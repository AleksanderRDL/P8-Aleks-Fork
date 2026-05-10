# Experiments Module

This module is the orchestration layer for the v2 rebuild. It turns flat CLI arguments into structured config objects, derives deterministic sub-seeds, generates workloads, trains MLQDS, evaluates baselines with higher-is-better F1, and writes the example outputs.

## Files

| File | Purpose |
| --- | --- |
| `experiment_cli.py` | `argparse` parser for `run_ais_experiment.py`. |
| `experiment_config.py` | Dataclasses for data, query, model, baseline, visualization, workload, and seed config. |
| `experiment_pipeline_helpers.py` | Trajectory split, workload generation, training/evaluation, and result dumping. |
| `geojson_writers.py` | GeoJSON and simplified-CSV writers plus trajectory length reporting helpers. |
| `run_ais_experiment.py` | Main entry point. Loads CSV or synthetic data and prints the result tables. |
| `run_inference.py` | Load a saved checkpoint and evaluate it on a CSV without retraining. |
| `benchmark_runtime.py` | Runtime benchmark wrapper that records environment, git state, commands, timings, and final metrics. |
| `benchmark_matrix.py` | Range-focused matrix runner for comparing runtime/batch/checkpoint-F1 variants. |

## CLI

`run_ais_experiment.py` accepts:

- `--csv_path`
- `--train_csv_path` / `--train_csv`
- `--eval_csv_path` / `--eval_csv`
- `--cache_dir`
- `--refresh_cache`
- `--n_ships`
- `--n_points`
- `--min_points_per_segment`
- `--max_points_per_segment` / `--max_points_per_ship`
- `--max_time_gap_seconds`
- `--max_segments`
- `--max_trajectories`
- `--n_queries`
- `--query_coverage` / `--target_query_coverage`
- `--max_queries`
- `--range_spatial_fraction`
- `--range_time_fraction`
- `--range_min_point_hits`
- `--range_max_point_hit_fraction`
- `--range_min_trajectory_hits`
- `--range_max_trajectory_hit_fraction`
- `--range_max_box_volume_fraction`
- `--range_duplicate_iou_threshold`
- `--range_acceptance_max_attempts`
- `--knn_k`
- `--epochs`
- `--lr`
- `--pointwise_loss_weight`
- `--gradient_clip_norm`
- `--train_batch_size`
- `--inference_batch_size`
- `--query_chunk_size`
- `--compression_ratio`
- `--model_type {baseline,turn_aware}`
- `--workload {range,knn,similarity,clustering}`
- `--train_workload_mix` / `--workload_mix_train`
- `--eval_workload_mix` / `--workload_mix_eval`
- `--early_stopping_patience`
- `--diagnostic_every`
- `--diagnostic_window_fraction`
- `--checkpoint_selection_metric {loss,f1,uniform_gap}`
- `--f1_diagnostic_every`
- `--checkpoint_uniform_gap_weight`
- `--checkpoint_type_penalty_weight`
- `--checkpoint_smoothing_window`
- `--checkpoint_f1_variant {answer,combined}`
- `--mlqds_temporal_fraction`
- `--mlqds_diversity_bonus`
- `--residual_label_mode {none,temporal}`
- `--float32_matmul_precision {highest,high,medium}`
- `--allow_tf32` / `--no-allow_tf32`
- `--amp_mode {off,bf16,fp16}`
- `--save_model`
- `--save_queries_dir`
- `--save_simplified_dir`
- `--seed`
- `--results_dir`

`run_inference.py` additionally accepts `--inference_device {auto,cpu,cuda}`;
`auto` uses CUDA for MLQDS model inference when available. It also accepts
the same matmul precision, TF32, and AMP flags; by default it reuses checkpoint
runtime precision settings when present.

If `--train_csv_path` and `--eval_csv_path` are supplied together, the training CSV is used only for training and the evaluation CSV is used only for evaluation/simplified-output writing. If `--csv_path` is supplied instead, trajectories are split at trajectory level as before. If all CSV arguments are omitted, synthetic AIS data is generated with `n_ships`, `n_points`, and `seed`.

CSV loading now segments each MMSI by temporal continuity before training. The default `--max_time_gap_seconds 3600` starts a new segment when consecutive AIS rows for one MMSI are more than one hour apart. Use `--max_time_gap_seconds 0` to disable gap-based segmentation for compatibility checks. CSV runs write loader audit stats into `example_run.json` under `data_audit`.

Use `--cache_dir` to persist segmented CSV data as Parquet. Cache entries are
keyed by source file identity and segmentation config, and `--refresh_cache`
forces a rebuild when you want to verify the source parser path.

## Config Objects

- `DataConfig` - CSV paths, synthetic data size, and legacy train/validation split fractions.
- `QueryConfig` - workload size, optional target point coverage, workload label, train/eval workload mixes, and `similarity_top_k`.
- `ModelConfig` - embedding sizes, transformer depth, chunk size, dropout, compression ratio, ranking-loss hyperparameters, checkpoint-selection diagnostics, and torch precision controls.
- `BaselineConfig` - baseline toggles such as `include_oracle`, enabled by default for matched-eval label Oracle comparisons.
- `VisualizationConfig` - current hook for optional plotting.
- `TypedQueryWorkload` - padded query features, original typed query dicts, and query type IDs.
- `ExperimentConfig` - top-level container that nests the other configs.
- `SeedBundle` - deterministic sub-seeds for split, train workload, eval workload, and torch.

## Pipeline

1. Use separate train/eval trajectory sets when provided, otherwise split one dataset into train, validation, and test sets at trajectory level.
2. Generate independent train and eval typed query workloads from the respective trajectory sets; range/kNN anchors use the 70/30 density sampler described in `src/queries`.
3. Train the query-aware model and restore the epoch with the selected checkpoint metric. The default is exact held-out query F1 on a validation workload using the same requested query count as final eval. `f1_diagnostic_every` controls how often held-out F1 is evaluated and therefore which epochs are eligible for F1-based checkpoint selection. `checkpoint_f1_variant=answer` selects on pure answer-set F1; `checkpoint_f1_variant=combined` compares against the legacy answer/support product. `checkpoint_selection_metric=uniform_gap` also scores the fair `uniform` baseline on the validation workload and penalizes checkpoints below that baseline. `checkpoint_smoothing_window` can select by a rolling mean of diagnostic scores instead of a single noisy epoch.
4. Evaluate MLQDS and baseline methods on the test set. Phase 3 benchmark runs should keep `uniform`, Douglas-Peucker, and label Oracle in the matched results.
5. Reuse one evaluation query cache across matched methods so full-data query answers and support masks are not recomputed for every baseline.
6. Write `example_run.json`, `matched_table.txt`, `shift_table.txt`, `geometric_distortion_table.txt`, `range_workload_diagnostics.json`, and `range_query_diagnostics.jsonl` under `results_dir` with aggregate/per-type F1 fields, retained-point spacing metrics such as `AvgPtGap`, length preservation, torch runtime precision settings, plus `best_epoch`, `best_loss`, `best_f1`, `checkpoint_selection_metric`, and `checkpoint_f1_variant` training metadata.
7. Optionally write eval queries as GeoJSON through `--save_queries_dir`, and simplified trajectory CSVs through `--save_simplified_dir`.

## Runtime Benchmark Wrapper

`benchmark_runtime.py` shells out to the existing training and inference CLIs so
the measured path matches normal usage. It writes `benchmark_runtime.json` under
`--results_dir`, along with child stdout logs. Use it before and after speed
changes so runtime, environment, GPU visibility, and final F1 are compared from
the same artifact schema.

```bash
cd QDS
../.venv/bin/python -m src.experiments.benchmark_runtime \
  --mode train \
  --profile range_real_usecase \
  --train_extra_args "--train_csv_path ../AISDATA/cleaned/aisdk-2026-02-02_cleaned.csv --eval_csv_path ../AISDATA/cleaned/aisdk-2026-02-03_cleaned.csv --cache_dir artifacts/cache/range_real_usecase" \
  --results_dir artifacts/benchmarks/runtime_range_real_usecase
```

Use `--train_batch_sizes 16,32,64,128` with `--mode train` to run one child
training benchmark per batch size. Each child gets its own output directory and
checkpoint, and the top-level JSON includes `train_batch_size_sweep` rows with
epoch-time summaries, peak CUDA memory, `best_f1`, and MLQDS aggregate F1.
Training modes require real CSV data in `--train_extra_args`; the old synthetic
`small`/`medium`/`serious` runtime profiles have been removed.
Use `--inference_batch_size` on the training or inference CLI to tune MLQDS
window batches independently of optimizer batches. It applies to matched/shift
evaluation, saved-checkpoint inference, and held-out validation query-F1
diagnostics.
Use `--amp_mode bf16` to benchmark CUDA autocast; the wrapper forwards the mode
to child training/inference commands and records effective AMP metadata in both
the child run JSON and the top-level benchmark artifact.

## Range Benchmark Matrix

`benchmark_matrix.py` defaults to the range workload and runs one child
experiment per configuration variant, then writes `benchmark_matrix.json`,
`benchmark_matrix.csv`, and a compact `benchmark_matrix.md` table. Non-range
workloads remain available through `--workloads`, but the current benchmark
track should stay range-only until the range model is stronger.

```bash
cd QDS
../.venv/bin/python -m src.experiments.benchmark_matrix \
  --profile range_real_usecase \
  --csv_path ../AISDATA/cleaned \
  --cache_dir artifacts/cache/range_real_usecase \
  --variants tf32_bf16_bs32_inf32 \
  --results_dir artifacts/benchmarks/range_real_usecase/runs/manual_range_real_usecase_a \
  --run_id manual_range_real_usecase_a
```

For the real-usecase AIS profile, pass the cleaned-data directory. The matrix
selects the first two sorted cleaned CSV files as train/eval days, warms their
segmented Parquet caches before measured runs, and then runs the variants
against cache hits. The profile uses 512 range queries, a 30% target query
coverage, `range_spatial_fraction=0.0165`, `range_time_fraction=0.033`, 5%
retained-point compression, `query_chunk_size=512`, 20 epochs, answer-set F1
checkpointing, no checkpoint smoothing, `mlqds_temporal_fraction=0.10`, and F1
diagnostics every epoch. It enables `early_stopping_patience=8` so non-improving
F1 runs can stop before all 20 epochs. Leave `--max_points_per_segment`,
`--max_segments`, and `--max_trajectories` unset for this profile so all valid
segments and points from both days are used.

Use `--train_csv_path` and `--eval_csv_path` to choose the two days manually.
Use `--no_cache_warmup` only when intentionally measuring cold-cache behavior.
The default variant is the current baseline candidate
`tf32_bf16_bs32_inf32`: TF32 enabled, BF16 autocast, train/inference batches of
32, `checkpoint_selection_metric=f1`, and `checkpoint_f1_variant=answer`. Pass
an explicit comma-separated `--variants` list when intentionally running a
comparison matrix.

Each matrix run writes a run-local guide and index:

```text
README.md
run_config.json
run_status.json
artifact_index.json
benchmark_matrix.{json,csv,md}
logs/
variants/<variant>/
  simplified_eval/
```

Use one run directory per benchmark attempt. The child experiment artifacts for
each variant live under `variants/<variant>/`; default simplified eval CSVs live
under `variants/<variant>/simplified_eval/`; tmux launcher logs live under
`logs/`. The family root keeps `runs_index.csv` with the latest status per run
and `runs_index_events.jsonl` with status history.
Child experiment stdout is streamed live through the matrix console and written
to `variants/<variant>/stdout.log`, so long-running variants should show phase
progress in tmux before the variant exits. The matrix keeps only a bounded
stdout tail in memory; the full child log remains on disk.

For long local runs, prefer the tmux launcher:

```bash
cd QDS
make benchmark-preflight
scripts/run_range_benchmark_tmux.sh
```

It creates one pane for the matrix and one pane for
`scripts/monitor_system.sh`. The monitor writes
`system_monitor.log` beside the benchmark artifact and samples RAM/swap, disk,
top RSS processes, GPU utilization, GPU memory, temperature, power draw, clocks,
visible CUDA processes, and kernel markers for OOM/GPU/reset/thermal events
emitted after the monitor starts.
If the matrix process exits abnormally before it can finalize `run_status.json`,
the launcher marks any stale `running` status as `failed` and updates the family
run index.

Operational helpers:

```bash
make list-runs
make clean-smoke-artifacts
make clean-smoke-artifacts CONFIRM=1
```

`make list-runs` reads the active family `runs_index.csv`. The cleanup target
dry-runs by default and deletes only known smoke/test artifact directories when
`CONFIRM=1` is set. The artifact policy is tracked in
`QDS/artifacts/README.md`.
`make benchmark-preflight` also reports available RAM, swap, and git worktree
state before expensive runs; these are warnings unless a required prerequisite
is missing.

## Workload Mixes

Experiment entrypoints now train one model per pure query workload. The common
path is `--workload {range,knn,similarity,clustering}`, which uses the same
pure type for train, validation, and eval workloads. Explicit
`--train_workload_mix` and `--eval_workload_mix` strings are still parsed for
compatibility, but they must contain exactly one positive query type.
