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
- `--compression_ratio`
- `--model_type {baseline,turn_aware}`
- `--workload`
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
- `--save_model`
- `--save_queries_dir`
- `--save_simplified_dir`
- `--seed`
- `--results_dir`

`run_inference.py` additionally accepts `--inference_device {auto,cpu,cuda}`;
`auto` uses CUDA for MLQDS model inference when available. It also accepts
the same matmul precision and TF32 flags; by default it reuses checkpoint
precision settings when present.

If `--train_csv_path` and `--eval_csv_path` are supplied together, the training CSV is used only for training and the evaluation CSV is used only for evaluation/simplified-output writing. If `--csv_path` is supplied instead, trajectories are split at trajectory level as before. If all CSV arguments are omitted, synthetic AIS data is generated with `n_ships`, `n_points`, and `seed`.

CSV loading now segments each MMSI by temporal continuity before training. The default `--max_time_gap_seconds 3600` starts a new segment when consecutive AIS rows for one MMSI are more than one hour apart. Use `--max_time_gap_seconds 0` to disable gap-based segmentation for compatibility checks. CSV runs write loader audit stats into `example_run.json` under `data_audit`.

Use `--cache_dir` to persist segmented CSV data as Parquet. Cache entries are
keyed by source file identity and segmentation config, and `--refresh_cache`
forces a rebuild when you want to verify the source parser path.

## Config Objects

- `DataConfig` - CSV paths, synthetic data size, and legacy train/validation split fractions.
- `QueryConfig` - workload size, optional target point coverage, workload label, train/eval workload mixes, and `similarity_top_k`.
- `ModelConfig` - embedding sizes, transformer depth, chunk size, dropout, compression ratio, ranking-loss hyperparameters, checkpoint-selection diagnostics, and torch precision controls.
- `BaselineConfig` - baseline toggles such as `include_oracle`.
- `VisualizationConfig` - current hook for optional plotting.
- `TypedQueryWorkload` - padded query features, original typed query dicts, and query type IDs.
- `ExperimentConfig` - top-level container that nests the other configs.
- `SeedBundle` - deterministic sub-seeds for split, train workload, eval workload, and torch.

## Pipeline

1. Use separate train/eval trajectory sets when provided, otherwise split one dataset into train, validation, and test sets at trajectory level.
2. Generate independent train and eval typed query workloads from the respective trajectory sets; range/kNN anchors use the 70/30 density sampler described in `src/queries`.
3. Train the query-aware model and restore the epoch with the selected checkpoint metric. The default is training loss; `checkpoint_selection_metric=f1` uses exact held-out query F1 on a validation workload. `checkpoint_selection_metric=uniform_gap` also scores the fair `uniform` baseline on the validation workload and penalizes checkpoints that hide weak range/kNN/similarity scores behind one strong type. `checkpoint_smoothing_window` can select by a rolling mean of diagnostic scores instead of a single noisy epoch.
4. Evaluate MLQDS and baseline methods on the test set. Phase 3 benchmark runs should keep `uniform`, Douglas-Peucker, and label Oracle in the matched results.
5. Reuse one evaluation query cache across matched methods so full-data query answers and support masks are not recomputed for every baseline.
6. Write `example_run.json`, `matched_table.txt`, `shift_table.txt`, `geometric_distortion_table.txt`, `range_workload_diagnostics.json`, and `range_query_diagnostics.jsonl` under `results_dir` with aggregate/per-type F1 fields, retained-point spacing metrics such as `AvgPtGap`, length preservation, torch runtime precision settings, plus `best_epoch`, `best_loss`, and `best_f1` training metadata.
7. Optionally write eval queries as GeoJSON through `--save_queries_dir`, and simplified trajectory CSVs through `--save_simplified_dir`.

## Runtime Benchmark Wrapper

`benchmark_runtime.py` shells out to the existing training and inference CLIs so
the measured path matches normal usage. It writes `benchmark_runtime.json` under
`--results_dir`, along with child stdout logs. Use it before and after speed
changes so runtime, environment, GPU visibility, and final F1 are compared from
the same artifact schema.

```bash
cd QDS
../.venv/bin/python -m src.experiments.benchmark_runtime --mode train --profile small
```

## Workload Mixes

`resolve_workload_mixes` parses comma-separated strings such as `range=0.8,knn=0.2`. Explicit `--train_workload_mix` and `--eval_workload_mix` strings win. Otherwise, the `--workload` keyword is used for both train and eval mixes; the default `--workload mixed` resolves to `range=0.4,knn=0.2,similarity=0.2,clustering=0.2`. The older mixed-shift defaults (`range=0.8,knn=0.2` for training and `range=0.2,clustering=0.8` for evaluation) are now only used if the workload keyword is absent or unrecognized.
