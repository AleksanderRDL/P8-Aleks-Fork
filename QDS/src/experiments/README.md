# Experiments Module

This module is the orchestration layer for AIS-QDS runs. It converts CLI
arguments into structured config, loads data, generates pure query workloads,
trains MLQDS, evaluates baselines, and writes run artifacts.

## Files

| File | Purpose |
| --- | --- |
| `experiment_cli.py` | Shared `argparse` parser for training/evaluation runs. |
| `experiment_config.py` | Data, query, model, baseline, workload, and seed config dataclasses. |
| `experiment_pipeline_helpers.py` | Split, workload generation, diagnostics reuse, training, evaluation, and result dumping. |
| `benchmark_profiles.py` | Central benchmark profile constants such as `range_real_usecase`. |
| `benchmark_matrix.py` | Range-focused benchmark matrix runner. |
| `benchmark_runtime.py` | Runtime wrapper for train/inference timing experiments. |
| `run_ais_experiment.py` | Main train/evaluate entry point. |
| `run_inference.py` | Evaluate a saved checkpoint on a CSV without retraining. |

## Entry Points

Use `--help` on each entry point for the full current argument list.

```bash
../.venv/bin/python -m src.experiments.run_ais_experiment --help
../.venv/bin/python -m src.experiments.run_inference --help
../.venv/bin/python -m src.experiments.benchmark_matrix --help
../.venv/bin/python -m src.experiments.benchmark_runtime --help
```

Common argument groups:

- Data: `--csv_path`, `--train_csv_path`, `--eval_csv_path`, segmentation caps, `--cache_dir`, `--refresh_cache`.
- Query workload: `--workload`, `--n_queries`, `--query_coverage`, range footprint/acceptance controls.
- Training: epochs, LR, ranking sampler, batch sizes, query chunking, checkpoint selection, early stopping.
- Runtime: `--float32_matmul_precision`, `--allow_tf32`, `--amp_mode`, inference device/batch size.
- Outputs: `--results_dir`, `--save_model`, `--save_queries_dir`, `--save_simplified_dir`.

Training-specific behavior is documented in
[`../training/README.md`](../training/README.md). Query-generation behavior is
documented in [`../queries/README.md`](../queries/README.md).

## Data Modes

If `--train_csv_path` and `--eval_csv_path` are both supplied, the train CSV is
used only for training/checkpoint selection and the eval CSV is used only for
final evaluation and simplified-output CSVs. If only `--csv_path` is supplied,
one dataset is split by trajectory. If all CSV paths are omitted, the CLI
generates deterministic synthetic data.

CSV loading segments each MMSI by temporal continuity. The default
`--max_time_gap_seconds 3600` starts a new segment after a one-hour gap; use
`0` only for compatibility checks. `--cache_dir` persists post-segmentation
Parquet data keyed by source file identity and segmentation config.

## Pipeline

1. Resolve train/eval/selection trajectory sets.
2. Generate independent typed workloads for train, eval, and checkpoint selection.
3. Compute range diagnostics and reusable labels/query caches when applicable.
4. Train MLQDS and restore the best checkpoint according to the active selection metric.
5. Evaluate MLQDS, uniform, Douglas-Peucker, and label Oracle on the eval workload.
6. Write tables, JSON diagnostics, optional GeoJSON queries, and optional simplified CSVs.

Current sprint policy is one model per pure workload. Use
`--workload {range,knn,similarity,clustering}`. Explicit workload-mix arguments
remain for compatibility but must contain exactly one positive query type.

## Real-Usecase Range Profile

`benchmark_matrix.py --profile range_real_usecase` is the current benchmark
baseline. It selects the first two sorted cleaned CSV files from `--csv_path`
as train/eval days unless explicit train/eval paths are provided.

Profile shape:

| Setting | Value |
| --- | --- |
| Workload | pure `range` |
| Queries | `80` |
| Target coverage | `0.20` |
| Range footprint | `range_spatial_km=2.2`, `range_time_hours=3.0` fixed half-windows (`range_footprint_jitter=0.0`) |
| Compression | `0.05` retained points |
| Epoch budget | `20` with `early_stopping_patience=5` |
| Checkpoint selection | `checkpoint_selection_metric=f1`, `checkpoint_f1_variant=answer` |
| MLQDS scoring | pure workload `rank` mode, `mlqds_temporal_fraction=0.25`, `mlqds_score_temperature=1.0` |
| Attention chunk | `query_chunk_size=2048` |
| Range labels | pure point-F1 labels (`range_boundary_prior_weight=0.0`) |
| Diagnostics | `f1_diagnostic_every=1`, no smoothing (`checkpoint_smoothing_window=1`) |
| Runtime variant | `tf32_bf16_bs32_inf32` by default |
| Ranking sampler | `vectorized` by default |
| Caps | leave `max_points_per_segment`, `max_segments`, and `max_trajectories` unset |

`query_coverage` is a target used for workload generation and diagnostics, not
a guarantee. If the fixed query count/footprint cannot hit the target on a day,
the run prints a warning and records the realized coverage in `example_run.json`.

Direct matrix run:

```bash
../.venv/bin/python -m src.experiments.benchmark_matrix \
  --profile range_real_usecase \
  --workloads range \
  --csv_path ../AISDATA/cleaned \
  --cache_dir artifacts/cache/range_real_usecase \
  --variants tf32_bf16_bs32_inf32 \
  --results_dir artifacts/benchmarks/range_real_usecase/runs/manual_range_real_usecase_a \
  --run_id manual_range_real_usecase_a
```

Use `--no_cache_warmup` only when intentionally measuring cold-cache behavior.
Before changing range footprint or target coverage, estimate query counts first:

```bash
../.venv/bin/python scripts/estimate_range_coverage.py \
  --csv_path ../AISDATA/cleaned \
  --cache_dir artifacts/cache/range_real_usecase \
  --query_counts 64,72,80,88,96,112 \
  --sample_stride 20 \
  --target_coverage 0.20 \
  --range_spatial_km 2.2 \
  --range_time_hours 3.0 \
  --range_footprint_jitter 0.0
```

## Long Runs

Use tmux launchers for long local runs. They keep training alive outside the
IDE session and capture system telemetry beside the run.

```bash
make benchmark-preflight
ATTACH=0 BENCHMARK_RUN_ID=range_real_usecase_a make range-benchmark-tmux
```

Queue rows use a tab-separated plan file:

```text
range_real_usecase_base_seed42	42
range_real_usecase_rank_tie_seed42	42	--mlqds_score_mode rank_tie
range_real_usecase_pairs192_seed42	42	--ranking_pairs_per_type 192
```

Launch a sequential queue:

```bash
ATTACH=0 \
  BENCHMARK_PLAN_FILE=artifacts/benchmarks/range_real_usecase/queues/my_plan.tsv \
  BENCHMARK_CONTINUE_ON_FAILURE=1 \
  make range-benchmark-queue-tmux
```

The queue launcher validates each row's child args against
`run_ais_experiment` before tmux starts, so unsupported benchmark knobs fail
fast.

The monitor pane writes RAM/swap, disk, top RSS processes, GPU utilization,
GPU memory, temperature, power draw, clocks, CUDA processes, and relevant
kernel markers. If a launcher observes an abnormal exit, it marks stale
`run_status.json` files as failed.

## Artifacts And Timing

For comparisons, start with `benchmark_matrix.md` or `benchmark_matrix.csv`.
For model behavior, inspect the variant `example_run.json`,
`matched_table.txt`, and `range_workload_diagnostics.json`. Artifact layout and
cleanup rules live in [`../../artifacts/README.md`](../../artifacts/README.md).

Use `benchmark_runtime.py` only for targeted train/inference timing studies
such as batch-size sweeps. Use `benchmark_matrix.py` for model-quality
benchmark runs.

```bash
../.venv/bin/python -m src.experiments.benchmark_runtime \
  --mode train \
  --profile range_real_usecase \
  --train_extra_args "--train_csv_path ../AISDATA/cleaned/aisdk-2026-02-02_cleaned.csv --eval_csv_path ../AISDATA/cleaned/aisdk-2026-02-03_cleaned.csv --cache_dir artifacts/cache/range_real_usecase" \
  --train_batch_sizes 16,32,64 \
  --results_dir artifacts/benchmarks/runtime_range_real_usecase
```
