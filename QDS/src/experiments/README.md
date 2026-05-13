# Experiments Module

This module owns run orchestration: parse CLI args, load AIS data, generate one
pure query workload, train MLQDS, evaluate baselines, and write artifacts.

## Files

| File | Purpose |
| --- | --- |
| `experiment_cli.py` | Shared `argparse` parser for train/eval commands. |
| `experiment_config.py` | Structured config dataclasses. |
| `experiment_pipeline_helpers.py` | Split, workload, training, evaluation, and artifact helpers. |
| `benchmark_profiles.py` | Named benchmark profile constants. |
| `benchmark_runner.py` | Range benchmark runner for profiles and queued overrides. |
| `benchmark_runtime.py` | Targeted train/inference timing experiments. |
| `run_ais_experiment.py` | Main train/evaluate entry point. |
| `run_inference.py` | Evaluate a saved checkpoint without retraining. |

Use `--help` for the full current CLI surface:

```bash
../.venv/bin/python -m src.experiments.run_ais_experiment --help
../.venv/bin/python -m src.experiments.benchmark_runner --help
../.venv/bin/python -m src.experiments.run_inference --help
```

## Data Modes

- Three CSV paths: `--train_csv_path`, `--validation_csv_path`, and
  `--eval_csv_path` are used as train, checkpoint-validation, and final-eval
  days.
- Train/eval CSV paths only: checkpoint validation is split out of train data
  for compatibility.
- One `--csv_path`: trajectories are split from one dataset.
- No CSV path: deterministic synthetic data is generated.

CSV loading segments MMSI tracks by time gaps. The default
`--max_time_gap_seconds 3600` starts a new segment after a one-hour gap.
`--cache_dir` persists post-segmentation Parquet data keyed by source file and
segmentation config.

## Current Benchmark Baseline

The active benchmark profile is `range_testing_baseline`. It is a pure range
workload profile for cleaned AIS CSV days:

| Area | Default |
| --- | --- |
| Data split | first three sorted cleaned CSVs = train, validation, eval |
| Workload | range only |
| Query generation | minimum `80`, target `20%` coverage, cap `2048` |
| Range footprint | `range_spatial_km=2.2`, `range_time_hours=5.0`, no jitter |
| Compression | `5%` retained points |
| Training | `20` epochs, early stopping patience `5` |
| Checkpoint target | `checkpoint_f1_variant=range_usefulness` |
| Loss | `budget_topk` over budgets `1%,2%,5%,10%` |
| MLQDS simplification | score mode `rank`, temporal fraction `0.25` |
| Runtime | TF32 enabled, BF16 AMP, train/inference batch size `64` |
| Query chunking | `query_chunk_size=2048`, also used as `max_queries` |
| Loader caps | no `max_points_per_segment`, `max_segments`, or `max_trajectories` |

Keep durable baseline defaults in `benchmark_profiles.py`. For experiments,
use profile overrides in queue rows or `BENCHMARK_CHILD_EXTRA_ARGS`; promote an
override into a named profile only when it becomes a repeated baseline.

## Running Benchmarks

Run preflight first:

```bash
make benchmark-preflight
```

Launch one profile run in tmux:

```bash
ATTACH=0 BENCHMARK_RUN_ID=range_testing_baseline_a make range-benchmark-tmux
```

Launch a sequential queue:

```bash
ATTACH=0 \
  BENCHMARK_PLAN_FILE=artifacts/benchmarks/range_testing_baseline/queues/my_plan.tsv \
  BENCHMARK_CONTINUE_ON_FAILURE=1 \
  make range-benchmark-queue-tmux
```

Queue plan rows are tab-separated:

```text
range_testing_baseline_seed42	42
range_testing_baseline_score_rank_tie_seed42	42	--mlqds_score_mode rank_tie
range_testing_baseline_pairs192_seed42	42	--ranking_pairs_per_type 192
```

The queue launcher validates child args before tmux starts, so unsupported
overrides fail before an expensive run begins.

## Direct CLI Example

```bash
../.venv/bin/python -m src.experiments.benchmark_runner \
  --profile range_testing_baseline \
  --workloads range \
  --csv_path ../AISDATA/cleaned \
  --cache_dir artifacts/cache/range_testing_baseline \
  --results_dir artifacts/benchmarks/range_testing_baseline/runs/manual_a \
  --run_id manual_a
```

Use `--no_cache_warmup` only when intentionally measuring cold-cache behavior.

## Coverage Calibration

Estimate query count and coverage before changing footprint or target coverage:

```bash
../.venv/bin/python scripts/estimate_range_coverage.py \
  --csv_path ../AISDATA/cleaned \
  --cache_dir artifacts/cache/range_testing_baseline \
  --query_counts 80,384,512,640,1024,2048 \
  --sample_stride 20 \
  --target_coverage 0.20 \
  --range_spatial_km 2.2 \
  --range_time_hours 5.0 \
  --range_footprint_jitter 0.0
```

`query_coverage` is point-level query-signal coverage. When `max_queries` is
larger than `n_queries`, generation continues until the target is reached or
the cap is hit. Run artifacts record the final generated count and stop reason.

## Artifacts

For comparisons, start with `benchmark_report.md` or `benchmark_report.csv`.
Then inspect the child run files that explain behavior:

- `example_run.json`
- `matched_table.txt`
- `range_usefulness_table.txt`
- `range_workload_diagnostics.json`
- `learned_fill_diagnostics.json`
- `range_residual_objective_summary.json`

Artifact layout and cleanup rules live in
[`../../artifacts/README.md`](../../artifacts/README.md). Metric definitions
live in [`../evaluation/README.md`](../evaluation/README.md), and training
objective details live in [`../training/README.md`](../training/README.md).

## Timing Experiments

Use `benchmark_runtime.py` only for targeted runtime studies, not model-quality
benchmarking:

```bash
../.venv/bin/python -m src.experiments.benchmark_runtime \
  --mode train \
  --profile range_testing_baseline \
  --train_extra_args "--train_csv_path ../AISDATA/cleaned/aisdk-2026-02-02_cleaned.csv --validation_csv_path ../AISDATA/cleaned/aisdk-2026-02-03_cleaned.csv --eval_csv_path ../AISDATA/cleaned/aisdk-2026-02-04_cleaned.csv --cache_dir artifacts/cache/range_testing_baseline" \
  --train_batch_sizes 16,32,64 \
  --results_dir artifacts/benchmarks/runtime_range_testing_baseline
```
