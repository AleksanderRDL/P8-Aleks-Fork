# Experiments Module

Owns CLI parsing, config construction, benchmark profiles, run orchestration,
runtime metadata, and artifact writing.

Run CLI entry points from the repository root:

```bash
uv run --group dev -- python -m experiments.run_ais_experiment --help
uv run --group dev -- python -m experiments.benchmark_runner --help
uv run --group dev -- python -m experiments.run_inference --help
```

From `Range_QDS/`, use the Make targets for routine work:

```bash
make smoke
make benchmark-preflight
ATTACH=0 BENCHMARK_PROFILE=range_workload_v1_workload_blind_v2 make range-benchmark-tmux
ATTACH=0 BENCHMARK_SEEDS=42,43,44 make range-benchmark-queue-tmux
make list-runs
```

The benchmark Makefile keeps legacy artifact-family defaults for compatibility.
Set `BENCHMARK_FAMILY` and `BENCHMARK_CACHE` explicitly for comparable
query-driven batches.

## Key Files

| File | Purpose |
| --- | --- |
| `experiment_cli.py` / `experiment_config.py` | CLI flags and structured config. |
| `experiment_pipeline.py` | End-to-end train/eval orchestration. |
| `experiment_data.py` | Train, validation, selection, and eval data splits. |
| `experiment_workloads.py` | Workload generation and workload-map resolution. |
| `experiment_methods.py` | Evaluation method construction. |
| `experiment_outputs.py` | Run artifact payloads and writers. |
| `range_diagnostics.py` | Range workload, learned-fill, and gate diagnostics. |
| `benchmark_profiles.py` | Durable benchmark profile defaults. |
| `benchmark_runner.py` | Benchmark child-run and queue orchestration. |
| `benchmark_report.py` | Benchmark rows, summaries, and final-grid report logic. |
| `run_ais_experiment.py` | Main training/evaluation entry point. |
| `run_inference.py` | Evaluate a saved checkpoint without retraining. |

## Data Modes

- `--csv_path`: one cleaned CSV file or a directory split internally.
- `--train_csv_path`, `--validation_csv_path`, `--eval_csv_path`: explicit
  split sources. Comma-separated lists support multi-day splits.
- no CSV path: deterministic synthetic data.

CSV loading segments MMSI tracks by `--max_time_gap_seconds` and can cache
post-segmentation tensors with `--cache_dir`. `--max_segments`,
`--max_points_per_segment`, and split-specific caps are runtime controls; do
not treat tiny capped probes as scientific evidence.

## Benchmark Profiles

Current query-driven candidate profile:

- `range_workload_v1_workload_blind_v2`

This is the active QueryUsefulV1 workload-blind path. It is still blocked
unless the guide-required workload stability, support, predictability,
causality, global-sanity, and final-grid gates pass.

Legacy diagnostic profiles remain useful for regression and teacher checks:

- `range_workload_aware_diagnostic`
- `range_workload_blind_expected_usefulness`
- `range_workload_blind_retained_frequency`
- `range_workload_blind_teacher_distill`

Legacy profiles report diagnostic RangeUseful/scalar behavior and are not final
acceptance evidence.

Keep durable defaults in `benchmark_profiles.py`. Put one-off variations in
queue rows or `BENCHMARK_CHILD_EXTRA_ARGS`, not in profile defaults.

For coverage-grid checks, use:

```bash
uv run --group dev -- python -m experiments.benchmark_runner \
  --profile range_workload_v1_workload_blind_v2 \
  --workloads range \
  --coverage_targets 0.05,0.10,0.15,0.30
```

Do not also pass a conflicting `--query_coverage` through child extra args for
the same benchmark.

## Coverage Calibration

Use this before changing query count, footprint, or coverage target:

```bash
uv run --group dev -- python Range_QDS/scripts/estimate_range_coverage.py \
  --csv_path AISDATA/cleaned \
  --cache_dir Range_QDS/artifacts/cache/range_workload_v1 \
  --query_counts 32,48,64,96,128,256 \
  --sample_stride 20 \
  --target_coverage 0.10 \
  --range_spatial_km 2.2 \
  --range_time_hours 5.0 \
  --range_time_domain_mode anchor_day
```

`query_coverage` is point-level query-signal coverage. If `max_queries`
exceeds `n_queries`, generation continues until coverage is reached or the cap
is hit. `n_queries` remains a minimum. For coverage-sweep claims, use recorded
generated query count and actual coverage from `example_run.json`.

## Artifacts

Start with:

- run-local `README.md`
- `run_status.json`
- `benchmark_report.md`
- `benchmark_report.csv`
- `example_run.json`

Artifact layout and cleanup rules live in
[`../artifacts/README.md`](../artifacts/README.md). Use jq/Rich helpers from
[`../docs/dev-tooling-guide.md`](../docs/dev-tooling-guide.md) for one-run
inspection.

## Workload-Blind Rule

Final or diagnostic workload-blind claims require retained masks to be chosen
before held-out eval queries are scored. Treat a run as invalid if
`workload_blind_protocol.primary_masks_frozen_before_eval_query_scoring` or
`workload_blind_protocol.audit_masks_frozen_before_eval_query_scoring` is
false.

QueryUsefulV1 is the active primary metric for the rework. RangeUseful outputs
must remain under `legacy_range_useful_summary` or diagnostic fields, not
`final_claim_summary`.
