# Experiments Module

Owns CLI parsing, config construction, benchmark profiles, run orchestration,
and artifact writing.

Commands below assume the current directory is `QDS/`:

```bash
PYTHON="$(cd .. && pwd -P)/.venv/bin/python"
"$PYTHON" -m experiments.run_ais_experiment --help
"$PYTHON" -m experiments.benchmark_runner --help
"$PYTHON" -m experiments.run_inference --help
```

## Files

| File | Purpose |
| --- | --- |
| `experiment_cli.py` | Shared CLI parser. |
| `experiment_config.py` | Structured config dataclasses. |
| `experiment_pipeline.py` | End-to-end train/eval orchestration. |
| `experiment_data.py` | Train/selection/eval split and dataset construction. |
| `experiment_workloads.py` | Workload map resolution and query workload generation. |
| `experiment_methods.py` | Evaluation method construction and shift scoring. |
| `experiment_outputs.py` | Experiment result payload and artifact writing. |
| `range_diagnostics.py` | Range workload, learned-fill, and audit diagnostics. |
| `benchmark_profiles.py` | Durable benchmark profile defaults. |
| `benchmark_inputs.py` | Benchmark source resolution and profile argument construction. |
| `benchmark_process.py` | Streaming child-process execution and captured output. |
| `benchmark_report.py` | Benchmark report rows, summaries, and compact tables. |
| `benchmark_runner.py` | Profile runner and queue orchestration entry point. |
| `benchmark_artifacts.py` | Status files, family indexes, and artifact guides. |
| `benchmark_runtime.py` | Runtime-only train/inference studies. |
| `workload_cache.py` / `range_cache.py` | Workload, diagnostics, and label caches. |
| `run_ais_experiment.py` | Main train/evaluate entry point. |
| `run_inference.py` | Evaluate a saved checkpoint without retraining. |

## Data Modes

- `--train_csv_path`, `--validation_csv_path`, `--eval_csv_path`: explicit
  train, checkpoint-validation, and final-eval sources. Each accepts a
  comma-separated CSV list for multi-day splits.
- `--train_csv_path`, `--eval_csv_path`: validation is split from train data.
  The default `--validation_split_mode random` samples from combined train
  trajectories. Use `--validation_split_mode source_stratified` to hold out
  trajectories from each train CSV source for checkpoint-selection diagnostics.
- `--train_csv_path day1.csv,day2.csv,...`: train on multiple historical CSV
  days while keeping checkpoint-validation and final-eval sources explicit and
  distinct. Validation and eval can use the same comma-list syntax for
  week-level held-out evaluation.
- `--csv_path`: one file or sorted directory split internally.
- no CSV path: deterministic synthetic data.

CSV loading segments MMSI tracks by `--max_time_gap_seconds` and can cache the
post-segmentation tensors with `--cache_dir`.
`--max_segments` is the global segment cap. In explicit split-CSV mode,
`--train_max_segments`, `--validation_max_segments`, and `--eval_max_segments`
can override it per split; unset split caps fall back to `--max_segments`.

## Active Profiles

`range_workload_aware_diagnostic` is the workload-aware diagnostic profile. It
uses `model_type=range_aware`, so it may see the supplied range workload during
compression. Treat it as diagnostic/teacher evidence, not final workload-blind
evidence.

The workload-blind benchmark profiles are:

- `range_workload_blind_expected_usefulness`
- `range_workload_blind_retained_frequency`
- `range_workload_blind_teacher_distill`

Those profiles freeze retained masks before held-out eval query scoring and
evaluate the full compression audit grid. They use a small `n_queries=8`
minimum query floor so `query_coverage` controls coverage. Do not raise that
floor for final claims unless you explicitly want a high-query workload
setting; a large floor can keep adding duplicate or near-duplicate queries after
coverage is already reached.

| Setting | Value |
| --- | --- |
| Workload | range only |
| Data | first three sorted cleaned CSVs = train, validation, eval |
| Coverage target | default `20%`; required sweep `5%,10%,15%,30%` |
| Compression | default `5%`; required sweep `1%,2%,5%,10%,15%,20%,30%` |
| Range footprint | `2.2 km`, `5.0 h`, jitter `0.0`, `anchor_day` time clamp |
| Training | `8` epochs, early stopping patience `5`, `budget_topk` loss |
| Checkpointing | `checkpoint_score_variant=range_usefulness`, `checkpoint_selection_metric=uniform_gap` |
| Runtime | BF16 AMP, TF32 allowed, train/inference batch size `64`, query chunk `2048` |

Keep durable defaults in `benchmark_profiles.py`. Use queue rows or
`BENCHMARK_CHILD_EXTRA_ARGS` for one-off variations.

For explicit coverage-grid checks, `benchmark_runner.py` accepts
`--coverage_targets 0.05,0.10,0.15,0.30`. The runner creates one child run per
coverage target and appends `c05`, `c10`, etc. to the child run label. Do not
also pass `--query_coverage` in `--extra_args` for the same benchmark.

## Benchmark Commands

```bash
make benchmark-preflight
ATTACH=0 BENCHMARK_RUN_ID=range_workload_aware_diagnostic_a make range-benchmark-tmux
ATTACH=0 BENCHMARK_SEEDS=42,43,44 make range-benchmark-queue-tmux
```

Queue files are tab-separated:

```text
run_id	seed	extra_child_args
```

The old workload-aware coverage/compression grid is archived at
[`../../benchmark_plans/archive/range_aware_coverage_compression_grid.tsv`](../../benchmark_plans/archive/range_aware_coverage_compression_grid.tsv).
It is valid only as workload-aware diagnostic evidence.

## Coverage Calibration

Use this before changing query count, footprint, or coverage target:

```bash
"$PYTHON" scripts/estimate_range_coverage.py \
  --csv_path ../AISDATA/cleaned \
  --cache_dir artifacts/cache/range_workload_aware_diagnostic \
  --query_counts 80,384,512,640,1024,2048 \
  --sample_stride 20 \
  --target_coverage 0.20 \
  --range_spatial_km 2.2 \
  --range_time_hours 5.0 \
  --range_time_domain_mode anchor_day \
  --range_footprint_jitter 0.0 \
  --range_max_coverage_overshoot 0.02
```

`query_coverage` is point-level query-signal coverage. If `max_queries` exceeds
`n_queries`, generation continues until coverage is reached or the cap is hit.
`n_queries` is still a minimum. For coverage-sweep claims, keep it low enough
that it does not dominate the requested `query_coverage`; use the generated
query count and actual coverage recorded in `example_run.json` as the evidence.
`range_max_coverage_overshoot` is an optional absolute upper tolerance and
accepts fractions or percentages. With `query_coverage=0.05` and
`range_max_coverage_overshoot=0.02`, generated
workloads reject boxes that would push union point coverage above 7%.

`range_anchor_mode` controls the eval/checkpoint workload anchor prior.
`range_train_anchor_modes` is an optional comma-separated list cycled across
training workload replicates only. Use it to train blind supervision against
multiple generated workload priors while keeping held-out eval queries unseen.
`range_train_footprints` similarly cycles train-only absolute footprint
families such as `1.1:2.5,2.2:5.0,4.4:10.0`; eval and checkpoint selection keep
the configured `range_spatial_km` / `range_time_hours` footprint.
For replicated retained-frequency targets, `range_replicate_target_aggregation`
can average raw labels (`label_mean`), take the raw-label upper envelope
(`label_max`), or average per-workload retained-frequency targets
(`frequency_mean`).

## Artifacts

Start with `benchmark_report.md` or `benchmark_report.csv`, then inspect child
run files:

- `example_run.json`
- `matched_table.txt`
- `range_usefulness_table.txt`
- `range_workload_diagnostics.json`
- `learned_fill_diagnostics.json`
- `range_learned_fill_summary.json`
- `range_compression_audit.json`, when multi-budget audits are enabled

Artifact layout and cleanup rules live in [`../../artifacts/README.md`](../../artifacts/README.md).

## Workload-Blind Rule

For final workload-blind claims, choose retained masks before generating or
passing eval queries into the model, feature builder, or checkpoint selector.
Current workload-blind profiles record protocol flags in `example_run.json` and
benchmark rows. Treat a run as invalid for final claims if
`workload_blind_protocol.primary_masks_frozen_before_eval_query_scoring` or
`workload_blind_protocol.audit_masks_frozen_before_eval_query_scoring` is
false.
