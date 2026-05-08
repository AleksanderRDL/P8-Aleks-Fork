# AIS-QDS v2

AIS-QDS v2 is the current shift-aware rebuild of the AIS query-driven simplification pipeline. It loads AIS trajectories, generates typed query workloads, trains a query-conditioned ranking model, and evaluates the resulting simplifier against learned and geometric baselines under matched and shifted workloads.

## What Is In This Folder

- `requirements.txt` - Python dependencies for the v2 stack.
- `src/` - package code for loading data, building queries, training models, and running experiments.
- `tests/` - regression tests that guard the rebuild.
- `results/` - example outputs from the experiment runner.

## Quick Start

```bash
cd QDS
pip install -r requirements.txt
python -m src.experiments.run_ais_experiment --n_queries 128 --epochs 6 --workload mixed
```

## Phase 0 Environment And Smoke Checks

The sprint environment is the repository-level virtual environment at `../.venv`
when commands are run from `QDS`. Requirements are pinned in
`requirements.txt` to the versions used for the Phase 0 check.

```bash
cd QDS
../.venv/bin/python -m pip install -r requirements.txt
make check-env
make test
```

Use the local Makefile for repeatable smoke runs:

```bash
# Tiny synthetic train/eval run. Outputs go to artifacts/results/smoke_synthetic.
make smoke

# Tiny cleaned-CSV smoke run against AISDATA/cleaned with segmented Parquet cache.
make smoke-csv
```

The cleaned-CSV smoke target uses `--max_points_per_segment` and
`--max_segments` so it validates the real loader without turning into a full
research run. New sprint artifacts should be written under `artifacts/` or
another external run directory, not under `src/models/saved_models`.

To run on a CSV instead of synthetic data:

```bash
python -m src.experiments.run_ais_experiment \
  --csv_path "C:\path\to\ais.csv" \
  --max_points_per_segment 500 \
  --max_time_gap_seconds 3600 \
  --n_queries 250 \
  --epochs 20 \
  --workload mixed \
  --compression_ratio 0.20 \
  --model_type baseline
```

To train on one CSV and evaluate/clean a different CSV without trajectory splitting:

```bash
python -m src.experiments.run_ais_experiment \
  --train_csv_path "C:\path\to\train.csv" \
  --eval_csv_path "C:\path\to\eval.csv" \
  --query_coverage 0.30 \
  --max_queries 1000 \
  --range_spatial_fraction 0.02 \
  --range_time_fraction 0.04 \
  --epochs 20 \
  --lr 0.0005 \
  --pointwise_loss_weight 0.25 \
  --gradient_clip_norm 1.0 \
  --workload mixed \
  --compression_ratio 0.20 \
  --model_type baseline
```

`--query_coverage` accepts either `0.30` or `30` for 30% point coverage. Coverage mode still emits exactly `--n_queries`; the target coverage is used to bias later query anchors toward points that have not yet been covered. `--max_queries` is accepted only as a positive legacy safety parameter. When `--eval_csv_path` is used and `--save_simplified_dir` is not set, the experiment writes only the eval-set simplified CSV under `AISDATA/ML_processed_AIS_files` as `ML_simplified_eval.csv`.

For local smoke runs on large cleaned AIS files, `--max_points_per_segment`
downsamples each loaded trajectory segment, `--max_segments` caps the loader
output during segmentation, and `--max_trajectories` remains as a legacy
post-load cap. CSV loading splits MMSI tracks by default whenever consecutive
points are more than `--max_time_gap_seconds 3600` seconds apart; set this to
`0` to disable gap-based segmentation for compatibility checks. Benchmark runs
should record any caps explicitly or leave them unset.

Add `--cache_dir artifacts/cache/<run-name>` to CSV runs to reuse segmented
Parquet caches across experiments. Use `--refresh_cache` when changing loader
code or when you want to force a rebuild of a matching source/config entry.

Range and kNN workloads focus query anchors on dense areas with a 70/30 sampler: 70% density-map weighted by lat/lon grid cell occupancy, 30% uniform from all points. The same sampler is used by coverage-targeted generation, so coverage still controls when generation stops while density controls where range/kNN queries are anchored.

For range-heavy runs, `--range_spatial_fraction` and `--range_time_fraction` control range-box half-widths as fractions of the dataset latitude/longitude and time spans. Lower these when you want many range queries without covering most of the dataset; for example, `0.02` spatial and `0.04` time gives smaller local boxes than the default `0.08` and `0.15`.

Phase 2 range diagnostics can be enabled without changing the default generator.
Use optional acceptance filters such as `--range_min_point_hits`,
`--range_max_point_hit_fraction`, `--range_max_trajectory_hit_fraction`,
`--range_max_box_volume_fraction`, `--range_duplicate_iou_threshold`, and
`--range_acceptance_max_attempts` to reject broad, duplicate, or uninformative
range boxes during generation. Each run now writes range workload diagnostics
and range label/baseline signal diagnostics into the result directory.

Use `--model_type turn_aware` to include the extra `turn_score` point feature. Workload mixes can be overridden with `--train_workload_mix` and `--eval_workload_mix` (or the `..._mix_train` / `..._mix_eval` aliases).

Training uses a ranking loss plus balanced pointwise BCE supervision. Exact final query F1 can optionally be used for checkpoint selection with `--checkpoint_selection_metric f1`; this creates a held-out validation workload and restores the epoch with the best validation query F1. If diagnostics collapse to `pred_std=0`, prefer lowering `--lr`, keeping `--gradient_clip_norm` enabled, and increasing query diversity before changing the model architecture.

## Architecture At A Glance

1. `src/data/` loads AIS CSV files or generates deterministic synthetic trajectories.
2. `src/queries/` builds typed query workloads and executes range, kNN, similarity, and clustering queries.
3. `src/training/` computes typed F1-contribution labels, trains the model, restores the selected checkpoint epoch, and persists the scaler and checkpoint artifacts.
4. `src/models/` contains the query-conditioned trajectory transformer and the turn-aware variant.
5. `src/simplification/` keeps the highest-scoring points per trajectory with deterministic tie-breaking.
6. `src/evaluation/` runs learned and baseline methods and reports aggregate and per-type F1 scores.
7. `src/experiments/` wires the full pipeline together through the CLI.
8. `src/visualization/` is currently a placeholder package for future plotting hooks.

## Outputs

The experiment runner writes three files into `results/`:

- `example_run.json` - config, workload mixes, per-method metrics, training history, and selected-checkpoint metadata.
- `matched_table.txt` - fixed-width comparison table for the evaluation workload.
- `shift_table.txt` - shift table comparing the train workload against the eval workload.
- `range_workload_diagnostics.json` - Phase 2 range workload, label, Oracle, and baseline diagnostics for train/eval/selection workloads.
- `range_query_diagnostics.jsonl` - one JSON record per generated range query with hit counts, footprint, broad-query flag, and duplicate-query flag.

## Validation

The `tests/` folder focuses on the rebuild-specific regressions:

- `test_beats_random_in_distribution.py` - in-distribution performance guard.
- `test_no_cross_trajectory_attention_leakage.py` - attention leakage guard.
- `test_query_type_ids_required.py` - query type ID contract.
- `test_scaler_persisted.py` - scaler persistence.
- `test_topk_no_positional_bias.py` - deterministic top-k behavior.
- `test_training_does_not_collapse.py` - training stability.

For the legacy v1 system, see [qds_project/README.md](../qds_project/README.md).
