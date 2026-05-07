# Phase 2 Preparation: Range Workload And Label Diagnostics

## Goal

Prepare `qds_v2` for Phase 2 implementation and execution:

> Generate range-query workloads that are non-empty, non-trivial, not too broad,
> not heavily duplicated, and diagnostically strong enough to justify Range-QDS
> training in Phase 3.

Phase 2 should answer this before model training is trusted:

```text
Does the range workload contain learnable query-preservation signal, and do the
labels/oracle/baselines confirm that signal?
```

## Completion Status

Completed 2026-05-07. Phase 2 is accepted as ready for Phase 3
Range-QDS training and benchmarking.

Implemented:

- optional range acceptance filters for useful hit-count bands, broad boxes,
  near-duplicate boxes, and acceptance-attempt exhaustion
- train/eval/selection workload diagnostics and per-query range diagnostics
- range label diagnostics, label Oracle diagnostics, and Random,
  newUniformTemporal, and DouglasPeucker baseline diagnostics
- stable output artifacts:
  `range_workload_diagnostics.json`, `range_query_diagnostics.jsonl`, and
  `example_run.json.workload_diagnostics`
- CLI/config wiring and regression tests for the Phase 2 behavior

Validation summary:

- full test suite passed after implementation
- cleaned-CSV Phase 2 smoke with range filters produced healthy train/eval
  diagnostics
- F1 checkpoint-selection smoke produced healthy train/eval/selection
  diagnostics
- scaled cleaned-CSV runs, including F1-vs-loss checkpoint selection at 384
  segments, 160 points per segment, and 192 queries, preserved healthy workload
  diagnostics

Deferred:

- workload/label caching remains optional and was not added in Phase 2; the
  existing segmented CSV cache is sufficient for the completed diagnostics and
  smoke validation.

## Current Code Facts

- Range queries are created in `src/queries/query_generator.py` by sampling an
  anchor point, then building a spatiotemporal box around it.
- Because the anchor point is inside the generated box, empty range boxes should
  be rare/impossible for normal non-empty point clouds. The actual risk is broad
  boxes, redundant boxes, and weak point-hit distributions.
- Coverage mode already treats `n_queries` as a minimum and can use
  `max_queries` only to exceed that count when needed for target coverage.
- Range evaluation is point-level in `src/evaluation/evaluate_methods.py`.
  A method gets credit for preserved range-hit points, not just for preserving
  one point from a matching trajectory.
- Range labels in `src/training/importance_labels.py` currently mirror
  point-hit F1 contribution: every in-box point gets the singleton point-hit
  recovery gain for that query.
- Existing tests already cover coverage generation, range point evaluation, and
  range label semantics.

## Implementation Package

### 1. Range Query Diagnostics

Add a small diagnostics module, preferably:

```text
src/queries/workload_diagnostics.py
```

Core function:

```python
compute_range_workload_diagnostics(points, boundaries, typed_queries) -> dict
```

Per-query fields:

- `point_hits`
- `trajectory_hits`
- `point_hit_fraction`
- `trajectory_hit_fraction`
- `lat_span_fraction`
- `lon_span_fraction`
- `time_span_fraction`
- `box_volume_fraction`
- `is_empty`
- `is_too_broad`
- `near_duplicate_of`

Aggregate fields:

- `range_query_count`
- `empty_query_rate`
- `too_broad_query_rate`
- `near_duplicate_query_rate`
- `point_hit_count_p10/p50/p90`
- `trajectory_hit_count_p10/p50/p90`
- `point_hit_fraction_p10/p50/p90`
- `coverage_fraction`

### 2. Range Acceptance Filters

Extend range generation with optional acceptance controls. Keep defaults
compatible enough that existing smoke tests still pass.

Candidate controls:

```text
range_min_point_hits
range_max_point_hit_fraction
range_min_trajectory_hits
range_max_trajectory_hit_fraction
range_max_box_volume_fraction
range_duplicate_iou_threshold
range_acceptance_max_attempts
```

The generator should attempt candidates until it accepts enough queries or hits
the attempt limit. If it cannot satisfy strict filters, it should return the
best accepted set and emit diagnostics/warnings rather than silently producing
trivial workloads.

Suggested first defaults for real range runs:

```text
range_min_point_hits = 2
range_max_point_hit_fraction = 0.20
range_min_trajectory_hits = 1
range_max_trajectory_hit_fraction = 0.30
range_max_box_volume_fraction = 0.05
range_duplicate_iou_threshold = 0.85
range_acceptance_max_attempts = 50 * n_requested_range_queries
```

Use more permissive defaults in code if needed for backward compatibility, then
activate stricter settings through CLI for Phase 2 experiments.

### 3. Label And Baseline Diagnostics

Add diagnostics that connect workload quality to learning signal:

- range positive-label fraction
- range label p50/p90/p95/max
- count of points with nonzero range labels
- label Oracle range F1 at the target compression ratio
- Random range F1
- newUniformTemporal range F1
- DouglasPeucker range F1
- Oracle gap over best baseline

The fastest implementation can reuse:

- `compute_typed_importance_labels`
- `OracleMethod`
- `RandomMethod`
- `NewUniformTemporalMethod`
- `DouglasPeuckerMethod`
- `evaluate_method`

### 4. Output Artifacts

Write Phase 2 diagnostics under the run output directory:

```text
artifacts/results/<run-name>/range_workload_diagnostics.json
artifacts/results/<run-name>/range_query_diagnostics.jsonl
```

The JSON summary should be included in `example_run.json` under a stable key,
for example:

```json
"workload_diagnostics": {
  "train": {...},
  "eval": {...},
  "selection": {...}
}
```

### 5. Optional Workload Cache

Do not make workload caching the first blocker. Phase 2 can start with
diagnostics output. Once diagnostics are stable, add:

```text
src/queries/workload_cache.py
```

Cache key should include:

- source dataset identity or point-cloud fingerprint
- boundaries fingerprint
- query generation config
- workload mix
- seed
- query type
- acceptance thresholds

Cached payload:

- typed query dictionaries
- padded query features
- type IDs
- coverage metadata
- diagnostics summary

Labels can be cached separately after workload diagnostics are stable.

## Test Plan

Add focused tests before relying on Phase 2 runs:

1. `test_range_workload_diagnostics_reports_hit_distributions`
2. `test_range_diagnostics_marks_broad_queries`
3. `test_range_diagnostics_marks_near_duplicate_boxes`
4. `test_range_acceptance_rejects_overly_broad_queries`
5. `test_range_acceptance_keeps_requested_query_count_when_possible`
6. `test_range_label_diagnostics_reports_positive_fraction`
7. `test_phase2_diagnostics_dump_is_json_serializable`

Keep existing tests passing:

```bash
make test
```

## Phase 2 Smoke Commands

Synthetic diagnostics smoke:

```bash
python -m src.experiments.run_ais_experiment \
  --n_ships 6 \
  --n_points 80 \
  --workload range \
  --n_queries 64 \
  --query_coverage 0.30 \
  --range_spatial_fraction 0.02 \
  --range_time_fraction 0.04 \
  --epochs 1 \
  --compression_ratio 0.20 \
  --results_dir artifacts/results/phase2_range_synthetic_smoke
```

Cleaned-CSV diagnostics smoke:

```bash
python -m src.experiments.run_ais_experiment \
  --csv_path ../../AISDATA/cleaned/aisdk-2026-01-01.cleaned.csv \
  --cache_dir artifacts/cache/phase2_range_smoke \
  --max_segments 32 \
  --max_points_per_segment 96 \
  --workload range \
  --n_queries 64 \
  --query_coverage 0.30 \
  --range_spatial_fraction 0.02 \
  --range_time_fraction 0.04 \
  --epochs 1 \
  --compression_ratio 0.20 \
  --results_dir artifacts/results/phase2_range_cleaned_smoke
```

After diagnostics pass, use Phase 3 settings with validation F1 selection:

```bash
python -m src.experiments.run_ais_experiment \
  --train_csv_path <train-day-or-days> \
  --eval_csv_path <eval-day> \
  --cache_dir artifacts/cache/phase2_range \
  --workload range \
  --n_queries 250 \
  --query_coverage 0.30 \
  --range_spatial_fraction 0.02 \
  --range_time_fraction 0.04 \
  --checkpoint_selection_metric f1 \
  --f1_diagnostic_every 1 \
  --epochs 8 \
  --lr 0.0005 \
  --pointwise_loss_weight 0.25 \
  --gradient_clip_norm 1.0 \
  --compression_ratio 0.20 \
  --results_dir artifacts/results/phase2_range_validation
```

## Acceptance Criteria Before Phase 3

Phase 2 should be considered ready for Range-QDS training only when:

- generated range workload has exactly the intended query count unless strict
  filters make that impossible and the run reports why
- empty-query rate is near zero
- too-broad-query rate is below the configured threshold
- near-duplicate-query rate is below the configured threshold
- point-hit and trajectory-hit distributions are saved
- positive range-label fraction is nonzero and not dominated by a tiny handful
  of points
- label Oracle range F1 is clearly above Random, newUniformTemporal, and
  DouglasPeucker at the same compression ratio
- diagnostics are written for train, eval, and checkpoint-selection workloads
  when those workloads exist
- `make test` passes

## Recommended Implementation Order

1. Add pure range diagnostics module and tests.
2. Wire diagnostics into `run_experiment_pipeline` output JSON.
3. Add label/oracle/baseline diagnostics.
4. Add optional acceptance filters to range generation.
5. Add CLI/config fields for acceptance thresholds.
6. Run synthetic and cleaned-CSV Phase 2 smokes.
7. Only then add workload/label caching if repeated diagnostics become slow.
