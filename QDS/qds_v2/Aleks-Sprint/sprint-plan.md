# Sprint Plan: Query-Driven Simplification Models

## Purpose

This sprint plan turns `sprint-ambitions.md` into a realistic execution sequence.

The ambition document describes the full research direction. This plan narrows that into a practical sprint target:

> Build a reliable specialist-model pipeline and prove the first specialist wins before attempting broad claims across all four query workloads.

The sprint should prioritize Range-QDS and kNN-QDS because they are the most local, inspectable, and likely to produce a defensible first result. Similarity-QDS and Clustering-QDS should still be prepared, but they should be treated as secondary or stretch targets unless Range-QDS and kNN-QDS are already stable.

## Current Merge Position

Status on 2026-05-08 after incorporating the non-artifact parts of
`V2_Revamp`:

- Phase 0, Phase 1, and Phase 2 remain the accepted local baseline for this
  branch.
- The `V2_Revamp` result artifacts are useful historical/reference outputs, but
  they do not replace the Phase 2 diagnostics or define the Phase 3 benchmark
  protocol.
- The useful incoming code changes are checkpoint smoothing, explicit MLQDS
  gaps against baselines, length-preservation reporting, `AnswerF1` versus
  `CombinedF1` reporting, selectable validation F1 variant, per-head
  rank-normalized MLQDS score mixing, true recursive Douglas-Peucker, large
  tensor sampling/quantile safeguards, optional stationary trimming in the
  upstream AIS cleaning pipeline, and a multi-file CSV combine helper.
- The incoming baseline rename from `newUniformTemporal` to `uniform` and the
  removal of `Random` from matched evaluation are canonical. Phase 3 matched
  runs should compare MLQDS against `uniform`, Douglas-Peucker, and label
  Oracle at equal compression.
- The incoming range sqrt-normalization change is intentionally not carried
  forward. Range labels should normalize by query count like the other query
  types until a measured Phase 3 run justifies changing label scale.
- Multi-day CSV combination should preserve MMSIs by default and let the
  segmented loader split by timestamp gaps; MMSI offsetting is only a
  compatibility option.

## Realistic Sprint Targets

### Primary Targets

1. **Make the data pipeline trustworthy**

   Add trajectory segmentation, cached preprocessing, and data audit outputs so experiments train on validated trajectory segments rather than raw MMSI-day tracks.

2. **Build validated Range-QDS and kNN-QDS workloads**

   Add query diagnostics and acceptance filters so generated range and kNN workloads are non-empty, non-trivial, and informative.

3. **Train and benchmark Range-QDS**

   Produce a repeatable Range-QDS benchmark where Range-QDS beats `uniform` and true Douglas-Peucker at equal compression across at least 3 seeds.

4. **Train and benchmark kNN-QDS**

   Produce the same kind of benchmark for kNN-QDS, or document exactly which data/query/label issue blocks it.

5. **Create the benchmark harness needed for specialist claims**

   Add a benchmark runner that can evaluate specialist models against baselines using fixed splits, repeated seeds, best-baseline gaps, label Oracle gaps, and geometry metrics.

### Secondary Targets

1. Add initial workload diagnostics for Similarity-QDS and Clustering-QDS.
2. Identify whether similarity and clustering fail because of query generation, labels, model conditioning, or simplification policy.
3. Add visual debugging outputs for retained vs removed points and per-query failures.
4. Validate the true recursive Douglas-Peucker baseline at larger AIS scale.

### Stretch Targets

1. Get Similarity-QDS to beat baselines on a small controlled workload.
2. Get Clustering-QDS to beat baselines on a small controlled workload.
3. Add learning-curve experiments across `3 -> 5 -> 10` cleaned AIS days.
4. Add a prototype global or workload-aware budget allocation policy.

## Non-Goals For This Sprint

These are important, but should not block the realistic sprint target:

- proving all four specialist models on full 30-day data
- training on `10k-50k` queries directly through full cross-attention
- building a final universal mixed-workload model
- solving fully differentiable final query F1 training
- making broad research claims from single-seed runs

## Working Dataset Plan

Use the existing cleaned AIS days as the main sprint dataset.

Recommended 10-day split:

```text
Train: Jan 01-Jan 07
Validation: Jan 08
Test: Jan 09-Jan 10
```

Use smaller splits only for debugging:

```text
Debug: 3 cleaned days
Minimum viable: 5 cleaned days
Sprint target: 10 cleaned days
```

Every final benchmark run should record:

- train days
- validation days
- test days
- segmentation config
- query config
- seeds
- compression ratio
- model config
- baseline versions

## Benchmark Protocol

### Minimum Protocol

```text
Models: Range-QDS, kNN-QDS
Baselines: uniform, true Douglas-Peucker, label Oracle
Seeds: at least 3
Compression: same ratio for all methods
Splits: fixed train/validation/test days
Metrics: target workload F1, best-baseline gap, label Oracle gap, SED/PED, length preservation
```

### Preferred Protocol

```text
Models: Range-QDS, kNN-QDS, Similarity-QDS, Clustering-QDS
Baselines: uniform, true Douglas-Peucker, label Oracle
Seeds: 5
Data volumes: 3, 5, 10 days
Reporting: mean, standard deviation, best-baseline gap, label Oracle gap, learning curves
```

## Sequence Of Work

### Phase 0: Environment And Guardrails

Goal: make the project runnable and repeatable before adding more experiments.

Status on 2026-05-06: complete for the local sprint setup. `QDS/qds_v2`
now has pinned requirements, a local Makefile with `check-env`, `test`,
`smoke`, and `smoke-csv` targets, documented environment commands in the root
qds_v2 README, and ignored `artifacts/` output for future local runs.

Tasks:

1. Create one documented Python environment for `qds_v2`.
2. Install a compatible `torch`, `pandas`, `numpy`, and `pytest`.
3. Pin dependency versions or add a reproducible environment file.
4. Add a simple test command.
5. Add a simple small benchmark smoke command.
6. Move or ignore generated checkpoints outside `src/models/saved_models` for future outputs.

Acceptance checks:

- tests can run locally
- one tiny synthetic experiment can run
- one tiny cleaned-CSV smoke experiment can run
- generated artifacts do not pollute source code paths

Deliverable:

- documented environment and smoke-test command

### Phase 1: Data Segmentation And Cached Preprocessing

Status: Completed for the current sprint scope on 2026-05-06.

Goal: stop treating one MMSI-day as the default trajectory unit.

Tasks:

1. Add trajectory segmentation by MMSI plus time gap.
2. Add configurable segment controls:
   - `min_points_per_segment`
   - `max_points_per_segment`
   - `max_time_gap_seconds`
   - `max_segments`
3. Add data audit output:
   - rows loaded
   - rows dropped
   - invalid COG count
   - duplicate timestamp count
   - segment count
   - segment length distribution
   - time-gap distribution
4. Add cached preprocessing artifacts:

   ```text
   cleaned CSV -> validated trajectory segments -> cached tensors/parquet/pt artifacts
   ```

5. Add real cleaned-CSV smoke tests using a small slice.

Acceptance checks:

- cached segments can be reused across runs
- train/validation/test splits are day-based and explicit
- segment stats are saved with every run
- long MMSI tracks with large gaps are split

Deliverable:

- reusable segmented dataset cache for the 10-day sprint split

Completion note:

- Implemented MMSI/time-gap segmentation, configurable segment controls, run-level data audit output, Parquet-backed segmented CSV cache, unit tests, and a cleaned-CSV cache smoke command. The cache is reusable per source CSV and segmentation config; building the full 10-day split cache remains a scale-up run using the completed cache API.

### Phase 2: Range Workload And Label Diagnostics

Goal: make Range-QDS training data informative enough to beat uniform sampling.

Tasks:

1. Add range query acceptance filters:
   - reject empty boxes
   - reject overly broad boxes
   - target useful hit-count bands
   - limit near-duplicate boxes
2. Add range diagnostics:
   - point-hit distribution
   - trajectory-hit distribution
   - positive label fraction
   - label Oracle F1
   - uniform F1
3. Add cached range workloads and labels.
4. Add visual debug output for range queries:
   - query box
   - original hits
   - retained MLQDS points
   - retained baseline points
5. Review range labels for redundancy.

Acceptance checks:

- generated range workloads are not empty, trivial, or too broad
- label Oracle is clearly above baselines
- Range-QDS validation workload is stable across seeds

Deliverable:

- validated Range-QDS workload generator and cached workload artifacts

Status:

- Completed 2026-05-07.

Completion note:

- Implemented optional range acceptance filters for point hits, trajectory hits,
  broad boxes, duplicate boxes, and acceptance-attempt exhaustion.
- Added train/eval/selection range workload diagnostics, per-query JSONL output,
  range label diagnostics, Oracle diagnostics, and baseline signal diagnostics.
- Wrote diagnostics to `range_workload_diagnostics.json`,
  `range_query_diagnostics.jsonl`, and `example_run.json`.
- Validated with cleaned-CSV smoke runs, F1/loss checkpoint-selection comparison,
  and scaled range smokes up to 384 segments, 160 points per segment, and 192
  queries.
- Phase 2 acceptance checks passed in those runs: intended query counts were met,
  empty/broad/duplicate query rates were zero, positive label signal was present,
  label Oracle stayed clearly above baselines, diagnostics were written for
  train/eval/selection workloads, and the test suite passed.
- Workload/label caching was intentionally deferred; Phase 2 can proceed to
  Phase 3 with generated workload diagnostics and the existing segmented CSV
  cache.

### Phase 3: Range-QDS Training And Benchmark

Goal: produce the first defensible specialist win.

Entry state:

- Range workload diagnostics are complete and accepted.
- The benchmark output now reports `AnswerF1`, `CombinedF1`, SED/PED,
  `LengthPres`, `F1xLen`, and MLQDS F1 gaps versus `uniform` and true
  Douglas-Peucker.
- Checkpoint selection can use a rolling diagnostic score through
  `checkpoint_smoothing_window`, but smoothing is a selection stabilizer, not a
  training objective.
- Checkpoint selection can use `checkpoint_f1_variant=answer` for the primary
  pure answer-set metric or `checkpoint_f1_variant=combined` for the legacy
  answer/support product. Phase 3 should default to `answer` unless the run is
  explicitly testing support-preservation selection.
- MLQDS matched evaluation now mixes query-type heads by within-trajectory
  ranks before applying workload weights. This prevents one uncalibrated head
  from dominating mixed-workload scores.

Tasks:

1. Train Range-QDS on the validated range workload.
2. Use validation query F1 or uniform-gap checkpoint selection, with
   `checkpoint_smoothing_window` tested on noisy runs.
3. Track:
   - prediction spread
   - positive fraction
   - label p95
   - ranking pairs
   - skipped windows
   - validation F1
   - validation gap to uniform
4. Evaluate against:
   - uniform
   - true Douglas-Peucker
   - label Oracle
5. Run at least 3 seeds.
6. Report mean, standard deviation, best-baseline gap, label Oracle gap, and geometry.
7. Compare pure learned scoring against temporal-hybrid scoring.

Acceptance checks:

- Range-QDS beats the best baseline on range F1 on average across seeds
- the win is not caused only by an oversized temporal base
- geometry remains within acceptable bounds
- label Oracle remains meaningfully above Range-QDS

Deliverable:

- Range-QDS benchmark card

### Phase 4: kNN Workload And Label Diagnostics

Goal: make kNN queries local and discriminative enough for kNN-QDS to learn.

Tasks:

1. Add configurable kNN time-window distribution.
2. Add configurable `k` distribution.
3. Add answer-cardinality diagnostics.
4. Add distance-margin diagnostics to identify ambiguous queries.
5. Reject queries where many trajectories are nearly tied.
6. Split kNN diagnostics by dense-port and open-water regions.
7. Cache kNN workloads and labels.
8. Add visual debug output for kNN:
   - query anchor
   - returned original trajectories
   - nearest representative points
   - retained MLQDS and baseline points

Acceptance checks:

- kNN workloads are not dominated by broad 6-hour windows
- query answer sets are stable and non-trivial
- label Oracle is clearly above baselines
- kNN labels focus on representative points that preserve nearest-neighbor membership

Deliverable:

- validated kNN-QDS workload generator and cached workload artifacts

### Phase 5: kNN-QDS Training And Benchmark

Goal: attempt the second defensible specialist win.

Tasks:

1. Train kNN-QDS on validated kNN workloads.
2. Use validation query F1 or uniform-gap checkpoint selection.
3. Evaluate at the same compression ratio as baselines.
4. Run at least 3 seeds.
5. Report mean, standard deviation, best-baseline gap, label Oracle gap, and geometry.
6. Inspect failure cases where kNN-QDS loses to `uniform` or Douglas-Peucker.

Acceptance checks:

- kNN-QDS beats the best baseline on kNN F1 across seeds
- or the benchmark identifies a concrete blocker:
  - query ambiguity
  - weak labels
  - insufficient trajectory segmentation
  - simplification budget issue
  - model-conditioning scale issue

Deliverable:

- kNN-QDS benchmark card or kNN blocker report

### Phase 6: Benchmark Harness

Goal: make specialist claims repeatable instead of manually assembled.

Tasks:

1. Add a benchmark runner for specialist models.
2. Support this matrix:

   ```text
   Train model       Eval: range   Eval: kNN   Eval: similarity   Eval: clustering
   Range-QDS
   kNN-QDS
   Similarity-QDS
   Clustering-QDS
   ```

3. Evaluate all baselines in every active cell.
4. Add multi-seed aggregation.
5. Add result fields:
   - best baseline
   - gap to best baseline
   - gap to label Oracle
   - per-type F1
   - geometry metrics
   - latency
6. Persist per-query scores and query diagnostics.
7. Produce a readable markdown or JSON benchmark summary.

Acceptance checks:

- a single command can run the Range-QDS benchmark
- a single command can run the kNN-QDS benchmark
- result summaries are comparable across seeds and workloads

Deliverable:

- repeatable specialist benchmark runner

### Phase 7: Similarity And Clustering Scoping

Goal: prepare the later specialists without letting them consume the whole sprint.

Tasks:

1. Add similarity diagnostics:
   - reference snippet length
   - answer count
   - distance separation
   - hard-negative candidates
2. Add clustering diagnostics:
   - cluster count
   - noise fraction
   - co-membership pair count
   - eps sensitivity
3. Identify whether query generation or label construction is the immediate bottleneck.
4. Build small controlled workloads for each type.
5. Run one smoke training pass for each if Range-QDS and kNN-QDS are stable.

Acceptance checks:

- similarity and clustering workloads have clear diagnostics
- blockers are explicitly documented
- no broad claims are made from unstable runs

Deliverable:

- Similarity-QDS and Clustering-QDS readiness report

## Task Board

| ID | Task | Priority | Output |
| --- | --- | --- | --- |
| P0.1 | Fix Python environment with torch and pytest | Critical | runnable local environment |
| P0.2 | Add smoke-test commands | Critical | documented commands |
| P1.1 | Add MMSI/time-gap segmentation | Critical | segmented trajectories |
| P1.2 | Add cached preprocessing artifacts | Critical | reusable dataset cache |
| P1.3 | Add data audit report | Critical | run-level data stats |
| P2.1 | Add range query acceptance filters | Critical | validated range workloads |
| P2.2 | Add range workload/label diagnostics | Critical | range diagnostics JSON |
| P3.0 | Reconcile V2_Revamp benchmark/checkpoint changes with Phase 2 baseline contract | Critical | done; merge-ready Phase 3 code |
| P3.1 | Train Range-QDS with validation F1 or uniform-gap selection | Critical | trained Range-QDS runs |
| P3.2 | Run 3-seed Range-QDS benchmark | Critical | Range-QDS benchmark card |
| P4.1 | Add configurable kNN time windows and k values | High | validated kNN config |
| P4.2 | Add kNN ambiguity and distance-margin diagnostics | High | kNN diagnostics JSON |
| P5.1 | Train kNN-QDS | High | trained kNN-QDS runs |
| P5.2 | Run 3-seed kNN-QDS benchmark | High | kNN-QDS benchmark card |
| P6.1 | Add specialist benchmark runner | High | repeatable benchmark command |
| P6.2 | Add multi-seed aggregation | High | mean/std benchmark report |
| P6.3 | Add best-baseline and label Oracle gap reporting | High | comparable result cards |
| P7.1 | Scope similarity workload blockers | Medium | readiness report |
| P7.2 | Scope clustering workload blockers | Medium | readiness report |
| P8.1 | Validate true recursive Douglas-Peucker baseline on larger AIS runs | Medium | scale/performance check |
| P8.2 | Add visual debug outputs | Medium | per-query inspection plots |

## Milestones

### Milestone 1: Runnable And Segmented

The project can run tests and smoke experiments. AIS data is segmented into validated trajectory artifacts with audit reports.

Exit criteria:

- environment works
- segmentation works
- cached dataset exists
- cleaned-CSV smoke test passes

### Milestone 2: Range-QDS Result

Range workload generation is validated, Range-QDS trains, and the benchmark produces a repeatable 3-seed result.

Exit criteria:

- range workload diagnostics pass
- Range-QDS benchmark card exists
- Range-QDS beats or clearly fails against best baseline with documented reason

### Milestone 3: kNN-QDS Result

kNN workload generation is validated, kNN-QDS trains, and the benchmark produces a repeatable 3-seed result or a concrete blocker report.

Exit criteria:

- kNN workload diagnostics pass
- kNN-QDS benchmark card or blocker report exists
- failure cases are inspectable through per-query diagnostics

### Milestone 4: Specialist Benchmark Harness

The project can run specialist benchmarks and aggregate results across seeds.

Exit criteria:

- one command for Range-QDS benchmark
- one command for kNN-QDS benchmark
- result cards include best-baseline gap and label Oracle gap
- geometry metrics are included

## Reporting Template

Each specialist result should be reported in this shape:

```text
Model: Range-QDS
Target workload: range
Train days:
Validation days:
Test days:
Segmentation config:
Query config:
Compression ratio:
Seeds:

Mean target F1:
Std target F1:
Best baseline:
Best baseline F1:
Gap to best baseline:
Label Oracle F1:
Gap to label Oracle:
Avg SED/PED:
Length preserved:
F1xLen:
Latency:

Conclusion:
Known blockers:
Next action:
```

## Risk Management

### Risk: Data segmentation changes the whole training distribution

Mitigation:

- compare old MMSI-day results against segmented results on a small split
- keep segmentation config in every result file
- inspect segment length distributions before training

### Risk: Range-QDS only wins because of temporal hybrid

Mitigation:

- report pure learned scoring and temporal-hybrid scoring separately
- vary `mlqds_temporal_fraction`
- require a measured learned contribution beyond uniform temporal

### Risk: kNN labels remain noisy

Mitigation:

- use shorter and configurable time windows
- reject ambiguous nearest-neighbor queries
- inspect distance margins
- visualize representative retained points

### Risk: Query volume becomes too expensive

Mitigation:

- cache workloads and labels
- start with smaller debug workloads
- use query sampling or prototypes before increasing query counts

### Risk: Benchmarks are too slow for multi-seed runs

Mitigation:

- run 3-day debug experiments first
- cache all reusable artifacts
- separate smoke benchmarks from final 10-day benchmarks

## Final Sprint Definition Of Done

The sprint is successful if it produces:

1. A documented, runnable environment.
2. Cached segmented AIS data for the sprint split.
3. Validated range and kNN workload generators with diagnostics.
4. A repeatable Range-QDS benchmark across at least 3 seeds.
5. A repeatable kNN-QDS benchmark across at least 3 seeds, or a concrete kNN blocker report.
6. A specialist benchmark runner with best-baseline and label Oracle gap reporting.
7. Clear next-step diagnostics for Similarity-QDS and Clustering-QDS.

The best realistic outcome is:

```text
Range-QDS beats the strongest baseline.
kNN-QDS beats the strongest baseline or has a concrete, well-diagnosed blocker.
Similarity-QDS and Clustering-QDS have validated workload diagnostics and scoped next actions.
```

The stretch outcome is:

```text
Range-QDS and kNN-QDS both beat the strongest baseline across seeds.
Similarity-QDS or Clustering-QDS shows a controlled small-workload win.
```
