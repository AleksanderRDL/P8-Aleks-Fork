# Range Training Redesign

Current implementation reference for the Range-QDS redesign.

## Objective

Build a workload-blind range compressor:

1. train from historical/generated range workloads as supervision
2. compress validation/eval trajectories without seeing their future queries
3. freeze one retained set per dataset and compression ratio
4. evaluate the frozen set against held-out range workloads

The current `range_aware` profile remains useful as a diagnostic and teacher,
but it is workload-aware. It cannot be used as final workload-blind success
evidence.

## Hard Rule

Query context is allowed while building training supervision. Query context is
not allowed when compressing validation/eval data for final claims.

Allowed:

- train labels from train workloads
- query-aware teachers on train workloads
- validation workloads for checkpoint selection after blind validation
  compression

Forbidden for final claims:

- eval queries passed into model, feature builder, or selector before retaining
  points
- eval point/query relation features
- query cross-attention during eval compression
- checkpoint selection on final eval-query performance

## Target User Outcome

Future range queries over simplified AIS data should still answer what ships
were present and how they moved inside the query window.

Important retained behavior:

- ship presence and per-ship point coverage
- entry/exit and boundary-crossing evidence
- temporal span without one large missing run
- turns, heading/speed changes, and local route shape
- reasonable global geometry and length preservation

Global trajectory geometry is secondary. It is still reported because poor
global geometry can make outputs nonsensical, but the primary product target is
future range-query usefulness.

Point-hit coverage matters, but it is not enough by itself.

## Evaluation Grid

Coverage targets:

- `5%`
- `10%`
- `15%`
- `30%`

Compression targets:

- `1%`
- `2%`
- `5%`
- `10%`
- `15%`
- `20%`
- `30%`

Coverage is a workload-generation target. Compression is the retained-point
budget. They are independent axes.

Each final grid cell must:

1. train on train data and train workloads
2. compress validation/eval trajectories blind
3. evaluate held-out queries only after masks are frozen
4. compare against uniform, Douglas-Peucker, temporal random fill where
   relevant, and workload-aware diagnostic upper bounds

## Metrics

Primary metric: `RangeUseful`.

Always report:

- `RangePointF1`
- `ShipF1`
- `ShipCov`
- `EntryExitF1`
- `CrossingF1`
- `TemporalCov`
- `GapCov`
- `TurnCov`
- `ShapeScore`
- global SED/PED-style distortion
- length preservation
- compression ratio
- latency/runtime

`RangeUseful` is the best current aggregate but still a versioned proxy. Report
components so wins and regressions are inspectable.

`GapCov` currently means index/count continuity: the largest run of missing
in-query point positions is normalized by the number of in-query points. This is
simple and stable, but it can misrepresent irregular AIS sampling. A missing run
of ten dense points is not equivalent to a missing run spanning hours or many
kilometers.

Required `GapCov` ablation:

- keep current count-normalized `GapCov` as the baseline metric
- add `GapCovTime`: largest missing time span normalized by full in-query time span
- add `GapCovDistance`: largest missing along-track distance span normalized by
  full in-query path length
- Maybe `gap_min` = min(`gap_time`, `gap_distance`) if strict continuity is important
- compare variants by final `RangeUseful`, component regressions, and checkpoint
  decisions before changing the default aggregate

## Baselines And Diagnostics

Final baselines:

- uniform temporal sampling
- Douglas-Peucker
- temporal random fill when evaluating learned residual fill

Diagnostics and upper references:

- `range_aware` model
- direct range-geometry labels
- query-aware teacher scores
- temporal oracle fill

Keep diagnostic results separate from final workload-blind claims.

## Required Model Direction

Add a workload-blind model type, likely `range_prior` or
`workload_blind_range`.

The blind model must:

- score points without eval queries
- avoid query cross-attention during compression
- avoid `range_aware` point/query relation features during compression
- produce reusable retained masks for every compression ratio
- support the full compression grid from one score vector where possible

Allowed compression-time features:

- time, lat, lon, speed, heading/course features
- turn score, gap size, speed/heading change, local density, endpoints, segment position
- historical train-derived priors that do not use current eval queries

Forbidden compression-time features:

- current eval query boxes
- containment in current eval queries
- distance to current eval query boundaries
- attention over current eval query embeddings

## Training Targets

The core strategy is query-conditional supervision for a query-blind student.

### Expected-Usefulness Labels

Build first.

1. Generate many training range workloads.
2. Compute point usefulness from range audit components for each workload.
3. Aggregate usefulness per point across workloads.
4. Train the blind model to predict expected future usefulness.

Aggregation variants:

- mean usefulness
- max usefulness
- positive-usefulness frequency
- budget-normalized usefulness
- ship-balanced usefulness
- component-balanced usefulness

The label must not collapse into "inside many boxes." It needs to preserve the
same concepts as `RangeUseful`.

### Teacher-Student Distillation

Build after the blind pipeline exists.

1. Train or reuse `range_aware` as query-aware teacher.
2. Score many train workloads.
3. Aggregate teacher signal per point.
4. Train the blind student on the aggregate target.

Teacher variants:

- raw teacher score mean
- per-workload rank-percentile mean
- top-k membership frequency across budgets
- teacher score blended with explicit `RangeUseful` labels

Improving the teacher can improve the student, but only if the improvement is
compressible into query-blind features. A better `range_aware` teacher should
produce cleaner supervision on training workloads: better ranking, better
budget calibration, and better boundary/crossing/shape scores. That can reduce
target noise.

The limit is the distillation gap:

- teacher target: what is useful for this known workload
- blind student target: what is generally likely to be useful for future
  workloads

The student can learn repeated priors such as dense traffic areas, common
entry/exit zones, crossing-heavy routes, port approaches, trajectory endpoints,
turns, gaps, and local route structure. It cannot learn exact future query-box
membership unless that membership is statistically predictable from blind
features.

Practical sequence:

1. Build the blind student pipeline with the current strong `range_aware`
   teacher.
2. Measure teacher quality, student quality, student-vs-teacher target fit, and
   blind validation `RangeUseful`.
3. Improve the teacher if the student tracks teacher quality and still has a
   clear gap below it.
4. Improve blind features, workload generation, aggregation, or label design if
   the student cannot fit or transfer the teacher signal.

Do not over-invest in teacher improvements before proving the student can use
the teacher signal. A better teacher only matters if its information can be
learned without future query inputs.

### Marginal-Gain Targets

Use later if independent labels are too weak.

- estimate each point's marginal contribution under retained budgets
- account for redundancy with nearby useful points
- train on set-aware targets instead of independent point labels

This is more aligned but more expensive.

### Pretraining

Pretraining is allowed only if it improves blind range compression without
leaking validation/eval queries into compression.

Relevant pretraining candidates:

- trajectory-dynamics pretraining: mask or predict local deltas, speed, heading,
  turn score, and gap structure before supervised range training
- range-teacher distillation pretraining: aggregate many query-aware teacher
  scores over train workloads, then initialize the blind student from that target
- component curriculum: train first on simpler range components such as temporal
  coverage, entry/exit, turn, gap, and shape, then fine-tune on the full
  `RangeUseful` target

Low-priority pretraining:

- generic reconstruction or autoencoding unless it proves transfer to
  `RangeUseful`
- any objective that mostly teaches point density without improving boundary,
  gap, shape, or ship-level utility

Decision rule: add pretraining only after the first blind pipeline exists, and
keep it only if it improves held-out blind `RangeUseful` at the target
compression ratios after the same checkpointing protocol.

## Loss And Checkpointing

Loss should match retained-budget use:

- train over multiple budgets, including the target compression grid
- prefer budget-top-k objectives over generic BCE
- monitor score collapse, score scale, and component mass

Checkpoint selection must stay workload-blind:

1. train candidate checkpoint
2. compress validation trajectories without validation queries
3. evaluate the frozen validation mask against validation workloads
4. select by validation `RangeUseful` or `uniform_gap`

`uniform_gap` means checkpoint score minus fair-uniform score on the same
validation workload and compression ratio. It does not replace `RangeUseful`.

## Workload Generation

The generator defines the prior the blind model learns. Final claims should not
depend on one narrow generator setting.

Required coverage targets:

- `5%`
- `10%`
- `15%`
- `30%`

Generator variants to test:

- footprint jitter
- multiple footprint families, not only fixed `2.2 km / 5 h`
- density-biased anchors and sparse/background anchors
- boundary-heavy and crossing-heavy anchors
- few-ship, many-ship, sparse, dense, short-window, and long-window buckets
- overlap limits to avoid near-duplicate queries
- held-out seeds and held-out generator settings

Time-domain rule for scale-up:

- Training queries on multi-day data must look structurally identical to
  single-day eval queries when eval remains day-based.
- Add an `anchor_day` time-domain mode: sample an anchor point, then clamp
  `t_start` / `t_end` to the calendar/source-file day containing that anchor.
- Prefer absolute time half-windows such as `range_time_hours=5.0` over
  dataset-relative time fractions for final range experiments.
- Do not let multi-day train bounds silently create multi-day query windows.

This is not cosmetic. If train queries can span multiple days but eval queries
cannot, the model learns a different query prior than the one used for final
claims.

First A/B set:

- fixed footprint
- `range_footprint_jitter=0.25`
- mixed footprint families

## Implementation Checklist

Protocol:

- add blind model type
- add query-free compression/inference path
- record `workload_aware` versus `workload_blind`
- fail loudly if final blind eval sees eval queries before compression

Targets:

- generate multiple train workloads per train day
- support `anchor_day` query time-domain generation for multi-day train data
- aggregate usefulness or teacher signal per point
- cache labels with schema version
- report component mass
- add count/time/distance `GapCov` label and metric variants behind explicit
  config
- add optional task-aligned pretraining stage only after the first blind model
  protocol is working

Benchmarks:

- run coverage/compression grid
- write retained masks before scoring
- evaluate held-out queries after masks exist
- report baselines separately from diagnostics

## Runtime Notes

Blind inference should be cheaper because compression no longer scales with
points times eval queries. Training may be more expensive because expected
usefulness needs many train workloads.

Quality-preserving optimizations:

- cache workloads
- cache range audit support
- cache aggregated blind labels
- reuse one score vector across compression-ratio audits
- vectorize label aggregation over point/query chunks

Approximate optimizations need ablation labels:

- sampled queries for label aggregation
- approximate spatial indexes
- approximate teacher aggregation

## Acceptance Criteria

The redesign is credible only if the workload-blind model:

- beats uniform and Douglas-Peucker on `RangeUseful` across most target cells
- remains competitive at `1%`, `2%`, and `5%` compression
- explains failures against temporal/random-fill baselines
- keeps global geometry and length preservation within acceptable limits
- generalizes across AIS days
- generalizes across held-out workload seeds
- survives at least one held-out generator setting

Expected outcome: the blind model will probably score below `range_aware`.
That is acceptable. The real target is beating blind baselines without future
query access.

## Immediate Plan

1. Implement workload-blind evaluation protocol.
2. Build expected-usefulness aggregated labels.
3. Add `anchor_day` query time-domain support before scaling to multi-day train
   datasets.
4. Train the first blind range model.
5. Evaluate the coverage/compression grid.
6. Compare against uniform, Douglas-Peucker, temporal random fill, and
   `range_aware` as a workload-aware upper bound.
7. Inspect component gaps, including count/time/distance `GapCov`, and label
   mass before increasing model size.
8. Add task-aligned pretraining or distillation if expected-usefulness
   labels are too weak.
