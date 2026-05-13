# Range Training Redesign

## Purpose

This document defines the next Range-QDS training redesign. It should be treated as the current implementation
reference.

The final target is workload-blind compression:

- train from historical/generated range workloads as supervision
- compress each eval dataset once without seeing future eval queries
- freeze the retained set
- evaluate that retained set against held-out range queries

The current `range_aware` experiments are still useful, but only as
workload-aware diagnostics. They show that range-query geometry contains strong
signal. They do not prove the final workload-blind compressor beats uniform or
Douglas-Peucker.

## Hard Rule

Query context is allowed during training. Query context is not allowed when
compressing eval/test data.

Allowed:

- generating train labels from train workloads
- using query-aware geometry to compute train targets
- using a query-aware teacher model on training workloads
- using validation workloads only for checkpoint selection after blind
  compression of the validation data

Not allowed for final claims:

- passing eval queries into the model before choosing retained points
- computing eval point/query relation features before compression
- using query cross-attention during eval compression
- selecting a checkpoint based on eval-query performance

The model may learn patterns from query semantics. It must not receive the
future query workload as an input at compression time.

## Target User Outcome

A user runs a future spatiotemporal range query over simplified AIS data. The
query result should preserve enough information to understand ships and their
movement inside the query window.

Important retained behavior:

- ships present in the query should remain represented
- each relevant ship should have enough retained points to interpret movement
- entry and exit behavior near the range boundary should survive
- boundary crossings should be inferable
- temporal coverage inside the queried interval should avoid one large missing
  run
- turns, heading/speed changes, and local route shape should remain usable
- statistical point-hit coverage still matters, but is not sufficient alone

Global trajectory geometry is secondary. It is still reported because poor
global geometry can make outputs nonsensical, but the primary product target is
future range-query usefulness.

## Evaluation Grid

Coverage targets:

- 5%
- 10%
- 15%
- 30%

Compression targets:

- 1%
- 2%
- 5%
- 10%
- 15%
- 20%
- 30%

Coverage targets are workload-generation targets. Compression targets are
retained-point budgets. They are separate axes.

For final claims, every cell must follow the workload-blind protocol:

1. train on train data and train workloads
2. compress validation/eval trajectories without validation/eval queries
3. evaluate the frozen retained masks against held-out range workloads
4. compare against uniform, Douglas-Peucker, and relevant temporal/random
   baselines at the same compression ratio

## Metrics

Primary comparison metric:

- `RangeUseful`

Required reported components:

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

`RangePointF1` is useful but too narrow. It measures retained in-box point
overlap. It does not by itself prove that ship movement remains interpretable.

`RangeUseful` is the best current aggregate, but it is still a versioned proxy.
Always report subcomponents so improvements are inspectable.

## Current Baselines And Diagnostics

Baselines that must stay in reports:

- uniform temporal sampling
- Douglas-Peucker
- temporal random fill when using a temporal spine
- temporal oracle fill where relevant as a residual-fill upper bound

Diagnostic upper bounds:

- `range_aware` model
- direct range-geometry labels
- query-aware teacher scores

These are not workload-blind final models. They answer a different question:
how much signal exists if the query workload is known.

## Required Model Direction

Add a workload-blind range model, likely named `range_prior` or
`workload_blind_range`.

The workload-blind model must:

- score points without eval queries
- avoid query cross-attention during compression
- avoid `range_aware` point/query relation features during compression
- produce one reusable retained set per dataset and compression ratio
- support multi-budget scoring across the target compression grid

Allowed input features at compression time:

- normalized time, lat, lon
- speed/course/heading-derived features already present in AIS data
- local trajectory features such as turn score, gap size, speed change, heading
  change, local density, endpoint flags, and segment position
- historical prior features that are computed without the current eval queries,
  such as train-derived spatial/temporal query-prior maps

Forbidden input features at eval compression time:

- current eval query boxes
- point containment in current eval queries
- distance to current eval query boundaries
- attention over current eval query embeddings

## Training Targets

The central idea is valid: use query-conditional context to build supervision,
then train a query-blind student.

### Expected-Usefulness Labels

First implementation path:

1. Generate many training range workloads.
2. For each workload, compute point usefulness from range audit components.
3. Aggregate each point's usefulness over workloads.
4. Train the blind model to predict expected future range usefulness from
   point/trajectory features only.

This is simple and should be built first.

Aggregation should be tested:

- mean usefulness across workloads
- max usefulness across workloads
- frequency of positive usefulness
- budget-normalized usefulness
- ship-balanced usefulness
- component-balanced usefulness

The label should not become only "was inside many boxes." It must preserve the
same concepts as `RangeUseful`: ship representation, boundary behavior,
crossings, gap coverage, turns, and shape.

### Teacher-Student Distillation

Second implementation path:

1. Train or reuse the current `range_aware` model as a query-aware teacher.
2. Run it over many train workloads.
3. Aggregate teacher scores per point into a query-blind target.
4. Train the blind student to imitate expected teacher usefulness.

This is more expensive but likely stronger. It may capture interactions the
local additive labels miss.

Distillation variants to test:

- raw teacher score mean
- per-workload rank percentile mean
- top-k membership frequency across budgets
- teacher score blended with explicit `RangeUseful` component labels

### Marginal-Gain Targets

Later path if local labels are insufficient:

- approximate each point's marginal contribution to `RangeUseful` under a
  retained budget
- compute labels after accounting for redundancy with already useful nearby
  points
- train on set-aware targets instead of independent point labels

This is likely more aligned but more expensive. Do not start here unless the
expected-usefulness and distillation paths fail.

## Loss And Checkpointing

The loss should match retained-budget use:

- train over multiple budgets, not only 5%
- include at least the compression grid `1%,2%,5%,10%,15%,20%,30%`
- prefer budget-top-k style objectives over generic pointwise BCE when possible
- monitor score collapse and score scale

Checkpoint selection must be workload-blind:

1. Train candidate checkpoint.
2. Compress validation trajectories without validation queries.
3. Evaluate the frozen validation retained set against validation workloads.
4. Select by validation `RangeUseful` or validation `uniform_gap`.

`uniform_gap` means checkpoint score minus uniform score on the same validation
workload and compression ratio. It does not replace `RangeUseful`; it uses
`RangeUseful` as the underlying metric when `checkpoint_f1_variant` is
`range_usefulness`.

## Workload Generation

The workload generator becomes more important in the blind setting. The model
will learn whatever prior the generator teaches it. If the generator is narrow
or unrealistic, the blind model will overfit that prior.

Required generator coverage:

- 5%
- 10%
- 15%
- 30%

Generator improvements to test:

- footprint jitter around spatial and temporal window sizes
- multiple footprint families, not only fixed 2.2 km / 5 hour boxes
- density-biased anchors for realistic traffic regions
- sparse/background anchors for generalization
- boundary-heavy anchors where many tracks enter/exit boxes
- crossing-heavy anchors where segment brackets cross query boundaries
- few-ship, many-ship, sparse, dense, short-window, and long-window buckets
- spatial/temporal overlap limits to avoid near-duplicate queries
- held-out generator seeds and held-out generator settings for robustness

Recommended first A/B:

- fixed footprint
- `range_footprint_jitter=0.25`
- mixed footprint families

Do not tune only to the current default generator. Final claims should include
held-out generator settings.

## Implementation Checklist

Protocol plumbing:

- add a workload-blind model type
- add an inference path that can compress with no workload/query input
- prevent final eval from calling `build_model_point_features(...,
  model_type="range_aware")`
- prevent final eval from using query cross-attention for blind model types
- record whether each run is `workload_aware` or `workload_blind`
- fail loudly if a final workload-blind benchmark sees eval queries before
  compression

Training target plumbing:

- generate multiple train workloads per train day
- aggregate point usefulness across workloads
- cache aggregated labels with a schema version
- support teacher-score aggregation as a separate target source
- report component mass so labels cannot silently collapse to point hits

Benchmark plumbing:

- run the requested coverage/compression grid
- write one retained mask per method and compression ratio before scoring
- evaluate held-out queries only after retained masks exist
- report baselines and diagnostic upper bounds separately
- keep exact final audit semantics

## Runtime And Optimization

The blind protocol may be cheaper at inference because it removes
points-by-queries feature construction for compression. Training may become
more expensive because many workloads are needed to estimate expected
usefulness.

Optimizations that should not change quality:

- cache generated workloads
- cache range audit support per workload
- cache aggregated blind labels per train split and label schema
- reuse model scores across compression-ratio audits
- evaluate multiple compression masks from one score vector
- vectorize label aggregation over point/query chunks

Approximate optimizations must be treated as ablations:

- sampled queries for label aggregation
- approximate spatial indexes
- approximate teacher-score aggregation

## Acceptance Criteria

The redesign is credible only if the workload-blind model:

- beats uniform and Douglas-Peucker on `RangeUseful` across most or all target
  coverage/compression cells
- still beats them at low budgets, especially 1%, 2%, and 5%
- beats or clearly explains failures against temporal/random-fill baselines
- keeps outputs sensical by not destroying length preservation and global
  geometry beyond acceptable limits
- generalizes across different AIS days
- generalizes across held-out workload-generator seeds
- survives at least one held-out generator setting, such as jitter or mixed
  footprint families

Expected outcome: the blind model will probably score below `range_aware`.
That is fine. The real target is beating simple blind baselines without seeing
future queries.

## Immediate Experiment Plan

1. Implement workload-blind evaluation protocol.
2. Build expected-usefulness aggregated labels from generated train workloads.
3. Train the first `range_prior` model with no query inputs at compression time.
4. Evaluate across the requested coverage/compression grid.
5. Compare against uniform, Douglas-Peucker, temporal random fill, and the
   workload-aware `range_aware` upper bound.
6. If it fails, inspect component gaps and label mass before changing model
   size.
7. Add teacher-student distillation if expected-usefulness labels are too weak.

The likely failure mode is that the blind student learns density and temporal
spacing but not boundary/crossing usefulness. The generator and label
aggregation should be designed to expose that early.
