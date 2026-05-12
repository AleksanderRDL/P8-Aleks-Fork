# Range Query Objective Redesign Notes

## Purpose

This note records the current project-level conclusion about what Range-QDS
should optimize. It is a research/design note, not a runnable command reference.

The end goal is to train a pure range-workload model that simplifies large AIS
trajectory datasets at multiple compression rates while preserving the
navigational usefulness of the data returned by realistic range queries.

## Target User Outcome

A user runs a spatiotemporal range query and receives simplified AIS points
inside that query. The simplified result should let the user understand the
ships and movement inside the requested area/time window as well as possible
for the available compression budget.

This is broader than retaining arbitrary in-box points. A good range-query
result should preserve:

- which ships appeared inside the query
- enough points per relevant ship to understand movement through the query
- entry and exit behavior near the query boundary
- turns, heading/speed changes, and other navigationally meaningful shape
  inside the query
- temporal coverage across the in-query segment instead of only dense bursts
- statistical point/query-hit coverage where that supports the above goals

Global geometric preservation is secondary. Shape matters primarily where it
helps interpret ships inside user queries.

## Compression Rates

The model should learn scores that remain useful across multiple retained-point
budgets, not only one fixed compression rate. Evaluation should include at
least:

- 1%
- 2%
- 5%
- 10%

The goal is to measure how far the model can hold up as compression becomes
more aggressive and whether query-driven training improves quality at each
budget.

## Current Metric And Audit

The previous range `AnswerF1` label was misleading. The metric is now reported
as `RangePointF1`: retained in-box point overlap. For each range query, it
scores how many original in-box point instances survived in the retained set.
Because simplified points are a subset of original points, this behaves like a
recall-derived point-hit metric.

That metric is useful, but incomplete. The evaluation path now also reports a
range usefulness audit:

- `RangePointF1`: retained in-box point-hit preservation
- `ShipF1`: whether ships/trajectories present in the query are still present
- `ShipCov`: average per-hit-ship point coverage
- `EntryExitF1`: whether range entry/exit points are retained
- `CrossingF1`: whether AIS point pairs bracketing range-box boundary
  crossings or between-sample pass-throughs are retained
- `TemporalCov`: how much of each in-query ship time span remains covered
- `GapCov`: whether retained in-query points avoid one large missing interior
  run
- `TurnCov`: whether high route-change points inside the query are retained
- `ShapeScore`: range-local route fidelity from shortcut error
- `RangeUseful`: weighted audit score over those components

Consequence: a model can improve the current training loss or point-hit proxy
without improving the actual user-facing usefulness of the simplified query
result.

### Audit Metric Status

The current audit components are useful first-pass diagnostics, not final
ground truth for the training objective:

- `ShipF1` measures whether every ship/trajectory present in a range query is
  still represented by at least one retained in-query point. It is correct for
  ship presence, but intentionally does not measure whether each ship has
  enough retained points to interpret movement.
- `ShipCov` averages point-subset F1 per hit ship, so a dense retained
  trajectory cannot hide a second ship that only has one retained point. It is
  still a point-count proxy, not a direct human interpretability metric.
- `EntryExitF1` measures whether sampled in-box entry/exit points are retained.
  It does not interpolate true geometric boundary crossings between AIS
  samples.
- `CrossingF1` measures retained AIS point pairs bracketing range-box boundary
  crossings and between-sample pass-throughs. It is closer to interpolation
  quality than sampled `EntryExitF1`, but it still scores retained brackets
  rather than exact crossing-time or crossing-location error.
- `TemporalCov` measures retained in-query time span divided by original
  in-query time span per ship. It is a useful coverage proxy, but intentionally
  does not penalize large interior gaps if endpoints are retained.
- `GapCov` penalizes the largest missing run between retained in-query points.
  It is a first-pass continuity proxy, not a full time/distance resampling
  metric.
- `TurnCov` scores weighted coverage of route-change points inside the query
  using local curvature and persisted turn-score features when available. It
  is useful for navigational interpretability, but still only measures sampled
  route-change points rather than semantic maneuver intent.
- `ShapeScore` measures range-local shortcut error using SED/PED-style
  fidelity. It is still a proxy, not a human route-quality metric.
- `RangeUseful` is a weighted audit score over these components. Treat it as a
  versioned comparison and checkpoint-selection diagnostic while the objective
  is being redesigned, not as the mathematically final utility target. Schema
  v7 weights are point `0.22`, ship presence `0.13`, ship coverage `0.13`,
  sampled entry/exit `0.10`, crossing brackets `0.10`, temporal span `0.10`,
  gap `0.09`, turn `0.07`, and shape `0.06`.

Before making these metrics central to the loss, add focused unit/property
tests for single-ship, multi-ship, no-hit, single-point-hit, duplicate-point,
boundary-entry/exit, crossing-bracket, interior-gap, turn, and curved-path
cases.

### Audit Runtime Direction

The exact final audit should stay exact. To reduce overhead without changing
semantics:

- precompute per-query full ship IDs, retained-independent entry/exit masks,
  crossing brackets, in-query offsets, turn weights, full time spans, and full
  local path lengths
- reuse this audit support across methods and compression ratios
- cache range masks through `EvaluationQueryCache`
- use sampled or cheaper approximations only for checkpoint selection if exact
  validation becomes too slow, while keeping final benchmark reporting exact

### Temporal Spine Finding

The `range_temporal_spine_sweep_20260512-073118` benchmark tested
`mlqds_temporal_fraction` values `0.25`, `0.50`, and `0.75` with the current
range-usefulness objective path. The `0.75` setting was best at every audited
compression ratio (`1%`, `2%`, `5%`, `10%`) and removed the collapse warning in
that run. It slightly beat Douglas-Peucker on `RangeUseful` at all audited
ratios, but still trailed the uniform temporal baseline.

Project consequence: `mlqds_temporal_fraction=0.75` is a useful high-scaffold
comparison, but it is no longer the real-usecase default. The baseline profile
uses `0.50` and `mlqds_diversity_bonus=0.0` so future runs leave more
retained-point budget under learned model control and do not add a second
spacing prior by default. A large temporal spine is currently carrying much of
the useful range-local shape/coverage behavior, so future objective work should
measure whether the learned score fill adds value beyond the temporal baseline
instead of only comparing against Douglas-Peucker.

### Temporal Fraction Interpretation

`mlqds_temporal_fraction` is not purely learned behavior. It reserves part of
the retained-point budget for an algorithmic evenly-spaced temporal spine, then
uses learned model scores to fill the remaining residual budget.

For example, at 5% retained points:

- `mlqds_temporal_fraction=0.75` means about 3.75% of the original points are
  selected by the temporal-spine rule and about 1.25% are selected by learned
  residual scores.
- `mlqds_temporal_fraction=0.80` means about 4.0% are selected by the temporal
  spine and about 1.0% are selected by learned residual scores.

This is useful as a stabilizer and as a strong baseline, but it is also a
warning sign. If quality improves mainly by increasing the temporal fraction,
the system is relying more on hardcoded temporal coverage and giving the model
less opportunity to learn query-specific retention behavior. High temporal
fractions should therefore be treated as provisional scaffolding, not proof
that the learned range model is solving the target task.

### Long-Term Default Risks

Some defaults are useful for stabilizing experiments but can also hide whether
the model is learning the desired query-driven behavior:

- high `mlqds_temporal_fraction`: preserves temporal coverage through a
  hardcoded evenly-spaced spine and leaves less budget for learned selection
- nonzero `mlqds_diversity_bonus`: adds another algorithmic spacing prior on
  top of the temporal spine, which can make learned residual quality harder to
  isolate
- per-trajectory rank scoring: makes scores easy to use for per-trajectory
  top-k simplification, but discards absolute score calibration across
  trajectories
- mandatory per-trajectory budget and endpoint retention: protects every
  trajectory, but may conflict with a purely query-usefulness objective if many
  trajectories are irrelevant to likely range workloads
- local/additive label modes: provide cheap supervision, but do not model
  retained-set interactions such as redundancy, ship-level coverage, or local
  shape preservation under a fixed budget
- high audit/checkpoint cadence: gives better selection feedback, but can make
  iteration slow enough that fewer objective variants are tested

These defaults should be separated into two categories in future benchmarks:
stabilizing scaffolds and learned-quality controls. A model-quality claim
should identify how much quality comes from the learned score fill after
subtracting the temporal/random-fill baseline.

### Residual Fill Bottleneck

The corrected `rangefix_*` queue has so far reinforced the residual-fill issue.
Across the first successful rows, MLQDS remained below uniform on
`RangeUseful`, even when it was close to or slightly above uniform on
`RangePointF1`. The best early rows were approximately:

- `mlqds_temporal_fraction=0.80`: `RangeUseful=0.4597`, still below uniform at
  about `0.4680`
- `budget_loss_temperature=0.05`: `RangeUseful=0.4598`, also below uniform and
  slower

More importantly, the learned-fill diagnostics showed MLQDS below
`TemporalRandomFill` in every first successful row. `TemporalOracleFill`
remained far higher. This implies the temporal spine is reasonable, but the
learned residual scores are not yet ranking the non-spine candidates better
than random. The next objective work should therefore focus on loss/objective
alignment for residual fill selection before spending effort optimizing the
current loss implementation.

Practical conclusion:

- keep `TemporalRandomFill` and `TemporalOracleFill` in benchmark reporting
- judge learned model progress by whether MLQDS beats `TemporalRandomFill`, not
  only whether it beats Douglas-Peucker
- avoid treating higher temporal fractions as a final solution unless learned
  residual fill quality also improves

## Training Objective Redesign Start

The old range labels were local/additive point-hit proxy labels:

- points inside range boxes receive query contribution mass
- points inside more/smaller/more informative boxes can accumulate higher
  labels
- an optional boundary prior can reweight in-box boundary points

Those labels do not model retained-set interactions. They do not know whether
nearby points already explain the same ship movement, whether a retained set
covers all ships in the query, or whether entry/exit and shape are preserved
under a fixed budget.

The first objective-redesign implementation adds `range_label_mode=usefulness`.
It is still a local/additive approximation, but it directly injects the same
concepts used by the range audit:

- point-hit retention for statistical in-box coverage
- ship-balanced mass so every hit ship can compete for budget
- sampled entry/exit points
- in-query temporal span endpoints
- local shape/turn points

The old point-hit proxy remains available as `range_label_mode=point_f1` for
ablation runs. The real-usecase profile now defaults to
`range_label_mode=usefulness`.

This is not the final retained-set objective. It should be evaluated as the
first concrete alignment step before moving to query-local set losses or
explicit marginal-gain approximations.

The second objective-redesign implementation adds `loss_objective=budget_topk`.
Instead of sampling pairwise local label order, it optimizes soft top-k retained
label mass across multiple retained-point budgets (`0.01,0.02,0.05,0.10` by
default). This better matches the final score-stream use: sort points by model
score, retain a limited budget, and measure query usefulness of that retained
set.

The real-usecase timing profile confirms this is also a runtime bottleneck:
loss construction can dominate epoch time even when forward/backward are much
smaller. That loss time currently pays for quantile selection, random ranking
pairs, and balanced pointwise sampling over local proxy labels. Optimizing this
implementation may help runtime, but it does not solve the main alignment
problem. The preferred sequence is now:

1. lock down the audit metrics with stronger tests - done
2. cache and optimize exact range usefulness evaluation - done
3. test `range_label_mode=usefulness` against the point-F1 label baseline - done
4. test `loss_objective=budget_topk` against the legacy ranking+BCE objective
5. if budget-top-k is insufficient, move to query-local set/value labels or
   explicit retained-set marginal-gain approximations
6. vectorize the final loss path after the target is clearer

### RangeUseful Alignment Guardrails

Loss/objective alignment should target the metric that represents the user
outcome, but `RangeUseful` itself must remain under scrutiny. The current
components are aligned with the desired range-query outcome directionally, but
they are still proxies:

- `RangePointF1` measures retained in-box point overlap, not interpretability.
- `ShipF1` measures whether a ship is represented, not whether its movement is
  understandable.
- `ShipCov` now measures average per-ship point coverage, reducing cases where
  `ShipF1=1.0` hides sparse representation for one queried ship.
- `EntryExitF1` measures sampled entry/exit support, not exact continuous
  boundary crossings.
- `CrossingF1` now measures retained brackets for range-box boundary crossings
  and between-sample pass-throughs.
- `TemporalCov` rewards retained in-query span, but can miss large interior
  gaps.
- `GapCov` now penalizes the largest missing run between retained in-query
  points. It is a first-pass continuity proxy, not a full time/distance
  resampling metric.
- `TurnCov` now measures weighted route-change preservation inside the query.
- `ShapeScore` now uses range-local SED/PED-style shortcut error normalized by
  original in-query segment scale. It is more route-fidelity oriented than the
  old path-length ratio, but it remains a proxy rather than a final human
  interpretability metric.

Potential future components or replacements:

- further route-fidelity refinements inside the query, such as stronger
  weighting of local SED/PED errors or distance from original in-query points
  to the simplified in-query polyline
- stronger gap coverage, especially time- or distance-normalized variants for
  irregular AIS sampling
- stronger per-ship interpretability, such as a minimum useful retained-point
  count or spacing-aware coverage inside each queried ship
- stronger turn/change-point preservation, especially heading, speed, or
  course-change semantics inside the query
- stronger entry/exit interpolation quality, including crossing time/location
  error rather than only retained segment-intersection brackets
- traffic-pattern coverage, measuring whether dense groups, crossings, or
  route alternatives remain understandable inside the query result

Therefore the redesign should follow two tracks:

1. improve the loss so learned residual scores optimize retained-set
   usefulness rather than local point labels
2. keep testing whether the usefulness score components actually match the
   target user outcome, especially on small constructed examples where the
   expected behavior is clear

Avoid baking every navigational preference into a single hidden score too
early. Report the sub-components alongside `RangeUseful` so it remains clear
whether a run improved point hits, ship presence, boundary behavior, temporal
span, gap coverage, turn preservation, or shape preservation. `RangeUseful` is
schema-versioned; old and new benchmark scores should be compared through
sub-components unless they use the same schema and weights.

### Pretraining Position

Trajectory pretraining may have merit later, but it should not be the next
priority. Reasonable pretraining ideas include:

- masked trajectory point reconstruction
- next-point, time-delta, speed, or heading prediction
- denoising corrupted AIS segments
- contrastive segment/trajectory representation learning
- query-aware pretraining that predicts range-query support or usefulness

However, current diagnostics point to a more immediate blocker: learned
residual fill is losing to `TemporalRandomFill`. Pretraining could improve
representations, but it will not fix a loss or scoring path that rewards the
wrong retained-set behavior. Revisit pretraining after the residual objective
can beat random fill under the same temporal-spine scaffold.

## Runtime Optimization Plan

Current real-usecase runs do not fully utilize the GPU because the dominant
work is not the dense transformer forward pass. Recent queue logs show a
typical epoch spending roughly:

- `forward`: about 1 second
- `loss`: about 30-40 seconds
- `backward`: about 18-22 seconds
- training diagnostics and held-out F1: about 20-25 seconds

This means simply increasing model size would not fix the bottleneck. The
runtime work should focus on removing Python/window-loop overhead, reducing
repeated exact diagnostics during sweeps, and then increasing batch size to
consume available VRAM.

### 1. Vectorize Budget-Top-K Loss - First Pass Implemented

Code targets:

- `QDS/src/training/train_model.py`
- `_budget_topk_recall_loss`
- `_budget_topk_temporal_residual_loss`
- the inner `for b in range(B)` loop in `train_model`

Original issue:

- training batches already contain multiple windows, but loss construction
  still iterates over each window
- each window then loops over budget ratios and performs small `topk`,
  `sigmoid`, masking, and pointwise-BCE operations
- this creates many small GPU launches and CPU-side orchestration overhead,
  so GPU utilization remains low even with BF16/TF32 enabled

Implemented first pass:

- budget-top-k row losses now operate on batched `[B, L]` tensors for
  predictions, labels, valid masks, and global indices
- compute per-row valid counts and per-row `k` values for each ratio
- perform batched sort/top-k threshold selection per budget ratio instead of
  calling the scalar helper for every active window
- compute ideal label mass and captured soft mass in one tensor expression
- handle temporal-residual masks per budget with batched residual candidate
  masks
- keep scalar helpers as correctness references in unit tests

Remaining scalar work:

- auxiliary pointwise BCE still samples negatives per active row
- legacy `loss_objective=ranking_bce` remains scalar because it is now an
  ablation path
- runtime impact still needs a focused before/after benchmark because the
  backward pass and pointwise term may now dominate more visibly

Correctness requirements:

- scalar-vs-batched deterministic tests are in place for normal and
  temporal-residual budget-top-k loss
- padded windows and windows with no residual positives are skipped
- temporal residual masks are handled independently for each budget ratio
- the configured `budget_loss_temperature` is preserved
- final evaluation semantics are unchanged

Expected impact:

- reduce the largest per-epoch cost
- increase GPU utilization by replacing many small operations with fewer,
  larger tensor operations
- make larger `train_batch_size` values more useful

### 2. Cheaper Checkpoint Validation - First Pass Implemented

Code targets:

- `QDS/src/training/train_model.py`
- `_validation_query_f1`
- checkpoint-selection logic around `f1_diagnostic_every`,
  `selection_history`, and best-state restoration
- `QDS/src/experiments/experiment_cli.py` and
  `QDS/src/experiments/experiment_config.py` for new knobs

Original issue:

- `f1_diagnostic_every=1` runs full validation every epoch
- exact validation is useful but costs about 17-19 seconds per epoch in the
  current profile

Implemented first pass:

- added `checkpoint_full_f1_every`, defaulting to current behavior at `1`
- added `checkpoint_candidate_pool_size`, defaulting to `1`
- when `checkpoint_full_f1_every > 1`, each eligible diagnostic epoch records a
  cheap candidate score based on loss, sampled rank diagnostic, and prediction
  spread
- the best cheap candidate snapshots are held until the next full-validation
  round, where exact validation F1/`RangeUseful` is run only for those
  candidates
- exact validation remains the only score that can promote the canonical best
  checkpoint
- keep final evaluation exact

Correctness requirements:

- candidate filtering uses the existing collapse-penalized cheap selection
  score, so low loss alone cannot promote a collapsed model
- when `checkpoint_full_f1_every=1`, behavior matches the current exact
  validation path
- training history marks checkpoint candidates, full-validation due epochs,
  evaluated candidates, and promoted candidates

Expected impact:

- reduce validation overhead without blindly selecting only every Nth epoch
- preserve robustness against noisy validation F1 spikes
- make longer epoch budgets less expensive

### 3. Cache Or Skip Repeated Range Diagnostics During Sweeps - First Pass Implemented

Code targets:

- `QDS/src/experiments/experiment_pipeline_helpers.py`
- `_range_workload_diagnostics`
- `_range_signal_diagnostics`
- the `range-diagnostics` phase in `run_experiment_pipeline`

Original issue:

- range workloads are cached, but full diagnostics are recomputed for each run
- when only model/loss/scoring parameters change, train/eval/selection queries
  are identical and the expensive range diagnostic/signals are identical too
- current logs show this phase can cost multiple minutes before training

Implemented first pass:

- added `--range_diagnostics_mode full|cached`
- the real-usecase profile uses `cached`; without `--cache_dir` this falls back
  to normal computation
- range usefulness label diagnostics now include per-component positive label
  mass and mass fractions, making it easier to see whether the local/additive
  proxy is overemphasizing point hits, ship presence, crossing brackets,
  temporal span, gap coverage, turn points, or shape points; component mass is
  reported before the final training-label clamp so saturation remains visible
- persistent cache entries are stored under `<cache_dir>/range_diagnostics/`
- cache keys include points/boundaries fingerprints, typed query digest,
  workload map, compression ratio, range label mode, boundary prior, diagnostic
  filter settings, seed, and schema version
- cached entries reuse workload summary, per-query diagnostic rows,
  range labels/labelled masks, label diagnostics, baseline/oracle signal
  summaries, and rebuild the runtime `EvaluationQueryCache` from the cached
  workload
- `--refresh_cache` forces recomputation

Correctness requirements:

- cache key includes values that change labels or diagnostic scores
- final matched evaluation remains exact with respect to method masks
- stale, missing, or unreadable diagnostics cache entries fail closed by
  recomputing

Expected impact:

- large speedup for queues that test only loss/scoring/model parameters
- less CPU/RAM churn before training
- cleaner separation between workload-generation experiments and model-training
  experiments

### 4. Use Available VRAM With Larger Training Batches

Code targets:

- `QDS/src/experiments/benchmark_matrix.py`
- `MatrixVariant`
- `QDS/src/experiments/benchmark_runtime.py`
- `QDS/src/training/trajectory_batching.py`
- `QDS/src/training/train_model.py`

Current issue:

- current real-usecase runs use about 2.6GB of a 16GB GPU
- `train_batch_size=32` leaves substantial VRAM unused
- larger batches can reduce optimizer/Python overhead and provide larger GPU
  work units, but they do not fully solve the loss bottleneck by themselves

Prepared implementation direction:

- after vectorizing the loss, benchmark `train_batch_size=64` and `128`
- matrix variants are available:
  - `tf32_bf16_bs64_inf32`
  - `tf32_bf16_bs128_inf32`
- compare runtime, peak VRAM, best epoch, `RangeUseful`, and collapse warnings
- keep inference batch size separate; inference can likely also increase if
  validation remains expensive

Correctness requirements:

- batch-size changes alter optimizer step count per epoch, so quality must be
  compared, not only speed
- monitor VRAM and host RAM; WSL/host memory pressure may still dominate

Expected impact:

- moderate speedup before loss vectorization
- larger speedup after loss vectorization because the GPU receives larger,
  fewer kernels

### 5. Optimization Benchmark Order

After the current corrected queue finishes:

1. analyze fixed queue quality and timing
2. run one runtime+quality comparison against the previous scalar-loss baseline
3. benchmark `train_batch_size=64` and `128`

Do not change all optimization knobs at once. The first comparison should
answer whether the batched loss path is faster without changing quality. The
second should answer how much repeated diagnostics cost can be removed. The
third should tune checkpoint cadence and batch size.

## Current Benchmark Evidence

The `range_objective_audit_20260512-032736` real-usecase audit showed MLQDS did
not beat simple baselines at 1%, 2%, 5%, or 10% retained points. At 5%
retention, MLQDS was close on `RangePointF1` but much weaker on
`RangeUseful`, mainly because `TemporalCov` and `ShapeScore` trailed uniform
and Douglas-Peucker.

The `range_usefulness_checkpoint_ab_20260512-042307` A/B compared default
`checkpoint_f1_variant=answer` against `checkpoint_f1_variant=range_usefulness`.
The aligned checkpoint-selection score did not materially improve final audit
quality:

- at 5% retention, `RangeUseful` was about `0.37319` with answer selection and
  `0.37306` with range-usefulness selection
- at 10% retention, range-usefulness selection was only slightly higher
  (`0.47790` vs `0.47707`)
- both variants remained well below uniform and Douglas-Peucker on
  `RangeUseful`

Despite the small single-run delta, `range_usefulness` is now the default
checkpoint target for range-specialist training because it is the aligned
selection metric. The older `answer` target is retained only for explicit
legacy ablations.

Conclusion: checkpoint selection is not the main blocker. The more likely
blocker is the training objective/label construction itself.

### Reporting Cleanup Notes

The first reporting cleanup pass is implemented:

- benchmark rows report `collapse_warning_any`, `collapse_warning_count`,
  `best_epoch_collapse_warning`, `min_pred_std`, and `best_epoch_pred_std` so a
  transient collapsed epoch is not confused with a collapsed selected
  checkpoint
- validation history/logs report `val_selection_score` plus explicit
  `val_range_point_f1` and `val_range_usefulness`, so range-usefulness
  checkpointing is not mislabeled as generic query F1
- workload generation metadata separates minimum requested queries, max query
  cap, final generated query count, target coverage, final coverage, and stop
  reason
- benchmark matrix rows include temporal-random-fill usefulness,
  MLQDS-vs-random-fill usefulness, and temporal-oracle-fill usefulness gap

## Range Workload Generation Direction

Range workload generation defines the user-query distribution the model learns.
It should be realistic enough to train useful behavior without overfitting to a
single narrow usage pattern.

Future range query generation should consider a mixture of realistic query
patterns such as:

- dense traffic regions and shipping lanes
- port approaches, chokepoints, anchorage areas, and crossings when available
- variable spatial windows around meaningful maritime regions
- variable temporal windows that represent plausible analyst questions
- some background/uniform queries to avoid overfitting only dense hotspots

Concrete generator improvements to consider:

- anchor more queries on actual traffic structures instead of only sampled
  points, for example high-density tiles, route corridors, ports, and
  chokepoints
- stratify queries by hit shape: single-ship, few-ship, many-ship, dense-area,
  sparse-area, entry/exit-heavy, and long-transit cases
- explicitly control temporal-window families, such as short event windows,
  watch windows of several hours, and full-shift/daypart windows
- enforce query diversity using spatial/temporal overlap limits so coverage is
  not achieved by many near-duplicate boxes
- keep a reserved background/uniform slice to measure generalization outside
  dense hotspots
- report the final generated query count, coverage, hit-count distribution,
  trajectory-hit distribution, and stop reason for every workload split

The generator should continue to report coverage and query diagnostics, but
coverage alone should not define workload quality.

## Benchmarking Cadence

Prefer smaller iterative benchmark runs over large broad queues. The default
research loop should be:

1. formulate one concrete hypothesis
2. run a focused A/B comparison or small sweep
3. inspect quality, runtime, and diagnostics
4. update the next hypothesis

Large queues are useful only when the factor under test is already clear or
when the computer would otherwise be idle for a long period. Avoid mixing too
many unrelated factors in one queue because it increases waiting time and makes
causal interpretation weaker.

## Next Step

Run a focused benchmark comparison between:

- `loss_objective=budget_topk`
- `loss_objective=ranking_bce`

Keep the rest of the real-usecase profile fixed. Evaluate at `0.01,0.02,0.05,0.10`
retained ratios and compare `RangePointF1`, `RangeUseful`, `ShipF1`,
`ShipCov`, `EntryExitF1`, `CrossingF1`, `TemporalCov`, `GapCov`, `TurnCov`,
and `ShapeScore`.

If budget-top-k improves audit quality, repeat across seeds. If it does not,
move to query-local set/value labels or explicit retained-set marginal-gain
approximations.
