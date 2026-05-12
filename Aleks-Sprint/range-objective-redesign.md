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
- `EntryExitF1`: whether range entry/exit points are retained
- `TemporalCov`: how much of each in-query ship time span remains covered
- `ShapeScore`: range-local retained path-length preservation
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
- `EntryExitF1` measures whether sampled in-box entry/exit points are retained.
  It does not interpolate true geometric boundary crossings between AIS
  samples.
- `TemporalCov` measures retained in-query time span divided by original
  in-query time span per ship. It is a useful coverage proxy, but it does not
  yet penalize large interior gaps if the endpoints are retained.
- `ShapeScore` measures retained in-query path-length preservation. It is a
  cheap local shape proxy, not full local SED/PED fidelity.
- `RangeUseful` is a weighted audit score over these components. Treat it as a
  comparison and checkpoint-selection diagnostic while the objective is being
  redesigned, not as the mathematically final utility target.

Before making these metrics central to the loss, add focused unit/property
tests for single-ship, multi-ship, no-hit, single-point-hit, duplicate-point,
boundary-entry/exit, interior-gap, and curved-path cases.

### Audit Runtime Direction

The exact final audit should stay exact. To reduce overhead without changing
semantics:

- precompute per-query full ship IDs, retained-independent entry/exit masks,
  in-query offsets, full time spans, and full local path lengths
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

Project consequence: the real-usecase profile now uses
`mlqds_temporal_fraction=0.75` as the provisional default, pending seed-repeat
confirmation. This is an objective-design signal too: a large temporal spine is
currently carrying most of the useful range-local shape/coverage behavior, so
future objective work should measure whether the learned score fill adds value
beyond the temporal baseline instead of only comparing against Douglas-Peucker.

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
- `EntryExitF1` measures sampled entry/exit support, not exact continuous
  boundary crossings.
- `TemporalCov` rewards retained in-query span, but can miss large interior
  gaps.
- `ShapeScore` currently rewards retained path-length coverage; it is not a
  full geometric route-fidelity metric.

Therefore the redesign should follow two tracks:

1. improve the loss so learned residual scores optimize retained-set
   usefulness rather than local point labels
2. keep testing whether the usefulness score components actually match the
   target user outcome, especially on small constructed examples where the
   expected behavior is clear

Avoid baking every navigational preference into a single hidden score too
early. Report the sub-components alongside `RangeUseful` so it remains clear
whether a run improved point hits, ship presence, boundary behavior, temporal
coverage, or shape preservation.

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

### 1. Vectorize Budget-Top-K Loss

Code targets:

- `QDS/src/training/train_model.py`
- `_budget_topk_recall_loss`
- `_budget_topk_temporal_residual_loss`
- the inner `for b in range(B)` loop in `train_model`

Current issue:

- training batches already contain multiple windows, but loss construction
  still iterates over each window
- each window then loops over budget ratios and performs small `topk`,
  `sigmoid`, masking, and pointwise-BCE operations
- this creates many small GPU launches and CPU-side orchestration overhead,
  so GPU utilization remains low even with BF16/TF32 enabled

Prepared implementation direction:

- build batched tensors shaped approximately `[B, L]` for predictions, labels,
  valid masks, and global indices
- create one residual candidate mask per budget ratio with shape `[R, B, L]`
  where `R = len(budget_loss_ratios)`
- compute per-row valid counts and per-row `k` values for each ratio
- perform batched top-k/threshold selection across flattened `(R * B)` rows
  instead of calling the scalar helper for every window
- compute ideal label mass and captured soft mass in one tensor expression
- return both loss and useful diagnostics such as active row count and skipped
  no-positive rows
- keep the existing scalar helpers temporarily for unit tests and correctness
  comparison, then remove them once the vectorized path is trusted

Correctness requirements:

- must match current scalar behavior on small deterministic examples
- must handle padded windows and windows with no residual positives
- must handle temporal residual masks for each budget independently
- must preserve the configured `budget_loss_temperature`
- must leave final evaluation semantics unchanged

Expected impact:

- reduce the largest per-epoch cost
- increase GPU utilization by replacing many small operations with fewer,
  larger tensor operations
- make larger `train_batch_size` values more useful

### 2. Cheaper Checkpoint Validation

Code targets:

- `QDS/src/training/train_model.py`
- `_validation_query_f1`
- checkpoint-selection logic around `f1_diagnostic_every`,
  `selection_history`, and best-state restoration
- `QDS/src/experiments/experiment_cli.py` and
  `QDS/src/experiments/experiment_config.py` for new knobs

Current issue:

- `f1_diagnostic_every=1` runs full validation every epoch
- exact validation is useful but costs about 17-19 seconds per epoch in the
  current profile

Prepared implementation direction:

- add a candidate-checkpoint tournament:
  - every epoch computes cheap signals such as loss, prediction std, and sampled
    diagnostic statistics
  - keep the top `K` candidate model snapshots since the previous full
    validation
  - every `N` epochs, run exact validation F1/`RangeUseful` only on those
    candidate snapshots
  - promote the best validated snapshot to canonical best checkpoint
- add config knobs:
  - `checkpoint_full_f1_every`, defaulting to current behavior at `1`
  - `checkpoint_candidate_pool_size`, likely `2` or `3`
  - optionally `checkpoint_candidate_metric`, initially based on loss plus
    non-collapse prediction std
- keep final evaluation exact

Correctness requirements:

- exact validation remains the only score that can promote the canonical best
  checkpoint
- candidate filtering must never select a collapsed model solely due to low
  loss
- when `checkpoint_full_f1_every=1`, behavior should match the current path
  except for harmless refactoring

Expected impact:

- reduce validation overhead without blindly selecting only every Nth epoch
- preserve robustness against noisy validation F1 spikes
- make longer epoch budgets less expensive

### 3. Cache Or Skip Repeated Range Diagnostics During Sweeps

Code targets:

- `QDS/src/experiments/experiment_pipeline_helpers.py`
- `_range_workload_diagnostics`
- `_range_signal_diagnostics`
- the `range-diagnostics` phase in `run_experiment_pipeline`

Current issue:

- range workloads are cached, but full diagnostics are recomputed for each run
- when only model/loss/scoring parameters change, train/eval/selection queries
  are identical and the expensive range diagnostic/signals are identical too
- current logs show this phase can cost multiple minutes before training

Prepared implementation direction:

- add persistent diagnostics cache keyed by:
  - points/boundaries fingerprint
  - typed query workload cache key or query digest
  - compression ratio
  - range label mode
  - range boundary prior weight
  - diagnostics schema version
- reuse cached:
  - workload summary
  - per-query diagnostics rows
  - range labels/labelled masks
  - evaluation query cache inputs where safely reconstructable
  - baseline/oracle signal summaries
- add a sweep mode such as `--reuse_range_diagnostics` or
  `--range_diagnostics_mode cached|full|minimal`
- for parameter sweeps, use cached/minimal diagnostics by default when workload
  generation settings are unchanged
- force full diagnostics for audit runs and after query-generation changes

Correctness requirements:

- cache key must include every value that can change labels or diagnostic
  scores
- final matched evaluation remains exact and uncached with respect to method
  masks
- stale diagnostics must fail closed by recomputing

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
- add matrix variants:
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
2. implement vectorized budget-top-k loss
3. run one A/B runtime+quality comparison against the current scalar loss
4. implement cached/minimal range diagnostics for repeated model sweeps
5. implement candidate-checkpoint tournament
6. benchmark `train_batch_size=64` and `128`

Do not change all optimization knobs at once. The first comparison should
answer whether the vectorized loss is numerically equivalent enough and faster.
The second should answer how much repeated diagnostics cost can be removed. The
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

The benchmark table and logs need two cleanup passes after the active queue is
finished:

- `collapse_warning` currently means any epoch had a low prediction standard
  deviation, not necessarily that the selected checkpoint collapsed. Replace or
  extend it with fields such as `collapse_warning_any`,
  `collapse_warning_count`, `best_epoch_collapse_warning`, and `min_pred_std`.
- `val_query_f1` in logs is semantically confusing when checkpoint selection
  uses `checkpoint_f1_variant=range_usefulness`. Rename the selected value to
  `val_selection_score` and report `val_range_point_f1` /
  `val_range_usefulness` separately.
- workload generation metadata currently reports `requested_queries` even when
  target-coverage generation produces more queries. Separate minimum requested
  queries, max query cap, final generated query count, target coverage, final
  coverage, and stop reason.

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

The generator should continue to report coverage and query diagnostics, but
coverage alone should not define workload quality.

## Next Step

Run a focused benchmark comparison between:

- `loss_objective=budget_topk`
- `loss_objective=ranking_bce`

Keep the rest of the real-usecase profile fixed. Evaluate at `0.01,0.02,0.05,0.10`
retained ratios and compare `RangePointF1`, `RangeUseful`, `ShipF1`,
`EntryExitF1`, `TemporalCov`, and `ShapeScore`.

If budget-top-k improves audit quality, repeat across seeds. If it does not,
move to query-local set/value labels or explicit retained-set marginal-gain
approximations.
