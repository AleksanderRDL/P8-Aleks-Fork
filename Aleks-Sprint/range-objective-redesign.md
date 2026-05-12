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
