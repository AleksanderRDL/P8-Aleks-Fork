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

## Current Metric Limitation

The current range `AnswerF1` implementation is effectively retained in-box
point overlap: for each range query, it scores how many original in-box point
instances survived in the retained set. Because simplified points are a subset
of original points, this behaves like a recall-derived point-hit metric.

That metric is useful, but incomplete. It does not directly measure whether the
simplified query result preserves per-ship interpretability, entry/exit
behavior, temporal coverage, or range-local trajectory shape.

Consequence: a model can improve the current training loss or point-hit proxy
without improving the actual user-facing usefulness of the simplified query
result.

## Current Training Objective Limitation

Current range labels are local/additive proxy labels:

- points inside range boxes receive query contribution mass
- points inside more/smaller/more informative boxes can accumulate higher
  labels
- an optional boundary prior can reweight in-box boundary points

These labels do not model retained-set interactions. They do not know whether
nearby points already explain the same ship movement, whether a retained set
covers all ships in the query, or whether entry/exit and shape are preserved
under a fixed budget.

The current training loss can therefore optimize the proxy while final
retained-set range usefulness stagnates or degrades.

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

Before redesigning labels or loss, run a range objective alignment audit. It
should compare current MLQDS, uniform, Douglas-Peucker, and oracle diagnostics
across 1%, 2%, 5%, and 10% retained-point budgets.

The audit should report at least:

- current retained in-box point F1
- ship/trajectory recall inside range queries
- entry/exit preservation
- range-local trajectory shape/geometric distortion
- range-local temporal coverage
- combined range usefulness score candidates
- correlation between training loss, proxy labels, validation F1, and final
  retained-set metrics by epoch

Only after that audit should the project redesign the objective. Candidate
directions include query-local set/value labels, budget-aware labels, retained
set marginal-gain approximations, shape-aware range metrics, and losses that
better match the final simplification policy.
