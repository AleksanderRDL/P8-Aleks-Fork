# Evaluation Module

This module compares query-aware ML simplification against temporal, geometric, and label-Oracle baselines, then reports per-type and aggregate query F1 where higher is better.

## Files

| File | Purpose |
| --- | --- |
| `baselines.py` | Simplification methods: `MLQDSMethod`, `UniformTemporalMethod`, `DouglasPeuckerMethod`, and `OracleMethod`. |
| `metrics.py` | F1 functions and the `MethodEvaluation` container. |
| `evaluate_methods.py` | Runs a method on flattened points and boundaries, caches reusable query results, then formats the comparison tables. |

## Methods

- `MLQDSMethod` uses the trained model, persisted scaler, and eval workload to produce per-point scores for one explicit workload type. The default score mode rank-normalizes that score stream within each trajectory before simplification. Benchmarkable alternatives are `rank_tie`, `raw`, `sigmoid`, `temperature_sigmoid`, `zscore_sigmoid`, and `rank_confidence`. Model inference uses CUDA by default when available, while retained masks stay on the original point tensor device for evaluation.
- `ScoreHybridMethod` is a diagnostic helper: it keeps the same temporal base as
  MLQDS and fills the remaining budget with caller-supplied scores. Experiments
  use it for random-fill and oracle-fill comparisons without adding those rows
  to the main matched-method table.
- `UniformTemporalMethod` (`uniform` in result tables) keeps truly evenly spaced points in each trajectory and is the default temporal baseline.
- `DouglasPeuckerMethod` is a true recursive Douglas-Peucker baseline that keeps endpoints and repeatedly splits the current highest-error segment until the compression budget is filled.
- `OracleMethod` is an additive-label greedy diagnostic. It uses oracle labels for the explicit workload, but it is not an exact combinatorial optimizer for final retained-set F1.

## Metrics

- Range queries report `RangePointF1`, the retained point-hit metric inside the
  spatiotemporal box. Sparse point retention is measured by how much of the
  original query-hit mass it preserves rather than by one retained point
  recovering an entire trajectory. Exact duplicate AIS rows are counted as
  separate point instances.
- This range score is a useful retained-point proxy, not the full target for
  range-query navigational usefulness. It does not directly score per-ship
  interpretability, entry/exit preservation, or range-local trajectory shape.
  See `../../../Aleks-Sprint/range-objective-redesign.md` for the current
  objective-redesign conclusion.
- `RangeUseful` is a versioned audit score combining `RangePointF1`, ship
  presence, per-ship point coverage, entry/exit preservation, temporal span
  coverage, in-query gap coverage, and range-local path-shape preservation. It
  is reported separately so it can guide objective redesign without pretending
  to be a mathematically final target. Schema v3 weights are
  `0.25/0.15/0.15/0.15/0.12/0.10/0.08` for point, ship presence, ship
  coverage, entry/exit, temporal span, gap, and shape respectively.
- `EntryExitF1` is reported separately for range workloads. It measures
  retained in-box boundary-crossing points and is a shape-preservation
  diagnostic, not part of `RangePointF1`.
- Audit component interpretation:
  - `ShipF1` is ship presence only; one retained in-query point can recover a
    ship.
  - `ShipCov` averages point-subset F1 per hit ship, so dense ships do not hide
    sparse representation of another queried ship.
  - `EntryExitF1` uses sampled AIS entry/exit points, not interpolated true box
    crossings.
  - `TemporalCov` scores retained in-query time span. It intentionally does
    not penalize large interior gaps when endpoints survive.
  - `GapCov` scores the largest missing run between retained in-query points,
    so endpoints-only simplifications no longer look complete on straight
    tracks.
  - `ShapeScore` scores range-local route fidelity from SED/PED-style shortcut
    error normalized by the original in-query segment scale. It is still a
    proxy, but it now penalizes geometric shortcuts more directly than retained
    path-length ratio.
- kNN, similarity, and clustering queries report pure answer-set agreement as `AnswerF1`; `CombinedF1` additionally multiplies answer agreement by retained support-point quality for diagnostic comparison.
- `f1_score(original, simplified)` - harmonic-mean agreement between original and simplified answer sets.
- `clustering_f1(original_labels, simplified_labels)` - F1 over same-cluster trajectory co-membership pairs, ignoring noise label `-1`.
- Retained point gap reports the average original-index spacing between consecutive retained points per trajectory. Lower values mean retained points are more evenly dense along the original trajectory, and the JSON output also includes normalized and max gap values.
- `MethodEvaluation` stores aggregate F1, per-type F1, compression ratio, retained point gap, latency in milliseconds, geometry distortion, and length preservation. The legacy `avg_length_loss` property remains available as `1 - avg_length_preserved`.

## Reporting

- `evaluate_method` scores one method against one typed workload. Reuse an
  `EvaluationQueryCache` when several methods share the same points,
  boundaries, and query list.
- `print_method_comparison_table` reports `RangePointF1`/`RangeUseful` for
  range workloads and `AnswerF1`/`CombinedF1` for non-range answer-set
  workloads.
- `print_range_usefulness_table` reports the detailed range audit components.
- `print_geometric_distortion_table` reports SED/PED, length preservation, and
  `F1xLen`.
- Read F1 and length-preservation columns as higher-is-better. Read `AvgPtGap`
  as lower-is-better.

## Audit Caching

Keep final benchmark audits exact. `EvaluationQueryCache` precomputes
retained-independent per-query range support: full ship IDs, compact entry/exit
indices, in-query offsets, full local time spans, and full local path lengths.
Reuse that cache across MLQDS, baselines, Oracle, and compression ratios. Only
use sampled approximations for checkpoint-selection diagnostics when needed;
final reported audit metrics should stay exact.
