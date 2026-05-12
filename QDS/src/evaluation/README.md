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
- `UniformTemporalMethod` (`uniform` in result tables) keeps truly evenly spaced points in each trajectory and is the default temporal baseline.
- `DouglasPeuckerMethod` is a true recursive Douglas-Peucker baseline that keeps endpoints and repeatedly splits the current highest-error segment until the compression budget is filled.
- `OracleMethod` is an additive-label greedy diagnostic. It uses oracle labels for the explicit workload, but it is not an exact combinatorial optimizer for final retained-set F1.

## Metrics

- Range queries are scored over retained point hits inside the spatiotemporal box, so sparse point retention is measured by how much of the original query-hit mass it preserves rather than by one retained point recovering an entire trajectory. Exact duplicate AIS rows are counted as separate point instances.
- This range score is a useful retained-point proxy, not the full target for
  range-query navigational usefulness. It does not directly score per-ship
  interpretability, entry/exit preservation, or range-local trajectory shape.
  See `../../../Aleks-Sprint/range-objective-redesign.md` for the current
  objective-redesign conclusion.
- `BoundaryF1` is reported separately for range workloads. It measures retained in-box boundary-crossing points and is a shape-preservation diagnostic, not part of pure range F1.
- kNN, similarity, and clustering queries report pure answer-set agreement as `AnswerF1`; `CombinedF1` additionally multiplies answer agreement by retained support-point quality for diagnostic comparison.
- `f1_score(original, simplified)` - harmonic-mean agreement between original and simplified answer sets.
- `clustering_f1(original_labels, simplified_labels)` - F1 over same-cluster trajectory co-membership pairs, ignoring noise label `-1`.
- Retained point gap reports the average original-index spacing between consecutive retained points per trajectory. Lower values mean retained points are more evenly dense along the original trajectory, and the JSON output also includes normalized and max gap values.
- `MethodEvaluation` stores aggregate F1, per-type F1, compression ratio, retained point gap, latency in milliseconds, geometry distortion, and length preservation. The legacy `avg_length_loss` property remains available as `1 - avg_length_preserved`.

## Reporting

- `evaluate_method` scores one method against one typed workload. Reuse an
  `EvaluationQueryCache` when several methods share the same points,
  boundaries, and query list.
- `print_method_comparison_table` reports F1, `AvgPtGap`, range `BoundaryF1`,
  and MLQDS gaps versus `uniform`/Douglas-Peucker when available.
- `print_geometric_distortion_table` reports SED/PED, length preservation, and
  `F1xLen`.
- Read F1 and length-preservation columns as higher-is-better. Read `AvgPtGap`
  as lower-is-better.
