# Evaluation Module

This module compares query-aware ML simplification against temporal, geometric, and label-Oracle baselines, then reports per-type and aggregate query F1 where higher is better.

## Files

| File | Purpose |
| --- | --- |
| `baselines.py` | Simplification methods: `MLQDSMethod`, `NewUniformTemporalMethod`, `DouglasPeuckerMethod`, and `OracleMethod`. |
| `metrics.py` | F1 functions and the `MethodEvaluation` container. |
| `evaluate_methods.py` | Runs a method on flattened points and boundaries, caches reusable query results, then formats the comparison tables. |

## Methods

- `MLQDSMethod` uses the trained model, persisted scaler, and eval workload to produce per-point scores. Query-type heads are rank-normalized within each trajectory before weighting; current experiment entrypoints use one pure workload per model. Model inference uses CUDA by default when available, while retained masks stay on the original point tensor device for evaluation.
- `NewUniformTemporalMethod` (`uniform` in result tables) keeps truly evenly spaced points in each trajectory and is the default temporal baseline.
- `DouglasPeuckerMethod` is a true recursive Douglas-Peucker baseline that keeps endpoints and repeatedly splits the current highest-error segment until the compression budget is filled.
- `OracleMethod` is a diagnostic upper bound that uses oracle labels directly.

## Metrics

- Range queries are scored over retained point hits inside the spatiotemporal box, so sparse point retention is measured by how much of the original query-hit mass it preserves rather than by one retained point recovering an entire trajectory. Exact duplicate AIS rows are counted as separate point instances.
- kNN, similarity, and clustering queries report pure answer-set agreement as `AnswerF1`; `CombinedF1` additionally multiplies answer agreement by retained support-point quality for diagnostic comparison.
- `f1_score(original, simplified)` - harmonic-mean agreement between original and simplified answer sets.
- `clustering_f1(original_labels, simplified_labels)` - F1 over same-cluster trajectory co-membership pairs, ignoring noise label `-1`.
- Retained point gap reports the average original-index spacing between consecutive retained points per trajectory. Lower values mean retained points are more evenly dense along the original trajectory, and the JSON output also includes normalized and max gap values.
- `MethodEvaluation` stores aggregate F1, per-type F1, compression ratio, retained point gap, latency in milliseconds, geometry distortion, and length preservation. The legacy `avg_length_loss` property remains available as `1 - avg_length_preserved`.

## Reporting

`evaluate_method` evaluates one simplification method against a typed query workload. Pass an `EvaluationQueryCache` when several methods are evaluated on the same points, boundaries, and query list; it reuses full-data query answers and support masks while still recomputing simplified-query answers for each retained mask. `print_method_comparison_table` renders F1 values to six decimals so close methods are not hidden by rounding, includes `AvgPtGap` for retained-point spacing, and appends MLQDS gaps versus `uniform` and Douglas-Peucker when those baselines are present. `print_geometric_distortion_table` reports SED/PED plus `LengthPres` and `F1xLen`, where both length columns are higher-is-better. `print_shift_table` renders the train-vs-eval workload shift table written by the experiment pipeline. Tables should be read as higher-is-better F1 scores, while lower `AvgPtGap` means smaller average spacing between retained points.
