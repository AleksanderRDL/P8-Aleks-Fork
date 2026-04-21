# Evaluation Module

This module compares query-aware ML simplification against query-blind and geometric baselines, then reports per-type and aggregate query error.

## Files

| File | Purpose |
| --- | --- |
| `baselines.py` | Simplification methods: `MLQDSMethod`, `QueryBlindMLMethod`, `RandomMethod`, `UniformTemporalMethod`, `DouglasPeuckerMethod`, and `OracleMethod`. |
| `metrics.py` | Error functions and the `MethodEvaluation` container. |
| `evaluate_methods.py` | Runs a method on flattened points and boundaries, then formats the comparison tables. |

## Methods

- `MLQDSMethod` uses the trained model, the persisted scaler, and the eval workload to produce per-point scores.
- `QueryBlindMLMethod` is the ablation trained with random query input.
- `RandomMethod` retains random points per trajectory.
- `UniformTemporalMethod` keeps approximately evenly spaced points in each trajectory.
- `DouglasPeuckerMethod` approximates geometric importance from perpendicular distance to the trajectory endpoints.
- `OracleMethod` is a diagnostic upper bound that uses oracle labels directly.

## Metrics

- `range_error(full, simplified)` - normalized absolute error on the aggregate range answer.
- `knn_error(full, simplified)` - 1 minus Jaccard overlap.
- `similarity_error(full, simplified)` - 1 minus rank-biased overlap.
- `clustering_error(full, simplified)` - normalized cluster-count error.
- `MethodEvaluation` stores aggregate error, per-type error, compression ratio, and latency in milliseconds.

## Reporting

`evaluate_method` evaluates one simplification method against a typed query workload. `print_method_comparison_table` renders the matched-workload summary table, and `print_shift_table` renders the train-vs-eval workload shift table written by the experiment pipeline.
