# Evaluation Module

This module compares query-aware ML simplification against stochastic and geometric baselines, then reports per-type and aggregate query F1 where higher is better.

## Files

| File | Purpose |
| --- | --- |
| `baselines.py` | Simplification methods: `MLQDSMethod`, `RandomMethod`, `UniformTemporalMethod`, `DouglasPeuckerMethod`, and `OracleMethod`. |
| `metrics.py` | F1 functions and the `MethodEvaluation` container. |
| `evaluate_methods.py` | Runs a method on flattened points and boundaries, then formats the comparison tables. |

## Methods

- `MLQDSMethod` uses the trained model, the persisted scaler, and the eval workload to produce per-point scores.
- `RandomMethod` retains random points per trajectory.
- `UniformTemporalMethod` keeps approximately evenly spaced points in each trajectory.
- `DouglasPeuckerMethod` approximates geometric importance from perpendicular distance to the trajectory endpoints.
- `OracleMethod` is a diagnostic upper bound that uses oracle labels directly.

## Metrics

- `f1_score(original, simplified)` - harmonic-mean agreement between original and simplified answer sets.
- `clustering_f1(original_labels, simplified_labels)` - F1 over same-cluster trajectory co-membership pairs, ignoring noise label `-1`.
- `MethodEvaluation` stores aggregate F1, per-type F1, compression ratio, and latency in milliseconds.

## Reporting

`evaluate_method` evaluates one simplification method against a typed query workload. `print_method_comparison_table` renders the matched-workload summary table, and `print_shift_table` renders the train-vs-eval workload shift table written by the experiment pipeline. Tables should be read as higher-is-better F1 scores.
