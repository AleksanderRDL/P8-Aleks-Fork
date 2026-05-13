# Queries Module

Defines typed query formats, workload generation, and query execution.

## Files

| File | Purpose |
| --- | --- |
| `workload.py` | `TypedQueryWorkload` container. |
| `query_types.py` | Query IDs, pure workload validation, feature padding. |
| `query_generator.py` | Deterministic workload generation. |
| `query_executor.py` | Range, kNN, similarity, and clustering execution. |

## Query Types

| Type | ID | Meaning |
| --- | --- | --- |
| `range` | 0 | Spatiotemporal box; range evaluation also scores retained point support. |
| `knn` | 1 | Nearest distinct trajectories around an anchor and time window. |
| `similarity` | 2 | Reference-snippet similarity inside a centroid/radius/time filter. |
| `clustering` | 3 | DBSCAN labels over trajectory representatives inside a box. |

Workloads are pure: one active query type per model, e.g. `{"range": 1.0}`.
`pad_query_features` converts typed query dicts into `[M, 12]` features plus
`[M]` type IDs.

## Generation

`generate_typed_query_workload` returns the shared workload container used by
training and evaluation.

Range generation controls:

- `range_spatial_fraction`, `range_time_fraction`: dataset-relative footprint.
- `range_spatial_km`, `range_time_hours`: absolute half-window footprint.
- `range_footprint_jitter`: random footprint scaling.
- `target_coverage`: point-level query-signal coverage target.
- `max_queries`: optional cap when generation continues past `n_queries`.

Range and kNN anchors use density-biased sampling mixed with uniform sampling.
Similarity and clustering currently use uniform anchors.

Use `scripts/estimate_range_coverage.py` before changing query count,
footprint, or coverage targets.

## Execution

- `execute_range_query`: trajectory IDs with points inside the box.
- `execute_knn_query`: nearest distinct trajectory IDs in the time window.
- `execute_similarity_query`: ranked trajectory IDs; evaluation consumes set F1.
- `execute_clustering_query`: per-trajectory cluster labels.
- `execute_typed_query`: dispatch by the query `type` field.

The generator defines the future-query prior for workload-blind training. Do
not tune final claims only to one narrow generator setting.
