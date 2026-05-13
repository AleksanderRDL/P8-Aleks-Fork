# Queries Module

Defines range query formats, workload generation, and query execution.

Range is the active product/query surface. Historical kNN, similarity, and
clustering code may still exist in the package until it is removed, but it is
legacy and should not shape new training, evaluation, or benchmark design.

## Files

| File | Purpose |
| --- | --- |
| `workload.py` | `TypedQueryWorkload` container. |
| `query_types.py` | Query IDs, pure workload validation, feature padding. |
| `query_generator.py` | Deterministic workload generation. |
| `query_executor.py` | Range query execution. |
| `range_geometry.py` | Shared range-box and geographic distance helpers. |
| `workload_diagnostics.py` | Range workload quality and label diagnostics. |

## Query Types

| Type | ID | Meaning |
| --- | --- | --- |
| `range` | 0 | Spatiotemporal box; range evaluation also scores retained point support. |

Workloads are range-only for active experiments, e.g. `{"range": 1.0}`.
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

Range anchors use density-biased sampling mixed with uniform sampling.

Use `scripts/estimate_range_coverage.py` before changing query count,
footprint, or coverage targets.

## Execution

- `execute_range_query`: trajectory IDs with points inside the box.
- `execute_typed_query`: dispatch by the query `type` field.

The generator defines the future-query prior for workload-blind training. Do
not tune final claims only to one narrow generator setting.
