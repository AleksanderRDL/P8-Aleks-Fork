# Queries Module

This module defines the typed query format, workload generator, and concrete query executors that produce labels and evaluation targets.

## Files

| File | Purpose |
| --- | --- |
| `workload.py` | `TypedQueryWorkload` container shared by generation, training, and evaluation. |
| `query_types.py` | Query type IDs, pure workload validation, and padded query feature conversion. |
| `query_generator.py` | Deterministic typed workload generation from a trajectory set. |
| `query_executor.py` | Concrete range, kNN, similarity, and clustering query execution. |

## Query Types

| Name | ID | Input params | Return value |
| --- | --- | --- | --- |
| `range` | 0 | spatial-temporal box | matching trajectory IDs plus point support for range F1 |
| `knn` | 1 | anchor point, time window, `k` | set of nearest distinct trajectory IDs |
| `similarity` | 2 | centroid, time box, radius, reference snippet | ranked trajectory IDs, consumed as a set for F1 |
| `clustering` | 3 | box, `eps`, `min_samples` | per-trajectory DBSCAN labels over trajectory centroids |

## Workload Format

- `normalize_pure_workload_map` lowercases the keys, drops non-positive weights,
  validates query names, and requires exactly one active query type.
- `pad_query_features` converts heterogeneous typed query dicts into a `[M, 12]` tensor plus a `[M]` tensor of type IDs.
- Feature layout:
    - range: `lat_min`, `lat_max`, `lon_min`, `lon_max`, `t_start`, `t_end`
    - knn: `lat`, `lon`, `t_center`, `t_half_window`, `k`
    - similarity: `lat_query_centroid`, `lon_query_centroid`, `t_start`, `t_end`, `radius`, plus the mean of the reference snippet in slots 5-7 when present
    - clustering: `lat_min`, `lat_max`, `lon_min`, `lon_max`, `t_start`, `t_end`, `eps`, `min_samples`

## Workload Generation

- `generate_typed_query_workload` returns the `TypedQueryWorkload` container
  used by training and evaluation. Workloads are pure, e.g. `{"range": 1.0}`,
  so one model targets one query type.
- Range query generation defines the user-query distribution the model learns.
  Future range workloads should represent realistic usage patterns without
  overfitting to one narrow concept; see
  `../../../Aleks-Sprint/range-training-redesign.md`.
- Range and kNN anchors use a 70/30 sampler: 70% density-weighted lat/lon cells,
  30% uniform points. Similarity and clustering use uniform anchors.
- Range footprint can be dataset-relative (`range_spatial_fraction`,
  `range_time_fraction`) or absolute (`range_spatial_km`, `range_time_hours`).
  The current testing baseline uses absolute half-windows for day-to-day
  stability.
- `range_footprint_jitter` randomizes footprint size. Benchmark profiles set it
  to `0.0` for exact, reproducible footprints.
- `target_coverage` is point-level query-signal coverage. With `max_queries`
  unset, generation keeps `n_queries` fixed and may miss the target. With a
  larger `max_queries`, `n_queries` is the minimum and generation continues
  until the target is reached or the cap is hit.
- Use `scripts/estimate_range_coverage.py` before changing query count,
  footprint, or coverage targets.

## Execution Semantics

- `execute_range_query` returns trajectory IDs with points inside the box; the
  evaluation layer additionally scores retained point hits inside the same box.
- `execute_knn_query` computes the nearest point per trajectory inside the time window and returns the `k` nearest distinct trajectory IDs.
- `execute_similarity_query` filters trajectories by centroid and radius, then ranks trajectory IDs with a lightweight DTW-like distance; evaluation drops ranking and uses set F1.
- `execute_clustering_query` clusters per-trajectory representatives inside the box and returns labels indexed by trajectory ID.
- `execute_typed_query` dispatches by the `type` field and returns the type-specific result object.
