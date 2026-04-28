# Queries Module

This module defines the typed query format used by v2, the workload generator, and the concrete query executors that produce the labels and evaluation targets.

## Files

| File | Purpose |
| --- | --- |
| `query_types.py` | Query type IDs, workload-mix parsing, and padded query feature conversion. |
| `query_generator.py` | Deterministic typed workload generation from a trajectory set. |
| `query_executor.py` | Concrete range, kNN, similarity, and clustering query execution. |

## Query Types

| Name | ID | Input params | Return value |
| --- | --- | --- | --- |
| `range` | 0 | spatial-temporal box | sum of speed in the box |
| `knn` | 1 | anchor point, time window, `k` | set of point indices |
| `similarity` | 2 | centroid, time box, radius, reference snippet | ranked trajectory indices |
| `clustering` | 3 | box, `eps`, `min_samples` | DBSCAN cluster count |

## Workload Format

- `normalize_workload_mix` lowercases the keys, drops non-positive weights, validates query names, and normalizes the mix to sum to 1.
- `parse_workload_mix` accepts CLI strings like `range=0.8,knn=0.2`.
- `pad_query_features` converts heterogeneous typed query dicts into a `[M, 12]` tensor plus a `[M]` tensor of type IDs.
- Feature layout:
    - range: `lat_min`, `lat_max`, `lon_min`, `lon_max`, `t_start`, `t_end`
    - knn: `lat`, `lon`, `t_center`, `t_half_window`, `k`
    - similarity: `lat_query_centroid`, `lon_query_centroid`, `t_start`, `t_end`, `radius`, plus the mean of the reference snippet in slots 5-7 when present
    - clustering: `lat_min`, `lat_max`, `lon_min`, `lon_max`, `t_start`, `t_end`, `eps`, `min_samples`

## Workload Generation

`generate_typed_query_workload` builds a mixed workload from the full trajectory set, allocates query counts from the requested mix, shuffles the result deterministically, and returns the `TypedQueryWorkload` container used by experiments and training.

When `target_coverage` is provided, the generator switches to dynamic mode: it creates queries one at a time, anchors new queries on currently uncovered points when possible, and stops once the union of query-covered points reaches the target or `max_queries` is hit. Coverage is measured as point-level query signal coverage: range/clustering boxes, dense kNN neighbourhoods, and similarity spatiotemporal radius regions.

## Execution Semantics

- `execute_range_query` returns the speed sum inside the box.
- `execute_knn_query` returns the point-index set selected by a spatial-plus-temporal distance.
- `execute_similarity_query` filters trajectories by centroid and radius, then ranks them with a lightweight DTW-like distance.
- `execute_clustering_query` returns the DBSCAN cluster count inside the box.
- `execute_typed_query` dispatches by the `type` field and returns the type-specific result object.
