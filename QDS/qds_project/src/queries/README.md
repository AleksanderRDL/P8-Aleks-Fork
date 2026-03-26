# Queries Module

Generates and executes spatiotemporal range queries over AIS trajectory data.

---

## Components

### `query_generator.py`

Generates random spatiotemporal range queries derived from the actual
trajectory data bounds. Each query is a 6-element vector:

```
[lat_min, lat_max, lon_min, lon_max, time_start, time_end]
```

Query extents are sampled as fractions of the total data range so that each
query covers a realistic portion of the dataset.

#### Query Generation Strategies

**`generate_uniform_queries(trajectories, n_queries, spatial_fraction, temporal_fraction)`**  
Query centres are drawn independently and uniformly from the full lat/lon/time
extent of the dataset. Models an unbiased, ad-hoc workload where any location
is equally likely to be queried.

```
center_lat  ~ Uniform(lat_min,  lat_max)
center_lon  ~ Uniform(lon_min,  lon_max)
time_center ~ Uniform(time_min, time_max)
width       ~ Uniform(w_min, w_max) * effective_range
```

**`generate_density_biased_queries(trajectories, n_queries, spatial_fraction, temporal_fraction)`**  
Query centres are anchored to real AIS data points rather than drawn from the
bounding box. Because the point cloud is denser in high-traffic areas (ports,
straits, shipping lanes), this naturally concentrates queries where vessel
activity is greatest, closely approximating a realistic maritime workload.

```
anchor      = random AIS data point
center_lat  = anchor.lat
center_lon  = anchor.lon
time_center ~ Uniform(time_min, time_max)
```

**`generate_mixed_queries(trajectories, total_queries, density_ratio, ...)`**  
A shuffled blend of uniform and density-biased queries. The `density_ratio`
parameter (default 0.5) controls the fraction that are density-biased.

**`generate_spatiotemporal_queries(trajectories, n_queries, ...)`** *(deprecated)*  
Backward-compatible wrapper that delegates to `generate_density_biased_queries`
or `generate_uniform_queries` based on the `anchor_to_data` flag.

#### Robust Spatial Bounds

Width sampling uses a clipped spread estimate based on the 5th–95th
percentiles (`_effective_spatial_ranges`) to prevent extremely wide queries
when the global bounding box is dominated by outliers. Center placement uses
1st–99th percentile quantile bounds (`_effective_spatial_bounds`).

---

### `query_executor.py`

Executes spatiotemporal range queries against a point cloud.

A query selects all points whose `(lat, lon, time)` fall inside the query
rectangle and returns the **SUM of the speed column** for those points.

**`run_query(points, query)`**  
Run a single query against a `[N, 5]` point cloud. Returns a scalar tensor.

**`run_queries(points, queries)`**  
Run all `M` queries in a vectorised, chunked manner against the point cloud
to avoid large `[N, M]` memory allocations. Returns a `[M]` result tensor.

Point tensor columns: `[time, lat, lon, speed, heading]`  
Query tensor columns: `[lat_min, lat_max, lon_min, lon_max, time_start, time_end]`

---

### `query_masks.py`

Shared masking utilities used across query execution, label generation,
visualisation, and experiment diagnostics.

- **`spatial_inclusion_mask(points, queries)`** → `[N, M]` boolean mask
- **`spatiotemporal_inclusion_mask(points, queries, spatial_mask=None)`** → `[N, M]`
- **`sum_speed_by_query(points, inclusion_mask, absolute=False)`** → `[M]`

This centralises point/query column semantics and keeps masking logic
consistent across modules.
