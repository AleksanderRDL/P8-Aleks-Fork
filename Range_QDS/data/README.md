# Data Module

Loads AIS trajectories into tensors and exposes flattened points plus
trajectory boundaries.

## Files

| File | Purpose |
| --- | --- |
| `ais_loader.py` | Load cleaned AIS CSVs or generate synthetic trajectories. |
| `combine_days.py` | Concatenate preprocessed CSVs while preserving MMSIs. |
| `trajectory_cache.py` | Parquet cache for segmented trajectory tensors. |
| `trajectory_dataset.py` | Flattened points and trajectory boundary helper. |
| `trajectory_index.py` | Shared boundary, split, and trajectory-id helpers for flattened tensors. |

## Tensor Schema

Each trajectory tensor has 8 columns:

| Col | Meaning |
| --- | --- |
| 0 | time in seconds |
| 1 | latitude |
| 2 | longitude |
| 3 | speed |
| 4 | heading |
| 5 | `is_start` flag |
| 6 | `is_end` flag |
| 7 | `turn_score` |

## Loader Behavior

- Accepts common AIS column aliases for MMSI, lat/lon, speed, heading, and time.
- Groups by vessel, sorts by timestamp, and drops segments shorter than 4 points.
- Splits one MMSI track when consecutive points exceed `max_time_gap_seconds`
  (`3600` by default).
- `max_points_per_segment`, `max_segments`, split-specific segment caps, and
  `min_points_per_segment` are smoke/runtime controls, not benchmark-quality
  defaults.
- `return_audit=True` returns invalid-row, duplicate, segmentation, and
  downsampling diagnostics.
- Synthetic runs use independent random routes by default. Set
  `--synthetic_route_families N` to generate ships around `N` shared corridors
  when testing same-support query-prior behavior.

## Cache

`trajectory_cache.py` stores post-segmentation tensors as `points.parquet` plus
`manifest.json`. The cache key includes source identity, schema version, and
loader config. Use `--cache_dir` to enable and `--refresh_cache` to rebuild.

## Boundary Contract

`TrajectoryDataset.get_all_points()` returns `[N, F]`;
`get_trajectory_boundaries()` returns `(start, end)` pairs. Training,
simplification, and evaluation rely on these boundaries to avoid crossing
trajectory ownership.
