# Data Module

This module loads AIS trajectories into per-trajectory tensors and provides the flattened and boundary views used by training, simplification, and evaluation.

## Files

| File | Purpose |
| --- | --- |
| `ais_loader.py` | Load AIS CSV files or generate deterministic synthetic trajectories. |
| `combine_days.py` | Utility for concatenating preprocessed CSVs while preserving MMSIs by default. |
| `trajectory_cache.py` | Persist segmented AIS tensors as Parquet plus manifest/audit metadata. |
| `trajectory_dataset.py` | Wrap a list of trajectory tensors and expose flattened points plus trajectory boundaries. |

## AIS Tensor Schema

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
| 7 | `turn_score` (normalized absolute heading change) |

## Loader Behavior

- `load_ais_csv(...)` accepts common column aliases: `mmsi` / `ship_id` / `vessel_id`, `lat` / `latitude`, `lon` / `longitude`, `speed` / `sog`, `heading` / `cog`, and `timestamp` / `time` / `datetime`.
- Rows are grouped by vessel id, sorted by timestamp, and trajectories shorter than 4 points are dropped.
- By default, one MMSI track is split into new trajectory segments when consecutive points are more than `3600` seconds apart.
- `min_points_per_segment` controls short-segment dropping.
- `max_points_per_segment` down-samples long segments with evenly spaced indices.
- `max_time_gap_seconds=None` disables time-gap segmentation.
- `max_segments` is a loader-level safety cap for smoke runs.
- `return_audit=True` appends an `AISLoadAudit` object containing invalid-row counts, duplicate timestamp counts, time-gap stats, segment length stats, segment counts, dropped short segments, and downsampling counts.
- `generate_synthetic_ais_data(n_ships=24, n_points_per_ship=200, seed=42)` produces deterministic pseudo-realistic trajectories when no CSV is supplied.

## Segmented Cache

`trajectory_cache.py` stores the post-cleaning, post-segmentation tensor rows as
`points.parquet` with a sibling `manifest.json`. The manifest is keyed by source
file path, file size, modification time, cache schema version, and segmentation
config, so changing the source file or controls creates a separate cache entry.
Use `--cache_dir` on experiment/inference commands to enable it, and
`--refresh_cache` to rebuild a matching entry.

## Multi-Day CSV Combination

Use `python -m src.data.combine_days --input ... --output ...` for ad hoc
multi-day experiments that need one combined CSV. The current testing-baseline
benchmark normally uses separate train/eval CSV days instead. The utility
preserves MMSIs so `load_ais_csv` can segment continuous vessels by timestamp
gaps across file boundaries.

## Dataset Helpers

- `TrajectoryDataset.get_all_points()` concatenates all trajectories into one `[N, F]` tensor.
- `TrajectoryDataset.get_trajectory_boundaries()` returns `(start, end)` pairs for each trajectory in flattened order.
- Trajectory boundaries are the contract used by the model, training loop, simplification, and evaluation code to keep attention and metrics trajectory-local.
