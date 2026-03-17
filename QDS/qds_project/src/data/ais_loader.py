"""
ais_loader.py

Load AIS (Automatic Identification System) trajectory data from CSV files
and provide a synthetic data generator for testing.

AIS CSV format expected: mmsi, timestamp, lat, lon, speed, heading
Each row represents a position report for a vessel.

The loader groups rows by MMSI (vessel identifier), sorts each vessel's
track by timestamp, and returns tensors of shape [T, 7] where the columns
are [time, lat, lon, speed, heading, is_start, is_end].

The ``is_start`` and ``is_end`` binary flags are set to 1.0 for the first
and last points of each trajectory respectively, and 0.0 otherwise.  They
allow the model to learn that trajectory endpoints may have higher importance.
"""

from __future__ import annotations

import math

import os
from typing import List, Optional

import torch
from torch import Tensor


def _make_endpoint_flags(t: int, start: bool) -> "numpy.ndarray":
    """Return a [T, 1] float32 array with 1.0 at the start or end index."""
    import numpy as np
    flags = np.zeros((t, 1), dtype="float32")
    if t > 0:
        flags[0 if start else -1, 0] = 1.0
    return flags


def _normalize_column_name(name: str) -> str:
    """Normalise CSV column names for schema matching."""
    return name.strip().lower().lstrip("#").strip()


def _pick_column(columns: list[str], candidates: list[str]) -> Optional[str]:
    """Return the first matching candidate column from a normalised column list."""
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def load_ais_csv(filepath: str) -> List[Tensor]:
    """Load AIS trajectories from a CSV file.

    Supports common schema variants, including AISDK-style fields like
    ``Latitude``, ``Longitude``, and ``SOG``. If timestamp values are missing,
    synthetic per-vessel timestamps are generated from row order so the
    spatiotemporal pipeline can still run.

    Args:
        filepath: Path to the CSV file.

    Returns:
        List of tensors, one per vessel, each of shape [T, 7] with columns
        [time, lat, lon, speed, heading, is_start, is_end].
    """
    import pandas as pd  # optional dependency; checked at call-time

    header = pd.read_csv(filepath, nrows=0)
    raw_columns = list(header.columns)
    norm_to_raw = {_normalize_column_name(col): col for col in raw_columns}

    available = list(norm_to_raw.keys())
    mmsi_col_norm = _pick_column(available, ["mmsi"])
    lat_col_norm = _pick_column(available, ["lat", "latitude"])
    lon_col_norm = _pick_column(available, ["lon", "longitude"])
    speed_col_norm = _pick_column(available, ["speed", "sog"])
    heading_col_norm = _pick_column(available, ["heading", "cog"])
    timestamp_col_norm = _pick_column(available, ["timestamp", "time", "datetime"])

    required_missing: list[str] = []
    if mmsi_col_norm is None:
        required_missing.append("mmsi")
    if lat_col_norm is None:
        required_missing.append("lat/latitude")
    if lon_col_norm is None:
        required_missing.append("lon/longitude")
    if speed_col_norm is None:
        required_missing.append("speed/sog")
    if required_missing:
        raise ValueError(f"CSV is missing required columns: {set(required_missing)}")

    selected_norm = [mmsi_col_norm, lat_col_norm, lon_col_norm, speed_col_norm]
    if heading_col_norm is not None:
        selected_norm.append(heading_col_norm)
    if timestamp_col_norm is not None:
        selected_norm.append(timestamp_col_norm)

    usecols = [norm_to_raw[n] for n in selected_norm]
    df = pd.read_csv(filepath, usecols=usecols, low_memory=False)
    df.columns = [_normalize_column_name(col) for col in df.columns]

    mmsi_col = mmsi_col_norm
    lat_col = lat_col_norm
    lon_col = lon_col_norm
    speed_col = speed_col_norm
    heading_col = heading_col_norm
    timestamp_col = timestamp_col_norm

    # Convert required numeric fields.
    df[mmsi_col] = pd.to_numeric(df[mmsi_col], errors="coerce")
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df[speed_col] = pd.to_numeric(df[speed_col], errors="coerce").fillna(0.0)

    if heading_col is not None:
        df[heading_col] = pd.to_numeric(df[heading_col], errors="coerce").fillna(0.0)
    else:
        heading_col = "_synthetic_heading"
        df[heading_col] = 0.0

    # Remove rows with invalid vessel/position fields.
    df = df.dropna(subset=[mmsi_col, lat_col, lon_col])

    # Build/clean timestamp column.
    if timestamp_col is not None:
        df[timestamp_col] = pd.to_numeric(df[timestamp_col], errors="coerce")

    has_valid_timestamp = (
        timestamp_col is not None and bool(df[timestamp_col].notna().any())
    )

    if has_valid_timestamp:
        # Fill remaining missing timestamps with per-vessel monotonic fallback.
        seq = df.groupby(mmsi_col, sort=False).cumcount().astype(float)
        vessel_base = df.groupby(mmsi_col, sort=False)[timestamp_col].transform("min")
        missing_base = (pd.Series(range(len(df)), index=df.index, dtype="float64") * 60.0)
        vessel_base = vessel_base.fillna(missing_base)
        df[timestamp_col] = df[timestamp_col].fillna(vessel_base + seq * 60.0)
    else:
        # No timestamp information available: synthesize contiguous global timeline.
        timestamp_col = "_synthetic_timestamp"
        df[timestamp_col] = pd.Series(range(len(df)), index=df.index, dtype="float64") * 60.0

    trajectories: List[Tensor] = []
    import numpy as np
    for _, group in df.groupby(mmsi_col, sort=False):
        group = group.sort_values(timestamp_col, kind="mergesort")
        # Keep only [time, lat, lon, speed, heading] — drop mmsi
        values = group[[timestamp_col, lat_col, lon_col, speed_col, heading_col]].to_numpy(
            dtype="float32",
            copy=True,
        )
        t = values.shape[0]
        is_start = _make_endpoint_flags(t, start=True)
        is_end = _make_endpoint_flags(t, start=False)
        values = np.concatenate([values, is_start, is_end], axis=1)
        trajectories.append(torch.from_numpy(values))

    return trajectories


def generate_synthetic_ais_data(
    n_ships: int = 10,
    n_points_per_ship: int = 100,
    save_path: Optional[str] = None,
) -> List[Tensor]:
    """Generate synthetic AIS trajectory data for testing.

    Ships move on straight-ish paths with small random perturbations,
    placed in a North Sea–like region (lat 50–60, lon 0–20).

    Args:
        n_ships:           Number of vessels to simulate.
        n_points_per_ship: Number of position reports per vessel.
        save_path:         If provided, save the data as a CSV to this path.

    Returns:
        List of tensors, one per vessel, each of shape [T, 7] with columns
        [time, lat, lon, speed, heading, is_start, is_end].
    """
    torch.manual_seed(42)  # reproducibility

    # Geographic bounds (North Sea-like region)
    LAT_MIN, LAT_MAX = 50.0, 60.0
    LON_MIN, LON_MAX = 0.0, 20.0
    SPEED_MEAN, SPEED_STD = 12.0, 3.0   # knots
    TIME_STEP = 600.0                     # seconds between reports (10 min)

    all_rows: list = []  # for optional CSV export
    trajectories: List[Tensor] = []

    for ship_id in range(n_ships):
        # Random starting position
        start_lat = LAT_MIN + torch.rand(1).item() * (LAT_MAX - LAT_MIN)
        start_lon = LON_MIN + torch.rand(1).item() * (LON_MAX - LON_MIN)

        # Random constant heading (degrees) with small drift each step
        heading = torch.rand(1).item() * 360.0
        speed = max(1.0, SPEED_MEAN + torch.randn(1).item() * SPEED_STD)

        lat, lon = start_lat, start_lon
        base_time = float(ship_id * 1000)  # stagger start times

        points: list = []
        for t in range(n_points_per_ship):
            timestamp = base_time + t * TIME_STEP
            # Small random heading drift
            heading = (heading + torch.randn(1).item() * 5.0) % 360.0
            step_speed = max(0.5, speed + torch.randn(1).item() * 0.5)

            # Convert speed (knots) + heading to lat/lon displacement
            # 1 knot ≈ 0.000278 degrees lat per second
            heading_rad = heading * math.pi / 180.0
            dt_hours = TIME_STEP / 3600.0
            dlat = math.cos(heading_rad) * step_speed * dt_hours / 60.0
            dlon = math.sin(heading_rad) * step_speed * dt_hours / (
                60.0 * math.cos(lat * math.pi / 180.0) + 1e-9
            )

            lat = max(LAT_MIN, min(LAT_MAX, lat + dlat))
            lon = max(LON_MIN, min(LON_MAX, lon + dlon))

            points.append([timestamp, lat, lon, step_speed, heading])
            if save_path is not None:
                all_rows.append(
                    {
                        "mmsi": 100000000 + ship_id,
                        "timestamp": timestamp,
                        "lat": lat,
                        "lon": lon,
                        "speed": step_speed,
                        "heading": heading,
                    }
                )

        tensor = torch.tensor(points, dtype=torch.float32)  # [T, 5]
        n_pts = tensor.shape[0]
        is_start = torch.zeros(n_pts, 1, dtype=torch.float32)
        is_end = torch.zeros(n_pts, 1, dtype=torch.float32)
        if n_pts > 0:
            is_start[0, 0] = 1.0
            is_end[-1, 0] = 1.0
        trajectories.append(torch.cat([tensor, is_start, is_end], dim=1))  # [T, 7]

    # Optionally persist to CSV
    if save_path is not None:
        import pandas as pd

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        pd.DataFrame(all_rows).to_csv(save_path, index=False)

    return trajectories
