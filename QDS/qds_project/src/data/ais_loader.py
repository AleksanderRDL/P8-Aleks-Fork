"""AIS data loading utilities. See src/data/README.md for full documentation."""

from __future__ import annotations

import math

import os
from typing import TYPE_CHECKING, List, Optional

import torch
from torch import Tensor

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

CSV_COLUMN_CANDIDATES: dict[str, list[str]] = {
    "mmsi": ["mmsi"],
    "lat": ["lat", "latitude"],
    "lon": ["lon", "longitude"],
    "speed": ["speed", "sog"],
    "heading": ["heading", "cog"],
    "timestamp": ["timestamp", "time", "datetime"],
}


def _make_endpoint_flags(t: int, start: bool) -> "np.ndarray":
    """Return a [T, 1] float32 array with 1.0 at the start or end."""
    import numpy as np
    flags = np.zeros((t, 1), dtype="float32")
    if t > 0:
        flags[0 if start else -1, 0] = 1.0
    return flags


def _normalize_turn_score_method(method: str) -> str:
    """Normalize and validate turn-score method names."""
    norm = method.strip().lower()
    if norm not in {"heading", "geometry"}:
        raise ValueError(
            "Unknown turn_score_method "
            f"'{method}'. Choose from: heading, geometry."
        )
    return norm


def _compute_turn_scores_geometry(lat_lon: "np.ndarray") -> "np.ndarray":
    """Vectorized geometry-based turn score from [lat, lon] points."""
    import numpy as np

    t = lat_lon.shape[0]
    scores = np.zeros((t, 1), dtype="float32")
    if t < 3:
        return scores

    # Interior points use the angle between incoming and outgoing displacement vectors.
    v1 = lat_lon[1:-1] - lat_lon[:-2]
    v2 = lat_lon[2:] - lat_lon[1:-1]

    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    den = n1 * n2
    valid = den > 1e-12

    cos_angle = np.zeros_like(den, dtype="float64")
    dot = np.einsum("ij,ij->i", v1, v2)
    cos_angle[valid] = dot[valid] / den[valid]
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angles = np.arccos(cos_angle) / math.pi
    angles[~valid] = 0.0
    scores[1:-1, 0] = angles.astype("float32")
    return scores


def _compute_turn_scores_heading(heading: "np.ndarray") -> "np.ndarray":
    """Vectorized heading/COG-based turn score using wrapped angular deltas."""
    import numpy as np

    heading_vec = np.asarray(heading, dtype="float64").reshape(-1)
    t = heading_vec.shape[0]
    scores = np.zeros((t, 1), dtype="float32")
    if t < 3:
        return scores

    # Compare heading before/after each interior point using shortest angular distance.
    delta = np.abs(((heading_vec[2:] - heading_vec[:-2] + 180.0) % 360.0) - 180.0)
    delta = np.nan_to_num(delta, nan=0.0, posinf=180.0, neginf=180.0)
    scores[1:-1, 0] = (delta / 180.0).astype("float32")
    return scores


def _compute_turn_scores(
    lat_lon: "np.ndarray",
    heading: Optional["np.ndarray"] = None,
    method: str = "heading",
) -> "np.ndarray":
    """Compute per-point turn scores in [0, 1] using geometry or heading deltas."""
    method_norm = _normalize_turn_score_method(method)
    if method_norm == "geometry":
        return _compute_turn_scores_geometry(lat_lon)
    if heading is None:
        raise ValueError("heading-based turn scores require a heading/COG array.")
    return _compute_turn_scores_heading(heading)


def _normalize_column_name(name: str) -> str:
    """Normalise CSV column names for schema matching."""
    return name.strip().lower().lstrip("#").strip()


def _pick_column(columns: list[str], candidates: list[str]) -> Optional[str]:
    """Return the first matching candidate column from a normalised column list."""
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _resolve_csv_columns(available_columns: list[str]) -> dict[str, Optional[str]]:
    """Resolve normalized CSV column names to the expected semantic fields."""
    resolved = {
        key: _pick_column(available_columns, candidates)
        for key, candidates in CSV_COLUMN_CANDIDATES.items()
    }

    required_missing: list[str] = []
    if resolved["mmsi"] is None:
        required_missing.append("mmsi")
    if resolved["lat"] is None:
        required_missing.append("lat/latitude")
    if resolved["lon"] is None:
        required_missing.append("lon/longitude")
    if resolved["speed"] is None:
        required_missing.append("speed/sog")
    if required_missing:
        raise ValueError(f"CSV is missing required columns: {set(required_missing)}")

    return resolved


def _coerce_numeric_columns(
    df: "pd.DataFrame",
    *,
    mmsi_col: str,
    lat_col: str,
    lon_col: str,
    speed_col: str,
    heading_col: Optional[str],
) -> str:
    """Convert core numeric columns and ensure a usable heading column exists."""
    import pandas as pd

    df[mmsi_col] = pd.to_numeric(df[mmsi_col], errors="coerce")
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df[speed_col] = pd.to_numeric(df[speed_col], errors="coerce").fillna(0.0)

    if heading_col is not None:
        df[heading_col] = pd.to_numeric(df[heading_col], errors="coerce").fillna(0.0)
        return heading_col

    synthetic_heading_col = "_synthetic_heading"
    df[synthetic_heading_col] = 0.0
    return synthetic_heading_col


def _ensure_timestamp_column(
    df: "pd.DataFrame",
    *,
    mmsi_col: str,
    timestamp_col: Optional[str],
) -> str:
    """Return a valid timestamp column, synthesizing one when needed."""
    import pandas as pd

    if timestamp_col is not None:
        df[timestamp_col] = pd.to_numeric(df[timestamp_col], errors="coerce")

    has_valid_timestamp = (
        timestamp_col is not None and bool(df[timestamp_col].notna().any())
    )
    if has_valid_timestamp:
        seq = df.groupby(mmsi_col, sort=False).cumcount().astype(float)
        vessel_base = df.groupby(mmsi_col, sort=False)[timestamp_col].transform("min")
        missing_base = pd.Series(range(len(df)), index=df.index, dtype="float64") * 60.0
        vessel_base = vessel_base.fillna(missing_base)
        df[timestamp_col] = df[timestamp_col].fillna(vessel_base + seq * 60.0)
        return timestamp_col

    synthetic_timestamp_col = "_synthetic_timestamp"
    df[synthetic_timestamp_col] = (
        pd.Series(range(len(df)), index=df.index, dtype="float64") * 60.0
    )
    return synthetic_timestamp_col


def _augment_values_with_flags_and_turn_scores(
    values: "np.ndarray",
    *,
    method: str,
) -> "np.ndarray":
    """Append [is_start, is_end, turn_score] columns to base [T, 5] values."""
    import numpy as np

    n_points = values.shape[0]
    is_start = _make_endpoint_flags(n_points, start=True)
    is_end = _make_endpoint_flags(n_points, start=False)
    turn_scores = _compute_turn_scores(
        values[:, 1:3],
        heading=values[:, 4],
        method=method,
    )
    return np.concatenate([values, is_start, is_end, turn_scores], axis=1)


def _group_to_trajectory_tensor(
    group: "pd.DataFrame",
    *,
    timestamp_col: str,
    lat_col: str,
    lon_col: str,
    speed_col: str,
    heading_col: str,
    method: str,
) -> Tensor:
    """Convert one vessel group to [T, 8] tensor with endpoint flags and turn scores."""
    ordered_group = group.sort_values(timestamp_col, kind="mergesort")
    values = ordered_group[
        [timestamp_col, lat_col, lon_col, speed_col, heading_col]
    ].to_numpy(dtype="float32", copy=True)
    return torch.from_numpy(
        _augment_values_with_flags_and_turn_scores(values, method=method)
    )


def load_ais_csv(
    filepath: str,
    turn_score_method: str = "heading",
) -> List[Tensor]:
    """Load AIS trajectories from a CSV file."""
    import pandas as pd  # optional dependency; checked at call-time

    method = _normalize_turn_score_method(turn_score_method)

    header = pd.read_csv(filepath, nrows=0)
    raw_columns = list(header.columns)
    norm_to_raw = {_normalize_column_name(col): col for col in raw_columns}
    resolved = _resolve_csv_columns(list(norm_to_raw.keys()))
    assert resolved["mmsi"] is not None
    assert resolved["lat"] is not None
    assert resolved["lon"] is not None
    assert resolved["speed"] is not None

    selected_norm = [
        resolved["mmsi"],
        resolved["lat"],
        resolved["lon"],
        resolved["speed"],
    ]
    if resolved["heading"] is not None:
        selected_norm.append(resolved["heading"])
    if resolved["timestamp"] is not None:
        selected_norm.append(resolved["timestamp"])

    usecols = [norm_to_raw[n] for n in selected_norm]
    df = pd.read_csv(filepath, usecols=usecols, low_memory=False)
    df.columns = [_normalize_column_name(col) for col in df.columns]

    mmsi_col = resolved["mmsi"]
    lat_col = resolved["lat"]
    lon_col = resolved["lon"]
    speed_col = resolved["speed"]
    heading_col = _coerce_numeric_columns(
        df,
        mmsi_col=mmsi_col,
        lat_col=lat_col,
        lon_col=lon_col,
        speed_col=speed_col,
        heading_col=resolved["heading"],
    )

    # Remove rows with invalid vessel/position fields.
    df = df.dropna(subset=[mmsi_col, lat_col, lon_col])
    timestamp_col = _ensure_timestamp_column(
        df,
        mmsi_col=mmsi_col,
        timestamp_col=resolved["timestamp"],
    )

    trajectories: List[Tensor] = []
    for _, group in df.groupby(mmsi_col, sort=False):
        trajectories.append(
            _group_to_trajectory_tensor(
                group,
                timestamp_col=timestamp_col,
                lat_col=lat_col,
                lon_col=lon_col,
                speed_col=speed_col,
                heading_col=heading_col,
                method=method,
            )
        )

    return trajectories


def generate_synthetic_ais_data(
    n_ships: int = 10,
    n_points_per_ship: int = 100,
    save_path: Optional[str] = None,
    turn_score_method: str = "heading",
) -> List[Tensor]:
    """Generate synthetic AIS trajectory data for testing."""
    torch.manual_seed(42)  # reproducibility
    method = _normalize_turn_score_method(turn_score_method)

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

        values = torch.tensor(points, dtype=torch.float32).numpy()  # [T, 5]
        augmented = _augment_values_with_flags_and_turn_scores(values, method=method)
        trajectories.append(torch.from_numpy(augmented))  # [T, 8]

    # Optionally persist to CSV
    if save_path is not None:
        import pandas as pd

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        pd.DataFrame(all_rows).to_csv(save_path, index=False)

    return trajectories
