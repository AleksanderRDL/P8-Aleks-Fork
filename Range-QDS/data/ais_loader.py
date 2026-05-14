"""AIS trajectory loading and synthetic generation utilities. See data/README.md for details."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any, Literal, overload, cast

import pandas as pd
import torch

DEFAULT_MIN_POINTS_PER_SEGMENT = 4
DEFAULT_MAX_TIME_GAP_SECONDS = 3600.0


@dataclass
class AISLoadConfig:
    """Configuration for CSV cleaning, segmentation, and optional downsampling."""

    min_points_per_segment: int = DEFAULT_MIN_POINTS_PER_SEGMENT
    max_points_per_segment: int | None = None
    max_time_gap_seconds: float | None = DEFAULT_MAX_TIME_GAP_SECONDS
    max_segments: int | None = None

    def validate(self) -> None:
        """Validate segmentation controls before loading data."""
        if int(self.min_points_per_segment) <= 0:
            raise ValueError("min_points_per_segment must be positive.")
        if self.max_points_per_segment is not None and int(self.max_points_per_segment) <= 0:
            raise ValueError("max_points_per_segment must be positive when provided.")
        if self.max_time_gap_seconds is not None and float(self.max_time_gap_seconds) <= 0.0:
            raise ValueError("max_time_gap_seconds must be positive when provided.")
        if self.max_segments is not None and int(self.max_segments) <= 0:
            raise ValueError("max_segments must be positive when provided.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize load config for experiment artifacts."""
        return asdict(self)


@dataclass
class AISLoadAudit:
    """Data-audit summary produced while loading and segmenting an AIS CSV."""

    source_path: str
    rows_loaded: int
    rows_after_cleaning: int
    rows_dropped_invalid: int
    invalid_time_rows: int
    invalid_lat_rows: int
    invalid_lon_rows: int
    invalid_speed_rows: int
    invalid_heading_rows: int
    duplicate_timestamp_rows: int
    input_mmsi_count: int
    output_segment_count: int
    output_point_count: int
    dropped_short_segments: int
    downsampled_segments: int
    segment_limit_reached: bool
    time_gap_over_threshold_count: int
    segment_length_stats: dict[str, float] = field(default_factory=dict)
    time_gap_stats: dict[str, float] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize audit for JSON result dumps."""
        return asdict(self)


def _to_time_seconds(series: pd.Series) -> pd.Series:
    """Convert timestamp-like values to floating-point seconds. See data/README.md for details."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    dt = pd.to_datetime(series, utc=True, errors="coerce")
    if dt.isna().all():
        return pd.Series(range(len(series)), dtype=float)
    base = dt.min()
    return (dt - base).dt.total_seconds().astype(float)


def _resolve_col(df: pd.DataFrame, aliases: list[str]) -> str:
    """Resolve a canonical column from aliases. See data/README.md for details."""
    lowered = {c.lower(): c for c in df.columns}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    raise ValueError(f"Missing required column aliases: {aliases}")


def _summary_stats(values: list[float]) -> dict[str, float]:
    """Return compact distribution stats for audit logs."""
    if not values:
        return {
            "count": 0.0,
            "min": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "mean": 0.0,
        }
    t = torch.tensor(values, dtype=torch.float32)
    return {
        "count": float(t.numel()),
        "min": float(t.min().item()),
        "p50": float(torch.quantile(t, 0.50).item()),
        "p75": float(torch.quantile(t, 0.75).item()),
        "p95": float(torch.quantile(t, 0.95).item()),
        "p99": float(torch.quantile(t, 0.99).item()),
        "max": float(t.max().item()),
        "mean": float(t.mean().item()),
    }


def _safe_int(value: Any) -> int:
    """Convert MMSI-like values to stable integer IDs for downstream writers."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _trajectory_tensor_from_group(
    grp: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    speed_col: str,
    heading_col: str,
) -> torch.Tensor:
    """Convert one cleaned, time-continuous segment into the AIS tensor schema."""
    t = torch.tensor(grp["_time"].to_numpy(), dtype=torch.float32)
    lat = torch.tensor(grp[lat_col].to_numpy(), dtype=torch.float32)
    lon = torch.tensor(grp[lon_col].to_numpy(), dtype=torch.float32)
    speed = torch.tensor(grp[speed_col].to_numpy(), dtype=torch.float32)
    heading = torch.tensor(grp[heading_col].to_numpy(), dtype=torch.float32)

    is_start = torch.zeros_like(t)
    is_end = torch.zeros_like(t)
    is_start[0] = 1.0
    is_end[-1] = 1.0

    turn_score = torch.zeros_like(t)
    if len(t) > 2:
        d = torch.abs(heading[1:] - heading[:-1])
        d = torch.minimum(d, 360.0 - d)
        turn_score[1:] = d / 180.0

    return torch.stack([t, lat, lon, speed, heading, is_start, is_end, turn_score], dim=1)


LoadAISReturn = (
    list[torch.Tensor]
    | tuple[list[torch.Tensor], list[int]]
    | tuple[list[torch.Tensor], AISLoadAudit]
    | tuple[list[torch.Tensor], list[int], AISLoadAudit]
)


@overload
def load_ais_csv(
    csv_path: str,
    *,
    return_mmsis: Literal[True],
    min_points_per_segment: int = DEFAULT_MIN_POINTS_PER_SEGMENT,
    max_points_per_segment: int | None = None,
    max_time_gap_seconds: float | None = DEFAULT_MAX_TIME_GAP_SECONDS,
    max_segments: int | None = None,
    return_audit: Literal[True],
) -> tuple[list[torch.Tensor], list[int], AISLoadAudit]: ...


@overload
def load_ais_csv(
    csv_path: str,
    *,
    return_mmsis: Literal[True],
    min_points_per_segment: int = DEFAULT_MIN_POINTS_PER_SEGMENT,
    max_points_per_segment: int | None = None,
    max_time_gap_seconds: float | None = DEFAULT_MAX_TIME_GAP_SECONDS,
    max_segments: int | None = None,
    return_audit: Literal[False] = False,
) -> tuple[list[torch.Tensor], list[int]]: ...


@overload
def load_ais_csv(
    csv_path: str,
    *,
    return_mmsis: Literal[False] = False,
    min_points_per_segment: int = DEFAULT_MIN_POINTS_PER_SEGMENT,
    max_points_per_segment: int | None = None,
    max_time_gap_seconds: float | None = DEFAULT_MAX_TIME_GAP_SECONDS,
    max_segments: int | None = None,
    return_audit: Literal[True],
) -> tuple[list[torch.Tensor], AISLoadAudit]: ...


@overload
def load_ais_csv(
    csv_path: str,
    *,
    return_mmsis: Literal[False] = False,
    min_points_per_segment: int = DEFAULT_MIN_POINTS_PER_SEGMENT,
    max_points_per_segment: int | None = None,
    max_time_gap_seconds: float | None = DEFAULT_MAX_TIME_GAP_SECONDS,
    max_segments: int | None = None,
    return_audit: Literal[False] = False,
) -> list[torch.Tensor]: ...


@overload
def load_ais_csv(
    csv_path: str,
    *,
    return_mmsis: bool = False,
    min_points_per_segment: int = DEFAULT_MIN_POINTS_PER_SEGMENT,
    max_points_per_segment: int | None = None,
    max_time_gap_seconds: float | None = DEFAULT_MAX_TIME_GAP_SECONDS,
    max_segments: int | None = None,
    return_audit: bool = False,
) -> LoadAISReturn: ...


def load_ais_csv(
    csv_path: str,
    *,
    return_mmsis: bool = False,
    min_points_per_segment: int = DEFAULT_MIN_POINTS_PER_SEGMENT,
    max_points_per_segment: int | None = None,
    max_time_gap_seconds: float | None = DEFAULT_MAX_TIME_GAP_SECONDS,
    max_segments: int | None = None,
    return_audit: bool = False,
) -> LoadAISReturn:
    """Load AIS trajectories from CSV into per-trajectory tensors.

    If ``return_mmsis=True``, also return the original MMSI identifiers aligned
    with the trajectory list so downstream writers can preserve vessel IDs.
    If ``return_audit=True``, append an ``AISLoadAudit`` to the return tuple.
    See ``data/README.md`` for details.
    """
    config = AISLoadConfig(
        min_points_per_segment=int(min_points_per_segment),
        max_points_per_segment=max_points_per_segment,
        max_time_gap_seconds=max_time_gap_seconds,
        max_segments=max_segments,
    )
    config.validate()

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV path does not exist: {csv_path}")

    df = pd.read_csv(path, low_memory=False)
    rows_loaded = int(len(df))
    df.columns = [str(c).lstrip("#").strip() for c in df.columns]
    mmsi_col = _resolve_col(df, ["mmsi", "ship_id", "vessel_id"])
    lat_col = _resolve_col(df, ["lat", "latitude"])
    lon_col = _resolve_col(df, ["lon", "longitude"])
    speed_col = _resolve_col(df, ["speed", "sog"])
    heading_col = _resolve_col(df, ["heading", "cog"])
    time_col = _resolve_col(df, ["timestamp", "time", "datetime"])

    df = df[[mmsi_col, lat_col, lon_col, speed_col, heading_col, time_col]].copy()
    df["_time"] = _to_time_seconds(cast(pd.Series, df[time_col]))

    # Coerce numeric columns so non-numeric entries become NaN, then drop
    # invalid rows. AIS feeds frequently contain missing heading/speed and
    # sentinel values (e.g. heading=511) that would propagate as NaN through
    # min-max normalization and collapse training to loss=NaN from epoch 1.
    for col in (lat_col, lon_col, speed_col, heading_col):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    time_values = cast(pd.Series, df["_time"])
    lat_values = cast(pd.Series, df[lat_col])
    lon_values = cast(pd.Series, df[lon_col])
    speed_values = cast(pd.Series, df[speed_col])
    heading_values = cast(pd.Series, df[heading_col])
    invalid_time = time_values.isna()
    invalid_lat = lat_values.isna() | (lat_values < -90) | (lat_values > 90)
    invalid_lon = lon_values.isna() | (lon_values < -180) | (lon_values > 180)
    invalid_speed = speed_values.isna() | (speed_values < 0) | (speed_values > 102.2)
    invalid_heading = heading_values.isna() | (heading_values < 0) | (heading_values >= 360)
    duplicate_key_frame = cast(pd.DataFrame, df[[mmsi_col, "_time"]])
    duplicate_timestamp_rows = int(duplicate_key_frame.duplicated(keep=False).sum())

    df.loc[invalid_heading, heading_col] = float("nan")
    df.loc[invalid_speed, speed_col] = float("nan")
    df.loc[invalid_lat, lat_col] = float("nan")
    df.loc[invalid_lon, lon_col] = float("nan")
    valid_rows = ~(invalid_time | invalid_lat | invalid_lon | invalid_speed | invalid_heading)
    df = df.loc[valid_rows].copy()

    df = df.sort_values([mmsi_col, "_time"]).reset_index(drop=True)

    trajectories: list[torch.Tensor] = []
    mmsis: list[int] = []
    segment_lengths: list[float] = []
    time_gaps: list[float] = []
    dropped_short_segments = 0
    downsampled_segments = 0
    time_gap_over_threshold_count = 0
    segment_limit_reached = False

    for mmsi_val, grp in df.groupby(mmsi_col, sort=False):
        if len(grp) == 0:
            continue
        gap_series = grp["_time"].diff()
        group_gaps = [float(v) for v in gap_series.dropna().tolist()]
        time_gaps.extend(group_gaps)
        if config.max_time_gap_seconds is None:
            split_mask = pd.Series(False, index=grp.index)
        else:
            split_mask = gap_series > float(config.max_time_gap_seconds)
            time_gap_over_threshold_count += int(split_mask.sum())
        segment_ids = split_mask.fillna(False).cumsum()

        for _, seg in grp.groupby(segment_ids, sort=False):
            if len(seg) < int(config.min_points_per_segment):
                dropped_short_segments += 1
                continue
            if config.max_points_per_segment is not None and len(seg) > int(config.max_points_per_segment):
                idx = torch.linspace(0, len(seg) - 1, steps=int(config.max_points_per_segment)).long().tolist()
                seg = seg.iloc[idx]
                downsampled_segments += 1

            trajectories.append(_trajectory_tensor_from_group(seg, lat_col, lon_col, speed_col, heading_col))
            mmsis.append(_safe_int(mmsi_val))
            segment_lengths.append(float(len(seg)))

            if config.max_segments is not None and len(trajectories) >= int(config.max_segments):
                segment_limit_reached = True
                break
        if segment_limit_reached:
            break

    if not trajectories:
        raise ValueError("No valid trajectories found in CSV.")

    rows_after_cleaning = int(len(df))
    audit = AISLoadAudit(
        source_path=str(path),
        rows_loaded=rows_loaded,
        rows_after_cleaning=rows_after_cleaning,
        rows_dropped_invalid=int(rows_loaded - rows_after_cleaning),
        invalid_time_rows=int(invalid_time.sum()),
        invalid_lat_rows=int(invalid_lat.sum()),
        invalid_lon_rows=int(invalid_lon.sum()),
        invalid_speed_rows=int(invalid_speed.sum()),
        invalid_heading_rows=int(invalid_heading.sum()),
        duplicate_timestamp_rows=duplicate_timestamp_rows,
        input_mmsi_count=int(cast(pd.Series, df[mmsi_col]).nunique()),
        output_segment_count=int(len(trajectories)),
        output_point_count=int(sum(t.shape[0] for t in trajectories)),
        dropped_short_segments=int(dropped_short_segments),
        downsampled_segments=int(downsampled_segments),
        segment_limit_reached=bool(segment_limit_reached),
        time_gap_over_threshold_count=int(time_gap_over_threshold_count),
        segment_length_stats=_summary_stats(segment_lengths),
        time_gap_stats=_summary_stats(time_gaps),
        config=config.to_dict(),
    )

    if return_mmsis and return_audit:
        return trajectories, mmsis, audit
    if return_mmsis:
        return trajectories, mmsis
    if return_audit:
        return trajectories, audit
    return trajectories


def generate_synthetic_ais_data(
    n_ships: int = 24,
    n_points_per_ship: int = 200,
    seed: int = 42,
) -> list[torch.Tensor]:
    """Generate synthetic AIS trajectories with realistic temporal continuity. See data/README.md for details."""
    g = torch.Generator()
    g.manual_seed(int(seed))

    trajectories: list[torch.Tensor] = []
    for ship_idx in range(n_ships):
        time = torch.arange(n_points_per_ship, dtype=torch.float32)
        time = time * 60.0 + 1000.0 * ship_idx

        start_lat = 30.0 + 20.0 * torch.rand(1, generator=g).item()
        start_lon = -20.0 + 40.0 * torch.rand(1, generator=g).item()

        drift_lat = (torch.rand(1, generator=g).item() - 0.5) * 0.02
        drift_lon = (torch.rand(1, generator=g).item() - 0.5) * 0.02

        wave = torch.sin(torch.linspace(0, 8.0 * math.pi, n_points_per_ship))
        lat_noise = 0.002 * torch.randn(n_points_per_ship, generator=g)
        lon_noise = 0.002 * torch.randn(n_points_per_ship, generator=g)

        lat = start_lat + drift_lat * torch.arange(n_points_per_ship) + 0.05 * wave + lat_noise
        lon = start_lon + drift_lon * torch.arange(n_points_per_ship) + 0.05 * torch.cos(wave) + lon_noise

        speed = 8.0 + 4.0 * torch.rand(n_points_per_ship, generator=g)
        heading = (torch.atan2(torch.diff(lat, prepend=lat[:1]), torch.diff(lon, prepend=lon[:1])) * 180.0 / math.pi) % 360.0

        is_start = torch.zeros(n_points_per_ship, dtype=torch.float32)
        is_end = torch.zeros(n_points_per_ship, dtype=torch.float32)
        is_start[0] = 1.0
        is_end[-1] = 1.0

        turn = torch.zeros(n_points_per_ship, dtype=torch.float32)
        if n_points_per_ship > 2:
            hd = torch.abs(heading[1:] - heading[:-1])
            hd = torch.minimum(hd, 360.0 - hd)
            turn[1:] = hd / 180.0

        traj = torch.stack([time, lat, lon, speed, heading, is_start, is_end, turn], dim=1)
        trajectories.append(traj)

    return trajectories
