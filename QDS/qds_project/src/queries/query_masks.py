"""Shared spatiotemporal point-query masking helpers."""

from __future__ import annotations

from torch import Tensor

# Point schema: [time, lat, lon, speed, ...]
TIME_COL = 0
LAT_COL = 1
LON_COL = 2
SPEED_COL = 3

# Query schema: [lat_min, lat_max, lon_min, lon_max, time_start, time_end]
Q_LAT_MIN_COL = 0
Q_LAT_MAX_COL = 1
Q_LON_MIN_COL = 2
Q_LON_MAX_COL = 3
Q_TIME_START_COL = 4
Q_TIME_END_COL = 5


def spatial_inclusion_mask(points: Tensor, queries: Tensor) -> Tensor:
    """Return [N, M] mask for points inside each query's spatial bounds."""
    return (
        (points[:, None, LAT_COL] >= queries[None, :, Q_LAT_MIN_COL])
        & (points[:, None, LAT_COL] <= queries[None, :, Q_LAT_MAX_COL])
        & (points[:, None, LON_COL] >= queries[None, :, Q_LON_MIN_COL])
        & (points[:, None, LON_COL] <= queries[None, :, Q_LON_MAX_COL])
    )


def spatiotemporal_inclusion_mask(
    points: Tensor,
    queries: Tensor,
    spatial_mask: Tensor | None = None,
) -> Tensor:
    """Return [N, M] mask for points inside each query's full spatiotemporal bounds."""
    base_spatial_mask = (
        spatial_mask
        if spatial_mask is not None
        else spatial_inclusion_mask(points, queries)
    )
    return base_spatial_mask & (
        (points[:, None, TIME_COL] >= queries[None, :, Q_TIME_START_COL])
        & (points[:, None, TIME_COL] <= queries[None, :, Q_TIME_END_COL])
    )


def sum_values_by_query(values: Tensor, inclusion_mask: Tensor) -> Tensor:
    """Aggregate query results by summing a per-point value vector for each query."""
    if values.dim() != 1:
        raise ValueError(f"values must be 1-D, got shape={tuple(values.shape)}")
    return (inclusion_mask.float() * values.unsqueeze(1)).sum(dim=0)


def sum_speed_by_query(
    points: Tensor,
    inclusion_mask: Tensor,
    *,
    absolute: bool = False,
) -> Tensor:
    """Aggregate query results by summing point speed for each query column."""
    speed = points[:, SPEED_COL]
    if absolute:
        speed = speed.abs()
    return sum_values_by_query(speed, inclusion_mask)
