"""Spatiotemporal query generation. See src/queries/README.md for strategies."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from src.queries.query_masks import LAT_COL, LON_COL, TIME_COL


@dataclass(frozen=True)
class _QueryBounds:
    """Sampling and clamping bounds used to build query tensors."""

    time_min: Tensor
    time_max: Tensor
    lat_min: Tensor
    lat_max: Tensor
    lon_min: Tensor
    lon_max: Tensor
    time_range: Tensor
    lat_width_range: Tensor
    lon_width_range: Tensor


def _concat_trajectories(trajectories: list[Tensor]) -> Tensor:
    """Flatten trajectory list into one point cloud tensor."""
    if len(trajectories) == 0:
        raise ValueError("trajectories must contain at least one tensor.")
    return torch.cat(trajectories, dim=0)


def _guard_range(r: Tensor) -> Tensor:
    """Return *r* if positive, else a safe fallback span of 1.0."""
    return torch.where(r > 1e-6, r, torch.ones_like(r))


def _rand_like_scalar(size: int, ref: Tensor) -> Tensor:
    """Sample random tensor on the same dtype/device as a scalar reference."""
    return torch.rand(size, dtype=ref.dtype, device=ref.device)


def _effective_spatial_ranges(
    all_points: Tensor,
    lat_range: Tensor,
    lon_range: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return robust spatial ranges for width sampling using 5th–95th percentiles."""
    q = torch.tensor([0.05, 0.95], dtype=all_points.dtype, device=all_points.device)
    lat_q = torch.quantile(all_points[:, LAT_COL], q)
    lon_q = torch.quantile(all_points[:, LON_COL], q)

    robust_lat_range = _guard_range(lat_q[1] - lat_q[0])
    robust_lon_range = _guard_range(lon_q[1] - lon_q[0])

    eff_lat_range = _guard_range(torch.minimum(lat_range, robust_lat_range * 1.5))
    eff_lon_range = _guard_range(torch.minimum(lon_range, robust_lon_range * 1.5))
    return eff_lat_range, eff_lon_range


def _effective_spatial_bounds(
    all_points: Tensor,
    *,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return robust spatial center bounds via quantile clipping."""
    if not (0.0 <= lower_q < upper_q <= 1.0):
        raise ValueError(
            f"Invalid quantile bounds: lower_q={lower_q}, upper_q={upper_q}."
        )

    q = torch.tensor([lower_q, upper_q], dtype=all_points.dtype, device=all_points.device)
    lat_q = torch.quantile(all_points[:, LAT_COL], q)
    lon_q = torch.quantile(all_points[:, LON_COL], q)

    lat_lo, lat_hi = lat_q[0], lat_q[1]
    lon_lo, lon_hi = lon_q[0], lon_q[1]

    if bool(lat_hi - lat_lo <= 1e-6):
        lat_lo = all_points[:, LAT_COL].min()
        lat_hi = all_points[:, LAT_COL].max()
    if bool(lon_hi - lon_lo <= 1e-6):
        lon_lo = all_points[:, LON_COL].min()
        lon_hi = all_points[:, LON_COL].max()

    return lat_lo, lat_hi, lon_lo, lon_hi


def _prepare_query_bounds(
    all_points: Tensor,
    *,
    use_robust_spatial_bounds: bool,
    spatial_bound_lower_quantile: float = 0.01,
    spatial_bound_upper_quantile: float = 0.99,
) -> _QueryBounds:
    """Build query bounds/context from the point cloud."""
    time_min = all_points[:, TIME_COL].min()
    time_max = all_points[:, TIME_COL].max()

    if use_robust_spatial_bounds:
        lat_min, lat_max, lon_min, lon_max = _effective_spatial_bounds(
            all_points,
            lower_q=spatial_bound_lower_quantile,
            upper_q=spatial_bound_upper_quantile,
        )
    else:
        lat_min = all_points[:, LAT_COL].min()
        lat_max = all_points[:, LAT_COL].max()
        lon_min = all_points[:, LON_COL].min()
        lon_max = all_points[:, LON_COL].max()

    lat_range = _guard_range(lat_max - lat_min)
    lon_range = _guard_range(lon_max - lon_min)
    time_range = _guard_range(time_max - time_min)
    lat_width_range, lon_width_range = _effective_spatial_ranges(
        all_points,
        lat_range,
        lon_range,
    )
    return _QueryBounds(
        time_min=time_min,
        time_max=time_max,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        time_range=time_range,
        lat_width_range=lat_width_range,
        lon_width_range=lon_width_range,
    )


def _ensure_strict_order_within_bounds(
    q_min: Tensor,
    q_max: Tensor,
    bound_min: Tensor,
    bound_max: Tensor,
    eps: float = 1e-4,
) -> tuple[Tensor, Tensor]:
    """Ensure q_min < q_max while keeping both values inside bounds."""
    eps_t = torch.as_tensor(eps, dtype=q_min.dtype, device=q_min.device)
    span = bound_max - bound_min

    if bool(span <= eps_t):
        return bound_min.expand_as(q_min), bound_max.expand_as(q_max)

    too_small = q_max <= q_min
    if too_small.any():
        upper_room = q_min + eps_t <= bound_max
        lower_room = q_max - eps_t >= bound_min

        new_max = torch.where(upper_room, q_min + eps_t, q_max)
        new_min = torch.where(~upper_room & lower_room, q_max - eps_t, q_min)

        q_min = torch.where(too_small, new_min, q_min)
        q_max = torch.where(too_small, new_max, q_max)

    q_min = torch.clamp(q_min, min=bound_min, max=bound_max)
    q_max = torch.clamp(q_max, min=bound_min, max=bound_max)
    return q_min, q_max


def _sample_center_uniform(min_value: Tensor, max_value: Tensor, n_queries: int) -> Tensor:
    """Sample uniformly in [min_value, max_value] with matching dtype/device."""
    return min_value + _rand_like_scalar(n_queries, min_value) * _guard_range(max_value - min_value)


def _build_queries(
    centers_lat: Tensor,
    centers_lon: Tensor,
    centers_time: Tensor,
    bounds: _QueryBounds,
    spatial_fraction: float,
    temporal_fraction: float,
) -> Tensor:
    """Sample query widths, clamp to bounds, and assemble [M, 6] query tensor."""
    n_queries = centers_lat.shape[0]

    if spatial_fraction <= 0.0:
        raise ValueError(f"spatial_fraction must be > 0, got {spatial_fraction}.")
    if temporal_fraction <= 0.0:
        raise ValueError(f"temporal_fraction must be > 0, got {temporal_fraction}.")

    min_spatial_fraction = min(0.0025, spatial_fraction)
    min_temporal_fraction = min(0.01, temporal_fraction)

    spatial_f = (
        _rand_like_scalar(n_queries, centers_lat)
        * (spatial_fraction - min_spatial_fraction)
        + min_spatial_fraction
    )
    temporal_f = (
        _rand_like_scalar(n_queries, centers_time)
        * (temporal_fraction - min_temporal_fraction)
        + min_temporal_fraction
    )

    hw_lat = spatial_f * bounds.lat_width_range / 2.0
    hw_lon = spatial_f * bounds.lon_width_range / 2.0
    hw_time = temporal_f * bounds.time_range / 2.0

    q_lat_min = torch.clamp(centers_lat - hw_lat, min=bounds.lat_min, max=bounds.lat_max)
    q_lat_max = torch.clamp(centers_lat + hw_lat, min=bounds.lat_min, max=bounds.lat_max)
    q_lon_min = torch.clamp(centers_lon - hw_lon, min=bounds.lon_min, max=bounds.lon_max)
    q_lon_max = torch.clamp(centers_lon + hw_lon, min=bounds.lon_min, max=bounds.lon_max)
    q_time_min = torch.clamp(centers_time - hw_time, min=bounds.time_min, max=bounds.time_max)
    q_time_max = torch.clamp(centers_time + hw_time, min=bounds.time_min, max=bounds.time_max)

    q_lat_min, q_lat_max = _ensure_strict_order_within_bounds(
        q_lat_min,
        q_lat_max,
        bounds.lat_min,
        bounds.lat_max,
    )
    q_lon_min, q_lon_max = _ensure_strict_order_within_bounds(
        q_lon_min,
        q_lon_max,
        bounds.lon_min,
        bounds.lon_max,
    )
    q_time_min, q_time_max = _ensure_strict_order_within_bounds(
        q_time_min,
        q_time_max,
        bounds.time_min,
        bounds.time_max,
    )

    return torch.stack(
        [q_lat_min, q_lat_max, q_lon_min, q_lon_max, q_time_min, q_time_max],
        dim=1,
    )


def generate_uniform_queries(
    trajectories: list[Tensor],
    n_queries: int = 100,
    spatial_fraction: float = 0.03,
    temporal_fraction: float = 0.10,
    spatial_bound_lower_quantile: float = 0.01,
    spatial_bound_upper_quantile: float = 0.99,
) -> Tensor:
    """Generate queries with centers sampled uniformly from robust spatial bounds."""
    all_points = _concat_trajectories(trajectories)
    bounds = _prepare_query_bounds(
        all_points,
        use_robust_spatial_bounds=True,
        spatial_bound_lower_quantile=spatial_bound_lower_quantile,
        spatial_bound_upper_quantile=spatial_bound_upper_quantile,
    )

    centers_lat = _sample_center_uniform(bounds.lat_min, bounds.lat_max, n_queries)
    centers_lon = _sample_center_uniform(bounds.lon_min, bounds.lon_max, n_queries)
    centers_time = _sample_center_uniform(bounds.time_min, bounds.time_max, n_queries)

    return _build_queries(
        centers_lat,
        centers_lon,
        centers_time,
        bounds,
        spatial_fraction,
        temporal_fraction,
    )


def generate_density_biased_queries(
    trajectories: list[Tensor],
    n_queries: int = 100,
    spatial_fraction: float = 0.03,
    temporal_fraction: float = 0.10,
) -> Tensor:
    """Generate queries with spatial centers anchored to real AIS points."""
    all_points = _concat_trajectories(trajectories)
    bounds = _prepare_query_bounds(
        all_points,
        use_robust_spatial_bounds=False,
    )

    anchor_idx = torch.randint(
        0,
        all_points.shape[0],
        (n_queries,),
        device=all_points.device,
    )
    anchors = all_points[anchor_idx]
    centers_lat = anchors[:, LAT_COL]
    centers_lon = anchors[:, LON_COL]
    centers_time = _sample_center_uniform(bounds.time_min, bounds.time_max, n_queries)

    return _build_queries(
        centers_lat,
        centers_lon,
        centers_time,
        bounds,
        spatial_fraction,
        temporal_fraction,
    )


def generate_mixed_queries(
    trajectories: list[Tensor],
    total_queries: int = 100,
    density_ratio: float = 0.5,
    spatial_fraction: float = 0.03,
    temporal_fraction: float = 0.10,
    spatial_bound_lower_quantile: float = 0.01,
    spatial_bound_upper_quantile: float = 0.99,
) -> Tensor:
    """Generate a shuffled blend of uniform and density-biased queries."""
    if not (0.0 <= density_ratio <= 1.0):
        raise ValueError(f"density_ratio must be in [0, 1], got {density_ratio}.")

    n_density = int(total_queries * density_ratio)
    n_uniform = total_queries - n_density
    parts: list[Tensor] = []

    if n_density > 0:
        parts.append(
            generate_density_biased_queries(
                trajectories,
                n_queries=n_density,
                spatial_fraction=spatial_fraction,
                temporal_fraction=temporal_fraction,
            )
        )
    if n_uniform > 0:
        parts.append(
            generate_uniform_queries(
                trajectories,
                n_queries=n_uniform,
                spatial_fraction=spatial_fraction,
                temporal_fraction=temporal_fraction,
                spatial_bound_lower_quantile=spatial_bound_lower_quantile,
                spatial_bound_upper_quantile=spatial_bound_upper_quantile,
            )
        )

    if not parts:
        all_points = _concat_trajectories(trajectories)
        return torch.empty((0, 6), dtype=all_points.dtype, device=all_points.device)

    combined = torch.cat(parts, dim=0)
    return combined[torch.randperm(combined.shape[0], device=combined.device)]


def generate_spatiotemporal_queries(
    trajectories: list[Tensor],
    n_queries: int = 100,
    spatial_fraction: float = 0.03,
    temporal_fraction: float = 0.10,
    anchor_to_data: bool = True,
) -> Tensor:
    """Backward-compatible wrapper for density-biased or uniform workloads."""
    if anchor_to_data:
        return generate_density_biased_queries(
            trajectories,
            n_queries=n_queries,
            spatial_fraction=spatial_fraction,
            temporal_fraction=temporal_fraction,
        )
    return generate_uniform_queries(
        trajectories,
        n_queries=n_queries,
        spatial_fraction=spatial_fraction,
        temporal_fraction=temporal_fraction,
    )
