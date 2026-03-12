"""
query_generator.py

Generates random spatiotemporal range queries derived from actual trajectory
data bounds.

Each query is a 6-tuple:
    [lat_min, lat_max, lon_min, lon_max, time_start, time_end]

Query extents are sampled as fractions of the total data range so that
queries cover a realistic portion of the dataset.

Three query generation strategies are provided:

* ``generate_uniform_queries`` — centres sampled uniformly from the
  bounding box, simulating random ad-hoc queries.
* ``generate_density_biased_queries`` — centres anchored to real AIS
  data points, concentrating queries where vessel traffic is dense and
  simulating realistic maritime workloads (ports, straits, shipping lanes).
* ``generate_mixed_queries`` — a weighted blend of the two strategies.
"""

from __future__ import annotations

from typing import List

import torch
from torch import Tensor


def _compute_bounds(
    all_points: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return (time_min, time_max, lat_min, lat_max, lon_min, lon_max) from a
    flat point cloud tensor of shape [N, 5]."""
    time_min = all_points[:, 0].min()
    time_max = all_points[:, 0].max()
    lat_min  = all_points[:, 1].min()
    lat_max  = all_points[:, 1].max()
    lon_min  = all_points[:, 2].min()
    lon_max  = all_points[:, 2].max()
    return time_min, time_max, lat_min, lat_max, lon_min, lon_max


def _guard_range(r: Tensor) -> Tensor:
    """Return *r* unchanged if it is meaningfully positive, else 1.0."""
    return r if r > 1e-6 else torch.tensor(1.0)


def _effective_spatial_ranges(
    all_points: Tensor,
    lat_range: Tensor,
    lon_range: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return robust spatial ranges for query width sampling.

    Uses a clipped spread estimate based on the 5th-95th percentiles to
    prevent extremely wide queries when the global bounding box is dominated
    by outliers or sparse extremes.
    """
    q = torch.tensor([0.05, 0.95], dtype=all_points.dtype, device=all_points.device)
    lat_q = torch.quantile(all_points[:, 1], q)
    lon_q = torch.quantile(all_points[:, 2], q)

    robust_lat_range = _guard_range(lat_q[1] - lat_q[0])
    robust_lon_range = _guard_range(lon_q[1] - lon_q[0])

    scale = torch.tensor(1.5, dtype=all_points.dtype, device=all_points.device)
    eff_lat_range = _guard_range(torch.minimum(lat_range, robust_lat_range * scale))
    eff_lon_range = _guard_range(torch.minimum(lon_range, robust_lon_range * scale))

    return eff_lat_range, eff_lon_range


def _effective_spatial_bounds(
    all_points: Tensor,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return robust spatial bounds for center sampling/clamping.

    Bounds are based on quantiles to reduce the influence of a small number
    of extreme outliers in lat/lon coordinates.
    """
    if not (0.0 <= lower_q < upper_q <= 1.0):
        raise ValueError(
            f"Invalid quantile bounds: lower_q={lower_q}, upper_q={upper_q}."
        )

    q = torch.tensor([lower_q, upper_q], dtype=all_points.dtype, device=all_points.device)
    lat_q = torch.quantile(all_points[:, 1], q)
    lon_q = torch.quantile(all_points[:, 2], q)

    lat_lo = lat_q[0]
    lat_hi = lat_q[1]
    lon_lo = lon_q[0]
    lon_hi = lon_q[1]

    # Fallback to strict min/max if quantile span is degenerate.
    if bool(lat_hi - lat_lo <= 1e-6):
        lat_lo = all_points[:, 1].min()
        lat_hi = all_points[:, 1].max()
    if bool(lon_hi - lon_lo <= 1e-6):
        lon_lo = all_points[:, 2].min()
        lon_hi = all_points[:, 2].max()

    return lat_lo, lat_hi, lon_lo, lon_hi


def _ensure_strict_order_within_bounds(
    q_min: Tensor,
    q_max: Tensor,
    bound_min: Tensor,
    bound_max: Tensor,
    eps: float = 1e-4,
) -> tuple[Tensor, Tensor]:
    """Ensure q_min < q_max while keeping both values inside bounds."""
    eps_t = torch.tensor(eps, dtype=q_min.dtype, device=q_min.device)
    span = bound_max - bound_min

    # If the global span is tiny, best effort is to stay inside bounds.
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


def _build_queries(
    centers_lat: Tensor,
    centers_lon: Tensor,
    centers_time: Tensor,
    lat_width_range: Tensor,
    lon_width_range: Tensor,
    time_range: Tensor,
    lat_min: Tensor,
    lat_max: Tensor,
    lon_min: Tensor,
    lon_max: Tensor,
    time_min: Tensor,
    time_max: Tensor,
    spatial_fraction: float,
    temporal_fraction: float,
) -> Tensor:
    """Sample half-widths, clamp to data bounds, and assemble query tensor.

    Args:
        centers_lat / centers_lon / centers_time: Query centre coordinates,
            each of shape [M].
        lat_width_range / lon_width_range / time_range: Extents used to scale
            half-widths.
        lat_min / lat_max / lon_min / lon_max / time_min / time_max: Data
            bounds used for clamping.
        spatial_fraction:  Maximum query width as a fraction of the spatial
            range.
        temporal_fraction: Maximum query width as a fraction of the time
            range.

    Returns:
        Tensor of shape [M, 6] with columns
        [lat_min, lat_max, lon_min, lon_max, time_start, time_end].
    """
    n_queries = centers_lat.shape[0]

    if spatial_fraction <= 0.0:
        raise ValueError(f"spatial_fraction must be > 0, got {spatial_fraction}.")
    if temporal_fraction <= 0.0:
        raise ValueError(f"temporal_fraction must be > 0, got {temporal_fraction}.")

    # Keep a small lower bound so boxes do not collapse to near-zero size.
    min_spatial_fraction = min(0.0025, spatial_fraction)
    min_temporal_fraction = min(0.01, temporal_fraction)

    spatial_f = (
        torch.rand(n_queries) * (spatial_fraction - min_spatial_fraction)
        + min_spatial_fraction
    )
    temporal_f = (
        torch.rand(n_queries) * (temporal_fraction - min_temporal_fraction)
        + min_temporal_fraction
    )

    # Sample half-widths (always positive, proportional to effective ranges)
    hw_lat = spatial_f * lat_width_range / 2.0
    hw_lon = spatial_f * lon_width_range / 2.0
    hw_time = temporal_f * time_range / 2.0

    # Build bounds and clamp to data range
    q_lat_min  = torch.clamp(centers_lat  - hw_lat,  min=lat_min,  max=lat_max)
    q_lat_max  = torch.clamp(centers_lat  + hw_lat,  min=lat_min,  max=lat_max)
    q_lon_min  = torch.clamp(centers_lon  - hw_lon,  min=lon_min,  max=lon_max)
    q_lon_max  = torch.clamp(centers_lon  + hw_lon,  min=lon_min,  max=lon_max)
    q_time_min = torch.clamp(centers_time - hw_time, min=time_min, max=time_max)
    q_time_max = torch.clamp(centers_time + hw_time, min=time_min, max=time_max)

    # Ensure min < max (may become equal after clamping at boundary), while
    # staying strictly inside dataset bounds.
    q_lat_min, q_lat_max = _ensure_strict_order_within_bounds(
        q_lat_min,
        q_lat_max,
        lat_min,
        lat_max,
    )
    q_lon_min, q_lon_max = _ensure_strict_order_within_bounds(
        q_lon_min,
        q_lon_max,
        lon_min,
        lon_max,
    )
    q_time_min, q_time_max = _ensure_strict_order_within_bounds(
        q_time_min,
        q_time_max,
        time_min,
        time_max,
    )

    return torch.stack(
        [q_lat_min, q_lat_max, q_lon_min, q_lon_max, q_time_min, q_time_max],
        dim=1,
    )  # [M, 6]


def generate_uniform_queries(
    trajectories: List[Tensor],
    n_queries: int = 100,
    spatial_fraction: float = 0.03,
    temporal_fraction: float = 0.10,
    spatial_bound_lower_quantile: float = 0.01,
    spatial_bound_upper_quantile: float = 0.99,
) -> Tensor:
    """Generate spatiotemporal queries with centres sampled **uniformly** from
    the dataset bounding box.

    Query centres are drawn independently and uniformly from the full
    lat/lon/time extent of the dataset.  This models an unbiased, ad-hoc
    query workload where any location is equally likely to be queried.

    Algorithm:
        center_lat  ~ Uniform(lat_min,  lat_max)
        center_lon  ~ Uniform(lon_min,  lon_max)
        time_start  ~ Uniform(time_min, time_max)
        width_lat   ~ Uniform(w_min,    w_max) * lat_range
        width_lon   ~ Uniform(w_min,    w_max) * lon_range
        time_length ~ Uniform(t_min,    t_max) * time_range

    Args:
        trajectories:      List of trajectory tensors, each [T, 5] with
                           columns [time, lat, lon, speed, heading].
        n_queries:         Number of queries to generate.
        spatial_fraction:  Maximum query width as a fraction of the lat/lon
                           range.
        temporal_fraction: Maximum query width as a fraction of the time
                           range.
        spatial_bound_lower_quantile: Lower quantile used to define robust
                   spatial bounds for center placement and clamping.
        spatial_bound_upper_quantile: Upper quantile used to define robust
                   spatial bounds for center placement and clamping.

    Returns:
        Tensor of shape [n_queries, 6] with columns
        [lat_min, lat_max, lon_min, lon_max, time_start, time_end].
    """
    all_points = torch.cat(trajectories, dim=0)  # [N, 5]
    time_min, time_max, _, _, _, _ = _compute_bounds(all_points)

    lat_min, lat_max, lon_min, lon_max = _effective_spatial_bounds(
        all_points,
        lower_q=spatial_bound_lower_quantile,
        upper_q=spatial_bound_upper_quantile,
    )

    lat_range = _guard_range(lat_max - lat_min)
    lon_range = _guard_range(lon_max - lon_min)
    time_range = _guard_range(time_max - time_min)
    lat_width_range, lon_width_range = _effective_spatial_ranges(
        all_points,
        lat_range,
        lon_range,
    )

    # Uniform centre sampling from the bounding box
    centers_lat  = lat_min  + torch.rand(n_queries) * lat_range
    centers_lon  = lon_min  + torch.rand(n_queries) * lon_range
    centers_time = time_min + torch.rand(n_queries) * time_range

    return _build_queries(
        centers_lat, centers_lon, centers_time,
        lat_width_range, lon_width_range, time_range,
        lat_min, lat_max, lon_min, lon_max, time_min, time_max,
        spatial_fraction, temporal_fraction,
    )


def generate_density_biased_queries(
    trajectories: List[Tensor],
    n_queries: int = 100,
    spatial_fraction: float = 0.03,
    temporal_fraction: float = 0.10,
) -> Tensor:
    """Generate spatiotemporal queries **centred on real AIS data points**.

    Query centres are drawn by sampling uniformly from the actual AIS point
    cloud rather than from the bounding box.  Because the point cloud is
    denser in high-traffic areas (ports, straits, shipping lanes), this
    naturally concentrates queries where vessel activity is greatest —
    closely approximating a traffic-density sampling distribution and
    simulating realistic maritime workloads.

    Algorithm:
        point       = random AIS data point
        center_lat  = point.lat
        center_lon  = point.lon
        time_start  ~ Uniform(time_min, time_max)
        width_lat   ~ Uniform(w_min,    w_max) * lat_range
        width_lon   ~ Uniform(w_min,    w_max) * lon_range
        time_length ~ Uniform(t_min,    t_max) * time_range

    Args:
        trajectories:      List of trajectory tensors, each [T, 5] with
                           columns [time, lat, lon, speed, heading].
        n_queries:         Number of queries to generate.
        spatial_fraction:  Maximum query width as a fraction of the lat/lon
                           range.
        temporal_fraction: Maximum query width as a fraction of the time
                           range.

    Returns:
        Tensor of shape [n_queries, 6] with columns
        [lat_min, lat_max, lon_min, lon_max, time_start, time_end].
    """
    all_points = torch.cat(trajectories, dim=0)  # [N, 5]
    time_min, time_max, lat_min, lat_max, lon_min, lon_max = _compute_bounds(all_points)

    lat_range = _guard_range(lat_max - lat_min)
    lon_range = _guard_range(lon_max - lon_min)
    time_range = _guard_range(time_max - time_min)
    lat_width_range, lon_width_range = _effective_spatial_ranges(
        all_points,
        lat_range,
        lon_range,
    )

    # Density-biased centre sampling: anchor to real AIS data points so that
    # queries are concentrated in high-traffic regions.
    anchor_idx   = torch.randint(0, all_points.shape[0], (n_queries,))
    anchors      = all_points[anchor_idx]  # [M, 5]
    centers_lat  = anchors[:, 1]
    centers_lon  = anchors[:, 2]
    # Time window centre is still drawn uniformly to allow temporal variety.
    centers_time = time_min + torch.rand(n_queries) * time_range

    return _build_queries(
        centers_lat, centers_lon, centers_time,
        lat_width_range, lon_width_range, time_range,
        lat_min, lat_max, lon_min, lon_max, time_min, time_max,
        spatial_fraction, temporal_fraction,
    )


def generate_mixed_queries(
    trajectories: List[Tensor],
    total_queries: int = 100,
    density_ratio: float = 0.5,
    spatial_fraction: float = 0.03,
    temporal_fraction: float = 0.10,
    spatial_bound_lower_quantile: float = 0.01,
    spatial_bound_upper_quantile: float = 0.99,
) -> Tensor:
    """Generate a **mixed workload** combining uniform and density-biased queries.

    The resulting tensor is a shuffled blend of uniform queries (representing
    general exploratory traffic) and density-biased queries (representing
    focused queries on busy maritime areas).

    Args:
        trajectories:      List of trajectory tensors, each [T, 5] with
                           columns [time, lat, lon, speed, heading].
        total_queries:     Total number of queries to generate.
        density_ratio:     Fraction in [0, 1] of queries that are
                           density-biased.  The remainder are uniform.
                           E.g. 0.7 → 70 % density-biased, 30 % uniform.
        spatial_fraction:  Maximum query width as a fraction of the lat/lon
                           range.
        temporal_fraction: Maximum query width as a fraction of the time
                           range.
        spatial_bound_lower_quantile: Lower quantile used to define robust
                   spatial bounds for uniform component.
        spatial_bound_upper_quantile: Upper quantile used to define robust
                   spatial bounds for uniform component.

    Returns:
        Tensor of shape [total_queries, 6] with columns
        [lat_min, lat_max, lon_min, lon_max, time_start, time_end].
    """
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

    combined = torch.cat(parts, dim=0)  # [total_queries, 6]

    # Shuffle so density-biased and uniform queries are interleaved
    perm = torch.randperm(combined.shape[0])
    return combined[perm]


def generate_spatiotemporal_queries(
    trajectories: List[Tensor],
    n_queries: int = 100,
    spatial_fraction: float = 0.03,
    temporal_fraction: float = 0.10,
    anchor_to_data: bool = True,
) -> Tensor:
    """Generate random spatiotemporal range queries from trajectory bounds.

    .. deprecated::
        Prefer ``generate_uniform_queries`` or
        ``generate_density_biased_queries`` for new code.  This function is
        kept for backward compatibility and delegates to one of those two
        generators based on the ``anchor_to_data`` flag.

    Args:
        trajectories:      List of trajectory tensors, each [T, 5] with
                           columns [time, lat, lon, speed, heading].
        n_queries:         Number of queries to generate.
        spatial_fraction:  Typical query width as a fraction of the lat/lon range.
        temporal_fraction: Typical query width as a fraction of the time range.
        anchor_to_data:    If True, delegates to ``generate_density_biased_queries``
                           (centres on real AIS points).  If False, delegates to
                           ``generate_uniform_queries`` (centres drawn from bounding
                           box).

    Returns:
        Tensor of shape [M, 6] with columns
        [lat_min, lat_max, lon_min, lon_max, time_start, time_end].
    """
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
