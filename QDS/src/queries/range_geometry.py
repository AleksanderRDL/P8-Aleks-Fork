"""Shared geometry helpers for range-query space-time boxes."""

from __future__ import annotations

from typing import Mapping

import torch


def _range_box_bounds(params: Mapping[str, float], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return inclusive ``[time, lat, lon]`` lower and upper bounds."""
    lows = torch.tensor(
        [float(params["t_start"]), float(params["lat_min"]), float(params["lon_min"])],
        dtype=torch.float32,
        device=device,
    )
    highs = torch.tensor(
        [float(params["t_end"]), float(params["lat_max"]), float(params["lon_max"])],
        dtype=torch.float32,
        device=device,
    )
    return lows, highs


def points_in_range_box(points: torch.Tensor, params: Mapping[str, float]) -> torch.Tensor:
    """Return points inside the inclusive ``[time, lat, lon]`` range box."""
    if points.shape[0] == 0:
        return torch.empty((0,), dtype=torch.bool, device=points.device)
    xyz = points[:, [0, 1, 2]].to(dtype=torch.float32)
    lows, highs = _range_box_bounds(params, points.device)
    return ((xyz >= lows) & (xyz <= highs)).all(dim=1)


def segment_box_crossings(points: torch.Tensor, params: Mapping[str, float]) -> torch.Tensor:
    """Return true for segments crossing or passing through a range box.

    Fully inside segments are not crossing support; they are already covered by
    in-box point metrics. Outside-to-inside, inside-to-outside, and
    outside-to-outside pass-through segments are crossing support because their
    retained endpoint pair preserves continuous boundary context.
    """
    if points.shape[0] < 2:
        return torch.empty((0,), dtype=torch.bool, device=points.device)

    start_xyz = points[:-1, [0, 1, 2]].to(dtype=torch.float32)
    end_xyz = points[1:, [0, 1, 2]].to(dtype=torch.float32)
    delta = end_xyz - start_xyz
    lows, highs = _range_box_bounds(params, points.device)

    start_inside = ((start_xyz >= lows) & (start_xyz <= highs)).all(dim=1)
    end_inside = ((end_xyz >= lows) & (end_xyz <= highs)).all(dim=1)

    u_min = torch.zeros((start_xyz.shape[0],), dtype=torch.float32, device=points.device)
    u_max = torch.ones((start_xyz.shape[0],), dtype=torch.float32, device=points.device)
    valid = torch.ones((start_xyz.shape[0],), dtype=torch.bool, device=points.device)
    eps = 1e-12
    for dim in range(3):
        dim_delta = delta[:, dim]
        dim_start = start_xyz[:, dim]
        parallel = torch.abs(dim_delta) <= eps
        valid &= (~parallel) | ((dim_start >= lows[dim]) & (dim_start <= highs[dim]))

        non_parallel = ~parallel
        if bool(non_parallel.any().item()):
            u1 = (lows[dim] - dim_start[non_parallel]) / dim_delta[non_parallel]
            u2 = (highs[dim] - dim_start[non_parallel]) / dim_delta[non_parallel]
            u_low = torch.minimum(u1, u2)
            u_high = torch.maximum(u1, u2)
            u_min[non_parallel] = torch.maximum(u_min[non_parallel], u_low)
            u_max[non_parallel] = torch.minimum(u_max[non_parallel], u_high)

    intersects = valid & (u_max >= u_min) & (u_max >= 0.0) & (u_min <= 1.0)
    return intersects & ~(start_inside & end_inside)


def segment_box_bracket_mask(
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    params: Mapping[str, float],
) -> torch.Tensor:
    """Return point mask for endpoint pairs bracketing range-box crossings."""
    bracket_full = torch.zeros((points.shape[0],), dtype=torch.bool, device=points.device)
    for start, end in boundaries:
        if end - start < 2:
            continue
        crossing_offsets = torch.where(segment_box_crossings(points[start:end], params))[0]
        if crossing_offsets.numel() == 0:
            continue
        bracket_full[start + crossing_offsets] = True
        bracket_full[start + crossing_offsets + 1] = True
    return bracket_full


def segment_box_bracket_indices(
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    params: Mapping[str, float],
) -> torch.Tensor:
    """Return sorted point indices bracketing range-box crossings."""
    return torch.where(segment_box_bracket_mask(points, boundaries, params))[0].to(dtype=torch.long)
