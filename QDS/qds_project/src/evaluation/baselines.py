"""Baseline simplification methods for comparison. See src/evaluation/README.md."""

from __future__ import annotations

import torch
from torch import Tensor


def random_sampling(points: Tensor, ratio: float) -> Tensor:
    """Retain a uniformly random subset of the trajectory point cloud."""
    n = points.shape[0]
    k = max(1, int(round(ratio * n)))
    indices = torch.randperm(n, device=points.device)[:k]
    return points[indices]


def uniform_temporal_sampling(points: Tensor, ratio: float) -> Tensor:
    """Sample uniformly in time by retaining every k-th point."""
    n = points.shape[0]
    k = max(1, int(round(ratio * n)))

    # Sort by time column (index 0)
    sorted_indices = torch.argsort(points[:, 0])
    sorted_points = points[sorted_indices]

    # Pick every step-th point to get approximately k points
    step = max(1, n // k)
    sampled_indices = torch.arange(0, n, step, device=points.device)[:k]
    return sorted_points[sampled_indices]


def _perpendicular_distance(point: Tensor, line_start: Tensor, line_end: Tensor) -> float:
    """Perpendicular distance from a 2D point to a line segment."""
    if torch.allclose(line_start, line_end):
        return float(torch.norm(point - line_start))

    d = line_end - line_start
    # |cross product| / |d|
    cross = d[0] * (line_start[1] - point[1]) - (line_start[0] - point[0]) * d[1]
    return float(torch.abs(cross) / torch.norm(d))


def _max_distance_in_segment(points_2d: Tensor, start_idx: int, end_idx: int) -> tuple[int, float]:
    """Return the interior point index with max perpendicular distance to the segment."""
    if end_idx - start_idx <= 1:
        return -1, 0.0

    start_pt = points_2d[start_idx]
    end_pt = points_2d[end_idx]
    interior = points_2d[start_idx + 1 : end_idx]
    if interior.numel() == 0:
        return -1, 0.0

    segment = end_pt - start_pt
    seg_norm = torch.norm(segment)

    if float(seg_norm.item()) == 0.0:
        distances = torch.norm(interior - start_pt, dim=1)
    else:
        rel = interior - start_pt
        cross = segment[0] * rel[:, 1] - rel[:, 0] * segment[1]
        distances = torch.abs(cross) / seg_norm

    max_dist, local_idx = torch.max(distances, dim=0)
    split_idx = start_idx + 1 + int(local_idx.item())
    return split_idx, float(max_dist.item())


def _douglas_peucker_indices(
    points_2d: Tensor,
    indices: list[int],
    epsilon: float,
) -> list[int]:
    """Douglas-Peucker algorithm; returns sorted list of indices to keep."""
    if len(indices) <= 2:
        return indices

    index_tensor = torch.as_tensor(indices, dtype=torch.long, device=points_2d.device)
    keep = torch.zeros(index_tensor.shape[0], dtype=torch.bool, device=points_2d.device)
    keep[0] = True
    keep[-1] = True

    # Work with positions in `indices` to support arbitrary index subsets.
    stack: list[tuple[int, int]] = [(0, len(indices) - 1)]

    while stack:
        left_pos, right_pos = stack.pop()
        if right_pos - left_pos <= 1:
            continue

        start_idx = int(index_tensor[left_pos].item())
        end_idx = int(index_tensor[right_pos].item())

        interior_idx = index_tensor[left_pos + 1 : right_pos]
        if interior_idx.numel() == 0:
            continue

        start_pt = points_2d[start_idx]
        end_pt = points_2d[end_idx]
        interior = points_2d[interior_idx]

        segment = end_pt - start_pt
        seg_norm = torch.norm(segment)
        if float(seg_norm.item()) == 0.0:
            distances = torch.norm(interior - start_pt, dim=1)
        else:
            rel = interior - start_pt
            cross = segment[0] * rel[:, 1] - rel[:, 0] * segment[1]
            distances = torch.abs(cross) / seg_norm

        max_dist, local_idx = torch.max(distances, dim=0)
        if float(max_dist.item()) > epsilon:
            split_pos = left_pos + 1 + int(local_idx.item())
            keep[split_pos] = True
            stack.append((left_pos, split_pos))
            stack.append((split_pos, right_pos))

    return index_tensor[keep].tolist()


def douglas_peucker(points: Tensor, epsilon: float = 0.01) -> Tensor:
    """Simplify a trajectory using Douglas-Peucker on lat/lon coordinates."""
    n = points.shape[0]
    if n <= 2:
        return points

    # Work on the lat/lon columns only (indices 1 and 2)
    points_2d = points[:, 1:3]  # [N, 2]

    keep = torch.zeros(n, dtype=torch.bool, device=points.device)
    keep[0] = True
    keep[-1] = True

    # Iterative stack avoids Python recursion overhead on large inputs.
    stack: list[tuple[int, int]] = [(0, n - 1)]
    while stack:
        start_idx, end_idx = stack.pop()
        split_idx, max_dist = _max_distance_in_segment(points_2d, start_idx, end_idx)

        if split_idx >= 0 and max_dist > epsilon:
            keep[split_idx] = True
            stack.append((start_idx, split_idx))
            stack.append((split_idx, end_idx))

    kept_indices = torch.nonzero(keep, as_tuple=False).flatten()
    return points.index_select(0, kept_indices)
