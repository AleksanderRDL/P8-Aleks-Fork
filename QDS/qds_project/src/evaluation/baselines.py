"""
baselines.py

Baseline trajectory simplification strategies for comparison with
the TrajectoryQDSModel.

Baselines
---------
1. random_sampling           — retain a uniformly random subset of points.
2. uniform_temporal_sampling — retain every k-th point sorted by time.
3. douglas_peucker           — recursive 2D line simplification on lat/lon.
"""

from __future__ import annotations

import torch
from torch import Tensor


def random_sampling(points: Tensor, ratio: float) -> Tensor:
    """Retain a uniformly random subset of the trajectory point cloud.

    Args:
        points: Tensor of shape [N, 5] with columns
                [time, lat, lon, speed, heading].
        ratio:  Fraction of points to retain, in (0, 1].

    Returns:
        Tensor of shape [K, 5] where K = max(1, round(ratio * N)).
    """
    n = points.shape[0]
    k = max(1, int(round(ratio * n)))
    indices = torch.randperm(n)[:k]
    return points[indices]


def uniform_temporal_sampling(points: Tensor, ratio: float) -> Tensor:
    """Sample uniformly in time by retaining every k-th point.

    Points are first sorted by the time column, then every k-th point is
    kept so that the retained set is spread evenly across the time axis.

    Args:
        points: Tensor of shape [N, 5] with columns
                [time, lat, lon, speed, heading].
        ratio:  Fraction of points to retain, in (0, 1].

    Returns:
        Tensor of shape [K, 5].
    """
    n = points.shape[0]
    k = max(1, int(round(ratio * n)))

    # Sort by time column (index 0)
    sorted_indices = torch.argsort(points[:, 0])
    sorted_points = points[sorted_indices]

    # Pick every step-th point to get approximately k points
    step = max(1, n // k)
    sampled_indices = torch.arange(0, n, step)[:k]
    return sorted_points[sampled_indices]


def _perpendicular_distance(point: Tensor, line_start: Tensor, line_end: Tensor) -> float:
    """Compute the perpendicular distance from a 2D point to a line segment.

    Args:
        point:      2D tensor [lat, lon].
        line_start: 2D tensor [lat, lon] — start of the segment.
        line_end:   2D tensor [lat, lon] — end of the segment.

    Returns:
        Perpendicular distance as a float.
    """
    if torch.allclose(line_start, line_end):
        return float(torch.norm(point - line_start))

    d = line_end - line_start
    # |cross product| / |d|
    cross = d[0] * (line_start[1] - point[1]) - (line_start[0] - point[0]) * d[1]
    return float(torch.abs(cross) / torch.norm(d))


def _douglas_peucker_indices(
    points_2d: Tensor,
    indices: list[int],
    epsilon: float,
) -> list[int]:
    """Recursive Douglas-Peucker algorithm returning indices to keep.

    Args:
        points_2d: Tensor of shape [N, 2] with [lat, lon].
        indices:   List of point indices in the current segment.
        epsilon:   Maximum allowable perpendicular distance.

    Returns:
        Sorted list of indices to retain.
    """
    if len(indices) <= 2:
        return indices

    start_idx = indices[0]
    end_idx   = indices[-1]
    start_pt  = points_2d[start_idx]
    end_pt    = points_2d[end_idx]

    # Find the point with the maximum perpendicular distance
    max_dist  = 0.0
    max_index = 0
    for i in indices[1:-1]:
        dist = _perpendicular_distance(points_2d[i], start_pt, end_pt)
        if dist > max_dist:
            max_dist  = dist
            max_index = i

    if max_dist > epsilon:
        # Recurse on both sub-segments
        left_idx  = indices[: indices.index(max_index) + 1]
        right_idx = indices[indices.index(max_index):]
        left_result  = _douglas_peucker_indices(points_2d, left_idx,  epsilon)
        right_result = _douglas_peucker_indices(points_2d, right_idx, epsilon)
        # Merge (avoid duplicating the split point)
        return left_result[:-1] + right_result
    else:
        # All interior points are within epsilon — keep only endpoints
        return [start_idx, end_idx]


def douglas_peucker(points: Tensor, epsilon: float = 0.01) -> Tensor:
    """Simplify a trajectory using the Douglas-Peucker algorithm on lat/lon.

    Recursively removes points that deviate less than ``epsilon`` degrees
    from the straight-line path in 2D lat/lon space.

    Args:
        points:  Tensor of shape [N, 5] with columns
                 [time, lat, lon, speed, heading].
        epsilon: Maximum allowable perpendicular distance in degrees.
                 Smaller values preserve more detail.

    Returns:
        Tensor of shape [K, 5] — the subset of input points that were
        retained by the algorithm.
    """
    n = points.shape[0]
    if n <= 2:
        return points

    # Work on the lat/lon columns only (indices 1 and 2)
    points_2d = points[:, 1:3]  # [N, 2]

    kept_indices = _douglas_peucker_indices(points_2d, list(range(n)), epsilon)
    kept_indices = sorted(set(kept_indices))
    return points[kept_indices]
