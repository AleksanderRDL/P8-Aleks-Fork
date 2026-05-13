"""Shared helpers for flattened trajectory tensors and boundary indexes."""

from __future__ import annotations

from collections.abc import Iterable

import torch

TrajectoryBoundary = tuple[int, int]


def boundaries_from_trajectories(trajectories: list[torch.Tensor]) -> list[TrajectoryBoundary]:
    """Return flattened ``(start, end)`` boundaries for trajectory tensors."""
    boundaries: list[TrajectoryBoundary] = []
    cursor = 0
    for trajectory in trajectories:
        end = cursor + int(trajectory.shape[0])
        boundaries.append((cursor, end))
        cursor = end
    return boundaries


def default_boundaries(
    points: torch.Tensor,
    boundaries: list[TrajectoryBoundary] | None,
) -> list[TrajectoryBoundary]:
    """Use explicit boundaries, or treat all points as one trajectory."""
    return boundaries if boundaries is not None else [(0, int(points.shape[0]))]


def split_by_boundaries(points: torch.Tensor, boundaries: list[TrajectoryBoundary]) -> list[torch.Tensor]:
    """Split flattened points into trajectory tensors by boundaries."""
    return [points[start:end] for start, end in boundaries]


def trajectory_ids_for_points(
    n_points: int,
    boundaries: list[TrajectoryBoundary],
    device: torch.device,
) -> torch.Tensor:
    """Return a per-point trajectory-id lookup, with ``-1`` for unassigned rows."""
    trajectory_ids = torch.full((int(n_points),), -1, dtype=torch.long, device=device)
    for trajectory_id, (start, end) in enumerate(boundaries):
        if end > start:
            trajectory_ids[start:end] = int(trajectory_id)
    return trajectory_ids


def trajectory_id_mask(point_trajectory_ids: torch.Tensor, trajectory_ids: Iterable[int]) -> torch.Tensor:
    """Return a point mask for rows belonging to any supplied trajectory id."""
    mask = torch.zeros_like(point_trajectory_ids, dtype=torch.bool)
    for trajectory_id in trajectory_ids:
        mask |= point_trajectory_ids == int(trajectory_id)
    return mask


def trajectory_ids_from_mask(mask: torch.Tensor, point_trajectory_ids: torch.Tensor) -> list[int]:
    """Return sorted trajectory ids with at least one true point in ``mask``."""
    if not bool(mask.any().item()):
        return []
    ids = torch.unique(point_trajectory_ids[mask])
    return sorted(int(value) for value in ids.detach().cpu().tolist() if int(value) >= 0)


def trajectory_ids_intersecting_indices(
    indices: torch.Tensor,
    boundaries: list[TrajectoryBoundary],
) -> set[int]:
    """Map flattened point indices to stable trajectory ids derived from boundaries."""
    if indices.numel() == 0:
        return set()
    trajectory_ids: set[int] = set()
    for trajectory_id, (start, end) in enumerate(boundaries):
        if end <= start:
            continue
        if bool(((indices >= start) & (indices < end)).any().item()):
            trajectory_ids.add(int(trajectory_id))
    return trajectory_ids
