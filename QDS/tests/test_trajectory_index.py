from __future__ import annotations

import torch

from src.data.trajectory_index import (
    boundaries_from_trajectories,
    default_boundaries,
    split_by_boundaries,
    trajectory_id_mask,
    trajectory_ids_for_points,
    trajectory_ids_from_mask,
    trajectory_ids_intersecting_indices,
)


def test_boundaries_and_split_round_trip() -> None:
    trajectories = [
        torch.tensor([[1.0], [2.0]], dtype=torch.float32),
        torch.tensor([[3.0]], dtype=torch.float32),
    ]

    boundaries = boundaries_from_trajectories(trajectories)

    assert boundaries == [(0, 2), (2, 3)]
    assert [part.tolist() for part in split_by_boundaries(torch.cat(trajectories, dim=0), boundaries)] == [
        [[1.0], [2.0]],
        [[3.0]],
    ]


def test_trajectory_id_helpers_ignore_unassigned_points() -> None:
    trajectory_ids = trajectory_ids_for_points(
        n_points=5,
        boundaries=[(0, 2), (3, 5)],
        device=torch.device("cpu"),
    )
    mask = torch.tensor([False, True, True, True, False])

    assert trajectory_ids.tolist() == [0, 0, -1, 1, 1]
    assert trajectory_id_mask(trajectory_ids, {1}).tolist() == [False, False, False, True, True]
    assert trajectory_ids_from_mask(mask, trajectory_ids) == [0, 1]


def test_default_boundaries_and_index_intersection() -> None:
    points = torch.zeros((4, 3), dtype=torch.float32)

    assert default_boundaries(points, None) == [(0, 4)]
    assert trajectory_ids_intersecting_indices(torch.tensor([0, 3]), [(0, 2), (2, 4)]) == {0, 1}
