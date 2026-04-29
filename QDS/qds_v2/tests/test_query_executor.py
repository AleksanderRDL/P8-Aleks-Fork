"""Tests typed query execution semantics. See src/queries/README.md for details."""

from __future__ import annotations

import torch

from src.queries.query_executor import execute_knn_query


def test_knn_returns_nearest_distinct_trajectories() -> None:
    """Assert kNN selects nearest trajectories instead of duplicate points from one vessel."""
    points = torch.tensor(
        [
            [0.0, 0.0, 0.000, 1.0],
            [0.0, 0.0, 0.001, 1.0],
            [0.0, 0.0, 0.002, 1.0],
            [0.0, 0.0, 0.010, 1.0],
            [0.0, 1.0, 1.000, 1.0],
        ],
        dtype=torch.float32,
    )
    boundaries = [(0, 3), (3, 4), (4, 5)]
    params = {
        "lat": 0.0,
        "lon": 0.0,
        "t_center": 0.0,
        "t_half_window": 1.0,
        "k": 2,
    }

    assert execute_knn_query(points, params, boundaries) == {0, 1}
