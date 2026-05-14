"""Tests range query execution semantics. See queries/README.md for details."""

from __future__ import annotations

import pytest
import torch

from queries.query_executor import execute_range_query, execute_typed_query


def test_range_query_returns_intersecting_trajectories() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.1, 0.1],
            [2.0, 5.0, 5.0],
            [3.0, 5.1, 5.1],
        ],
        dtype=torch.float32,
    )
    boundaries = [(0, 2), (2, 4)]
    params = {
        "lat_min": -0.5,
        "lat_max": 0.5,
        "lon_min": -0.5,
        "lon_max": 0.5,
        "t_start": 0.0,
        "t_end": 1.5,
    }

    assert execute_range_query(points, params, boundaries) == {0}


def test_typed_query_rejects_non_range_workloads() -> None:
    points = torch.zeros((1, 3), dtype=torch.float32)

    with pytest.raises(ValueError, match="Only range queries"):
        execute_typed_query(points, {"type": "legacy", "params": {}}, [(0, 1)])
