"""Tests range-query evaluation uses point-level preservation."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from src.evaluation.evaluate_methods import evaluate_method


@dataclass
class FixedMaskMethod:
    """Test method that returns a precomputed retention mask."""

    retained_mask: torch.Tensor
    name: str = "FixedMask"

    def simplify(
        self,
        points: torch.Tensor,
        boundaries: list[tuple[int, int]],
        compression_ratio: float,
    ) -> torch.Tensor:
        return self.retained_mask.clone()


def test_range_evaluation_scores_point_hits_not_trajectory_presence() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.1, 0.1, 1.0],
            [2.0, 0.2, 0.2, 1.0],
            [3.0, 0.3, 0.3, 1.0],
            [4.0, 9.0, 9.0, 1.0],
        ],
        dtype=torch.float32,
    )
    boundaries = [(0, 5)]
    query = {
        "type": "range",
        "params": {
            "lat_min": -1.0,
            "lat_max": 1.0,
            "lon_min": -1.0,
            "lon_max": 1.0,
            "t_start": -1.0,
            "t_end": 3.5,
        },
    }
    method = FixedMaskMethod(torch.tensor([False, True, False, False, True]))

    result = evaluate_method(
        method=method,
        points=points,
        boundaries=boundaries,
        typed_queries=[query],
        workload_map={"range": 1.0},
        compression_ratio=0.4,
    )

    assert result.per_type_f1["range"] == pytest.approx(0.4)
    assert result.aggregate_f1 == pytest.approx(0.4)
    assert result.range_point_f1 == pytest.approx(0.4)
    assert 0.0 <= result.range_usefulness_score <= 1.0


def test_range_evaluation_counts_duplicate_rows_as_distinct_point_hits() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 9.0, 9.0, 1.0],
        ],
        dtype=torch.float32,
    )
    boundaries = [(0, 3)]
    query = {
        "type": "range",
        "params": {
            "lat_min": -1.0,
            "lat_max": 1.0,
            "lon_min": -1.0,
            "lon_max": 1.0,
            "t_start": -1.0,
            "t_end": 1.0,
        },
    }
    method = FixedMaskMethod(torch.tensor([True, False, True]))

    result = evaluate_method(
        method=method,
        points=points,
        boundaries=boundaries,
        typed_queries=[query],
        workload_map={"range": 1.0},
        compression_ratio=1.0 / 3.0,
    )

    assert result.per_type_f1["range"] == pytest.approx(2.0 / 3.0)
    assert result.aggregate_f1 == pytest.approx(2.0 / 3.0)
    assert result.range_point_f1 == pytest.approx(2.0 / 3.0)
