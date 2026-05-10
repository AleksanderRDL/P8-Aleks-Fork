"""Tests for Phase 2 range workload diagnostics and acceptance filters."""

from __future__ import annotations

import json

import pytest
import torch

from src.data.ais_loader import generate_synthetic_ais_data
from src.queries.query_generator import generate_typed_query_workload
from src.queries.query_types import QUERY_TYPE_ID_RANGE
from src.queries.workload_diagnostics import (
    compute_range_label_diagnostics,
    compute_range_workload_diagnostics,
)
from src.training.importance_labels import compute_typed_importance_labels


def _points_and_boundaries() -> tuple[torch.Tensor, list[tuple[int, int]]]:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.1, 0.1, 1.0],
            [2.0, 0.2, 0.2, 1.0],
            [3.0, 5.0, 5.0, 1.0],
            [4.0, 5.1, 5.1, 1.0],
        ],
        dtype=torch.float32,
    )
    return points, [(0, 3), (3, 5)]


def _range_query(lat_min: float, lat_max: float, lon_min: float, lon_max: float, t_start: float, t_end: float) -> dict:
    return {
        "type": "range",
        "params": {
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_min": lon_min,
            "lon_max": lon_max,
            "t_start": t_start,
            "t_end": t_end,
        },
    }


def test_range_workload_diagnostics_reports_hit_distributions() -> None:
    points, boundaries = _points_and_boundaries()
    queries = [
        _range_query(-1.0, 1.0, -1.0, 1.0, -1.0, 2.5),
        _range_query(4.9, 5.2, 4.9, 5.2, 2.5, 4.5),
    ]

    diagnostics = compute_range_workload_diagnostics(points, boundaries, queries)

    assert diagnostics["summary"]["range_query_count"] == 2
    assert diagnostics["queries"][0]["point_hits"] == 3
    assert diagnostics["queries"][0]["trajectory_hits"] == 1
    assert diagnostics["queries"][1]["point_hits"] == 2
    assert diagnostics["summary"]["point_hit_count_p50"] == pytest.approx(2.5)
    assert diagnostics["summary"]["coverage_fraction"] == pytest.approx(1.0)


def test_range_diagnostics_marks_broad_queries() -> None:
    points, boundaries = _points_and_boundaries()
    queries = [_range_query(-1.0, 6.0, -1.0, 6.0, -1.0, 10.0)]

    diagnostics = compute_range_workload_diagnostics(
        points,
        boundaries,
        queries,
        max_point_hit_fraction=0.50,
        max_trajectory_hit_fraction=0.50,
        max_box_volume_fraction=0.50,
    )

    assert diagnostics["queries"][0]["is_too_broad"] is True
    assert diagnostics["summary"]["too_broad_query_rate"] == pytest.approx(1.0)


def test_range_diagnostics_marks_near_duplicate_boxes() -> None:
    points, boundaries = _points_and_boundaries()
    queries = [
        _range_query(-1.0, 1.0, -1.0, 1.0, -1.0, 2.5),
        _range_query(-1.0, 1.0, -1.0, 1.0, -1.0, 2.5),
    ]

    diagnostics = compute_range_workload_diagnostics(
        points,
        boundaries,
        queries,
        duplicate_iou_threshold=0.85,
    )

    assert diagnostics["queries"][0]["near_duplicate_of"] is None
    assert diagnostics["queries"][1]["near_duplicate_of"] == 0
    assert diagnostics["summary"]["near_duplicate_query_rate"] == pytest.approx(0.5)


def test_range_acceptance_rejects_overly_broad_queries() -> None:
    trajectories = generate_synthetic_ais_data(n_ships=4, n_points_per_ship=20, seed=9)

    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=5,
        workload_mix={"range": 1.0},
        seed=1,
        range_spatial_fraction=1.0,
        range_time_fraction=1.0,
        range_max_box_volume_fraction=0.0,
        range_acceptance_max_attempts=4,
    )

    assert workload.generation_diagnostics is not None
    generation = workload.generation_diagnostics["range_acceptance"]
    assert len(workload.typed_queries) == 0
    assert generation["exhausted"] is True
    assert generation["rejected"] == 4
    assert generation["rejection_reasons"]["too_broad"] == 4


def test_range_acceptance_keeps_requested_query_count_when_possible() -> None:
    trajectories = generate_synthetic_ais_data(n_ships=4, n_points_per_ship=30, seed=10)

    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=8,
        workload_mix={"range": 1.0},
        seed=2,
        range_spatial_fraction=0.02,
        range_time_fraction=0.04,
        range_min_point_hits=1,
        range_acceptance_max_attempts=40,
    )

    assert workload.generation_diagnostics is not None
    generation = workload.generation_diagnostics["range_acceptance"]
    assert len(workload.typed_queries) == 8
    assert generation["accepted"] == 8
    assert generation["rejected"] == 0
    assert generation["exhausted"] is False


def test_range_label_diagnostics_reports_positive_fraction() -> None:
    points, boundaries = _points_and_boundaries()
    queries = [_range_query(-1.0, 1.0, -1.0, 1.0, -1.0, 1.5)]

    labels, labelled_mask = compute_typed_importance_labels(points, boundaries, queries, seed=1)
    diagnostics = compute_range_label_diagnostics(labels, labelled_mask)

    assert bool(labelled_mask[:, QUERY_TYPE_ID_RANGE].all().item())
    assert diagnostics["labelled_point_count"] == 5
    assert diagnostics["positive_point_count"] == 2
    assert diagnostics["positive_label_fraction"] == pytest.approx(0.4)
    assert diagnostics["positive_label_max"] > 0.0


def test_phase2_diagnostics_dump_is_json_serializable() -> None:
    points, boundaries = _points_and_boundaries()
    queries = [_range_query(-1.0, 1.0, -1.0, 1.0, -1.0, 2.5)]

    diagnostics = compute_range_workload_diagnostics(points, boundaries, queries)

    json.dumps(diagnostics)
