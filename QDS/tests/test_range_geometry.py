from __future__ import annotations

import pytest
import torch

from src.queries.range_geometry import (
    haversine_km_to_point,
    points_in_range_box,
    segment_box_bracket_mask,
    segment_box_crossings,
)


def _range_params() -> dict[str, float]:
    return {
        "lat_min": -1.0,
        "lat_max": 1.0,
        "lon_min": -1.0,
        "lon_max": 1.0,
        "t_start": 0.0,
        "t_end": 10.0,
    }


def test_segment_box_crossings_include_between_sample_pass_through() -> None:
    points = torch.tensor(
        [
            [1.0, 0.0, -2.0, 1.0],
            [2.0, 0.0, 2.0, 1.0],
        ],
        dtype=torch.float32,
    )

    assert segment_box_crossings(points, _range_params()).tolist() == [True]
    assert segment_box_bracket_mask(points, [(0, 2)], _range_params()).tolist() == [True, True]


def test_segment_box_crossings_ignore_fully_inside_segments() -> None:
    points = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0],
            [2.0, 0.5, 0.5, 1.0],
            [3.0, 0.0, 2.0, 1.0],
        ],
        dtype=torch.float32,
    )

    assert segment_box_crossings(points, _range_params()).tolist() == [False, True]
    assert segment_box_bracket_mask(points, [(0, 3)], _range_params()).tolist() == [False, True, True]


def test_segment_box_crossings_reject_time_disjoint_segments() -> None:
    points = torch.tensor(
        [
            [20.0, 0.0, -2.0, 1.0],
            [21.0, 0.0, 2.0, 1.0],
        ],
        dtype=torch.float32,
    )

    assert segment_box_crossings(points, _range_params()).tolist() == [False]
    assert segment_box_bracket_mask(points, [(0, 2)], _range_params()).tolist() == [False, False]


def test_segment_box_brackets_do_not_cross_trajectory_boundaries() -> None:
    points = torch.tensor(
        [
            [1.0, 0.0, -2.0, 1.0],
            [2.0, 0.0, -2.0, 1.0],
            [3.0, 0.0, 2.0, 1.0],
            [4.0, 0.0, 2.0, 1.0],
        ],
        dtype=torch.float32,
    )

    separated = segment_box_bracket_mask(points, [(0, 2), (2, 4)], _range_params())
    combined = segment_box_bracket_mask(points, [(0, 4)], _range_params())

    assert separated.tolist() == [False, False, False, False]
    assert combined.tolist() == [False, True, True, False]


def test_points_in_range_box_uses_time_lat_lon_columns() -> None:
    points = torch.tensor(
        [
            [5.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [5.0, 2.0, 0.0],
        ],
        dtype=torch.float32,
    )

    assert points_in_range_box(points, _range_params()).tolist() == [True, False, False]


def test_haversine_km_to_point_matches_equator_degree_scale() -> None:
    distances = haversine_km_to_point(
        torch.tensor([0.0, 0.0], dtype=torch.float32),
        torch.tensor([0.0, 1.0], dtype=torch.float32),
        0.0,
        0.0,
    )

    assert distances[0].item() == pytest.approx(0.0, abs=1e-6)
    assert distances[1].item() == pytest.approx(111.19, abs=0.05)
