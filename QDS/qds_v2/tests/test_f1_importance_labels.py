"""Tests F1-contribution training labels. See src/training/README.md for details."""

from __future__ import annotations

import pytest
import torch

from src.queries.query_types import QUERY_TYPE_ID_CLUSTERING, QUERY_TYPE_ID_KNN, QUERY_TYPE_ID_RANGE
from src.training.importance_labels import compute_typed_importance_labels


def test_range_labels_match_singleton_f1_contribution() -> None:
    """Assert range labels equal the F1 gain of recovering one matching trajectory."""
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 3.0],
            [1.0, 5.0, 5.0, 9.0],
            [0.0, 0.1, 0.1, 1.0],
            [1.0, 6.0, 6.0, 7.0],
        ],
        dtype=torch.float32,
    )
    boundaries = [(0, 2), (2, 4)]
    queries = [
        {
            "type": "range",
            "params": {
                "lat_min": -1.0,
                "lat_max": 1.0,
                "lon_min": -1.0,
                "lon_max": 1.0,
                "t_start": -1.0,
                "t_end": 0.5,
            },
        }
    ]

    labels, labelled_mask = compute_typed_importance_labels(points, boundaries, queries, seed=1)

    expected_gain = 2.0 / 3.0
    assert labels[0, QUERY_TYPE_ID_RANGE].item() == pytest.approx(expected_gain)
    assert labels[2, QUERY_TYPE_ID_RANGE].item() == pytest.approx(expected_gain)
    assert labels[1, QUERY_TYPE_ID_RANGE].item() == pytest.approx(0.0)
    assert labels[3, QUERY_TYPE_ID_RANGE].item() == pytest.approx(0.0)
    assert bool(labelled_mask[:, QUERY_TYPE_ID_RANGE].all().item())


def test_range_labels_distribute_one_hit_gain_over_interchangeable_points() -> None:
    """Assert duplicate in-box points from one trajectory share one trajectory-hit gain."""
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.1, 0.1, 1.0],
            [2.0, 5.0, 5.0, 1.0],
        ],
        dtype=torch.float32,
    )
    boundaries = [(0, 3)]
    queries = [
        {
            "type": "range",
            "params": {
                "lat_min": -1.0,
                "lat_max": 1.0,
                "lon_min": -1.0,
                "lon_max": 1.0,
                "t_start": -1.0,
                "t_end": 1.5,
            },
        }
    ]

    labels, _ = compute_typed_importance_labels(points, boundaries, queries, seed=1)

    assert labels[0, QUERY_TYPE_ID_RANGE].item() == pytest.approx(0.5)
    assert labels[1, QUERY_TYPE_ID_RANGE].item() == pytest.approx(0.5)
    assert labels[2, QUERY_TYPE_ID_RANGE].item() == pytest.approx(0.0)


def test_knn_labels_follow_f1_trajectory_hits_not_speed() -> None:
    """Assert kNN labels are driven by F1 hit membership instead of speed heuristics."""
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [10.0, 4.0, 4.0, 100.0],
            [0.0, 2.0, 2.0, 200.0],
            [10.0, 8.0, 8.0, 300.0],
        ],
        dtype=torch.float32,
    )
    boundaries = [(0, 2), (2, 4)]
    queries = [
        {
            "type": "knn",
            "params": {
                "lat": 0.0,
                "lon": 0.0,
                "t_center": 0.0,
                "t_half_window": 0.5,
                "k": 1,
            },
        }
    ]

    labels, _ = compute_typed_importance_labels(points, boundaries, queries, seed=1)

    assert labels[0, QUERY_TYPE_ID_KNN].item() == pytest.approx(1.0)
    assert labels[1, QUERY_TYPE_ID_KNN].item() == pytest.approx(0.0)
    assert labels[2, QUERY_TYPE_ID_KNN].item() == pytest.approx(0.0)
    assert labels[3, QUERY_TYPE_ID_KNN].item() == pytest.approx(0.0)


def test_knn_labels_focus_nearest_representatives_not_whole_time_window() -> None:
    """Assert kNN labels do not reward every point in the selected trajectory window."""
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.1, 0.0, 0.001, 1.0],
            [0.2, 0.0, 0.002, 1.0],
            [0.3, 20.0, 20.0, 1.0],
            [0.0, 5.0, 5.0, 1.0],
        ],
        dtype=torch.float32,
    )
    boundaries = [(0, 4), (4, 5)]
    queries = [
        {
            "type": "knn",
            "params": {
                "lat": 0.0,
                "lon": 0.0,
                "t_center": 0.0,
                "t_half_window": 1.0,
                "k": 1,
            },
        }
    ]

    labels, _ = compute_typed_importance_labels(points, boundaries, queries, seed=1)

    assert labels[:3, QUERY_TYPE_ID_KNN].sum().item() == pytest.approx(1.0)
    assert labels[3, QUERY_TYPE_ID_KNN].item() == pytest.approx(0.0)
    assert labels[4, QUERY_TYPE_ID_KNN].item() == pytest.approx(0.0)


def test_clustering_labels_use_pairwise_f1_membership() -> None:
    """Assert clustering labels reward trajectories that participate in true co-membership pairs."""
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.01, 0.0, 1.0],
            [0.0, 0.02, 0.0, 1.0],
            [0.0, 5.0, 5.0, 1.0],
        ],
        dtype=torch.float32,
    )
    boundaries = [(0, 1), (1, 2), (2, 3), (3, 4)]
    queries = [
        {
            "type": "clustering",
            "params": {
                "lat_min": -1.0,
                "lat_max": 6.0,
                "lon_min": -1.0,
                "lon_max": 6.0,
                "t_start": -1.0,
                "t_end": 1.0,
                "eps": 0.05,
                "min_samples": 2,
            },
        }
    ]

    labels, _ = compute_typed_importance_labels(points, boundaries, queries, seed=1)

    assert labels[0, QUERY_TYPE_ID_CLUSTERING].item() > 0.0
    assert labels[1, QUERY_TYPE_ID_CLUSTERING].item() > 0.0
    assert labels[2, QUERY_TYPE_ID_CLUSTERING].item() > 0.0
    assert labels[3, QUERY_TYPE_ID_CLUSTERING].item() == pytest.approx(0.0)