"""Tests F1-contribution training labels. See src/training/README.md for details."""

from __future__ import annotations

import pytest
import torch

from src.queries.query_types import (
    QUERY_TYPE_ID_CLUSTERING,
    QUERY_TYPE_ID_KNN,
    QUERY_TYPE_ID_RANGE,
    QUERY_TYPE_ID_SIMILARITY,
)
from src.training.importance_labels import (
    RANGE_USEFULNESS_LABEL_COMPONENTS,
    compute_typed_importance_labels,
    compute_typed_importance_labels_with_range_components,
)


def test_range_labels_match_singleton_point_f1_contribution() -> None:
    """Assert range labels equal the F1 gain of recovering one matching point."""
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


def test_range_labels_reward_each_in_box_point() -> None:
    """Assert duplicate in-box points are scored as individual range-query hits."""
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

    expected_gain = 2.0 / 3.0
    assert labels[0, QUERY_TYPE_ID_RANGE].item() == pytest.approx(expected_gain)
    assert labels[1, QUERY_TYPE_ID_RANGE].item() == pytest.approx(expected_gain)
    assert labels[2, QUERY_TYPE_ID_RANGE].item() == pytest.approx(0.0)


def test_large_range_labels_do_not_use_quadratic_proximity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Assert range labels do not allocate an all-pairs distance matrix."""
    n_per_trajectory = 2500
    time = torch.arange(n_per_trajectory * 2, dtype=torch.float32)
    lat = torch.cat(
        [
            torch.linspace(0.0, 1.0, n_per_trajectory),
            torch.linspace(0.1, 1.1, n_per_trajectory),
        ]
    )
    lon = torch.cat(
        [
            torch.linspace(0.0, 1.0, n_per_trajectory),
            torch.linspace(0.1, 1.1, n_per_trajectory),
        ]
    )
    speed = torch.ones_like(time)
    heading = torch.zeros_like(time)
    is_start = torch.zeros_like(time)
    is_end = torch.zeros_like(time)
    turn = torch.zeros_like(time)
    points = torch.stack([time, lat, lon, speed, heading, is_start, is_end, turn], dim=1)
    boundaries = [(0, n_per_trajectory), (n_per_trajectory, n_per_trajectory * 2)]
    queries = [
        {
            "type": "range",
            "params": {
                "lat_min": -1.0,
                "lat_max": 2.0,
                "lon_min": -1.0,
                "lon_max": 2.0,
                "t_start": -1.0,
                "t_end": float(points[-1, 0].item()) + 1.0,
            },
        }
    ]

    def fail_cdist(*args, **kwargs):
        raise AssertionError("large range labels should not call dense torch.cdist")

    monkeypatch.setattr(torch, "cdist", fail_cdist)

    labels, labelled_mask = compute_typed_importance_labels(points, boundaries, queries, seed=1)

    assert bool(labelled_mask[:, QUERY_TYPE_ID_RANGE].all().item())
    assert float(labels[:, QUERY_TYPE_ID_RANGE].sum().item()) > 0.0
    assert torch.isfinite(labels).all()


def test_range_boundary_prior_is_optional_and_mass_preserving() -> None:
    """Assert boundary weighting is opt-in and preserves total query label mass."""
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.1, 1.0],
            [2.0, 0.0, 0.2, 1.0],
            [3.0, 0.0, 0.3, 1.0],
            [4.0, 9.0, 9.0, 1.0],
        ],
        dtype=torch.float32,
    )
    boundaries = [(0, 5)]
    queries = [
        {
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
    ]

    pure, _ = compute_typed_importance_labels(points, boundaries, queries, seed=1)
    boundary, _ = compute_typed_importance_labels(
        points,
        boundaries,
        queries,
        seed=1,
        range_boundary_prior_weight=1.0,
    )

    pure_values = pure[:4, QUERY_TYPE_ID_RANGE]
    boundary_values = boundary[:4, QUERY_TYPE_ID_RANGE]
    assert pure_values.tolist() == pytest.approx([0.4, 0.4, 0.4, 0.4])
    assert boundary_values[0].item() > boundary_values[1].item()
    assert boundary_values[3].item() > boundary_values[2].item()
    assert float(boundary_values.sum().item()) == pytest.approx(float(pure_values.sum().item()))


def test_range_usefulness_labels_prioritize_ship_span_and_shape_points() -> None:
    """Assert usefulness labels add navigational signal beyond uniform in-box points."""
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.3, 1.0, 0.0, 0.0, 1.0, 0.0],
            [1.5, 0.8, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    boundaries = [(0, 4), (4, 5)]
    queries = [
        {
            "type": "range",
            "params": {
                "lat_min": -1.0,
                "lat_max": 1.0,
                "lon_min": -1.0,
                "lon_max": 1.0,
                "t_start": -1.0,
                "t_end": 4.0,
            },
        }
    ]

    point_labels, _ = compute_typed_importance_labels(
        points,
        boundaries,
        queries,
        seed=1,
        range_label_mode="point_f1",
    )
    useful_labels, _ = compute_typed_importance_labels(
        points,
        boundaries,
        queries,
        seed=1,
        range_label_mode="usefulness",
    )
    point_values = point_labels[:, QUERY_TYPE_ID_RANGE]
    useful_values = useful_labels[:, QUERY_TYPE_ID_RANGE]

    assert point_values[:5].tolist() == pytest.approx([1.0 / 3.0] * 5)
    assert useful_values[0].item() > useful_values[1].item()
    assert useful_values[3].item() > useful_values[2].item()
    assert useful_values[4].item() > useful_values[1].item()


def test_range_usefulness_component_labels_sum_to_training_labels() -> None:
    """Assert component diagnostics decompose the unclipped usefulness target."""
    points = torch.tensor(
        [
            [0.0, -2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 0.2, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [3.0, 2.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    queries = [
        {
            "type": "range",
            "params": {
                "lat_min": -1.0,
                "lat_max": 1.0,
                "lon_min": -1.0,
                "lon_max": 1.0,
                "t_start": -1.0,
                "t_end": 4.0,
            },
        }
    ]

    labels, labelled_mask, component_labels = compute_typed_importance_labels_with_range_components(
        points,
        [(0, 4)],
        queries,
        seed=1,
    )

    component_sum = torch.stack(
        [component_labels[name][:, QUERY_TYPE_ID_RANGE] for name in RANGE_USEFULNESS_LABEL_COMPONENTS]
    ).sum(dim=0)
    assert bool(labelled_mask[:, QUERY_TYPE_ID_RANGE].all().item())
    assert torch.allclose(labels[:, QUERY_TYPE_ID_RANGE], component_sum)
    assert component_labels["range_crossing_f1"][0, QUERY_TYPE_ID_RANGE].item() > 0.0
    assert component_labels["range_crossing_f1"][3, QUERY_TYPE_ID_RANGE].item() > 0.0
    assert component_labels["range_point_f1"][0, QUERY_TYPE_ID_RANGE].item() == pytest.approx(0.0)
    assert component_labels["range_point_f1"][1, QUERY_TYPE_ID_RANGE].item() > 0.0


def test_range_usefulness_labels_preserve_point_component_mass() -> None:
    """Assert usefulness labels remain finite and include crossing brackets."""
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.1, 1.0],
            [2.0, 0.0, 0.2, 1.0],
            [3.0, 9.0, 9.0, 1.0],
        ],
        dtype=torch.float32,
    )
    queries = [
        {
            "type": "range",
            "params": {
                "lat_min": -1.0,
                "lat_max": 1.0,
                "lon_min": -1.0,
                "lon_max": 1.0,
                "t_start": -1.0,
                "t_end": 3.0,
            },
        }
    ]

    labels, labelled_mask = compute_typed_importance_labels(
        points,
        [(0, 4)],
        queries,
        seed=1,
        range_label_mode="usefulness",
    )

    values = labels[:, QUERY_TYPE_ID_RANGE]
    assert bool(labelled_mask[:, QUERY_TYPE_ID_RANGE].all().item())
    assert torch.isfinite(values).all()
    assert bool((values[:3] > 0.0).all().item())
    assert values[3].item() > 0.0
    assert values[3].item() < values[:3].max().item()


def test_range_usefulness_labels_include_between_sample_crossing_brackets() -> None:
    points = torch.tensor(
        [
            [0.0, -2.0, 0.0, 1.0],
            [1.0, 2.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    queries = [
        {
            "type": "range",
            "params": {
                "lat_min": -1.0,
                "lat_max": 1.0,
                "lon_min": -1.0,
                "lon_max": 1.0,
                "t_start": -1.0,
                "t_end": 2.0,
            },
        }
    ]

    point_labels, _ = compute_typed_importance_labels(points, [(0, 2)], queries, seed=1, range_label_mode="point_f1")
    useful_labels, labelled_mask = compute_typed_importance_labels(
        points,
        [(0, 2)],
        queries,
        seed=1,
        range_label_mode="usefulness",
    )

    assert bool(labelled_mask[:, QUERY_TYPE_ID_RANGE].all().item())
    assert point_labels[:, QUERY_TYPE_ID_RANGE].tolist() == pytest.approx([0.0, 0.0])
    assert bool((useful_labels[:, QUERY_TYPE_ID_RANGE] > 0.0).all().item())


def test_range_label_mode_rejects_unknown_mode() -> None:
    points = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
    queries = [
        {
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
    ]

    with pytest.raises(ValueError, match="range_label_mode"):
        compute_typed_importance_labels(
            points,
            [(0, 1)],
            queries,
            seed=1,
            range_label_mode="not-a-mode",
        )


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


def test_knn_labels_all_inwindow_points_when_below_representative_cap() -> None:
    """Assert small selected trajectories keep dense kNN supervision."""
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],    # traj 0, point 0 — inside window
            [0.1, 0.0, 0.001, 1.0],  # traj 0, point 1 — inside window
            [0.2, 0.0, 0.002, 1.0],  # traj 0, point 2 — inside window
            [0.3, 20.0, 20.0, 1.0],  # traj 0, point 3 — inside window (far, but still in)
            [0.0, 5.0, 5.0, 1.0],    # traj 1         — NOT selected by k=1
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

    # All 4 in-window points of trajectory 0 receive positive label because they are below the cap.
    assert labels[:4, QUERY_TYPE_ID_KNN].sum().item() > 0.0
    # Trajectory 1 is not selected (k=1 picks closest trajectory, which is traj 0)
    assert labels[4, QUERY_TYPE_ID_KNN].item() == pytest.approx(0.0)
    assert labels[:4, QUERY_TYPE_ID_KNN].sum().item() == pytest.approx(1.0)


def test_knn_labels_keep_nearest_high_cap_representatives() -> None:
    """Assert kNN labels cap very dense trajectories at the nearest 64 representatives."""
    near_points = [[float(i), 0.0, float(i) * 0.001, 1.0] for i in range(64)]
    far_points = [[float(64 + i), 10.0, 10.0 + float(i) * 0.001, 1.0] for i in range(6)]
    points = torch.tensor(near_points + far_points, dtype=torch.float32)
    boundaries = [(0, len(points))]
    queries = [
        {
            "type": "knn",
            "params": {
                "lat": 0.0,
                "lon": 0.0,
                "t_center": 35.0,
                "t_half_window": 100.0,
                "k": 1,
            },
        }
    ]

    labels, _ = compute_typed_importance_labels(points, boundaries, queries, seed=1)

    positives = labels[:, QUERY_TYPE_ID_KNN] > 0.0
    assert int(positives.sum().item()) == 64
    assert bool(positives[:64].all().item())
    assert not bool(positives[64:].any().item())
    assert labels[:, QUERY_TYPE_ID_KNN].sum().item() == pytest.approx(1.0)


def test_similarity_labels_keep_reference_nearest_representatives() -> None:
    """Assert dense similarity labels focus on points nearest the reference snippet."""
    near_points = [[float(i), 0.0, float(i) * 0.001, 1.0] for i in range(64)]
    far_points = [[float(64 + i), 10.0, 10.0 + float(i) * 0.001, 1.0] for i in range(6)]
    points = torch.tensor(near_points + far_points, dtype=torch.float32)
    boundaries = [(0, len(points))]
    queries = [
        {
            "type": "similarity",
            "params": {
                "lat_query_centroid": 0.0,
                "lon_query_centroid": 0.0,
                "t_start": -1.0,
                "t_end": 100.0,
                "radius": 100.0,
                "top_k": 1,
            },
            "reference": points[:5, :3].tolist(),
        }
    ]

    labels, _ = compute_typed_importance_labels(points, boundaries, queries, seed=1)

    positives = labels[:, QUERY_TYPE_ID_SIMILARITY] > 0.0
    assert int(positives.sum().item()) == 64
    assert bool(positives[:64].all().item())
    assert not bool(positives[64:].any().item())
    assert labels[:, QUERY_TYPE_ID_SIMILARITY].sum().item() == pytest.approx(1.0)


def test_knn_labels_exclude_out_of_window_points() -> None:
    """Assert kNN labels exclude points of the selected trajectory that are outside the time window."""
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],   # traj 0, inside window  [t_center=0 ± 0.5]
            [0.3, 0.0, 0.0, 1.0],   # traj 0, inside window
            [5.0, 0.0, 0.0, 1.0],   # traj 0, OUTSIDE window
        ],
        dtype=torch.float32,
    )
    boundaries = [(0, 3)]
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

    # The two in-window points are labeled; the out-of-window point is not
    assert labels[0, QUERY_TYPE_ID_KNN].item() > 0.0
    assert labels[1, QUERY_TYPE_ID_KNN].item() > 0.0
    assert labels[2, QUERY_TYPE_ID_KNN].item() == pytest.approx(0.0)


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


def test_similarity_labels_follow_executor_centroid_geometry() -> None:
    """Assert similarity labels support trajectories selected by execution geometry."""
    points = torch.tensor(
        [
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [0.0, 5.0, 5.0, 1.0],
            [1.0, 6.0, 5.0, 1.0],
        ],
        dtype=torch.float32,
    )
    boundaries = [(0, 2), (2, 4)]
    queries = [
        {
            "type": "similarity",
            "params": {
                "lat_query_centroid": 0.0,
                "lon_query_centroid": 0.0,
                "t_start": -1.0,
                "t_end": 2.0,
                "radius": 0.25,
                "top_k": 1,
            },
            "reference": [[0.0, -1.0, 0.0], [1.0, 1.0, 0.0]],
        }
    ]

    labels, _ = compute_typed_importance_labels(points, boundaries, queries, seed=1)

    assert labels[0, QUERY_TYPE_ID_SIMILARITY].item() == pytest.approx(0.5)
    assert labels[1, QUERY_TYPE_ID_SIMILARITY].item() == pytest.approx(0.5)
    assert labels[2, QUERY_TYPE_ID_SIMILARITY].item() == pytest.approx(0.0)
    assert labels[3, QUERY_TYPE_ID_SIMILARITY].item() == pytest.approx(0.0)
