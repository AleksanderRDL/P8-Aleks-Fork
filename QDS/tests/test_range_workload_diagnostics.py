"""Tests for Phase 2 range workload diagnostics and acceptance filters."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from data.ais_loader import generate_synthetic_ais_data
from experiments.experiment_config import build_experiment_config
from queries.workload import TypedQueryWorkload
from experiments.experiment_pipeline_helpers import _range_workload_diagnostics
from experiments.range_cache import RangeRuntimeCache, prepare_range_label_cache as _prepare_range_label_cache
from queries.query_generator import generate_typed_query_workload
from queries.query_types import QUERY_TYPE_ID_RANGE, pad_query_features
from queries.workload_diagnostics import (
    compute_range_label_diagnostics,
    compute_range_workload_diagnostics,
)
from training.importance_labels import (
    compute_typed_importance_labels,
    compute_typed_importance_labels_with_range_components,
)


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


def test_range_workload_diagnostics_can_reuse_known_coverage_fraction() -> None:
    points, boundaries = _points_and_boundaries()
    queries = [
        _range_query(-1.0, 1.0, -1.0, 1.0, -1.0, 2.5),
        _range_query(4.9, 5.2, 4.9, 5.2, 2.5, 4.5),
    ]

    diagnostics = compute_range_workload_diagnostics(
        points,
        boundaries,
        queries,
        coverage_fraction=0.42,
    )

    assert diagnostics["summary"]["coverage_fraction"] == pytest.approx(0.42)
    assert diagnostics["queries"][0]["point_hits"] == 3
    assert diagnostics["queries"][0]["trajectory_hits"] == 1
    assert diagnostics["queries"][1]["point_hits"] == 2
    assert diagnostics["queries"][1]["trajectory_hits"] == 1


def test_range_workload_diagnostics_uses_supplied_masks() -> None:
    points, boundaries = _points_and_boundaries()
    queries = [
        _range_query(-100.0, -99.0, -100.0, -99.0, -100.0, -99.0),
        _range_query(-100.0, -99.0, -100.0, -99.0, -100.0, -99.0),
    ]
    supplied_masks = {
        0: torch.tensor([True, True, True, False, False]),
        1: torch.tensor([False, False, False, True, True]),
    }
    calls: list[int] = []

    def mask_provider(query_index: int, _query: dict) -> torch.Tensor:
        calls.append(query_index)
        return supplied_masks[query_index]

    diagnostics = compute_range_workload_diagnostics(
        points,
        boundaries,
        queries,
        mask_provider=mask_provider,
    )

    assert calls == [0, 1]
    assert diagnostics["queries"][0]["point_hits"] == 3
    assert diagnostics["queries"][1]["point_hits"] == 2
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
        workload_map={"range": 1.0},
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
        workload_map={"range": 1.0},
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


def test_range_label_diagnostics_reports_component_mass() -> None:
    points, boundaries = _points_and_boundaries()
    queries = [_range_query(-1.0, 1.0, -1.0, 1.0, -1.0, 2.5)]

    labels, labelled_mask, component_labels = compute_typed_importance_labels_with_range_components(
        points,
        boundaries,
        queries,
        seed=1,
    )
    diagnostics = compute_range_label_diagnostics(labels, labelled_mask, component_labels)
    fractions = diagnostics["component_positive_label_mass_fraction"]

    assert diagnostics["positive_label_mass"] > 0.0
    assert fractions["range_point_f1"] > 0.0
    assert fractions["range_ship_f1"] > 0.0
    assert sum(float(value) for value in fractions.values()) == pytest.approx(1.0)


def test_range_diagnostics_dump_is_json_serializable() -> None:
    points, boundaries = _points_and_boundaries()
    queries = [_range_query(-1.0, 1.0, -1.0, 1.0, -1.0, 2.5)]

    diagnostics = compute_range_workload_diagnostics(points, boundaries, queries)

    json.dumps(diagnostics)


def test_range_workload_diagnostics_cache_reuses_labels(tmp_path: Path) -> None:
    points, boundaries = _points_and_boundaries()
    queries = [_range_query(-1.0, 1.0, -1.0, 1.0, -1.0, 2.5)]
    features, type_ids = pad_query_features(queries)
    workload = TypedQueryWorkload(
        query_features=features,
        typed_queries=queries,
        type_ids=type_ids,
        coverage_fraction=0.60,
        covered_points=3,
        total_points=5,
    )
    cfg = build_experiment_config(
        cache_dir=str(tmp_path / "cache"),
        range_diagnostics_mode="cached",
        compression_ratio=0.4,
        workload="range",
    )

    first_cache = RangeRuntimeCache()
    first_summary, first_rows = _range_workload_diagnostics(
        "train",
        points,
        boundaries,
        workload,
        {"range": 1.0},
        cfg,
        seed=123,
        runtime_cache=first_cache,
    )
    assert first_summary["range_diagnostics_cache"]["hit"] is False
    assert first_cache.labels is not None
    assert first_cache.labelled_mask is not None
    first_labels = first_cache.labels.clone()

    second_cache = RangeRuntimeCache()
    second_summary, second_rows = _range_workload_diagnostics(
        "train",
        points,
        boundaries,
        workload,
        {"range": 1.0},
        cfg,
        seed=123,
        runtime_cache=second_cache,
    )

    assert second_summary["range_diagnostics_cache"]["hit"] is True
    assert second_rows == first_rows
    assert second_cache.labels is not None
    assert torch.equal(second_cache.labels, first_labels)
    assert second_cache.query_cache is not None


def test_range_workload_diagnostics_populates_runtime_query_cache_masks(tmp_path: Path) -> None:
    points, boundaries = _points_and_boundaries()
    queries = [_range_query(-1.0, 1.0, -1.0, 1.0, -1.0, 2.5)]
    features, type_ids = pad_query_features(queries)
    workload = TypedQueryWorkload(
        query_features=features,
        typed_queries=queries,
        type_ids=type_ids,
        coverage_fraction=0.60,
        covered_points=3,
        total_points=5,
    )
    cfg = build_experiment_config(
        cache_dir=str(tmp_path / "cache"),
        range_diagnostics_mode="full",
        compression_ratio=0.4,
        workload="range",
    )

    runtime_cache = RangeRuntimeCache()
    _summary, _rows = _range_workload_diagnostics(
        "eval",
        points,
        boundaries,
        workload,
        {"range": 1.0},
        cfg,
        seed=123,
        runtime_cache=runtime_cache,
    )

    assert runtime_cache.query_cache is not None
    assert set(runtime_cache.query_cache.support_masks) == {0}
    assert runtime_cache.labels is not None
    assert runtime_cache.labelled_mask is not None
    assert runtime_cache.component_labels is not None


def test_eval_range_label_cache_reuses_tensor_cache(tmp_path: Path) -> None:
    points, boundaries = _points_and_boundaries()
    queries = [_range_query(-1.0, 1.0, -1.0, 1.0, -1.0, 2.5)]
    features, type_ids = pad_query_features(queries)
    workload = TypedQueryWorkload(
        query_features=features,
        typed_queries=queries,
        type_ids=type_ids,
        coverage_fraction=0.60,
        covered_points=3,
        total_points=5,
    )
    cfg = build_experiment_config(
        cache_dir=str(tmp_path / "cache"),
        range_diagnostics_mode="cached",
        compression_ratio=0.4,
        workload="range",
    )

    first_cache = RangeRuntimeCache()
    first = _prepare_range_label_cache(
        cache_label="eval",
        points=points,
        boundaries=boundaries,
        workload=workload,
        workload_map={"range": 1.0},
        config=cfg,
        seed=123,
        runtime_cache=first_cache,
        range_boundary_prior_weight=0.0,
    )
    assert first is not None
    assert first_cache.labels is not None
    assert first_cache.component_labels is not None

    second_cache = RangeRuntimeCache()
    second = _prepare_range_label_cache(
        cache_label="eval",
        points=points,
        boundaries=boundaries,
        workload=workload,
        workload_map={"range": 1.0},
        config=cfg,
        seed=123,
        runtime_cache=second_cache,
        range_boundary_prior_weight=0.0,
    )

    assert second is not None
    assert second_cache.labels is not None
    assert second_cache.component_labels is not None
    assert torch.equal(second_cache.labels, first_cache.labels)
    assert second_cache.labelled_mask is not None


def test_range_label_cache_key_ignores_compression_ratio(tmp_path: Path) -> None:
    points, boundaries = _points_and_boundaries()
    queries = [_range_query(-1.0, 1.0, -1.0, 1.0, -1.0, 2.5)]
    features, type_ids = pad_query_features(queries)
    workload = TypedQueryWorkload(
        query_features=features,
        typed_queries=queries,
        type_ids=type_ids,
        coverage_fraction=0.60,
        covered_points=3,
        total_points=5,
    )
    cache_dir = tmp_path / "cache"
    cfg_40 = build_experiment_config(
        cache_dir=str(cache_dir),
        range_diagnostics_mode="cached",
        compression_ratio=0.4,
        workload="range",
    )
    cfg_10 = build_experiment_config(
        cache_dir=str(cache_dir),
        range_diagnostics_mode="cached",
        compression_ratio=0.1,
        workload="range",
    )

    first_cache = RangeRuntimeCache()
    first = _prepare_range_label_cache(
        cache_label="eval",
        points=points,
        boundaries=boundaries,
        workload=workload,
        workload_map={"range": 1.0},
        config=cfg_40,
        seed=123,
        runtime_cache=first_cache,
        range_boundary_prior_weight=0.0,
    )
    second_cache = RangeRuntimeCache()
    second = _prepare_range_label_cache(
        cache_label="eval",
        points=points,
        boundaries=boundaries,
        workload=workload,
        workload_map={"range": 1.0},
        config=cfg_10,
        seed=123,
        runtime_cache=second_cache,
        range_boundary_prior_weight=0.0,
    )

    assert first is not None
    assert second is not None
    assert second_cache.labels is not None
    assert first_cache.labels is not None
    assert torch.equal(second_cache.labels, first_cache.labels)
    assert len(list((cache_dir / "range_diagnostics").glob("eval-*.pt"))) == 1
