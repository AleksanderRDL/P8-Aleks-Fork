"""Tests coverage-targeted query generation. See src/queries/README.md for details."""

from __future__ import annotations

from pathlib import Path

import torch

from src.data.ais_loader import generate_synthetic_ais_data, load_ais_csv
from src.experiments.experiment_config import build_experiment_config
from src.experiments.workload_cache import generate_typed_query_workload_for_config
from src.queries.coverage_estimator import best_query_count, estimate_range_coverage, sample_trajectories_by_stride
from src.queries.query_generator import generate_typed_query_workload, point_coverage_mask_for_query


def _density_test_trajectories() -> list[torch.Tensor]:
    """Build a point cloud with one intentionally dense spatial region."""
    trajectories: list[torch.Tensor] = []
    for idx in range(120):
        trajectories.append(torch.tensor([[float(idx), 10.0, 10.0, 1.0]], dtype=torch.float32))
    for idx in range(24):
        trajectories.append(torch.tensor([[float(idx), 20.0, 20.0, 1.0]], dtype=torch.float32))
    trajectories.append(torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32))
    trajectories.append(torch.tensor([[1.0, 30.0, 30.0, 1.0]], dtype=torch.float32))
    return trajectories


def _near_dense_region(lat: float, lon: float) -> bool:
    return abs(float(lat) - 10.0) <= 1.0 and abs(float(lon) - 10.0) <= 1.0


def test_query_generation_can_expand_toward_coverage_target() -> None:
    """Assert coverage-targeted query generation may use max_queries to improve coverage."""
    trajectories = generate_synthetic_ais_data(n_ships=6, n_points_per_ship=80, seed=321)

    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=10,
        workload_map={"range": 1.0},
        seed=11,
        target_coverage=0.95,
        max_queries=300,
    )

    assert workload.coverage_fraction is not None
    assert workload.covered_points is not None
    assert workload.total_points == 6 * 80
    assert 10 <= len(workload.typed_queries) <= 300


def test_coverage_generation_keeps_requested_query_count_after_target_is_met() -> None:
    """Assert coverage mode treats n_queries as a minimum, not only a coverage stop hint."""
    trajectories = generate_synthetic_ais_data(n_ships=4, n_points_per_ship=40, seed=222)

    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=25,
        workload_map={"range": 1.0},
        seed=3,
        target_coverage=0.01,
        max_queries=200,
    )

    assert workload.coverage_fraction is not None
    assert workload.coverage_fraction >= 0.01
    assert len(workload.typed_queries) == 25
    assert workload.generation_diagnostics is not None
    generation = workload.generation_diagnostics["query_generation"]
    assert generation["mode"] == "target_coverage"
    assert generation["minimum_queries"] == 25
    assert generation["max_queries"] == 200
    assert generation["final_query_count"] == 25
    assert generation["type_counts"] == {"range": 25}
    assert generation["stop_reason"] == "target_coverage_reached"


def test_smaller_range_fraction_reduces_query_footprint() -> None:
    """Assert range footprint controls let high query counts avoid blanket coverage."""
    trajectories = generate_synthetic_ais_data(n_ships=6, n_points_per_ship=80, seed=456)

    default_workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=80,
        workload_map={"range": 1.0},
        seed=8,
    )
    small_workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=80,
        workload_map={"range": 1.0},
        seed=8,
        range_spatial_fraction=0.02,
        range_time_fraction=0.04,
    )

    assert small_workload.coverage_fraction is not None
    assert default_workload.coverage_fraction is not None
    assert small_workload.coverage_fraction < default_workload.coverage_fraction


def test_absolute_range_controls_are_stable_workload_footprint() -> None:
    trajectories = generate_synthetic_ais_data(n_ships=6, n_points_per_ship=80, seed=456)

    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=20,
        workload_map={"range": 1.0},
        seed=8,
        range_spatial_km=2.2,
        range_time_hours=6.0,
    )

    assert len(workload.typed_queries) == 20
    assert workload.coverage_fraction is not None
    assert 0.0 <= workload.coverage_fraction <= 1.0


def test_range_footprint_jitter_can_be_disabled() -> None:
    trajectories = generate_synthetic_ais_data(n_ships=4, n_points_per_ship=40, seed=456)

    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=5,
        workload_map={"range": 1.0},
        seed=8,
        range_spatial_km=2.2,
        range_time_hours=6.0,
        range_footprint_jitter=0.0,
    )

    for query in workload.typed_queries:
        params = query["params"]
        assert params["t_end"] - params["t_start"] <= 12.0 * 3600.0


def test_sampled_range_coverage_estimator_returns_reproducible_rows() -> None:
    trajectories = generate_synthetic_ais_data(n_ships=6, n_points_per_ship=30, seed=456)

    sampled = sample_trajectories_by_stride(trajectories, 2)
    rows_a = estimate_range_coverage(
        trajectories=trajectories,
        query_counts=[4, 8],
        seeds=[5],
        sample_stride=2,
        target_coverage=0.20,
        range_spatial_km=2.2,
        range_time_hours=6.0,
        range_footprint_jitter=0.0,
    )
    rows_b = estimate_range_coverage(
        trajectories=trajectories,
        query_counts=[4, 8],
        seeds=[5],
        sample_stride=2,
        target_coverage=0.20,
        range_spatial_km=2.2,
        range_time_hours=6.0,
        range_footprint_jitter=0.0,
    )

    assert len(sampled) == 3
    assert [row.to_dict() for row in rows_a] == [row.to_dict() for row in rows_b]
    assert {row.query_count for row in rows_a} == {4, 8}
    assert best_query_count(rows_a, 0.20).query_count in {4, 8}


def test_sampled_range_coverage_estimator_works_on_loaded_cleaned_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "cleaned.csv"
    lines = ["MMSI,# Timestamp,Latitude,Longitude,SOG,COG"]
    for mmsi, lat0, lon0 in ((100, 55.0, 12.0), (200, 55.4, 12.4)):
        for idx in range(6):
            lines.append(f"{mmsi},{idx * 600},{lat0 + idx * 0.01},{lon0 + idx * 0.01},8.0,90.0")
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    trajectories = load_ais_csv(str(csv_path), min_points_per_segment=4)

    rows = estimate_range_coverage(
        trajectories=trajectories,
        source=str(csv_path),
        query_counts=[2, 4],
        seeds=[7],
        sample_stride=1,
        target_coverage=0.20,
        range_spatial_km=5.0,
        range_time_hours=1.0,
        range_footprint_jitter=0.0,
    )

    assert len(rows) == 2
    assert {row.source for row in rows} == {str(csv_path)}
    assert {row.query_count for row in rows} == {2, 4}
    assert all(row.sampled_trajectories == 2 for row in rows)
    assert all(row.sampled_points == 12 for row in rows)
    assert all(0.0 <= row.coverage_fraction <= 1.0 for row in rows)


def test_configured_workload_expands_to_max_queries_when_target_needs_more_queries() -> None:
    """Assert coverage-targeted config treats n_queries as a minimum and max_queries as the cap."""
    trajectories = generate_synthetic_ais_data(n_ships=5, n_points_per_ship=60, seed=457)
    cfg = build_experiment_config(
        n_queries=4,
        query_coverage=1.0,
        max_queries=12,
        workload="range",
        range_spatial_fraction=0.01,
        range_time_fraction=0.01,
    )

    workload = generate_typed_query_workload_for_config(
        trajectories=trajectories,
        n_queries=4,
        workload_map={"range": 1.0},
        seed=12,
        config=cfg,
    )

    assert len(workload.typed_queries) == 12
    assert float(workload.coverage_fraction or 0.0) < 1.0
    assert workload.generation_diagnostics is not None
    generation = workload.generation_diagnostics["query_generation"]
    assert generation["minimum_queries"] == 4
    assert generation["max_queries"] == 12
    assert generation["final_query_count"] == 12
    assert generation["type_counts"] == {"range": 12}
    assert generation["stop_reason"] == "max_queries_reached"


def test_configured_workload_uses_persistent_workload_cache(tmp_path: Path) -> None:
    trajectories = generate_synthetic_ais_data(n_ships=3, n_points_per_ship=24, seed=462)
    cfg = build_experiment_config(
        n_queries=6,
        query_coverage=0.10,
        max_queries=20,
        workload="range",
        cache_dir=str(tmp_path / "cache"),
        range_spatial_fraction=0.05,
        range_time_fraction=0.05,
    )

    first = generate_typed_query_workload_for_config(
        trajectories=trajectories,
        n_queries=6,
        workload_map={"range": 1.0},
        seed=12,
        config=cfg,
        cache_label="train",
    )
    second = generate_typed_query_workload_for_config(
        trajectories=trajectories,
        n_queries=6,
        workload_map={"range": 1.0},
        seed=12,
        config=cfg,
        cache_label="train",
    )

    first_cache = (first.generation_diagnostics or {})["workload_cache"]
    second_cache = (second.generation_diagnostics or {})["workload_cache"]
    assert first_cache["hit"] is False
    assert second_cache["hit"] is True
    assert Path(second_cache["path"]).exists()
    assert first.typed_queries == second.typed_queries
    assert torch.equal(first.query_features, second.query_features)


def test_coverage_generation_allows_overlapping_query_hits() -> None:
    """Assert coverage-targeted generation can cover the same point more than once."""
    trajectories = generate_synthetic_ais_data(n_ships=5, n_points_per_ship=50, seed=777)
    points = torch.cat(trajectories, dim=0)

    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=10,
        workload_map={"range": 1.0},
        seed=4,
        target_coverage=0.60,
        max_queries=200,
    )

    coverage_counts = torch.zeros((points.shape[0],), dtype=torch.long)
    for query in workload.typed_queries:
        coverage_counts += point_coverage_mask_for_query(points, query).long()

    assert 10 <= len(workload.typed_queries) <= 200
    assert bool((coverage_counts >= 2).any().item())


def test_query_generation_accepts_percent_coverage() -> None:
    """Assert percent-style coverage arguments are normalized."""
    trajectories = generate_synthetic_ais_data(n_ships=4, n_points_per_ship=60, seed=123)

    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=10,
        workload_map={"range": 1.0},
        seed=22,
        target_coverage=30,
        max_queries=250,
    )

    assert workload.coverage_fraction is not None
    assert 10 <= len(workload.typed_queries) <= 250


def test_range_and_knn_generation_biases_dense_regions() -> None:
    """Assert range/kNN anchors are mostly drawn from dense spatial cells."""
    trajectories = _density_test_trajectories()

    range_workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=80,
        workload_map={"range": 1.0},
        seed=101,
    )
    range_dense = 0
    for query in range_workload.typed_queries:
        params = query["params"]
        lat_center = 0.5 * (float(params["lat_min"]) + float(params["lat_max"]))
        lon_center = 0.5 * (float(params["lon_min"]) + float(params["lon_max"]))
        range_dense += int(_near_dense_region(lat_center, lon_center))

    knn_workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=80,
        workload_map={"knn": 1.0},
        seed=202,
    )
    knn_dense = sum(
        int(_near_dense_region(query["params"]["lat"], query["params"]["lon"]))
        for query in knn_workload.typed_queries
    )

    assert range_dense / len(range_workload.typed_queries) >= 0.70
    assert knn_dense / len(knn_workload.typed_queries) >= 0.70


def test_knn_generation_uses_configured_k() -> None:
    trajectories = _density_test_trajectories()

    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=12,
        workload_map={"knn": 1.0},
        seed=404,
        knn_k=12,
    )

    assert {int(query["params"]["k"]) for query in workload.typed_queries} == {12}


def test_coverage_generation_uses_density_biased_knn_anchors() -> None:
    """Assert dynamic coverage mode still applies the dense-region anchor sampler."""
    trajectories = _density_test_trajectories()

    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=10,
        workload_map={"knn": 1.0},
        seed=303,
        target_coverage=0.80,
        max_queries=120,
    )
    dense = sum(
        int(_near_dense_region(query["params"]["lat"], query["params"]["lon"]))
        for query in workload.typed_queries
    )

    assert workload.coverage_fraction is not None
    assert 10 <= len(workload.typed_queries) <= 120
    assert dense / len(workload.typed_queries) >= 0.70
