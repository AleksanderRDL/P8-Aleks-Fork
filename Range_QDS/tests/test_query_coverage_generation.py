"""Tests coverage-targeted query generation. See queries/README.md for details."""

from __future__ import annotations

from pathlib import Path

import torch

from data.ais_loader import generate_synthetic_ais_data, load_ais_csv
from experiments.experiment_config import build_experiment_config, derive_seed_bundle
from experiments.experiment_workloads import generate_experiment_workloads
from experiments.workload_cache import generate_typed_query_workload_for_config
from queries.coverage_estimator import best_query_count, estimate_range_coverage, sample_trajectories_by_stride
from queries.query_generator import (
    _dataset_bounds,
    _make_range_query,
    generate_typed_query_workload,
    point_coverage_mask_for_query,
)


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
    assert generation["target_reached_query_count"] is not None
    assert 1 <= generation["target_reached_query_count"] <= 25
    assert generation["coverage_at_target_reached"] >= 0.01
    assert generation["extra_queries_after_target_reached"] == (
        25 - generation["target_reached_query_count"]
    )


def test_coverage_generation_profile_calibrated_mode_keeps_requested_query_floor() -> None:
    """Assert final profiles keep the requested query count after reaching target coverage."""
    trajectories = generate_synthetic_ais_data(n_ships=8, n_points_per_ship=128, seed=1818)

    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=48,
        workload_map={"range": 1.0},
        seed=1818,
        target_coverage=0.05,
        max_queries=300,
        workload_profile_id="range_workload_v1",
        range_max_coverage_overshoot=0.0075,
        range_acceptance_max_attempts=6000,
    )

    assert workload.coverage_fraction is not None
    assert workload.coverage_fraction >= 0.05
    assert workload.generation_diagnostics is not None
    generation = workload.generation_diagnostics["query_generation"]
    assert generation["mode"] == "target_coverage"
    assert generation["query_count_mode"] == "calibrated_to_coverage"
    assert generation["coverage_calibration_mode"] == "profile_sampled_query_count"
    assert generation["minimum_queries"] == 48
    assert generation["stop_reason"] == "target_coverage_reached"
    assert generation["target_reached_query_count"] is not None
    assert generation["coverage_at_target_reached"] is not None
    assert generation["final_query_count"] >= generation["minimum_queries"]
    assert generation["extra_queries_after_target_reached"] >= 0
    assert generation["profile_query_plan"]["requested_queries"] == 300
    assert len(workload.typed_queries) == generation["final_query_count"]


def test_coverage_overshoot_guard_rejects_over_broad_queries() -> None:
    trajectories = generate_synthetic_ais_data(n_ships=3, n_points_per_ship=24, seed=782)

    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=4,
        workload_map={"range": 1.0},
        seed=3,
        target_coverage=0.01,
        max_queries=8,
        range_spatial_fraction=1.0,
        range_time_fraction=1.0,
        range_footprint_jitter=0.0,
        range_acceptance_max_attempts=5,
        range_max_coverage_overshoot=0.01,
    )

    assert workload.coverage_fraction == 0.0
    assert len(workload.typed_queries) == 0
    assert workload.generation_diagnostics is not None
    generation = workload.generation_diagnostics["query_generation"]
    acceptance = workload.generation_diagnostics["range_acceptance"]
    assert generation["coverage_guard_enabled"] is True
    assert abs(float(generation["max_allowed_coverage"]) - 0.02) < 1e-9
    assert generation["stop_reason"] == "range_coverage_guard_exhausted"
    assert generation["target_reached_query_count"] is None
    assert generation["extra_queries_after_target_reached"] is None
    assert acceptance["enabled"] is True
    assert acceptance["exhausted"] is True
    assert acceptance["rejection_reasons"]["coverage_overshoot"] == 5


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


def test_anchor_day_time_domain_clamps_queries_to_anchor_day() -> None:
    points = torch.tensor(
        [
            [0.0, 55.0, 12.0, 1.0],
            [90_000.0, 55.0, 12.0, 1.0],
            [172_800.0, 55.0, 12.0, 1.0],
        ],
        dtype=torch.float32,
    )
    generator = torch.Generator().manual_seed(3)
    bounds = _dataset_bounds(points)
    anchor_mask = torch.tensor([False, True, False])

    dataset_query = _make_range_query(
        points,
        bounds,
        generator,
        anchor_mask=anchor_mask,
        range_spatial_km=1.0,
        range_time_hours=30.0,
        range_footprint_jitter=0.0,
        range_time_domain_mode="dataset",
    )
    anchor_day_query = _make_range_query(
        points,
        bounds,
        generator,
        anchor_mask=anchor_mask,
        range_spatial_km=1.0,
        range_time_hours=30.0,
        range_footprint_jitter=0.0,
        range_time_domain_mode="anchor_day",
    )

    dataset_params = dataset_query["params"]
    anchor_day_params = anchor_day_query["params"]
    assert dataset_params["t_start"] == 0.0
    assert dataset_params["t_end"] == 172_800.0
    assert anchor_day_params["t_start"] == 86_400.0
    assert anchor_day_params["t_end"] == 172_800.0


def test_generated_anchor_day_workload_records_time_domain_mode() -> None:
    trajectories = [
        torch.tensor(
            [
                [0.0, 55.0, 12.0, 1.0],
                [10_000.0, 55.1, 12.1, 1.0],
                [90_000.0, 55.2, 12.2, 1.0],
                [100_000.0, 55.3, 12.3, 1.0],
            ],
            dtype=torch.float32,
        )
    ]

    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=20,
        workload_map={"range": 1.0},
        seed=14,
        range_spatial_km=5.0,
        range_time_hours=20.0,
        range_footprint_jitter=0.0,
        range_time_domain_mode="anchor_day",
    )

    assert workload.generation_diagnostics is not None
    generation = workload.generation_diagnostics["query_generation"]
    assert generation["range_time_domain_mode"] == "anchor_day"
    for query in workload.typed_queries:
        params = query["params"]
        assert params["t_end"] - params["t_start"] <= 86_400.0


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


def test_workload_generation_warns_when_train_or_eval_coverage_overshoots(capsys) -> None:
    trajectories = generate_synthetic_ais_data(n_ships=3, n_points_per_ship=24, seed=782)
    points = torch.cat(trajectories, dim=0)
    boundaries: list[tuple[int, int]] = []
    cursor = 0
    for trajectory in trajectories:
        next_cursor = cursor + int(trajectory.shape[0])
        boundaries.append((cursor, next_cursor))
        cursor = next_cursor
    cfg = build_experiment_config(
        n_queries=4,
        query_coverage=0.01,
        max_queries=8,
        workload="range",
        range_spatial_fraction=1.0,
        range_time_fraction=1.0,
    )

    generate_experiment_workloads(
        config=cfg,
        seeds=derive_seed_bundle(782),
        train_traj=trajectories,
        test_traj=trajectories,
        selection_traj=None,
        train_points=points,
        test_points=points,
        selection_points=None,
        train_boundaries=boundaries,
        test_boundaries=boundaries,
        selection_boundaries=None,
        train_workload_map={"range": 1.0},
        eval_workload_map={"range": 1.0},
    )

    output = capsys.readouterr().out
    assert "WARNING: train_r0 workload remains above requested coverage" in output
    assert "WARNING: eval workload remains above requested coverage" in output


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

    anchor_cfg = build_experiment_config(
        n_queries=6,
        query_coverage=0.10,
        max_queries=20,
        workload="range",
        cache_dir=str(tmp_path / "cache"),
        range_spatial_fraction=0.05,
        range_time_fraction=0.05,
        range_time_domain_mode="anchor_day",
    )
    anchor = generate_typed_query_workload_for_config(
        trajectories=trajectories,
        n_queries=6,
        workload_map={"range": 1.0},
        seed=12,
        config=anchor_cfg,
        cache_label="train",
    )
    anchor_cache = (anchor.generation_diagnostics or {})["workload_cache"]
    assert anchor_cache["hit"] is False
    assert anchor_cache["key"] != first_cache["key"]

    sparse_cfg = build_experiment_config(
        n_queries=6,
        query_coverage=0.10,
        max_queries=20,
        workload="range",
        cache_dir=str(tmp_path / "cache"),
        range_spatial_fraction=0.05,
        range_time_fraction=0.05,
        range_anchor_mode="sparse",
    )
    sparse = generate_typed_query_workload_for_config(
        trajectories=trajectories,
        n_queries=6,
        workload_map={"range": 1.0},
        seed=12,
        config=sparse_cfg,
        cache_label="train",
    )
    sparse_cache = (sparse.generation_diagnostics or {})["workload_cache"]
    assert sparse_cache["hit"] is False
    assert sparse_cache["key"] != first_cache["key"]

    override = generate_typed_query_workload_for_config(
        trajectories=trajectories,
        n_queries=6,
        workload_map={"range": 1.0},
        seed=12,
        config=cfg,
        cache_label="train_sparse_override",
        range_anchor_mode="sparse",
    )
    override_diagnostics = override.generation_diagnostics or {}
    override_cache = override_diagnostics["workload_cache"]
    override_generation = override_diagnostics["query_generation"]
    assert override_generation["range_anchor_mode"] == "sparse"
    assert override_cache["hit"] is False
    assert override_cache["key"] != first_cache["key"]

    guarded_cfg = build_experiment_config(
        n_queries=6,
        query_coverage=0.10,
        max_queries=20,
        workload="range",
        cache_dir=str(tmp_path / "cache"),
        range_spatial_fraction=0.05,
        range_time_fraction=0.05,
        range_max_coverage_overshoot=0.02,
    )
    guarded = generate_typed_query_workload_for_config(
        trajectories=trajectories,
        n_queries=6,
        workload_map={"range": 1.0},
        seed=12,
        config=guarded_cfg,
        cache_label="train",
    )
    guarded_diagnostics = guarded.generation_diagnostics or {}
    guarded_cache = guarded_diagnostics["workload_cache"]
    guarded_generation = guarded_diagnostics["query_generation"]
    assert guarded_generation["range_max_coverage_overshoot"] == 0.02
    assert guarded_generation["coverage_guard_enabled"] is True
    assert guarded_cache["hit"] is False
    assert guarded_cache["key"] != first_cache["key"]


def test_train_workload_replicates_can_cycle_anchor_modes() -> None:
    trajectories = generate_synthetic_ais_data(n_ships=4, n_points_per_ship=32, seed=731)
    points = torch.cat(trajectories, dim=0)
    boundaries: list[tuple[int, int]] = []
    cursor = 0
    for trajectory in trajectories:
        next_cursor = cursor + int(trajectory.shape[0])
        boundaries.append((cursor, next_cursor))
        cursor = next_cursor

    cfg = build_experiment_config(
        n_queries=8,
        query_coverage=0.15,
        max_queries=20,
        workload="range",
        range_spatial_fraction=0.10,
        range_time_fraction=0.10,
        range_anchor_mode="mixed_density",
        range_train_workload_replicates=3,
        range_train_anchor_modes=["mixed_density", "sparse"],
    )

    workloads = generate_experiment_workloads(
        config=cfg,
        seeds=derive_seed_bundle(19),
        train_traj=trajectories,
        test_traj=trajectories,
        selection_traj=None,
        train_points=points,
        test_points=points,
        selection_points=None,
        train_boundaries=boundaries,
        test_boundaries=boundaries,
        selection_boundaries=None,
        train_workload_map={"range": 1.0},
        eval_workload_map={"range": 1.0},
    )

    train_modes = [
        (workload.generation_diagnostics or {})["query_generation"]["range_anchor_mode"]
        for workload in workloads.train_label_workloads
    ]
    eval_mode = (workloads.eval_workload.generation_diagnostics or {})["query_generation"]["range_anchor_mode"]
    assert train_modes == ["mixed_density", "sparse", "mixed_density"]
    assert eval_mode == "mixed_density"


def test_train_workload_replicates_can_cycle_footprint_families() -> None:
    trajectories = generate_synthetic_ais_data(n_ships=4, n_points_per_ship=32, seed=732)
    points = torch.cat(trajectories, dim=0)
    boundaries: list[tuple[int, int]] = []
    cursor = 0
    for trajectory in trajectories:
        next_cursor = cursor + int(trajectory.shape[0])
        boundaries.append((cursor, next_cursor))
        cursor = next_cursor

    cfg = build_experiment_config(
        n_queries=8,
        query_coverage=0.15,
        max_queries=20,
        workload="range",
        range_spatial_km=2.2,
        range_time_hours=5.0,
        range_footprint_jitter=0.0,
        range_time_domain_mode="anchor_day",
        range_train_workload_replicates=4,
        range_train_footprints=["1.1:2.5", "2.2:5.0", "4.4:10.0"],
    )

    workloads = generate_experiment_workloads(
        config=cfg,
        seeds=derive_seed_bundle(20),
        train_traj=trajectories,
        test_traj=trajectories,
        selection_traj=None,
        train_points=points,
        test_points=points,
        selection_points=None,
        train_boundaries=boundaries,
        test_boundaries=boundaries,
        selection_boundaries=None,
        train_workload_map={"range": 1.0},
        eval_workload_map={"range": 1.0},
    )

    train_footprints = [
        (
            (workload.generation_diagnostics or {})["query_generation"]["range_spatial_km"],
            (workload.generation_diagnostics or {})["query_generation"]["range_time_hours"],
        )
        for workload in workloads.train_label_workloads
    ]
    eval_generation = (workloads.eval_workload.generation_diagnostics or {})["query_generation"]

    assert train_footprints == [(1.1, 2.5), (2.2, 5.0), (4.4, 10.0), (1.1, 2.5)]
    assert eval_generation["range_spatial_km"] == 2.2
    assert eval_generation["range_time_hours"] == 5.0


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


def test_range_generation_biases_dense_regions() -> None:
    """Assert range anchors are mostly drawn from dense spatial cells."""
    trajectories = _density_test_trajectories()

    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=80,
        workload_map={"range": 1.0},
        seed=101,
    )
    dense = 0
    for query in workload.typed_queries:
        params = query["params"]
        lat_center = 0.5 * (float(params["lat_min"]) + float(params["lat_max"]))
        lon_center = 0.5 * (float(params["lon_min"]) + float(params["lon_max"]))
        dense += int(_near_dense_region(lat_center, lon_center))

    assert dense / len(workload.typed_queries) >= 0.70


def test_range_anchor_modes_expose_dense_uniform_and_sparse_priors() -> None:
    trajectories = _density_test_trajectories()

    def dense_fraction(anchor_mode: str) -> float:
        workload = generate_typed_query_workload(
            trajectories=trajectories,
            n_queries=160,
            workload_map={"range": 1.0},
            seed=101,
            range_anchor_mode=anchor_mode,
        )
        dense = 0
        for query in workload.typed_queries:
            params = query["params"]
            lat_center = 0.5 * (float(params["lat_min"]) + float(params["lat_max"]))
            lon_center = 0.5 * (float(params["lon_min"]) + float(params["lon_max"]))
            dense += int(_near_dense_region(lat_center, lon_center))
        return dense / len(workload.typed_queries)

    dense_ratio = dense_fraction("dense")
    uniform_ratio = dense_fraction("uniform")
    sparse_ratio = dense_fraction("sparse")

    assert dense_ratio > uniform_ratio
    assert uniform_ratio > sparse_ratio
    assert sparse_ratio < 0.50
