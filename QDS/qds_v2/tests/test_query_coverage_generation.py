"""Tests coverage-targeted query generation. See src/queries/README.md for details."""

from __future__ import annotations

from src.data.ais_loader import generate_synthetic_ais_data
from src.queries.query_generator import generate_typed_query_workload


def test_query_generation_reaches_target_coverage() -> None:
    """Assert dynamic query generation stops only after reaching the requested point coverage."""
    trajectories = generate_synthetic_ais_data(n_ships=6, n_points_per_ship=80, seed=321)

    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=10,
        workload_mix={"range": 1.0},
        seed=11,
        target_coverage=0.30,
        max_queries=300,
    )

    assert workload.coverage_fraction is not None
    assert workload.coverage_fraction >= 0.30
    assert workload.covered_points is not None
    assert workload.total_points == 6 * 80
    assert len(workload.typed_queries) <= 300


def test_query_generation_accepts_percent_coverage() -> None:
    """Assert percent-style coverage arguments are normalized."""
    trajectories = generate_synthetic_ais_data(n_ships=4, n_points_per_ship=60, seed=123)

    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=10,
        workload_mix={"range": 1.0},
        seed=22,
        target_coverage=30,
        max_queries=250,
    )

    assert workload.coverage_fraction is not None
    assert workload.coverage_fraction >= 0.30
