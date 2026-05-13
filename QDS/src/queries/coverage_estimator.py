"""Sampled query-coverage estimation for benchmark workload calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from src.queries.query_generator import (
    DEFAULT_RANGE_FOOTPRINT_JITTER,
    DEFAULT_RANGE_SPATIAL_FRACTION,
    DEFAULT_RANGE_TIME_FRACTION,
    generate_typed_query_workload,
)


@dataclass(frozen=True)
class RangeCoverageEstimate:
    """One sampled range-workload coverage estimate."""

    source: str | None
    query_count: int
    seed: int
    sample_stride: int
    sampled_trajectories: int
    sampled_points: int
    coverage_fraction: float
    covered_points: int
    total_points: int
    target_coverage: float | None
    range_spatial_km: float | None
    range_time_hours: float | None
    range_spatial_fraction: float
    range_time_fraction: float
    range_footprint_jitter: float

    def to_dict(self) -> dict[str, int | float | str | None]:
        """Serialize estimate for JSON artifacts."""
        return asdict(self)


def sample_trajectories_by_stride(trajectories: list[torch.Tensor], sample_stride: int) -> list[torch.Tensor]:
    """Return every Nth trajectory, preserving whole sampled trajectories."""
    stride = max(1, int(sample_stride))
    return [trajectory for trajectory_idx, trajectory in enumerate(trajectories) if trajectory_idx % stride == 0]


def estimate_range_coverage(
    *,
    trajectories: list[torch.Tensor],
    query_counts: list[int],
    seeds: list[int],
    source: str | None = None,
    sample_stride: int = 20,
    target_coverage: float | None = None,
    range_spatial_fraction: float = DEFAULT_RANGE_SPATIAL_FRACTION,
    range_time_fraction: float = DEFAULT_RANGE_TIME_FRACTION,
    range_spatial_km: float | None = None,
    range_time_hours: float | None = None,
    range_footprint_jitter: float = DEFAULT_RANGE_FOOTPRINT_JITTER,
) -> list[RangeCoverageEstimate]:
    """Estimate range workload point coverage on a deterministic trajectory sample."""
    sampled_trajectories = sample_trajectories_by_stride(trajectories, sample_stride)
    if not sampled_trajectories:
        raise ValueError("Cannot estimate coverage for an empty trajectory sample.")
    sampled_points = int(sum(int(trajectory.shape[0]) for trajectory in sampled_trajectories))
    estimates: list[RangeCoverageEstimate] = []
    for query_count in query_counts:
        query_count_int = int(query_count)
        if query_count_int <= 0:
            raise ValueError("query_counts must all be positive.")
        for seed in seeds:
            workload = generate_typed_query_workload(
                trajectories=sampled_trajectories,
                n_queries=query_count_int,
                workload_map={"range": 1.0},
                seed=int(seed),
                target_coverage=target_coverage,
                range_spatial_fraction=range_spatial_fraction,
                range_time_fraction=range_time_fraction,
                range_spatial_km=range_spatial_km,
                range_time_hours=range_time_hours,
                range_footprint_jitter=range_footprint_jitter,
            )
            estimates.append(
                RangeCoverageEstimate(
                    source=source,
                    query_count=query_count_int,
                    seed=int(seed),
                    sample_stride=max(1, int(sample_stride)),
                    sampled_trajectories=len(sampled_trajectories),
                    sampled_points=sampled_points,
                    coverage_fraction=float(workload.coverage_fraction or 0.0),
                    covered_points=int(workload.covered_points or 0),
                    total_points=int(workload.total_points or 0),
                    target_coverage=None if target_coverage is None else float(target_coverage),
                    range_spatial_km=range_spatial_km,
                    range_time_hours=range_time_hours,
                    range_spatial_fraction=float(range_spatial_fraction),
                    range_time_fraction=float(range_time_fraction),
                    range_footprint_jitter=float(range_footprint_jitter),
                )
            )
    return estimates


def best_query_count(estimates: list[RangeCoverageEstimate], target_coverage: float) -> RangeCoverageEstimate:
    """Return the estimate closest to a desired coverage fraction."""
    if not estimates:
        raise ValueError("estimates must not be empty.")
    target = float(target_coverage)
    return min(estimates, key=lambda estimate: abs(float(estimate.coverage_fraction) - target))
