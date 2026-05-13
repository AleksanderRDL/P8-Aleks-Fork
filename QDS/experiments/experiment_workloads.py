"""Workload map resolution and query workload generation for experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from experiments.experiment_config import ExperimentConfig, SeedBundle
from experiments.workload_cache import (
    coverage_name,
    generate_typed_query_workload_for_config,
    workload_cache_name,
)
from queries.workload import TypedQueryWorkload


@dataclass
class ExperimentWorkloads:
    """Train, eval, and optional checkpoint-selection query workloads."""

    train_workload: TypedQueryWorkload
    eval_workload: TypedQueryWorkload
    selection_workload: TypedQueryWorkload | None


def workload_name(workload_map: dict[str, float]) -> str:
    """Build compact string name for a pure workload map."""
    return ",".join(
        f"{query_type}={weight:.1f}"
        for query_type, weight in sorted(workload_map.items())
    )


def _normalized_coverage_target(value: float | None) -> float | None:
    """Normalize coverage target for pipeline warnings."""
    if value is None:
        return None
    target = float(value)
    return target / 100.0 if target > 1.0 else target


def validation_query_count(config: ExperimentConfig) -> int:
    """Use the same minimum query count for validation and final eval workloads."""
    return max(1, int(config.query.n_queries))


def generate_experiment_workloads(
    *,
    config: ExperimentConfig,
    seeds: SeedBundle,
    train_traj: list[torch.Tensor],
    test_traj: list[torch.Tensor],
    selection_traj: list[torch.Tensor] | None,
    train_points: torch.Tensor,
    test_points: torch.Tensor,
    selection_points: torch.Tensor | None,
    train_boundaries: list[tuple[int, int]],
    test_boundaries: list[tuple[int, int]],
    selection_boundaries: list[tuple[int, int]] | None,
    train_workload_map: dict[str, float],
    eval_workload_map: dict[str, float],
) -> ExperimentWorkloads:
    """Generate train, eval, and optional checkpoint-selection query workloads."""
    knn_front_load = int(train_workload_map.get("knn", 0.0) * config.query.n_queries)
    train_workload = generate_typed_query_workload_for_config(
        trajectories=train_traj,
        n_queries=config.query.n_queries,
        workload_map=train_workload_map,
        seed=seeds.train_query_seed,
        front_load_knn=knn_front_load,
        config=config,
        points=train_points,
        boundaries=train_boundaries,
        cache_label="train",
    )
    eval_workload = generate_typed_query_workload_for_config(
        trajectories=test_traj,
        n_queries=config.query.n_queries,
        workload_map=eval_workload_map,
        seed=seeds.eval_query_seed,
        config=config,
        points=test_points,
        boundaries=test_boundaries,
        cache_label="eval",
    )
    selection_workload = None
    if selection_traj:
        selection_workload = generate_typed_query_workload_for_config(
            trajectories=selection_traj,
            n_queries=validation_query_count(config),
            workload_map=eval_workload_map,
            seed=seeds.eval_query_seed + 17,
            config=config,
            points=selection_points,
            boundaries=selection_boundaries,
            cache_label="selection",
        )
    _print_workload_summary("train", train_workload)
    _print_workload_summary("eval", eval_workload)
    if selection_workload is not None:
        _print_workload_summary("selection", selection_workload)
    _print_coverage_warnings(config, train_workload, eval_workload, selection_workload)
    return ExperimentWorkloads(
        train_workload=train_workload,
        eval_workload=eval_workload,
        selection_workload=selection_workload,
    )


def _print_workload_summary(label: str, workload: TypedQueryWorkload) -> None:
    print(
        f"  {label}_workload={len(workload.typed_queries)} queries  "
        f"coverage={coverage_name(workload)}  cache={workload_cache_name(workload)}",
        flush=True,
    )


def _print_coverage_warnings(
    config: ExperimentConfig,
    train_workload: TypedQueryWorkload,
    eval_workload: TypedQueryWorkload,
    selection_workload: TypedQueryWorkload | None,
) -> None:
    target = _normalized_coverage_target(config.query.target_coverage)
    if target is None:
        return
    workloads_to_check = [("train", train_workload), ("eval", eval_workload)]
    if selection_workload is not None:
        workloads_to_check.append(("selection", selection_workload))
    for label, workload in workloads_to_check:
        coverage = float(workload.coverage_fraction or 0.0)
        if coverage + 1e-9 < target:
            print(
                f"  WARNING: {label} workload stopped below requested coverage "
                f"({coverage:.2%} < {target:.2%}); raise --max_queries "
                "or query footprint to cover more points.",
                flush=True,
            )
        elif label == "selection" and coverage > target + 0.05:
            print(
                f"  WARNING: {label} workload remains above requested coverage "
                f"({coverage:.2%} > {target:.2%}); lower --n_queries or query footprint.",
                flush=True,
            )


def _workload_keyword_to_map(keyword: str | None) -> dict[str, float] | None:
    """Translate a --workload keyword to a concrete pure workload map, or return None."""
    if not keyword:
        return None
    k = keyword.strip().lower()
    if k in {"mixed", "local_mixed", "global_mixed"}:
        raise ValueError(
            f"workload='{k}' is no longer supported for model runs; use one pure type: "
            "range, knn, similarity, or clustering."
        )
    if k in {"range", "knn", "similarity", "clustering"}:
        return {k: 1.0}
    return None


def resolve_workload_maps(workload_keyword: str | None = None) -> tuple[dict[str, float], dict[str, float]]:
    """Return identical pure train/eval workload maps for one model run."""
    keyword_map = _workload_keyword_to_map(workload_keyword)
    workload_map = keyword_map if keyword_map is not None else {"range": 1.0}
    return workload_map, workload_map
