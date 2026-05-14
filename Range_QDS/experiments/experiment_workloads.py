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
from queries.query_generator import RANGE_ANCHOR_MODES
from queries.workload import TypedQueryWorkload


@dataclass
class ExperimentWorkloads:
    """Train, eval, and optional checkpoint-selection query workloads."""

    train_workload: TypedQueryWorkload
    train_label_workloads: list[TypedQueryWorkload]
    train_label_workload_seeds: list[int]
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


def _normalized_coverage_tolerance(value: float | None, *, default: float) -> float:
    """Normalize coverage tolerances for pipeline warnings."""
    if value is None:
        return float(default)
    tolerance = float(value)
    if tolerance > 1.0 and tolerance <= 100.0:
        tolerance = tolerance / 100.0
    return max(0.0, tolerance)


def validation_query_count(config: ExperimentConfig) -> int:
    """Use the same minimum query count for validation and final eval workloads."""
    return max(1, int(config.query.n_queries))


def _training_anchor_modes(config: ExperimentConfig) -> list[str]:
    """Return anchor-prior modes to cycle across train label workloads."""
    raw_modes = getattr(config.query, "range_train_anchor_modes", []) or []
    modes = [str(mode).strip().lower() for mode in raw_modes if str(mode).strip()]
    if not modes:
        modes = [str(getattr(config.query, "range_anchor_mode", "mixed_density")).strip().lower()]
    invalid = [mode for mode in modes if mode not in RANGE_ANCHOR_MODES]
    if invalid:
        raise ValueError(f"range_train_anchor_modes must contain only {RANGE_ANCHOR_MODES}; got {invalid}.")
    return modes


def _parse_train_footprint(spec: str) -> tuple[float, float]:
    """Parse one train-only footprint family as spatial_km:time_hours."""
    normalized = str(spec).strip().lower().replace("x", ":")
    parts = [part.strip() for part in normalized.split(":")]
    if len(parts) != 2:
        raise ValueError("range_train_footprints entries must use spatial_km:time_hours.")
    spatial_km = float(parts[0])
    time_hours = float(parts[1])
    if spatial_km <= 0.0 or time_hours <= 0.0:
        raise ValueError("range_train_footprints entries must be positive.")
    return spatial_km, time_hours


def _training_footprints(config: ExperimentConfig) -> list[tuple[float | None, float | None]]:
    """Return train-only range footprints to cycle across train label workloads."""
    raw_specs = getattr(config.query, "range_train_footprints", []) or []
    specs = [str(spec).strip() for spec in raw_specs if str(spec).strip()]
    if not specs:
        return [(None, None)]
    return [_parse_train_footprint(spec) for spec in specs]


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
    train_anchor_modes = _training_anchor_modes(config)
    train_footprints = _training_footprints(config)
    train_anchor_mode = train_anchor_modes[0]
    train_spatial_km, train_time_hours = train_footprints[0]
    train_workload = generate_typed_query_workload_for_config(
        trajectories=train_traj,
        n_queries=config.query.n_queries,
        workload_map=train_workload_map,
        seed=seeds.train_query_seed,
        config=config,
        points=train_points,
        boundaries=train_boundaries,
        cache_label="train",
        range_anchor_mode=train_anchor_mode,
        range_spatial_km=train_spatial_km,
        range_time_hours=train_time_hours,
    )
    train_label_workloads = [train_workload]
    train_label_workload_seeds = [int(seeds.train_query_seed)]
    train_replicates = max(1, int(getattr(config.query, "range_train_workload_replicates", 1) or 1))
    for replicate_index in range(1, train_replicates):
        replicate_seed = int(seeds.train_query_seed + 1009 * replicate_index)
        replicate_anchor_mode = train_anchor_modes[replicate_index % len(train_anchor_modes)]
        replicate_spatial_km, replicate_time_hours = train_footprints[replicate_index % len(train_footprints)]
        replicate_workload = generate_typed_query_workload_for_config(
            trajectories=train_traj,
            n_queries=config.query.n_queries,
            workload_map=train_workload_map,
            seed=replicate_seed,
            config=config,
            points=train_points,
            boundaries=train_boundaries,
            cache_label=f"train_r{replicate_index}",
            range_anchor_mode=replicate_anchor_mode,
            range_spatial_km=replicate_spatial_km,
            range_time_hours=replicate_time_hours,
        )
        train_label_workloads.append(replicate_workload)
        train_label_workload_seeds.append(replicate_seed)
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
    for replicate_index, replicate_workload in enumerate(train_label_workloads[1:], start=1):
        _print_workload_summary(f"train_r{replicate_index}", replicate_workload)
    _print_workload_summary("eval", eval_workload)
    if selection_workload is not None:
        _print_workload_summary("selection", selection_workload)
    _print_coverage_warnings(config, train_label_workloads, eval_workload, selection_workload)
    return ExperimentWorkloads(
        train_workload=train_workload,
        train_label_workloads=train_label_workloads,
        train_label_workload_seeds=train_label_workload_seeds,
        eval_workload=eval_workload,
        selection_workload=selection_workload,
    )


def _print_workload_summary(label: str, workload: TypedQueryWorkload) -> None:
    query_generation = (workload.generation_diagnostics or {}).get("query_generation") or {}
    anchor_mode = query_generation.get("range_anchor_mode", "unknown")
    spatial_km = query_generation.get("range_spatial_km")
    time_hours = query_generation.get("range_time_hours")
    footprint = f"{spatial_km:g}km/{time_hours:g}h" if spatial_km and time_hours else "fractional"
    print(
        f"  {label}_workload={len(workload.typed_queries)} queries  "
        f"coverage={coverage_name(workload)}  anchor_mode={anchor_mode}  footprint={footprint}  "
        f"cache={workload_cache_name(workload)}",
        flush=True,
    )


def _print_coverage_warnings(
    config: ExperimentConfig,
    train_workloads: list[TypedQueryWorkload],
    eval_workload: TypedQueryWorkload,
    selection_workload: TypedQueryWorkload | None,
) -> None:
    target = _normalized_coverage_target(config.query.target_coverage)
    if target is None:
        return
    overshoot = getattr(config.query, "range_max_coverage_overshoot", None)
    upper_tolerance = _normalized_coverage_tolerance(overshoot, default=0.05)
    upper_limit = target + upper_tolerance
    workloads_to_check = [(f"train_r{idx}", workload) for idx, workload in enumerate(train_workloads)]
    workloads_to_check.append(("eval", eval_workload))
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
        elif coverage > upper_limit + 1e-9:
            print(
                f"  WARNING: {label} workload remains above requested coverage "
                f"({coverage:.2%} > {upper_limit:.2%}); lower --n_queries or query footprint.",
                flush=True,
            )


def _workload_keyword_to_map(keyword: str | None) -> dict[str, float] | None:
    """Translate a --workload keyword to a concrete pure workload map, or return None."""
    if not keyword:
        return None
    k = keyword.strip().lower()
    if k in {"mixed", "local_mixed", "global_mixed"}:
        raise ValueError(f"workload='{k}' is no longer supported; only range workloads are supported.")
    if k == "range":
        return {k: 1.0}
    raise ValueError(f"workload='{k}' is no longer supported; only range workloads are supported.")


def resolve_workload_maps(workload_keyword: str | None = None) -> tuple[dict[str, float], dict[str, float]]:
    """Return identical pure train/eval workload maps for one model run."""
    keyword_map = _workload_keyword_to_map(workload_keyword)
    workload_map = keyword_map if keyword_map is not None else {"range": 1.0}
    return workload_map, workload_map
