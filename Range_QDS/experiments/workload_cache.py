"""Persistent typed-workload generation cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from experiments.experiment_config import ExperimentConfig
from queries.query_generator import generate_typed_query_workload
from queries.workload import TypedQueryWorkload

WORKLOAD_CACHE_SCHEMA_VERSION = 2


def coverage_name(workload: TypedQueryWorkload) -> str:
    """Format workload point-coverage metadata for logs."""
    if workload.coverage_fraction is None:
        return "unknown"
    covered = workload.covered_points if workload.covered_points is not None else 0
    total = workload.total_points if workload.total_points is not None else 0
    return f"{100.0 * workload.coverage_fraction:.2f}% ({covered}/{total})"


def trajectory_boundaries_for_cache(trajectories: list[torch.Tensor]) -> list[tuple[int, int]]:
    """Return flattened trajectory boundaries without constructing a dataset object."""
    boundaries: list[tuple[int, int]] = []
    cursor = 0
    for trajectory in trajectories:
        end = cursor + int(trajectory.shape[0])
        boundaries.append((cursor, end))
        cursor = end
    return boundaries


def tensor_cache_digest(tensor: torch.Tensor) -> str:
    """Return an exact digest for workload and diagnostics cache data identity checks."""
    value = tensor.detach().cpu().contiguous()
    hasher = hashlib.sha256()
    hasher.update(str(value.dtype).encode("utf-8"))
    hasher.update(json.dumps(list(value.shape), separators=(",", ":")).encode("utf-8"))
    hasher.update(value.numpy().tobytes())
    return hasher.hexdigest()


def _workload_cache_root(config: ExperimentConfig) -> Path | None:
    """Return persistent workload-cache root when data caching is configured."""
    if not config.data.cache_dir:
        return None
    return Path(config.data.cache_dir) / "workloads"


def _workload_cache_payload(
    *,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    n_queries: int,
    workload_map: dict[str, float],
    seed: int,
    config: ExperimentConfig,
    range_anchor_mode: str | None = None,
    range_spatial_km: float | None = None,
    range_time_hours: float | None = None,
) -> dict[str, Any]:
    """Build the canonical workload-cache key payload."""
    query_config = config.query
    effective_range_anchor_mode = range_anchor_mode or query_config.range_anchor_mode
    effective_range_spatial_km = query_config.range_spatial_km if range_spatial_km is None else range_spatial_km
    effective_range_time_hours = query_config.range_time_hours if range_time_hours is None else range_time_hours
    return {
        "schema_version": WORKLOAD_CACHE_SCHEMA_VERSION,
        "points_sha256": tensor_cache_digest(points),
        "boundaries_sha256": hashlib.sha256(
            json.dumps(boundaries, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "n_queries": int(n_queries),
        "seed": int(seed),
        "workload_map": {key: float(workload_map[key]) for key in sorted(workload_map)},
        "target_coverage": query_config.target_coverage,
        "max_queries": query_config.max_queries,
        "range_spatial_fraction": query_config.range_spatial_fraction,
        "range_time_fraction": query_config.range_time_fraction,
        "range_spatial_km": effective_range_spatial_km,
        "range_time_hours": effective_range_time_hours,
        "range_footprint_jitter": query_config.range_footprint_jitter,
        "range_time_domain_mode": query_config.range_time_domain_mode,
        "range_anchor_mode": effective_range_anchor_mode,
        "range_min_point_hits": query_config.range_min_point_hits,
        "range_max_point_hit_fraction": query_config.range_max_point_hit_fraction,
        "range_min_trajectory_hits": query_config.range_min_trajectory_hits,
        "range_max_trajectory_hit_fraction": query_config.range_max_trajectory_hit_fraction,
        "range_max_box_volume_fraction": query_config.range_max_box_volume_fraction,
        "range_duplicate_iou_threshold": query_config.range_duplicate_iou_threshold,
        "range_acceptance_max_attempts": query_config.range_acceptance_max_attempts,
        "range_max_coverage_overshoot": query_config.range_max_coverage_overshoot,
        "workload_profile_id": query_config.workload_profile_id,
        "coverage_calibration_mode": query_config.coverage_calibration_mode,
    }


def _workload_cache_key(payload: dict[str, Any]) -> str:
    """Return a stable cache key for a workload payload."""
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _attach_workload_cache_info(
    workload: TypedQueryWorkload,
    *,
    hit: bool,
    path: Path,
    key: str,
) -> TypedQueryWorkload:
    """Attach cache hit/miss metadata to workload diagnostics."""
    diagnostics = dict(workload.generation_diagnostics or {})
    diagnostics["workload_cache"] = {
        "hit": bool(hit),
        "path": str(path),
        "key": key,
    }
    workload.generation_diagnostics = diagnostics
    return workload


def generate_typed_query_workload_for_config(
    *,
    trajectories: list[torch.Tensor],
    n_queries: int,
    workload_map: dict[str, float],
    seed: int,
    config: ExperimentConfig,
    points: torch.Tensor | None = None,
    boundaries: list[tuple[int, int]] | None = None,
    cache_label: str | None = None,
    range_anchor_mode: str | None = None,
    range_spatial_km: float | None = None,
    range_time_hours: float | None = None,
) -> TypedQueryWorkload:
    """Generate a typed workload from config, expanding to target coverage up to max_queries."""
    query_config = config.query
    effective_range_anchor_mode = range_anchor_mode or query_config.range_anchor_mode
    effective_range_spatial_km = query_config.range_spatial_km if range_spatial_km is None else range_spatial_km
    effective_range_time_hours = query_config.range_time_hours if range_time_hours is None else range_time_hours
    points_for_cache = points if points is not None else torch.cat(trajectories, dim=0)
    boundaries_for_cache = boundaries if boundaries is not None else trajectory_boundaries_for_cache(trajectories)
    cache_root = _workload_cache_root(config)
    cache_path: Path | None = None
    cache_key: str | None = None
    if cache_root is not None:
        payload = _workload_cache_payload(
            points=points_for_cache,
            boundaries=boundaries_for_cache,
            n_queries=n_queries,
            workload_map=workload_map,
            seed=seed,
            config=config,
            range_anchor_mode=effective_range_anchor_mode,
            range_spatial_km=effective_range_spatial_km,
            range_time_hours=effective_range_time_hours,
        )
        cache_key = _workload_cache_key(payload)
        cache_name = f"{cache_label or 'workload'}-{cache_key[:16]}.json"
        cache_path = cache_root / cache_name
        if cache_path.exists() and not config.data.refresh_cache:
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("schema_version") == WORKLOAD_CACHE_SCHEMA_VERSION and cached.get("key") == cache_key:
                    workload = TypedQueryWorkload.from_dict(cached["workload"])
                    return _attach_workload_cache_info(workload, hit=True, path=cache_path, key=cache_key)
            except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
                print(f"  WARNING: ignoring unreadable workload cache {cache_path}: {exc}", flush=True)

    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=n_queries,
        workload_map=workload_map,
        seed=seed,
        target_coverage=query_config.target_coverage,
        max_queries=query_config.max_queries,
        range_spatial_fraction=query_config.range_spatial_fraction,
        range_time_fraction=query_config.range_time_fraction,
        range_spatial_km=effective_range_spatial_km,
        range_time_hours=effective_range_time_hours,
        range_footprint_jitter=query_config.range_footprint_jitter,
        range_time_domain_mode=query_config.range_time_domain_mode,
        range_anchor_mode=effective_range_anchor_mode,
        range_min_point_hits=query_config.range_min_point_hits,
        range_max_point_hit_fraction=query_config.range_max_point_hit_fraction,
        range_min_trajectory_hits=query_config.range_min_trajectory_hits,
        range_max_trajectory_hit_fraction=query_config.range_max_trajectory_hit_fraction,
        range_max_box_volume_fraction=query_config.range_max_box_volume_fraction,
        range_duplicate_iou_threshold=query_config.range_duplicate_iou_threshold,
        range_acceptance_max_attempts=query_config.range_acceptance_max_attempts,
        range_max_coverage_overshoot=query_config.range_max_coverage_overshoot,
        workload_profile_id=query_config.workload_profile_id,
        coverage_calibration_mode=query_config.coverage_calibration_mode,
    )
    if cache_path is not None and cache_key is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_payload = {
                "schema_version": WORKLOAD_CACHE_SCHEMA_VERSION,
                "key": cache_key,
                "workload": workload.to_dict(),
            }
            cache_path.write_text(json.dumps(cache_payload, indent=2), encoding="utf-8")
            workload = _attach_workload_cache_info(workload, hit=False, path=cache_path, key=cache_key)
        except OSError as exc:
            print(f"  WARNING: could not write workload cache {cache_path}: {exc}", flush=True)
    return workload


def workload_cache_name(workload: TypedQueryWorkload) -> str:
    """Format workload-cache hit/miss metadata for logs."""
    cache_info = (workload.generation_diagnostics or {}).get("workload_cache")
    if not isinstance(cache_info, dict):
        return "disabled"
    hit = "hit" if cache_info.get("hit") else "miss"
    key = str(cache_info.get("key", ""))[:12]
    return f"{hit} ({key})"
