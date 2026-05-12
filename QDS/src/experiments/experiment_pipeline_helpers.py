"""Experiment orchestration helpers for training and evaluation runs. See src/experiments/README.md for details."""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@contextmanager
def _phase(name: str):
    """Log a named phase with wall-clock timing."""
    print(f"[{name}] starting...", flush=True)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        print(f"[{name}] done in {dt:.2f}s", flush=True)

from src.data.trajectory_dataset import TrajectoryDataset
from src.evaluation.baselines import (
    DouglasPeuckerMethod,
    Method,
    MLQDSMethod,
    OracleMethod,
    ScoreHybridMethod,
    UniformTemporalMethod,
)
from src.evaluation.evaluate_methods import (
    EvaluationQueryCache,
    evaluate_method,
    print_geometric_distortion_table,
    print_method_comparison_table,
    print_range_usefulness_table,
    print_shift_table,
    score_retained_mask,
)
from src.evaluation.metrics import MethodEvaluation
from src.experiments.experiment_config import ExperimentConfig, TypedQueryWorkload, derive_seed_bundle
from src.experiments.geojson_writers import report_trajectory_length_loss, write_queries_geojson, write_simplified_csv
from src.queries.query_generator import generate_typed_query_workload
from src.queries.query_types import single_workload_type
from src.queries.workload_diagnostics import compute_range_label_diagnostics, compute_range_workload_diagnostics
from src.simplification.mlqds_scoring import workload_type_head
from src.training.importance_labels import compute_typed_importance_labels
from src.training.train_model import train_model
from src.training.training_pipeline import ModelArtifacts, save_checkpoint
from src.experiments.torch_runtime import (
    amp_runtime_snapshot,
    cuda_memory_snapshot,
    reset_cuda_peak_memory_stats,
    torch_runtime_snapshot,
)


@dataclass
class ExperimentOutputs:
    """Experiment run output payload. See src/experiments/README.md for details."""

    matched_table: str
    shift_table: str
    metrics_dump: dict
    geometric_table: str = ""
    range_usefulness_table: str = ""
    range_objective_audit_table: str = ""


@dataclass
class RangeRuntimeCache:
    """Non-serialized tensors/caches reused across range diagnostics, training, and evaluation."""

    labels: torch.Tensor | None = None
    labelled_mask: torch.Tensor | None = None
    query_cache: EvaluationQueryCache | None = None


WORKLOAD_CACHE_SCHEMA_VERSION = 1
RANGE_DIAGNOSTICS_CACHE_SCHEMA_VERSION = 1


def split_trajectories(
    trajectories: list[torch.Tensor],
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Deterministically split trajectories at trajectory level. See src/experiments/README.md for details."""
    n = len(trajectories)
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g).tolist()

    n_train = max(1, int(n * train_fraction))
    n_val = max(1, int(n * val_fraction)) if n - n_train > 1 else 0
    n_test = max(1, n - n_train - n_val)
    if n_train + n_val + n_test > n:
        n_test = n - n_train - n_val

    train = [trajectories[i] for i in perm[:n_train]]
    val = [trajectories[i] for i in perm[n_train : n_train + n_val]]
    test = [trajectories[i] for i in perm[n_train + n_val :]]
    if not test:
        test = val if val else train
    return train, val, test


def _workload_name(workload_map: dict[str, float]) -> str:
    """Build compact string name for a pure workload map."""
    return ",".join(f"{k}={v:.1f}" for k, v in sorted(workload_map.items()))


def _coverage_name(workload: TypedQueryWorkload) -> str:
    """Format workload point-coverage metadata for logs."""
    if workload.coverage_fraction is None:
        return "unknown"
    covered = workload.covered_points if workload.covered_points is not None else 0
    total = workload.total_points if workload.total_points is not None else 0
    return f"{100.0 * workload.coverage_fraction:.2f}% ({covered}/{total})"


def _normalized_coverage_target(value: float | None) -> float | None:
    """Normalize coverage target for pipeline warnings."""
    if value is None:
        return None
    target = float(value)
    return target / 100.0 if target > 1.0 else target


def _validation_query_count(config: ExperimentConfig) -> int:
    """Use the same minimum query count for validation and final eval workloads."""
    return max(1, int(config.query.n_queries))


def _trajectory_boundaries_for_cache(trajectories: list[torch.Tensor]) -> list[tuple[int, int]]:
    """Return flattened trajectory boundaries without constructing a dataset object."""
    boundaries: list[tuple[int, int]] = []
    cursor = 0
    for trajectory in trajectories:
        end = cursor + int(trajectory.shape[0])
        boundaries.append((cursor, end))
        cursor = end
    return boundaries


def _tensor_cache_digest(tensor: torch.Tensor) -> str:
    """Return an exact digest for workload-cache data identity checks."""
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
    front_load_knn: int,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Build the canonical workload-cache key payload."""
    query_config = config.query
    return {
        "schema_version": WORKLOAD_CACHE_SCHEMA_VERSION,
        "points_sha256": _tensor_cache_digest(points),
        "boundaries_sha256": hashlib.sha256(
            json.dumps(boundaries, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "n_queries": int(n_queries),
        "seed": int(seed),
        "front_load_knn": int(front_load_knn),
        "workload_map": {key: float(workload_map[key]) for key in sorted(workload_map)},
        "target_coverage": query_config.target_coverage,
        "max_queries": query_config.max_queries,
        "range_spatial_fraction": query_config.range_spatial_fraction,
        "range_time_fraction": query_config.range_time_fraction,
        "range_spatial_km": query_config.range_spatial_km,
        "range_time_hours": query_config.range_time_hours,
        "range_footprint_jitter": query_config.range_footprint_jitter,
        "knn_k": query_config.knn_k,
        "range_min_point_hits": query_config.range_min_point_hits,
        "range_max_point_hit_fraction": query_config.range_max_point_hit_fraction,
        "range_min_trajectory_hits": query_config.range_min_trajectory_hits,
        "range_max_trajectory_hit_fraction": query_config.range_max_trajectory_hit_fraction,
        "range_max_box_volume_fraction": query_config.range_max_box_volume_fraction,
        "range_duplicate_iou_threshold": query_config.range_duplicate_iou_threshold,
        "range_acceptance_max_attempts": query_config.range_acceptance_max_attempts,
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


def _generate_typed_query_workload_for_config(
    *,
    trajectories: list[torch.Tensor],
    n_queries: int,
    workload_map: dict[str, float],
    seed: int,
    config: ExperimentConfig,
    front_load_knn: int = 0,
    points: torch.Tensor | None = None,
    boundaries: list[tuple[int, int]] | None = None,
    cache_label: str | None = None,
) -> TypedQueryWorkload:
    """Generate a typed workload from config, expanding to target coverage up to max_queries."""
    query_config = config.query
    points_for_cache = points if points is not None else torch.cat(trajectories, dim=0)
    boundaries_for_cache = boundaries if boundaries is not None else _trajectory_boundaries_for_cache(trajectories)
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
            front_load_knn=front_load_knn,
            config=config,
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
        range_spatial_km=query_config.range_spatial_km,
        range_time_hours=query_config.range_time_hours,
        range_footprint_jitter=query_config.range_footprint_jitter,
        knn_k=query_config.knn_k,
        front_load_knn=front_load_knn,
        range_min_point_hits=query_config.range_min_point_hits,
        range_max_point_hit_fraction=query_config.range_max_point_hit_fraction,
        range_min_trajectory_hits=query_config.range_min_trajectory_hits,
        range_max_trajectory_hit_fraction=query_config.range_max_trajectory_hit_fraction,
        range_max_box_volume_fraction=query_config.range_max_box_volume_fraction,
        range_duplicate_iou_threshold=query_config.range_duplicate_iou_threshold,
        range_acceptance_max_attempts=query_config.range_acceptance_max_attempts,
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


def _workload_cache_name(workload: TypedQueryWorkload) -> str:
    """Format workload-cache hit/miss metadata for logs."""
    cache_info = (workload.generation_diagnostics or {}).get("workload_cache")
    if not isinstance(cache_info, dict):
        return "disabled"
    hit = "hit" if cache_info.get("hit") else "miss"
    key = str(cache_info.get("key", ""))[:12]
    return f"{hit} ({key})"


def _range_diagnostics_cache_root(config: ExperimentConfig) -> Path | None:
    """Return persistent range-diagnostics cache root when enabled."""
    if str(getattr(config.data, "range_diagnostics_mode", "full")).lower() != "cached":
        return None
    if not config.data.cache_dir:
        return None
    return Path(config.data.cache_dir) / "range_diagnostics"


def _typed_queries_digest(typed_queries: list[dict[str, Any]]) -> str:
    """Return a stable digest for a typed query list."""
    encoded = json.dumps(typed_queries, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _range_diagnostics_cache_payload(
    *,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    workload: TypedQueryWorkload,
    workload_map: dict[str, float],
    config: ExperimentConfig,
    seed: int,
) -> dict[str, Any]:
    """Build the canonical cache key payload for range workload diagnostics."""
    return {
        "schema_version": RANGE_DIAGNOSTICS_CACHE_SCHEMA_VERSION,
        "points_sha256": _tensor_cache_digest(points),
        "boundaries_sha256": hashlib.sha256(
            json.dumps(boundaries, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "typed_queries_sha256": _typed_queries_digest(workload.typed_queries),
        "workload_map": {key: float(workload_map[key]) for key in sorted(workload_map)},
        "seed": int(seed),
        "compression_ratio": float(config.model.compression_ratio),
        "range_label_mode": str(getattr(config.model, "range_label_mode", "usefulness")),
        "range_boundary_prior_weight": float(getattr(config.model, "range_boundary_prior_weight", 0.0)),
        "max_point_hit_fraction": config.query.range_max_point_hit_fraction,
        "max_trajectory_hit_fraction": config.query.range_max_trajectory_hit_fraction,
        "max_box_volume_fraction": config.query.range_max_box_volume_fraction,
        "duplicate_iou_threshold": _range_diagnostic_duplicate_threshold(config),
    }


def _range_diagnostics_cache_key(payload: dict[str, Any]) -> str:
    """Return a stable cache key for range diagnostics."""
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _range_diagnostics_cache_paths(config: ExperimentConfig, label: str, key: str) -> tuple[Path, Path] | None:
    """Return JSON and tensor paths for a range diagnostics cache entry."""
    cache_root = _range_diagnostics_cache_root(config)
    if cache_root is None:
        return None
    stem = f"{label}-{key[:16]}"
    return cache_root / f"{stem}.json", cache_root / f"{stem}.pt"


def _load_range_diagnostics_cache(
    *,
    config: ExperimentConfig,
    label: str,
    key: str,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    scored_queries: list[dict[str, Any]],
    runtime_cache: RangeRuntimeCache | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    """Load cached range diagnostics and runtime tensors when the entry is complete."""
    paths = _range_diagnostics_cache_paths(config, label, key)
    if paths is None or config.data.refresh_cache:
        return None
    json_path, tensor_path = paths
    if not json_path.exists() or not tensor_path.exists():
        return None
    try:
        cached = json.loads(json_path.read_text(encoding="utf-8"))
        if cached.get("schema_version") != RANGE_DIAGNOSTICS_CACHE_SCHEMA_VERSION or cached.get("key") != key:
            return None
        tensors = torch.load(tensor_path, map_location="cpu", weights_only=True)
        if not isinstance(tensors, dict):
            return None
        labels = tensors.get("labels")
        labelled_mask = tensors.get("labelled_mask")
        if not isinstance(labels, torch.Tensor) or not isinstance(labelled_mask, torch.Tensor):
            return None
        if runtime_cache is not None:
            runtime_cache.labels = labels
            runtime_cache.labelled_mask = labelled_mask
            runtime_cache.query_cache = EvaluationQueryCache.for_workload(points, boundaries, scored_queries)
        summary = cached["summary"]
        rows = cached["rows"]
        if not isinstance(summary, dict) or not isinstance(rows, list):
            return None
        summary = dict(summary)
        cache_info = dict(summary.get("range_diagnostics_cache") or {})
        cache_info.update({"hit": True, "path": str(json_path), "tensor_path": str(tensor_path), "key": key})
        summary["range_diagnostics_cache"] = cache_info
        return summary, rows
    except (OSError, KeyError, TypeError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"  WARNING: ignoring unreadable range diagnostics cache {json_path}: {exc}", flush=True)
        return None


def _write_range_diagnostics_cache(
    *,
    config: ExperimentConfig,
    label: str,
    key: str,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    runtime_cache: RangeRuntimeCache | None,
) -> None:
    """Persist range diagnostics plus label tensors for reuse in repeated sweeps."""
    paths = _range_diagnostics_cache_paths(config, label, key)
    if paths is None or runtime_cache is None or runtime_cache.labels is None or runtime_cache.labelled_mask is None:
        return
    json_path, tensor_path = paths
    try:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"labels": runtime_cache.labels.cpu(), "labelled_mask": runtime_cache.labelled_mask.cpu()},
            tensor_path,
        )
        cache_summary = dict(summary)
        cache_info = dict(cache_summary.get("range_diagnostics_cache") or {})
        cache_info.update({"hit": False, "path": str(json_path), "tensor_path": str(tensor_path), "key": key})
        cache_summary["range_diagnostics_cache"] = cache_info
        json_path.write_text(
            json.dumps(
                {
                    "schema_version": RANGE_DIAGNOSTICS_CACHE_SCHEMA_VERSION,
                    "key": key,
                    "summary": cache_summary,
                    "rows": rows,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        summary["range_diagnostics_cache"] = cache_info
    except OSError as exc:
        print(f"  WARNING: could not write range diagnostics cache {json_path}: {exc}", flush=True)


def _range_diagnostic_duplicate_threshold(config: ExperimentConfig) -> float | None:
    """Use explicit duplicate threshold for diagnostics, or a diagnostic-only default."""
    threshold = config.query.range_duplicate_iou_threshold
    return 0.85 if threshold is None else threshold


def _range_only_queries(typed_queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only range queries from a typed workload."""
    return [query for query in typed_queries if str(query.get("type", "")).lower() == "range"]


def _range_signal_diagnostics(
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    range_queries: list[dict[str, Any]],
    workload_map: dict[str, float],
    compression_ratio: float,
    seed: int,
    range_label_mode: str = "usefulness",
    range_boundary_prior_weight: float = 0.0,
    runtime_cache: RangeRuntimeCache | None = None,
    cache_typed_queries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute label, Oracle, and baseline diagnostics for range workloads."""
    if not range_queries:
        return {
            "range_query_count": 0,
            "range_label_mode": str(range_label_mode),
            "labels": compute_range_label_diagnostics(
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0, 4), dtype=torch.bool),
            ),
            "methods": {},
            "best_baseline": None,
            "best_baseline_range_f1": 0.0,
            "oracle_range_f1": 0.0,
            "oracle_gap_over_best_baseline": 0.0,
        }

    labels, labelled_mask = compute_typed_importance_labels(
        points=points,
        boundaries=boundaries,
        typed_queries=range_queries,
        seed=seed,
        range_label_mode=range_label_mode,
        range_boundary_prior_weight=range_boundary_prior_weight,
    )
    if runtime_cache is not None:
        runtime_cache.labels = labels
        runtime_cache.labelled_mask = labelled_mask
    label_diagnostics = compute_range_label_diagnostics(labels, labelled_mask)
    oracle_labels = labels
    methods: list[Method] = [
        UniformTemporalMethod(),
        DouglasPeuckerMethod(),
        OracleMethod(labels=oracle_labels, workload_type="range"),
    ]
    method_scores: dict[str, dict[str, float]] = {}
    scored_queries = cache_typed_queries if cache_typed_queries is not None else range_queries
    query_cache = EvaluationQueryCache.for_workload(
        points,
        boundaries,
        scored_queries,
    )
    if runtime_cache is not None:
        runtime_cache.query_cache = query_cache
    for method in methods:
        retained_mask = method.simplify(points, boundaries, compression_ratio)
        aggregate, per_type, _, _ = score_retained_mask(
            points=points,
            boundaries=boundaries,
            retained_mask=retained_mask,
            typed_queries=scored_queries,
            workload_map={"range": 1.0},
            query_cache=query_cache,
        )
        method_scores[method.name] = {
            "aggregate_f1": float(aggregate),
            "range_f1": float(per_type.get("range", 0.0)),
        }

    baseline_names = ["uniform", "DouglasPeucker"]
    best_baseline = max(baseline_names, key=lambda name: method_scores.get(name, {}).get("range_f1", 0.0))
    best_baseline_range_f1 = float(method_scores[best_baseline]["range_f1"])
    oracle_range_f1 = float(method_scores.get("Oracle", {}).get("range_f1", 0.0))
    normalized_map = sum(float(v) for v in workload_map.values())
    range_weight = float(workload_map.get("range", 0.0)) / normalized_map if normalized_map > 0.0 else 0.0
    return {
        "range_query_count": int(len(range_queries)),
        "range_workload_weight": float(range_weight),
        "range_boundary_prior_weight": float(range_boundary_prior_weight),
        "range_boundary_prior_enabled": bool(float(range_boundary_prior_weight) > 0.0),
        "range_label_mode": str(range_label_mode),
        "oracle_label_mode": str(range_label_mode),
        "oracle_kind": "additive_label_greedy",
        "oracle_exact_optimum": False,
        "labels": label_diagnostics,
        "methods": method_scores,
        "best_baseline": best_baseline,
        "best_baseline_range_f1": best_baseline_range_f1,
        "oracle_range_f1": oracle_range_f1,
        "oracle_gap_over_best_baseline": float(oracle_range_f1 - best_baseline_range_f1),
    }


def _range_workload_diagnostics(
    label: str,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    workload: TypedQueryWorkload,
    workload_map: dict[str, float],
    config: ExperimentConfig,
    seed: int,
    runtime_cache: RangeRuntimeCache | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build summary and JSONL rows for one workload."""
    range_queries = _range_only_queries(workload.typed_queries)
    is_pure_range_workload = len(range_queries) == len(workload.typed_queries)
    scored_queries = workload.typed_queries if is_pure_range_workload else range_queries
    cache_payload = _range_diagnostics_cache_payload(
        points=points,
        boundaries=boundaries,
        workload=workload,
        workload_map=workload_map,
        config=config,
        seed=seed,
    )
    cache_key = _range_diagnostics_cache_key(cache_payload)
    cached = _load_range_diagnostics_cache(
        config=config,
        label=label,
        key=cache_key,
        points=points,
        boundaries=boundaries,
        scored_queries=scored_queries,
        runtime_cache=runtime_cache,
    )
    if cached is not None:
        return cached

    workload_diagnostics = compute_range_workload_diagnostics(
        points=points,
        boundaries=boundaries,
        typed_queries=workload.typed_queries,
        max_point_hit_fraction=config.query.range_max_point_hit_fraction,
        max_trajectory_hit_fraction=config.query.range_max_trajectory_hit_fraction,
        max_box_volume_fraction=config.query.range_max_box_volume_fraction,
        duplicate_iou_threshold=_range_diagnostic_duplicate_threshold(config),
        coverage_fraction=workload.coverage_fraction,
    )
    signal = _range_signal_diagnostics(
        points=points,
        boundaries=boundaries,
        range_queries=range_queries,
        workload_map=workload_map,
        compression_ratio=config.model.compression_ratio,
        seed=seed,
        range_label_mode=str(getattr(config.model, "range_label_mode", "usefulness")),
        range_boundary_prior_weight=float(getattr(config.model, "range_boundary_prior_weight", 0.0)),
        runtime_cache=runtime_cache,
        cache_typed_queries=workload.typed_queries if is_pure_range_workload else None,
    )
    summary = {
        "range": workload_diagnostics["summary"],
        "range_signal": signal,
        "generation": workload.generation_diagnostics or {},
    }
    rows = [{"workload": label, **row} for row in workload_diagnostics["queries"]]
    _write_range_diagnostics_cache(
        config=config,
        label=label,
        key=cache_key,
        summary=summary,
        rows=rows,
        runtime_cache=runtime_cache,
    )
    return summary, rows


def _compact_range_workload_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Extract comparable workload-shape fields from verbose diagnostics."""
    range_summary = summary.get("range", {}) if isinstance(summary, dict) else {}
    range_signal = summary.get("range_signal", {}) if isinstance(summary, dict) else {}
    fields = (
        "range_query_count",
        "coverage_fraction",
        "empty_query_rate",
        "too_broad_query_rate",
        "near_duplicate_query_rate",
        "point_hit_count_p50",
        "point_hit_count_p90",
        "trajectory_hit_count_p50",
        "trajectory_hit_count_p90",
        "point_hit_fraction_p50",
        "point_hit_fraction_p90",
        "trajectory_hit_fraction_p50",
        "trajectory_hit_fraction_p90",
        "box_volume_fraction_p50",
        "box_volume_fraction_p90",
    )
    compact = {field: range_summary.get(field) for field in fields}
    compact["oracle_gap_over_best_baseline"] = range_signal.get("oracle_gap_over_best_baseline")
    compact["best_baseline"] = range_signal.get("best_baseline")
    return compact


def _range_workload_distribution_comparison(summaries: dict[str, Any]) -> dict[str, Any]:
    """Compare train/selection workload shape against final eval workload shape."""
    compact = {label: _compact_range_workload_summary(summary) for label, summary in summaries.items()}
    eval_summary = compact.get("eval", {})
    numeric_fields = (
        "range_query_count",
        "coverage_fraction",
        "empty_query_rate",
        "too_broad_query_rate",
        "near_duplicate_query_rate",
        "point_hit_count_p50",
        "trajectory_hit_count_p50",
        "point_hit_fraction_p50",
        "trajectory_hit_fraction_p50",
        "box_volume_fraction_p50",
        "oracle_gap_over_best_baseline",
    )
    deltas: dict[str, dict[str, float | None]] = {}
    for label, row in compact.items():
        if label == "eval":
            continue
        label_delta: dict[str, float | None] = {}
        for field in numeric_fields:
            left = row.get(field)
            right = eval_summary.get(field)
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                label_delta[f"{field}_minus_eval"] = float(left) - float(right)
            else:
                label_delta[f"{field}_minus_eval"] = None
        deltas[label] = label_delta
    return {
        "summaries": compact,
        "deltas_vs_eval": deltas,
    }


def _range_audit_ratios(config: ExperimentConfig) -> list[float]:
    """Return configured multi-budget range-audit ratios, deduped and sorted."""
    raw = getattr(config.model, "range_audit_compression_ratios", []) or []
    return sorted({float(value) for value in raw if 0.0 < float(value) <= 1.0})


def _evaluation_metrics_payload(metrics: MethodEvaluation) -> dict[str, Any]:
    """Serialize method metrics with explicit range aliases."""
    return {
        "aggregate_f1": metrics.aggregate_f1,
        "per_type_f1": metrics.per_type_f1,
        "compression_ratio": metrics.compression_ratio,
        "latency_ms": metrics.latency_ms,
        "avg_retained_point_gap": metrics.avg_retained_point_gap,
        "avg_retained_point_gap_norm": metrics.avg_retained_point_gap_norm,
        "max_retained_point_gap": metrics.max_retained_point_gap,
        "geometric_distortion": metrics.geometric_distortion,
        "avg_length_preserved": metrics.avg_length_preserved,
        "avg_length_loss": metrics.avg_length_loss,
        "combined_query_shape_score": metrics.combined_query_shape_score,
        "range_point_f1": metrics.range_point_f1,
        "pure_range_f1": metrics.range_point_f1,
        "range_ship_f1": metrics.range_ship_f1,
        "range_ship_coverage": metrics.range_ship_coverage,
        "range_entry_exit_f1": metrics.range_entry_exit_f1,
        "range_boundary_f1": metrics.range_entry_exit_f1,
        "range_temporal_coverage": metrics.range_temporal_coverage,
        "range_gap_coverage": metrics.range_gap_coverage,
        "range_turn_coverage": metrics.range_turn_coverage,
        "range_shape_score": metrics.range_shape_score,
        "range_usefulness_score": metrics.range_usefulness_score,
        "range_usefulness_schema_version": metrics.range_usefulness_schema_version,
        "range_audit": metrics.range_audit,
    }


def run_experiment_pipeline(
    config: ExperimentConfig,
    trajectories: list[torch.Tensor],
    results_dir: str,
    save_model: str | None = None,
    save_queries_dir: str | None = None,
    save_simplified_dir: str | None = None,
    trajectory_mmsis: list[int] | None = None,
    validation_trajectories: list[torch.Tensor] | None = None,
    eval_trajectories: list[torch.Tensor] | None = None,
    eval_trajectory_mmsis: list[int] | None = None,
    data_audit: dict[str, Any] | None = None,
) -> ExperimentOutputs:
    """Run training, matched evaluation, and shifted evaluation tables. See src/experiments/README.md for details."""
    pipeline_t0 = time.perf_counter()
    train_workload_map, eval_workload_map = resolve_workload_maps(config.query.workload)
    if eval_trajectories is None:
        print(
            f"[pipeline] {len(trajectories)} trajectories, workload={_workload_name(eval_workload_map)}",
            flush=True,
        )
    else:
        validation_part = (
            f", validation={len(validation_trajectories)} trajectories"
            if validation_trajectories is not None
            else ""
        )
        print(
            f"[pipeline] train={len(trajectories)} trajectories{validation_part}, "
            f"eval={len(eval_trajectories)} trajectories, "
            f"workload={_workload_name(eval_workload_map)}",
            flush=True,
        )

    seeds = derive_seed_bundle(config.data.seed)
    selection_metric = str(getattr(config.model, "checkpoint_selection_metric", "f1")).lower()
    f1_diag_every = int(getattr(config.model, "f1_diagnostic_every", 0) or 0)
    needs_validation_f1 = selection_metric in {"f1", "uniform_gap"} or f1_diag_every > 0
    with _phase("split"):
        selection_traj: list[torch.Tensor] | None = None
        if eval_trajectories is None:
            # Reproduce split_trajectories' permutation here so we can align the
            # MMSI list with the test split (the helper itself doesn't carry ids).
            n = len(trajectories)
            g = torch.Generator().manual_seed(int(seeds.split_seed))
            perm = torch.randperm(n, generator=g).tolist()
            n_train = max(1, int(n * config.data.train_fraction))
            n_val = max(1, int(n * config.data.val_fraction)) if n - n_train > 1 else 0
            train_traj = [trajectories[i] for i in perm[:n_train]]
            _val_traj = [trajectories[i] for i in perm[n_train : n_train + n_val]]
            test_traj = [trajectories[i] for i in perm[n_train + n_val :]]
            if not test_traj:
                test_traj = _val_traj if _val_traj else train_traj
            selection_traj = _val_traj if needs_validation_f1 and _val_traj else None
            if trajectory_mmsis is not None and len(trajectory_mmsis) == n:
                train_mmsis = [trajectory_mmsis[i] for i in perm[:n_train]]
                test_mmsis = [trajectory_mmsis[i] for i in perm[n_train + n_val :]]
                if not test_mmsis:
                    test_mmsis = [trajectory_mmsis[i] for i in perm[n_train : n_train + n_val]] or \
                                 [trajectory_mmsis[i] for i in perm[:n_train]]
            else:
                train_mmsis = None
                test_mmsis = None
            print(f"  split mode=single dataset  train={len(train_traj)}  test={len(test_traj)}", flush=True)
        else:
            train_traj = trajectories
            test_traj = eval_trajectories
            train_mmsis = trajectory_mmsis
            test_mmsis = eval_trajectory_mmsis
            if validation_trajectories is not None:
                selection_traj = validation_trajectories if needs_validation_f1 else None
            elif needs_validation_f1:
                n = len(train_traj)
                g = torch.Generator().manual_seed(int(seeds.split_seed))
                perm = torch.randperm(n, generator=g).tolist()
                n_val = max(1, int(n * config.data.val_fraction)) if n > 1 else 0
                val_idx = set(perm[:n_val])
                selection_traj = [traj for idx, traj in enumerate(train_traj) if idx in val_idx]
                train_traj = [traj for idx, traj in enumerate(train_traj) if idx not in val_idx]
                if train_mmsis is not None and len(train_mmsis) == n:
                    train_mmsis = [mmsi for idx, mmsi in enumerate(train_mmsis) if idx not in val_idx]
            print(f"  split mode=separate CSVs  train={len(train_traj)}  eval={len(test_traj)}", flush=True)
        if selection_traj:
            print(f"  checkpoint-selection validation={len(selection_traj)} trajectories", flush=True)

    with _phase("build-datasets"):
        train_ds = TrajectoryDataset(train_traj)
        test_ds = TrajectoryDataset(test_traj)
        selection_ds = TrajectoryDataset(selection_traj) if selection_traj else None
        train_points = train_ds.get_all_points()
        test_points = test_ds.get_all_points()
        selection_points = selection_ds.get_all_points() if selection_ds is not None else None
        train_boundaries = train_ds.get_trajectory_boundaries()
        test_boundaries = test_ds.get_trajectory_boundaries()
        selection_boundaries = selection_ds.get_trajectory_boundaries() if selection_ds is not None else None

    with _phase("generate-workloads"):
        # Front-load all kNN queries before proportional scheduling for training
        # so kNN always gets its full quota even if n_queries is small.
        knn_front_load = int(train_workload_map.get("knn", 0.0) * config.query.n_queries)
        train_workload = _generate_typed_query_workload_for_config(
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
        eval_workload = _generate_typed_query_workload_for_config(
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
            selection_workload = _generate_typed_query_workload_for_config(
                trajectories=selection_traj,
                n_queries=_validation_query_count(config),
                workload_map=eval_workload_map,
                seed=seeds.eval_query_seed + 17,
                config=config,
                points=selection_points,
                boundaries=selection_boundaries,
                cache_label="selection",
            )
        print(
            f"  train_workload={len(train_workload.typed_queries)} queries  "
            f"coverage={_coverage_name(train_workload)}  cache={_workload_cache_name(train_workload)}",
            flush=True,
        )
        print(
            f"  eval_workload={len(eval_workload.typed_queries)} queries  "
            f"coverage={_coverage_name(eval_workload)}  cache={_workload_cache_name(eval_workload)}",
            flush=True,
        )
        if selection_workload is not None:
            print(
                f"  selection_workload={len(selection_workload.typed_queries)} queries  "
                f"coverage={_coverage_name(selection_workload)}  cache={_workload_cache_name(selection_workload)}",
                flush=True,
            )
        target = _normalized_coverage_target(config.query.target_coverage)
        if target is not None:
            workloads_to_check = [("train", train_workload), ("eval", eval_workload)]
            if selection_workload is not None:
                workloads_to_check.append(("selection", selection_workload))
            for label, workload in workloads_to_check:
                coverage = float(workload.coverage_fraction or 0.0)
                if coverage + 1e-9 < target:
                    print(
                        f"  WARNING: {label} workload stopped below requested coverage "
                        f"({coverage:.2%} < {target:.2%}); raise --max_queries or query footprint to cover more points.",
                        flush=True,
                    )
                elif label == "selection" and coverage > target + 0.05:
                    print(
                        f"  WARNING: {label} workload remains above requested coverage "
                        f"({coverage:.2%} > {target:.2%}); lower --n_queries or query footprint.",
                        flush=True,
                    )

    range_diagnostics_summary: dict[str, Any] = {}
    range_diagnostics_rows: list[dict[str, Any]] = []
    range_runtime_caches = {
        "train": RangeRuntimeCache(),
        "eval": RangeRuntimeCache(),
        "selection": RangeRuntimeCache(),
    }
    with _phase("range-diagnostics"):
        train_summary, train_rows = _range_workload_diagnostics(
            "train",
            train_points,
            train_boundaries,
            train_workload,
            train_workload_map,
            config,
            seeds.train_query_seed,
            range_runtime_caches["train"],
        )
        eval_summary, eval_rows = _range_workload_diagnostics(
            "eval",
            test_points,
            test_boundaries,
            eval_workload,
            eval_workload_map,
            config,
            seeds.eval_query_seed,
            range_runtime_caches["eval"],
        )
        range_diagnostics_summary["train"] = train_summary
        range_diagnostics_summary["eval"] = eval_summary
        range_diagnostics_rows.extend(train_rows)
        range_diagnostics_rows.extend(eval_rows)
        if selection_workload is not None and selection_points is not None and selection_boundaries is not None:
            selection_summary, selection_rows = _range_workload_diagnostics(
                "selection",
                selection_points,
                selection_boundaries,
                selection_workload,
                eval_workload_map,
                config,
                seeds.eval_query_seed + 17,
                range_runtime_caches["selection"],
            )
            range_diagnostics_summary["selection"] = selection_summary
            range_diagnostics_rows.extend(selection_rows)
        for label, summary in range_diagnostics_summary.items():
            range_summary = summary["range"]
            signal = summary["range_signal"]
            diagnostics_cache = summary.get("range_diagnostics_cache") or {}
            cache_text = "disabled"
            if isinstance(diagnostics_cache, dict) and diagnostics_cache:
                cache_text = "hit" if diagnostics_cache.get("hit") else "miss"
            print(
                f"  {label}: range_queries={range_summary['range_query_count']}  "
                f"empty={range_summary['empty_query_rate']:.2%}  "
                f"broad={range_summary['too_broad_query_rate']:.2%}  "
                f"duplicates={range_summary['near_duplicate_query_rate']:.2%}  "
                f"oracle_gap={signal['oracle_gap_over_best_baseline']:+.6f}  "
                f"diag_cache={cache_text}",
                flush=True,
            )
        workload_distribution_comparison = _range_workload_distribution_comparison(range_diagnostics_summary)
        for label, delta in workload_distribution_comparison["deltas_vs_eval"].items():
            coverage_delta = delta.get("coverage_fraction_minus_eval")
            point_p50_delta = delta.get("point_hit_count_p50_minus_eval")
            traj_p50_delta = delta.get("trajectory_hit_count_p50_minus_eval")
            coverage_text = f"{coverage_delta:+.4f}" if isinstance(coverage_delta, (int, float)) else "n/a"
            point_text = f"{point_p50_delta:+.2f}" if isinstance(point_p50_delta, (int, float)) else "n/a"
            traj_text = f"{traj_p50_delta:+.2f}" if isinstance(traj_p50_delta, (int, float)) else "n/a"
            print(
                f"  {label}_vs_eval: "
                f"coverage_delta={coverage_text}  "
                f"point_hit_p50_delta={point_text}  "
                f"trajectory_hit_p50_delta={traj_text}",
                flush=True,
            )

    if save_queries_dir:
        with _phase("write-queries-geojson"):
            write_queries_geojson(save_queries_dir, eval_workload.typed_queries)

    reset_cuda_peak_memory_stats()
    train_cache = range_runtime_caches["train"]
    train_labels: tuple[torch.Tensor, torch.Tensor] | None = None
    if (
        train_cache.labels is not None
        and train_cache.labelled_mask is not None
        and len(_range_only_queries(train_workload.typed_queries)) == len(train_workload.typed_queries)
    ):
        train_labels = (train_cache.labels, train_cache.labelled_mask)
    selection_cache = range_runtime_caches["selection"]
    selection_query_cache: EvaluationQueryCache | None = None
    if (
        selection_workload is not None
        and selection_cache.query_cache is not None
        and len(_range_only_queries(selection_workload.typed_queries)) == len(selection_workload.typed_queries)
    ):
        selection_query_cache = selection_cache.query_cache
    with _phase(f"train-model ({config.model.epochs} epochs)"):
        trained = train_model(
            train_trajectories=train_traj,
            train_boundaries=train_boundaries,
            workload=train_workload,
            model_config=config.model,
            seed=seeds.torch_seed,
            train_workload_map=train_workload_map,
            validation_trajectories=selection_traj,
            validation_boundaries=selection_boundaries,
            validation_workload=selection_workload,
            validation_workload_map=eval_workload_map if selection_workload is not None else None,
            precomputed_labels=train_labels,
            validation_points=selection_points,
            precomputed_validation_query_cache=selection_query_cache,
        )
    training_cuda_memory = cuda_memory_snapshot()
    if training_cuda_memory.get("available"):
        print(
            f"  train_cuda_peak_allocated={training_cuda_memory['max_allocated_mb']:.1f} MiB  "
            f"peak_reserved={training_cuda_memory['max_reserved_mb']:.1f} MiB",
            flush=True,
        )

    if save_model:
        with _phase("save-model"):
            artifacts = ModelArtifacts(
                model=trained.model,
                scaler=trained.scaler,
                config=config,
                epochs_trained=trained.epochs_trained,
                workload_type=single_workload_type(eval_workload_map),
            )
            save_checkpoint(save_model, artifacts)
            print(
                f"  saved checkpoint to {save_model}  "
                f"(epochs_trained={trained.epochs_trained}, "
                f"best_epoch={trained.best_epoch}, best_loss={trained.best_loss:.8f}, "
                f"workload={_workload_name(eval_workload_map)})",
                flush=True,
            )
    methods = [
        MLQDSMethod(
            name="MLQDS",
            trained=trained,
            workload=eval_workload,
            workload_type=single_workload_type(eval_workload_map),
            score_mode=config.model.mlqds_score_mode,
            score_temperature=config.model.mlqds_score_temperature,
            rank_confidence_weight=config.model.mlqds_rank_confidence_weight,
            temporal_fraction=config.model.mlqds_temporal_fraction,
            diversity_bonus=config.model.mlqds_diversity_bonus,
            inference_batch_size=config.model.inference_batch_size,
            amp_mode=config.model.amp_mode,
        ),
        UniformTemporalMethod(),
        DouglasPeuckerMethod(),
    ]

    matched: dict[str, MethodEvaluation] = {}
    oracle_method: OracleMethod | None = None
    eval_labels: torch.Tensor | None = None
    save_masks = bool(save_simplified_dir)
    with _phase("evaluate-matched"):
        eval_query_cache = (
            range_runtime_caches["eval"].query_cache
            if len(_range_only_queries(eval_workload.typed_queries)) == len(eval_workload.typed_queries)
            else None
        )
        if eval_query_cache is None:
            eval_query_cache = EvaluationQueryCache.for_workload(
                test_points,
                test_boundaries,
                eval_workload.typed_queries,
            )
        else:
            eval_query_cache.validate(test_points, test_boundaries, eval_workload.typed_queries)
        for method in methods:
            with _phase(f"  eval {method.name}"):
                matched[method.name] = evaluate_method(
                    method=method,
                    points=test_points,
                    boundaries=test_boundaries,
                    typed_queries=eval_workload.typed_queries,
                    workload_map=eval_workload_map,
                    compression_ratio=config.model.compression_ratio,
                    return_mask=method.name == "MLQDS" or save_masks,
                    query_cache=eval_query_cache,
                )

        if config.baselines.include_oracle:
            if eval_labels is None:
                eval_labels, _ = compute_typed_importance_labels(
                    points=test_points,
                    boundaries=test_boundaries,
                    typed_queries=eval_workload.typed_queries,
                    seed=seeds.eval_query_seed,
                    range_label_mode=str(getattr(config.model, "range_label_mode", "usefulness")),
                    range_boundary_prior_weight=0.0,
                )
            oracle_method = OracleMethod(labels=eval_labels, workload_type=single_workload_type(eval_workload_map))
            with _phase(f"  eval {oracle_method.name}"):
                matched[oracle_method.name] = evaluate_method(
                    method=oracle_method,
                    points=test_points,
                    boundaries=test_boundaries,
                    typed_queries=eval_workload.typed_queries,
                    workload_map=eval_workload_map,
                    compression_ratio=config.model.compression_ratio,
                    query_cache=eval_query_cache,
                )

    learned_fill_diagnostics: dict[str, MethodEvaluation] = {"MLQDS": matched["MLQDS"]}
    learned_fill_table = ""
    diagnostic_methods: list[Method] = []
    if len(_range_only_queries(eval_workload.typed_queries)) == len(eval_workload.typed_queries):
        if eval_labels is None:
            eval_labels, _ = compute_typed_importance_labels(
                points=test_points,
                boundaries=test_boundaries,
                typed_queries=eval_workload.typed_queries,
                seed=seeds.eval_query_seed,
                range_label_mode=str(getattr(config.model, "range_label_mode", "usefulness")),
                range_boundary_prior_weight=0.0,
            )
        assert eval_labels is not None
        _, eval_type_id = workload_type_head(single_workload_type(eval_workload_map))
        random_generator = torch.Generator().manual_seed(int(seeds.eval_query_seed) + 404)
        random_scores = torch.rand((test_points.shape[0],), generator=random_generator)
        diagnostic_methods = [
            ScoreHybridMethod(
                name="TemporalRandomFill",
                scores=random_scores,
                temporal_fraction=config.model.mlqds_temporal_fraction,
                diversity_bonus=config.model.mlqds_diversity_bonus,
            ),
            ScoreHybridMethod(
                name="TemporalOracleFill",
                scores=eval_labels[:, eval_type_id].float(),
                temporal_fraction=config.model.mlqds_temporal_fraction,
                diversity_bonus=config.model.mlqds_diversity_bonus,
            ),
        ]
        with _phase("learned-fill-diagnostics"):
            for method in diagnostic_methods:
                with _phase(f"  fill {method.name}"):
                    learned_fill_diagnostics[method.name] = evaluate_method(
                        method=method,
                        points=test_points,
                        boundaries=test_boundaries,
                        typed_queries=eval_workload.typed_queries,
                        workload_map=eval_workload_map,
                        compression_ratio=config.model.compression_ratio,
                        query_cache=eval_query_cache,
                    )
        learned_fill_table = print_range_usefulness_table(learned_fill_diagnostics)

    matched_table = print_method_comparison_table(matched)
    geometric_table = print_geometric_distortion_table(matched)
    range_usefulness_table = print_range_usefulness_table(matched)
    range_objective_audit: dict[str, dict[str, Any]] = {}
    range_objective_audit_table = ""
    audit_ratios = _range_audit_ratios(config)
    if audit_ratios:
        audit_methods = [*methods, *diagnostic_methods]
        if oracle_method is not None:
            audit_methods.append(oracle_method)
        audit_sections: list[str] = []
        with _phase("range-objective-audit"):
            for ratio in audit_ratios:
                if abs(float(ratio) - float(config.model.compression_ratio)) <= 1e-9:
                    ratio_results = {
                        **matched,
                        **{
                            name: metrics
                            for name, metrics in learned_fill_diagnostics.items()
                            if name not in matched
                        },
                    }
                else:
                    ratio_results: dict[str, MethodEvaluation] = {}
                    for method in audit_methods:
                        with _phase(f"  audit {method.name} ratio={ratio:.4f}"):
                            ratio_results[method.name] = evaluate_method(
                                method=method,
                                points=test_points,
                                boundaries=test_boundaries,
                                typed_queries=eval_workload.typed_queries,
                                workload_map=eval_workload_map,
                                compression_ratio=float(ratio),
                                query_cache=eval_query_cache,
                            )
                ratio_key = f"{float(ratio):.4f}"
                range_objective_audit[ratio_key] = {
                    name: _evaluation_metrics_payload(metrics) for name, metrics in ratio_results.items()
                }
                audit_sections.append(f"compression_ratio={ratio_key}\n{print_range_usefulness_table(ratio_results)}")
        range_objective_audit_table = "\n\n".join(audit_sections)

    with _phase("evaluate-shift"):
        train_name = _workload_name(train_workload_map)
        eval_name = _workload_name(eval_workload_map)
        shift_pairs = {train_name: {eval_name: float(matched["MLQDS"].aggregate_f1)}}
        if train_name == eval_name:
            shift_pairs[train_name][train_name] = float(matched["MLQDS"].aggregate_f1)
        else:
            train_query_cache = EvaluationQueryCache.for_workload(
                test_points,
                test_boundaries,
                train_workload.typed_queries,
            )
            shift_pairs[train_name][train_name] = float(
                evaluate_method(
                    method=MLQDSMethod(
                        name="MLQDS",
                        trained=trained,
                        workload=train_workload,
                        workload_type=single_workload_type(train_workload_map),
                        score_mode=config.model.mlqds_score_mode,
                        score_temperature=config.model.mlqds_score_temperature,
                        rank_confidence_weight=config.model.mlqds_rank_confidence_weight,
                        temporal_fraction=config.model.mlqds_temporal_fraction,
                        diversity_bonus=config.model.mlqds_diversity_bonus,
                        inference_batch_size=config.model.inference_batch_size,
                        amp_mode=config.model.amp_mode,
                    ),
                    points=test_points,
                    boundaries=test_boundaries,
                    typed_queries=train_workload.typed_queries,
                    workload_map=train_workload_map,
                    compression_ratio=config.model.compression_ratio,
                    query_cache=train_query_cache,
                ).aggregate_f1
            )
    shift_table = print_shift_table(shift_pairs)

    dump = {
        "config": config.to_dict(),
        "workload": single_workload_type(eval_workload_map),
        "train_query_count": len(train_workload.typed_queries),
        "eval_query_count": len(eval_workload.typed_queries),
        "selection_query_count": len(selection_workload.typed_queries) if selection_workload is not None else None,
        "train_query_coverage": train_workload.coverage_fraction,
        "eval_query_coverage": eval_workload.coverage_fraction,
        "selection_query_coverage": selection_workload.coverage_fraction if selection_workload is not None else None,
        "query_generation_diagnostics": {
            "train": train_workload.generation_diagnostics,
            "eval": eval_workload.generation_diagnostics,
            "selection": selection_workload.generation_diagnostics if selection_workload is not None else None,
        },
        "matched": {name: _evaluation_metrics_payload(m) for name, m in matched.items()},
        "learned_fill_diagnostics": {
            name: _evaluation_metrics_payload(metrics) for name, metrics in learned_fill_diagnostics.items()
        },
        "range_objective_audit": range_objective_audit,
        "shift": shift_pairs,
        "training_history": trained.history,
        "training_target_diagnostics": trained.target_diagnostics,
        "best_epoch": trained.best_epoch,
        "best_loss": trained.best_loss,
        "best_selection_score": trained.best_selection_score,
        "best_f1": trained.best_f1,
        "checkpoint_selection_metric": config.model.checkpoint_selection_metric,
        "checkpoint_f1_variant": config.model.checkpoint_f1_variant,
        "checkpoint_smoothing_window": config.model.checkpoint_smoothing_window,
        "mlqds_score_mode": config.model.mlqds_score_mode,
        "mlqds_score_temperature": config.model.mlqds_score_temperature,
        "mlqds_rank_confidence_weight": config.model.mlqds_rank_confidence_weight,
        "oracle_diagnostic": {
            "kind": "additive_label_greedy",
            "exact_optimum": False,
        },
        "range_label_mode": config.model.range_label_mode,
        "range_boundary_prior_weight": config.model.range_boundary_prior_weight,
        "range_boundary_prior_enabled": config.model.range_boundary_prior_weight > 0.0,
        "data_audit": data_audit,
        "workload_diagnostics": range_diagnostics_summary,
        "workload_distribution_comparison": workload_distribution_comparison,
        "torch_runtime": {
            **torch_runtime_snapshot(),
            "amp": amp_runtime_snapshot(config.model.amp_mode),
        },
        "cuda_memory": {
            "training": training_cuda_memory,
        },
    }

    with _phase("write-results"):
        out_dir = Path(results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "matched_table.txt").write_text(matched_table + "\n", encoding="utf-8")
        (out_dir / "shift_table.txt").write_text(shift_table + "\n", encoding="utf-8")
        (out_dir / "geometric_distortion_table.txt").write_text(geometric_table + "\n", encoding="utf-8")
        (out_dir / "range_usefulness_table.txt").write_text(range_usefulness_table + "\n", encoding="utf-8")
        if learned_fill_table:
            (out_dir / "learned_fill_diagnostics_table.txt").write_text(
                learned_fill_table + "\n",
                encoding="utf-8",
            )
        (out_dir / "learned_fill_diagnostics.json").write_text(
            json.dumps(
                {name: _evaluation_metrics_payload(metrics) for name, metrics in learned_fill_diagnostics.items()},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if range_objective_audit:
            (out_dir / "range_objective_audit.json").write_text(
                json.dumps(range_objective_audit, indent=2) + "\n",
                encoding="utf-8",
            )
            (out_dir / "range_objective_audit_table.txt").write_text(
                range_objective_audit_table + "\n",
                encoding="utf-8",
            )
        (out_dir / "range_workload_diagnostics.json").write_text(
            json.dumps(range_diagnostics_summary, indent=2) + "\n",
            encoding="utf-8",
        )
        (out_dir / "range_workload_distribution_comparison.json").write_text(
            json.dumps(workload_distribution_comparison, indent=2) + "\n",
            encoding="utf-8",
        )
        with open(out_dir / "range_query_diagnostics.jsonl", "w", encoding="utf-8") as f:
            for row in range_diagnostics_rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        with open(out_dir / "example_run.json", "w", encoding="utf-8") as f:
            json.dump(dump, f, indent=2)
        print(f"  wrote results to {out_dir}", flush=True)

    if save_simplified_dir:
        with _phase("write-simplified-csv"):
            out_dir = Path(save_simplified_dir)
            eval_mask = matched["MLQDS"].retained_mask
            if eval_mask is None:
                eval_mlqds = MLQDSMethod(
                    name="MLQDS",
                    trained=trained,
                    workload=eval_workload,
                    workload_type=single_workload_type(eval_workload_map),
                    score_mode=config.model.mlqds_score_mode,
                    score_temperature=config.model.mlqds_score_temperature,
                    rank_confidence_weight=config.model.mlqds_rank_confidence_weight,
                    temporal_fraction=config.model.mlqds_temporal_fraction,
                    diversity_bonus=config.model.mlqds_diversity_bonus,
                    inference_batch_size=config.model.inference_batch_size,
                    amp_mode=config.model.amp_mode,
                )
                eval_mask = eval_mlqds.simplify(test_points, test_boundaries, config.model.compression_ratio)
            write_simplified_csv(
                str(out_dir / "ML_simplified_eval.csv"),
                test_points,
                test_boundaries,
                eval_mask,
                trajectory_mmsis=test_mmsis,
            )
            for ref_name, csv_name in (("uniform", "uniform_simplified_eval.csv"),
                                       ("DouglasPeucker", "DP_simplified_eval.csv")):
                ref_eval = matched.get(ref_name)
                ref_mask = ref_eval.retained_mask if ref_eval is not None else None
                if ref_mask is not None:
                    write_simplified_csv(
                        str(out_dir / csv_name),
                        test_points,
                        test_boundaries,
                        ref_mask,
                        trajectory_mmsis=test_mmsis,
                    )

        with _phase("trajectory-length-loss"):
            report_trajectory_length_loss(test_points, test_boundaries, eval_mask, top_k=25, trajectory_mmsis=test_mmsis)

    print(f"[pipeline] total runtime {time.perf_counter() - pipeline_t0:.2f}s", flush=True)
    return ExperimentOutputs(
        matched_table=matched_table,
        shift_table=shift_table,
        metrics_dump=dump,
        geometric_table=geometric_table,
        range_usefulness_table=range_usefulness_table,
        range_objective_audit_table=range_objective_audit_table,
    )


def _workload_keyword_to_map(keyword: str | None) -> dict[str, float] | None:
    """Translate a --workload keyword to a concrete pure workload map, or return None.

    - "range"/"knn"/"similarity"/"clustering" -> 100% that type.
    - anything else -> None (fall back to caller default).
    """
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
