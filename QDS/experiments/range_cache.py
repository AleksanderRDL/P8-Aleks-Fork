"""Range diagnostics, query-cache, and label-cache helpers for experiment runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from evaluation.query_cache import EvaluationQueryCache
from experiments.experiment_config import ExperimentConfig
from experiments.workload_cache import tensor_cache_digest
from queries.workload import TypedQueryWorkload
from training.importance_labels import (
    compute_typed_importance_labels,
    compute_typed_importance_labels_with_range_components,
)

RANGE_DIAGNOSTICS_CACHE_SCHEMA_VERSION = 2


@dataclass
class RangeRuntimeCache:
    """Non-serialized tensors/caches reused across range diagnostics, training, and evaluation."""

    labels: torch.Tensor | None = None
    labelled_mask: torch.Tensor | None = None
    component_labels: dict[str, torch.Tensor] | None = None
    query_cache: EvaluationQueryCache | None = None


def runtime_evaluation_query_cache(
    runtime_cache: RangeRuntimeCache | None,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, Any]],
) -> EvaluationQueryCache:
    """Return a reusable query cache for one runtime split."""
    if runtime_cache is not None and runtime_cache.query_cache is not None:
        try:
            runtime_cache.query_cache.validate(points, boundaries, typed_queries)
            return runtime_cache.query_cache
        except ValueError:
            runtime_cache.query_cache = None

    query_cache = EvaluationQueryCache.for_workload(points, boundaries, typed_queries)
    if runtime_cache is not None:
        runtime_cache.query_cache = query_cache
    return query_cache


def ensure_range_runtime_labels(
    *,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    range_queries: list[dict[str, Any]],
    seed: int,
    range_label_mode: str,
    range_boundary_prior_weight: float,
    runtime_cache: RangeRuntimeCache | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return range labels, populating the runtime cache once per split."""
    if runtime_cache is not None and runtime_cache.labels is not None and runtime_cache.labelled_mask is not None:
        return runtime_cache.labels, runtime_cache.labelled_mask

    range_label_mode = str(range_label_mode).lower()
    if range_label_mode in {"usefulness", "usefulness_balanced"}:
        labels, labelled_mask, component_labels = compute_typed_importance_labels_with_range_components(
            points=points,
            boundaries=boundaries,
            typed_queries=range_queries,
            seed=seed,
            range_boundary_prior_weight=range_boundary_prior_weight,
            range_label_mode=range_label_mode,
        )
    else:
        labels, labelled_mask = compute_typed_importance_labels(
            points=points,
            boundaries=boundaries,
            typed_queries=range_queries,
            seed=seed,
            range_label_mode=range_label_mode,
            range_boundary_prior_weight=range_boundary_prior_weight,
        )
        component_labels = None

    if runtime_cache is not None:
        runtime_cache.labels = labels
        runtime_cache.labelled_mask = labelled_mask
        runtime_cache.component_labels = component_labels
    return labels, labelled_mask


def range_diagnostic_duplicate_threshold(config: ExperimentConfig) -> float | None:
    """Use explicit duplicate threshold for diagnostics, or a diagnostic-only default."""
    threshold = config.query.range_duplicate_iou_threshold
    return 0.85 if threshold is None else threshold


def range_only_queries(typed_queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only range queries from a typed workload."""
    return [query for query in typed_queries if str(query.get("type", "")).lower() == "range"]


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


def range_diagnostics_cache_payload(
    *,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    workload: TypedQueryWorkload,
    workload_map: dict[str, float],
    config: ExperimentConfig,
    seed: int,
    range_label_mode: str | None = None,
    range_boundary_prior_weight: float | None = None,
) -> dict[str, Any]:
    """Build the canonical cache key payload for range workload diagnostics."""
    label_mode = str(
        range_label_mode
        if range_label_mode is not None
        else getattr(config.model, "range_label_mode", "usefulness")
    )
    prior_weight = float(
        range_boundary_prior_weight
        if range_boundary_prior_weight is not None
        else getattr(config.model, "range_boundary_prior_weight", 0.0)
    )
    return {
        "schema_version": RANGE_DIAGNOSTICS_CACHE_SCHEMA_VERSION,
        "points_sha256": tensor_cache_digest(points),
        "boundaries_sha256": hashlib.sha256(
            json.dumps(boundaries, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "typed_queries_sha256": _typed_queries_digest(workload.typed_queries),
        "workload_map": {key: float(workload_map[key]) for key in sorted(workload_map)},
        "seed": int(seed),
        "compression_ratio": float(config.model.compression_ratio),
        "range_label_mode": label_mode,
        "range_boundary_prior_weight": prior_weight,
        "max_point_hit_fraction": config.query.range_max_point_hit_fraction,
        "max_trajectory_hit_fraction": config.query.range_max_trajectory_hit_fraction,
        "max_box_volume_fraction": config.query.range_max_box_volume_fraction,
        "duplicate_iou_threshold": range_diagnostic_duplicate_threshold(config),
    }


def range_diagnostics_cache_key(payload: dict[str, Any]) -> str:
    """Return a stable cache key for range diagnostics."""
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def range_label_cache_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the compression-independent subset needed for label tensor caching."""
    label_payload = dict(payload)
    label_payload["cache_kind"] = "range_labels"
    label_payload.pop("compression_ratio", None)
    label_payload.pop("duplicate_iou_threshold", None)
    return label_payload


def _range_diagnostics_cache_paths(config: ExperimentConfig, label: str, key: str) -> tuple[Path, Path] | None:
    """Return JSON and tensor paths for a range diagnostics cache entry."""
    cache_root = _range_diagnostics_cache_root(config)
    if cache_root is None:
        return None
    stem = f"{label}-{key[:16]}"
    return cache_root / f"{stem}.json", cache_root / f"{stem}.pt"


def load_range_diagnostics_cache(
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
            component_labels = tensors.get("component_labels")
            runtime_cache.component_labels = (
                {str(key): value for key, value in component_labels.items() if isinstance(value, torch.Tensor)}
                if isinstance(component_labels, dict)
                else None
            )
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


def _load_range_label_tensor_cache(
    *,
    config: ExperimentConfig,
    label: str,
    key: str,
    runtime_cache: RangeRuntimeCache,
) -> bool:
    """Load early-written range label tensors when full diagnostics JSON is not available yet."""
    paths = _range_diagnostics_cache_paths(config, label, key)
    if paths is None or config.data.refresh_cache:
        return False
    _json_path, tensor_path = paths
    if not tensor_path.exists():
        return False
    try:
        tensors = torch.load(tensor_path, map_location="cpu", weights_only=True)
        if not isinstance(tensors, dict):
            return False
        labels = tensors.get("labels")
        labelled_mask = tensors.get("labelled_mask")
        if not isinstance(labels, torch.Tensor) or not isinstance(labelled_mask, torch.Tensor):
            return False
        runtime_cache.labels = labels
        runtime_cache.labelled_mask = labelled_mask
        component_labels = tensors.get("component_labels")
        runtime_cache.component_labels = (
            {str(key): value for key, value in component_labels.items() if isinstance(value, torch.Tensor)}
            if isinstance(component_labels, dict)
            else None
        )
        print(f"  range label cache hit: {tensor_path}", flush=True)
        return True
    except (OSError, TypeError, RuntimeError) as exc:
        print(f"  WARNING: ignoring unreadable range label cache {tensor_path}: {exc}", flush=True)
        return False


def _write_range_label_tensor_cache(
    *,
    config: ExperimentConfig,
    label: str,
    key: str,
    runtime_cache: RangeRuntimeCache,
) -> None:
    """Persist range label tensors immediately after training prep."""
    paths = _range_diagnostics_cache_paths(config, label, key)
    if paths is None or runtime_cache.labels is None or runtime_cache.labelled_mask is None:
        return
    _json_path, tensor_path = paths
    try:
        tensor_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "labels": runtime_cache.labels.cpu(),
            "labelled_mask": runtime_cache.labelled_mask.cpu(),
        }
        if runtime_cache.component_labels is not None:
            payload["component_labels"] = {key: value.cpu() for key, value in runtime_cache.component_labels.items()}
        torch.save(payload, tensor_path)
        print(f"  range label cache wrote: {tensor_path}", flush=True)
    except OSError as exc:
        print(f"  WARNING: could not write range label cache {tensor_path}: {exc}", flush=True)


def write_range_diagnostics_cache(
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
        payload: dict[str, Any] = {
            "labels": runtime_cache.labels.cpu(),
            "labelled_mask": runtime_cache.labelled_mask.cpu(),
        }
        if runtime_cache.component_labels is not None:
            payload["component_labels"] = {key: value.cpu() for key, value in runtime_cache.component_labels.items()}
        torch.save(payload, tensor_path)
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


def prepare_range_label_cache(
    *,
    cache_label: str,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    workload: TypedQueryWorkload,
    workload_map: dict[str, float],
    config: ExperimentConfig,
    seed: int,
    runtime_cache: RangeRuntimeCache,
    range_boundary_prior_weight: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Populate range-label tensors, reusing persistent caches when available."""
    range_queries = range_only_queries(workload.typed_queries)
    if len(range_queries) != len(workload.typed_queries):
        return None
    if runtime_cache.labels is not None and runtime_cache.labelled_mask is not None:
        return runtime_cache.labels, runtime_cache.labelled_mask

    cache_payload = range_diagnostics_cache_payload(
        points=points,
        boundaries=boundaries,
        workload=workload,
        workload_map=workload_map,
        config=config,
        seed=seed,
        range_boundary_prior_weight=range_boundary_prior_weight,
    )
    cache_key = range_diagnostics_cache_key(range_label_cache_payload(cache_payload))
    if _load_range_label_tensor_cache(
        config=config,
        label=cache_label,
        key=cache_key,
        runtime_cache=runtime_cache,
    ):
        assert runtime_cache.labels is not None
        assert runtime_cache.labelled_mask is not None
        return runtime_cache.labels, runtime_cache.labelled_mask

    range_label_mode = str(getattr(config.model, "range_label_mode", "usefulness")).lower()
    prior_weight = float(
        range_boundary_prior_weight
        if range_boundary_prior_weight is not None
        else getattr(config.model, "range_boundary_prior_weight", 0.0)
    )
    if range_label_mode in {"usefulness", "usefulness_balanced"}:
        labels, labelled_mask, component_labels = compute_typed_importance_labels_with_range_components(
            points=points,
            boundaries=boundaries,
            typed_queries=range_queries,
            seed=seed,
            range_boundary_prior_weight=prior_weight,
            range_label_mode=range_label_mode,
        )
        runtime_cache.component_labels = component_labels
    else:
        labels, labelled_mask = compute_typed_importance_labels(
            points=points,
            boundaries=boundaries,
            typed_queries=range_queries,
            seed=seed,
            range_label_mode=range_label_mode,
            range_boundary_prior_weight=prior_weight,
        )
        runtime_cache.component_labels = None

    runtime_cache.labels = labels
    runtime_cache.labelled_mask = labelled_mask
    _write_range_label_tensor_cache(
        config=config,
        label=cache_label,
        key=cache_key,
        runtime_cache=runtime_cache,
    )
    return labels, labelled_mask


def prepare_range_training_cache(
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    workload: TypedQueryWorkload,
    workload_map: dict[str, float],
    config: ExperimentConfig,
    seed: int,
    runtime_cache: RangeRuntimeCache,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Populate train-label tensors needed before model training starts."""
    return prepare_range_label_cache(
        cache_label="train",
        points=points,
        boundaries=boundaries,
        workload=workload,
        workload_map=workload_map,
        config=config,
        seed=seed,
        runtime_cache=runtime_cache,
    )
