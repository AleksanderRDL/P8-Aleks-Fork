"""Workload diagnostics for query generation quality checks."""

from __future__ import annotations

from typing import Any

import torch

from src.queries.query_types import QUERY_TYPE_ID_RANGE


def _dataset_bounds(points: torch.Tensor) -> dict[str, float]:
    """Return global bounds used to normalize query footprint metrics."""
    if points.numel() == 0:
        return {
            "t_min": 0.0,
            "t_max": 0.0,
            "lat_min": 0.0,
            "lat_max": 0.0,
            "lon_min": 0.0,
            "lon_max": 0.0,
        }
    return {
        "t_min": float(points[:, 0].min().item()),
        "t_max": float(points[:, 0].max().item()),
        "lat_min": float(points[:, 1].min().item()),
        "lat_max": float(points[:, 1].max().item()),
        "lon_min": float(points[:, 2].min().item()),
        "lon_max": float(points[:, 2].max().item()),
    }


def range_box_mask(points: torch.Tensor, params: dict[str, float]) -> torch.Tensor:
    """Return point mask for one range-query box."""
    if points.numel() == 0:
        return torch.zeros((0,), dtype=torch.bool, device=points.device)
    return (
        (points[:, 1] >= float(params["lat_min"]))
        & (points[:, 1] <= float(params["lat_max"]))
        & (points[:, 2] >= float(params["lon_min"]))
        & (points[:, 2] <= float(params["lon_max"]))
        & (points[:, 0] >= float(params["t_start"]))
        & (points[:, 0] <= float(params["t_end"]))
    )


def _trajectory_hits(mask: torch.Tensor, boundaries: list[tuple[int, int]]) -> int:
    """Count trajectories containing at least one masked point."""
    hits = 0
    for start, end in boundaries:
        if end > start and bool(mask[start:end].any().item()):
            hits += 1
    return hits


def _safe_span(max_value: float, min_value: float) -> float:
    """Return a nonzero span denominator for fraction metrics."""
    return max(float(max_value) - float(min_value), 1e-9)


def _range_span_fractions(params: dict[str, float], bounds: dict[str, float]) -> tuple[float, float, float, float]:
    """Return normalized lat/lon/time spans and their product."""
    lat_fraction = max(0.0, float(params["lat_max"]) - float(params["lat_min"])) / _safe_span(
        bounds["lat_max"],
        bounds["lat_min"],
    )
    lon_fraction = max(0.0, float(params["lon_max"]) - float(params["lon_min"])) / _safe_span(
        bounds["lon_max"],
        bounds["lon_min"],
    )
    time_fraction = max(0.0, float(params["t_end"]) - float(params["t_start"])) / _safe_span(
        bounds["t_max"],
        bounds["t_min"],
    )
    lat_fraction = min(1.0, float(lat_fraction))
    lon_fraction = min(1.0, float(lon_fraction))
    time_fraction = min(1.0, float(time_fraction))
    return lat_fraction, lon_fraction, time_fraction, float(lat_fraction * lon_fraction * time_fraction)


def range_box_iou(params_a: dict[str, float], params_b: dict[str, float]) -> float:
    """Compute axis-aligned spatiotemporal box IoU for two range queries."""
    axes = [
        ("lat_min", "lat_max"),
        ("lon_min", "lon_max"),
        ("t_start", "t_end"),
    ]
    intersection = 1.0
    volume_a = 1.0
    volume_b = 1.0
    for lo_key, hi_key in axes:
        a_lo = float(params_a[lo_key])
        a_hi = float(params_a[hi_key])
        b_lo = float(params_b[lo_key])
        b_hi = float(params_b[hi_key])
        a_span = max(0.0, a_hi - a_lo)
        b_span = max(0.0, b_hi - b_lo)
        overlap = max(0.0, min(a_hi, b_hi) - max(a_lo, b_lo))
        intersection *= overlap
        volume_a *= a_span
        volume_b *= b_span
    union = volume_a + volume_b - intersection
    if union <= 1e-12:
        return 1.0 if volume_a <= 1e-12 and volume_b <= 1e-12 else 0.0
    return float(intersection / union)


def range_query_diagnostic(
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    query: dict[str, Any],
    *,
    query_index: int = 0,
    previous_range_queries: list[dict[str, Any]] | None = None,
    bounds: dict[str, float] | None = None,
    max_point_hit_fraction: float | None = None,
    max_trajectory_hit_fraction: float | None = None,
    max_box_volume_fraction: float | None = None,
    duplicate_iou_threshold: float | None = 0.85,
) -> dict[str, Any]:
    """Return JSON-safe diagnostics for one range query."""
    params = query["params"]
    bounds = bounds or _dataset_bounds(points)
    mask = range_box_mask(points, params)
    point_hits = int(mask.sum().item())
    trajectory_hits = _trajectory_hits(mask, boundaries)
    total_points = int(points.shape[0])
    total_trajectories = int(len(boundaries))
    point_hit_fraction = float(point_hits / total_points) if total_points > 0 else 0.0
    trajectory_hit_fraction = float(trajectory_hits / total_trajectories) if total_trajectories > 0 else 0.0
    lat_fraction, lon_fraction, time_fraction, box_volume_fraction = _range_span_fractions(params, bounds)

    near_duplicate_of: int | None = None
    near_duplicate_iou = 0.0
    if duplicate_iou_threshold is not None and previous_range_queries:
        threshold = float(duplicate_iou_threshold)
        for previous in previous_range_queries:
            previous_params = previous["params"]
            iou = range_box_iou(params, previous_params)
            if iou > near_duplicate_iou:
                near_duplicate_iou = float(iou)
            if iou >= threshold:
                near_duplicate_of = int(previous.get("query_index", previous.get("_query_index", 0)))
                break

    too_broad = False
    if max_point_hit_fraction is not None and point_hit_fraction > float(max_point_hit_fraction):
        too_broad = True
    if max_trajectory_hit_fraction is not None and trajectory_hit_fraction > float(max_trajectory_hit_fraction):
        too_broad = True
    if max_box_volume_fraction is not None and box_volume_fraction > float(max_box_volume_fraction):
        too_broad = True

    return {
        "query_index": int(query_index),
        "point_hits": point_hits,
        "trajectory_hits": trajectory_hits,
        "point_hit_fraction": point_hit_fraction,
        "trajectory_hit_fraction": trajectory_hit_fraction,
        "lat_span_fraction": float(lat_fraction),
        "lon_span_fraction": float(lon_fraction),
        "time_span_fraction": float(time_fraction),
        "box_volume_fraction": float(box_volume_fraction),
        "is_empty": bool(point_hits == 0),
        "is_too_broad": bool(too_broad),
        "near_duplicate_of": near_duplicate_of,
        "near_duplicate_iou": float(near_duplicate_iou),
    }


def _quantile(values: list[float], q: float) -> float:
    """Return a scalar quantile from a possibly empty list."""
    if not values:
        return 0.0
    return float(torch.quantile(torch.tensor(values, dtype=torch.float32), float(q)).item())


def _summary_from_query_diagnostics(query_rows: list[dict[str, Any]], coverage_fraction: float) -> dict[str, Any]:
    """Aggregate per-query diagnostics into a compact summary."""
    count = len(query_rows)
    point_hits = [float(row["point_hits"]) for row in query_rows]
    trajectory_hits = [float(row["trajectory_hits"]) for row in query_rows]
    point_hit_fractions = [float(row["point_hit_fraction"]) for row in query_rows]
    trajectory_hit_fractions = [float(row["trajectory_hit_fraction"]) for row in query_rows]
    box_volume_fractions = [float(row["box_volume_fraction"]) for row in query_rows]
    return {
        "range_query_count": int(count),
        "empty_query_rate": float(sum(1 for row in query_rows if row["is_empty"]) / count) if count else 0.0,
        "too_broad_query_rate": float(sum(1 for row in query_rows if row["is_too_broad"]) / count) if count else 0.0,
        "near_duplicate_query_rate": (
            float(sum(1 for row in query_rows if row["near_duplicate_of"] is not None) / count) if count else 0.0
        ),
        "point_hit_count_p10": _quantile(point_hits, 0.10),
        "point_hit_count_p50": _quantile(point_hits, 0.50),
        "point_hit_count_p90": _quantile(point_hits, 0.90),
        "trajectory_hit_count_p10": _quantile(trajectory_hits, 0.10),
        "trajectory_hit_count_p50": _quantile(trajectory_hits, 0.50),
        "trajectory_hit_count_p90": _quantile(trajectory_hits, 0.90),
        "point_hit_fraction_p10": _quantile(point_hit_fractions, 0.10),
        "point_hit_fraction_p50": _quantile(point_hit_fractions, 0.50),
        "point_hit_fraction_p90": _quantile(point_hit_fractions, 0.90),
        "trajectory_hit_fraction_p10": _quantile(trajectory_hit_fractions, 0.10),
        "trajectory_hit_fraction_p50": _quantile(trajectory_hit_fractions, 0.50),
        "trajectory_hit_fraction_p90": _quantile(trajectory_hit_fractions, 0.90),
        "box_volume_fraction_p50": _quantile(box_volume_fractions, 0.50),
        "box_volume_fraction_p90": _quantile(box_volume_fractions, 0.90),
        "coverage_fraction": float(coverage_fraction),
    }


def compute_range_workload_diagnostics(
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, Any]],
    *,
    max_point_hit_fraction: float | None = None,
    max_trajectory_hit_fraction: float | None = None,
    max_box_volume_fraction: float | None = None,
    duplicate_iou_threshold: float | None = 0.85,
) -> dict[str, Any]:
    """Compute range-query workload quality diagnostics."""
    bounds = _dataset_bounds(points)
    previous: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    covered = torch.zeros((points.shape[0],), dtype=torch.bool, device=points.device)
    for query_index, query in enumerate(typed_queries):
        if str(query.get("type", "")).lower() != "range":
            continue
        row = range_query_diagnostic(
            points,
            boundaries,
            query,
            query_index=query_index,
            previous_range_queries=previous,
            bounds=bounds,
            max_point_hit_fraction=max_point_hit_fraction,
            max_trajectory_hit_fraction=max_trajectory_hit_fraction,
            max_box_volume_fraction=max_box_volume_fraction,
            duplicate_iou_threshold=duplicate_iou_threshold,
        )
        rows.append(row)
        previous.append({"params": query["params"], "query_index": query_index})
        covered |= range_box_mask(points, query["params"])

    coverage_fraction = float(covered.float().mean().item()) if points.shape[0] > 0 else 0.0
    return {
        "summary": _summary_from_query_diagnostics(rows, coverage_fraction),
        "queries": rows,
    }


def compute_range_label_diagnostics(labels: torch.Tensor, labelled_mask: torch.Tensor) -> dict[str, Any]:
    """Summarize range-label density and magnitude."""
    if labels.numel() == 0 or labelled_mask.numel() == 0:
        return {
            "labelled_point_count": 0,
            "positive_point_count": 0,
            "positive_label_fraction": 0.0,
            "positive_label_p50": 0.0,
            "positive_label_p90": 0.0,
            "positive_label_p95": 0.0,
            "positive_label_max": 0.0,
        }

    active = labelled_mask[:, QUERY_TYPE_ID_RANGE]
    values = labels[:, QUERY_TYPE_ID_RANGE]
    positive = active & (values > 0)
    positive_values = values[positive].detach().cpu().float().tolist()
    labelled_count = int(active.sum().item())
    positive_count = int(positive.sum().item())
    return {
        "labelled_point_count": labelled_count,
        "positive_point_count": positive_count,
        "positive_label_fraction": float(positive_count / max(1, labelled_count)),
        "positive_label_p50": _quantile(positive_values, 0.50),
        "positive_label_p90": _quantile(positive_values, 0.90),
        "positive_label_p95": _quantile(positive_values, 0.95),
        "positive_label_max": float(max(positive_values)) if positive_values else 0.0,
    }
