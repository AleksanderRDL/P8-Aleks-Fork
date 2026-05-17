"""learned_segment_budget_v1 selector."""

from __future__ import annotations

import math
from typing import Any

import torch

from simplification.simplify_trajectories import deterministic_topk_with_jitter

LEARNED_SEGMENT_BUDGET_SCHEMA_VERSION = 2
LEARNED_SEGMENT_BUDGET_TRACE_SCHEMA_VERSION = 3
SEGMENT_ALLOCATION_WEIGHT_FLOOR = 0.50
GEOMETRY_TIE_BREAKER_WEIGHT = 0.12
SEGMENT_SCORE_POINT_BLEND_WEIGHT = 0.05


def blend_segment_support_scores(
    *,
    segment_scores: torch.Tensor | None,
    path_length_support_scores: torch.Tensor | None,
    path_length_support_weight: float,
) -> torch.Tensor | None:
    """Blend query segment scores with learned query-free path-length support scores."""
    weight = max(0.0, min(1.0, float(path_length_support_weight)))
    if weight <= 0.0 or path_length_support_scores is None:
        return None if segment_scores is None else segment_scores.detach().cpu().float()
    path_scores = path_length_support_scores.detach().cpu().float()
    if segment_scores is None:
        return path_scores
    segment = segment_scores.detach().cpu().float()
    if segment.shape != path_scores.shape:
        raise ValueError(
            "segment_scores and path_length_support_scores must have matching shape: "
            f"got {tuple(segment.shape)} and {tuple(path_scores.shape)}."
        )
    return (1.0 - weight) * segment + weight * path_scores


def _total_budget(boundaries: list[tuple[int, int]], compression_ratio: float) -> int:
    """Return the comparable per-trajectory total budget."""
    total = 0
    ratio = min(1.0, max(0.0, float(compression_ratio)))
    for start, end in boundaries:
        count = int(end - start)
        if count <= 0:
            continue
        total += min(count, max(2, int(math.ceil(ratio * count))))
    return total


def _max_skeleton_fraction(compression_ratio: float) -> float:
    """Return guide-recommended maximum skeleton share."""
    ratio = float(compression_ratio)
    if ratio <= 0.01:
        return 0.50
    if ratio <= 0.02:
        return 0.40
    if ratio <= 0.05:
        return 0.25
    if ratio <= 0.10:
        return 0.20
    return 0.15


def _segment_rows(
    scores: torch.Tensor,
    boundaries: list[tuple[int, int]],
    segment_size: int,
    segment_scores: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    """Return candidate segment rows with predicted value."""
    rows: list[dict[str, Any]] = []
    size = max(1, int(segment_size))
    segment_values = scores if segment_scores is None else segment_scores.to(device=scores.device)
    for trajectory_id, (start, end) in enumerate(boundaries):
        for seg_start in range(int(start), int(end), size):
            seg_end = min(int(end), seg_start + size)
            if seg_end <= seg_start:
                continue
            if segment_scores is None:
                local_segment = scores[seg_start:seg_end].float()
                top_count = min(
                    int(local_segment.numel()),
                    max(1, int(math.ceil(0.20 * int(local_segment.numel())))),
                )
                segment_score = float(torch.topk(local_segment, k=top_count).values.mean().item())
                segment_score_source = "point_score_top20_mean"
            else:
                local_segment = segment_values[seg_start:seg_end].float()
                head_top_count = min(
                    int(local_segment.numel()),
                    max(1, int(math.ceil(0.20 * int(local_segment.numel())))),
                )
                segment_score = float(torch.topk(local_segment, k=head_top_count).values.mean().item())
                segment_score_source = "segment_budget_head_top20_mean"
            rows.append(
                {
                    "segment_index": int(len(rows)),
                    "trajectory_id": int(trajectory_id),
                    "start": int(seg_start),
                    "end": int(seg_end),
                    "score": segment_score,
                    "score_source": segment_score_source,
                    "length": int(seg_end - seg_start),
                }
            )
    return rows


def _entropy(counts: list[int]) -> tuple[float, float]:
    """Return raw and normalized Shannon entropy for positive counts."""
    positive = [int(count) for count in counts if int(count) > 0]
    total = sum(positive)
    if total <= 0:
        return 0.0, 0.0
    entropy = 0.0
    for count in positive:
        probability = float(count) / float(total)
        entropy -= probability * math.log(probability)
    if len(positive) <= 1:
        return float(entropy), 0.0
    return float(entropy), float(entropy / math.log(float(len(positive))))


def _trajectory_counts(mask: torch.Tensor, boundaries: list[tuple[int, int]]) -> list[int]:
    """Count selected points per trajectory."""
    return [int(mask[int(start):int(end)].sum().item()) for start, end in boundaries]


def _mask_indices_payload(mask: torch.Tensor | None) -> dict[str, Any]:
    """Return a compact JSON payload for a diagnostic retained mask."""
    if mask is None:
        return {
            "available": False,
            "diagnostic_only": True,
            "query_free": True,
            "reason": "not_run",
            "retained_count": None,
            "indices": [],
        }
    mask_bool = mask.detach().cpu().bool().flatten()
    indices = [int(idx) for idx in torch.where(mask_bool)[0].tolist()]
    return {
        "available": True,
        "diagnostic_only": True,
        "query_free": True,
        "retained_count": int(len(indices)),
        "indices": indices,
    }


def _descending_ranks(values: list[float]) -> list[int]:
    """Return 1-based descending ranks for deterministic diagnostic ordering."""
    order = sorted(range(len(values)), key=lambda idx: (float(values[idx]), -idx), reverse=True)
    ranks = [0 for _value in values]
    for rank, idx in enumerate(order, start=1):
        ranks[int(idx)] = int(rank)
    return ranks


def _segment_source_attribution(
    *,
    segment_rows: list[dict[str, Any]],
    segment_allocations: dict[int, int],
    retained: torch.Tensor,
    skeleton_mask: torch.Tensor,
    learned_mask: torch.Tensor,
    fallback_mask: torch.Tensor,
    length_repair_mask: torch.Tensor,
) -> dict[str, Any]:
    """Return query-free retained-source attribution per selector segment."""
    if not segment_rows:
        return {"available": False, "reason": "segment_rows_missing"}
    point_count = int(retained.numel())
    masks = {
        "retained": retained.detach().cpu().bool(),
        "skeleton": skeleton_mask.detach().cpu().bool(),
        "learned": learned_mask.detach().cpu().bool(),
        "fallback": fallback_mask.detach().cpu().bool(),
        "length_repair": length_repair_mask.detach().cpu().bool(),
    }
    if any(int(mask.numel()) != point_count for mask in masks.values()):
        return {"available": False, "reason": "mask_shape_mismatch"}
    attributed = masks["skeleton"] | masks["learned"] | masks["fallback"] | masks["length_repair"]
    score_values = [
        float(row.get("score", 0.0)) if math.isfinite(float(row.get("score", 0.0))) else 0.0
        for row in segment_rows
    ]
    score_ranks = _descending_ranks(score_values)
    rows: list[dict[str, Any]] = []
    summary_counts = {
        "retained": 0,
        "skeleton": 0,
        "learned": 0,
        "fallback": 0,
        "length_repair": 0,
        "unattributed": 0,
        "allocation": 0,
    }
    segment_presence = {
        "retained": 0,
        "skeleton": 0,
        "learned": 0,
        "fallback": 0,
        "length_repair": 0,
        "allocation": 0,
    }
    for allocation_order_index, row in enumerate(segment_rows):
        start = int(row["start"])
        end = int(row["end"])
        if start < 0 or end > point_count or end <= start:
            continue
        canonical_segment_index = int(row.get("segment_index", allocation_order_index))
        retained_count = int(masks["retained"][start:end].sum().item())
        skeleton_count = int(masks["skeleton"][start:end].sum().item())
        learned_count = int(masks["learned"][start:end].sum().item())
        fallback_count = int(masks["fallback"][start:end].sum().item())
        length_repair_count = int(masks["length_repair"][start:end].sum().item())
        unattributed_count = int((masks["retained"][start:end] & ~attributed[start:end]).sum().item())
        allocation_count = int(segment_allocations.get(allocation_order_index, 0))
        summary_counts["retained"] += retained_count
        summary_counts["skeleton"] += skeleton_count
        summary_counts["learned"] += learned_count
        summary_counts["fallback"] += fallback_count
        summary_counts["length_repair"] += length_repair_count
        summary_counts["unattributed"] += unattributed_count
        summary_counts["allocation"] += allocation_count
        if retained_count > 0:
            segment_presence["retained"] += 1
        if skeleton_count > 0:
            segment_presence["skeleton"] += 1
        if learned_count > 0:
            segment_presence["learned"] += 1
        if fallback_count > 0:
            segment_presence["fallback"] += 1
        if length_repair_count > 0:
            segment_presence["length_repair"] += 1
        if allocation_count > 0:
            segment_presence["allocation"] += 1
        rows.append(
            {
                "segment_index": canonical_segment_index,
                "allocation_order_index": int(allocation_order_index),
                "trajectory_id": int(row["trajectory_id"]),
                "start": start,
                "end": end,
                "length": int(end - start),
                "segment_score": float(score_values[allocation_order_index]),
                "segment_score_rank": int(score_ranks[allocation_order_index]),
                "segment_score_source": str(row.get("score_source", "")),
                "segment_allocation_count": allocation_count,
                "retained_count": retained_count,
                "retained_fraction": float(retained_count / max(1, end - start)),
                "skeleton_count": skeleton_count,
                "learned_count": learned_count,
                "fallback_count": fallback_count,
                "length_repair_count": length_repair_count,
                "unattributed_count": unattributed_count,
            }
        )
    return {
        "available": True,
        "diagnostic_only": True,
        "query_free": True,
        "description": "Per-segment retained-point attribution by selector source before eval-query scoring.",
        "segment_count": int(len(rows)),
        "summary": {
            "retained_count_total": int(summary_counts["retained"]),
            "skeleton_count_total": int(summary_counts["skeleton"]),
            "learned_count_total": int(summary_counts["learned"]),
            "fallback_count_total": int(summary_counts["fallback"]),
            "length_repair_count_total": int(summary_counts["length_repair"]),
            "unattributed_count_total": int(summary_counts["unattributed"]),
            "segment_allocation_count_total": int(summary_counts["allocation"]),
            "segments_with_retained": int(segment_presence["retained"]),
            "segments_with_skeleton": int(segment_presence["skeleton"]),
            "segments_with_learned": int(segment_presence["learned"]),
            "segments_with_fallback": int(segment_presence["fallback"]),
            "segments_with_length_repair": int(segment_presence["length_repair"]),
            "segments_with_allocation": int(segment_presence["allocation"]),
        },
        "rows": rows,
    }


def _polyline_length_km(points: torch.Tensor) -> float:
    """Return approximate retained polyline length in km for lat/lon points."""
    if int(points.shape[0]) < 2:
        return 0.0
    lats = points[:, 1].float()
    lons = points[:, 2].float()
    lat_rad = torch.deg2rad(lats)
    lon_rad = torch.deg2rad(lons)
    dlat = lat_rad[1:] - lat_rad[:-1]
    dlon = lon_rad[1:] - lon_rad[:-1]
    a = torch.sin(dlat / 2.0) ** 2 + torch.cos(lat_rad[:-1]) * torch.cos(lat_rad[1:]) * torch.sin(dlon / 2.0) ** 2
    c = 2.0 * torch.atan2(torch.sqrt(a), torch.sqrt(torch.clamp(1.0 - a, min=1e-9)))
    return float((6371.0 * c).sum().item())


def _quantile(values: list[float], q: float) -> float | None:
    """Return a deterministic linear quantile for small diagnostic lists."""
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, float(q))) * float(len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - float(lower)
    return float((1.0 - fraction) * ordered[lower] + fraction * ordered[upper])


def _retained_index_gap_stats(mask: torch.Tensor) -> tuple[int, float]:
    """Return max and mean retained-index gap for one trajectory mask."""
    indices = torch.where(mask.detach().cpu().bool())[0]
    if int(indices.numel()) < 2:
        return 0, 0.0
    gaps = (indices[1:] - indices[:-1]).float()
    return int(gaps.max().item()), float(gaps.mean().item())


def _selector_geometry_diagnostics(
    *,
    points: torch.Tensor | None,
    retained: torch.Tensor,
    skeleton_mask: torch.Tensor,
    learned_mask: torch.Tensor,
    fallback_mask: torch.Tensor,
    length_repair_mask: torch.Tensor,
    boundaries: list[tuple[int, int]],
) -> dict[str, Any]:
    """Return query-free path-length diagnostics for the selector trace."""
    if points is None or int(points.shape[0]) != int(retained.numel()) or int(points.shape[-1]) < 3:
        return {"available": False, "reason": "missing_or_mismatched_points"}

    points_cpu = points.detach().cpu().float()
    retained_cpu = retained.detach().cpu().bool()
    skeleton_cpu = skeleton_mask.detach().cpu().bool()
    learned_cpu = learned_mask.detach().cpu().bool()
    fallback_cpu = fallback_mask.detach().cpu().bool()
    repair_cpu = length_repair_mask.detach().cpu().bool()
    skeleton_plus_learned_cpu = skeleton_cpu | learned_cpu

    total_original_km = 0.0
    total_retained_km = 0.0
    total_skeleton_km = 0.0
    total_skeleton_plus_learned_km = 0.0
    total_fallback_km = 0.0
    trajectory_rows: list[dict[str, float | int]] = []
    preservation_values: list[float] = []
    max_gap_values: list[float] = []
    mean_gap_values: list[float] = []

    for trajectory_id, (start, end) in enumerate(boundaries):
        start_i = int(start)
        end_i = int(end)
        if end_i - start_i < 2:
            continue
        local_points = points_cpu[start_i:end_i]
        original_km = _polyline_length_km(local_points)
        if original_km <= 1e-9:
            continue
        local_retained = retained_cpu[start_i:end_i]
        local_skeleton = skeleton_cpu[start_i:end_i]
        local_skeleton_plus_learned = skeleton_plus_learned_cpu[start_i:end_i]
        local_fallback = fallback_cpu[start_i:end_i]

        retained_km = (
            _polyline_length_km(local_points[local_retained])
            if int(local_retained.sum().item()) >= 2
            else 0.0
        )
        skeleton_km = (
            _polyline_length_km(local_points[local_skeleton])
            if int(local_skeleton.sum().item()) >= 2
            else 0.0
        )
        skeleton_plus_learned_km = (
            _polyline_length_km(local_points[local_skeleton_plus_learned])
            if int(local_skeleton_plus_learned.sum().item()) >= 2
            else 0.0
        )
        fallback_with_skeleton = local_skeleton | local_fallback
        fallback_km = (
            _polyline_length_km(local_points[fallback_with_skeleton])
            if int(fallback_with_skeleton.sum().item()) >= 2
            else 0.0
        )
        max_gap, mean_gap = _retained_index_gap_stats(local_retained)
        preservation = float(max(0.0, min(1.0, retained_km / original_km)))

        total_original_km += original_km
        total_retained_km += retained_km
        total_skeleton_km += skeleton_km
        total_skeleton_plus_learned_km += skeleton_plus_learned_km
        total_fallback_km += fallback_km
        preservation_values.append(preservation)
        max_gap_values.append(float(max_gap))
        mean_gap_values.append(float(mean_gap))
        trajectory_rows.append(
            {
                "trajectory_id": int(trajectory_id),
                "original_length_km": float(original_km),
                "retained_length_km": float(retained_km),
                "retained_length_preservation": preservation,
                "retained_count": int(local_retained.sum().item()),
                "learned_count": int(learned_cpu[start_i:end_i].sum().item()),
                "length_repair_count": int(repair_cpu[start_i:end_i].sum().item()),
                "max_retained_index_gap": int(max_gap),
                "mean_retained_index_gap": float(mean_gap),
            }
        )

    if total_original_km <= 1e-9:
        return {"available": False, "reason": "zero_original_length"}

    worst = sorted(
        trajectory_rows,
        key=lambda row: (float(row["retained_length_preservation"]), -float(row["original_length_km"])),
    )[:5]
    retained_preservation = float(total_retained_km / total_original_km)
    skeleton_preservation = float(total_skeleton_km / total_original_km)
    skeleton_plus_learned_preservation = float(total_skeleton_plus_learned_km / total_original_km)
    return {
        "available": True,
        "trajectory_count": int(len(trajectory_rows)),
        "total_original_length_km": float(total_original_km),
        "retained_length_km": float(total_retained_km),
        "retained_length_preservation": retained_preservation,
        "skeleton_length_preservation": skeleton_preservation,
        "skeleton_plus_learned_length_preservation": skeleton_plus_learned_preservation,
        "fallback_with_skeleton_length_preservation": float(total_fallback_km / total_original_km),
        "learned_length_gain_over_skeleton": float(skeleton_plus_learned_preservation - skeleton_preservation),
        "fallback_length_gain_over_skeleton": float(total_fallback_km / total_original_km - skeleton_preservation),
        "trajectory_length_preservation_min": _quantile(preservation_values, 0.0),
        "trajectory_length_preservation_p10": _quantile(preservation_values, 0.10),
        "trajectory_length_preservation_p50": _quantile(preservation_values, 0.50),
        "trajectory_length_preservation_p90": _quantile(preservation_values, 0.90),
        "trajectory_length_preservation_below_0_8_count": int(sum(value < 0.80 for value in preservation_values)),
        "trajectory_length_preservation_below_0_8_fraction": float(
            sum(value < 0.80 for value in preservation_values) / max(1, len(preservation_values))
        ),
        "trajectory_length_preservation_below_0_5_count": int(sum(value < 0.50 for value in preservation_values)),
        "trajectory_max_retained_index_gap_p50": _quantile(max_gap_values, 0.50),
        "trajectory_max_retained_index_gap_p90": _quantile(max_gap_values, 0.90),
        "trajectory_max_retained_index_gap_max": _quantile(max_gap_values, 1.0),
        "trajectory_mean_retained_index_gap_p50": _quantile(mean_gap_values, 0.50),
        "worst_trajectories": worst,
    }


def _mask_length_preservation(
    *,
    points: torch.Tensor | None,
    mask: torch.Tensor,
    boundaries: list[tuple[int, int]],
) -> float | None:
    """Return aggregate retained/original path-length ratio for one mask."""
    if points is None or int(points.shape[0]) != int(mask.numel()) or int(points.shape[-1]) < 3:
        return None
    points_cpu = points.detach().cpu().float()
    mask_cpu = mask.detach().cpu().bool()
    original_km = 0.0
    retained_km = 0.0
    for start, end in boundaries:
        start_i = int(start)
        end_i = int(end)
        if end_i - start_i < 2:
            continue
        local_points = points_cpu[start_i:end_i]
        local_original = _polyline_length_km(local_points)
        if local_original <= 1e-9:
            continue
        local_mask = mask_cpu[start_i:end_i]
        local_retained = (
            _polyline_length_km(local_points[local_mask])
            if int(local_mask.sum().item()) >= 2
            else 0.0
        )
        original_km += local_original
        retained_km += local_retained
    if original_km <= 1e-9:
        return None
    return float(retained_km / original_km)


def _fill_missing_by_length_gain(
    *,
    retained: torch.Tensor,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    budget: int,
) -> int:
    """Fill any diagnostic under-budget slots by query-free path-length gain."""
    missing = max(0, int(budget) - int(retained.sum().item()))
    if missing <= 0:
        return 0
    filled = 0
    points_cpu = points.detach().cpu().float()
    for trajectory_id, (start, end) in enumerate(boundaries):
        if filled >= missing:
            break
        start_i = int(start)
        end_i = int(end)
        if end_i - start_i < 3:
            continue
        local_retained = retained[start_i:end_i].detach().cpu().bool()
        retained_indices = torch.where(local_retained)[0]
        if int(retained_indices.numel()) < 2:
            continue
        candidate_scores = torch.zeros((end_i - start_i,), dtype=torch.float32)
        candidate_scores[local_retained] = -float("inf")
        while filled < missing:
            finite = torch.isfinite(candidate_scores)
            if not bool(finite.any().item()):
                break
            retained_indices = torch.where(local_retained)[0]
            gain_scores = _length_gain_scores(points_cpu[start_i:end_i], retained_indices, candidate_scores)
            positive_gain = finite & (gain_scores > 1e-9)
            if not bool(positive_gain.any().item()):
                break
            gain_scores[~positive_gain] = -float("inf")
            choice = deterministic_topk_with_jitter(
                gain_scores,
                1,
                trajectory_id * 11003 + filled,
            )
            if int(choice.numel()) == 0:
                break
            idx = int(choice[0].item())
            local_retained[idx] = True
            candidate_scores[idx] = -float("inf")
            retained[start_i + idx] = True
            filled += 1
    return int(filled)


def _allocation_point_selection_diagnostics(
    *,
    scores: torch.Tensor,
    points: torch.Tensor | None,
    boundaries: list[tuple[int, int]],
    compression_ratio: float,
    segment_rows: list[dict[str, Any]],
    segment_allocations: dict[int, int],
    skeleton_retained: torch.Tensor,
    primary_retained: torch.Tensor,
    budget: int,
    min_temporal_spacing_fraction_within_segment: float,
) -> dict[str, Any]:
    """Separate segment-allocation length capacity from within-segment point choice."""
    if points is None or int(points.shape[0]) != int(scores.numel()) or not segment_allocations:
        return {"available": False, "reason": "missing_points_or_allocations"}

    primary_length = _mask_length_preservation(points=points, mask=primary_retained, boundaries=boundaries)
    if primary_length is None:
        return {"available": False, "reason": "length_preservation_unavailable"}

    length_only_retained = skeleton_retained.detach().cpu().bool().clone()
    scores_cpu = scores.detach().cpu().float()
    points_cpu = points.detach().cpu().float()
    for segment_idx, keep_count in segment_allocations.items():
        if int(keep_count) <= 0:
            continue
        row = segment_rows[int(segment_idx)]
        start = int(row["start"])
        end = int(row["end"])
        trajectory_id = int(row["trajectory_id"])
        trajectory_start, trajectory_end = boundaries[trajectory_id]
        trajectory_start = int(trajectory_start)
        trajectory_end = int(trajectory_end)
        trajectory_scores = torch.full(
            (max(0, trajectory_end - trajectory_start),),
            -float("inf"),
            dtype=torch.float32,
        )
        segment_local_start = max(0, start - trajectory_start)
        segment_local_end = min(int(trajectory_scores.numel()), end - trajectory_start)
        if segment_local_end <= segment_local_start:
            continue
        trajectory_scores[segment_local_start:segment_local_end] = scores_cpu[start:end]
        existing = torch.where(length_only_retained[trajectory_start:trajectory_end])[0]
        min_spacing = int(math.floor(float(end - start) * float(min_temporal_spacing_fraction_within_segment)))
        selected = _select_with_spacing(
            trajectory_scores,
            int(keep_count),
            trajectory_id=trajectory_id,
            existing_indices=existing,
            min_spacing=min_spacing,
            local_points=points_cpu[trajectory_start:trajectory_end],
            geometry_gain_weight=1.0,
        )
        absolute_selected = trajectory_start + selected
        length_only_retained[absolute_selected] = True

    length_fill_count = _fill_missing_by_length_gain(
        retained=length_only_retained,
        points=points_cpu,
        boundaries=boundaries,
        budget=int(budget),
    )
    length_only_preservation = _mask_length_preservation(
        points=points_cpu,
        mask=length_only_retained,
        boundaries=boundaries,
    )
    if length_only_preservation is None:
        return {"available": False, "reason": "counterfactual_length_unavailable"}
    retained_count = int(length_only_retained.sum().item())
    gate_target = 0.80
    return {
        "available": True,
        "diagnostic_only": True,
        "description": "Same learned segment allocations, but length-only point choice inside those allocations.",
        "compression_ratio": float(compression_ratio),
        "total_budget_count": int(budget),
        "primary_retained_stage": "pre_length_repair",
        "primary_retained_count": int(primary_retained.sum().item()),
        "counterfactual_retained_count": retained_count,
        "counterfactual_under_budget_count": max(0, int(budget) - retained_count),
        "counterfactual_length_fill_count": int(length_fill_count),
        "primary_length_preservation": float(primary_length),
        "same_allocation_length_only_point_selection_preservation": float(length_only_preservation),
        "same_allocation_length_only_delta": float(length_only_preservation - primary_length),
        "length_gate_target": float(gate_target),
        "same_allocation_length_only_gate_would_pass": bool(length_only_preservation >= gate_target),
        "component_diagnosis": (
            "point_selection_can_clear_length_with_current_allocation"
            if length_only_preservation >= gate_target
            else "current_segment_allocation_cannot_clear_length_even_with_length_only_point_selection"
        ),
    }


def _segment_score_stats(segment_rows: list[dict[str, Any]]) -> dict[str, float | int]:
    """Return compact segment score diagnostics."""
    if not segment_rows:
        return {
            "segment_score_count": 0,
            "segment_score_min": 0.0,
            "segment_score_max": 0.0,
            "segment_score_mean": 0.0,
            "segment_score_std": 0.0,
            "segment_score_span": 0.0,
        }
    values = torch.tensor(
        [
            float(row.get("score", 0.0))
            if math.isfinite(float(row.get("score", 0.0)))
            else 0.0
            for row in segment_rows
        ],
        dtype=torch.float32,
    )
    return {
        "segment_score_count": int(values.numel()),
        "segment_score_min": float(values.min().item()),
        "segment_score_max": float(values.max().item()),
        "segment_score_mean": float(values.mean().item()),
        "segment_score_std": float(values.std(unbiased=False).item()),
        "segment_score_span": float((values.max() - values.min()).item()),
    }


def _segment_allocation_weights(segment_rows: list[dict[str, Any]]) -> list[float]:
    """Return positive row weights; equal scores degrade to uniform allocation."""
    if not segment_rows:
        return []
    raw_scores = [
        float(row.get("score", 0.0))
        if math.isfinite(float(row.get("score", 0.0)))
        else 0.0
        for row in segment_rows
    ]
    min_score = min(raw_scores)
    max_score = max(raw_scores)
    span = max_score - min_score
    if span <= 1e-12:
        return [1.0 for _row in segment_rows]
    return [
        SEGMENT_ALLOCATION_WEIGHT_FLOOR + ((score - min_score) / span)
        for score in raw_scores
    ]


def _allocate_segment_budgets(
    *,
    segment_rows: list[dict[str, Any]],
    retained: torch.Tensor,
    remaining: int,
    budget: int,
    boundaries: list[tuple[int, int]],
    max_budget_share_per_ship: float,
    fairness_preallocation_enabled: bool = True,
) -> dict[int, int]:
    """Allocate learned slots with score-weighted diminishing returns."""
    if remaining <= 0 or not segment_rows:
        return {}
    valid_trajectory_count = sum(1 for start, end in boundaries if int(end - start) > 0)
    share_cap = int(math.ceil(float(budget) * max(0.01, min(1.0, float(max_budget_share_per_ship)))))
    fair_share_cap = int(math.ceil(float(budget) / float(max(1, valid_trajectory_count))))
    max_per_ship = max(1, share_cap, fair_share_cap)
    ship_allocations = {idx: int(retained[start:end].sum().item()) for idx, (start, end) in enumerate(boundaries)}
    segment_allocations: dict[int, int] = {}
    weights = _segment_allocation_weights(segment_rows)
    remaining_slots = int(remaining)

    # Trajectories with enough total learned budget should not be reduced to
    # endpoints-only retention. This is query-free sanity structure, so expose
    # it as a switch and as a diagnostic ablation rather than hiding it.
    if (
        fairness_preallocation_enabled
        and remaining_slots >= max(1, valid_trajectory_count)
    ):
        trajectory_best_rows: dict[int, tuple[float, int, int]] = {}
        for segment_idx, row in enumerate(segment_rows):
            trajectory_id = int(row["trajectory_id"])
            start = int(row["start"])
            score = float(row["score"])
            best = trajectory_best_rows.get(trajectory_id)
            if best is None or score > best[0] or (score == best[0] and start < best[1]):
                trajectory_best_rows[trajectory_id] = (score, start, segment_idx)

        for _, _start, segment_idx in sorted(
            trajectory_best_rows.values(),
            key=lambda item: (float(item[0]), -int(item[1])),
            reverse=True,
        ):
            if remaining_slots <= 0:
                break
            row = segment_rows[segment_idx]
            trajectory_id = int(row["trajectory_id"])
            if ship_allocations.get(trajectory_id, 0) >= max_per_ship:
                continue
            start = int(row["start"])
            end = int(row["end"])
            capacity = int(row["length"]) - int(retained[start:end].sum().item()) - int(segment_allocations.get(segment_idx, 0))
            if capacity <= 0:
                continue
            segment_allocations[segment_idx] = int(segment_allocations.get(segment_idx, 0)) + 1
            ship_allocations[trajectory_id] = int(ship_allocations.get(trajectory_id, 0)) + 1
            remaining_slots -= 1

    if remaining_slots <= 0:
        return segment_allocations

    while remaining_slots > 0:
        best_idx: int | None = None
        best_key: tuple[float, int, float, int] | None = None
        for segment_idx, row in enumerate(segment_rows):
            trajectory_id = int(row["trajectory_id"])
            if ship_allocations.get(trajectory_id, 0) >= max_per_ship:
                continue
            current = int(segment_allocations.get(segment_idx, 0))
            start = int(row["start"])
            end = int(row["end"])
            capacity = int(row["length"]) - int(retained[start:end].sum().item()) - current
            if capacity <= 0:
                continue
            weight = max(1e-6, float(weights[segment_idx]))
            priority = math.log(weight) - math.log(float(current + 1))
            key = (priority, -current, float(row["score"]), -start)
            if best_key is None or key > best_key:
                best_key = key
                best_idx = segment_idx
        if best_idx is None:
            break
        row = segment_rows[best_idx]
        trajectory_id = int(row["trajectory_id"])
        segment_allocations[best_idx] = int(segment_allocations.get(best_idx, 0)) + 1
        ship_allocations[trajectory_id] = int(ship_allocations.get(trajectory_id, 0)) + 1
        remaining_slots -= 1
    return segment_allocations


def _normalize_candidate_values(values: torch.Tensor, finite: torch.Tensor) -> torch.Tensor:
    """Min-max normalize finite candidate values, keeping invalid entries at -inf."""
    out = torch.full_like(values.float(), -float("inf"))
    if not bool(finite.any().item()):
        return out
    finite_values = values.float()[finite]
    min_value = finite_values.min()
    span = finite_values.max() - min_value
    if float(span.item()) <= 1e-12:
        out[finite] = 0.0
    else:
        out[finite] = (finite_values - min_value) / span
    return out


def _local_distance_km(local_points: torch.Tensor, left_idx: torch.Tensor, right_idx: torch.Tensor) -> torch.Tensor:
    """Return approximate lat/lon distance in km for local index pairs."""
    left = local_points[left_idx.long()]
    right = local_points[right_idx.long()]
    lat1 = left[:, 1].float()
    lon1 = left[:, 2].float()
    lat2 = right[:, 1].float()
    lon2 = right[:, 2].float()
    lat_mid = torch.deg2rad((lat1 + lat2) * 0.5)
    dy = (lat2 - lat1) * 111.32
    dx = (lon2 - lon1) * 111.32 * torch.clamp(torch.cos(lat_mid).abs(), min=0.10)
    return torch.sqrt(dx * dx + dy * dy)


def _length_gain_scores(
    local_points: torch.Tensor | None,
    retained_indices: torch.Tensor,
    candidate_scores: torch.Tensor,
) -> torch.Tensor:
    """Return path-length gain from adding each candidate between retained neighbors."""
    if local_points is None or int(local_points.shape[0]) != int(candidate_scores.numel()):
        return torch.zeros_like(candidate_scores.float())
    finite = torch.isfinite(candidate_scores)
    retained_sorted = retained_indices.to(device=candidate_scores.device, dtype=torch.long).unique(sorted=True)
    if int(retained_sorted.numel()) < 2 or not bool(finite.any().item()):
        return torch.zeros_like(candidate_scores.float())
    candidate_idx = torch.arange(int(candidate_scores.numel()), device=candidate_scores.device)
    pos = torch.searchsorted(retained_sorted, candidate_idx)
    valid = finite & (pos > 0) & (pos < int(retained_sorted.numel()))
    gains = torch.zeros_like(candidate_scores.float())
    if not bool(valid.any().item()):
        return gains
    valid_idx = candidate_idx[valid]
    valid_pos = pos[valid]
    left_idx = retained_sorted[valid_pos - 1]
    right_idx = retained_sorted[valid_pos]
    local_points_device = local_points.to(device=candidate_scores.device)
    via_candidate = (
        _local_distance_km(local_points_device, left_idx, valid_idx)
        + _local_distance_km(local_points_device, valid_idx, right_idx)
    )
    shortcut = _local_distance_km(local_points_device, left_idx, right_idx)
    gains[valid] = torch.clamp(via_candidate - shortcut, min=0.0)
    return gains


def _length_loss_scores(
    local_points: torch.Tensor | None,
    retained_indices: torch.Tensor,
    removable_indices: torch.Tensor,
) -> torch.Tensor:
    """Return path-length loss from removing retained candidate indices."""
    if (
        local_points is None
        or int(retained_indices.numel()) < 3
        or int(removable_indices.numel()) <= 0
        or int(local_points.shape[0]) <= 0
    ):
        return torch.full((int(removable_indices.numel()),), float("inf"), dtype=torch.float32)
    retained_sorted = retained_indices.to(dtype=torch.long).unique(sorted=True)
    removable = removable_indices.to(device=retained_sorted.device, dtype=torch.long)
    pos = torch.searchsorted(retained_sorted, removable)
    valid = (pos > 0) & (pos < int(retained_sorted.numel()) - 1)
    losses = torch.full((int(removable.numel()),), float("inf"), dtype=torch.float32, device=retained_sorted.device)
    if not bool(valid.any().item()):
        return losses.cpu()
    valid_removable = removable[valid]
    valid_pos = pos[valid]
    left_idx = retained_sorted[valid_pos - 1]
    right_idx = retained_sorted[valid_pos + 1]
    local_points_device = local_points.to(device=retained_sorted.device)
    via_removed = (
        _local_distance_km(local_points_device, left_idx, valid_removable)
        + _local_distance_km(local_points_device, valid_removable, right_idx)
    )
    shortcut = _local_distance_km(local_points_device, left_idx, right_idx)
    losses[valid] = torch.clamp(via_removed - shortcut, min=0.0)
    return losses.cpu()


def _apply_length_repair_swaps(
    *,
    scores: torch.Tensor,
    points: torch.Tensor | None,
    boundaries: list[tuple[int, int]],
    retained: torch.Tensor,
    learned_mask: torch.Tensor,
    fallback_mask: torch.Tensor,
    length_repair_mask: torch.Tensor,
    repair_fraction: float,
) -> int:
    """Swap a bounded share of learned slots toward query-free path-length gain."""
    if points is None or int(points.shape[0]) != int(scores.numel()) or float(repair_fraction) <= 0.0:
        return 0
    fraction = max(0.0, min(1.0, float(repair_fraction)))
    total_swaps = 0
    points_cpu = points.detach().cpu().float()
    scores_cpu = scores.detach().cpu().float()

    for trajectory_id, (start, end) in enumerate(boundaries):
        start_i = int(start)
        end_i = int(end)
        if end_i - start_i < 3:
            continue
        local_retained = retained[start_i:end_i].detach().cpu().bool().clone()
        local_learned = learned_mask[start_i:end_i].detach().cpu().bool().clone()
        local_fallback = fallback_mask[start_i:end_i].detach().cpu().bool().clone()
        local_repair = length_repair_mask[start_i:end_i].detach().cpu().bool().clone()
        local_removable = local_learned | local_fallback
        removable_count = int(local_removable.sum().item())
        max_swaps = min(removable_count, int(math.ceil(fraction * float(removable_count))))
        if max_swaps <= 0:
            continue
        local_scores = scores_cpu[start_i:end_i]
        local_points = points_cpu[start_i:end_i]

        for step in range(max_swaps):
            retained_indices = torch.where(local_retained)[0]
            if int(retained_indices.numel()) < 3:
                break
            candidate_scores = local_scores.clone()
            candidate_scores[local_retained] = -float("inf")
            finite_candidates = torch.isfinite(candidate_scores)
            if not bool(finite_candidates.any().item()):
                break
            gain_scores = _length_gain_scores(local_points, retained_indices, candidate_scores)
            positive_gain = finite_candidates & (gain_scores > 1e-9)
            if not bool(positive_gain.any().item()):
                break
            normalized_gain = _normalize_candidate_values(gain_scores, positive_gain)
            normalized_score = _normalize_candidate_values(candidate_scores, positive_gain)
            candidate_key = 0.90 * normalized_gain + 0.10 * normalized_score
            candidate_key[~positive_gain] = -float("inf")
            add_idx_tensor = deterministic_topk_with_jitter(
                candidate_key,
                1,
                trajectory_id * 65537 + step,
            )
            if int(add_idx_tensor.numel()) <= 0:
                break
            add_idx = int(add_idx_tensor[0].item())

            removable_indices = torch.where(local_removable & local_retained)[0]
            if int(removable_indices.numel()) <= 0:
                break
            removal_losses = _length_loss_scores(local_points, retained_indices, removable_indices)
            finite_removable = torch.isfinite(removal_losses)
            if not bool(finite_removable.any().item()):
                break
            removable_scores = local_scores[removable_indices].float()
            normalized_loss = _normalize_candidate_values(removal_losses, finite_removable)
            normalized_removable_score = _normalize_candidate_values(removable_scores, finite_removable)
            removal_key = (1.0 - normalized_loss) + 0.10 * (1.0 - normalized_removable_score)
            removal_key[~finite_removable] = -float("inf")
            remove_choice = deterministic_topk_with_jitter(
                removal_key,
                1,
                trajectory_id * 91733 + step,
            )
            if int(remove_choice.numel()) <= 0:
                break
            remove_idx = int(removable_indices[int(remove_choice[0].item())].item())
            if float(gain_scores[add_idx].item()) <= float(removal_losses[int(remove_choice[0].item())].item()) + 1e-9:
                break

            local_retained[remove_idx] = False
            local_learned[remove_idx] = False
            local_fallback[remove_idx] = False
            local_removable[remove_idx] = False
            local_retained[add_idx] = True
            local_repair[add_idx] = True
            total_swaps += 1

        retained[start_i:end_i] = local_retained.to(device=retained.device)
        learned_mask[start_i:end_i] = local_learned.to(device=learned_mask.device)
        fallback_mask[start_i:end_i] = local_fallback.to(device=fallback_mask.device)
        length_repair_mask[start_i:end_i] = local_repair.to(device=length_repair_mask.device)

    return int(total_swaps)


def _selector_trace(
    *,
    retained: torch.Tensor,
    skeleton_mask: torch.Tensor,
    learned_mask: torch.Tensor,
    fallback_mask: torch.Tensor,
    length_repair_mask: torch.Tensor,
    boundaries: list[tuple[int, int]],
    compression_ratio: float,
    budget: int,
    skeleton_cap: int,
    segment_rows: list[dict[str, Any]] | None,
    segment_allocations: dict[int, int],
    segment_count: int,
    segment_score_source: str,
    segment_score_stats: dict[str, float | int] | None = None,
    segment_budget_allocation_method: str = "none",
    fairness_preallocation_enabled: bool = True,
    geometry_gain_weight: float = GEOMETRY_TIE_BREAKER_WEIGHT,
    segment_score_point_blend_weight: float = SEGMENT_SCORE_POINT_BLEND_WEIGHT,
    length_repair_fraction: float = 0.0,
    length_repair_swap_count: int = 0,
    points: torch.Tensor | None = None,
    allocation_point_selection_diagnostics: dict[str, Any] | None = None,
    pre_repair_segment_source_attribution: dict[str, Any] | None = None,
    pre_repair_retained_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Return JSON-serializable attribution for the retained mask."""
    retained_count = int(retained.sum().item())
    skeleton_count = int(skeleton_mask.sum().item())
    learned_count = int(learned_mask.sum().item())
    fallback_count = int(fallback_mask.sum().item())
    length_repair_count = int(length_repair_mask.sum().item())
    attributed = skeleton_mask | learned_mask | fallback_mask | length_repair_mask
    unattributed_count = int((retained & ~attributed).sum().item())
    trajectory_learned_counts = _trajectory_counts(learned_mask, boundaries)
    trajectories_with_learned = sum(1 for count in trajectory_learned_counts if int(count) > 0)
    valid_trajectory_count = sum(1 for start, end in boundaries if int(end - start) > 0)
    entropy, entropy_normalized = _entropy(list(segment_allocations.values()))
    return {
        "schema_version": int(LEARNED_SEGMENT_BUDGET_TRACE_SCHEMA_VERSION),
        "selector_type": "learned_segment_budget_v1",
        "compression_ratio": float(compression_ratio),
        "total_point_count": int(retained.numel()),
        "total_budget_count": int(budget),
        "retained_count": retained_count,
        "minimal_skeleton_slot_cap": int(skeleton_cap),
        "skeleton_retained_count": skeleton_count,
        "skeleton_cap_exceeded_for_endpoint_sanity": bool(skeleton_count > int(skeleton_cap)),
        "learned_controlled_retained_slots": learned_count,
        "learned_controlled_retained_slot_fraction": float(learned_count / max(1, int(budget))),
        "learned_fraction_of_retained_count": float(learned_count / max(1, retained_count)),
        "fallback_retained_count": fallback_count,
        "length_repair_retained_count": length_repair_count,
        "length_repair_fraction": float(length_repair_fraction),
        "length_repair_swap_count": int(length_repair_swap_count),
        "unattributed_retained_count": unattributed_count,
        "trajectory_count": int(valid_trajectory_count),
        "trajectories_with_at_least_one_learned_decision": int(trajectories_with_learned),
        "trajectories_with_zero_learned_decisions": int(max(0, valid_trajectory_count - trajectories_with_learned)),
        "trajectory_learned_decision_counts": trajectory_learned_counts,
        "trajectory_skeleton_counts": _trajectory_counts(skeleton_mask, boundaries),
        "trajectory_fallback_counts": _trajectory_counts(fallback_mask, boundaries),
        "segments_considered_count": int(segment_count),
        "segments_with_learned_budget": int(sum(1 for count in segment_allocations.values() if int(count) > 0)),
        "segment_budget_allocation_count": int(sum(int(count) for count in segment_allocations.values())),
        "segment_budget_entropy": entropy,
        "segment_budget_entropy_normalized": entropy_normalized,
        "segment_score_source": str(segment_score_source),
        "segment_budget_allocation_method": str(segment_budget_allocation_method),
        "trajectory_fairness_preallocation_enabled": bool(fairness_preallocation_enabled),
        "geometry_tie_breaker_weight": float(geometry_gain_weight),
        "segment_score_point_blend_weight": float(segment_score_point_blend_weight),
        "no_fixed_85_percent_temporal_scaffold": True,
        "point_attribution_available": True,
        "geometry_diagnostics": _selector_geometry_diagnostics(
            points=points,
            retained=retained,
            skeleton_mask=skeleton_mask,
            learned_mask=learned_mask,
            fallback_mask=fallback_mask,
            length_repair_mask=length_repair_mask,
            boundaries=boundaries,
        ),
        "allocation_point_selection_diagnostics": (
            allocation_point_selection_diagnostics
            if allocation_point_selection_diagnostics is not None
            else {"available": False, "reason": "not_run"}
        ),
        "segment_source_attribution": _segment_source_attribution(
            segment_rows=[] if segment_rows is None else segment_rows,
            segment_allocations=segment_allocations,
            retained=retained,
            skeleton_mask=skeleton_mask,
            learned_mask=learned_mask,
            fallback_mask=fallback_mask,
            length_repair_mask=length_repair_mask,
        ),
        "pre_repair_segment_source_attribution": (
            pre_repair_segment_source_attribution
            if pre_repair_segment_source_attribution is not None
            else {"available": False, "reason": "not_run"}
        ),
        "pre_repair_retained_mask": _mask_indices_payload(pre_repair_retained_mask),
        **(segment_score_stats or {}),
    }


def _select_with_spacing(
    local_scores: torch.Tensor,
    keep_count: int,
    *,
    trajectory_id: int,
    existing_indices: torch.Tensor,
    min_spacing: int,
    local_points: torch.Tensor | None = None,
    geometry_gain_weight: float = 0.05,
    segment_aux_scores: torch.Tensor | None = None,
    segment_score_weight: float = 0.0,
) -> torch.Tensor:
    """Select top scores with simple non-maximum spacing."""
    keep = max(0, min(int(keep_count), int(local_scores.numel())))
    if keep <= 0:
        return torch.empty((0,), dtype=torch.long, device=local_scores.device)
    candidate_scores = local_scores.clone()
    if int(existing_indices.numel()) > 0:
        candidate_scores[existing_indices.to(device=local_scores.device, dtype=torch.long)] = -float("inf")
    selected: list[torch.Tensor] = []
    retained_indices = existing_indices.to(device=local_scores.device, dtype=torch.long)
    spacing = max(0, int(min_spacing))
    for step in range(keep):
        finite = torch.isfinite(candidate_scores)
        if not bool(finite.any().item()):
            break
        segment_weight = max(0.0, min(1.0, float(segment_score_weight)))
        score_for_selection = candidate_scores.clone()
        if segment_aux_scores is not None and segment_weight > 0.0:
            segment_scores = segment_aux_scores.to(device=candidate_scores.device, dtype=torch.float32).clone()
            segment_scores[~finite] = -float("inf")
            segment_finite = torch.isfinite(segment_scores)
            if bool(segment_finite.any().item()):
                point_scores_norm = _normalize_candidate_values(score_for_selection, finite)
                segment_scores_norm = _normalize_candidate_values(segment_scores, segment_finite)
                blended = (1.0 - segment_weight) * point_scores_norm + segment_weight * segment_scores_norm
                blended[~finite] = -float("inf")
                score_for_selection = blended

        gain_scores = _length_gain_scores(local_points, retained_indices, score_for_selection)
        normalized_scores = _normalize_candidate_values(score_for_selection, finite)
        normalized_gain = _normalize_candidate_values(gain_scores, finite)
        weight = max(0.0, min(1.0, float(geometry_gain_weight)))
        combined_scores = (1.0 - weight) * normalized_scores + weight * normalized_gain
        combined_scores[~finite] = -float("inf")
        choice = deterministic_topk_with_jitter(combined_scores, 1, trajectory_id * 4099 + step)
        if int(choice.numel()) == 0:
            break
        idx = int(choice[0].item())
        selected.append(choice)
        retained_indices = torch.cat([retained_indices, choice.to(dtype=torch.long)]).unique(sorted=True)
        left = max(0, idx - spacing)
        right = min(int(candidate_scores.numel()), idx + spacing + 1)
        candidate_scores[left:right] = -float("inf")
    if len(selected) < keep:
        finite = torch.isfinite(candidate_scores)
        if bool(finite.any().item()):
            gain_scores = _length_gain_scores(local_points, retained_indices, candidate_scores)
            normalized_scores = _normalize_candidate_values(candidate_scores, finite)
            normalized_gain = _normalize_candidate_values(gain_scores, finite)
            weight = max(0.0, min(1.0, float(geometry_gain_weight)))
            combined_scores = (1.0 - weight) * normalized_scores + weight * normalized_gain
            combined_scores[~finite] = -float("inf")
            fallback = deterministic_topk_with_jitter(
                combined_scores,
                keep - len(selected),
                trajectory_id * 9173 + keep,
            )
            selected.append(fallback)
        if not selected:
            return torch.empty((0,), dtype=torch.long, device=local_scores.device)
    if not selected:
        return torch.empty((0,), dtype=torch.long, device=local_scores.device)
    return torch.cat(selected).unique(sorted=True)[:keep]


def simplify_with_learned_segment_budget_v1_with_trace(
    scores: torch.Tensor,
    boundaries: list[tuple[int, int]],
    compression_ratio: float,
    *,
    segment_size: int = 32,
    min_temporal_spacing_fraction_within_segment: float = 0.10,
    max_budget_share_per_ship: float = 0.20,
    segment_scores: torch.Tensor | None = None,
    segment_point_scores: torch.Tensor | None = None,
    points: torch.Tensor | None = None,
    geometry_gain_weight: float = GEOMETRY_TIE_BREAKER_WEIGHT,
    segment_score_point_blend_weight: float = SEGMENT_SCORE_POINT_BLEND_WEIGHT,
    fairness_preallocation_enabled: bool = True,
    length_repair_fraction: float = 0.0,
    segment_score_source_label: str | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Retain points and return skeleton/learned/fallback attribution."""
    point_total = int(scores.numel())
    retained = torch.zeros((point_total,), dtype=torch.bool, device=scores.device)
    skeleton_mask = torch.zeros_like(retained)
    learned_mask = torch.zeros_like(retained)
    fallback_mask = torch.zeros_like(retained)
    length_repair_mask = torch.zeros_like(retained)
    if point_total <= 0:
        trace = _selector_trace(
            retained=retained,
            skeleton_mask=skeleton_mask,
            learned_mask=learned_mask,
            fallback_mask=fallback_mask,
            length_repair_mask=length_repair_mask,
            boundaries=boundaries,
            compression_ratio=compression_ratio,
            budget=0,
            skeleton_cap=0,
            segment_rows=[],
            segment_allocations={},
            segment_count=0,
            segment_score_source="none",
            segment_budget_allocation_method="none",
            fairness_preallocation_enabled=fairness_preallocation_enabled,
            geometry_gain_weight=geometry_gain_weight,
            segment_score_point_blend_weight=segment_score_point_blend_weight,
            length_repair_fraction=0.0,
            length_repair_swap_count=0,
            points=points,
        )
        return retained, trace
    budget = min(point_total, _total_budget(boundaries, compression_ratio))
    skeleton_cap = int(math.floor(float(budget) * _max_skeleton_fraction(compression_ratio)))

    skeleton_count = 0
    skeleton_candidates: list[tuple[float, int, int]] = []
    for trajectory_id, (start, end) in enumerate(boundaries):
        count = int(end - start)
        if count <= 0:
            continue
        local_budget = min(count, max(2, int(math.ceil(float(compression_ratio) * count))))
        if local_budget >= 2:
            for point_idx in (int(start), int(end - 1)):
                if not retained[point_idx]:
                    retained[point_idx] = True
                    skeleton_mask[point_idx] = True
                    skeleton_count += 1
        else:
            mid = int(start + count // 2)
            candidates = torch.tensor([int(start), mid, int(end - 1)], dtype=torch.long, device=scores.device).unique()
            best = candidates[torch.argmax(scores[candidates].float())]
            skeleton_candidates.append((float(scores[best].item()), trajectory_id, int(best.item())))
    optional_skeleton_slots = max(0, int(skeleton_cap) - int(skeleton_count))
    if optional_skeleton_slots > 0 and skeleton_candidates:
        skeleton_candidates.sort(key=lambda item: (item[0], -item[2]), reverse=True)
        for _score, _trajectory_id, point_idx in skeleton_candidates[:optional_skeleton_slots]:
            if not retained[point_idx]:
                retained[point_idx] = True
                skeleton_mask[point_idx] = True
                skeleton_count += 1
                if skeleton_count >= skeleton_cap:
                    break

    remaining = budget - int(retained.sum().item())
    if remaining <= 0:
        trace = _selector_trace(
            retained=retained,
            skeleton_mask=skeleton_mask,
            learned_mask=learned_mask,
            fallback_mask=fallback_mask,
            length_repair_mask=length_repair_mask,
            boundaries=boundaries,
            compression_ratio=compression_ratio,
            budget=budget,
            skeleton_cap=skeleton_cap,
            segment_rows=[],
            segment_allocations={},
            segment_count=0,
            segment_score_source="none",
            segment_budget_allocation_method="none",
            fairness_preallocation_enabled=fairness_preallocation_enabled,
            geometry_gain_weight=geometry_gain_weight,
            segment_score_point_blend_weight=segment_score_point_blend_weight,
            length_repair_fraction=0.0,
            length_repair_swap_count=0,
            points=points,
        )
        return retained, trace

    if segment_scores is not None and int(segment_scores.numel()) != point_total:
        raise ValueError(
            "segment_scores must match scores length: "
            f"got {int(segment_scores.numel())}, expected {point_total}."
        )
    if segment_point_scores is not None and int(segment_point_scores.numel()) != point_total:
        raise ValueError(
            "segment_point_scores must match scores length: "
            f"got {int(segment_point_scores.numel())}, expected {point_total}."
        )
    point_segment_scores = segment_scores if segment_point_scores is None else segment_point_scores
    segment_rows = _segment_rows(scores, boundaries, segment_size, segment_scores=segment_scores)
    segment_rows.sort(key=lambda row: (float(row["score"]), -int(row["start"])), reverse=True)
    segment_score_source = (
        str(segment_score_source_label)
        if segment_score_source_label is not None
        else "segment_budget_head_mean"
        if segment_scores is not None
        else "point_score_top20_mean"
    )
    segment_score_stats = _segment_score_stats(segment_rows)
    segment_allocations = _allocate_segment_budgets(
        segment_rows=segment_rows,
        retained=retained,
        remaining=remaining,
        budget=budget,
        boundaries=boundaries,
        max_budget_share_per_ship=max_budget_share_per_ship,
        fairness_preallocation_enabled=fairness_preallocation_enabled,
    )
    skeleton_retained_for_diagnostic = retained.detach().cpu().bool().clone()

    for segment_idx, keep_count in segment_allocations.items():
        row = segment_rows[segment_idx]
        start = int(row["start"])
        end = int(row["end"])
        trajectory_id = int(row["trajectory_id"])
        trajectory_start, trajectory_end = boundaries[trajectory_id]
        trajectory_start = int(trajectory_start)
        trajectory_end = int(trajectory_end)
        trajectory_scores = torch.full(
            (max(0, trajectory_end - trajectory_start),),
            -float("inf"),
            dtype=torch.float32,
            device=scores.device,
        )
        segment_local_start = max(0, start - trajectory_start)
        segment_local_end = min(int(trajectory_scores.numel()), end - trajectory_start)
        if segment_local_end <= segment_local_start:
            continue
        trajectory_scores[segment_local_start:segment_local_end] = scores[start:end].float()
        local_retained = retained[trajectory_start:trajectory_end]
        existing = torch.where(local_retained)[0]
        min_spacing = int(math.floor(float(end - start) * float(min_temporal_spacing_fraction_within_segment)))
        segment_aux_scores = None
        segment_score_weight = 0.0
        if point_segment_scores is not None:
            segment_aux_scores = torch.full_like(trajectory_scores, -float("inf"))
            segment_aux_scores[segment_local_start:segment_local_end] = point_segment_scores[start:end].to(
                device=trajectory_scores.device,
                dtype=torch.float32,
            )
            segment_aux_local_scores = segment_aux_scores[segment_local_start:segment_local_end]
            segment_score_finite = torch.isfinite(segment_aux_local_scores)
            if bool(segment_score_finite.any().item()):
                segment_score_weight = float(segment_score_point_blend_weight)
        selected = _select_with_spacing(
            trajectory_scores,
            int(keep_count),
            trajectory_id=trajectory_id,
            existing_indices=existing,
            min_spacing=min_spacing,
            local_points=None if points is None else points[trajectory_start:trajectory_end],
            geometry_gain_weight=float(geometry_gain_weight),
            segment_aux_scores=segment_aux_scores,
            segment_score_weight=float(segment_score_weight) if point_segment_scores is not None else 0.0,
        )
        absolute_selected = trajectory_start + selected
        new_selected = absolute_selected[~retained[absolute_selected]]
        retained[new_selected] = True
        learned_mask[new_selected] = True

    if int(retained.sum().item()) < budget:
        candidate_scores = scores.float().clone()
        candidate_scores[retained] = -float("inf")
        missing = min(budget - int(retained.sum().item()), int(torch.isfinite(candidate_scores).sum().item()))
        if missing > 0:
            fallback_selected = deterministic_topk_with_jitter(candidate_scores, missing, point_total + 31337)
            retained[fallback_selected] = True
            fallback_mask[fallback_selected] = True
    pre_repair_retained_for_diagnostic = retained.detach().cpu().bool().clone()
    pre_repair_learned_for_diagnostic = learned_mask.detach().cpu().bool().clone()
    pre_repair_fallback_for_diagnostic = fallback_mask.detach().cpu().bool().clone()
    pre_repair_length_repair_for_diagnostic = torch.zeros_like(pre_repair_retained_for_diagnostic)
    allocation_diagnostics = _allocation_point_selection_diagnostics(
        scores=scores,
        points=points,
        boundaries=boundaries,
        compression_ratio=float(compression_ratio),
        segment_rows=segment_rows,
        segment_allocations=segment_allocations,
        skeleton_retained=skeleton_retained_for_diagnostic,
        primary_retained=pre_repair_retained_for_diagnostic,
        budget=budget,
        min_temporal_spacing_fraction_within_segment=float(min_temporal_spacing_fraction_within_segment),
    )
    pre_repair_source_attribution = _segment_source_attribution(
        segment_rows=segment_rows,
        segment_allocations=segment_allocations,
        retained=pre_repair_retained_for_diagnostic,
        skeleton_mask=skeleton_retained_for_diagnostic,
        learned_mask=pre_repair_learned_for_diagnostic,
        fallback_mask=pre_repair_fallback_for_diagnostic,
        length_repair_mask=pre_repair_length_repair_for_diagnostic,
    )
    length_repair_swap_count = _apply_length_repair_swaps(
        scores=scores,
        points=points,
        boundaries=boundaries,
        retained=retained,
        learned_mask=learned_mask,
        fallback_mask=fallback_mask,
        length_repair_mask=length_repair_mask,
        repair_fraction=float(length_repair_fraction),
    )
    trace = _selector_trace(
        retained=retained,
        skeleton_mask=skeleton_mask,
        learned_mask=learned_mask,
        fallback_mask=fallback_mask,
        length_repair_mask=length_repair_mask,
        boundaries=boundaries,
        compression_ratio=compression_ratio,
        budget=budget,
        skeleton_cap=skeleton_cap,
        segment_rows=segment_rows,
        segment_allocations=segment_allocations,
        segment_count=len(segment_rows),
        segment_score_source=segment_score_source,
        segment_score_stats=segment_score_stats,
        segment_budget_allocation_method="score_weighted_diminishing_priority",
        fairness_preallocation_enabled=fairness_preallocation_enabled,
        geometry_gain_weight=geometry_gain_weight,
        segment_score_point_blend_weight=segment_score_point_blend_weight,
        length_repair_fraction=float(length_repair_fraction),
        length_repair_swap_count=length_repair_swap_count,
        points=points,
        allocation_point_selection_diagnostics=allocation_diagnostics,
        pre_repair_segment_source_attribution=pre_repair_source_attribution,
        pre_repair_retained_mask=pre_repair_retained_for_diagnostic,
    )
    return retained, trace


def simplify_with_learned_segment_budget_v1(
    scores: torch.Tensor,
    boundaries: list[tuple[int, int]],
    compression_ratio: float,
    *,
    segment_size: int = 32,
    min_temporal_spacing_fraction_within_segment: float = 0.10,
    max_budget_share_per_ship: float = 0.20,
    segment_scores: torch.Tensor | None = None,
    segment_point_scores: torch.Tensor | None = None,
    points: torch.Tensor | None = None,
    geometry_gain_weight: float = GEOMETRY_TIE_BREAKER_WEIGHT,
    segment_score_point_blend_weight: float = SEGMENT_SCORE_POINT_BLEND_WEIGHT,
    fairness_preallocation_enabled: bool = True,
    length_repair_fraction: float = 0.0,
    segment_score_source_label: str | None = None,
) -> torch.Tensor:
    """Retain a minimal skeleton, then allocate remaining budget by learned segment value."""
    retained, _trace = simplify_with_learned_segment_budget_v1_with_trace(
        scores,
        boundaries,
        compression_ratio,
        segment_size=segment_size,
        min_temporal_spacing_fraction_within_segment=min_temporal_spacing_fraction_within_segment,
        max_budget_share_per_ship=max_budget_share_per_ship,
        segment_scores=segment_scores,
        segment_point_scores=segment_point_scores,
        points=points,
        geometry_gain_weight=geometry_gain_weight,
        segment_score_point_blend_weight=segment_score_point_blend_weight,
        fairness_preallocation_enabled=fairness_preallocation_enabled,
        length_repair_fraction=length_repair_fraction,
        segment_score_source_label=segment_score_source_label,
    )
    return retained


def learned_segment_budget_diagnostics(
    boundaries: list[tuple[int, int]],
    compression_ratios: list[float] | tuple[float, ...],
) -> dict[str, Any]:
    """Return selector contribution diagnostics independent of model scores."""
    rows: list[dict[str, Any]] = []
    trajectory_count = sum(1 for start, end in boundaries if int(end - start) > 0)
    for ratio in compression_ratios:
        budget = _total_budget(boundaries, float(ratio))
        skeleton_cap = int(math.floor(float(budget) * _max_skeleton_fraction(float(ratio))))
        learned_slots = max(0, int(budget) - int(skeleton_cap))
        rows.append(
            {
                "compression_ratio": float(ratio),
                "trajectory_count": int(trajectory_count),
                "total_budget_count": int(budget),
                "minimal_skeleton_slot_cap": int(skeleton_cap),
                "learned_slot_count": int(learned_slots),
                "learned_slot_fraction_of_budget": float(learned_slots / max(1, int(budget))),
                "no_fixed_85_percent_temporal_scaffold": True,
            }
        )
    return {
        "schema_version": int(LEARNED_SEGMENT_BUDGET_SCHEMA_VERSION),
        "selector_type": "learned_segment_budget_v1",
        "budget_rows": rows,
    }
