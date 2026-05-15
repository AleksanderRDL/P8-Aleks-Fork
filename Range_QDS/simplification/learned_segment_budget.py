"""learned_segment_budget_v1 selector."""

from __future__ import annotations

import math
from typing import Any

import torch

from simplification.simplify_trajectories import deterministic_topk_with_jitter

LEARNED_SEGMENT_BUDGET_SCHEMA_VERSION = 2
LEARNED_SEGMENT_BUDGET_TRACE_SCHEMA_VERSION = 1
SEGMENT_ALLOCATION_WEIGHT_FLOOR = 0.50
GEOMETRY_TIE_BREAKER_WEIGHT = 0.12
SEGMENT_SCORE_POINT_BLEND_WEIGHT = 0.05


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
                segment_score_source = "segment_budget_head_mean"
            rows.append(
                {
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
    # endpoints-only retention. Guarantee one learned slot on at least one segment
    # per active trajectory before score-weighted diminishing allocations.
    if remaining_slots >= max(1, valid_trajectory_count):
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


def _selector_trace(
    *,
    retained: torch.Tensor,
    skeleton_mask: torch.Tensor,
    learned_mask: torch.Tensor,
    fallback_mask: torch.Tensor,
    boundaries: list[tuple[int, int]],
    compression_ratio: float,
    budget: int,
    skeleton_cap: int,
    segment_allocations: dict[int, int],
    segment_count: int,
    segment_score_source: str,
    segment_score_stats: dict[str, float | int] | None = None,
    segment_budget_allocation_method: str = "none",
) -> dict[str, Any]:
    """Return JSON-serializable attribution for the retained mask."""
    retained_count = int(retained.sum().item())
    skeleton_count = int(skeleton_mask.sum().item())
    learned_count = int(learned_mask.sum().item())
    fallback_count = int(fallback_mask.sum().item())
    attributed = skeleton_mask | learned_mask | fallback_mask
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
        "no_fixed_85_percent_temporal_scaffold": True,
        "point_attribution_available": True,
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
    points: torch.Tensor | None = None,
    geometry_gain_weight: float = GEOMETRY_TIE_BREAKER_WEIGHT,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Retain points and return skeleton/learned/fallback attribution."""
    point_total = int(scores.numel())
    retained = torch.zeros((point_total,), dtype=torch.bool, device=scores.device)
    skeleton_mask = torch.zeros_like(retained)
    learned_mask = torch.zeros_like(retained)
    fallback_mask = torch.zeros_like(retained)
    if point_total <= 0:
        trace = _selector_trace(
            retained=retained,
            skeleton_mask=skeleton_mask,
            learned_mask=learned_mask,
            fallback_mask=fallback_mask,
            boundaries=boundaries,
            compression_ratio=compression_ratio,
            budget=0,
            skeleton_cap=0,
            segment_allocations={},
            segment_count=0,
            segment_score_source="none",
            segment_budget_allocation_method="none",
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
            boundaries=boundaries,
            compression_ratio=compression_ratio,
            budget=budget,
            skeleton_cap=skeleton_cap,
            segment_allocations={},
            segment_count=0,
            segment_score_source="none",
            segment_budget_allocation_method="none",
        )
        return retained, trace

    if segment_scores is not None and int(segment_scores.numel()) != point_total:
        raise ValueError(
            "segment_scores must match scores length: "
            f"got {int(segment_scores.numel())}, expected {point_total}."
        )
    segment_rows = _segment_rows(scores, boundaries, segment_size, segment_scores=segment_scores)
    segment_rows.sort(key=lambda row: (float(row["score"]), -int(row["start"])), reverse=True)
    segment_score_source = (
        "segment_budget_head_mean" if segment_scores is not None else "point_score_top20_mean"
    )
    segment_score_stats = _segment_score_stats(segment_rows)
    segment_allocations = _allocate_segment_budgets(
        segment_rows=segment_rows,
        retained=retained,
        remaining=remaining,
        budget=budget,
        boundaries=boundaries,
        max_budget_share_per_ship=max_budget_share_per_ship,
    )

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
        if segment_scores is not None:
            segment_aux_scores = torch.full_like(trajectory_scores, -float("inf"))
            segment_aux_scores[segment_local_start:segment_local_end] = segment_scores[start:end].to(
                device=trajectory_scores.device,
                dtype=torch.float32,
            )
            segment_aux_local_scores = segment_aux_scores[segment_local_start:segment_local_end]
            segment_score_finite = torch.isfinite(segment_aux_local_scores)
            if bool(segment_score_finite.any().item()):
                segment_score_weight = float(SEGMENT_SCORE_POINT_BLEND_WEIGHT)
        selected = _select_with_spacing(
            trajectory_scores,
            int(keep_count),
            trajectory_id=trajectory_id,
            existing_indices=existing,
            min_spacing=min_spacing,
            local_points=None if points is None else points[trajectory_start:trajectory_end],
            geometry_gain_weight=float(geometry_gain_weight),
            segment_aux_scores=segment_aux_scores,
            segment_score_weight=float(segment_score_weight) if segment_scores is not None else 0.0,
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
    trace = _selector_trace(
        retained=retained,
        skeleton_mask=skeleton_mask,
        learned_mask=learned_mask,
        fallback_mask=fallback_mask,
        boundaries=boundaries,
        compression_ratio=compression_ratio,
        budget=budget,
        skeleton_cap=skeleton_cap,
        segment_allocations=segment_allocations,
        segment_count=len(segment_rows),
        segment_score_source=segment_score_source,
        segment_score_stats=segment_score_stats,
        segment_budget_allocation_method="score_weighted_diminishing_priority",
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
    points: torch.Tensor | None = None,
    geometry_gain_weight: float = GEOMETRY_TIE_BREAKER_WEIGHT,
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
        points=points,
        geometry_gain_weight=geometry_gain_weight,
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
