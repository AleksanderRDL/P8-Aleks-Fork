"""Factorized QueryUsefulV1 target construction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from queries.range_geometry import points_in_range_box, segment_box_bracket_indices
from queries.query_types import NUM_QUERY_TYPES, QUERY_TYPE_ID_RANGE
from training.factorized_target_diagnostics import (
    factorized_target_diagnostics,
    support_fraction_by_threshold,
)

QUERY_USEFUL_V1_TARGET_MODES = frozenset({"query_useful_v1_factorized"})
QUERY_USEFUL_V1_HEAD_NAMES = (
    "query_hit_probability",
    "conditional_behavior_utility",
    "boundary_event_utility",
    "replacement_representative_value",
    "segment_budget_target",
    "path_length_support_target",
)


@dataclass
class QueryUsefulTargetBundle:
    """Scalar and factorized training labels for QueryUsefulV1."""

    labels: torch.Tensor
    labelled_mask: torch.Tensor
    head_targets: torch.Tensor
    head_mask: torch.Tensor
    diagnostics: dict[str, Any]


def _normalize_0_1(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Normalize non-negative values to [0, 1] on the selected support."""
    out = values.float().clamp(min=0.0)
    support = mask if mask is not None else torch.ones_like(out, dtype=torch.bool)
    if not bool(support.any().item()):
        return torch.zeros_like(out)
    local = out[support]
    max_value = float(local.max().item()) if int(local.numel()) > 0 else 0.0
    if max_value <= 1e-12:
        return torch.zeros_like(out)
    return (out / max_value).clamp(0.0, 1.0)


def _trajectory_change_weights(points: torch.Tensor, boundaries: list[tuple[int, int]]) -> torch.Tensor:
    """Return sparse query-free behavior-change weights per point."""
    n_points = int(points.shape[0])
    if n_points <= 0:
        return torch.empty((0,), dtype=torch.float32, device=points.device)
    weights = torch.zeros((n_points,), dtype=torch.float32, device=points.device)
    indices = torch.arange(n_points, device=points.device)
    prev_idx = torch.clamp(indices - 1, min=0)
    next_idx = torch.clamp(indices + 1, max=n_points - 1)
    if points.shape[1] > 7:
        weights = torch.maximum(weights, points[:, 7].float().clamp(min=0.0))
    if points.shape[1] > 4:
        prev_heading = torch.abs(points[:, 4].float() - points[prev_idx, 4].float())
        next_heading = torch.abs(points[next_idx, 4].float() - points[:, 4].float())
        prev_heading = torch.minimum(prev_heading, 360.0 - prev_heading).clamp(min=0.0) / 180.0
        next_heading = torch.minimum(next_heading, 360.0 - next_heading).clamp(min=0.0) / 180.0
        weights = torch.maximum(weights, torch.maximum(prev_heading, next_heading))
    if points.shape[1] > 3:
        prev_speed = torch.abs(points[:, 3].float() - points[prev_idx, 3].float())
        next_speed = torch.abs(points[next_idx, 3].float() - points[:, 3].float())
        speed_change = torch.maximum(prev_speed, next_speed)
        weights = torch.maximum(weights, _normalize_0_1(speed_change))
    weights = weights.clamp(0.0, 1.0)
    sparse = torch.zeros_like(weights)
    for start, end in boundaries:
        local = weights[int(start) : int(end)]
        if int(local.numel()) <= 0:
            continue
        max_value = local.max()
        min_value = local.min()
        if float((max_value - min_value).item()) <= 1e-6:
            continue
        threshold = torch.quantile(local, 0.70)
        sparse[int(start) : int(end)] = ((local - threshold) / (max_value - threshold).clamp(min=1e-6)).clamp(0.0, 1.0)
    return sparse


def _boundary_indices_for_query(
    range_mask: torch.Tensor,
    boundaries: list[tuple[int, int]],
) -> torch.Tensor:
    """Return in-box entry/exit point indices for one query."""
    parts: list[torch.Tensor] = []
    mask_cpu = range_mask.detach().cpu()
    for start, end in boundaries:
        if end <= start:
            continue
        local = mask_cpu[start:end]
        if not bool(local.any().item()):
            continue
        enters = torch.zeros_like(local)
        exits = torch.zeros_like(local)
        enters[1:] = local[1:] & ~local[:-1]
        enters[0] = local[0]
        exits[:-1] = local[:-1] & ~local[1:]
        exits[-1] = local[-1]
        offsets = torch.where(enters | exits)[0]
        if int(offsets.numel()) > 0:
            parts.append(offsets.to(dtype=torch.long) + int(start))
    if not parts:
        return torch.empty((0,), dtype=torch.long, device=range_mask.device)
    return torch.cat(parts).to(device=range_mask.device, dtype=torch.long).unique(sorted=True)


def _segment_budget_targets(
    point_value: torch.Tensor,
    boundaries: list[tuple[int, int]],
    segment_size: int,
) -> torch.Tensor:
    """Assign each point its segment's normalized query-local value mass."""
    out = torch.zeros_like(point_value.float())
    segment_masses: list[torch.Tensor] = []
    segment_slices: list[tuple[int, int]] = []
    size = max(1, int(segment_size))
    for start, end in boundaries:
        for seg_start in range(int(start), int(end), size):
            seg_end = min(int(end), seg_start + size)
            if seg_end <= seg_start:
                continue
            mass = point_value[seg_start:seg_end].float().clamp(min=0.0).sum()
            segment_masses.append(mass)
            segment_slices.append((seg_start, seg_end))
    if not segment_masses:
        return out
    masses = torch.stack(segment_masses)
    max_mass = masses.max().clamp(min=1e-6)
    normalized = (masses / max_mass).clamp(0.0, 1.0)
    for value, (seg_start, seg_end) in zip(normalized, segment_slices, strict=True):
        out[seg_start:seg_end] = value
    return out


def _lat_lon_distance_km(points: torch.Tensor, left_idx: torch.Tensor, right_idx: torch.Tensor) -> torch.Tensor:
    """Return approximate lat/lon distance in km for local index pairs."""
    left = points[left_idx.long()]
    right = points[right_idx.long()]
    lat1 = left[:, 1].float()
    lon1 = left[:, 2].float()
    lat2 = right[:, 1].float()
    lon2 = right[:, 2].float()
    lat_mid = torch.deg2rad((lat1 + lat2) * 0.5)
    dy = (lat2 - lat1) * 111.32
    dx = (lon2 - lon1) * 111.32 * torch.clamp(torch.cos(lat_mid).abs(), min=0.10)
    return torch.sqrt(dx * dx + dy * dy)


def _path_length_support_targets(
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    segment_size: int,
    highpass_quantile: float = 0.50,
) -> torch.Tensor:
    """Assign each point its segment's normalized query-free path-length support."""
    n_points = int(points.shape[0])
    raw = torch.zeros((n_points,), dtype=torch.float32, device=points.device)
    if n_points <= 0 or int(points.shape[1]) < 3:
        return raw

    for start, end in boundaries:
        start_i = int(start)
        end_i = int(end)
        count = int(end_i - start_i)
        if count < 3:
            continue
        local = points[start_i:end_i]
        mid_idx = torch.arange(1, count - 1, dtype=torch.long, device=points.device)
        prev_idx = mid_idx - 1
        next_idx = mid_idx + 1
        via_mid = _lat_lon_distance_km(local, prev_idx, mid_idx) + _lat_lon_distance_km(local, mid_idx, next_idx)
        shortcut = _lat_lon_distance_km(local, prev_idx, next_idx)
        raw[start_i + mid_idx] = torch.clamp(via_mid - shortcut, min=0.0)

    out = torch.zeros_like(raw)
    size = max(1, int(segment_size))
    quantile = max(0.0, min(1.0, float(highpass_quantile)))
    for start, end in boundaries:
        segment_masses: list[torch.Tensor] = []
        segment_slices: list[tuple[int, int]] = []
        for seg_start in range(int(start), int(end), size):
            seg_end = min(int(end), seg_start + size)
            if seg_end <= seg_start:
                continue
            segment_masses.append(raw[seg_start:seg_end].float().clamp(min=0.0).sum())
            segment_slices.append((seg_start, seg_end))
        if not segment_masses:
            continue
        masses = torch.stack(segment_masses)
        if float(masses.max().item()) <= 1e-12:
            continue
        if int(masses.numel()) == 1:
            normalized = masses / masses.max().clamp(min=1e-6)
        else:
            threshold = torch.quantile(masses, quantile)
            span = (masses.max() - threshold).clamp(min=1e-6)
            normalized = ((masses - threshold) / span).clamp(0.0, 1.0)
        for value, (seg_start, seg_end) in zip(normalized, segment_slices, strict=True):
            out[seg_start:seg_end] = value
    return out


def _query_replacement_support(
    query_value: torch.Tensor,
    boundaries: list[tuple[int, int]],
    keep_fraction: float = 0.50,
) -> torch.Tensor:
    """Return sparse query-local representative support for replacement-value labels."""
    support = torch.zeros_like(query_value, dtype=torch.bool)
    ratio = min(1.0, max(0.0, float(keep_fraction)))
    positive = query_value > 0.0
    for start, end in boundaries:
        cursor = int(start)
        while cursor < int(end):
            while cursor < int(end) and not bool(positive[cursor].item()):
                cursor += 1
            run_start = cursor
            while cursor < int(end) and bool(positive[cursor].item()):
                cursor += 1
            run_end = cursor
            run_len = int(run_end - run_start)
            if run_len <= 0:
                continue
            keep_count = min(run_len, max(1, int(math.ceil(ratio * run_len))))
            local_values = query_value[run_start:run_end]
            local_idx = torch.topk(local_values, k=keep_count, largest=True).indices
            support[run_start + local_idx] = True
    return support


def _rank_correlation(x: torch.Tensor, y: torch.Tensor, valid: torch.Tensor) -> float | None:
    """Return a lightweight Spearman-style rank correlation on valid positions."""
    valid = valid.to(dtype=torch.bool)
    if int(valid.sum().item()) < 2:
        return None
    xv = x[valid].float()
    yv = y[valid].float()
    if float(xv.std(unbiased=False).item()) <= 1e-12 or float(yv.std(unbiased=False).item()) <= 1e-12:
        return None
    x_order = torch.argsort(xv, stable=True)
    y_order = torch.argsort(yv, stable=True)
    x_rank = torch.empty_like(xv)
    y_rank = torch.empty_like(yv)
    rank_values = torch.arange(int(xv.numel()), dtype=torch.float32, device=xv.device)
    x_rank[x_order] = rank_values
    y_rank[y_order] = rank_values
    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    denom = x_centered.norm() * y_centered.norm()
    if float(denom.item()) <= 1e-12:
        return None
    return float((x_centered * y_centered).sum().item() / float(denom.item()))


def _topk_overlap_and_mass_recall(
    *,
    ranker: torch.Tensor,
    reference: torch.Tensor,
    valid: torch.Tensor,
    ratio: float,
) -> dict[str, float]:
    """Return overlap and reference-mass recall when ranking by another target."""
    valid = valid.to(dtype=torch.bool)
    valid_count = int(valid.sum().item())
    if valid_count <= 0:
        return {"overlap": 0.0, "reference_mass_recall": 0.0}
    keep = min(valid_count, max(1, int(math.ceil(float(ratio) * valid_count))))
    ranker_values = ranker[valid].float()
    reference_values = reference[valid].float().clamp(min=0.0)
    reference_mass = float(reference_values.sum().item())
    if reference_mass <= 1e-12:
        return {"overlap": 0.0, "reference_mass_recall": 0.0}
    selected_by_ranker = torch.topk(ranker_values, k=keep, largest=True).indices
    selected_by_reference = torch.topk(reference_values, k=keep, largest=True).indices
    selected_mask = torch.zeros((valid_count,), dtype=torch.bool, device=ranker.device)
    reference_mask = torch.zeros((valid_count,), dtype=torch.bool, device=ranker.device)
    selected_mask[selected_by_ranker] = True
    reference_mask[selected_by_reference] = True
    ideal_mass = float(reference_values[selected_by_reference].sum().item())
    selected_mass = float(reference_values[selected_by_ranker].sum().item())
    return {
        "overlap": float((selected_mask & reference_mask).sum().item() / keep),
        "reference_mass_recall": float(selected_mass / max(ideal_mass, 1e-12)),
    }


def _conditional_behavior_target_alignment(
    *,
    behavior: torch.Tensor,
    valid: torch.Tensor,
    references: dict[str, torch.Tensor],
    ratio: float = 0.05,
) -> dict[str, Any]:
    """Summarize whether behavior labels rank points like query-useful reference labels."""
    valid = valid.to(dtype=torch.bool)
    out: dict[str, Any] = {
        "valid_point_count": int(valid.sum().item()),
        "topk_ratio": float(ratio),
    }
    for reference_name, reference in references.items():
        safe_name = str(reference_name)
        out[f"spearman_with_{safe_name}"] = _rank_correlation(behavior, reference, valid)
        topk = _topk_overlap_and_mass_recall(
            ranker=behavior,
            reference=reference,
            valid=valid,
            ratio=ratio,
        )
        out[f"topk_overlap_with_{safe_name}"] = topk["overlap"]
        out[f"topk_{safe_name}_mass_recall_ranked_by_behavior"] = topk["reference_mass_recall"]
    return out


def _ship_query_pair_coverage_at_topk(
    *,
    ranker: torch.Tensor,
    valid: torch.Tensor,
    query_hit_masks: list[torch.Tensor],
    boundaries: list[tuple[int, int]],
    ratio: float,
) -> dict[str, Any]:
    """Return trajectory-query coverage when retaining top-ranked hit points."""
    valid = valid.to(dtype=torch.bool)
    valid_count = int(valid.sum().item())
    out: dict[str, Any] = {
        "ship_query_topk_ratio": float(ratio),
        "ship_query_topk_selected_point_count": 0,
        "ship_query_pair_count": 0,
        "ship_query_pair_covered_count": 0,
        "ship_query_pair_coverage_at_topk": 0.0,
    }
    if valid_count <= 0 or not query_hit_masks:
        return out

    keep = min(valid_count, max(1, int(math.ceil(float(ratio) * valid_count))))
    valid_indices = torch.where(valid)[0]
    top_local = torch.topk(ranker[valid].float(), k=keep, largest=True).indices
    selected = torch.zeros_like(valid, dtype=torch.bool)
    selected[valid_indices[top_local]] = True

    pair_count = 0
    covered_count = 0
    for query_mask in query_hit_masks:
        query_hit = query_mask.to(device=valid.device, dtype=torch.bool)
        for start, end in boundaries:
            local_hit = query_hit[int(start) : int(end)]
            if not bool(local_hit.any().item()):
                continue
            pair_count += 1
            if bool((selected[int(start) : int(end)] & local_hit).any().item()):
                covered_count += 1
    out.update(
        {
            "ship_query_topk_selected_point_count": keep,
            "ship_query_pair_count": pair_count,
            "ship_query_pair_covered_count": covered_count,
            "ship_query_pair_coverage_at_topk": float(covered_count / max(1, pair_count)),
        }
    )
    return out


def _conditional_behavior_candidate_diagnostics(
    *,
    current_behavior: torch.Tensor,
    valid: torch.Tensor,
    replacement: torch.Tensor,
    segment_budget: torch.Tensor,
    references: dict[str, torch.Tensor],
    query_hit_masks: list[torch.Tensor] | None = None,
    boundaries: list[tuple[int, int]] | None = None,
    ratio: float = 0.05,
) -> dict[str, Any]:
    """Return diagnostics for behavior-target candidates without changing labels."""
    valid = valid.to(dtype=torch.bool)
    replacement_signal = _normalize_0_1(replacement, valid)
    segment_signal = _normalize_0_1(segment_budget, valid)
    candidates = {
        "current_local_behavior": current_behavior,
        "replacement_gated_local_behavior": current_behavior * (0.25 + 0.75 * replacement_signal),
        "segment_gated_local_behavior": current_behavior * (0.25 + 0.75 * segment_signal),
        "replacement_support_only_local_behavior": torch.where(
            replacement > 0.0,
            current_behavior,
            torch.zeros_like(current_behavior),
        ),
        "replacement_segment_gated_local_behavior": current_behavior
        * (0.20 + 0.50 * replacement_signal + 0.30 * segment_signal),
    }
    out: dict[str, Any] = {}
    for name, candidate in candidates.items():
        valid_candidate = candidate[valid].float().clamp(min=0.0)
        row = _conditional_behavior_target_alignment(
            behavior=candidate,
            valid=valid,
            references=references,
        )
        row["support_fraction_by_threshold"] = support_fraction_by_threshold(candidate, valid)
        row["target_mass"] = float(valid_candidate.sum().item())
        row["target_mean"] = float(valid_candidate.mean().item()) if int(valid_candidate.numel()) > 0 else 0.0
        row["target_std"] = (
            float(valid_candidate.std(unbiased=False).item()) if int(valid_candidate.numel()) > 0 else 0.0
        )
        if query_hit_masks is not None and boundaries is not None:
            row.update(
                _ship_query_pair_coverage_at_topk(
                    ranker=candidate,
                    valid=valid,
                    query_hit_masks=query_hit_masks,
                    boundaries=boundaries,
                    ratio=ratio,
                )
            )
        out[str(name)] = row
    return out


def build_query_useful_v1_targets(
    *,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, Any]],
    segment_size: int = 32,
) -> QueryUsefulTargetBundle:
    """Build factorized QueryUsefulV1 labels from training range workloads only."""
    n_points = int(points.shape[0])
    device = points.device
    labels = torch.zeros((n_points, NUM_QUERY_TYPES), dtype=torch.float32, device=device)
    labelled_mask = torch.zeros((n_points, NUM_QUERY_TYPES), dtype=torch.bool, device=device)
    head_targets = torch.zeros((n_points, len(QUERY_USEFUL_V1_HEAD_NAMES)), dtype=torch.float32, device=device)
    head_mask = torch.zeros_like(head_targets, dtype=torch.bool)
    range_queries = [query for query in typed_queries if str(query.get("type", "")).lower() == "range"]
    if n_points <= 0 or not range_queries:
        diagnostics = factorized_target_diagnostics(head_targets, head_mask, QUERY_USEFUL_V1_HEAD_NAMES, boundaries)
        return QueryUsefulTargetBundle(labels, labelled_mask, head_targets, head_mask, diagnostics)

    behavior_base = _trajectory_change_weights(points, boundaries)
    query_hit_count = torch.zeros((n_points,), dtype=torch.float32, device=device)
    behavior_mass = torch.zeros_like(query_hit_count)
    boundary_mass = torch.zeros_like(query_hit_count)
    replacement_mass = torch.zeros_like(query_hit_count)
    ship_query_evidence_mass = torch.zeros_like(query_hit_count)
    query_hit_masks: list[torch.Tensor] = []

    points_cpu = points.detach().cpu()
    for query in range_queries:
        mask = points_in_range_box(points, query["params"]).to(device=device, dtype=torch.bool)
        if not bool(mask.any().item()):
            continue
        query_hit_masks.append(mask)
        query_hit_count[mask] += 1.0
        behavior_mass[mask] += behavior_base[mask]
        for start, end in boundaries:
            local_mask = mask[int(start) : int(end)]
            if not bool(local_mask.any().item()):
                continue
            local_hit_count = int(local_mask.sum().item())
            if local_hit_count <= 0:
                continue
            local_indices = torch.where(local_mask)[0] + int(start)
            ship_query_evidence_mass[local_indices] += float(1.0 / local_hit_count)
        boundary_idx = _boundary_indices_for_query(mask, boundaries)
        if int(boundary_idx.numel()) > 0:
            boundary_mass[boundary_idx] += 1.0
        crossing_idx = segment_box_bracket_indices(points_cpu, boundaries, query["params"]).to(device=device)
        if int(crossing_idx.numel()) > 0:
            boundary_mass[crossing_idx] += 1.0
        query_value = torch.zeros_like(query_hit_count)
        query_value[mask] = 0.50 + 0.35 * behavior_base[mask]
        if int(boundary_idx.numel()) > 0:
            query_value[boundary_idx] += 0.40
        if int(crossing_idx.numel()) > 0:
            query_value[crossing_idx] += 0.30
        replacement_support = _query_replacement_support(query_value, boundaries)
        replacement_mass[replacement_support] += query_value[replacement_support].clamp(min=0.0)

    query_count = float(max(1, len(range_queries)))
    q_hit = (query_hit_count / query_count).clamp(0.0, 1.0)
    behavior = torch.zeros_like(q_hit)
    hit_positive = query_hit_count > 0
    behavior[hit_positive] = (
        behavior_mass[hit_positive] / query_hit_count[hit_positive].clamp(min=1.0)
    ).clamp(0.0, 1.0)
    boundary = (boundary_mass / query_count).clamp(0.0, 1.0).square()
    replacement = torch.zeros_like(q_hit)
    replacement[hit_positive] = (
        replacement_mass[hit_positive] / query_hit_count[hit_positive].clamp(min=1.0)
    ).clamp(0.0, 1.0)
    ship_query_evidence = (ship_query_evidence_mass / query_count).clamp(0.0, 1.0)
    final_score = (q_hit * (0.5 + behavior) * (0.75 + 0.25 * replacement) + 0.25 * boundary).clamp(0.0, 1.0)
    segment_budget = _segment_budget_targets(
        final_score,
        boundaries,
        segment_size,
    )
    path_length_support = _path_length_support_targets(
        points,
        boundaries,
        segment_size,
    )

    head_targets[:, 0] = q_hit
    head_targets[:, 1] = behavior
    head_targets[:, 2] = boundary
    head_targets[:, 3] = replacement
    head_targets[:, 4] = segment_budget
    head_targets[:, 5] = path_length_support
    head_mask[:] = True
    head_mask[:, 1] = hit_positive

    labels[:, QUERY_TYPE_ID_RANGE] = final_score
    labelled_mask[:, QUERY_TYPE_ID_RANGE] = True
    diagnostics = factorized_target_diagnostics(
        head_targets,
        head_mask,
        QUERY_USEFUL_V1_HEAD_NAMES,
        boundaries,
    )
    diagnostics.update(
        {
            "target_family": "QueryUsefulV1Factorized",
            "range_query_count": int(len(range_queries)),
            "segment_size_points": int(segment_size),
            "segment_budget_target_training": "point_repeated_plus_segment_level_listwise_loss",
            "segment_budget_target_base_source": "query_useful_v1_final_score",
            "segment_budget_segment_level_loss_enabled": True,
            "path_length_support_target_training": "query_free_segment_path_length_removal_loss_highpass",
            "path_length_support_target_base_source": "per_point_path_length_removal_loss_segment_highpass_mass",
            "path_length_support_target_query_free": True,
            "path_length_support_target_highpass_quantile": 0.50,
            "behavior_change_highpass_quantile": 0.70,
            "conditional_behavior_utility_training": "masked_to_query_hit_points",
            "replacement_representative_value_normalization": "conditional_on_query_hit",
            "replacement_value_is_true_counterfactual_marginal_gain": False,
            "final_label_formula": "query_hit_times_behavior_with_conditional_replacement_modulation_plus_boundary",
            "final_boundary_bonus_uses_squared_event_probability": True,
            "final_label_positive_fraction": float((final_score > 0.0).float().mean().item()),
            "final_label_support_fraction_by_threshold": support_fraction_by_threshold(final_score),
            "final_label_mass": float(final_score.sum().item()),
            "conditional_behavior_target_alignment": _conditional_behavior_target_alignment(
                behavior=behavior,
                valid=hit_positive,
                references={
                    "final_score": final_score,
                    "query_hit_probability": q_hit,
                    "ship_query_evidence": ship_query_evidence,
                    "replacement_representative_value": replacement,
                    "segment_budget_target": segment_budget,
                    "path_length_support_target": path_length_support,
                },
            ),
            "conditional_behavior_candidate_alignment": _conditional_behavior_candidate_diagnostics(
                current_behavior=behavior,
                valid=hit_positive,
                replacement=replacement,
                segment_budget=segment_budget,
                references={
                    "final_score": final_score,
                    "query_hit_probability": q_hit,
                    "ship_query_evidence": ship_query_evidence,
                    "replacement_representative_value": replacement,
                    "segment_budget_target": segment_budget,
                    "path_length_support_target": path_length_support,
                },
                query_hit_masks=query_hit_masks,
                boundaries=boundaries,
            ),
        }
    )
    return QueryUsefulTargetBundle(labels, labelled_mask, head_targets, head_mask, diagnostics)
