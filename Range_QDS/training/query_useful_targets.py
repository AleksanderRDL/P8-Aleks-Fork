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

    points_cpu = points.detach().cpu()
    for query in range_queries:
        mask = points_in_range_box(points, query["params"]).to(device=device, dtype=torch.bool)
        if not bool(mask.any().item()):
            continue
        query_hit_count[mask] += 1.0
        behavior_mass[mask] += behavior_base[mask]
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
    behavior[hit_positive] = (behavior_mass[hit_positive] / query_hit_count[hit_positive].clamp(min=1.0)).clamp(0.0, 1.0)
    boundary = (boundary_mass / query_count).clamp(0.0, 1.0).square()
    replacement = (replacement_mass / query_count).clamp(0.0, 1.0)
    final_score = (q_hit * replacement * (0.5 + behavior) + 0.25 * boundary).clamp(0.0, 1.0)
    segment_budget = _segment_budget_targets(
        final_score,
        boundaries,
        segment_size,
    )

    head_targets[:, 0] = q_hit
    head_targets[:, 1] = behavior
    head_targets[:, 2] = boundary
    head_targets[:, 3] = replacement
    head_targets[:, 4] = segment_budget
    head_mask[:] = True

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
            "segment_budget_segment_level_loss_enabled": True,
            "behavior_change_highpass_quantile": 0.70,
            "replacement_representative_value_normalization": "expected_per_query",
            "replacement_value_is_true_counterfactual_marginal_gain": False,
            "final_boundary_bonus_uses_squared_event_probability": True,
            "final_label_positive_fraction": float((final_score > 0.0).float().mean().item()),
            "final_label_support_fraction_by_threshold": support_fraction_by_threshold(final_score),
            "final_label_mass": float(final_score.sum().item()),
        }
    )
    return QueryUsefulTargetBundle(labels, labelled_mask, head_targets, head_mask, diagnostics)


def build(*args: object, **kwargs: object) -> QueryUsefulTargetBundle:
    """Compatibility wrapper for the factorized target builder."""
    return build_query_useful_v1_targets(*args, **kwargs)
