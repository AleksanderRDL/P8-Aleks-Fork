"""Shared MLQDS scoring helpers used by validation and final evaluation."""

from __future__ import annotations

import torch

from queries.query_types import QUERY_NAME_TO_ID
from simplification.learned_segment_budget import simplify_with_learned_segment_budget_v1
from simplification.simplify_trajectories import simplify_with_temporal_score_hybrid

MLQDS_SCORE_MODES = (
    "rank",
    "rank_tie",
    "sigmoid",
    "raw",
    "zscore_sigmoid",
    "rank_confidence",
    "temperature_sigmoid",
)


def workload_type_head(workload_type: str) -> tuple[str, int]:
    """Return the model head index for one explicit workload type."""
    name = workload_type.lower()
    try:
        return name, QUERY_NAME_TO_ID[name]
    except KeyError as exc:
        raise ValueError(f"Unknown MLQDS workload type: {workload_type}") from exc


def _ordinal_rank_0_1(values: torch.Tensor) -> torch.Tensor:
    """Return 0..1 ordinal ranks for one trajectory."""
    length = int(values.numel())
    if length <= 0:
        return values.new_empty((0,), dtype=torch.float32)
    denom = float(max(1, length - 1))
    return values.argsort().argsort().to(torch.float32) / denom


def _tie_aware_rank_0_1(values: torch.Tensor) -> torch.Tensor:
    """Return 0..1 average ranks, assigning exact ties the same score."""
    length = int(values.numel())
    if length <= 0:
        return values.new_empty((0,), dtype=torch.float32)
    if length == 1:
        return values.new_zeros((1,), dtype=torch.float32)

    sorted_values, order = torch.sort(values)
    starts = torch.ones((length,), dtype=torch.bool, device=values.device)
    starts[1:] = sorted_values[1:] != sorted_values[:-1]
    start_idx = torch.where(starts)[0]
    end_idx = torch.cat([start_idx[1:], torch.tensor([length], dtype=torch.long, device=values.device)])

    ranks = values.new_empty((length,), dtype=torch.float32)
    denom = float(length - 1)
    for start, end in zip(start_idx.tolist(), end_idx.tolist()):
        average_rank = 0.5 * float(start + end - 1)
        ranks[order[start:end]] = average_rank / denom
    return ranks


def _trajectory_zscore(values: torch.Tensor) -> torch.Tensor:
    """Return per-trajectory z-scores, falling back to zeros for flat logits."""
    if values.numel() <= 1:
        return values.new_zeros(values.shape, dtype=torch.float32)
    centered = values.float() - values.float().mean()
    std = values.float().std(unbiased=False)
    if float(std.item()) <= 1e-6:
        return values.new_zeros(values.shape, dtype=torch.float32)
    return centered / std


def pure_workload_scores(
    predictions: torch.Tensor,
    boundaries: list[tuple[int, int]],
    workload_type: str,
    score_mode: str = "rank",
    score_temperature: float = 1.0,
    rank_confidence_weight: float = 0.15,
) -> torch.Tensor:
    """Convert pure workload logits into final MLQDS simplification scores."""
    if predictions.ndim == 1:
        head = predictions.float()
    elif predictions.ndim == 2 and predictions.shape[1] == 1:
        head = predictions[:, 0].float()
    else:
        raise ValueError("predictions must be pure single-workload scores with shape [n_points] or [n_points, 1].")

    mode = score_mode.lower()
    if mode not in MLQDS_SCORE_MODES:
        raise ValueError(f"score_mode must be one of {MLQDS_SCORE_MODES}; got {score_mode}.")

    workload_type_head(workload_type)
    if mode == "raw":
        return head.to(predictions.dtype)
    if mode == "sigmoid":
        return torch.sigmoid(head).to(predictions.dtype)
    if mode == "temperature_sigmoid":
        temperature = max(float(score_temperature), 1e-6)
        return torch.sigmoid(head / temperature).to(predictions.dtype)

    score = head.new_zeros((head.shape[0],))
    temperature = max(float(score_temperature), 1e-6)
    confidence_weight = max(0.0, min(1.0, float(rank_confidence_weight)))
    for start, end in boundaries:
        length = int(end - start)
        if length <= 0:
            continue
        local_head = head[start:end]
        if mode == "rank":
            local_score = _ordinal_rank_0_1(local_head)
        elif mode == "rank_tie":
            local_score = _tie_aware_rank_0_1(local_head)
        else:
            zscore_confidence = torch.sigmoid(_trajectory_zscore(local_head) / temperature)
            if mode == "zscore_sigmoid":
                local_score = zscore_confidence
            else:
                rank = _ordinal_rank_0_1(local_head)
                local_score = (1.0 - confidence_weight) * rank + confidence_weight * zscore_confidence
        score[start:end] = local_score.to(score.dtype)
    return score


def mlqds_simplification_scores(
    predictions: torch.Tensor,
    boundaries: list[tuple[int, int]],
    workload_type: str,
    score_mode: str = "rank",
    score_temperature: float = 1.0,
    rank_confidence_weight: float = 0.15,
    range_geometry_scores: torch.Tensor | None = None,
    range_geometry_blend: float = 0.0,
) -> torch.Tensor:
    """Convert model predictions into reusable MLQDS simplification scores."""
    scores = pure_workload_scores(
        predictions,
        boundaries,
        workload_type,
        score_mode=score_mode,
        score_temperature=score_temperature,
        rank_confidence_weight=rank_confidence_weight,
    )
    geometry_blend = max(0.0, min(1.0, float(range_geometry_blend)))
    if geometry_blend > 0.0:
        if range_geometry_scores is None:
            raise ValueError("range_geometry_scores are required when range_geometry_blend > 0.")
        if str(workload_type).lower() != "range":
            raise ValueError("range_geometry_blend is only supported for range workloads.")
        if int(range_geometry_scores.numel()) != int(scores.numel()):
            raise ValueError(
                "range_geometry_scores must match prediction count: "
                f"got {int(range_geometry_scores.numel())}, expected {int(scores.numel())}."
            )
        geometry = pure_workload_scores(
            range_geometry_scores.to(device=scores.device, dtype=scores.dtype),
            boundaries,
            workload_type,
            score_mode="rank",
        )
        scores = (1.0 - geometry_blend) * scores + geometry_blend * geometry
    return scores


def simplify_mlqds_predictions(
    predictions: torch.Tensor,
    boundaries: list[tuple[int, int]],
    workload_type: str,
    compression_ratio: float,
    temporal_fraction: float,
    diversity_bonus: float,
    hybrid_mode: str = "fill",
    score_mode: str = "rank",
    score_temperature: float = 1.0,
    rank_confidence_weight: float = 0.15,
    range_geometry_scores: torch.Tensor | None = None,
    range_geometry_blend: float = 0.0,
    stratified_center_weight: float = 0.0,
    min_learned_swaps: int = 0,
    selector_type: str = "temporal_hybrid",
    segment_scores: torch.Tensor | None = None,
    points: torch.Tensor | None = None,
) -> torch.Tensor:
    """Simplify using canonical MLQDS score conversion and retained-mask logic."""
    scores = mlqds_simplification_scores(
        predictions,
        boundaries,
        workload_type,
        score_mode=score_mode,
        score_temperature=score_temperature,
        rank_confidence_weight=rank_confidence_weight,
        range_geometry_scores=range_geometry_scores,
        range_geometry_blend=range_geometry_blend,
    )
    if str(selector_type).lower() == "learned_segment_budget_v1":
        return simplify_with_learned_segment_budget_v1(
            scores,
            boundaries,
            compression_ratio,
            segment_scores=segment_scores,
            points=points,
        )
    return simplify_with_temporal_score_hybrid(
        scores,
        boundaries,
        compression_ratio,
        temporal_fraction=temporal_fraction,
        diversity_bonus=diversity_bonus,
        hybrid_mode=hybrid_mode,
        stratified_center_weight=stratified_center_weight,
        min_learned_swaps=min_learned_swaps,
    )
