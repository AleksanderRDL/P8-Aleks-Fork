"""Trajectory simplification via predicted importance scores. See src/simplification/README.md."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor

from src.models.trajectory_qds_model import TrajectoryQDSModel
from src.models.trajectory_qds_model import normalize_points_and_queries
from src.models.turn_aware_qds_model import TurnAwareQDSModel
from src.training.importance_labels import compute_importance


def _infer_trajectory_boundaries(points: Tensor) -> List[Tuple[int, int]]:
    """Infer trajectory (start, end) index pairs from the ``is_start`` flag (column 5)."""
    if points.shape[1] < 6:
        return [(0, points.shape[0])]

    start_flags = (points[:, 5] > 0.5).nonzero(as_tuple=True)[0].tolist()
    if not start_flags:
        return [(0, points.shape[0])]

    n = points.shape[0]
    boundaries: List[Tuple[int, int]] = []
    for i, start in enumerate(start_flags):
        end = start_flags[i + 1] if i + 1 < len(start_flags) else n
        boundaries.append((start, end))
    return boundaries


def _enforce_trajectory_constraints(
    mask: Tensor,
    scores: Tensor,
    boundaries: List[Tuple[int, int]],
    min_points_per_trajectory: int,
) -> None:
    """Apply endpoint-retention and minimum-points constraints in-place."""
    for traj_start, traj_end in boundaries:
        if traj_start >= traj_end:
            continue

        # Always retain first and last points of the trajectory.
        mask[traj_start] = True
        mask[traj_end - 1] = True

        # Enforce minimum point count.
        traj_len = traj_end - traj_start
        needed = min(int(min_points_per_trajectory), traj_len)
        retained_count = int(mask[traj_start:traj_end].sum().item())

        if retained_count < needed:
            additional = needed - retained_count
            traj_scores = scores[traj_start:traj_end].clone()
            # Exclude already-retained points from the candidate set.
            traj_scores[mask[traj_start:traj_end]] = -float("inf")
            k = min(additional, int((~mask[traj_start:traj_end]).sum().item()))
            if k > 0:
                top_local = torch.topk(traj_scores, k=k).indices
                for li in top_local:
                    mask[traj_start + li.item()] = True


def _simplify_per_trajectory(
    mask: Tensor,
    scores: Tensor,
    boundaries: List[Tuple[int, int]],
    compression_ratio: float,
    min_points_per_trajectory: int,
) -> None:
    """Apply per-trajectory top-k compression in-place."""
    for traj_start, traj_end in boundaries:
        if traj_start >= traj_end:
            continue

        traj_len = traj_end - traj_start
        points_to_keep = max(
            int(min_points_per_trajectory),
            int(compression_ratio * traj_len),
        )
        points_to_keep = min(points_to_keep, traj_len)

        if points_to_keep >= traj_len:
            # Keep the entire trajectory.
            mask[traj_start:traj_end] = True
            continue

        # Clone trajectory scores and boost endpoints to +inf so they are always
        # selected regardless of their raw importance score.  This guarantees
        traj_scores = scores[traj_start:traj_end].clone()
        traj_scores[0] = float("inf")
        traj_scores[traj_len - 1] = float("inf")

        top_local = torch.topk(traj_scores, k=points_to_keep).indices
        for li in top_local:
            mask[traj_start + int(li.item())] = True


def simplify_trajectories(
    points: Tensor,
    model: TrajectoryQDSModel | TurnAwareQDSModel,
    queries: Tensor,
    threshold: float = 0.5,
    query_scores: Tensor | None = None,
    model_max_points: int | None = 300_000,
    importance_chunk_size: int = 200_000,
    trajectory_boundaries: Optional[List[Tuple[int, int]]] = None,
    min_points_per_trajectory: int = 3,
    compression_ratio: float | None = 0.2,
    turn_bias_weight: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Simplify AIS trajectory data using predicted per-point importance scores."""
    run_model = model_max_points is None or points.shape[0] <= int(model_max_points)

    model_scores: Tensor | None = None
    if run_model:
        if isinstance(model, TurnAwareQDSModel):
            model_points = points if points.shape[1] >= 8 else torch.cat(
                [points, torch.zeros(points.shape[0], 1, device=points.device)], dim=1
            )
        else:
            model_points = points[:, :7]
        norm_points, norm_queries = normalize_points_and_queries(model_points, queries)
        _first_param = next(model.parameters(), None)
        model_device = _first_param.device if _first_param is not None else torch.device("cpu")
        norm_points = norm_points.to(model_device)
        norm_queries = norm_queries.to(model_device)
        model.eval()
        with torch.no_grad():
            model_scores = model(norm_points, norm_queries).to(points.device)

    if query_scores is None:
        query_scores = compute_importance(points, queries, chunk_size=importance_chunk_size)

    if model_scores is None:
        scores = query_scores
    else:
        if query_scores.device != model_scores.device:
            query_scores = query_scores.to(model_scores.device)

        top_k = max(1, int(points.shape[0] * 0.01))
        top_idx = torch.topk(model_scores, k=top_k).indices

        score_span = (model_scores.max() - model_scores.min()).item()
        top_query_mean = query_scores[top_idx].mean().item()
        global_query_mean = query_scores.mean().item()

        degenerate_scores = score_span < 1e-6
        weak_query_alignment = top_query_mean <= (global_query_mean + 1e-4)

        scores = query_scores if (degenerate_scores or weak_query_alignment) else model_scores

    # Apply optional turn-score bias (column 7, if present).
    if turn_bias_weight > 0.0 and points.shape[1] >= 8:
        turn_scores = points[:, 7]
        if turn_scores.device != scores.device:
            turn_scores = turn_scores.to(scores.device)
        scores = scores + turn_bias_weight * turn_scores

    boundaries = (
        trajectory_boundaries
        if trajectory_boundaries is not None
        else _infer_trajectory_boundaries(points)
    )

    mask = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)

    if compression_ratio is not None:
        # ---- Per-trajectory top-k compression --------------------------------
        _simplify_per_trajectory(mask, scores, boundaries, compression_ratio, min_points_per_trajectory)
    else:
        # ---- Legacy global threshold mode ------------------------------------
        mask = scores >= threshold

        # Always retain at least one point.
        if mask.sum() == 0:
            best_idx = scores.argmax()
            mask[best_idx] = True

        # Apply trajectory-aware constraints: retain endpoints and enforce
        # a minimum number of points per trajectory.
        _enforce_trajectory_constraints(mask, scores, boundaries, min_points_per_trajectory)

    return points[mask], mask, scores
