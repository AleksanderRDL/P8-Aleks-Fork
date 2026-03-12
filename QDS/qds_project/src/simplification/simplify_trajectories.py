"""
simplify_trajectories.py

Compress AIS trajectory data by removing low-importance points as
predicted by a trained TrajectoryQDSModel.

Points whose predicted importance score falls below a threshold are
discarded; the retained set preserves query accuracy while reducing
dataset size.
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.models.trajectory_qds_model import TrajectoryQDSModel
from src.models.trajectory_qds_model import normalize_points_and_queries
from src.training.importance_labels import compute_importance


def simplify_trajectories(
    points: Tensor,
    model: TrajectoryQDSModel,
    queries: Tensor,
    threshold: float = 0.5,
    query_scores: Tensor | None = None,
    model_max_points: int | None = 300_000,
    importance_chunk_size: int = 200_000,
) -> tuple[Tensor, Tensor, Tensor]:
    """Remove low-importance trajectory points below a threshold.

    Uses the trained model to predict per-point importance scores, then
    retains only the points whose score meets or exceeds the threshold.

    If model scores appear degenerate (flat or poorly aligned with the
    query-driven label structure), the function falls back to exact
    query-driven scores from ``compute_importance`` to avoid retaining
    clearly irrelevant points.

    At least one point is always retained to prevent empty outputs.

    Args:
        points:    Tensor of shape [N, 5] with columns
                   [time, lat, lon, speed, heading].
        model:     Trained TrajectoryQDSModel instance.
        queries:   Tensor of shape [M, 6] — the query workload used to
                   guide which points matter.
        threshold: Importance score threshold in [0, 1].  Points with
                   score >= threshold are retained.
        query_scores: Optional precomputed query-driven importance scores.
        model_max_points: Optional upper bound for running full-set model
                  inference. Above this, query scores are used directly.
        importance_chunk_size: Chunk size used if query scores must be computed.

    Returns:
        A tuple of:
        - simplified_points: Tensor of shape [K, 5] where K ≤ N, containing
          the simplified trajectory point cloud.
        - retained_mask: Boolean tensor of shape [N] where True indicates a
          retained point and False indicates a removed point.
        - importance_scores: Tensor of shape [N] with the scores used for
          simplification (model or query-driven fallback).
    """
    run_model = model_max_points is None or points.shape[0] <= int(model_max_points)

    model_scores: Tensor | None = None
    if run_model:
        norm_points, norm_queries = normalize_points_and_queries(points, queries)
        model.eval()
        with torch.no_grad():
            model_scores = model(norm_points, norm_queries)  # [N]

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

    mask = scores >= threshold

    # Always retain at least one point
    if mask.sum() == 0:
        best_idx = scores.argmax()
        mask[best_idx] = True

    return points[mask], mask, scores
