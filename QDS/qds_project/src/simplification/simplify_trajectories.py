"""
simplify_trajectories.py

Compress AIS trajectory data by removing low-importance points as
predicted by a trained TrajectoryQDSModel.

Two simplification modes are supported:

1. **Per-trajectory compression** (recommended, default when
   ``compression_ratio`` is given):

   Each trajectory is simplified independently.  A point budget is
   derived per trajectory::

       points_to_keep = max(min_points_per_trajectory,
                            int(compression_ratio * trajectory_length))

   The top-``points_to_keep`` points are selected by descending
   importance score.  Trajectory endpoints are always retained by
   boosting their scores before the top-k selection.

2. **Global threshold mode** (legacy, used when ``compression_ratio``
   is ``None``):

   Points whose predicted importance score falls below ``threshold``
   are discarded.  Trajectory-aware constraints (endpoint retention and
   minimum-point floor) are applied afterwards.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor

from src.models.trajectory_qds_model import TrajectoryQDSModel
from src.models.trajectory_qds_model import normalize_points_and_queries
from src.models.turn_aware_qds_model import TurnAwareQDSModel
from src.training.importance_labels import compute_importance


def _infer_trajectory_boundaries(points: Tensor) -> List[Tuple[int, int]]:
    """Infer trajectory start/end index pairs from the ``is_start`` feature.

    Each point where column 5 (``is_start``) equals 1.0 begins a new
    trajectory.  The trajectory ends just before the next start, or at the
    last point in the tensor.

    Args:
        points: Tensor of shape [N, 7] with ``is_start`` in column 5.

    Returns:
        List of (start_index, end_index) pairs where end_index is exclusive.
        Returns ``[(0, N)]`` if no ``is_start`` flags are found.
    """
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
    """Apply endpoint-retention and minimum-points constraints in-place.

    For each trajectory defined by *boundaries*:
    1. Force the first and last points into *mask*.
    2. If the retained count is still below *min_points_per_trajectory*,
       promote additional points from highest to lowest importance score
       until the minimum is satisfied (or the trajectory is exhausted).

    Args:
        mask:       Boolean tensor [N] modified in-place.
        scores:     Importance scores [N] used for top-k promotion.
        boundaries: List of (start, end) index pairs (exclusive end).
        min_points_per_trajectory: Minimum retained points per trajectory.
    """
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
    """Apply per-trajectory top-k compression in-place.

    For each trajectory the number of points to retain is::

        points_to_keep = max(min_points_per_trajectory,
                             int(compression_ratio * trajectory_length))

    The top-``points_to_keep`` points (by importance score) are selected.
    Trajectory endpoints (first and last point) are guaranteed to be
    included by temporarily boosting their scores to ``+inf`` before the
    top-k selection; the original *scores* tensor is **not** modified.

    Args:
        mask:       Boolean tensor [N] modified in-place.
        scores:     Importance scores [N].  Not modified.
        boundaries: List of (start, end) index pairs (exclusive end).
        compression_ratio: Fraction of each trajectory's points to keep
                    (before applying the minimum-points floor).
        min_points_per_trajectory: Minimum retained points per trajectory.
    """
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
        # trajectory structural integrity without modifying the caller's tensor.
        traj_scores = scores[traj_start:traj_end].clone()
        traj_scores[0] = float("inf")                # trajectory start — must be kept
        traj_scores[traj_len - 1] = float("inf")     # trajectory end   — must be kept

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
    """Simplify AIS trajectory data using predicted per-point importance.

    Two modes are available, selected by the *compression_ratio* argument:

    **Per-trajectory compression** (``compression_ratio`` is not ``None``):
        Each trajectory is compressed independently.  A per-trajectory
        point budget is computed as::

            points_to_keep = max(min_points_per_trajectory,
                                 int(compression_ratio * traj_len))

        The top-``points_to_keep`` points are retained, with trajectory
        endpoints guaranteed by boosting their scores before selection.
        This distributes compression evenly across all trajectories so
        that no trajectory is fully removed.

    **Global threshold mode** (``compression_ratio`` is ``None``):
        Points whose score falls below *threshold* are discarded.
        Trajectory-aware endpoint-retention and minimum-point-floor
        constraints are still applied afterwards.

    In both modes, if model scores appear degenerate the function falls
    back to query-driven importance scores.

    When *turn_bias_weight* is positive and *points* contains a
    ``turn_score`` column (column 7), a small additive bias is applied::

        final_importance = predicted_importance
                         + turn_bias_weight * turn_score

    This nudges the ranker toward retaining points at trajectory bends
    while keeping endpoints and query-relevant points dominant.

    Args:
        points:    Tensor of shape [N, 7] or [N, 8] with columns
                   [time, lat, lon, speed, heading, is_start, is_end]
                   and optionally [turn_score].
        model:     Trained TrajectoryQDSModel or TurnAwareQDSModel instance.
        queries:   Tensor of shape [M, 6] — the query workload.
        threshold: Score threshold used in global threshold mode.
        query_scores: Optional precomputed query-driven importance scores.
        model_max_points: Optional upper bound for model inference.  Above
                   this, query scores are used directly.
        importance_chunk_size: Chunk size for query-score computation.
        trajectory_boundaries: Optional list of (start, end) index pairs
                   (exclusive end).  Inferred from ``is_start`` (column 5)
                   when ``None``.
        min_points_per_trajectory: Minimum retained points per trajectory.
        compression_ratio: Per-trajectory compression parameter in (0, 1].
                   When set, enables per-trajectory top-k mode and
                   *threshold* is ignored.  Defaults to 0.2.
        turn_bias_weight: Additive weight for turn-score bias.  When > 0
                   and points has 8+ columns, ``turn_score`` (column 7) is
                   added to the importance scores with this weight.
                   Recommended default is 0.1 for the turn-aware model.
                   Defaults to 0.0 (no bias).

    Returns:
        A tuple of:
        - simplified_points: Tensor of shape [K, F] where K ≤ N and F is
          the number of feature columns in *points*.
        - retained_mask: Boolean tensor of shape [N].
        - importance_scores: Tensor of shape [N] (model or fallback scores).
    """
    run_model = model_max_points is None or points.shape[0] <= int(model_max_points)

    model_scores: Tensor | None = None
    if run_model:
        # TurnAwareQDSModel expects 8-feature points; baseline model expects 7.
        if isinstance(model, TurnAwareQDSModel):
            model_points = points if points.shape[1] >= 8 else torch.cat(
                [points, torch.zeros(points.shape[0], 1, device=points.device)], dim=1
            )
        else:
            model_points = points[:, :7]
        norm_points, norm_queries = normalize_points_and_queries(model_points, queries)
        # Move tensors to the model's device once before inference.
        _first_param = next(model.parameters(), None)
        model_device = _first_param.device if _first_param is not None else torch.device("cpu")
        norm_points = norm_points.to(model_device)
        norm_queries = norm_queries.to(model_device)
        model.eval()
        with torch.no_grad():
            model_scores = model(norm_points, norm_queries).to(points.device)  # [N]

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

    # Apply optional turn-score bias when turn_bias_weight > 0 and turn_score
    # column is available (column 7 in an 8-feature point tensor).
    if turn_bias_weight > 0.0 and points.shape[1] >= 8:
        turn_scores = points[:, 7]
        if turn_scores.device != scores.device:
            turn_scores = turn_scores.to(scores.device)
        scores = scores + turn_bias_weight * turn_scores

    # Resolve trajectory boundaries (used by both modes).
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
