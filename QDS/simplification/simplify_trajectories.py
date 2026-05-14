"""Per-trajectory top-k simplification utilities. See simplification/README.md for details."""

from __future__ import annotations

import math

import torch


def deterministic_topk_with_jitter(
    scores: torch.Tensor,
    keep_count: int,
    trajectory_id: int,
) -> torch.Tensor:
    """Select top-k indices with deterministic pseudo-random tie jitter."""
    score_count = scores.numel()
    if keep_count >= score_count:
        return torch.arange(score_count, dtype=torch.long, device=scores.device)

    positions = torch.arange(score_count, device=scores.device, dtype=torch.float32)
    # Deterministic hash-like noise in [-0.5, 0.5].
    noise = torch.frac(torch.sin(positions * 12.9898 + float(trajectory_id) * 78.233) * 43758.5453) - 0.5
    jittered = scores + 1e-6 * noise
    top = torch.topk(jittered, k=keep_count, largest=True).indices
    return torch.sort(top).values


def evenly_spaced_indices(point_count: int, keep_count: int, device: torch.device) -> torch.Tensor:
    """Return deterministic evenly spaced local indices, including endpoints when possible."""
    point_count = int(point_count)
    keep_count = max(0, min(int(keep_count), point_count))
    if keep_count <= 0 or point_count <= 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    if keep_count >= point_count:
        return torch.arange(point_count, dtype=torch.long, device=device)
    local_indices = torch.linspace(0, point_count - 1, steps=keep_count, device=device).round().long().unique()
    if local_indices.numel() < keep_count:
        filler_indices = torch.arange(point_count, dtype=torch.long, device=device)
        missing_indices = filler_indices[~torch.isin(filler_indices, local_indices)][
            : keep_count - local_indices.numel()
        ]
        local_indices = torch.cat([local_indices, missing_indices])
    return torch.sort(local_indices).values


def simplify_with_scores(
    scores: torch.Tensor,
    boundaries: list[tuple[int, int]],
    compression_ratio: float,
) -> torch.Tensor:
    """Build retained mask by per-trajectory score top-k. See simplification/README.md for details."""
    retained = torch.zeros(scores.shape[0], dtype=torch.bool, device=scores.device)
    for trajectory_id, (start, end) in enumerate(boundaries):
        local_scores = scores[start:end]
        point_count = local_scores.numel()
        if point_count <= 0:
            continue
        keep_count = max(2, int(math.ceil(compression_ratio * point_count)))
        keep_count = min(keep_count, point_count)
        local_indices = deterministic_topk_with_jitter(local_scores, keep_count=keep_count, trajectory_id=trajectory_id)
        retained[start:end][local_indices] = True
        retained[start] = True
        retained[end - 1] = True
    return retained


def simplify_with_temporal_score_hybrid(
    scores: torch.Tensor,
    boundaries: list[tuple[int, int]],
    compression_ratio: float,
    temporal_fraction: float = 0.50,
    diversity_bonus: float = 0.0,
    hybrid_mode: str = "fill",
) -> torch.Tensor:
    """Retain a temporal coverage base, then fill remaining slots by learned score.

    Pure top-k scoring tends to over-select neighbouring points with similar
    logits.  This hybrid keeps most of the strong evenly-spaced temporal
    baseline and lets MLQDS spend the remaining budget on query-aware points.
    """
    retained = torch.zeros(scores.shape[0], dtype=torch.bool, device=scores.device)
    base_fraction = min(1.0, max(0.0, float(temporal_fraction)))
    bonus = max(0.0, float(diversity_bonus))
    mode = str(hybrid_mode).lower()
    if mode not in {"fill", "swap"}:
        raise ValueError("hybrid_mode must be 'fill' or 'swap'.")

    for trajectory_id, (start, end) in enumerate(boundaries):
        local_scores = scores[start:end]
        point_count = local_scores.numel()
        if point_count <= 0:
            continue
        total_keep_count = max(2, int(math.ceil(float(compression_ratio) * point_count)))
        total_keep_count = min(total_keep_count, point_count)
        if mode == "swap":
            base_indices = evenly_spaced_indices(point_count, total_keep_count, scores.device)
            retained[start + base_indices] = True
            protected_count = min(total_keep_count, max(2, int(math.ceil(total_keep_count * base_fraction))))
            swap_count = min(total_keep_count - protected_count, point_count - total_keep_count)
            if swap_count <= 0:
                continue
            removable_indices = base_indices[(base_indices != 0) & (base_indices != point_count - 1)]
            swap_count = min(swap_count, int(removable_indices.numel()))
            if swap_count <= 0:
                continue

            remove_positions = deterministic_topk_with_jitter(
                -local_scores[removable_indices],
                keep_count=swap_count,
                trajectory_id=trajectory_id,
            )
            remove_indices = removable_indices[remove_positions]
            candidate_scores = local_scores.clone()
            candidate_scores[base_indices] = -float("inf")
            add_indices = deterministic_topk_with_jitter(
                candidate_scores,
                keep_count=swap_count,
                trajectory_id=trajectory_id,
            )
            retained[start + remove_indices] = False
            retained[start + add_indices] = True
            continue

        base_keep_count = 0
        if base_fraction > 0.0:
            base_keep_count = min(total_keep_count, max(2, int(math.ceil(total_keep_count * base_fraction))))
        base_indices = evenly_spaced_indices(point_count, base_keep_count, scores.device)
        retained[start + base_indices] = True

        remaining_count = total_keep_count - int(base_indices.numel())
        if remaining_count <= 0:
            continue

        candidate_scores = local_scores.clone()
        candidate_scores[base_indices] = -float("inf")
        if bonus > 0.0 and base_indices.numel() > 0 and point_count > 1:
            positions = torch.arange(point_count, dtype=torch.float32, device=scores.device)
            distance_to_base = torch.abs(
                positions.unsqueeze(1) - base_indices.float().unsqueeze(0)
            ).min(dim=1).values
            candidate_scores = candidate_scores + bonus * (distance_to_base / float(max(1, point_count - 1)))
            candidate_scores[base_indices] = -float("inf")

        fill_indices = deterministic_topk_with_jitter(
            candidate_scores,
            keep_count=remaining_count,
            trajectory_id=trajectory_id,
        )
        retained[start + fill_indices] = True

    return retained
