"""Per-trajectory top-k simplification utilities. See src/simplification/README.md for details."""

from __future__ import annotations

import math

import torch


def deterministic_topk_with_jitter(
    scores: torch.Tensor,
    k: int,
    trajectory_id: int,
) -> torch.Tensor:
    """Select top-k indices with deterministic pseudo-random tie jitter. See src/simplification/README.md for details."""
    n = scores.numel()
    if k >= n:
        return torch.arange(n, dtype=torch.long, device=scores.device)

    pos = torch.arange(n, device=scores.device, dtype=torch.float32)
    # Deterministic hash-like noise in [-0.5, 0.5].
    noise = torch.frac(torch.sin(pos * 12.9898 + float(trajectory_id) * 78.233) * 43758.5453) - 0.5
    jittered = scores + 1e-6 * noise
    top = torch.topk(jittered, k=k, largest=True).indices
    return torch.sort(top).values


def evenly_spaced_indices(n: int, k: int, device: torch.device) -> torch.Tensor:
    """Return deterministic evenly spaced local indices, including endpoints when possible."""
    n = int(n)
    k = max(0, min(int(k), n))
    if k <= 0 or n <= 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    if k >= n:
        return torch.arange(n, dtype=torch.long, device=device)
    local_idx = torch.linspace(0, n - 1, steps=k, device=device).round().long().unique()
    if local_idx.numel() < k:
        filler = torch.arange(n, dtype=torch.long, device=device)
        missing = filler[~torch.isin(filler, local_idx)][: k - local_idx.numel()]
        local_idx = torch.cat([local_idx, missing])
    return torch.sort(local_idx).values


def simplify_with_scores(
    scores: torch.Tensor,
    boundaries: list[tuple[int, int]],
    compression_ratio: float,
) -> torch.Tensor:
    """Build retained mask by per-trajectory score top-k. See src/simplification/README.md for details."""
    retained = torch.zeros(scores.shape[0], dtype=torch.bool, device=scores.device)
    for tid, (start, end) in enumerate(boundaries):
        local = scores[start:end]
        n = local.numel()
        if n <= 0:
            continue
        k = max(2, int(math.ceil(compression_ratio * n)))
        k = min(k, n)
        idx = deterministic_topk_with_jitter(local, k=k, trajectory_id=tid)
        retained[start:end][idx] = True
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

    for tid, (start, end) in enumerate(boundaries):
        local = scores[start:end]
        n = local.numel()
        if n <= 0:
            continue
        k_total = max(2, int(math.ceil(float(compression_ratio) * n)))
        k_total = min(k_total, n)
        if mode == "swap":
            base_idx = evenly_spaced_indices(n, k_total, scores.device)
            retained[start + base_idx] = True
            protected = min(k_total, max(2, int(math.ceil(k_total * base_fraction))))
            swap_count = min(k_total - protected, n - k_total)
            if swap_count <= 0:
                continue
            removable_idx = base_idx[(base_idx != 0) & (base_idx != n - 1)]
            swap_count = min(swap_count, int(removable_idx.numel()))
            if swap_count <= 0:
                continue

            remove_pos = deterministic_topk_with_jitter(-local[removable_idx], k=swap_count, trajectory_id=tid)
            remove_idx = removable_idx[remove_pos]
            candidate_scores = local.clone()
            candidate_scores[base_idx] = -float("inf")
            add_idx = deterministic_topk_with_jitter(candidate_scores, k=swap_count, trajectory_id=tid)
            retained[start + remove_idx] = False
            retained[start + add_idx] = True
            continue

        k_base = 0 if base_fraction <= 0.0 else min(k_total, max(2, int(math.ceil(k_total * base_fraction))))
        base_idx = evenly_spaced_indices(n, k_base, scores.device)
        retained[start + base_idx] = True

        remaining = k_total - int(base_idx.numel())
        if remaining <= 0:
            continue

        candidate_scores = local.clone()
        candidate_scores[base_idx] = -float("inf")
        if bonus > 0.0 and base_idx.numel() > 0 and n > 1:
            pos = torch.arange(n, dtype=torch.float32, device=scores.device)
            dist_to_base = torch.abs(pos.unsqueeze(1) - base_idx.float().unsqueeze(0)).min(dim=1).values
            candidate_scores = candidate_scores + bonus * (dist_to_base / float(max(1, n - 1)))
            candidate_scores[base_idx] = -float("inf")

        fill_idx = deterministic_topk_with_jitter(candidate_scores, k=remaining, trajectory_id=tid)
        retained[start + fill_idx] = True

    return retained
