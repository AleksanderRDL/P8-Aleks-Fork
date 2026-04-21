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
        k = max(2, int(math.ceil(compression_ratio * n)))
        idx = deterministic_topk_with_jitter(local, k=k, trajectory_id=tid)
        retained[start:end][idx] = True
        retained[start] = True
        retained[end - 1] = True
    return retained
