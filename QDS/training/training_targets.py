"""Training target construction helpers for trajectory ranking models."""

from __future__ import annotations

import math

import torch

from simplification.simplify_trajectories import evenly_spaced_indices
from training.training_losses import _safe_quantile


def _scaled_training_target_for_type(
    labels: torch.Tensor,
    labelled_mask: torch.Tensor,
    type_idx: int,
) -> torch.Tensor:
    """Rescale one pure-workload F1 label stream while preserving rank order."""
    target = labels[:, type_idx].clone()
    positive = labelled_mask[:, type_idx] & (labels[:, type_idx] > 0)
    if not bool(positive.any().item()):
        return target.zero_()
    scale = _safe_quantile(labels[positive, type_idx].detach(), 0.95).clamp(min=1e-6)
    return torch.clamp(target / scale, 0.0, 1.0)

def _apply_temporal_residual_labels(
    labels: torch.Tensor,
    labelled_mask: torch.Tensor,
    boundaries: list[tuple[int, int]],
    compression_ratio: float,
    temporal_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop supervision for points the temporal base already keeps."""
    residual_labels = labels.clone()
    residual_mask = labelled_mask.clone()
    base_mask = torch.zeros((labels.shape[0],), dtype=torch.bool, device=labels.device)
    base_fraction = min(1.0, max(0.0, float(temporal_fraction)))

    for start, end in boundaries:
        point_count = int(end - start)
        if point_count <= 0:
            continue
        k_total = min(point_count, max(2, int(math.ceil(float(compression_ratio) * point_count))))
        k_base = 0 if base_fraction <= 0.0 else min(k_total, max(2, int(math.ceil(k_total * base_fraction))))
        base_idx = evenly_spaced_indices(point_count, k_base, labels.device)
        base_mask[start + base_idx] = True

    residual_labels[base_mask] = 0.0
    residual_mask[base_mask] = False
    return residual_labels, residual_mask
