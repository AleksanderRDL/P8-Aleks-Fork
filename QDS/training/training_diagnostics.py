"""Training target and rank-correlation diagnostics."""

from __future__ import annotations

from typing import Any

import torch

from training.training_losses import _safe_quantile

KENDALL_TIE_THRESHOLD = 1e-4


def _training_target_diagnostics(
    *,
    labels: torch.Tensor,
    labelled_mask: torch.Tensor,
    workload_type_id: int,
    configured_budget_ratios: tuple[float, ...],
    effective_budget_ratios: tuple[float, ...],
    temporal_residual_budget_masks: tuple[tuple[float, float, torch.Tensor], ...],
    temporal_residual_label_mode: str,
    loss_objective: str,
    temporal_fraction: float,
) -> dict[str, Any]:
    """Summarize the effective supervised target after residual masking."""
    active = labelled_mask[:, workload_type_id].bool()
    values = labels[:, workload_type_id].float()
    positive = active & (values > 0.0)
    n_points = int(labels.shape[0])
    labelled_count = int(active.sum().item())
    positive_count = int(positive.sum().item())
    total_positive_label_mass = float(values[positive].sum().item()) if bool(positive.any().item()) else 0.0
    diagnostics: dict[str, Any] = {
        "workload_type_id": int(workload_type_id),
        "temporal_residual_label_mode": str(temporal_residual_label_mode),
        "loss_objective": str(loss_objective),
        "mlqds_temporal_fraction": float(temporal_fraction),
        "configured_budget_loss_ratios": [float(value) for value in configured_budget_ratios],
        "effective_budget_loss_ratios": [float(value) for value in effective_budget_ratios],
        "point_count": n_points,
        "labelled_point_count": labelled_count,
        "positive_label_count": positive_count,
        "positive_label_fraction": float(positive_count / max(1, labelled_count)),
        "positive_label_mass": total_positive_label_mass,
        "budget_rows": [],
    }

    rows: list[dict[str, Any]] = []
    for total_ratio, effective_ratio, base_mask in temporal_residual_budget_masks:
        base = base_mask.to(device=labels.device, dtype=torch.bool)
        candidate = ~base
        residual_labelled = active & candidate
        residual_positive = positive & candidate
        base_positive = positive & base
        base_count = int(base.sum().item())
        candidate_count = int(candidate.sum().item())
        residual_labelled_count = int(residual_labelled.sum().item())
        residual_positive_count = int(residual_positive.sum().item())
        base_label_mass = float(values[base_positive].sum().item()) if bool(base_positive.any().item()) else 0.0
        residual_label_mass = (
            float(values[residual_positive].sum().item()) if bool(residual_positive.any().item()) else 0.0
        )
        rows.append(
            {
                "total_budget_ratio": float(total_ratio),
                "effective_fill_budget_ratio": float(effective_ratio),
                "temporal_base_point_count": base_count,
                "temporal_base_point_fraction": float(base_count / max(1, n_points)),
                "candidate_point_count": candidate_count,
                "candidate_point_fraction": float(candidate_count / max(1, n_points)),
                "base_positive_label_count": int(base_positive.sum().item()),
                "residual_labelled_point_count": residual_labelled_count,
                "residual_positive_label_count": residual_positive_count,
                "residual_positive_label_fraction": float(residual_positive_count / max(1, residual_labelled_count)),
                "temporal_base_label_mass": base_label_mass,
                "residual_label_mass": residual_label_mass,
                "temporal_base_label_mass_fraction": float(
                    base_label_mass / max(1e-12, total_positive_label_mass)
                ),
                "residual_label_mass_fraction": float(
                    residual_label_mass / max(1e-12, total_positive_label_mass)
                ),
            }
        )
    diagnostics["budget_rows"] = rows
    return diagnostics

def _discriminative_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_each: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return top+bottom-quantile subsample for more reliable rank correlation. See training/README.md for details.

    Computing Kendall tau on all labelled points is O(N^2) and noisy when the
    label distribution has many near-tied pairs.  Restricting to the top and
    bottom quantiles focuses the statistic on the pairs the ranker is expected
    to separate, where the signal is strongest.
    """
    target_count = target.numel()
    if target_count <= 2 * n_each:
        return pred, target
    quartiles = _safe_quantile(target, torch.tensor([0.25, 0.75], dtype=torch.float32, device=target.device))
    bottom_quantile_indices = torch.where(target <= quartiles[0])[0]
    top_quantile_indices = torch.where(target >= quartiles[1])[0]
    bottom_sample_order = torch.randperm(bottom_quantile_indices.numel(), generator=generator)[:n_each]
    top_sample_order = torch.randperm(top_quantile_indices.numel(), generator=generator)[:n_each]
    sampled_indices = torch.cat(
        [
            bottom_quantile_indices[bottom_sample_order],
            top_quantile_indices[top_sample_order],
        ]
    )
    return pred[sampled_indices], target[sampled_indices]


def _kendall_tau(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute Kendall tau for small vectors without external deps. See training/README.md for details."""
    sample_count = int(predictions.numel())
    if sample_count < 2:
        return 0.0
    prediction_delta = predictions.unsqueeze(0) - predictions.unsqueeze(1)
    target_delta = targets.unsqueeze(0) - targets.unsqueeze(1)
    upper_triangle = torch.triu(torch.ones_like(prediction_delta, dtype=torch.bool), diagonal=1)
    tied_target = target_delta.abs() < KENDALL_TIE_THRESHOLD
    pair_order = prediction_delta * target_delta
    concordant = int(((pair_order > 0) & upper_triangle & ~tied_target).sum().item())
    discordant = int(((pair_order < 0) & upper_triangle & ~tied_target).sum().item())
    denom = max(1, concordant + discordant)
    return float((concordant - discordant) / denom)
