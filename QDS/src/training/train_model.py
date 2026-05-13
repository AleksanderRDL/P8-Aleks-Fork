"""Ranking-based model training on trajectory windows. See src/training/README.md for details."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from src.experiments.experiment_config import ModelConfig
from src.queries.workload import TypedQueryWorkload
from src.experiments.torch_runtime import normalize_amp_mode, torch_autocast_context
from src.models.trajectory_qds_model import TrajectoryQDSModel
from src.models.turn_aware_qds_model import TurnAwareQDSModel
from src.queries.query_types import (
    ID_TO_QUERY_NAME,
    NUM_QUERY_TYPES,
    normalize_pure_workload_map,
    single_workload_type,
)
from src.simplification.mlqds_scoring import simplify_mlqds_predictions
from src.simplification.simplify_trajectories import evenly_spaced_indices
from src.training.checkpoint_selection import (
    CheckpointCandidate,
    normalize_checkpoint_selection_metric,
    record_validation_stats,
    selection_from_stats,
    selection_score,
)
from src.training.importance_labels import compute_typed_importance_labels
from src.training.model_features import build_model_point_features, build_model_point_features_for_dim
from src.training.scaler import FeatureScaler
from src.training.trajectory_batching import TrajectoryBatch, batch_windows, build_trajectory_windows

KENDALL_TIE_THRESHOLD = 1e-4


_QUANTILE_SUBSAMPLE_CAP = 1_000_000  # torch.quantile errors past 2^24 on some builds.


def _safe_quantile(values: torch.Tensor, quantile: float | torch.Tensor) -> torch.Tensor:
    """Quantile that tolerates very large input tensors.

    For tensors larger than ~1M elements, torch.quantile can fail with
    ``input tensor is too large``. This helper subsamples uniformly to a
    1M-element view, which gives a sufficiently accurate quantile estimate
    for diagnostic logging and label-rescaling purposes.
    """
    if values.numel() <= _QUANTILE_SUBSAMPLE_CAP:
        return torch.quantile(values, quantile)
    if values.is_floating_point():
        flat = values.detach().reshape(-1)
    else:
        flat = values.reshape(-1)
    sampled_indices = torch.randperm(flat.numel(), device=flat.device)[:_QUANTILE_SUBSAMPLE_CAP]
    return torch.quantile(flat[sampled_indices], quantile)


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


def _balanced_pointwise_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    generator: torch.Generator,
    negatives_per_positive: int = 3,
) -> torch.Tensor:
    """Compute balanced BCE on all positives plus a bounded random set of zero labels."""
    valid_idx = torch.where(valid_mask)[0]
    if valid_idx.numel() == 0:
        return pred.new_tensor(0.0)

    valid_target = target[valid_idx]
    positive_idx = valid_idx[valid_target > 0]
    if positive_idx.numel() == 0:
        return pred.new_tensor(0.0)

    zero_idx = valid_idx[valid_target <= 0]
    max_zero = int(positive_idx.numel() * max(1, negatives_per_positive))
    if zero_idx.numel() > max_zero:
        zero_sample_order = torch.randperm(zero_idx.numel(), generator=generator)[:max_zero]
        zero_idx = zero_idx[zero_sample_order.to(zero_idx.device)]

    pointwise_idx = torch.cat([positive_idx, zero_idx]) if zero_idx.numel() > 0 else positive_idx
    return F.binary_cross_entropy_with_logits(pred[pointwise_idx], target[pointwise_idx].clamp(0.0, 1.0))


def _balanced_pointwise_loss_rows(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    generator: torch.Generator,
    negatives_per_positive: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return balanced pointwise BCE for each padded row in one tensor path."""
    if pred.ndim != 2 or target.shape != pred.shape or valid_mask.shape != pred.shape:
        raise ValueError("pred, target, and valid_mask must have matching shape [batch, window_length].")

    positive = valid_mask & (target > 0.0)
    zero = valid_mask & (target <= 0.0)
    positive_count = positive.sum(dim=1)
    max_zero = positive_count * max(1, int(negatives_per_positive))
    active_rows = positive_count > 0

    if pred.device.type == "cpu":
        random_values = torch.rand(pred.shape, dtype=pred.dtype, device=pred.device, generator=generator)
    else:
        random_values = torch.rand(pred.shape, dtype=pred.dtype, device=pred.device)
    zero_order = random_values.masked_fill(~zero, float("inf")).argsort(dim=1)
    zero_rank = torch.empty_like(zero_order)
    rank_values = torch.arange(pred.shape[1], dtype=zero_order.dtype, device=pred.device).unsqueeze(0)
    zero_rank.scatter_(1, zero_order, rank_values.expand_as(zero_order))
    selected_zero = zero & (zero_rank < max_zero.unsqueeze(1))
    selected = positive | selected_zero

    per_element = F.binary_cross_entropy_with_logits(
        pred,
        target.clamp(0.0, 1.0),
        reduction="none",
    )
    selected_float = selected.to(dtype=per_element.dtype)
    denom = selected_float.sum(dim=1).clamp(min=1.0)
    row_loss = (per_element * selected_float).sum(dim=1) / denom
    return row_loss, active_rows


def _budget_topk_recall_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    budget_ratios: tuple[float, ...],
    temperature: float,
) -> torch.Tensor:
    """Differentiable retained-budget label-mass loss for one trajectory window.

    The loss approximates the final simplification decision more directly than
    pairwise ranking: for each configured retained-point budget, it asks how
    much target mass would be captured by the model's soft top-k points.
    """
    valid_idx = torch.where(valid_mask)[0]
    if valid_idx.numel() < 2:
        return pred.new_tensor(0.0)

    valid_scores = pred[valid_idx]
    valid_targets = target[valid_idx].clamp(min=0.0)
    if not bool((valid_targets > 0).any().item()):
        return pred.new_tensor(0.0)

    valid_count = int(valid_targets.numel())
    soft_keep_temperature = max(float(temperature), 1e-4)
    losses: list[torch.Tensor] = []
    for raw_ratio in budget_ratios:
        ratio = min(1.0, max(0.0, float(raw_ratio)))
        if ratio <= 0.0:
            continue
        keep_count = min(valid_count, max(1, int(math.ceil(ratio * valid_count))))
        ideal_mass = torch.topk(valid_targets, k=keep_count).values.sum().detach()
        if float(ideal_mass.item()) <= 1e-12:
            continue
        threshold = torch.topk(valid_scores.detach(), k=keep_count).values[-1]
        soft_keep = torch.sigmoid((valid_scores - threshold) / soft_keep_temperature)
        soft_keep = soft_keep * (float(keep_count) / soft_keep.sum().clamp(min=1e-6))
        soft_keep = soft_keep.clamp(max=1.0)
        captured_mass = (soft_keep * valid_targets).sum()
        recall = (captured_mass / ideal_mass.clamp(min=1e-6)).clamp(0.0, 1.0)
        losses.append(1.0 - recall)

    if not losses:
        return pred.new_tensor(0.0)
    return torch.stack(losses).mean()


def _budget_topk_recall_loss_rows(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    budget_ratios: tuple[float, ...],
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return budget-top-k loss for each row in a padded prediction batch."""
    if pred.ndim != 2 or target.shape != pred.shape or valid_mask.shape != pred.shape:
        raise ValueError("pred, target, and valid_mask must have matching shape [batch, window_length].")

    batch_size, window_length = pred.shape
    nonnegative_target = target.clamp(min=0.0)
    valid_counts = valid_mask.sum(dim=1)
    has_positive = (valid_mask & (nonnegative_target > 0.0)).any(dim=1)
    row_loss_sum = pred.new_zeros((batch_size,))
    row_loss_count = torch.zeros((batch_size,), dtype=torch.long, device=pred.device)
    soft_keep_temperature = max(float(temperature), 1e-4)

    target_sortable = nonnegative_target.masked_fill(~valid_mask, float("-inf"))
    score_sortable = pred.masked_fill(~valid_mask, float("-inf"))
    sorted_target = torch.sort(target_sortable, dim=1, descending=True).values
    sorted_score = torch.sort(score_sortable, dim=1, descending=True).values
    target_cumsum = sorted_target.clamp(min=0.0).cumsum(dim=1)

    for raw_ratio in budget_ratios:
        ratio = min(1.0, max(0.0, float(raw_ratio)))
        if ratio <= 0.0:
            continue
        keep_count_float = torch.ceil(valid_counts.float() * ratio)
        keep_count = keep_count_float.to(dtype=torch.long).clamp(min=1, max=window_length)
        active = (valid_counts >= 2) & has_positive
        if not bool(active.any().item()):
            continue

        gather_idx = (keep_count - 1).unsqueeze(1)
        ideal_mass = target_cumsum.gather(1, gather_idx).squeeze(1).detach()
        threshold = sorted_score.gather(1, gather_idx).squeeze(1).detach()
        active = active & (ideal_mass > 1e-12)
        if not bool(active.any().item()):
            continue

        soft_keep = torch.sigmoid((pred - threshold.unsqueeze(1)) / soft_keep_temperature) * valid_mask.float()
        soft_keep = soft_keep * (keep_count.float() / soft_keep.sum(dim=1).clamp(min=1e-6)).unsqueeze(1)
        soft_keep = soft_keep.clamp(max=1.0)
        captured_mass = (soft_keep * nonnegative_target).sum(dim=1)
        recall = (captured_mass / ideal_mass.clamp(min=1e-6)).clamp(0.0, 1.0)
        ratio_loss = 1.0 - recall
        row_loss_sum = torch.where(active, row_loss_sum + ratio_loss, row_loss_sum)
        row_loss_count = torch.where(active, row_loss_count + 1, row_loss_count)

    active_rows = row_loss_count > 0
    row_loss = row_loss_sum / row_loss_count.clamp(min=1).float()
    return row_loss, active_rows


def _budget_loss_ratios(model_config: ModelConfig) -> tuple[float, ...]:
    """Return configured retained-budget ratios for budget-aware loss."""
    raw = getattr(model_config, "budget_loss_ratios", None) or []
    if not raw:
        raw = getattr(model_config, "range_audit_compression_ratios", None) or []
    if not raw:
        raw = [float(getattr(model_config, "compression_ratio", 0.05))]
    ratios = sorted({float(value) for value in raw if 0.0 < float(value) <= 1.0})
    if not ratios:
        ratios = [float(getattr(model_config, "compression_ratio", 0.05))]
    return tuple(ratios)


def _effective_budget_loss_ratios(model_config: ModelConfig, temporal_residual_label_mode: str) -> tuple[float, ...]:
    """Return retained-budget ratios in the candidate set the model actually controls."""
    ratios = _budget_loss_ratios(model_config)
    if temporal_residual_label_mode != "temporal":
        return ratios

    temporal_fraction = min(1.0, max(0.0, float(getattr(model_config, "mlqds_temporal_fraction", 0.0))))
    if temporal_fraction <= 0.0:
        return ratios

    effective: list[float] = []
    for ratio in ratios:
        total_ratio = min(1.0, max(0.0, float(ratio)))
        base_ratio = min(total_ratio, total_ratio * temporal_fraction)
        fill_ratio = max(0.0, total_ratio - base_ratio)
        candidate_ratio = max(1e-6, 1.0 - base_ratio)
        value = fill_ratio / candidate_ratio
        if value > 0.0:
            effective.append(min(1.0, value))
    return tuple(effective) if effective else ratios


def _temporal_base_masks_for_budget_ratios(
    *,
    n_points: int,
    boundaries: list[tuple[int, int]],
    budget_ratios: tuple[float, ...],
    temporal_fraction: float,
    device: torch.device,
) -> tuple[tuple[float, float, torch.Tensor], ...]:
    """Return per-budget temporal-base masks and learned-fill ratios."""
    base_fraction = min(1.0, max(0.0, float(temporal_fraction)))
    if base_fraction <= 0.0:
        return ()

    masks: list[tuple[float, float, torch.Tensor]] = []
    for raw_ratio in budget_ratios:
        total_ratio = min(1.0, max(0.0, float(raw_ratio)))
        if total_ratio <= 0.0:
            continue
        base_mask = torch.zeros((n_points,), dtype=torch.bool, device=device)
        for start, end in boundaries:
            point_count = int(end - start)
            if point_count <= 0:
                continue
            k_total = min(point_count, max(2, int(math.ceil(total_ratio * point_count))))
            k_base = min(k_total, max(2, int(math.ceil(k_total * base_fraction))))
            base_idx = evenly_spaced_indices(point_count, k_base, device)
            base_mask[start + base_idx] = True
        base_ratio = float(base_mask.float().mean().item()) if n_points > 0 else 0.0
        fill_ratio = max(0.0, total_ratio - base_ratio)
        candidate_ratio = max(1e-6, 1.0 - base_ratio)
        effective_ratio = min(1.0, max(1e-6, fill_ratio / candidate_ratio))
        masks.append((total_ratio, effective_ratio, base_mask))
    return tuple(masks)


def _budget_topk_temporal_residual_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    global_idx: torch.Tensor,
    temporal_base_masks: tuple[tuple[float, float, torch.Tensor], ...],
    temperature: float,
) -> torch.Tensor:
    """Budget-top-k loss over only the per-budget learned-fill candidate points."""
    losses: list[torch.Tensor] = []
    for _total_ratio, effective_ratio, base_mask in temporal_base_masks:
        residual_mask = valid_mask & (~base_mask[global_idx])
        if not bool((residual_mask & (target > 0)).any().item()):
            continue
        losses.append(
            _budget_topk_recall_loss(
                pred=pred,
                target=target,
                valid_mask=residual_mask,
                budget_ratios=(effective_ratio,),
                temperature=temperature,
            )
        )
    if not losses:
        return pred.new_tensor(0.0)
    return torch.stack(losses).mean()


def _budget_topk_temporal_residual_loss_rows(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    global_idx: torch.Tensor,
    temporal_base_masks: tuple[tuple[float, float, torch.Tensor], ...],
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-row budget-top-k loss over learned-fill candidate points."""
    if global_idx.shape != pred.shape:
        raise ValueError("global_idx must match pred shape [batch, window_length].")

    row_loss_sum = pred.new_zeros((pred.shape[0],))
    row_loss_count = torch.zeros((pred.shape[0],), dtype=torch.long, device=pred.device)
    safe_idx = global_idx.clamp(min=0)
    for _total_ratio, effective_ratio, base_mask in temporal_base_masks:
        base_for_window = base_mask[safe_idx] & valid_mask
        residual_mask = valid_mask & (~base_for_window)
        ratio_loss, active_rows = _budget_topk_recall_loss_rows(
            pred=pred,
            target=target,
            valid_mask=residual_mask,
            budget_ratios=(effective_ratio,),
            temperature=temperature,
        )
        row_loss_sum = torch.where(active_rows, row_loss_sum + ratio_loss, row_loss_sum)
        row_loss_count = torch.where(active_rows, row_loss_count + 1, row_loss_count)

    active_rows = row_loss_count > 0
    row_loss = row_loss_sum / row_loss_count.clamp(min=1).float()
    return row_loss, active_rows


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
        "residual_label_mode": str(temporal_residual_label_mode),
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
    """Return top+bottom-quantile subsample for more reliable rank correlation. See src/training/README.md for details.

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


@dataclass
class TrainingOutputs:
    """Training artifact container. See src/training/README.md for details."""

    model: TrajectoryQDSModel
    scaler: FeatureScaler
    labels: torch.Tensor
    labelled_mask: torch.Tensor
    history: list[dict[str, float]]
    epochs_trained: int = 0
    best_epoch: int = 0
    best_loss: float = float("inf")
    best_selection_score: float = 0.0
    best_f1: float = 0.0
    target_diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Keep legacy best_f1 and explicit best_selection_score in sync."""
        if self.best_selection_score == 0.0 and self.best_f1 != 0.0:
            self.best_selection_score = float(self.best_f1)
        elif self.best_f1 == 0.0 and self.best_selection_score != 0.0:
            self.best_f1 = float(self.best_selection_score)


def _kendall_tau(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute Kendall tau for small vectors without external deps. See src/training/README.md for details."""
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


def _model_state_on_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Copy a model state dict to CPU tensors for best-epoch restoration."""
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _ranking_loss_for_type(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    pairs_per_type: int,
    top_quantile: float,
    margin: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    """Compute top-boundary-focused pairwise ranking loss for one type. See src/training/README.md for details."""
    valid_idx = torch.where(valid_mask)[0]
    if valid_idx.numel() < 2:
        return pred.new_tensor(0.0), 0

    valid_targets = target[valid_idx]
    top_threshold = _safe_quantile(
        valid_targets,
        torch.tensor(top_quantile, dtype=torch.float32, device=valid_targets.device),
    )
    top_idx = valid_idx[valid_targets >= top_threshold]
    strict_top_idx = valid_idx[valid_targets > top_threshold]
    if strict_top_idx.numel() > 0 and top_idx.numel() > max(4, valid_idx.numel() // 2):
        top_idx = strict_top_idx
    if top_idx.numel() == 0:
        top_idx = valid_idx

    sample_count = max(1, int(pairs_per_type))
    # The run-level generator is CPU-backed. Draw small position tensors on
    # CPU for deterministic consumption, then move only the sampled positions
    # to the model device instead of synchronizing labels/indices back to CPU.
    top_sample_pos = torch.randint(0, top_idx.numel(), (sample_count,), generator=generator)
    comparison_sample_pos = torch.randint(0, valid_idx.numel(), (sample_count,), generator=generator)
    top_sample_idx = top_idx[top_sample_pos.to(top_idx.device)]
    comparison_idx = valid_idx[comparison_sample_pos.to(valid_idx.device)]
    keep_pair = (top_sample_idx != comparison_idx) & ~torch.isclose(
        target[top_sample_idx],
        target[comparison_idx],
    )
    if not bool(keep_pair.any().item()):
        return pred.new_tensor(0.0), 0
    top_sample_idx = top_sample_idx[keep_pair]
    comparison_idx = comparison_idx[keep_pair]

    target_order = torch.sign(target[top_sample_idx] - target[comparison_idx])
    return (
        F.margin_ranking_loss(pred[top_sample_idx], pred[comparison_idx], target_order, margin=margin),
        int(top_sample_idx.numel()),
    )


def _workload_map_tensor(workload_map: dict[str, float], device: torch.device) -> torch.Tensor:
    """Return normalized pure-workload weights in query-type ID order."""
    normalized = normalize_pure_workload_map(workload_map)
    values = torch.tensor(
        [
            float(normalized.get("range", 0.0)),
            float(normalized.get("knn", 0.0)),
            float(normalized.get("similarity", 0.0)),
            float(normalized.get("clustering", 0.0)),
        ],
        dtype=torch.float32,
        device=device,
    )
    return values


def _query_frequency_workload_map(workload: TypedQueryWorkload) -> dict[str, float]:
    """Infer type weights from a workload when no explicit training workload map is provided."""
    counts = torch.bincount(workload.type_ids.detach().cpu(), minlength=NUM_QUERY_TYPES).float()
    return {
        "range": float(counts[0].item()),
        "knn": float(counts[1].item()),
        "similarity": float(counts[2].item()),
        "clustering": float(counts[3].item()),
    }


def _single_active_type_id(type_weights: torch.Tensor) -> int:
    """Return the one active query type for pure-workload training."""
    active = torch.where(type_weights.detach().cpu() > 0.0)[0]
    if int(active.numel()) != 1:
        raise ValueError("Pure-workload training requires exactly one active query type.")
    return int(active[0].item())


def _pure_query_type_id(type_ids: torch.Tensor) -> int:
    """Return the only query type id in a pure workload."""
    unique_ids = torch.unique(type_ids.detach().cpu())
    if int(unique_ids.numel()) != 1:
        raise ValueError("Pure-workload training/evaluation requires exactly one query type id.")
    return int(unique_ids[0].item())


def _window_has_positive_supervision(
    window: TrajectoryBatch,
    training_target: torch.Tensor,
    labelled_mask: torch.Tensor,
) -> bool:
    """Return whether a pure-workload window has positive supervision."""
    global_indices = window.global_indices.reshape(-1)
    valid_points = global_indices >= 0
    if not bool(valid_points.any().item()):
        return False
    valid_indices = global_indices[valid_points].to(device=training_target.device, dtype=torch.long)
    return bool((labelled_mask[valid_indices] & (training_target[valid_indices] > 0)).any().item())


def _filter_supervised_windows(
    windows: list[TrajectoryBatch],
    training_target: torch.Tensor,
    labelled_mask: torch.Tensor,
    active_type_id: int,
) -> tuple[list[TrajectoryBatch], torch.Tensor]:
    """Drop windows that cannot contribute loss for the active pure workload."""
    if not windows:
        return windows, torch.zeros((NUM_QUERY_TYPES,), dtype=torch.long)

    kept: list[TrajectoryBatch] = []
    filtered_zero_windows = torch.zeros((NUM_QUERY_TYPES,), dtype=torch.long)
    for window in windows:
        if _window_has_positive_supervision(window, training_target, labelled_mask):
            kept.append(window)
            continue
        filtered_zero_windows[active_type_id] += 1

    if not kept:
        return windows, torch.zeros((NUM_QUERY_TYPES,), dtype=torch.long)
    return kept, filtered_zero_windows


def _trajectory_batch_to_device(batch: TrajectoryBatch, device: torch.device) -> TrajectoryBatch:
    """Move one already-batched trajectory window group to the model device."""
    return TrajectoryBatch(
        points=batch.points.to(device=device, non_blocking=True),
        padding_mask=batch.padding_mask.to(device=device, non_blocking=True),
        trajectory_ids=batch.trajectory_ids,
        global_indices=batch.global_indices.to(device=device, non_blocking=True),
    )


def _predict_workload_logits(
    model: TrajectoryQDSModel,
    scaler: FeatureScaler,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    workload: TypedQueryWorkload,
    model_config: ModelConfig,
    device: torch.device,
) -> torch.Tensor:
    """Predict per-point pure-workload scores for exact validation-score diagnostics."""
    model_points = build_model_point_features_for_dim(points, workload, model.point_dim)
    norm_points, norm_queries = scaler.transform(model_points, workload.query_features)
    norm_points_dev = norm_points.to(device)
    norm_queries_dev = norm_queries.to(device)
    type_ids_dev = workload.type_ids.to(device)
    _pure_query_type_id(workload.type_ids)
    windows = build_trajectory_windows(
        points=norm_points,
        boundaries=boundaries,
        window_length=model_config.window_length,
        stride=model_config.window_stride,
    )
    inference_batch_size = max(1, int(getattr(model_config, "inference_batch_size", 16)))
    windows = batch_windows(windows, inference_batch_size)
    point_score_sum = norm_points_dev.new_zeros((norm_points_dev.shape[0],))
    point_score_count = norm_points_dev.new_zeros((norm_points_dev.shape[0],))
    amp_mode = normalize_amp_mode(getattr(model_config, "amp_mode", "off"))

    model.eval()
    with torch.no_grad():
        for window_batch_cpu in windows:
            window = _trajectory_batch_to_device(window_batch_cpu, device)
            with torch_autocast_context(device, amp_mode):
                window_scores = model(
                    points=window.points,
                    queries=norm_queries_dev,
                    query_type_ids=type_ids_dev,
                    padding_mask=window.padding_mask,
                )
            window_scores = window_scores.float()
            for batch_idx in range(window_scores.shape[0]):
                global_idx = window.global_indices[batch_idx]
                valid = global_idx >= 0
                point_score_sum[global_idx[valid]] = (
                    point_score_sum[global_idx[valid]] + window_scores[batch_idx, valid]
                )
                point_score_count[global_idx[valid]] = point_score_count[global_idx[valid]] + 1.0

    point_score_count = point_score_count.clamp(min=1.0)
    return (point_score_sum / point_score_count).detach().cpu()


def _validation_checkpoint_scores(
    model: TrajectoryQDSModel,
    scaler: FeatureScaler,
    trajectories: list[torch.Tensor],
    boundaries: list[tuple[int, int]],
    workload: TypedQueryWorkload,
    workload_map: dict[str, float],
    model_config: ModelConfig,
    device: torch.device,
    validation_points: torch.Tensor | None = None,
    query_cache: Any | None = None,
    range_geometry_scores: torch.Tensor | None = None,
) -> tuple[float, dict[str, float], dict[str, float]]:
    """Evaluate a checkpoint and return selected score plus explicit validation metrics."""
    from src.evaluation.evaluate_methods import score_range_usefulness, score_retained_mask

    points = validation_points if validation_points is not None else torch.cat(trajectories, dim=0)
    predictions = _predict_workload_logits(
        model=model,
        scaler=scaler,
        points=points,
        boundaries=boundaries,
        workload=workload,
        model_config=model_config,
        device=device,
    )
    retained_mask = simplify_mlqds_predictions(
        predictions,
        boundaries,
        single_workload_type(workload_map),
        model_config.compression_ratio,
        temporal_fraction=float(getattr(model_config, "mlqds_temporal_fraction", 0.50)),
        diversity_bonus=float(getattr(model_config, "mlqds_diversity_bonus", 0.0)),
        hybrid_mode=str(getattr(model_config, "mlqds_hybrid_mode", "fill")),
        score_mode=str(getattr(model_config, "mlqds_score_mode", "rank")),
        score_temperature=float(getattr(model_config, "mlqds_score_temperature", 1.0)),
        rank_confidence_weight=float(getattr(model_config, "mlqds_rank_confidence_weight", 0.15)),
        range_geometry_scores=range_geometry_scores,
        range_geometry_blend=float(getattr(model_config, "mlqds_range_geometry_blend", 0.0)),
    )
    answer_agg, answer_pt, combined_agg, combined_pt = score_retained_mask(
        points=points,
        boundaries=boundaries,
        retained_mask=retained_mask,
        typed_queries=workload.typed_queries,
        workload_map=workload_map,
        query_cache=query_cache,
    )
    metrics = {
        "answer_f1": float(answer_agg),
        "combined_f1": float(combined_agg),
        "range_point_f1": float(answer_pt.get("range", 0.0)),
    }
    range_audit: dict[str, Any] | None = None
    if any(str(query.get("type", "")).lower() == "range" for query in workload.typed_queries):
        range_audit = score_range_usefulness(
            points=points,
            boundaries=boundaries,
            retained_mask=retained_mask,
            typed_queries=workload.typed_queries,
            query_cache=query_cache,
        )
        metrics.update(
            {
                "range_usefulness": float(range_audit["range_usefulness_score"]),
                "range_ship_f1": float(range_audit["range_ship_f1"]),
                "range_ship_coverage": float(range_audit["range_ship_coverage"]),
                "range_entry_exit_f1": float(range_audit["range_entry_exit_f1"]),
                "range_crossing_f1": float(range_audit["range_crossing_f1"]),
                "range_temporal_coverage": float(range_audit["range_temporal_coverage"]),
                "range_gap_coverage": float(range_audit["range_gap_coverage"]),
                "range_turn_coverage": float(range_audit["range_turn_coverage"]),
                "range_shape_score": float(range_audit["range_shape_score"]),
            }
        )
    variant = str(getattr(model_config, "checkpoint_score_variant", "range_usefulness")).lower()
    if variant == "range_usefulness":
        if range_audit is None:
            return float(answer_agg), answer_pt, metrics
        score = float(range_audit["range_usefulness_score"])
        return score, {"range": score}, metrics
    if variant == "combined":
        return float(combined_agg), combined_pt, metrics
    return float(answer_agg), answer_pt, metrics


def _validation_query_score(
    model: TrajectoryQDSModel,
    scaler: FeatureScaler,
    trajectories: list[torch.Tensor],
    boundaries: list[tuple[int, int]],
    workload: TypedQueryWorkload,
    workload_map: dict[str, float],
    model_config: ModelConfig,
    device: torch.device,
    validation_points: torch.Tensor | None = None,
    query_cache: Any | None = None,
    range_geometry_scores: torch.Tensor | None = None,
) -> tuple[float, dict[str, float]]:
    """Return the active held-out validation score for checkpoint selection."""
    score, per_type, _metrics = _validation_checkpoint_scores(
        model=model,
        scaler=scaler,
        trajectories=trajectories,
        boundaries=boundaries,
        workload=workload,
        workload_map=workload_map,
        model_config=model_config,
        device=device,
        validation_points=validation_points,
        query_cache=query_cache,
        range_geometry_scores=range_geometry_scores,
    )
    return score, per_type


def _validation_query_f1(
    model: TrajectoryQDSModel,
    scaler: FeatureScaler,
    trajectories: list[torch.Tensor],
    boundaries: list[tuple[int, int]],
    workload: TypedQueryWorkload,
    workload_map: dict[str, float],
    model_config: ModelConfig,
    device: torch.device,
    validation_points: torch.Tensor | None = None,
    query_cache: Any | None = None,
    range_geometry_scores: torch.Tensor | None = None,
) -> tuple[float, dict[str, float]]:
    """Legacy alias for _validation_query_score."""
    return _validation_query_score(
        model=model,
        scaler=scaler,
        trajectories=trajectories,
        boundaries=boundaries,
        workload=workload,
        workload_map=workload_map,
        model_config=model_config,
        device=device,
        validation_points=validation_points,
        query_cache=query_cache,
        range_geometry_scores=range_geometry_scores,
    )


def _validation_uniform_score(
    trajectories: list[torch.Tensor],
    boundaries: list[tuple[int, int]],
    workload: TypedQueryWorkload,
    workload_map: dict[str, float],
    model_config: ModelConfig,
    validation_points: torch.Tensor | None = None,
    query_cache: Any | None = None,
) -> tuple[float, dict[str, float]]:
    """Evaluate fair uniform on the held-out validation workload once per run."""
    from src.evaluation.baselines import UniformTemporalMethod
    from src.evaluation.evaluate_methods import score_range_usefulness, score_retained_mask

    points = validation_points if validation_points is not None else torch.cat(trajectories, dim=0)
    retained_mask = UniformTemporalMethod().simplify(
        points=points,
        boundaries=boundaries,
        compression_ratio=model_config.compression_ratio,
    )
    answer_agg, answer_pt, combined_agg, combined_pt = score_retained_mask(
        points=points,
        boundaries=boundaries,
        retained_mask=retained_mask,
        typed_queries=workload.typed_queries,
        workload_map=workload_map,
        query_cache=query_cache,
    )
    variant = str(getattr(model_config, "checkpoint_score_variant", "range_usefulness")).lower()
    if variant == "range_usefulness":
        audit = score_range_usefulness(
            points=points,
            boundaries=boundaries,
            retained_mask=retained_mask,
            typed_queries=workload.typed_queries,
            query_cache=query_cache,
        )
        score = float(audit["range_usefulness_score"])
        return score, {"range": score}
    if variant == "combined":
        return combined_agg, combined_pt
    return answer_agg, answer_pt


def train_model(
    train_trajectories: list[torch.Tensor],
    train_boundaries: list[tuple[int, int]],
    workload: TypedQueryWorkload,
    model_config: ModelConfig,
    seed: int,
    train_workload_map: dict[str, float] | None = None,
    validation_trajectories: list[torch.Tensor] | None = None,
    validation_boundaries: list[tuple[int, int]] | None = None,
    validation_workload: TypedQueryWorkload | None = None,
    validation_workload_map: dict[str, float] | None = None,
    precomputed_labels: tuple[torch.Tensor, torch.Tensor] | None = None,
    validation_points: torch.Tensor | None = None,
    precomputed_validation_query_cache: Any | None = None,
    precomputed_validation_geometry_scores: torch.Tensor | None = None,
) -> TrainingOutputs:
    """Train one pure-workload model with trajectory-window ranking losses."""
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    all_points = torch.cat(train_trajectories, dim=0)
    points = build_model_point_features(all_points, workload, model_config.model_type)
    point_dim = int(points.shape[1])

    if precomputed_labels is None:
        labels, labelled_mask = compute_typed_importance_labels(
            points=all_points,
            boundaries=train_boundaries,
            typed_queries=workload.typed_queries,
            seed=seed,
            range_label_mode=str(getattr(model_config, "range_label_mode", "usefulness")),
            range_boundary_prior_weight=float(getattr(model_config, "range_boundary_prior_weight", 0.0)),
        )
    else:
        labels, labelled_mask = precomputed_labels
        expected_shape = (all_points.shape[0], NUM_QUERY_TYPES)
        if labels.shape != expected_shape or labelled_mask.shape != expected_shape:
            raise ValueError(
                "precomputed_labels must match flattened training points and query type count: "
                f"expected {expected_shape}, got labels={tuple(labels.shape)} mask={tuple(labelled_mask.shape)}"
            )
    temporal_residual_label_mode = str(getattr(model_config, "temporal_residual_label_mode", "none")).lower()
    if temporal_residual_label_mode not in {"none", "temporal"}:
        raise ValueError("temporal_residual_label_mode must be 'none' or 'temporal'.")
    loss_objective = str(getattr(model_config, "loss_objective", "budget_topk")).lower()
    if loss_objective not in {"ranking_bce", "budget_topk"}:
        raise ValueError("loss_objective must be 'ranking_bce' or 'budget_topk'.")
    configured_budget_ratios = _budget_loss_ratios(model_config)
    budget_ratios = configured_budget_ratios
    temporal_residual_budget_masks: tuple[tuple[float, float, torch.Tensor], ...] = ()
    temporal_residual_union_mask: torch.Tensor | None = None
    workload_type_id = _pure_query_type_id(workload.type_ids)
    if temporal_residual_label_mode == "temporal" and loss_objective == "budget_topk":
        budget_ratios = _effective_budget_loss_ratios(model_config, temporal_residual_label_mode)
        temporal_residual_budget_masks = _temporal_base_masks_for_budget_ratios(
            n_points=int(labels.shape[0]),
            boundaries=train_boundaries,
            budget_ratios=configured_budget_ratios,
            temporal_fraction=float(getattr(model_config, "mlqds_temporal_fraction", 0.50)),
            device=labels.device,
        )
        if temporal_residual_budget_masks:
            temporal_residual_union_mask = torch.zeros((labels.shape[0],), dtype=torch.bool, device=labels.device)
            for _total_ratio, _effective_ratio, base_mask in temporal_residual_budget_masks:
                temporal_residual_union_mask |= base_mask
    elif temporal_residual_label_mode == "temporal":
        labels, labelled_mask = _apply_temporal_residual_labels(
            labels=labels,
            labelled_mask=labelled_mask,
            boundaries=train_boundaries,
            compression_ratio=model_config.compression_ratio,
            temporal_fraction=float(getattr(model_config, "mlqds_temporal_fraction", 0.50)),
        )
    training_target = _scaled_training_target_for_type(labels, labelled_mask, workload_type_id)
    training_labelled_mask = labelled_mask[:, workload_type_id]

    scaler = FeatureScaler.fit(points, workload.query_features)
    norm_points, norm_queries = scaler.transform(points, workload.query_features)

    model_cls = TurnAwareQDSModel if model_config.model_type == "turn_aware" else TrajectoryQDSModel
    model = model_cls(
        point_dim=point_dim,
        query_dim=norm_queries.shape[1],
        embed_dim=model_config.embed_dim,
        num_heads=model_config.num_heads,
        num_layers=model_config.num_layers,
        type_embed_dim=model_config.type_embed_dim,
        query_chunk_size=model_config.query_chunk_size,
        dropout=model_config.dropout,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    norm_points_dev = norm_points.to(device)
    norm_queries_dev = norm_queries.to(device)
    type_ids_dev = workload.type_ids.to(device)
    training_target_dev = training_target.to(device)
    labelled_mask_dev = training_labelled_mask.to(device)
    base_type_weights = _workload_map_tensor(
        train_workload_map or _query_frequency_workload_map(workload),
        device=device,
    )
    active_type_id = _single_active_type_id(base_type_weights)
    if active_type_id != workload_type_id:
        raise ValueError("Training workload map and workload query type must refer to the same pure workload.")
    active_type_ids = [active_type_id]
    amp_mode = normalize_amp_mode(getattr(model_config, "amp_mode", "off"))
    budget_loss_temperature = float(getattr(model_config, "budget_loss_temperature", 0.10))
    run_tag = "main"
    target_diagnostics = _training_target_diagnostics(
        labels=labels,
        labelled_mask=labelled_mask,
        workload_type_id=workload_type_id,
        configured_budget_ratios=configured_budget_ratios,
        effective_budget_ratios=budget_ratios,
        temporal_residual_budget_masks=temporal_residual_budget_masks,
        temporal_residual_label_mode=temporal_residual_label_mode,
        loss_objective=loss_objective,
        temporal_fraction=float(getattr(model_config, "mlqds_temporal_fraction", 0.50)),
    )
    if budget_ratios != configured_budget_ratios:
        print(
            f"  [{run_tag}] effective_budget_loss_ratios={list(budget_ratios)} "
            f"from configured={list(configured_budget_ratios)} "
            f"temporal_residual_label_mode={temporal_residual_label_mode} "
            f"mlqds_temporal_fraction={float(getattr(model_config, 'mlqds_temporal_fraction', 0.0)):.3f}",
            flush=True,
        )
    for row in target_diagnostics.get("budget_rows", []):
        print(
            f"  [{run_tag}] residual_budget total={row['total_budget_ratio']:.4f} "
            f"effective_fill={row['effective_fill_budget_ratio']:.4f} "
            f"base_points={row['temporal_base_point_count']} "
            f"candidates={row['candidate_point_count']} "
            f"residual_pos={row['residual_positive_label_count']}",
            flush=True,
        )
    if temporal_residual_budget_masks:
        temporal_residual_budget_masks = tuple(
            (total_ratio, effective_ratio, base_mask.to(device=device, non_blocking=True))
            for total_ratio, effective_ratio, base_mask in temporal_residual_budget_masks
        )
    if temporal_residual_union_mask is not None:
        temporal_residual_union_mask = temporal_residual_union_mask.to(device=device, non_blocking=True)

    opt = torch.optim.Adam(model.parameters(), lr=model_config.lr)
    grad_scaler = torch.amp.GradScaler("cuda", enabled=(amp_mode == "fp16" and device.type == "cuda"))
    windows_cpu = build_trajectory_windows(
        points=norm_points,
        boundaries=train_boundaries,
        window_length=model_config.window_length,
        stride=model_config.window_stride,
    )
    raw_window_count = len(windows_cpu)
    windows_cpu, prefiltered_zero_windows = _filter_supervised_windows(
        windows=windows_cpu,
        training_target=training_target,
        labelled_mask=training_labelled_mask,
        active_type_id=active_type_id,
    )
    if int(prefiltered_zero_windows.sum().item()) > 0:
        filtered_parts = []
        for type_idx in active_type_ids:
            type_name = ID_TO_QUERY_NAME.get(type_idx, f"t{type_idx}")
            filtered_parts.append(f"{type_name}={int(prefiltered_zero_windows[type_idx].item())}")
        print(
            f"  [{run_tag}] filtered {raw_window_count - len(windows_cpu)}/{raw_window_count} "
            f"zero-positive training windows before forward ({', '.join(filtered_parts)})",
            flush=True,
        )
    train_batch_size = max(1, int(getattr(model_config, "train_batch_size", 1)))
    windows = batch_windows(windows_cpu, train_batch_size)
    trained_window_count = len(windows_cpu)
    # Keep diagnostics as sampleable single windows, then batch the selected
    # subset before forward so sampled diagnostics still use useful GPU work.
    diag_windows = windows_cpu
    diag_every = max(1, int(getattr(model_config, "diagnostic_every", 1)))
    diag_fraction = float(getattr(model_config, "diagnostic_window_fraction", 1.0))
    diag_fraction = min(1.0, max(0.05, diag_fraction))
    selection_metric = normalize_checkpoint_selection_metric(
        getattr(model_config, "checkpoint_selection_metric", "score")
    )
    if selection_metric not in {"loss", "score", "uniform_gap"}:
        raise ValueError("checkpoint_selection_metric must be 'loss', 'score', 'f1', or 'uniform_gap'.")
    validation_score_every = int(getattr(model_config, "validation_score_every", 0) or 0)
    has_validation_score = (
        validation_trajectories is not None
        and validation_boundaries is not None
        and validation_workload is not None
        and validation_workload_map is not None
    )
    if selection_metric in {"score", "uniform_gap"} and not has_validation_score:
        print(
            f"  [{run_tag}] WARNING: checkpoint_selection_metric={selection_metric} "
            "requested without validation workload; "
            "falling back to loss selection.",
            flush=True,
        )
        selection_metric = "loss"
    if selection_metric in {"score", "uniform_gap"} and validation_score_every <= 0:
        validation_score_every = diag_every
    validation_points_for_score: torch.Tensor | None = None
    validation_query_cache: Any | None = None
    if has_validation_score:
        from src.evaluation.query_cache import EvaluationQueryCache

        assert validation_trajectories is not None
        assert validation_boundaries is not None
        assert validation_workload is not None
        validation_points_for_score = (
            validation_points
            if validation_points is not None
            else torch.cat(validation_trajectories, dim=0)
        )
        if precomputed_validation_query_cache is None:
            validation_query_cache = EvaluationQueryCache.for_workload(
                validation_points_for_score,
                validation_boundaries,
                validation_workload.typed_queries,
            )
        else:
            precomputed_validation_query_cache.validate(
                validation_points_for_score,
                validation_boundaries,
                validation_workload.typed_queries,
            )
            validation_query_cache = precomputed_validation_query_cache
    validation_uniform_result: tuple[float, dict[str, float]] | None = None
    if selection_metric == "uniform_gap" and has_validation_score:
        assert validation_trajectories is not None
        assert validation_boundaries is not None
        assert validation_workload is not None
        validation_uniform_result = _validation_uniform_score(
            trajectories=validation_trajectories,
            boundaries=validation_boundaries,
            workload=validation_workload,
            workload_map=validation_workload_map or {},
            model_config=model_config,
            validation_points=validation_points_for_score,
            query_cache=validation_query_cache,
        )
        uniform_score, uniform_per_type = validation_uniform_result
        print(
            f"  [{run_tag}] validation uniform_score={uniform_score:.6f}  "
            f"range={uniform_per_type.get('range', 0.0):.6f}  "
            f"knn={uniform_per_type.get('knn', 0.0):.6f}  "
            f"similarity={uniform_per_type.get('similarity', 0.0):.6f}  "
            f"clustering={uniform_per_type.get('clustering', 0.0):.6f}",
            flush=True,
        )

    training_sample_generator = torch.Generator().manual_seed(int(seed) + 99)
    # Separate fixed-seed generator for diagnostics so the tau subsample
    # stays consistent across epochs and doesn't oscillate with training state.
    diagnostic_sample_generator = torch.Generator().manual_seed(int(seed) + 777)
    history: list[dict[str, float]] = []

    effective_epochs = max(1, int(model_config.epochs))
    patience = int(getattr(model_config, "early_stopping_patience", 0) or 0)
    smoothing_window = max(1, int(getattr(model_config, "checkpoint_smoothing_window", 1) or 1))
    checkpoint_full_score_every = max(1, int(getattr(model_config, "checkpoint_full_score_every", 1) or 1))
    checkpoint_candidate_pool_size = max(1, int(getattr(model_config, "checkpoint_candidate_pool_size", 1) or 1))
    checkpoint_candidates: list[CheckpointCandidate] = []
    selection_history: list[float] = []
    best_selection = float("-inf")
    best_loss = float("inf")
    best_selection_score = 0.0
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] | None = None
    epochs_no_improve = 0
    epoch_label_width = len(str(effective_epochs))
    epochs_trained = 0
    for epoch in range(effective_epochs):
        epoch_t0 = time.perf_counter()
        epoch_timing = {
            "forward_s": 0.0,
            "loss_s": 0.0,
            "backward_s": 0.0,
            "diagnostic_s": 0.0,
            "validation_score_s": 0.0,
        }
        evaluated_checkpoint_candidates: list[CheckpointCandidate] = []
        model.train()
        epoch_loss = torch.tensor(0.0, device=device)
        positive_windows = torch.zeros((NUM_QUERY_TYPES,), dtype=torch.long)
        skipped_zero_windows = prefiltered_zero_windows.clone()
        ranking_pair_counts = torch.zeros((NUM_QUERY_TYPES,), dtype=torch.long)

        for window_batch_cpu in windows:
            window_batch = _trajectory_batch_to_device(window_batch_cpu, device)
            forward_t0 = time.perf_counter()
            with torch_autocast_context(device, amp_mode):
                pred_batch = model(
                    points=window_batch.points,
                    queries=norm_queries_dev,
                    query_type_ids=type_ids_dev,
                    padding_mask=window_batch.padding_mask,
                )
            epoch_timing["forward_s"] += time.perf_counter() - forward_t0
            loss_t0 = time.perf_counter()
            pred_batch = pred_batch.float()
            # pred_batch: (batch, window_length). Build the active objective in batched tensor
            # paths where possible so batch-size increases create larger GPU
            # work units instead of many tiny per-window loss calls.
            loss: torch.Tensor | None = None
            batch_size = pred_batch.shape[0]
            batch_global_idx = window_batch.global_indices.to(device=device)
            valid_batch = batch_global_idx >= 0
            safe_global_idx = batch_global_idx.clamp(min=0)
            batch_labels = training_target_dev[safe_global_idx]
            batch_label_mask = labelled_mask_dev[safe_global_idx] & valid_batch
            positive_row_mask = (batch_label_mask & (batch_labels > 0)).any(dim=1)
            positive_windows[active_type_id] += int(positive_row_mask.sum().item())
            skipped_zero_windows[active_type_id] += int((~positive_row_mask).sum().item())
            pointwise_mask_batch = batch_label_mask
            if temporal_residual_union_mask is not None:
                base_for_batch = temporal_residual_union_mask[safe_global_idx] & valid_batch
                pointwise_mask_batch = batch_label_mask & (~base_for_batch)
            pointwise_loss_rows, _pointwise_active_rows = _balanced_pointwise_loss_rows(
                pred=pred_batch,
                target=batch_labels,
                valid_mask=pointwise_mask_batch,
                generator=training_sample_generator,
            )

            if loss_objective == "budget_topk":
                if temporal_residual_budget_masks:
                    rank_loss_rows, _rank_active_rows = _budget_topk_temporal_residual_loss_rows(
                        pred=pred_batch,
                        target=batch_labels,
                        valid_mask=batch_label_mask,
                        global_idx=safe_global_idx,
                        temporal_base_masks=temporal_residual_budget_masks,
                        temperature=budget_loss_temperature,
                    )
                else:
                    rank_loss_rows, _rank_active_rows = _budget_topk_recall_loss_rows(
                        pred=pred_batch,
                        target=batch_labels,
                        valid_mask=batch_label_mask,
                        budget_ratios=budget_ratios,
                        temperature=budget_loss_temperature,
                    )

                if bool(positive_row_mask.any().item()):
                    row_losses = rank_loss_rows + model_config.pointwise_loss_weight * pointwise_loss_rows
                    loss = (
                        row_losses[positive_row_mask].sum() / float(batch_size)
                        + model_config.l2_score_weight * (pred_batch ** 2).mean()
                    )
            else:
                loss_terms: list[torch.Tensor] = []
                for row_index in torch.where(positive_row_mask.detach().cpu())[0].tolist():
                    row = int(row_index)
                    window_global_idx = batch_global_idx[row]
                    valid_window = window_global_idx >= 0
                    valid_global_idx = window_global_idx[valid_window]
                    valid_pred = pred_batch[row][valid_window]
                    rank_loss, pair_count = _ranking_loss_for_type(
                        pred=valid_pred,
                        target=training_target_dev[valid_global_idx],
                        valid_mask=labelled_mask_dev[valid_global_idx],
                        pairs_per_type=model_config.ranking_pairs_per_type,
                        top_quantile=model_config.ranking_top_quantile,
                        margin=model_config.rank_margin,
                        generator=training_sample_generator,
                    )
                    ranking_pair_counts[active_type_id] += int(pair_count)
                    loss_terms.append(rank_loss + model_config.pointwise_loss_weight * pointwise_loss_rows[row])
                if loss_terms:
                    loss = (
                        torch.stack(loss_terms).sum() / float(batch_size)
                        + model_config.l2_score_weight * (pred_batch ** 2).mean()
                    )
            epoch_timing["loss_s"] += time.perf_counter() - loss_t0

            if loss is not None:
                backward_t0 = time.perf_counter()
                opt.zero_grad(set_to_none=True)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite training loss with amp_mode={amp_mode}: {float(loss.item())}")
                clip_norm = float(getattr(model_config, "gradient_clip_norm", 0.0) or 0.0)
                if grad_scaler.is_enabled():
                    grad_scaler.scale(loss).backward()
                    if clip_norm > 0.0:
                        grad_scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
                    grad_scaler.step(opt)
                    grad_scaler.update()
                else:
                    loss.backward()
                    if clip_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
                    opt.step()
                epoch_loss = epoch_loss + loss.detach()
                epoch_timing["backward_s"] += time.perf_counter() - backward_t0

        # Diagnostic pass only on selected epochs (every `diag_every` epochs and
        # the final epoch).  Subsample windows by `diag_fraction` to further cut
        # cost: pred_std and tau are statistical aggregates and noise from a
        # ~20% sample is tiny compared to the training noise we're measuring.
        is_last_epoch = (epoch + 1) == effective_epochs
        is_diag_epoch = ((epoch + 1) % diag_every == 0) or is_last_epoch or epoch == 0
        if is_diag_epoch:
            diagnostic_t0 = time.perf_counter()
            if diag_fraction < 1.0 and len(diag_windows) > 8:
                diagnostic_window_count = max(8, int(len(diag_windows) * diag_fraction))
                sample_indices = torch.randperm(
                    len(diag_windows),
                    generator=diagnostic_sample_generator,
                )[:diagnostic_window_count].tolist()
                diagnostic_windows = [diag_windows[i] for i in sample_indices]
            else:
                diagnostic_windows = diag_windows

            model.eval()
            with torch.no_grad():
                diagnostic_score_sum = norm_points_dev.new_zeros((norm_points_dev.shape[0],))
                diagnostic_score_count = norm_points_dev.new_zeros((norm_points_dev.shape[0],))
                diagnostic_batch_size = max(1, int(getattr(model_config, "inference_batch_size", train_batch_size)))
                for diagnostic_batch_cpu in batch_windows(diagnostic_windows, diagnostic_batch_size):
                    diagnostic_batch = _trajectory_batch_to_device(diagnostic_batch_cpu, device)
                    with torch_autocast_context(device, amp_mode):
                        window_scores = model(
                            points=diagnostic_batch.points,
                            queries=norm_queries_dev,
                            query_type_ids=type_ids_dev,
                            padding_mask=diagnostic_batch.padding_mask,
                        )
                    window_scores = window_scores.float()
                    for batch_idx in range(window_scores.shape[0]):
                        point_indices = diagnostic_batch.global_indices[batch_idx]
                        valid_points = point_indices >= 0
                        diagnostic_score_sum[point_indices[valid_points]] = (
                            diagnostic_score_sum[point_indices[valid_points]]
                            + window_scores[batch_idx, valid_points]
                        )
                        diagnostic_score_count[point_indices[valid_points]] = (
                            diagnostic_score_count[point_indices[valid_points]] + 1.0
                        )
                covered_mask = diagnostic_score_count > 0
                diagnostic_score_count = diagnostic_score_count.clamp(min=1.0)
                full_scores = diagnostic_score_sum / diagnostic_score_count

            stats: dict[str, float] = {
                "epoch": float(epoch),
                "loss": float(epoch_loss.item() / max(1, len(windows))),
                "pred_std": (
                    float(full_scores[covered_mask].std().item())
                    if bool(covered_mask.any().item())
                    else 0.0
                ),
            }
            for type_idx in range(NUM_QUERY_TYPES):
                stats[f"positive_windows_t{type_idx}"] = float(positive_windows[type_idx].item())
                stats[f"skipped_zero_windows_t{type_idx}"] = float(skipped_zero_windows[type_idx].item())
                stats[f"ranking_pairs_t{type_idx}"] = float(ranking_pair_counts[type_idx].item())
                stats[f"pred_p50_t{type_idx}"] = 0.0
                stats[f"pred_p90_t{type_idx}"] = 0.0
                stats[f"pred_p99_t{type_idx}"] = 0.0
                stats[f"positive_fraction_t{type_idx}"] = 0.0
                stats[f"label_p95_t{type_idx}"] = 0.0
                stats[f"kendall_tau_t{type_idx}"] = 0.0
            for t in range(NUM_QUERY_TYPES):
                if t != active_type_id:
                    continue
                type_scores = full_scores
                stats[f"pred_p50_t{t}"] = float(_safe_quantile(type_scores, 0.50).item())
                stats[f"pred_p90_t{t}"] = float(_safe_quantile(type_scores, 0.90).item())
                stats[f"pred_p99_t{t}"] = float(_safe_quantile(type_scores, 0.99).item())
                labelled_type = labelled_mask_dev
                positive_type = labelled_type & (training_target_dev > 0)
                labelled_count = max(1, int(labelled_type.sum().item()))
                stats[f"positive_fraction_t{t}"] = float(positive_type.sum().item() / labelled_count)
                if bool(positive_type.any().item()):
                    stats[f"label_p95_t{t}"] = float(_safe_quantile(training_target_dev[positive_type], 0.95).item())
                else:
                    stats[f"label_p95_t{t}"] = 0.0
                eval_mask = labelled_mask_dev & covered_mask
                if bool(eval_mask.any().item()):
                    # Reset the diagnostic generator each epoch so the diagnostic
                    # subsample is identical across epochs, giving stable tau trends.
                    diagnostic_sample_generator.manual_seed(int(seed) + 777)
                    pred_sample, target_sample = _discriminative_sample(
                        type_scores[eval_mask].detach().cpu(),
                        training_target_dev[eval_mask].detach().cpu(),
                        n_each=100,
                        generator=diagnostic_sample_generator,
                    )
                    stats[f"kendall_tau_t{t}"] = _kendall_tau(pred_sample, target_sample)
                else:
                    stats[f"kendall_tau_t{t}"] = 0.0

            if stats["pred_std"] < 1e-3:
                stats["collapse_warning"] = 1.0
            epoch_timing["diagnostic_s"] += time.perf_counter() - diagnostic_t0

            candidate_tau_vals = [stats[f"kendall_tau_t{t}"] for t in active_type_ids]
            candidate_avg_tau = sum(candidate_tau_vals) / max(1, len(candidate_tau_vals))
            validation_score_due = (
                validation_score_every <= 0
                or (epoch + 1) % validation_score_every == 0
                or is_last_epoch
                or epoch == 0
            )
            full_score_due = validation_score_due and (
                checkpoint_full_score_every <= 1
                or (epoch + 1) % checkpoint_full_score_every == 0
                or is_last_epoch
            )
            use_checkpoint_candidate_pool = (
                has_validation_score
                and validation_score_due
                and selection_metric in {"score", "uniform_gap"}
                and checkpoint_full_score_every > 1
            )
            should_run_validation_score = has_validation_score and full_score_due and (
                selection_metric in {"score", "uniform_gap"} or validation_score_every > 0
            ) and not use_checkpoint_candidate_pool
            if should_run_validation_score:
                score_t0 = time.perf_counter()
                assert validation_trajectories is not None
                assert validation_boundaries is not None
                assert validation_workload is not None
                validation_score, per_type_score, validation_metrics = _validation_checkpoint_scores(
                    model=model,
                    scaler=scaler,
                    trajectories=validation_trajectories,
                    boundaries=validation_boundaries,
                    workload=validation_workload,
                    workload_map=validation_workload_map or {},
                    model_config=model_config,
                    device=device,
                    validation_points=validation_points_for_score,
                    query_cache=validation_query_cache,
                    range_geometry_scores=precomputed_validation_geometry_scores,
                )
                epoch_timing["validation_score_s"] += time.perf_counter() - score_t0
                record_validation_stats(
                    stats,
                    validation_score=validation_score,
                    per_type_score=per_type_score,
                    validation_metrics=validation_metrics,
                    validation_uniform_result=validation_uniform_result,
                    validation_workload_map=validation_workload_map,
                )
            if has_validation_score and validation_score_due and selection_metric in {"score", "uniform_gap"}:
                stats["checkpoint_score_candidate"] = 1.0
                stats["checkpoint_f1_candidate"] = 1.0
                stats["checkpoint_candidate_cheap_score"] = selection_score(
                    candidate_avg_tau,
                    stats["pred_std"],
                    stats["loss"],
                )
                stats["checkpoint_full_score_due"] = 1.0 if full_score_due else 0.0
                stats["checkpoint_full_f1_due"] = stats["checkpoint_full_score_due"]
                if use_checkpoint_candidate_pool:
                    checkpoint_candidates.append(
                        CheckpointCandidate(
                            epoch_number=epoch + 1,
                            epoch_index=epoch,
                            cheap_score=float(stats["checkpoint_candidate_cheap_score"]),
                            loss=float(stats["loss"]),
                            state_dict=_model_state_on_cpu(model),
                            stats=stats,
                            avg_tau=candidate_avg_tau,
                        )
                    )
                    checkpoint_candidates.sort(key=lambda candidate: candidate.cheap_score, reverse=True)
                    checkpoint_candidates = checkpoint_candidates[:checkpoint_candidate_pool_size]
                    if full_score_due and checkpoint_candidates:
                        score_t0 = time.perf_counter()
                        assert validation_trajectories is not None
                        assert validation_boundaries is not None
                        assert validation_workload is not None
                        current_state_dict = _model_state_on_cpu(model)
                        for candidate in sorted(checkpoint_candidates, key=lambda item: item.epoch_number):
                            candidate_t0 = time.perf_counter()
                            model.load_state_dict(candidate.state_dict)
                            validation_score, per_type_score, validation_metrics = _validation_checkpoint_scores(
                                model=model,
                                scaler=scaler,
                                trajectories=validation_trajectories,
                                boundaries=validation_boundaries,
                                workload=validation_workload,
                                workload_map=validation_workload_map or {},
                                model_config=model_config,
                                device=device,
                                validation_points=validation_points_for_score,
                                query_cache=validation_query_cache,
                                range_geometry_scores=precomputed_validation_geometry_scores,
                            )
                            record_validation_stats(
                                candidate.stats,
                                validation_score=validation_score,
                                per_type_score=per_type_score,
                                validation_metrics=validation_metrics,
                                validation_uniform_result=validation_uniform_result,
                                validation_workload_map=validation_workload_map,
                            )
                            candidate.stats["checkpoint_candidate_evaluated"] = 1.0
                            candidate.stats["checkpoint_full_score_round_epoch"] = float(epoch + 1)
                            candidate.stats["checkpoint_full_f1_round_epoch"] = float(epoch + 1)
                            candidate.stats["checkpoint_validation_seconds"] = float(time.perf_counter() - candidate_t0)
                            evaluated_checkpoint_candidates.append(candidate)
                        model.load_state_dict(current_state_dict)
                        epoch_timing["validation_score_s"] += time.perf_counter() - score_t0
                        checkpoint_candidates = []
        else:
            # Skip diagnostics this epoch; log only loss.  Patience counters
            # are only updated on diagnostic epochs below.
            stats = {
                "epoch": float(epoch),
                "loss": float(epoch_loss.item() / max(1, len(windows))),
            }
            for type_idx in range(NUM_QUERY_TYPES):
                stats[f"positive_windows_t{type_idx}"] = float(positive_windows[type_idx].item())
                stats[f"skipped_zero_windows_t{type_idx}"] = float(skipped_zero_windows[type_idx].item())
                stats[f"ranking_pairs_t{type_idx}"] = float(ranking_pair_counts[type_idx].item())

        epoch_dt = time.perf_counter() - epoch_t0
        stats["epoch_seconds"] = float(epoch_dt)
        stats["epoch_forward_seconds"] = float(epoch_timing["forward_s"])
        stats["epoch_loss_seconds"] = float(epoch_timing["loss_s"])
        stats["epoch_backward_seconds"] = float(epoch_timing["backward_s"])
        stats["epoch_diagnostic_seconds"] = float(epoch_timing["diagnostic_s"])
        stats["epoch_validation_score_seconds"] = float(epoch_timing["validation_score_s"])
        stats["epoch_f1_seconds"] = stats["epoch_validation_score_seconds"]
        stats["raw_training_window_count"] = float(raw_window_count)
        stats["trained_training_window_count"] = float(trained_window_count)
        stats["filtered_zero_window_count"] = float(raw_window_count - trained_window_count)
        for type_idx in range(NUM_QUERY_TYPES):
            stats[f"filtered_zero_windows_t{type_idx}"] = float(prefiltered_zero_windows[type_idx].item())
        history.append(stats)

        epochs_trained = epoch + 1

        if is_diag_epoch:
            tau_vals = [stats[f"kendall_tau_t{t}"] for t in active_type_ids]
            avg_tau = sum(tau_vals) / max(1, len(tau_vals))
            collapse = "  COLLAPSE" if stats.get("collapse_warning") else ""
            selection: float | None = None
            smoothed_selection: float | None = None
            is_new_best_model = False
            validation_round_had_selection = False
            validation_round_improved = False
            if evaluated_checkpoint_candidates:
                for candidate in sorted(evaluated_checkpoint_candidates, key=lambda item: item.epoch_number):
                    candidate_selection = selection_from_stats(
                        stats=candidate.stats,
                        avg_tau=candidate.avg_tau,
                        selection_metric=selection_metric,
                        validation_uniform_result=validation_uniform_result,
                        validation_workload_map=validation_workload_map,
                        model_config=model_config,
                    )
                    if candidate_selection is None:
                        continue
                    validation_round_had_selection = True
                    candidate.stats["selection_score"] = candidate_selection
                    selection_history.append(float(candidate_selection))
                    window = selection_history[-smoothing_window:]
                    candidate_smoothed = float(sum(window) / len(window))
                    candidate.stats["selection_score_smoothed"] = candidate_smoothed
                    candidate_is_new_best = candidate_smoothed > best_selection + 1e-4 or (
                        abs(candidate_smoothed - best_selection) <= 1e-4 and candidate.loss < best_loss - 1e-8
                    )
                    if candidate_is_new_best:
                        validation_round_improved = True
                        best_selection = candidate_smoothed
                        best_loss = candidate.loss
                        best_selection_score = float(candidate.stats.get("val_selection_score", best_selection_score))
                        best_epoch = candidate.epoch_number
                        best_state_dict = candidate.state_dict
                        candidate.stats["checkpoint_promoted"] = 1.0
                    else:
                        candidate.stats["checkpoint_promoted"] = 0.0
                    if candidate.stats is not stats:
                        status = "promoted" if candidate_is_new_best else "checked"
                        print(
                            f"  [{run_tag}] checkpoint candidate epoch "
                            f"{candidate.epoch_number:0{epoch_label_width}d}/{effective_epochs}  "
                            f"cheap={candidate.cheap_score:+.3f}  "
                            f"select={candidate_selection:+.3f}  "
                            f"smoothed={candidate_smoothed:+.3f}  {status}",
                            flush=True,
                        )
                if "selection_score" in stats:
                    selection = float(stats["selection_score"])
                    smoothed_selection = float(stats["selection_score_smoothed"])
                    is_new_best_model = bool(stats.get("checkpoint_promoted", 0.0))
            else:
                selection = selection_from_stats(
                    stats=stats,
                    avg_tau=avg_tau,
                    selection_metric=selection_metric,
                    validation_uniform_result=validation_uniform_result,
                    validation_workload_map=validation_workload_map,
                    model_config=model_config,
                )
                if selection is not None:
                    validation_round_had_selection = True
                    stats["selection_score"] = selection
                    selection_history.append(float(selection))
                    window = selection_history[-smoothing_window:]
                    smoothed_score = float(sum(window) / len(window))
                    smoothed_selection = smoothed_score
                    stats["selection_score_smoothed"] = smoothed_score
                    # Use the smoothed score for "best" decisions: averages out
                    # epoch-to-epoch validation score noise so we don't lock onto a lucky
                    # spike. Single-epoch loss still tiebreaks on near-equal smoothed.
                    is_new_best_model = smoothed_score > best_selection + 1e-4 or (
                        abs(smoothed_score - best_selection) <= 1e-4 and stats["loss"] < best_loss - 1e-8
                    )
                    validation_round_improved = is_new_best_model
            markers = []
            if epoch > 0 and is_new_best_model:
                markers.append("*** NEW BEST MODEL ***")
            best_marker = ("  " + "  ".join(markers)) if markers else ""
            smoothed_label = (
                f"  smoothed_w{smoothing_window}={smoothed_selection:+.3f}"
                if smoothing_window > 1 and smoothed_selection is not None
                else ""
            )
            selection_text = f"{selection:+.3f}" if selection is not None else "skipped"
            print(
                f"  [{run_tag}] epoch {epoch + 1:0{epoch_label_width}d}/{effective_epochs}  "
                f"loss={stats['loss']:.8f}  avg_tau={avg_tau:+.3f}  "
                f"pred_std={stats['pred_std']:.6g}  select={selection_text}{smoothed_label}  "
                f"({epoch_dt:.2f}s){collapse}{best_marker}",
                flush=True,
            )
            if "val_selection_score" in stats:
                print(
                    f"    [{run_tag}] val_selection_score={stats['val_selection_score']:.6f}  "
                    f"range_point_f1={stats.get('val_range_point_f1', 0.0):.6f}  "
                    f"range_usefulness={stats.get('val_range_usefulness', 0.0):.6f}  "
                    f"answer_f1={stats.get('val_answer_f1', 0.0):.6f}  "
                    f"combined_f1={stats.get('val_combined_f1', 0.0):.6f}",
                    flush=True,
                )
            if "val_uniform_score" in stats:
                print(
                    f"    [{run_tag}] val_vs_uniform aggregate={stats['val_selection_uniform_gap']:+.6f}  "
                    f"type_deficit={stats['val_selection_type_deficit']:.6f}  "
                    f"range={stats.get('val_selection_score_gap_range', 0.0):+.6f}  "
                    f"knn={stats.get('val_selection_score_gap_knn', 0.0):+.6f}  "
                    f"similarity={stats.get('val_selection_score_gap_similarity', 0.0):+.6f}  "
                    f"clustering={stats.get('val_selection_score_gap_clustering', 0.0):+.6f}",
                    flush=True,
                )
            diag_parts = []
            for type_idx in active_type_ids:
                type_name = ID_TO_QUERY_NAME.get(type_idx, f"t{type_idx}")
                diag_parts.append(
                    f"{type_name}:pos={stats[f'positive_fraction_t{type_idx}']:.4f},"
                    f"p95={stats[f'label_p95_t{type_idx}']:.3f},"
                    f"pairs={int(stats[f'ranking_pairs_t{type_idx}'])},"
                    f"skip={int(stats[f'skipped_zero_windows_t{type_idx}'])},"
                    f"filtered={int(stats[f'filtered_zero_windows_t{type_idx}'])}"
                )
            if diag_parts:
                print(f"    [{run_tag}] label_diag  " + "  ".join(diag_parts), flush=True)
            print(
                f"    [{run_tag}] epoch_timing  "
                f"forward={stats['epoch_forward_seconds']:.2f}s  "
                f"loss={stats['epoch_loss_seconds']:.2f}s  "
                f"backward={stats['epoch_backward_seconds']:.2f}s  "
                f"diagnostic={stats['epoch_diagnostic_seconds']:.2f}s  "
                f"validation_score={stats['epoch_validation_score_seconds']:.2f}s  "
                f"filtered_zero_windows={int(stats['filtered_zero_window_count'])}",
                flush=True,
            )

            if is_new_best_model:
                best_selection = float(stats["selection_score_smoothed"])
                best_loss = stats["loss"]
                best_selection_score = float(stats.get("val_selection_score", best_selection_score))
                best_epoch = epoch + 1
                best_state_dict = _model_state_on_cpu(model)

            if patience > 0 and validation_round_had_selection:
                if is_new_best_model or validation_round_improved:
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(
                            f"  [{run_tag}] early stopping at epoch {epoch + 1:0{epoch_label_width}d}: "
                            f"selection score did not improve over {patience} diag epochs "
                            f"(best_selection={best_selection:+.3f}, best_loss={best_loss:.8f})",
                            flush=True,
                        )
                        break
        else:
            # Non-diagnostic epoch: log loss only, no tau / early-stopping update.
            print(
                f"  [{run_tag}] epoch {epoch + 1:0{epoch_label_width}d}/{effective_epochs}  "
                f"loss={stats['loss']:.8f}  (no-diag)  ({epoch_dt:.2f}s)",
                flush=True,
            )
            print(
                f"    [{run_tag}] epoch_timing  "
                f"forward={stats['epoch_forward_seconds']:.2f}s  "
                f"loss={stats['epoch_loss_seconds']:.2f}s  "
                f"backward={stats['epoch_backward_seconds']:.2f}s  "
                f"filtered_zero_windows={int(stats['filtered_zero_window_count'])}",
                flush=True,
            )

    model = model.to("cpu")
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(
            f"  [{run_tag}] restored best diagnostic epoch {best_epoch}/{epochs_trained} "
            f"(selection={best_selection:+.3f}, loss={best_loss:.8f}, "
            f"val_selection_score={best_selection_score:.6f})",
            flush=True,
        )
    return TrainingOutputs(
        model=model,
        scaler=scaler,
        labels=labels,
        labelled_mask=labelled_mask,
        history=history,
        epochs_trained=epochs_trained,
        best_epoch=best_epoch,
        best_loss=best_loss,
        best_selection_score=best_selection_score,
        best_f1=best_selection_score,
        target_diagnostics=target_diagnostics,
    )
