"""Ranking-based model training on trajectory windows. See src/training/README.md for details."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from src.experiments.experiment_config import ModelConfig, TypedQueryWorkload
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
from src.training.importance_labels import compute_typed_importance_labels
from src.training.scaler import FeatureScaler
from src.training.trajectory_batching import TrajectoryBatch, batch_windows, build_trajectory_windows

KENDALL_TIE_THRESHOLD = 1e-4


_QUANTILE_SUBSAMPLE_CAP = 1_000_000  # torch.quantile errors past 2^24 on some builds.


def _safe_quantile(t: torch.Tensor, q: float | torch.Tensor) -> torch.Tensor:
    """Quantile that tolerates very large input tensors.

    For tensors larger than ~1M elements, torch.quantile can fail with
    ``input tensor is too large``. This helper subsamples uniformly to a
    1M-element view, which gives a sufficiently accurate quantile estimate
    for diagnostic logging and label-rescaling purposes.
    """
    if t.numel() <= _QUANTILE_SUBSAMPLE_CAP:
        return torch.quantile(t, q)
    if t.is_floating_point():
        flat = t.detach().reshape(-1)
    else:
        flat = t.reshape(-1)
    perm = torch.randperm(flat.numel(), device=flat.device)[:_QUANTILE_SUBSAMPLE_CAP]
    return torch.quantile(flat[perm], q)


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
        n = int(end - start)
        if n <= 0:
            continue
        k_total = min(n, max(2, int(math.ceil(float(compression_ratio) * n))))
        k_base = 0 if base_fraction <= 0.0 else min(k_total, max(2, int(math.ceil(k_total * base_fraction))))
        base_idx = evenly_spaced_indices(n, k_base, labels.device)
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
        perm = torch.randperm(zero_idx.numel(), generator=generator)[:max_zero]
        zero_idx = zero_idx[perm.to(zero_idx.device)]

    pointwise_idx = torch.cat([positive_idx, zero_idx]) if zero_idx.numel() > 0 else positive_idx
    return F.binary_cross_entropy_with_logits(pred[pointwise_idx], target[pointwise_idx].clamp(0.0, 1.0))


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

    x = pred[valid_idx]
    y = target[valid_idx].clamp(min=0.0)
    if not bool((y > 0).any().item()):
        return pred.new_tensor(0.0)

    n = int(y.numel())
    tau = max(float(temperature), 1e-4)
    losses: list[torch.Tensor] = []
    for raw_ratio in budget_ratios:
        ratio = min(1.0, max(0.0, float(raw_ratio)))
        if ratio <= 0.0:
            continue
        k = min(n, max(1, int(math.ceil(ratio * n))))
        ideal_mass = torch.topk(y, k=k).values.sum().detach()
        if float(ideal_mass.item()) <= 1e-12:
            continue
        threshold = torch.topk(x.detach(), k=k).values[-1]
        soft_keep = torch.sigmoid((x - threshold) / tau)
        soft_keep = soft_keep * (float(k) / soft_keep.sum().clamp(min=1e-6))
        soft_keep = soft_keep.clamp(max=1.0)
        captured_mass = (soft_keep * y).sum()
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
        raise ValueError("pred, target, and valid_mask must have matching shape [B, L].")

    B, L = pred.shape
    y = target.clamp(min=0.0)
    valid_counts = valid_mask.sum(dim=1)
    has_positive = (valid_mask & (y > 0.0)).any(dim=1)
    row_loss_sum = pred.new_zeros((B,))
    row_loss_count = torch.zeros((B,), dtype=torch.long, device=pred.device)
    tau = max(float(temperature), 1e-4)

    y_sortable = y.masked_fill(~valid_mask, float("-inf"))
    x_sortable = pred.masked_fill(~valid_mask, float("-inf"))
    sorted_y = torch.sort(y_sortable, dim=1, descending=True).values
    sorted_x = torch.sort(x_sortable, dim=1, descending=True).values
    y_cumsum = sorted_y.clamp(min=0.0).cumsum(dim=1)

    for raw_ratio in budget_ratios:
        ratio = min(1.0, max(0.0, float(raw_ratio)))
        if ratio <= 0.0:
            continue
        k_float = torch.ceil(valid_counts.float() * ratio)
        k = k_float.to(dtype=torch.long).clamp(min=1, max=L)
        active = (valid_counts >= 2) & has_positive
        if not bool(active.any().item()):
            continue

        gather_idx = (k - 1).unsqueeze(1)
        ideal_mass = y_cumsum.gather(1, gather_idx).squeeze(1).detach()
        threshold = sorted_x.gather(1, gather_idx).squeeze(1).detach()
        active = active & (ideal_mass > 1e-12)
        if not bool(active.any().item()):
            continue

        soft_keep = torch.sigmoid((pred - threshold.unsqueeze(1)) / tau) * valid_mask.float()
        soft_keep = soft_keep * (k.float() / soft_keep.sum(dim=1).clamp(min=1e-6)).unsqueeze(1)
        soft_keep = soft_keep.clamp(max=1.0)
        captured_mass = (soft_keep * y).sum(dim=1)
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


def _effective_budget_loss_ratios(model_config: ModelConfig, residual_label_mode: str) -> tuple[float, ...]:
    """Return retained-budget ratios in the candidate set the model actually controls."""
    ratios = _budget_loss_ratios(model_config)
    if residual_label_mode != "temporal":
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
            n = int(end - start)
            if n <= 0:
                continue
            k_total = min(n, max(2, int(math.ceil(total_ratio * n))))
            k_base = min(k_total, max(2, int(math.ceil(k_total * base_fraction))))
            base_idx = evenly_spaced_indices(n, k_base, device)
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
        raise ValueError("global_idx must match pred shape [B, L].")

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
    residual_label_mode: str,
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
        "residual_label_mode": str(residual_label_mode),
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
    n = target.numel()
    if n <= 2 * n_each:
        return pred, target
    q = _safe_quantile(target, torch.tensor([0.25, 0.75], dtype=torch.float32, device=target.device))
    bot = torch.where(target <= q[0])[0]
    top = torch.where(target >= q[1])[0]
    perm_b = torch.randperm(bot.numel(), generator=generator)[:n_each]
    perm_t = torch.randperm(top.numel(), generator=generator)[:n_each]
    idx = torch.cat([bot[perm_b], top[perm_t]])
    return pred[idx], target[idx]


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


def _kendall_tau(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute Kendall tau for small vectors without external deps. See src/training/README.md for details."""
    n = int(x.numel())
    if n < 2:
        return 0.0
    dx = x.unsqueeze(0) - x.unsqueeze(1)
    dy = y.unsqueeze(0) - y.unsqueeze(1)
    upper = torch.triu(torch.ones_like(dx, dtype=torch.bool), diagonal=1)
    tie = dy.abs() < KENDALL_TIE_THRESHOLD
    prod = dx * dy
    concordant = int(((prod > 0) & upper & ~tie).sum().item())
    discordant = int(((prod < 0) & upper & ~tie).sum().item())
    denom = max(1, concordant + discordant)
    return float((concordant - discordant) / denom)


def _model_state_on_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Copy a model state dict to CPU tensors for best-epoch restoration."""
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


@dataclass
class _CheckpointCandidate:
    """Candidate snapshot awaiting exact validation in a full-F1 round."""

    epoch_number: int
    epoch_index: int
    cheap_score: float
    loss: float
    state_dict: dict[str, torch.Tensor]
    stats: dict[str, float]
    avg_tau: float


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

    y = target[valid_idx]
    q_val = _safe_quantile(y, torch.tensor(top_quantile, dtype=torch.float32, device=y.device))
    top_idx = valid_idx[y >= q_val]
    strict_top_idx = valid_idx[y > q_val]
    if strict_top_idx.numel() > 0 and top_idx.numel() > max(4, valid_idx.numel() // 2):
        top_idx = strict_top_idx
    if top_idx.numel() == 0:
        top_idx = valid_idx

    sample_count = max(1, int(pairs_per_type))
    # The run-level generator is CPU-backed. Draw small position tensors on
    # CPU for deterministic consumption, then move only the sampled positions
    # to the model device instead of synchronizing labels/indices back to CPU.
    a_pos = torch.randint(0, top_idx.numel(), (sample_count,), generator=generator)
    b_pos = torch.randint(0, valid_idx.numel(), (sample_count,), generator=generator)
    a_idx = top_idx[a_pos.to(top_idx.device)]
    b_idx = valid_idx[b_pos.to(valid_idx.device)]
    keep = (a_idx != b_idx) & ~torch.isclose(target[a_idx], target[b_idx])
    if not bool(keep.any().item()):
        return pred.new_tensor(0.0), 0
    a_idx = a_idx[keep]
    b_idx = b_idx[keep]

    sign = torch.sign(target[a_idx] - target[b_idx])
    return F.margin_ranking_loss(pred[a_idx], pred[b_idx], sign, margin=margin), int(a_idx.numel())


def _selection_score(avg_tau: float, pred_std: float, loss: float | None = None) -> float:
    """Score checkpoint quality while strongly penalizing collapsed predictions."""
    collapse_penalty = 1.0 if pred_std < 1e-3 else 0.0
    if loss is None:
        return float(avg_tau - collapse_penalty)
    return float(-float(loss) + 1e-3 * avg_tau - collapse_penalty)


def _f1_selection_score(query_f1: float, pred_std: float) -> float:
    """Score checkpoints by final query-F1 while rejecting collapsed predictions."""
    collapse_penalty = 1.0 if pred_std < 1e-3 else 0.0
    return float(query_f1 - collapse_penalty)


def _normalized_workload_map(workload_map: dict[str, float]) -> dict[str, float]:
    """Normalize a pure workload map into the fixed query-type key set."""
    names = ["range", "knn", "similarity", "clustering"]
    normalized = normalize_pure_workload_map(workload_map)
    return {name: float(normalized.get(name, 0.0)) for name in names}


def _uniform_type_deficit(
    per_type_f1: dict[str, float],
    uniform_per_type: dict[str, float],
    workload_map: dict[str, float],
) -> float:
    """Weighted amount by which a checkpoint loses to fair uniform per type."""
    type_weights = _normalized_workload_map(workload_map)
    return float(
        sum(
            type_weights[name] * max(0.0, float(uniform_per_type.get(name, 0.0)) - float(per_type_f1.get(name, 0.0)))
            for name in type_weights
        )
    )


def _uniform_gap_selection_score(
    query_f1: float,
    per_type_f1: dict[str, float],
    uniform_f1: float,
    uniform_per_type: dict[str, float],
    workload_map: dict[str, float],
    pred_std: float,
    aggregate_gap_weight: float = 0.5,
    type_penalty_weight: float = 1.0,
) -> float:
    """Score checkpoints by held-out F1 while penalizing losses to fair uniform."""
    collapse_penalty = 1.0 if pred_std < 1e-3 else 0.0
    aggregate_gap = float(query_f1) - float(uniform_f1)
    type_deficit = _uniform_type_deficit(per_type_f1, uniform_per_type, workload_map)
    return float(
        float(query_f1)
        + float(aggregate_gap_weight) * aggregate_gap
        - float(type_penalty_weight) * type_deficit
        - collapse_penalty
    )


def _record_validation_stats(
    stats: dict[str, float],
    *,
    validation_score: float,
    per_type_f1: dict[str, float],
    validation_metrics: dict[str, float],
    validation_uniform_result: tuple[float, dict[str, float]] | None,
    validation_workload_map: dict[str, float] | None,
) -> None:
    """Attach exact validation metrics to an epoch stats row."""
    stats["val_selection_score"] = float(validation_score)
    for metric_name, metric_value in validation_metrics.items():
        stats[f"val_{metric_name}"] = float(metric_value)
    for type_name, value in per_type_f1.items():
        stats[f"val_selection_score_{type_name}"] = float(value)
    if validation_uniform_result is not None:
        uniform_f1, uniform_per_type = validation_uniform_result
        stats["val_uniform_f1"] = float(uniform_f1)
        stats["val_query_uniform_gap"] = float(validation_score - uniform_f1)
        stats["val_query_type_deficit"] = _uniform_type_deficit(
            per_type_f1,
            uniform_per_type,
            validation_workload_map or {},
        )
        for type_name, value in uniform_per_type.items():
            stats[f"val_uniform_f1_{type_name}"] = float(value)
            stats[f"val_selection_score_gap_{type_name}"] = float(per_type_f1.get(type_name, 0.0) - value)


def _selection_from_stats(
    *,
    stats: dict[str, float],
    avg_tau: float,
    selection_metric: str,
    validation_uniform_result: tuple[float, dict[str, float]] | None,
    validation_workload_map: dict[str, float] | None,
    model_config: ModelConfig,
) -> float | None:
    """Return the active checkpoint selection score for one stats row."""
    if (
        selection_metric == "uniform_gap"
        and "val_selection_score" in stats
        and validation_uniform_result is not None
    ):
        uniform_f1, uniform_per_type = validation_uniform_result
        per_type_f1 = {
            name: stats.get(f"val_selection_score_{name}", 0.0)
            for name in ["range", "knn", "similarity", "clustering"]
        }
        return _uniform_gap_selection_score(
            query_f1=stats["val_selection_score"],
            per_type_f1=per_type_f1,
            uniform_f1=uniform_f1,
            uniform_per_type=uniform_per_type,
            workload_map=validation_workload_map or {},
            pred_std=stats["pred_std"],
            aggregate_gap_weight=float(getattr(model_config, "checkpoint_uniform_gap_weight", 0.5)),
            type_penalty_weight=float(getattr(model_config, "checkpoint_type_penalty_weight", 1.0)),
        )
    if selection_metric == "f1" and "val_selection_score" in stats:
        return _f1_selection_score(stats["val_selection_score"], stats["pred_std"])
    if selection_metric in {"f1", "uniform_gap"}:
        return None
    return _selection_score(avg_tau, stats["pred_std"], stats["loss"])


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
    global_idx = window.global_indices.reshape(-1)
    valid = global_idx >= 0
    if not bool(valid.any().item()):
        return False
    idx = global_idx[valid].to(device=training_target.device, dtype=torch.long)
    return bool((labelled_mask[idx] & (training_target[idx] > 0)).any().item())


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


def _predict_workload_logits(
    model: TrajectoryQDSModel,
    scaler: FeatureScaler,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    workload: TypedQueryWorkload,
    model_config: ModelConfig,
    device: torch.device,
) -> torch.Tensor:
    """Predict per-point pure-workload scores for exact query-F1 diagnostics."""
    point_dim = model.point_dim
    norm_points, norm_queries = scaler.transform(points[:, :point_dim].float(), workload.query_features)
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
    windows = [
        TrajectoryBatch(
            points=window.points.to(device),
            padding_mask=window.padding_mask.to(device),
            trajectory_ids=window.trajectory_ids,
            global_indices=window.global_indices.to(device),
        )
        for window in windows
    ]
    inference_batch_size = max(1, int(getattr(model_config, "inference_batch_size", 16)))
    windows = batch_windows(windows, inference_batch_size)
    all_pred = norm_points_dev.new_zeros((norm_points_dev.shape[0],))
    pred_count = norm_points_dev.new_zeros((norm_points_dev.shape[0],))
    amp_mode = normalize_amp_mode(getattr(model_config, "amp_mode", "off"))

    model.eval()
    with torch.no_grad():
        for window in windows:
            with torch_autocast_context(device, amp_mode):
                pred = model(
                    points=window.points,
                    queries=norm_queries_dev,
                    query_type_ids=type_ids_dev,
                    padding_mask=window.padding_mask,
                )
            pred = pred.float()
            for batch_idx in range(pred.shape[0]):
                global_idx = window.global_indices[batch_idx]
                valid = global_idx >= 0
                all_pred[global_idx[valid]] = all_pred[global_idx[valid]] + pred[batch_idx, valid]
                pred_count[global_idx[valid]] = pred_count[global_idx[valid]] + 1.0

    pred_count = pred_count.clamp(min=1.0)
    return (all_pred / pred_count).detach().cpu()


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
        score_mode=str(getattr(model_config, "mlqds_score_mode", "rank")),
        score_temperature=float(getattr(model_config, "mlqds_score_temperature", 1.0)),
        rank_confidence_weight=float(getattr(model_config, "mlqds_rank_confidence_weight", 0.15)),
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
                "range_entry_exit_f1": float(range_audit["range_entry_exit_f1"]),
                "range_temporal_coverage": float(range_audit["range_temporal_coverage"]),
                "range_gap_coverage": float(range_audit["range_gap_coverage"]),
                "range_shape_score": float(range_audit["range_shape_score"]),
            }
        )
    variant = str(getattr(model_config, "checkpoint_f1_variant", "range_usefulness")).lower()
    if variant == "range_usefulness":
        if range_audit is None:
            return float(answer_agg), answer_pt, metrics
        score = float(range_audit["range_usefulness_score"])
        return score, {"range": score}, metrics
    if variant == "combined":
        return float(combined_agg), combined_pt, metrics
    return float(answer_agg), answer_pt, metrics


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
) -> tuple[float, dict[str, float]]:
    """Backward-compatible validation selector used by focused tests."""
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
    )
    return score, per_type


def _validation_uniform_f1(
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
    variant = str(getattr(model_config, "checkpoint_f1_variant", "range_usefulness")).lower()
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
) -> TrainingOutputs:
    """Train one pure-workload model with trajectory-window ranking losses."""
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    all_points = torch.cat(train_trajectories, dim=0)
    point_dim = 8 if model_config.model_type == "turn_aware" else 7
    points = all_points[:, :point_dim].float()

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
    residual_label_mode = str(getattr(model_config, "residual_label_mode", "none")).lower()
    if residual_label_mode not in {"none", "temporal"}:
        raise ValueError("residual_label_mode must be 'none' or 'temporal'.")
    loss_objective = str(getattr(model_config, "loss_objective", "budget_topk")).lower()
    if loss_objective not in {"ranking_bce", "budget_topk"}:
        raise ValueError("loss_objective must be 'ranking_bce' or 'budget_topk'.")
    configured_budget_ratios = _budget_loss_ratios(model_config)
    budget_ratios = configured_budget_ratios
    temporal_residual_budget_masks: tuple[tuple[float, float, torch.Tensor], ...] = ()
    temporal_residual_union_mask: torch.Tensor | None = None
    workload_type_id = _pure_query_type_id(workload.type_ids)
    if residual_label_mode == "temporal" and loss_objective == "budget_topk":
        budget_ratios = _effective_budget_loss_ratios(model_config, residual_label_mode)
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
    elif residual_label_mode == "temporal":
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
    base_type_weights = _workload_map_tensor(train_workload_map or _query_frequency_workload_map(workload), device=device)
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
        residual_label_mode=residual_label_mode,
        loss_objective=loss_objective,
        temporal_fraction=float(getattr(model_config, "mlqds_temporal_fraction", 0.50)),
    )
    if budget_ratios != configured_budget_ratios:
        print(
            f"  [{run_tag}] effective_budget_loss_ratios={list(budget_ratios)} "
            f"from configured={list(configured_budget_ratios)} "
            f"residual_label_mode={residual_label_mode} "
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
    single_windows = [
        TrajectoryBatch(
            points=w.points.to(device),
            padding_mask=w.padding_mask.to(device),
            trajectory_ids=w.trajectory_ids,
            global_indices=w.global_indices.to(device),
        )
        for w in windows_cpu
    ]
    train_batch_size = max(1, int(getattr(model_config, "train_batch_size", 1)))
    windows = batch_windows(single_windows, train_batch_size)
    trained_window_count = len(single_windows)
    # Diagnostic pass operates on single windows so we can cheaply subsample a
    # fraction of them (batched diagnostics would waste the remaining lanes).
    diag_windows = single_windows
    diag_every = max(1, int(getattr(model_config, "diagnostic_every", 1)))
    diag_fraction = float(getattr(model_config, "diagnostic_window_fraction", 1.0))
    diag_fraction = min(1.0, max(0.05, diag_fraction))
    selection_metric = str(getattr(model_config, "checkpoint_selection_metric", "f1")).lower()
    if selection_metric not in {"loss", "f1", "uniform_gap"}:
        raise ValueError("checkpoint_selection_metric must be 'loss', 'f1', or 'uniform_gap'.")
    f1_diag_every = int(getattr(model_config, "f1_diagnostic_every", 0) or 0)
    has_validation_f1 = (
        validation_trajectories is not None
        and validation_boundaries is not None
        and validation_workload is not None
        and validation_workload_map is not None
    )
    if selection_metric in {"f1", "uniform_gap"} and not has_validation_f1:
        print(
            f"  [{run_tag}] WARNING: checkpoint_selection_metric={selection_metric} requested without validation workload; "
            "falling back to loss selection.",
            flush=True,
        )
        selection_metric = "loss"
    if selection_metric in {"f1", "uniform_gap"} and f1_diag_every <= 0:
        f1_diag_every = diag_every
    validation_points_for_f1: torch.Tensor | None = None
    validation_query_cache: Any | None = None
    if has_validation_f1:
        from src.evaluation.evaluate_methods import EvaluationQueryCache

        assert validation_trajectories is not None
        assert validation_boundaries is not None
        assert validation_workload is not None
        validation_points_for_f1 = (
            validation_points
            if validation_points is not None
            else torch.cat(validation_trajectories, dim=0)
        )
        if precomputed_validation_query_cache is None:
            validation_query_cache = EvaluationQueryCache.for_workload(
                validation_points_for_f1,
                validation_boundaries,
                validation_workload.typed_queries,
            )
        else:
            precomputed_validation_query_cache.validate(
                validation_points_for_f1,
                validation_boundaries,
                validation_workload.typed_queries,
            )
            validation_query_cache = precomputed_validation_query_cache
    validation_uniform_result: tuple[float, dict[str, float]] | None = None
    if selection_metric == "uniform_gap" and has_validation_f1:
        assert validation_trajectories is not None
        assert validation_boundaries is not None
        assert validation_workload is not None
        validation_uniform_result = _validation_uniform_f1(
            trajectories=validation_trajectories,
            boundaries=validation_boundaries,
            workload=validation_workload,
            workload_map=validation_workload_map or {},
            model_config=model_config,
            validation_points=validation_points_for_f1,
            query_cache=validation_query_cache,
        )
        uniform_f1, uniform_per_type = validation_uniform_result
        print(
            f"  [{run_tag}] validation uniform_f1={uniform_f1:.6f}  "
            f"range={uniform_per_type.get('range', 0.0):.6f}  "
            f"knn={uniform_per_type.get('knn', 0.0):.6f}  "
            f"similarity={uniform_per_type.get('similarity', 0.0):.6f}  "
            f"clustering={uniform_per_type.get('clustering', 0.0):.6f}",
            flush=True,
        )

    g = torch.Generator().manual_seed(int(seed) + 99)
    # Separate fixed-seed generator for diagnostics so the tau subsample
    # stays consistent across epochs and doesn't oscillate with training state.
    eval_g = torch.Generator().manual_seed(int(seed) + 777)
    history: list[dict[str, float]] = []

    effective_epochs = max(1, int(model_config.epochs))
    patience = int(getattr(model_config, "early_stopping_patience", 0) or 0)
    smoothing_window = max(1, int(getattr(model_config, "checkpoint_smoothing_window", 1) or 1))
    checkpoint_full_f1_every = max(1, int(getattr(model_config, "checkpoint_full_f1_every", 1) or 1))
    checkpoint_candidate_pool_size = max(1, int(getattr(model_config, "checkpoint_candidate_pool_size", 1) or 1))
    checkpoint_candidates: list[_CheckpointCandidate] = []
    selection_history: list[float] = []
    best_selection = float("-inf")
    best_loss = float("inf")
    best_selection_score = 0.0
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] | None = None
    epochs_no_improve = 0
    epoch_w = len(str(effective_epochs))
    epochs_trained = 0
    for epoch in range(effective_epochs):
        epoch_t0 = time.perf_counter()
        epoch_timing = {
            "forward_s": 0.0,
            "loss_s": 0.0,
            "backward_s": 0.0,
            "diagnostic_s": 0.0,
            "f1_s": 0.0,
        }
        evaluated_checkpoint_candidates: list[_CheckpointCandidate] = []
        model.train()
        epoch_loss = torch.tensor(0.0, device=device)
        positive_windows = torch.zeros((NUM_QUERY_TYPES,), dtype=torch.long)
        skipped_zero_windows = prefiltered_zero_windows.clone()
        ranking_pair_counts = torch.zeros((NUM_QUERY_TYPES,), dtype=torch.long)

        for w in windows:
            forward_t0 = time.perf_counter()
            with torch_autocast_context(device, amp_mode):
                pred_batch = model(
                    points=w.points,
                    queries=norm_queries_dev,
                    query_type_ids=type_ids_dev,
                    padding_mask=w.padding_mask,
                )
            epoch_timing["forward_s"] += time.perf_counter() - forward_t0
            loss_t0 = time.perf_counter()
            pred_batch = pred_batch.float()
            # pred_batch: (B, L).  Accumulate per-window loss terms across
            # the batch and backprop once per batch — this is what makes the
            # GPU actually saturated compared to the old batch=1 loop.
            loss_terms: list[torch.Tensor] = []
            B = pred_batch.shape[0]
            batch_global_idx = w.global_indices.to(device=device)
            valid_batch = batch_global_idx >= 0
            safe_global_idx = batch_global_idx.clamp(min=0)
            batch_labels = training_target_dev[safe_global_idx]
            batch_label_mask = labelled_mask_dev[safe_global_idx] & valid_batch
            positive_row_mask = (batch_label_mask & (batch_labels > 0)).any(dim=1)
            positive_windows[active_type_id] += int(positive_row_mask.sum().item())
            skipped_zero_windows[active_type_id] += int((~positive_row_mask).sum().item())

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

                for b in torch.where(positive_row_mask.detach().cpu())[0].tolist():
                    row = int(b)
                    idx = batch_global_idx[row]
                    valid_window = idx >= 0
                    global_idx = idx[valid_window]
                    pred_valid = pred_batch[row][valid_window]
                    t_labels = training_target_dev[global_idx]
                    t_mask = labelled_mask_dev[global_idx]
                    pointwise_mask = t_mask
                    if temporal_residual_union_mask is not None:
                        pointwise_mask = t_mask & (~temporal_residual_union_mask[global_idx])
                    pointwise_term = _balanced_pointwise_loss(pred_valid, t_labels, pointwise_mask, generator=g)
                    loss_terms.append(rank_loss_rows[row] + model_config.pointwise_loss_weight * pointwise_term)
            else:
                for b in torch.where(positive_row_mask.detach().cpu())[0].tolist():
                    row = int(b)
                    idx = batch_global_idx[row]
                    valid_window = idx >= 0
                    global_idx = idx[valid_window]
                    pred_valid = pred_batch[row][valid_window]
                    t_labels = training_target_dev[global_idx]
                    t_mask = labelled_mask_dev[global_idx]
                    rank_loss, pair_count = _ranking_loss_for_type(
                        pred=pred_valid,
                        target=t_labels,
                        valid_mask=t_mask,
                        pairs_per_type=model_config.ranking_pairs_per_type,
                        top_quantile=model_config.ranking_top_quantile,
                        margin=model_config.rank_margin,
                        generator=g,
                    )
                    ranking_pair_counts[active_type_id] += int(pair_count)
                    pointwise_mask = t_mask
                    if temporal_residual_union_mask is not None:
                        pointwise_mask = t_mask & (~temporal_residual_union_mask[global_idx])
                    pointwise_term = _balanced_pointwise_loss(pred_valid, t_labels, pointwise_mask, generator=g)
                    loss_terms.append(rank_loss + model_config.pointwise_loss_weight * pointwise_term)
            epoch_timing["loss_s"] += time.perf_counter() - loss_t0

            if loss_terms:
                backward_t0 = time.perf_counter()
                loss = (
                    torch.stack(loss_terms).sum() / float(B)
                    + model_config.l2_score_weight * (pred_batch ** 2).mean()
                )
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
                k = max(8, int(len(diag_windows) * diag_fraction))
                perm = torch.randperm(len(diag_windows), generator=eval_g)[:k].tolist()
                sample_windows = [diag_windows[i] for i in perm]
            else:
                sample_windows = diag_windows

            model.eval()
            with torch.no_grad():
                all_pred = norm_points_dev.new_zeros((norm_points_dev.shape[0],))
                pred_count = norm_points_dev.new_zeros((norm_points_dev.shape[0],))
                for w in sample_windows:
                    with torch_autocast_context(device, amp_mode):
                        wp = model(
                            points=w.points,
                            queries=norm_queries_dev,
                            query_type_ids=type_ids_dev,
                            padding_mask=w.padding_mask,
                        )[0]
                    wp = wp.float()
                    widx = w.global_indices[0]
                    valid = widx >= 0
                    all_pred[widx[valid]] = all_pred[widx[valid]] + wp[valid]
                    pred_count[widx[valid]] = pred_count[widx[valid]] + 1.0
                covered_mask = pred_count > 0
                pred_count = pred_count.clamp(min=1.0)
                full_pred = all_pred / pred_count

            stats: dict[str, float] = {
                "epoch": float(epoch),
                "loss": float(epoch_loss.item() / max(1, len(windows))),
                "pred_std": (
                    float(full_pred[covered_mask].std().item())
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
                pt = full_pred
                stats[f"pred_p50_t{t}"] = float(_safe_quantile(pt, 0.50).item())
                stats[f"pred_p90_t{t}"] = float(_safe_quantile(pt, 0.90).item())
                stats[f"pred_p99_t{t}"] = float(_safe_quantile(pt, 0.99).item())
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
                    # Reset eval_g to the same state each epoch so the diagnostic
                    # subsample is identical across epochs, giving stable tau trends.
                    eval_g.manual_seed(int(seed) + 777)
                    p_sample, y_sample = _discriminative_sample(
                        pt[eval_mask].detach().cpu(),
                        training_target_dev[eval_mask].detach().cpu(),
                        n_each=100,
                        generator=eval_g,
                    )
                    stats[f"kendall_tau_t{t}"] = _kendall_tau(p_sample, y_sample)
                else:
                    stats[f"kendall_tau_t{t}"] = 0.0

            if stats["pred_std"] < 1e-3:
                stats["collapse_warning"] = 1.0
            epoch_timing["diagnostic_s"] += time.perf_counter() - diagnostic_t0

            candidate_tau_vals = [stats[f"kendall_tau_t{t}"] for t in active_type_ids]
            candidate_avg_tau = sum(candidate_tau_vals) / max(1, len(candidate_tau_vals))
            f1_due = f1_diag_every <= 0 or ((epoch + 1) % f1_diag_every == 0 or is_last_epoch or epoch == 0)
            full_f1_due = f1_due and (
                checkpoint_full_f1_every <= 1
                or (epoch + 1) % checkpoint_full_f1_every == 0
                or is_last_epoch
                or epoch == 0
            )
            use_checkpoint_candidate_pool = (
                has_validation_f1
                and f1_due
                and selection_metric in {"f1", "uniform_gap"}
                and checkpoint_full_f1_every > 1
            )
            should_run_f1 = has_validation_f1 and full_f1_due and (
                selection_metric in {"f1", "uniform_gap"} or f1_diag_every > 0
            ) and not use_checkpoint_candidate_pool
            if should_run_f1:
                f1_t0 = time.perf_counter()
                assert validation_trajectories is not None
                assert validation_boundaries is not None
                assert validation_workload is not None
                validation_score, per_type_f1, validation_metrics = _validation_checkpoint_scores(
                    model=model,
                    scaler=scaler,
                    trajectories=validation_trajectories,
                    boundaries=validation_boundaries,
                    workload=validation_workload,
                    workload_map=validation_workload_map or {},
                    model_config=model_config,
                    device=device,
                    validation_points=validation_points_for_f1,
                    query_cache=validation_query_cache,
                )
                epoch_timing["f1_s"] += time.perf_counter() - f1_t0
                _record_validation_stats(
                    stats,
                    validation_score=validation_score,
                    per_type_f1=per_type_f1,
                    validation_metrics=validation_metrics,
                    validation_uniform_result=validation_uniform_result,
                    validation_workload_map=validation_workload_map,
                )
            if has_validation_f1 and f1_due and selection_metric in {"f1", "uniform_gap"}:
                stats["checkpoint_f1_candidate"] = 1.0
                stats["checkpoint_candidate_cheap_score"] = _selection_score(
                    candidate_avg_tau,
                    stats["pred_std"],
                    stats["loss"],
                )
                stats["checkpoint_full_f1_due"] = 1.0 if full_f1_due else 0.0
                if use_checkpoint_candidate_pool:
                    checkpoint_candidates.append(
                        _CheckpointCandidate(
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
                    if full_f1_due and checkpoint_candidates:
                        f1_t0 = time.perf_counter()
                        assert validation_trajectories is not None
                        assert validation_boundaries is not None
                        assert validation_workload is not None
                        current_state_dict = _model_state_on_cpu(model)
                        for candidate in sorted(checkpoint_candidates, key=lambda item: item.epoch_number):
                            candidate_t0 = time.perf_counter()
                            model.load_state_dict(candidate.state_dict)
                            validation_score, per_type_f1, validation_metrics = _validation_checkpoint_scores(
                                model=model,
                                scaler=scaler,
                                trajectories=validation_trajectories,
                                boundaries=validation_boundaries,
                                workload=validation_workload,
                                workload_map=validation_workload_map or {},
                                model_config=model_config,
                                device=device,
                                validation_points=validation_points_for_f1,
                                query_cache=validation_query_cache,
                            )
                            _record_validation_stats(
                                candidate.stats,
                                validation_score=validation_score,
                                per_type_f1=per_type_f1,
                                validation_metrics=validation_metrics,
                                validation_uniform_result=validation_uniform_result,
                                validation_workload_map=validation_workload_map,
                            )
                            candidate.stats["checkpoint_candidate_evaluated"] = 1.0
                            candidate.stats["checkpoint_full_f1_round_epoch"] = float(epoch + 1)
                            candidate.stats["checkpoint_validation_seconds"] = float(time.perf_counter() - candidate_t0)
                            evaluated_checkpoint_candidates.append(candidate)
                        model.load_state_dict(current_state_dict)
                        epoch_timing["f1_s"] += time.perf_counter() - f1_t0
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
        stats["epoch_f1_seconds"] = float(epoch_timing["f1_s"])
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
                    candidate_selection = _selection_from_stats(
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
                            f"{candidate.epoch_number:0{epoch_w}d}/{effective_epochs}  "
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
                selection = _selection_from_stats(
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
                    # epoch-to-epoch validation F1 noise so we don't lock onto a lucky
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
                f"  [{run_tag}] epoch {epoch + 1:0{epoch_w}d}/{effective_epochs}  "
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
            if "val_uniform_f1" in stats:
                print(
                    f"    [{run_tag}] val_vs_uniform aggregate={stats['val_query_uniform_gap']:+.6f}  "
                    f"type_deficit={stats['val_query_type_deficit']:.6f}  "
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
                f"f1={stats['epoch_f1_seconds']:.2f}s  "
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
                            f"  [{run_tag}] early stopping at epoch {epoch + 1:0{epoch_w}d}: "
                            f"selection score did not improve over {patience} diag epochs "
                            f"(best_selection={best_selection:+.3f}, best_loss={best_loss:.8f})",
                            flush=True,
                        )
                        break
        else:
            # Non-diagnostic epoch: log loss only, no tau / early-stopping update.
            print(
                f"  [{run_tag}] epoch {epoch + 1:0{epoch_w}d}/{effective_epochs}  "
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
