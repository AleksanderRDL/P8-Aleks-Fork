"""One-epoch optimization helpers for trajectory-window training."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol, cast

import torch
import torch.nn.functional as F

from experiments.experiment_config import ModelConfig
from experiments.torch_runtime import torch_autocast_context
from queries.query_types import NUM_QUERY_TYPES
from training.training_losses import (
    _balanced_pointwise_loss_rows,
    _budget_stratified_recall_loss_rows,
    _budget_temporal_cdf_loss_rows,
    _budget_topk_recall_loss_rows,
    _budget_topk_temporal_residual_loss_rows,
    _pointwise_bce_loss_rows,
    _ranking_loss_for_type,
)
from training.training_windows import _trajectory_batch_to_device
from training.trajectory_batching import TrajectoryBatch


class _GradScalerLike(Protocol):
    """Minimal GradScaler surface used by the epoch optimizer."""

    def is_enabled(self) -> bool: ...

    def scale(self, outputs: torch.Tensor) -> Any: ...

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None: ...

    def step(self, optimizer: torch.optim.Optimizer) -> Any: ...

    def update(self, new_scale: float | torch.Tensor | None = None) -> None: ...


@dataclass
class TrainingEpochResult:
    """Aggregated optimization result for one training epoch."""

    loss: torch.Tensor
    positive_windows: torch.Tensor
    skipped_zero_windows: torch.Tensor
    ranking_pair_counts: torch.Tensor
    timing: dict[str, float]


def _factorized_query_useful_loss(
    *,
    head_logits: torch.Tensor,
    head_targets: torch.Tensor,
    head_mask: torch.Tensor,
    global_indices: torch.Tensor | None = None,
    segment_ids: torch.Tensor | None = None,
    segment_size: int = 32,
) -> torch.Tensor:
    """Return auxiliary multi-head QueryUsefulV1 loss for v2 models."""
    if head_logits.shape != head_targets.shape or head_mask.shape != head_logits.shape:
        raise ValueError("factorized head logits, targets, and mask must have matching shape.")
    valid = head_mask.to(dtype=torch.bool)
    if not bool(valid.any().item()):
        return head_logits.new_tensor(0.0)
    # q_hit, boundary, and segment budget are probability-like.  Behavior and
    # replacement are smooth utilities but BCE keeps useful gradients near 0/1.
    per_element = F.binary_cross_entropy_with_logits(
        head_logits,
        head_targets.clamp(0.0, 1.0),
        reduction="none",
    )
    head_weights = head_logits.new_tensor([0.30, 0.25, 0.15, 0.20, 0.10]).view(1, 1, -1)
    weighted = per_element * head_weights * valid.to(dtype=per_element.dtype)
    denom = (head_weights * valid.to(dtype=per_element.dtype)).sum().clamp(min=1.0)
    point_loss = weighted.sum() / denom
    segment_loss = _segment_budget_head_segment_level_loss(
        head_logits=head_logits,
        head_targets=head_targets,
        head_mask=head_mask,
        global_indices=global_indices,
        segment_ids=segment_ids,
        segment_size=segment_size,
    )
    return point_loss + 0.25 * segment_loss


def _segment_budget_head_segment_level_loss(
    *,
    head_logits: torch.Tensor,
    head_targets: torch.Tensor,
    head_mask: torch.Tensor,
    global_indices: torch.Tensor | None = None,
    segment_ids: torch.Tensor | None = None,
    segment_size: int = 32,
) -> torch.Tensor:
    """Return segment-level/listwise loss for the segment-budget head."""
    if head_logits.shape != head_targets.shape or head_mask.shape != head_logits.shape:
        raise ValueError("factorized head logits, targets, and mask must have matching shape.")
    if int(head_logits.shape[-1]) <= 4:
        return head_logits.new_tensor(0.0)
    size = max(1, int(segment_size))
    seg_logits = head_logits[..., 4].float()
    seg_targets = head_targets[..., 4].float().clamp(0.0, 1.0)
    seg_mask = head_mask[..., 4].to(dtype=torch.bool)
    if global_indices is not None:
        seg_mask = seg_mask & (global_indices >= 0)
    if segment_ids is not None and segment_ids.shape[:2] != seg_mask.shape:
        raise ValueError("segment_ids must match factorized head batch dimensions.")
    losses: list[torch.Tensor] = []
    for row in range(int(seg_logits.shape[0])):
        valid_positions = torch.where(seg_mask[row])[0]
        if int(valid_positions.numel()) <= 0:
            continue
        row_segment_logits: list[torch.Tensor] = []
        row_segment_targets: list[torch.Tensor] = []
        if segment_ids is not None:
            row_segment_ids = segment_ids[row].to(device=seg_logits.device, dtype=torch.long)
            valid_positions = valid_positions[row_segment_ids[valid_positions] >= 0]
            if int(valid_positions.numel()) <= 0:
                continue
            unique_segment_ids = torch.unique(row_segment_ids[valid_positions], sorted=True)
            for segment_id in unique_segment_ids.tolist():
                positions = valid_positions[row_segment_ids[valid_positions] == int(segment_id)]
                if int(positions.numel()) <= 0:
                    continue
                row_segment_logits.append(seg_logits[row, positions].mean())
                row_segment_targets.append(seg_targets[row, positions].mean())
        else:
            for start in range(0, int(valid_positions.numel()), size):
                positions = valid_positions[start : start + size]
                if int(positions.numel()) <= 0:
                    continue
                row_segment_logits.append(seg_logits[row, positions].mean())
                row_segment_targets.append(seg_targets[row, positions].mean())
        if not row_segment_logits:
            continue
        pooled_logits = torch.stack(row_segment_logits)
        pooled_targets = torch.stack(row_segment_targets).clamp(0.0, 1.0)
        losses.append(F.binary_cross_entropy_with_logits(pooled_logits, pooled_targets, reduction="mean"))
        if int(pooled_logits.numel()) >= 2:
            target_diff = pooled_targets.unsqueeze(1) - pooled_targets.unsqueeze(0)
            logit_diff = pooled_logits.unsqueeze(1) - pooled_logits.unsqueeze(0)
            pair_mask = target_diff.abs() > 0.05
            if bool(pair_mask.any().item()):
                direction = torch.sign(target_diff[pair_mask])
                losses.append(F.softplus(-direction * logit_diff[pair_mask]).mean())
    if not losses:
        return head_logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def _train_one_epoch(
    *,
    model: torch.nn.Module,
    windows: list[TrajectoryBatch],
    opt: torch.optim.Optimizer,
    grad_scaler: _GradScalerLike,
    model_config: ModelConfig,
    device: torch.device,
    amp_mode: str,
    norm_queries_dev: torch.Tensor,
    type_ids_dev: torch.Tensor,
    training_target_dev: torch.Tensor,
    labelled_mask_dev: torch.Tensor,
    prefiltered_zero_windows: torch.Tensor,
    active_type_id: int,
    loss_objective: str,
    budget_ratios: tuple[float, ...],
    budget_loss_temperature: float,
    temporal_residual_budget_masks: tuple[tuple[float, float, torch.Tensor], ...],
    temporal_residual_union_mask: torch.Tensor | None,
    training_sample_generator: torch.Generator,
    factorized_targets_dev: torch.Tensor | None = None,
    factorized_mask_dev: torch.Tensor | None = None,
    canonical_segment_ids_dev: torch.Tensor | None = None,
) -> TrainingEpochResult:
    """Run forward/loss/backward optimization over all training windows."""
    timing = {
        "forward_s": 0.0,
        "loss_s": 0.0,
        "backward_s": 0.0,
    }
    model.train()
    epoch_loss = torch.tensor(0.0, device=device)
    positive_windows = torch.zeros((NUM_QUERY_TYPES,), dtype=torch.long)
    skipped_zero_windows = prefiltered_zero_windows.clone()
    ranking_pair_counts = torch.zeros((NUM_QUERY_TYPES,), dtype=torch.long)

    for window_batch_cpu in windows:
        window_batch = _trajectory_batch_to_device(window_batch_cpu, device)
        forward_t0 = time.perf_counter()
        with torch_autocast_context(device, amp_mode):
            forward_with_heads = getattr(model, "forward_with_heads", None)
            if factorized_targets_dev is not None and callable(forward_with_heads):
                forward_with_heads_fn = cast(
                    Callable[..., tuple[torch.Tensor, torch.Tensor]], forward_with_heads
                )
                pred_batch, head_logits_batch = forward_with_heads_fn(
                    window_batch.points,
                    padding_mask=window_batch.padding_mask,
                )
            else:
                pred_batch = model(
                    points=window_batch.points,
                    queries=norm_queries_dev,
                    query_type_ids=type_ids_dev,
                    padding_mask=window_batch.padding_mask,
                )
                head_logits_batch = None
        timing["forward_s"] += time.perf_counter() - forward_t0
        loss_t0 = time.perf_counter()
        pred_batch = pred_batch.float()
        loss: torch.Tensor | None = None
        batch_size = pred_batch.shape[0]
        batch_global_idx = window_batch.global_indices.to(device=device)
        valid_batch = batch_global_idx >= 0
        safe_global_idx = batch_global_idx.clamp(min=0)
        batch_labels = training_target_dev[safe_global_idx]
        batch_label_mask = labelled_mask_dev[safe_global_idx] & valid_batch
        aux_loss = pred_batch.new_tensor(0.0)
        if (
            head_logits_batch is not None
            and factorized_targets_dev is not None
            and factorized_mask_dev is not None
        ):
            batch_head_targets = factorized_targets_dev[safe_global_idx].float()
            batch_head_mask = factorized_mask_dev[safe_global_idx] & valid_batch.unsqueeze(-1)
            batch_segment_ids = None
            if canonical_segment_ids_dev is not None:
                batch_segment_ids = canonical_segment_ids_dev[safe_global_idx].to(device=device, dtype=torch.long)
                batch_segment_ids = torch.where(
                    valid_batch,
                    batch_segment_ids,
                    torch.full_like(batch_segment_ids, -1),
                )
            aux_loss = _factorized_query_useful_loss(
                head_logits=head_logits_batch.float(),
                head_targets=batch_head_targets,
                head_mask=batch_head_mask,
                global_indices=batch_global_idx,
                segment_ids=batch_segment_ids,
            )
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

        if loss_objective in {"budget_topk", "stratified_budget_topk"}:
            if temporal_residual_budget_masks:
                rank_loss_rows, _rank_active_rows = _budget_topk_temporal_residual_loss_rows(
                    pred=pred_batch,
                    target=batch_labels,
                    valid_mask=batch_label_mask,
                    global_idx=safe_global_idx,
                    temporal_base_masks=temporal_residual_budget_masks,
                    temperature=budget_loss_temperature,
                )
            elif loss_objective == "stratified_budget_topk":
                rank_loss_rows, _rank_active_rows = _budget_stratified_recall_loss_rows(
                    pred=pred_batch,
                    target=batch_labels,
                    valid_mask=batch_label_mask,
                    budget_ratios=budget_ratios,
                    temperature=budget_loss_temperature,
                    center_weight=float(getattr(model_config, "mlqds_stratified_center_weight", 0.0)),
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
                temporal_distribution_weight = float(
                    getattr(model_config, "temporal_distribution_loss_weight", 0.0) or 0.0
                )
                if temporal_distribution_weight > 0.0:
                    temporal_distribution_rows, _distribution_active_rows = _budget_temporal_cdf_loss_rows(
                        pred=pred_batch,
                        valid_mask=batch_label_mask,
                        budget_ratios=budget_ratios,
                        temperature=budget_loss_temperature,
                    )
                    row_losses = row_losses + temporal_distribution_weight * temporal_distribution_rows
                loss = (
                    row_losses[positive_row_mask].sum() / float(batch_size)
                    + 0.50 * aux_loss
                    + model_config.l2_score_weight * (pred_batch ** 2).mean()
                )
        elif loss_objective == "pointwise_bce":
            pointwise_direct_rows, pointwise_direct_active_rows = _pointwise_bce_loss_rows(
                pred=pred_batch,
                target=batch_labels,
                valid_mask=batch_label_mask,
            )
            if bool(pointwise_direct_active_rows.any().item()):
                loss = (
                    pointwise_direct_rows[pointwise_direct_active_rows].sum() / float(batch_size)
                    + 0.50 * aux_loss
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
                    + 0.50 * aux_loss
                    + model_config.l2_score_weight * (pred_batch ** 2).mean()
                )
        timing["loss_s"] += time.perf_counter() - loss_t0

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
            timing["backward_s"] += time.perf_counter() - backward_t0

    return TrainingEpochResult(
        loss=epoch_loss,
        positive_windows=positive_windows,
        skipped_zero_windows=skipped_zero_windows,
        ranking_pair_counts=ranking_pair_counts,
        timing=timing,
    )
