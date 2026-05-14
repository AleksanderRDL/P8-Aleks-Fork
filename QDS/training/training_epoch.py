"""One-epoch optimization helpers for trajectory-window training."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

import torch

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
            pred_batch = model(
                points=window_batch.points,
                queries=norm_queries_dev,
                query_type_ids=type_ids_dev,
                padding_mask=window_batch.padding_mask,
            )
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
