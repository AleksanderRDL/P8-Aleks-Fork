"""Training loss helpers for trajectory ranking models."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from experiments.experiment_config import ModelConfig
from simplification.simplify_trajectories import evenly_spaced_indices

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


def _pointwise_bce_loss_rows(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unbalanced pointwise BCE for every valid supervised point."""
    if pred.ndim != 2 or target.shape != pred.shape or valid_mask.shape != pred.shape:
        raise ValueError("pred, target, and valid_mask must have matching shape [batch, window_length].")

    per_element = F.binary_cross_entropy_with_logits(
        pred,
        target.clamp(0.0, 1.0),
        reduction="none",
    )
    valid_float = valid_mask.to(dtype=per_element.dtype)
    active_rows = valid_mask.any(dim=1)
    denom = valid_float.sum(dim=1).clamp(min=1.0)
    row_loss = (per_element * valid_float).sum(dim=1) / denom
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


def _budget_stratified_recall_loss_rows(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    budget_ratios: tuple[float, ...],
    temperature: float,
    center_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return loss matching the stratified retained-mask selector.

    The stratified selector always keeps endpoints, then picks one learned-score
    point inside each interior trajectory-order stratum. A global soft top-k
    loss trains the wrong decision surface for that selector, so this loss
    optimizes soft target-mass capture within each stratum independently.
    """
    if pred.ndim != 2 or target.shape != pred.shape or valid_mask.shape != pred.shape:
        raise ValueError("pred, target, and valid_mask must have matching shape [batch, window_length].")

    batch_size, window_length = pred.shape
    nonnegative_target = target.clamp(min=0.0)
    row_loss_sum = pred.new_zeros((batch_size,))
    row_loss_count = torch.zeros((batch_size,), dtype=torch.long, device=pred.device)
    softmax_temperature = max(float(temperature), 1e-4)
    center_penalty = max(0.0, float(center_weight))

    for row in range(batch_size):
        valid_idx = torch.where(valid_mask[row])[0]
        valid_count = int(valid_idx.numel())
        if valid_count < 3:
            continue
        row_targets = nonnegative_target[row, valid_idx]
        if not bool((row_targets > 0.0).any().item()):
            continue

        for raw_ratio in budget_ratios:
            ratio = min(1.0, max(0.0, float(raw_ratio)))
            if ratio <= 0.0:
                continue
            keep_count = min(valid_count, max(2, int(math.ceil(ratio * valid_count))))
            interior_count = valid_count - 2
            interior_slots = keep_count - 2
            if interior_slots <= 0 or interior_slots >= interior_count:
                continue

            ratio_loss_sum = pred.new_tensor(0.0)
            ratio_loss_count = 0
            for slot in range(interior_slots):
                left = 1 + int(math.floor(slot * interior_count / interior_slots))
                right = 1 + int(math.floor((slot + 1) * interior_count / interior_slots))
                if right <= left:
                    continue
                candidate_idx = valid_idx[left:right]
                candidate_targets = nonnegative_target[row, candidate_idx]
                ideal_mass = candidate_targets.max().detach()
                if float(ideal_mass.item()) <= 1e-12:
                    continue

                candidate_scores = pred[row, candidate_idx]
                if center_penalty > 0.0 and int(candidate_idx.numel()) > 1:
                    local_positions = torch.arange(left, right, dtype=pred.dtype, device=pred.device)
                    center = 0.5 * float(left + right - 1)
                    denom = max(1.0, 0.5 * float(right - left))
                    center_distance = torch.abs(local_positions - center) / denom
                    candidate_scores = candidate_scores - center_penalty * center_distance
                soft_choice = torch.softmax(candidate_scores / softmax_temperature, dim=0)
                captured_mass = (soft_choice * candidate_targets).sum()
                recall = (captured_mass / ideal_mass.clamp(min=1e-6)).clamp(0.0, 1.0)
                ratio_loss_sum = ratio_loss_sum + (1.0 - recall)
                ratio_loss_count += 1

            if ratio_loss_count > 0:
                row_loss_sum[row] = row_loss_sum[row] + ratio_loss_sum / float(ratio_loss_count)
                row_loss_count[row] += 1

    active_rows = row_loss_count > 0
    row_loss = row_loss_sum / row_loss_count.clamp(min=1).float()
    return row_loss, active_rows


def _budget_temporal_cdf_loss_rows(
    pred: torch.Tensor,
    valid_mask: torch.Tensor,
    budget_ratios: tuple[float, ...],
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-row soft top-k temporal distribution loss.

    This regularizer discourages budgeted soft-retention mass from collapsing
    into one local cluster. It compares the cumulative soft-keep mass over the
    valid trajectory order to a uniform temporal CDF.
    """
    if pred.ndim != 2 or valid_mask.shape != pred.shape:
        raise ValueError("pred and valid_mask must have matching shape [batch, window_length].")

    batch_size, window_length = pred.shape
    valid_counts = valid_mask.sum(dim=1)
    active_base = valid_counts >= 3
    if not bool(active_base.any().item()):
        return pred.new_zeros((batch_size,)), active_base

    score_sortable = pred.masked_fill(~valid_mask, float("-inf"))
    sorted_score = torch.sort(score_sortable, dim=1, descending=True).values
    valid_float = valid_mask.to(dtype=pred.dtype)
    valid_rank = valid_float.cumsum(dim=1)
    target_cdf = valid_rank / valid_counts.clamp(min=1).to(dtype=pred.dtype).unsqueeze(1)
    soft_keep_temperature = max(float(temperature), 1e-4)

    row_loss_sum = pred.new_zeros((batch_size,))
    row_loss_count = torch.zeros((batch_size,), dtype=torch.long, device=pred.device)
    for raw_ratio in budget_ratios:
        ratio = min(1.0, max(0.0, float(raw_ratio)))
        if ratio <= 0.0:
            continue
        keep_count = torch.ceil(valid_counts.float() * ratio).to(dtype=torch.long).clamp(min=1, max=window_length)
        active = active_base & (keep_count < valid_counts)
        if not bool(active.any().item()):
            continue

        threshold = sorted_score.gather(1, (keep_count - 1).unsqueeze(1)).squeeze(1).detach()
        soft_keep = torch.sigmoid((pred - threshold.unsqueeze(1)) / soft_keep_temperature) * valid_float
        soft_keep = soft_keep * (keep_count.float() / soft_keep.sum(dim=1).clamp(min=1e-6)).unsqueeze(1)
        soft_keep = soft_keep.clamp(max=1.0)
        keep_cdf = soft_keep.cumsum(dim=1) / keep_count.float().clamp(min=1.0).unsqueeze(1)
        ratio_loss = (((keep_cdf - target_cdf) ** 2) * valid_float).sum(dim=1) / valid_counts.clamp(min=1).float()
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


def _effective_temporal_residual_label_mode(
    model_config: ModelConfig,
    temporal_residual_label_mode: str,
) -> str:
    """Return the temporal-residual mode that matches the final selector.

    Stratified and global-budget selection have no reserved temporal base.
    Treating them as if they did makes training optimize a residual candidate
    set that inference never constructs.
    """
    mode = str(temporal_residual_label_mode).lower()
    if mode != "temporal":
        return mode
    hybrid_mode = str(getattr(model_config, "mlqds_hybrid_mode", "fill")).lower()
    if hybrid_mode in {"stratified", "global_budget"}:
        return "none"
    return "temporal"


def _effective_budget_loss_ratios(model_config: ModelConfig, temporal_residual_label_mode: str) -> tuple[float, ...]:
    """Return retained-budget ratios in the candidate set the model actually controls."""
    ratios = _budget_loss_ratios(model_config)
    if _effective_temporal_residual_label_mode(model_config, temporal_residual_label_mode) != "temporal":
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


def _ranking_loss_for_type(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    pairs_per_type: int,
    top_quantile: float,
    margin: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    """Compute top-boundary-focused pairwise ranking loss for one type. See training/README.md for details."""
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
