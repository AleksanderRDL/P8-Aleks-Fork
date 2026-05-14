"""Training target and rank-correlation diagnostics."""

from __future__ import annotations

from typing import Any

import torch

from simplification.mlqds_scoring import pure_workload_scores, simplify_mlqds_predictions
from simplification.simplify_trajectories import evenly_spaced_indices
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


def _fit_budget_ratios(model_config: object) -> tuple[float, ...]:
    """Return budget ratios for train-target fit diagnostics."""
    raw: list[float] = []
    raw.extend(float(value) for value in (getattr(model_config, "budget_loss_ratios", None) or []))
    raw.extend(float(value) for value in (getattr(model_config, "range_audit_compression_ratios", None) or []))
    raw.append(float(getattr(model_config, "compression_ratio", 0.05)))
    ratios = sorted({float(value) for value in raw if 0.0 < float(value) <= 1.0})
    return tuple(ratios) if ratios else (0.05,)


def _uniform_mask_for_ratio(
    point_count: int,
    boundaries: list[tuple[int, int]],
    ratio: float,
    device: torch.device,
) -> torch.Tensor:
    """Return the temporal-uniform retained mask for one ratio."""
    retained = torch.zeros((point_count,), dtype=torch.bool, device=device)
    for start, end in boundaries:
        local_count = int(end - start)
        if local_count <= 0:
            continue
        keep_count = min(local_count, max(2, int(torch.ceil(torch.tensor(float(ratio) * local_count)).item())))
        local_indices = evenly_spaced_indices(local_count, keep_count, device)
        retained[int(start) + local_indices] = True
    return retained


def _ideal_target_mask_for_ratio(
    target: torch.Tensor,
    labelled_mask: torch.Tensor,
    boundaries: list[tuple[int, int]],
    ratio: float,
) -> torch.Tensor:
    """Return a per-trajectory top-target mask used as a target-mass upper bound."""
    retained = torch.zeros_like(target, dtype=torch.bool)
    sortable = target.float().clamp(min=0.0).masked_fill(~labelled_mask.bool(), float("-inf"))
    for start, end in boundaries:
        local_count = int(end - start)
        if local_count <= 0:
            continue
        keep_count = min(local_count, max(2, int(torch.ceil(torch.tensor(float(ratio) * local_count)).item())))
        local_scores = sortable[int(start) : int(end)]
        finite = torch.isfinite(local_scores)
        if not bool(finite.any().item()):
            continue
        keep_count = min(keep_count, int(finite.sum().item()))
        if keep_count <= 0:
            continue
        local_indices = torch.topk(local_scores, k=keep_count, largest=True).indices
        retained[int(start) + local_indices] = True
    return retained


def _target_mass(target: torch.Tensor, labelled_mask: torch.Tensor, retained_mask: torch.Tensor) -> float:
    """Return positive target mass captured by a retained mask."""
    active = labelled_mask.bool() & retained_mask.bool()
    if not bool(active.any().item()):
        return 0.0
    return float(target.float().clamp(min=0.0)[active].sum().item())


def train_target_fit_diagnostics(
    *,
    predictions: torch.Tensor,
    target: torch.Tensor,
    labelled_mask: torch.Tensor,
    boundaries: list[tuple[int, int]],
    model_config: object,
    workload_type: str,
    seed: int,
) -> dict[str, Any]:
    """Summarize how well selected model scores fit the training target.

    This diagnostic is training-data only. It does not score eval queries and
    does not affect checkpoint selection.
    """
    if predictions.ndim != 1 or target.ndim != 1 or labelled_mask.ndim != 1:
        raise ValueError("predictions, target, and labelled_mask must be 1-D tensors.")
    if predictions.shape[0] != target.shape[0] or labelled_mask.shape[0] != target.shape[0]:
        raise ValueError("predictions, target, and labelled_mask must have matching lengths.")

    pred_cpu = predictions.detach().float().cpu()
    target_cpu = target.detach().float().cpu().clamp(min=0.0)
    labelled_cpu = labelled_mask.detach().bool().cpu()
    point_count = int(target_cpu.shape[0])
    positive = labelled_cpu & (target_cpu > 0.0)
    positive_count = int(positive.sum().item())
    target_mass_total = float(target_cpu[positive].sum().item()) if positive_count > 0 else 0.0

    generator = torch.Generator().manual_seed(int(seed) + 2029)
    if bool(labelled_cpu.any().item()):
        pred_sample, target_sample = _discriminative_sample(
            pred_cpu[labelled_cpu],
            target_cpu[labelled_cpu],
            n_each=200,
            generator=generator,
        )
        tau = _kendall_tau(pred_sample, target_sample)
    else:
        tau = 0.0

    selector_scores = pure_workload_scores(
        pred_cpu,
        boundaries,
        workload_type,
        score_mode=str(getattr(model_config, "mlqds_score_mode", "rank")),
        score_temperature=float(getattr(model_config, "mlqds_score_temperature", 1.0)),
        rank_confidence_weight=float(getattr(model_config, "mlqds_rank_confidence_weight", 0.15)),
    ).float()

    rows: list[dict[str, Any]] = []
    for ratio in _fit_budget_ratios(model_config):
        mlqds_mask = simplify_mlqds_predictions(
            pred_cpu,
            boundaries,
            workload_type,
            compression_ratio=float(ratio),
            temporal_fraction=float(getattr(model_config, "mlqds_temporal_fraction", 0.50)),
            diversity_bonus=float(getattr(model_config, "mlqds_diversity_bonus", 0.0)),
            hybrid_mode=str(getattr(model_config, "mlqds_hybrid_mode", "fill")),
            score_mode=str(getattr(model_config, "mlqds_score_mode", "rank")),
            score_temperature=float(getattr(model_config, "mlqds_score_temperature", 1.0)),
            rank_confidence_weight=float(getattr(model_config, "mlqds_rank_confidence_weight", 0.15)),
            range_geometry_scores=None,
            range_geometry_blend=0.0,
            stratified_center_weight=float(getattr(model_config, "mlqds_stratified_center_weight", 0.0)),
            min_learned_swaps=int(getattr(model_config, "mlqds_min_learned_swaps", 0)),
        ).cpu()
        uniform_mask = _uniform_mask_for_ratio(point_count, boundaries, float(ratio), target_cpu.device)
        ideal_mask = _ideal_target_mask_for_ratio(target_cpu, labelled_cpu, boundaries, float(ratio))
        mlqds_mass = _target_mass(target_cpu, labelled_cpu, mlqds_mask)
        uniform_mass = _target_mass(target_cpu, labelled_cpu, uniform_mask)
        ideal_mass = _target_mass(target_cpu, labelled_cpu, ideal_mask)
        mlqds_positive_retained = int((mlqds_mask & positive).sum().item())
        uniform_positive_retained = int((uniform_mask & positive).sum().item())
        ideal_positive_retained = int((ideal_mask & positive).sum().item())
        row = {
            "compression_ratio": float(ratio),
            "mlqds_target_mass": mlqds_mass,
            "uniform_target_mass": uniform_mass,
            "ideal_target_mass": ideal_mass,
            "mlqds_vs_uniform_target_mass": mlqds_mass - uniform_mass,
            "mlqds_target_recall": float(mlqds_mass / max(ideal_mass, 1e-12)),
            "uniform_target_recall": float(uniform_mass / max(ideal_mass, 1e-12)),
            "mlqds_vs_uniform_target_recall": float((mlqds_mass - uniform_mass) / max(ideal_mass, 1e-12)),
            "mlqds_positive_retained": mlqds_positive_retained,
            "uniform_positive_retained": uniform_positive_retained,
            "ideal_positive_retained": ideal_positive_retained,
            "mlqds_vs_uniform_positive_retained": mlqds_positive_retained - uniform_positive_retained,
            "ideal_overlap_fraction": float(
                int((mlqds_mask & ideal_mask).sum().item()) / max(1, int(ideal_mask.sum().item()))
            ),
            "mlqds_retained_count": int(mlqds_mask.sum().item()),
            "uniform_retained_count": int(uniform_mask.sum().item()),
            "ideal_retained_count": int(ideal_mask.sum().item()),
        }
        rows.append(row)

    matched_ratio = float(getattr(model_config, "compression_ratio", 0.05))
    matched_row = min(rows, key=lambda row: abs(float(row["compression_ratio"]) - matched_ratio)) if rows else {}
    low_rows = [row for row in rows if float(row["compression_ratio"]) <= 0.05 + 1e-12]
    return {
        "enabled": True,
        "workload_type": str(workload_type),
        "target_basis": "scaled_training_target_for_loss",
        "point_count": point_count,
        "labelled_point_count": int(labelled_cpu.sum().item()),
        "positive_label_count": positive_count,
        "positive_label_fraction": float(positive_count / max(1, int(labelled_cpu.sum().item()))),
        "positive_label_mass": target_mass_total,
        "prediction_std": float(pred_cpu.std(unbiased=False).item()) if point_count > 0 else 0.0,
        "selector_score_std": float(selector_scores.std(unbiased=False).item()) if point_count > 0 else 0.0,
        "score_target_kendall_tau": float(tau),
        "matched_compression_ratio": float(matched_row.get("compression_ratio", matched_ratio)),
        "matched_mlqds_target_recall": matched_row.get("mlqds_target_recall"),
        "matched_uniform_target_recall": matched_row.get("uniform_target_recall"),
        "matched_mlqds_vs_uniform_target_recall": matched_row.get("mlqds_vs_uniform_target_recall"),
        "low_budget_mean_mlqds_vs_uniform_target_recall": (
            float(sum(float(row["mlqds_vs_uniform_target_recall"]) for row in low_rows) / len(low_rows))
            if low_rows
            else None
        ),
        "budget_rows": rows,
    }
