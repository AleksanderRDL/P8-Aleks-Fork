"""Training target construction helpers for trajectory ranking models."""

from __future__ import annotations

import math

import torch

from evaluation.evaluate_methods import score_range_usefulness
from evaluation.query_cache import EvaluationQueryCache
from evaluation.range_usefulness import RANGE_USEFULNESS_WEIGHTS
from queries.range_geometry import points_in_range_box, segment_box_bracket_indices
from queries.query_types import QUERY_TYPE_ID_RANGE
from simplification.simplify_trajectories import (
    deterministic_topk_with_jitter,
    evenly_spaced_indices,
    simplify_with_global_score_budget,
    simplify_with_temporal_score_hybrid,
)
from training.model_features import (
    HISTORICAL_PRIOR_CLOCK_DIM,
    HISTORICAL_PRIOR_DENSITY_DIM,
    build_historical_prior_point_features,
)
from training.training_losses import _safe_quantile

LEGACY_RANGE_TARGET_MODES = frozenset(
    {
        "point_value",
        "retained_frequency",
        "global_budget_retained_frequency",
        "historical_prior_retained_frequency",
        "structural_retained_frequency",
        "component_retained_frequency",
        "continuity_retained_frequency",
        "marginal_coverage_frequency",
        "query_spine_frequency",
        "query_residual_frequency",
        "set_utility_frequency",
        "local_swap_utility_frequency",
        "local_swap_gain_cost_frequency",
    }
)
QUERY_USEFUL_V1_TARGET_MODES = frozenset({"query_useful_v1_factorized"})
RANGE_TRAINING_TARGET_MODES = (
    "point_value",
    "retained_frequency",
    "global_budget_retained_frequency",
    "historical_prior_retained_frequency",
    "structural_retained_frequency",
    "component_retained_frequency",
    "continuity_retained_frequency",
    "marginal_coverage_frequency",
    "query_spine_frequency",
    "query_residual_frequency",
    "set_utility_frequency",
    "local_swap_utility_frequency",
    "local_swap_gain_cost_frequency",
    "query_useful_v1_factorized",
)
RANGE_TARGET_BALANCE_MODES = ("none", "trajectory_unit_mass")
RANGE_CONTINUITY_TARGET_WEIGHTS = {
    "range_entry_exit_f1": 0.22,
    "range_crossing_f1": 0.16,
    "range_temporal_coverage": 0.22,
    "range_gap_coverage": 0.22,
    "range_turn_coverage": 0.08,
    "range_shape_score": 0.10,
}
RANGE_STRUCTURAL_TARGET_WEIGHTS = {
    "uniqueness": 0.40,
    "turn": 0.20,
    "gap": 0.15,
    "globality": 0.15,
    "endpoint": 0.10,
}


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


def _target_budget_ratios(model_config: object) -> tuple[float, ...]:
    """Return configured budgets used to convert label values into retained frequency."""
    raw = getattr(model_config, "budget_loss_ratios", None) or []
    if not raw:
        raw = getattr(model_config, "range_audit_compression_ratios", None) or []
    if not raw:
        raw = [float(getattr(model_config, "compression_ratio", 0.05))]
    ratios = sorted({float(value) for value in raw if 0.0 < float(value) <= 1.0})
    return tuple(ratios) if ratios else (float(getattr(model_config, "compression_ratio", 0.05)),)


def _target_budget_weights(model_config: object, ratios: tuple[float, ...]) -> tuple[float, ...]:
    """Return normalized retained-frequency target weights for budget ratios."""
    if not ratios:
        return ()
    power = max(0.0, float(getattr(model_config, "range_target_budget_weight_power", 0.0) or 0.0))
    if power <= 0.0:
        weight = 1.0 / float(len(ratios))
        return tuple(weight for _ratio in ratios)
    raw = [float(max(float(ratio), 1e-9)) ** (-power) for ratio in ratios]
    total = sum(raw)
    if total <= 1e-12:
        weight = 1.0 / float(len(ratios))
        return tuple(weight for _ratio in ratios)
    return tuple(value / total for value in raw)


def aggregate_range_label_sets(
    label_sets: list[tuple[torch.Tensor, torch.Tensor]],
    type_idx: int = QUERY_TYPE_ID_RANGE,
    source: str = "range_training_label_replicates",
    aggregation: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Aggregate range labels over independent training workloads.

    The aggregation is training-only supervision. It does not expose validation
    or eval queries to the blind compressor.
    """
    if not label_sets:
        raise ValueError("label_sets must contain at least one label/mask pair.")

    first_labels, first_mask = label_sets[0]
    if first_labels.ndim != 2 or first_mask.shape != first_labels.shape:
        raise ValueError("labels and labelled_mask must have matching shape [n_points, n_types].")
    if type_idx < 0 or type_idx >= first_labels.shape[1]:
        raise ValueError(f"type_idx={type_idx} is outside label shape {tuple(first_labels.shape)}.")
    aggregation_mode = aggregation.strip().lower()
    if aggregation_mode not in {"mean", "max"}:
        raise ValueError("aggregation must be 'mean' or 'max'.")

    label_values = torch.zeros_like(first_labels[:, type_idx], dtype=torch.float32)
    label_count = torch.zeros_like(label_values)
    aggregated_mask = first_mask.clone()
    for labels, labelled_mask in label_sets:
        if labels.shape != first_labels.shape or labelled_mask.shape != first_mask.shape:
            raise ValueError("all label sets must have identical shapes.")
        active = labelled_mask[:, type_idx].to(dtype=torch.bool)
        active_values = labels[active, type_idx].float()
        if aggregation_mode == "mean":
            label_values[active] += active_values
        else:
            label_values[active] = torch.maximum(label_values[active], active_values)
        label_count[active] += 1.0
        aggregated_mask |= labelled_mask

    aggregated = first_labels.clone()
    if aggregation_mode == "mean":
        target = label_values / label_count.clamp(min=1.0)
    else:
        target = label_values
    aggregated[:, type_idx] = target.to(dtype=aggregated.dtype)
    aggregated_mask[:, type_idx] = label_count > 0

    positive = aggregated_mask[:, type_idx] & (aggregated[:, type_idx] > 0.0)
    positive_count = int(positive.sum().item())
    diagnostics: dict[str, object] = {
        "source": source,
        "aggregation": aggregation_mode,
        "replicate_count": len(label_sets),
        "labelled_point_count": int((label_count > 0).sum().item()),
        "positive_label_count": positive_count,
        "positive_label_fraction": float(positive_count / max(1, int(aggregated.shape[0]))),
        "positive_label_mass": float(aggregated[positive, type_idx].sum().item()) if positive_count > 0 else 0.0,
    }
    return aggregated, aggregated_mask, diagnostics


def balance_range_training_target_by_trajectory(
    labels: torch.Tensor,
    labelled_mask: torch.Tensor,
    boundaries: list[tuple[int, int]],
    mode: str = "none",
    type_idx: int = QUERY_TYPE_ID_RANGE,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Optionally rebalance range target mass across train trajectories.

    This is a training-only transform. It is useful for diagnosing whether a
    blind prior is dominated by a few dense historical routes even though final
    retention budgets are allocated per trajectory.
    """
    if labels.ndim != 2 or labelled_mask.shape != labels.shape:
        raise ValueError("labels and labelled_mask must have matching shape [n_points, n_types].")
    if type_idx < 0 or type_idx >= labels.shape[1]:
        raise ValueError(f"type_idx={type_idx} is outside label shape {tuple(labels.shape)}.")
    balance_mode = str(mode).lower()
    if balance_mode not in RANGE_TARGET_BALANCE_MODES:
        raise ValueError(f"range target balance mode must be one of {RANGE_TARGET_BALANCE_MODES}.")

    target = labels[:, type_idx].float()
    positive = labelled_mask[:, type_idx].to(dtype=torch.bool) & (target > 0.0)
    before_mass = float(target[positive].sum().item()) if bool(positive.any().item()) else 0.0
    before_count = int(positive.sum().item())
    if balance_mode == "none":
        return labels, labelled_mask, {
            "enabled": False,
            "mode": "none",
            "positive_label_count": before_count,
            "positive_label_mass": before_mass,
        }

    balanced = labels.clone()
    balanced_target = balanced[:, type_idx].float()
    trajectory_masses: list[float] = []
    balanced_trajectory_count = 0
    for start, end in boundaries:
        start_i = int(start)
        end_i = int(end)
        if end_i <= start_i:
            continue
        local_positive = positive[start_i:end_i]
        if not bool(local_positive.any().item()):
            continue
        local_values = balanced_target[start_i:end_i]
        local_mass = float(local_values[local_positive].sum().item())
        if local_mass <= 1e-12:
            continue
        local_values[local_positive] = local_values[local_positive] / local_mass
        balanced_target[start_i:end_i] = local_values
        trajectory_masses.append(local_mass)
        balanced_trajectory_count += 1

    balanced[:, type_idx] = balanced_target.to(dtype=balanced.dtype).clamp(0.0, 1.0)
    after_positive = labelled_mask[:, type_idx].to(dtype=torch.bool) & (balanced[:, type_idx] > 0.0)
    after_count = int(after_positive.sum().item())
    after_mass = float(balanced[after_positive, type_idx].sum().item()) if after_count > 0 else 0.0
    mass_tensor = (
        torch.tensor(trajectory_masses, dtype=torch.float32, device=labels.device)
        if trajectory_masses
        else torch.empty((0,), dtype=torch.float32, device=labels.device)
    )
    diagnostics: dict[str, object] = {
        "enabled": True,
        "mode": balance_mode,
        "positive_label_count": after_count,
        "positive_label_mass": after_mass,
        "positive_label_count_before_balance": before_count,
        "positive_label_mass_before_balance": before_mass,
        "balanced_trajectory_count": int(balanced_trajectory_count),
    }
    if int(mass_tensor.numel()) > 0:
        diagnostics.update(
            {
                "trajectory_positive_mass_p50_before_balance": float(torch.quantile(mass_tensor, 0.50).item()),
                "trajectory_positive_mass_p90_before_balance": float(torch.quantile(mass_tensor, 0.90).item()),
                "trajectory_positive_mass_max_before_balance": float(mass_tensor.max().item()),
            }
        )
    return balanced, labelled_mask.clone(), diagnostics


def range_retained_frequency_training_labels(
    labels: torch.Tensor,
    labelled_mask: torch.Tensor,
    boundaries: list[tuple[int, int]],
    model_config: object,
    type_idx: int = QUERY_TYPE_ID_RANGE,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Convert range label values into oracle retained-set frequency targets.

    This remains workload-blind at inference: only training-workload labels are
    transformed, and the eval compressor still receives point features only.
    """
    if labels.ndim != 2 or labelled_mask.shape != labels.shape:
        raise ValueError("labels and labelled_mask must have matching shape [n_points, n_types].")
    if type_idx < 0 or type_idx >= labels.shape[1]:
        raise ValueError(f"type_idx={type_idx} is outside label shape {tuple(labels.shape)}.")

    ratios = _target_budget_ratios(model_config)
    budget_weights = _target_budget_weights(model_config, ratios)
    source_scores = labels[:, type_idx].float().clamp(min=0.0)
    source_positive = source_scores > 0.0
    retained_frequency = torch.zeros_like(source_scores)
    used = 0
    used_weight = 0.0
    for ratio, budget_weight in zip(ratios, budget_weights):
        mask = simplify_with_temporal_score_hybrid(
            source_scores,
            boundaries,
            float(ratio),
            temporal_fraction=float(getattr(model_config, "mlqds_temporal_fraction", 0.0)),
            diversity_bonus=float(getattr(model_config, "mlqds_diversity_bonus", 0.0)),
            hybrid_mode=str(getattr(model_config, "mlqds_hybrid_mode", "fill")),
            stratified_center_weight=float(getattr(model_config, "mlqds_stratified_center_weight", 0.0)),
            min_learned_swaps=int(getattr(model_config, "mlqds_min_learned_swaps", 0)),
        )
        retained_frequency += float(budget_weight) * (mask & source_positive).to(dtype=retained_frequency.dtype)
        used += 1
        used_weight += float(budget_weight)
    if used_weight > 1e-12:
        retained_frequency = retained_frequency / float(used_weight)
    retained_frequency, temporal_blend_diagnostics = _apply_temporal_target_blend(
        retained_frequency=retained_frequency,
        boundaries=boundaries,
        model_config=model_config,
        ratios=ratios,
    )
    transformed = labels.clone()
    transformed[:, type_idx] = retained_frequency.clamp(0.0, 1.0)
    transformed_mask = labelled_mask.clone()
    transformed_mask[:, type_idx] = True
    positive = transformed[:, type_idx] > 0.0
    positive_count = int(positive.sum().item())
    diagnostics: dict[str, object] = {
        "mode": "retained_frequency",
        "source": "range_training_labels",
        "budget_loss_ratios": list(ratios),
        "budget_weights": list(budget_weights),
        "budget_weight_power": float(getattr(model_config, "range_target_budget_weight_power", 0.0) or 0.0),
        "labelled_point_count": int(transformed.shape[0]),
        "positive_label_count": positive_count,
        "positive_label_fraction": float(positive_count / max(1, int(transformed.shape[0]))),
        "positive_label_mass": float(transformed[positive, type_idx].sum().item()) if positive_count > 0 else 0.0,
        "mlqds_temporal_fraction": float(getattr(model_config, "mlqds_temporal_fraction", 0.0)),
        "mlqds_hybrid_mode": str(getattr(model_config, "mlqds_hybrid_mode", "fill")),
    }
    diagnostics.update(temporal_blend_diagnostics)
    return transformed, transformed_mask, diagnostics


def _global_budget_retained_frequency_from_scores(
    source_scores: torch.Tensor,
    boundaries: list[tuple[int, int]],
    model_config: object,
    ratios: tuple[float, ...],
) -> tuple[torch.Tensor, int]:
    """Return retained frequency from global score allocation across trajectories."""
    source_positive = source_scores > 0.0
    retained_frequency = torch.zeros_like(source_scores, dtype=torch.float32)
    used = 0
    used_weight = 0.0
    for ratio, budget_weight in zip(ratios, _target_budget_weights(model_config, ratios)):
        mask = simplify_with_global_score_budget(
            source_scores,
            boundaries,
            float(ratio),
        )
        retained_frequency += float(budget_weight) * (mask & source_positive).to(
            dtype=retained_frequency.dtype
        )
        used += 1
        used_weight += float(budget_weight)
    if used_weight > 1e-12:
        retained_frequency = retained_frequency / float(used_weight)
    return retained_frequency, used


def range_global_budget_retained_frequency_training_labels(
    labels: torch.Tensor,
    labelled_mask: torch.Tensor,
    boundaries: list[tuple[int, int]],
    model_config: object,
    type_idx: int = QUERY_TYPE_ID_RANGE,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Convert range labels into global-scarcity retained-frequency targets.

    This is training-only supervision. It uses train-workload labels to ask
    which useful points would win under a database-level budget, then the eval
    compressor still scores points without eval queries.
    """
    if labels.ndim != 2 or labelled_mask.shape != labels.shape:
        raise ValueError("labels and labelled_mask must have matching shape [n_points, n_types].")
    if type_idx < 0 or type_idx >= labels.shape[1]:
        raise ValueError(f"type_idx={type_idx} is outside label shape {tuple(labels.shape)}.")

    ratios = _target_budget_ratios(model_config)
    budget_weights = _target_budget_weights(model_config, ratios)
    source_scores = labels[:, type_idx].float().clamp(min=0.0)
    retained_frequency, used = _global_budget_retained_frequency_from_scores(
        source_scores=source_scores,
        boundaries=boundaries,
        model_config=model_config,
        ratios=ratios,
    )
    retained_frequency, temporal_blend_diagnostics = _apply_temporal_target_blend(
        retained_frequency=retained_frequency,
        boundaries=boundaries,
        model_config=model_config,
        ratios=ratios,
    )

    transformed = labels.clone()
    transformed[:, type_idx] = retained_frequency.clamp(0.0, 1.0)
    transformed_mask = labelled_mask.clone()
    transformed_mask[:, type_idx] = True
    positive = transformed[:, type_idx] > 0.0
    positive_count = int(positive.sum().item())
    diagnostics: dict[str, object] = {
        "mode": "global_budget_retained_frequency",
        "source": "range_training_labels_global_budget",
        "budget_loss_ratios": list(ratios),
        "budget_weights": list(budget_weights),
        "budget_weight_power": float(getattr(model_config, "range_target_budget_weight_power", 0.0) or 0.0),
        "labelled_point_count": int(transformed.shape[0]),
        "positive_label_count": positive_count,
        "positive_label_fraction": float(positive_count / max(1, int(transformed.shape[0]))),
        "positive_label_mass": float(transformed[positive, type_idx].sum().item()) if positive_count > 0 else 0.0,
        "global_budget_frequency_budget_count": int(used),
        "global_budget_min_points_per_trajectory": 2,
    }
    diagnostics.update(temporal_blend_diagnostics)
    return transformed, transformed_mask, diagnostics


def _historical_prior_support_mask(
    targets: torch.Tensor,
    boundaries: list[tuple[int, int]],
    support_ratio: float,
) -> torch.Tensor:
    """Return per-trajectory top-target support for historical-prior teachers."""
    ratio = min(1.0, max(0.0, float(support_ratio)))
    support_mask = torch.zeros((int(targets.shape[0]),), dtype=torch.bool, device=targets.device)
    if ratio >= 1.0:
        support_mask[:] = True
        return support_mask
    if ratio <= 0.0:
        return support_mask

    for start, end in boundaries:
        point_count = int(end - start)
        if point_count <= 0:
            continue
        keep_count = min(point_count, max(1, int(math.ceil(ratio * point_count))))
        if keep_count >= point_count:
            support_mask[start:end] = True
            continue
        local_targets = targets[start:end].float()
        local_indices = torch.topk(local_targets, k=keep_count, largest=True).indices
        support_mask[start + local_indices] = True
    return support_mask


def _minmax_scale_feature_matrix(features: torch.Tensor) -> torch.Tensor:
    """Scale feature columns with the same min-max convention as FeatureScaler."""
    if features.ndim != 2:
        raise ValueError("features must be a matrix.")
    if int(features.shape[0]) == 0:
        return features.float()
    values = features.float()
    feature_min = values.min(dim=0).values
    feature_max = values.max(dim=0).values
    denom = (feature_max - feature_min).clamp(min=1e-6)
    return (values - feature_min) / denom


def _weight_historical_prior_features(features: torch.Tensor, model_config: object) -> torch.Tensor:
    """Apply historical-prior feature weights used by the KNN diagnostic model."""
    weighted = features.float().clone()
    point_dim = int(weighted.shape[1])
    density_dim = HISTORICAL_PRIOR_DENSITY_DIM if point_dim >= 21 else 0
    clock_dim = HISTORICAL_PRIOR_CLOCK_DIM if point_dim >= 23 else 0
    clock_weight = max(0.0, float(getattr(model_config, "historical_prior_clock_weight", 0.0)))
    density_weight = max(0.0, float(getattr(model_config, "historical_prior_density_weight", 1.0)))
    if clock_dim > 0 and clock_weight != 1.0:
        clock_start = point_dim - density_dim - clock_dim
        clock_end = point_dim - density_dim
        weighted[:, clock_start:clock_end] *= clock_weight
    if density_dim > 0 and density_weight != 1.0:
        weighted[:, -density_dim:] *= density_weight
    return weighted


def _historical_prior_teacher_scores(
    points: torch.Tensor,
    targets: torch.Tensor,
    boundaries: list[tuple[int, int]],
    model_config: object,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Score train points with a leave-one-out query-free historical KNN teacher."""
    if points.ndim != 2:
        raise ValueError("points must have shape [n_points, point_dim].")
    if targets.ndim != 1 or int(targets.shape[0]) != int(points.shape[0]):
        raise ValueError("targets must be a vector matching points.")
    if int(points.shape[0]) == 0:
        raise ValueError("historical-prior teacher requires at least one point.")

    features = _minmax_scale_feature_matrix(build_historical_prior_point_features(points))
    weighted_features = _weight_historical_prior_features(features, model_config)
    support_ratio = min(1.0, max(0.0, float(getattr(model_config, "historical_prior_support_ratio", 1.0))))
    support_mask = _historical_prior_support_mask(targets, boundaries, support_ratio)
    if not bool(support_mask.any().item()):
        raise ValueError("historical_prior_support_ratio removed every teacher support point.")
    pre_min_support_count = int(support_mask.sum().item())

    min_target = max(0.0, float(getattr(model_config, "historical_prior_min_target", 0.0)))
    if min_target > 0.0:
        support_mask &= targets >= min_target
        if not bool(support_mask.any().item()):
            raise ValueError(
                "historical_prior_min_target removed every teacher support point; "
                f"threshold={min_target:.6f}, max_target={float(targets.max().item()):.6f}."
            )

    support_indices = torch.where(support_mask)[0]
    support_features = weighted_features[support_indices]
    support_targets = targets[support_indices].float().clamp(0.0, 1.0)
    support_count = int(support_features.shape[0])
    k = min(max(1, int(getattr(model_config, "historical_prior_k", 32))), support_count)
    chunk_size = max(1, min(max(1, int(getattr(model_config, "query_chunk_size", 1024))), 1024))
    scores = torch.empty((int(points.shape[0]),), dtype=torch.float32, device=points.device)

    all_indices = torch.arange(int(points.shape[0]), dtype=support_indices.dtype, device=points.device)
    support_indices = support_indices.to(device=points.device)
    support_features = support_features.to(device=points.device)
    support_targets = support_targets.to(device=points.device)
    weighted_features = weighted_features.to(device=points.device)
    for start in range(0, int(points.shape[0]), chunk_size):
        end = min(int(points.shape[0]), start + chunk_size)
        distances = torch.cdist(weighted_features[start:end], support_features, p=2).square()
        same_point = all_indices[start:end].unsqueeze(1) == support_indices.unsqueeze(0)
        if bool(same_point.any().item()):
            distances = distances.masked_fill(same_point, float("inf"))
        nearest_distances, nearest_idx = torch.topk(distances, k=k, largest=False, dim=1)
        finite = torch.isfinite(nearest_distances)
        weights = torch.where(finite, 1.0 / (nearest_distances + 1e-4), torch.zeros_like(nearest_distances))
        denom = weights.sum(dim=1)
        local_scores = (weights * support_targets[nearest_idx]).sum(dim=1) / denom.clamp(min=1e-9)
        fallback = targets[start:end].float().clamp(0.0, 1.0)
        scores[start:end] = torch.where(denom > 1e-12, local_scores, fallback)

    positive = scores > 0.0
    diagnostics: dict[str, object] = {
        "historical_prior_teacher_k": int(k),
        "historical_prior_teacher_leave_one_out": True,
        "historical_prior_support_ratio": float(support_ratio),
        "historical_prior_support_pre_min_count": int(pre_min_support_count),
        "historical_prior_min_target": float(min_target),
        "historical_prior_stored_support_count": int(support_count),
        "historical_prior_stored_support_fraction": float(support_count / max(1, int(points.shape[0]))),
        "historical_prior_clock_weight": float(
            getattr(model_config, "historical_prior_clock_weight", 0.0)
        ),
        "historical_prior_density_weight": float(
            getattr(model_config, "historical_prior_density_weight", 1.0)
        ),
        "historical_prior_teacher_positive_score_count": int(positive.sum().item()),
        "historical_prior_teacher_positive_score_fraction": float(
            int(positive.sum().item()) / max(1, int(scores.numel()))
        ),
        "historical_prior_teacher_score_mass": float(scores[positive].sum().item())
        if bool(positive.any().item())
        else 0.0,
    }
    if bool(positive.any().item()):
        diagnostics.update(
            {
                "historical_prior_teacher_score_p50": float(_safe_quantile(scores[positive], 0.50).item()),
                "historical_prior_teacher_score_p95": float(_safe_quantile(scores[positive], 0.95).item()),
            }
        )
    return scores, diagnostics


def _numeric_diagnostic(diagnostics: dict[str, object], key: str, default: float = 0.0) -> float:
    """Read a numeric diagnostics field defensively."""
    value = diagnostics.get(key, default)
    if isinstance(value, (int, float)):
        return float(value)
    return float(default)


def range_historical_prior_retained_frequency_training_labels(
    labels: torch.Tensor,
    labelled_mask: torch.Tensor,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    model_config: object,
    type_idx: int = QUERY_TYPE_ID_RANGE,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Distill retained-frequency labels through a query-free historical KNN teacher.

    This is a training-only teacher. Eval compression still uses the configured
    neural blind model and receives no eval query boxes before masks are frozen.
    """
    retained_labels, retained_mask, base_diagnostics = range_retained_frequency_training_labels(
        labels=labels,
        labelled_mask=labelled_mask,
        boundaries=boundaries,
        model_config=model_config,
        type_idx=type_idx,
    )
    retained_target = retained_labels[:, type_idx].float().clamp(0.0, 1.0)
    teacher_scores, teacher_diagnostics = _historical_prior_teacher_scores(
        points=points,
        targets=retained_target,
        boundaries=boundaries,
        model_config=model_config,
    )
    ratios = _target_budget_ratios(model_config)
    retained_frequency, used = _retained_frequency_from_scores(
        source_scores=teacher_scores,
        boundaries=boundaries,
        model_config=model_config,
        ratios=ratios,
    )
    retained_frequency, temporal_blend_diagnostics = _apply_temporal_target_blend(
        retained_frequency=retained_frequency,
        boundaries=boundaries,
        model_config=model_config,
        ratios=ratios,
    )

    transformed = labels.clone()
    transformed[:, type_idx] = retained_frequency.clamp(0.0, 1.0)
    transformed_mask = retained_mask.clone()
    transformed_mask[:, type_idx] = True
    positive = transformed[:, type_idx] > 0.0
    positive_count = int(positive.sum().item())
    diagnostics: dict[str, object] = {
        "mode": "historical_prior_retained_frequency",
        "source": "range_retained_frequency_leave_one_out_historical_prior_teacher",
        "budget_loss_ratios": list(ratios),
        "budget_weights": list(_target_budget_weights(model_config, ratios)),
        "budget_weight_power": float(getattr(model_config, "range_target_budget_weight_power", 0.0) or 0.0),
        "labelled_point_count": int(transformed.shape[0]),
        "positive_label_count": positive_count,
        "positive_label_fraction": float(positive_count / max(1, int(transformed.shape[0]))),
        "positive_label_mass": float(transformed[positive, type_idx].sum().item()) if positive_count > 0 else 0.0,
        "teacher_retained_frequency_budget_count": int(used),
        "base_retained_frequency_positive_label_count": int(
            _numeric_diagnostic(base_diagnostics, "positive_label_count")
        ),
        "base_retained_frequency_positive_label_fraction": _numeric_diagnostic(
            base_diagnostics,
            "positive_label_fraction",
        ),
        "base_retained_frequency_positive_label_mass": _numeric_diagnostic(
            base_diagnostics,
            "positive_label_mass",
        ),
        "mlqds_temporal_fraction": float(getattr(model_config, "mlqds_temporal_fraction", 0.0)),
        "mlqds_hybrid_mode": str(getattr(model_config, "mlqds_hybrid_mode", "fill")),
    }
    diagnostics.update(teacher_diagnostics)
    diagnostics.update(temporal_blend_diagnostics)
    return transformed, transformed_mask, diagnostics


def _retained_frequency_from_scores(
    source_scores: torch.Tensor,
    boundaries: list[tuple[int, int]],
    model_config: object,
    ratios: tuple[float, ...],
) -> tuple[torch.Tensor, int]:
    """Return retained-set frequency from one nonnegative score stream."""
    source_positive = source_scores > 0.0
    retained_frequency = torch.zeros_like(source_scores, dtype=torch.float32)
    used = 0
    used_weight = 0.0
    for ratio, budget_weight in zip(ratios, _target_budget_weights(model_config, ratios)):
        mask = simplify_with_temporal_score_hybrid(
            source_scores,
            boundaries,
            float(ratio),
            temporal_fraction=float(getattr(model_config, "mlqds_temporal_fraction", 0.0)),
            diversity_bonus=float(getattr(model_config, "mlqds_diversity_bonus", 0.0)),
            hybrid_mode=str(getattr(model_config, "mlqds_hybrid_mode", "fill")),
            stratified_center_weight=float(getattr(model_config, "mlqds_stratified_center_weight", 0.0)),
            min_learned_swaps=int(getattr(model_config, "mlqds_min_learned_swaps", 0)),
        )
        retained_frequency += float(budget_weight) * (mask & source_positive).to(dtype=retained_frequency.dtype)
        used += 1
        used_weight += float(budget_weight)
    if used_weight > 1e-12:
        retained_frequency = retained_frequency / float(used_weight)
    return retained_frequency, used


def _temporal_retained_frequency(
    scores: torch.Tensor,
    boundaries: list[tuple[int, int]],
    ratios: tuple[float, ...],
    weights: tuple[float, ...] | None = None,
) -> tuple[torch.Tensor, int]:
    """Return retained frequency for pure evenly spaced temporal sampling."""
    retained_frequency = torch.zeros_like(scores, dtype=torch.float32)
    budget_weights = weights or tuple(1.0 / float(len(ratios)) for _ratio in ratios)
    used = 0
    used_weight = 0.0
    for ratio, budget_weight in zip(ratios, budget_weights):
        retained = torch.zeros_like(scores, dtype=torch.bool)
        for start, end in boundaries:
            point_count = int(end - start)
            if point_count <= 0:
                continue
            keep_count = min(point_count, max(2, int(math.ceil(float(ratio) * point_count))))
            local_indices = evenly_spaced_indices(point_count, keep_count, scores.device)
            retained[start + local_indices] = True
        retained_frequency += float(budget_weight) * retained.to(dtype=retained_frequency.dtype)
        used += 1
        used_weight += float(budget_weight)
    if used_weight > 1e-12:
        retained_frequency = retained_frequency / float(used_weight)
    return retained_frequency, used


def _apply_temporal_target_blend(
    retained_frequency: torch.Tensor,
    boundaries: list[tuple[int, int]],
    model_config: object,
    ratios: tuple[float, ...],
) -> tuple[torch.Tensor, dict[str, object]]:
    """Blend query-blind temporal anchor frequency into a retained-frequency target."""
    blend = max(0.0, min(1.0, float(getattr(model_config, "range_temporal_target_blend", 0.0) or 0.0)))
    diagnostics: dict[str, object] = {"temporal_target_blend": float(blend)}
    if blend <= 0.0:
        return retained_frequency, diagnostics

    temporal_frequency, used = _temporal_retained_frequency(
        scores=retained_frequency,
        boundaries=boundaries,
        ratios=ratios,
        weights=_target_budget_weights(model_config, ratios),
    )
    target = ((1.0 - blend) * retained_frequency + blend * temporal_frequency).clamp(0.0, 1.0)
    positive = temporal_frequency > 0.0
    diagnostics.update(
        {
            "temporal_target_budget_count": int(used),
            "temporal_target_positive_label_count": int(positive.sum().item()),
            "temporal_target_positive_label_mass": (
                float(temporal_frequency[positive].sum().item()) if bool(positive.any().item()) else 0.0
            ),
        }
    )
    return target, diagnostics


def _scale01(values: torch.Tensor) -> torch.Tensor:
    """Return per-vector min-max scores in [0, 1]."""
    if int(values.numel()) == 0:
        return values.float()
    values_f = values.float()
    span = values_f.max() - values_f.min()
    if float(span.item()) <= 1e-12:
        return torch.zeros_like(values_f)
    return (values_f - values_f.min()) / span.clamp(min=1e-12)


def _query_free_structural_scores(
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
) -> tuple[torch.Tensor, dict[str, object]]:
    """Return MLSimp-inspired query-free structural point scores.

    The score is deliberately train/eval-query blind: it uses only the observed
    trajectory geometry and dynamics. It is intended as a training regularizer
    for retained-frequency labels, not as an inference-time geometry blend.
    """
    if points.ndim != 2 or int(points.shape[1]) < 3:
        raise ValueError("points must have shape [n_points, point_dim>=3].")

    structural = torch.zeros((int(points.shape[0]),), dtype=torch.float32, device=points.device)
    component_mass = {
        "uniqueness": 0.0,
        "turn": 0.0,
        "gap": 0.0,
        "globality": 0.0,
        "endpoint": 0.0,
    }
    scored_trajectories = 0
    for start, end in boundaries:
        start_i = int(start)
        end_i = int(end)
        count = end_i - start_i
        if count <= 0:
            continue
        scored_trajectories += 1
        local = points[start_i:end_i].float()
        if count == 1:
            structural[start_i:end_i] = 1.0
            component_mass["endpoint"] += 1.0
            continue

        coords = local[:, 1:3]
        coord_min = coords.min(dim=0).values
        coord_span = (coords.max(dim=0).values - coord_min).clamp(min=1e-6)
        coords_norm = (coords - coord_min) / coord_span

        uniqueness = torch.zeros((count,), dtype=torch.float32, device=points.device)
        if count >= 3:
            midpoint = 0.5 * (coords_norm[:-2] + coords_norm[2:])
            uniqueness[1:-1] = torch.linalg.vector_norm(coords_norm[1:-1] - midpoint, dim=1)
            endpoint_value = float(uniqueness[1:-1].max().item()) if bool((uniqueness[1:-1] > 0).any().item()) else 1.0
        else:
            endpoint_value = 1.0
        uniqueness[0] = endpoint_value
        uniqueness[-1] = endpoint_value
        uniqueness = _scale01(uniqueness)

        turn = (
            _scale01(local[:, 7].clamp(min=0.0))
            if int(local.shape[1]) > 7
            else torch.zeros((count,), dtype=torch.float32, device=points.device)
        )

        times = local[:, 0]
        deltas = torch.diff(times).abs() if count >= 2 else times.new_empty((0,))
        prev_gap = torch.cat([times.new_zeros((1,)), deltas])
        next_gap = torch.cat([deltas, times.new_zeros((1,))])
        gap = _scale01(torch.maximum(prev_gap, next_gap))

        centroid = coords_norm.mean(dim=0)
        dist_to_centroid = torch.linalg.vector_norm(coords_norm - centroid.unsqueeze(0), dim=1)
        globality = 1.0 - _scale01(dist_to_centroid)

        endpoint = torch.zeros((count,), dtype=torch.float32, device=points.device)
        endpoint[0] = 1.0
        endpoint[-1] = 1.0
        if int(local.shape[1]) > 6:
            endpoint = torch.maximum(endpoint, local[:, 5].clamp(0.0, 1.0))
            endpoint = torch.maximum(endpoint, local[:, 6].clamp(0.0, 1.0))

        local_structural = (
            RANGE_STRUCTURAL_TARGET_WEIGHTS["uniqueness"] * uniqueness
            + RANGE_STRUCTURAL_TARGET_WEIGHTS["turn"] * turn
            + RANGE_STRUCTURAL_TARGET_WEIGHTS["gap"] * gap
            + RANGE_STRUCTURAL_TARGET_WEIGHTS["globality"] * globality
            + RANGE_STRUCTURAL_TARGET_WEIGHTS["endpoint"] * endpoint
        ).clamp(0.0, 1.0)
        structural[start_i:end_i] = local_structural
        component_mass["uniqueness"] += float(uniqueness.sum().item())
        component_mass["turn"] += float(turn.sum().item())
        component_mass["gap"] += float(gap.sum().item())
        component_mass["globality"] += float(globality.sum().item())
        component_mass["endpoint"] += float(endpoint.sum().item())

    positive = structural > 0.0
    diagnostics: dict[str, object] = {
        "structural_score_trajectory_count": int(scored_trajectories),
        "structural_score_positive_count": int(positive.sum().item()),
        "structural_score_positive_fraction": float(
            int(positive.sum().item()) / max(1, int(structural.numel()))
        ),
        "structural_score_positive_mass": (
            float(structural[positive].sum().item()) if bool(positive.any().item()) else 0.0
        ),
        "structural_score_component_weights": dict(RANGE_STRUCTURAL_TARGET_WEIGHTS),
        "structural_score_component_mass": component_mass,
    }
    if bool(positive.any().item()):
        diagnostics.update(
            {
                "structural_score_p50": float(_safe_quantile(structural[positive], 0.50).item()),
                "structural_score_p95": float(_safe_quantile(structural[positive], 0.95).item()),
            }
        )
    return structural, diagnostics


def range_structural_retained_frequency_training_labels(
    labels: torch.Tensor,
    labelled_mask: torch.Tensor,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    model_config: object,
    type_idx: int = QUERY_TYPE_ID_RANGE,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Blend train workload usefulness with query-free structural scores.

    This is a training-only target transform. The deployed scorer still sees
    only query-free point features and final eval masks are frozen before
    held-out range queries are scored.
    """
    if labels.ndim != 2 or labelled_mask.shape != labels.shape:
        raise ValueError("labels and labelled_mask must have matching shape [n_points, n_types].")
    if type_idx < 0 or type_idx >= labels.shape[1]:
        raise ValueError(f"type_idx={type_idx} is outside label shape {tuple(labels.shape)}.")
    if int(points.shape[0]) != int(labels.shape[0]):
        raise ValueError("points and labels must have the same point count.")

    ratios = _target_budget_ratios(model_config)
    blend = max(0.0, min(1.0, float(getattr(model_config, "range_structural_target_blend", 0.25) or 0.0)))
    label_scores = labels[:, type_idx].float().clamp(0.0, 1.0)
    structural_scores, structural_diagnostics = _query_free_structural_scores(points, boundaries)
    source_mode = str(getattr(model_config, "range_structural_target_source_mode", "blend")).lower()
    if source_mode not in {"blend", "boost"}:
        raise ValueError("range_structural_target_source_mode must be 'blend' or 'boost'.")
    if source_mode == "boost":
        source_scores = label_scores * (1.0 + blend * structural_scores)
    else:
        source_scores = ((1.0 - blend) * label_scores + blend * structural_scores).clamp(0.0, 1.0)

    base_frequency, base_used = _retained_frequency_from_scores(
        source_scores=label_scores,
        boundaries=boundaries,
        model_config=model_config,
        ratios=ratios,
    )
    retained_frequency, used = _retained_frequency_from_scores(
        source_scores=source_scores,
        boundaries=boundaries,
        model_config=model_config,
        ratios=ratios,
    )
    retained_frequency, temporal_blend_diagnostics = _apply_temporal_target_blend(
        retained_frequency=retained_frequency,
        boundaries=boundaries,
        model_config=model_config,
        ratios=ratios,
    )

    transformed = labels.clone()
    transformed[:, type_idx] = retained_frequency.clamp(0.0, 1.0)
    transformed_mask = labelled_mask.clone()
    transformed_mask[:, type_idx] = True
    positive = transformed[:, type_idx] > 0.0
    source_positive = source_scores > 0.0
    base_positive = base_frequency > 0.0
    positive_count = int(positive.sum().item())
    diagnostics: dict[str, object] = {
        "mode": "structural_retained_frequency",
        "source": "range_training_labels_plus_query_free_structural_scores",
        "budget_loss_ratios": list(ratios),
        "budget_weights": list(_target_budget_weights(model_config, ratios)),
        "budget_weight_power": float(getattr(model_config, "range_target_budget_weight_power", 0.0) or 0.0),
        "structural_target_blend": float(blend),
        "structural_target_source_mode": source_mode,
        "labelled_point_count": int(transformed.shape[0]),
        "positive_label_count": positive_count,
        "positive_label_fraction": float(positive_count / max(1, int(transformed.shape[0]))),
        "positive_label_mass": float(transformed[positive, type_idx].sum().item()) if positive_count > 0 else 0.0,
        "source_positive_label_count": int(source_positive.sum().item()),
        "source_positive_label_mass": (
            float(source_scores[source_positive].sum().item()) if bool(source_positive.any().item()) else 0.0
        ),
        "base_retained_frequency_budget_count": int(base_used),
        "base_retained_frequency_positive_label_count": int(base_positive.sum().item()),
        "base_retained_frequency_positive_label_mass": (
            float(base_frequency[base_positive].sum().item()) if bool(base_positive.any().item()) else 0.0
        ),
        "teacher_retained_frequency_budget_count": int(used),
        "mlqds_temporal_fraction": float(getattr(model_config, "mlqds_temporal_fraction", 0.0)),
        "mlqds_hybrid_mode": str(getattr(model_config, "mlqds_hybrid_mode", "fill")),
    }
    diagnostics.update(structural_diagnostics)
    diagnostics.update(temporal_blend_diagnostics)
    return transformed, transformed_mask, diagnostics


def _local_window_sum(values: torch.Tensor, radius: int) -> torch.Tensor:
    """Return inclusive fixed-radius window sums for one trajectory vector."""
    count = int(values.numel())
    if count <= 0:
        return values.new_empty((0,), dtype=torch.float32)
    if int(radius) <= 0:
        return values.float()
    positions = torch.arange(count, device=values.device)
    left = torch.clamp(positions - int(radius), min=0)
    right = torch.clamp(positions + int(radius) + 1, max=count)
    prefix = torch.cat([values.new_zeros((1,), dtype=torch.float32), values.float().cumsum(dim=0)])
    return prefix[right] - prefix[left]


def _erase_local_window(values: torch.Tensor, index: int, radius: int) -> None:
    """Mark one selected point's local label neighborhood as covered."""
    count = int(values.numel())
    if count <= 0:
        return
    left = max(0, int(index) - max(0, int(radius)))
    right = min(count, int(index) + max(0, int(radius)) + 1)
    values[left:right] = 0.0


def _marginal_coverage_mask_from_scores(
    source_scores: torch.Tensor,
    boundaries: list[tuple[int, int]],
    ratio: float,
    model_config: object,
) -> torch.Tensor:
    """Greedily retain points that cover remaining local label mass.

    Unlike plain top-k retained-frequency labels, selecting a point consumes
    nearby label mass before the next selection. This gives broad, redundant
    workloads a set-aware target without passing queries into the model.
    """
    retained = torch.zeros_like(source_scores, dtype=torch.bool)
    radius_scale = max(0.0, float(getattr(model_config, "range_marginal_target_radius_scale", 0.50) or 0.0))
    base_fraction = min(1.0, max(0.0, float(getattr(model_config, "mlqds_temporal_fraction", 0.0))))
    for trajectory_id, (start, end) in enumerate(boundaries):
        point_count = int(end - start)
        if point_count <= 0:
            continue
        total_keep_count = min(point_count, max(2, int(math.ceil(float(ratio) * point_count))))
        local_scores = source_scores[start:end].float().clamp(min=0.0)
        selected = torch.zeros((point_count,), dtype=torch.bool, device=source_scores.device)
        base_keep_count = 0
        if base_fraction > 0.0:
            base_keep_count = min(total_keep_count, max(2, int(math.ceil(total_keep_count * base_fraction))))
        base_indices = evenly_spaced_indices(point_count, base_keep_count, source_scores.device)
        if base_indices.numel() > 0:
            selected[base_indices] = True

        expected_spacing = float(point_count) / float(max(1, total_keep_count))
        radius = max(0, int(math.ceil(radius_scale * expected_spacing)))
        uncovered = local_scores.clone()
        for index in base_indices.detach().cpu().tolist():
            _erase_local_window(uncovered, int(index), radius)

        remaining = total_keep_count - int(selected.sum().item())
        positions = torch.arange(point_count, dtype=torch.float32, device=source_scores.device)
        for step in range(max(0, remaining)):
            available = ~selected
            if not bool(available.any().item()):
                break
            gains = _local_window_sum(uncovered, radius)
            if not bool((gains[available] > 1e-12).any().item()):
                if bool(selected.any().item()):
                    selected_positions = torch.where(selected)[0].float()
                    gains = torch.abs(positions.unsqueeze(1) - selected_positions.unsqueeze(0)).min(dim=1).values
                else:
                    gains = torch.ones_like(gains)
            gains = gains + 1e-3 * local_scores
            gains = gains.masked_fill(~available, float("-inf"))
            next_idx = deterministic_topk_with_jitter(
                gains,
                keep_count=1,
                trajectory_id=trajectory_id * 1009 + step,
            )
            if next_idx.numel() == 0 or not torch.isfinite(gains[next_idx[0]]):
                break
            idx = int(next_idx[0].item())
            selected[idx] = True
            _erase_local_window(uncovered, idx, radius)

        retained[start:end] = selected
    return retained


def range_marginal_coverage_training_labels(
    labels: torch.Tensor,
    labelled_mask: torch.Tensor,
    boundaries: list[tuple[int, int]],
    model_config: object,
    type_idx: int = QUERY_TYPE_ID_RANGE,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Convert range labels into neighborhood-marginal retained frequency."""
    if labels.ndim != 2 or labelled_mask.shape != labels.shape:
        raise ValueError("labels and labelled_mask must have matching shape [n_points, n_types].")
    if type_idx < 0 or type_idx >= labels.shape[1]:
        raise ValueError(f"type_idx={type_idx} is outside label shape {tuple(labels.shape)}.")

    ratios = _target_budget_ratios(model_config)
    budget_weights = _target_budget_weights(model_config, ratios)
    source_scores = labels[:, type_idx].float().clamp(min=0.0)
    retained_frequency = torch.zeros_like(source_scores)
    used = 0
    used_weight = 0.0
    for ratio, budget_weight in zip(ratios, budget_weights):
        mask = _marginal_coverage_mask_from_scores(
            source_scores=source_scores,
            boundaries=boundaries,
            ratio=float(ratio),
            model_config=model_config,
        )
        retained_frequency += float(budget_weight) * mask.to(dtype=retained_frequency.dtype)
        used += 1
        used_weight += float(budget_weight)
    if used_weight > 1e-12:
        retained_frequency = retained_frequency / float(used_weight)
    retained_frequency, temporal_blend_diagnostics = _apply_temporal_target_blend(
        retained_frequency=retained_frequency,
        boundaries=boundaries,
        model_config=model_config,
        ratios=ratios,
    )

    transformed = labels.clone()
    transformed[:, type_idx] = retained_frequency.clamp(0.0, 1.0)
    transformed_mask = labelled_mask.clone()
    transformed_mask[:, type_idx] = True
    positive = transformed[:, type_idx] > 0.0
    positive_count = int(positive.sum().item())
    diagnostics: dict[str, object] = {
        "mode": "marginal_coverage_frequency",
        "source": "range_training_labels",
        "budget_loss_ratios": list(ratios),
        "budget_weights": list(budget_weights),
        "budget_weight_power": float(getattr(model_config, "range_target_budget_weight_power", 0.0) or 0.0),
        "labelled_point_count": int(transformed.shape[0]),
        "positive_label_count": positive_count,
        "positive_label_fraction": float(positive_count / max(1, int(transformed.shape[0]))),
        "positive_label_mass": float(transformed[positive, type_idx].sum().item()) if positive_count > 0 else 0.0,
        "marginal_target_radius_scale": float(
            getattr(model_config, "range_marginal_target_radius_scale", 0.50) or 0.0
        ),
        "mlqds_temporal_fraction": float(getattr(model_config, "mlqds_temporal_fraction", 0.0)),
        "mlqds_hybrid_mode": str(getattr(model_config, "mlqds_hybrid_mode", "fill")),
    }
    diagnostics.update(temporal_blend_diagnostics)
    return transformed, transformed_mask, diagnostics


def _range_query_spine_scores(
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, object]],
    model_config: object,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Return query-derived temporal-spine source scores for training only.

    By default, each ``(query, trajectory-hit)`` group contributes equal mass.
    With ``range_query_spine_mass_mode="query"``, each train query contributes
    unit mass split equally across its hit trajectories. In both cases, each
    group's mass is spread over a small set of evenly spaced in-query anchors,
    with high-turn points added as extra shape anchors. This target is query
    aware only during supervision construction; the trained model still receives
    point features only at compression time.
    """
    source_scores = torch.zeros((points.shape[0],), dtype=torch.float32, device=points.device)
    spine_fraction = max(0.0, min(1.0, float(getattr(model_config, "range_query_spine_fraction", 0.10) or 0.0)))
    if spine_fraction <= 0.0:
        raise ValueError("range_query_spine_fraction must be positive for query_spine_frequency.")
    mass_mode = str(getattr(model_config, "range_query_spine_mass_mode", "hit_group")).lower()
    if mass_mode not in {"hit_group", "query"}:
        raise ValueError("range_query_spine_mass_mode must be 'hit_group' or 'query'.")
    turn_fraction = 0.25
    range_query_count = 0
    hit_group_count = 0
    selected_anchor_count = 0
    selected_turn_anchor_count = 0
    query_with_hits_count = 0
    max_query_hit_group_count = 0

    for query_index, query in enumerate(typed_queries):
        if str(query.get("type", "")).lower() != "range":
            continue
        range_query_count += 1
        params = query.get("params")
        if not isinstance(params, dict):
            continue
        box_support = points_in_range_box(points, params)
        query_groups: list[torch.Tensor] = []
        for trajectory_id, (start, end) in enumerate(boundaries):
            if end <= start:
                continue
            local_offsets = torch.where(box_support[start:end])[0]
            local_count = int(local_offsets.numel())
            if local_count <= 0:
                continue
            hit_group_count += 1
            spine_count = min(local_count, max(1, int(math.ceil(spine_fraction * local_count))))
            if local_count >= 2:
                spine_count = max(2, spine_count)
            local_spine_offsets = evenly_spaced_indices(local_count, spine_count, points.device)
            selected_offsets = local_offsets[local_spine_offsets]

            if points.shape[1] > 7 and local_count >= 3:
                turn_scores = points[start + local_offsets, 7].float().clamp(min=0.0)
                turn_count = min(local_count, max(1, int(math.ceil(turn_fraction * spine_count))))
                if bool((turn_scores > 0.0).any().item()):
                    turn_local = deterministic_topk_with_jitter(
                        turn_scores,
                        keep_count=turn_count,
                        trajectory_id=(query_index + 1) * 10007 + trajectory_id,
                    )
                    turn_offsets = local_offsets[turn_local]
                    selected_offsets = torch.unique(torch.cat([selected_offsets, turn_offsets]), sorted=True)
                    selected_turn_anchor_count += int(turn_offsets.numel())

            global_indices = start + selected_offsets
            if global_indices.numel() == 0:
                continue
            query_groups.append(global_indices)
            selected_anchor_count += int(global_indices.numel())

        query_hit_group_count = len(query_groups)
        if query_hit_group_count > 0:
            query_with_hits_count += 1
            max_query_hit_group_count = max(max_query_hit_group_count, query_hit_group_count)
            for global_indices in query_groups:
                group_mass = 1.0
                if mass_mode == "query":
                    group_mass = 1.0 / float(query_hit_group_count)
                source_scores[global_indices] += float(group_mass) / float(global_indices.numel())

    if range_query_count > 0:
        source_scores = source_scores / float(range_query_count)
    positive = source_scores > 0.0
    diagnostics = {
        "query_spine_fraction": float(spine_fraction),
        "query_spine_mass_mode": mass_mode,
        "query_spine_range_query_count": int(range_query_count),
        "query_spine_query_with_hits_count": int(query_with_hits_count),
        "query_spine_hit_group_count": int(hit_group_count),
        "query_spine_hit_groups_per_hit_query_mean": float(
            hit_group_count / max(1, query_with_hits_count)
        ),
        "query_spine_hit_groups_per_hit_query_max": int(max_query_hit_group_count),
        "query_spine_selected_anchor_count": int(selected_anchor_count),
        "query_spine_selected_turn_anchor_count": int(selected_turn_anchor_count),
        "query_spine_source_positive_count": int(positive.sum().item()),
        "query_spine_source_positive_mass": (
            float(source_scores[positive].sum().item()) if bool(positive.any().item()) else 0.0
        ),
    }
    return source_scores, diagnostics


def range_query_spine_frequency_training_labels(
    labels: torch.Tensor,
    labelled_mask: torch.Tensor,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, object]],
    model_config: object,
    type_idx: int = QUERY_TYPE_ID_RANGE,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Convert per-query temporal-spine source scores into retained frequency."""
    if labels.ndim != 2 or labelled_mask.shape != labels.shape:
        raise ValueError("labels and labelled_mask must have matching shape [n_points, n_types].")
    if type_idx < 0 or type_idx >= labels.shape[1]:
        raise ValueError(f"type_idx={type_idx} is outside label shape {tuple(labels.shape)}.")
    if int(points.shape[0]) != int(labels.shape[0]):
        raise ValueError("points and labels must have the same point count.")

    ratios = _target_budget_ratios(model_config)
    source_scores, spine_diagnostics = _range_query_spine_scores(
        points=points,
        boundaries=boundaries,
        typed_queries=typed_queries,
        model_config=model_config,
    )
    if not bool((source_scores > 0.0).any().item()):
        raise ValueError("query_spine_frequency target found no positive source scores.")
    retained_frequency, used = _retained_frequency_from_scores(
        source_scores=source_scores,
        boundaries=boundaries,
        model_config=model_config,
        ratios=ratios,
    )
    if used <= 0:
        raise ValueError("query_spine_frequency target did not use any budget ratios.")
    retained_frequency, temporal_blend_diagnostics = _apply_temporal_target_blend(
        retained_frequency=retained_frequency,
        boundaries=boundaries,
        model_config=model_config,
        ratios=ratios,
    )

    transformed = labels.clone()
    transformed[:, type_idx] = retained_frequency.clamp(0.0, 1.0)
    transformed_mask = labelled_mask.clone()
    transformed_mask[:, type_idx] = True
    positive = transformed[:, type_idx] > 0.0
    positive_count = int(positive.sum().item())
    diagnostics: dict[str, object] = {
        "mode": "query_spine_frequency",
        "source": "range_query_temporal_spines",
        "budget_loss_ratios": list(ratios),
        "budget_weights": list(_target_budget_weights(model_config, ratios)),
        "budget_weight_power": float(getattr(model_config, "range_target_budget_weight_power", 0.0) or 0.0),
        "labelled_point_count": int(transformed.shape[0]),
        "positive_label_count": positive_count,
        "positive_label_fraction": float(positive_count / max(1, int(transformed.shape[0]))),
        "positive_label_mass": float(transformed[positive, type_idx].sum().item()) if positive_count > 0 else 0.0,
        "mlqds_temporal_fraction": float(getattr(model_config, "mlqds_temporal_fraction", 0.0)),
        "mlqds_hybrid_mode": str(getattr(model_config, "mlqds_hybrid_mode", "fill")),
    }
    diagnostics.update(spine_diagnostics)
    diagnostics.update(temporal_blend_diagnostics)
    return transformed, transformed_mask, diagnostics


def _temporal_base_mask_for_ratio(
    *,
    n_points: int,
    boundaries: list[tuple[int, int]],
    ratio: float,
    temporal_fraction: float,
    device: torch.device,
) -> torch.Tensor:
    """Return the query-blind temporal base retained at one total budget."""
    base_mask = torch.zeros((n_points,), dtype=torch.bool, device=device)
    base_fraction = min(1.0, max(0.0, float(temporal_fraction)))
    if base_fraction <= 0.0:
        return base_mask

    total_ratio = min(1.0, max(0.0, float(ratio)))
    if total_ratio <= 0.0:
        return base_mask
    for start, end in boundaries:
        point_count = int(end - start)
        if point_count <= 0:
            continue
        keep_count = min(point_count, max(2, int(math.ceil(total_ratio * point_count))))
        base_count = min(keep_count, max(2, int(math.ceil(keep_count * base_fraction))))
        base_indices = evenly_spaced_indices(point_count, base_count, device)
        base_mask[start + base_indices] = True
    return base_mask


def _query_residual_priority_positions(
    *,
    points: torch.Tensor,
    global_indices: torch.Tensor,
    local_base_mask: torch.Tensor,
    query_keep_count: int,
) -> list[int]:
    """Return in-query local positions preferred for learned residual fill."""
    count = int(global_indices.numel())
    if count <= 0:
        return []

    selected: list[int] = []
    selected_set: set[int] = set()

    def add_positions(positions: list[int] | torch.Tensor) -> None:
        raw_positions = (
            positions.detach().cpu().tolist()
            if isinstance(positions, torch.Tensor)
            else positions
        )
        for value in raw_positions:
            pos = int(value)
            if pos < 0 or pos >= count or pos in selected_set:
                continue
            if bool(local_base_mask[pos].item()):
                continue
            selected_set.add(pos)
            selected.append(pos)

    # Boundary evidence is disproportionately useful for entry/exit and temporal span.
    add_positions([0, count - 1])

    base_positions = torch.where(local_base_mask)[0]
    if base_positions.numel() > 0:
        anchors = torch.cat(
            [
                torch.tensor([-1], dtype=torch.long, device=global_indices.device),
                base_positions.to(dtype=torch.long),
                torch.tensor([count], dtype=torch.long, device=global_indices.device),
            ]
        )
        gap_left = anchors[:-1]
        gap_right = anchors[1:]
        missing = gap_right - gap_left - 1
        ordered_gaps = torch.argsort(missing, descending=True)
        mids: list[int] = []
        for gap_idx in ordered_gaps.detach().cpu().tolist():
            if int(missing[int(gap_idx)].item()) <= 0:
                continue
            left = int(gap_left[int(gap_idx)].item())
            right = int(gap_right[int(gap_idx)].item())
            mids.append((left + right) // 2)
        add_positions(mids)
    else:
        add_positions([count // 2])

    if points.shape[1] > 7 and count >= 3:
        turn_count = min(count, max(1, int(math.ceil(0.25 * float(max(1, query_keep_count))))))
        turn_scores = points[global_indices, 7].float().clamp(min=0.0)
        if bool((turn_scores > 0.0).any().item()):
            turn_positions = deterministic_topk_with_jitter(
                turn_scores,
                keep_count=turn_count,
                trajectory_id=count * 7919 + query_keep_count,
            )
            add_positions(turn_positions)

    spaced_count = min(count, max(1, int(query_keep_count)))
    add_positions(evenly_spaced_indices(count, spaced_count, global_indices.device))
    add_positions(torch.arange(count, device=global_indices.device))
    return selected


def _range_query_residual_scores(
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, object]],
    model_config: object,
    ratios: tuple[float, ...],
) -> tuple[torch.Tensor, dict[str, object]]:
    """Return budgeted train-query residual-anchor frequencies.

    For each train range query and budget, this simulates the query-blind
    temporal base, then labels only the residual anchors needed to improve
    range continuity, boundary context, turns, and shape inside the query.
    """
    n_points = int(points.shape[0])
    target = torch.zeros((n_points,), dtype=torch.float32, device=points.device)
    temporal_fraction = float(getattr(model_config, "mlqds_temporal_fraction", 0.0))
    multiplier = max(0.0, float(getattr(model_config, "range_query_residual_multiplier", 1.0) or 0.0))
    if multiplier <= 0.0:
        raise ValueError("range_query_residual_multiplier must be positive for query_residual_frequency.")
    mass_mode = str(getattr(model_config, "range_query_residual_mass_mode", "query")).lower()
    if mass_mode not in {"query", "point"}:
        raise ValueError("range_query_residual_mass_mode must be 'query' or 'point'.")

    range_query_count = 0
    used_budget_count = 0
    total_hit_group_count = 0
    total_selected_anchor_count = 0
    total_selected_residual_count = 0
    total_base_anchor_count = 0
    per_budget: list[dict[str, object]] = []
    budget_weights = _target_budget_weights(model_config, ratios)
    used_weight = 0.0

    range_queries = [
        query
        for query in typed_queries
        if str(query.get("type", "")).lower() == "range" and isinstance(query.get("params"), dict)
    ]
    range_query_count = len(range_queries)
    if range_query_count <= 0:
        return target, {
            "query_residual_range_query_count": 0,
            "query_residual_used_budget_count": 0,
            "query_residual_multiplier": float(multiplier),
            "query_residual_mass_mode": mass_mode,
        }

    for ratio, budget_weight in zip(ratios, budget_weights):
        ratio_value = min(1.0, max(0.0, float(ratio)))
        if ratio_value <= 0.0:
            continue
        used_budget_count += 1
        budget_scores = torch.zeros_like(target)
        base_mask = _temporal_base_mask_for_ratio(
            n_points=n_points,
            boundaries=boundaries,
            ratio=ratio_value,
            temporal_fraction=temporal_fraction,
            device=points.device,
        )
        base_anchor_count = int(base_mask.sum().item())
        budget_hit_group_count = 0
        budget_selected_anchor_count = 0
        budget_selected_residual_count = 0

        for query in range_queries:
            params = query["params"]
            if not isinstance(params, dict):
                raise ValueError("Range query params must be a dictionary.")
            box_support = points_in_range_box(points, params)
            query_selected: list[torch.Tensor] = []
            for start, end in boundaries:
                if end <= start:
                    continue
                in_offsets = torch.where(box_support[start:end])[0]
                count = int(in_offsets.numel())
                if count <= 0:
                    continue
                budget_hit_group_count += 1
                local_global = start + in_offsets
                local_base = base_mask[local_global]
                query_keep_count = min(
                    count,
                    max(1, int(math.ceil(float(multiplier) * ratio_value * float(count)))),
                )
                if count >= 2:
                    query_keep_count = max(2, query_keep_count)
                residual_needed = max(0, query_keep_count - int(local_base.sum().item()))
                if residual_needed <= 0:
                    continue
                priority_positions = _query_residual_priority_positions(
                    points=points,
                    global_indices=local_global,
                    local_base_mask=local_base,
                    query_keep_count=query_keep_count,
                )
                selected_positions = priority_positions[:residual_needed]
                if not selected_positions:
                    continue
                selected_global = local_global[
                    torch.tensor(selected_positions, dtype=torch.long, device=points.device)
                ]
                query_selected.append(selected_global)
                budget_selected_anchor_count += int(query_keep_count)
                budget_selected_residual_count += int(selected_global.numel())

            if query_selected:
                selected = torch.unique(torch.cat(query_selected), sorted=False)
                if selected.numel() > 0:
                    mass = 1.0 / float(selected.numel()) if mass_mode == "query" else 1.0
                    budget_scores[selected] += float(mass)

        budget_scores = budget_scores / float(range_query_count)
        target += float(budget_weight) * budget_scores
        used_weight += float(budget_weight)
        total_hit_group_count += budget_hit_group_count
        total_selected_anchor_count += budget_selected_anchor_count
        total_selected_residual_count += budget_selected_residual_count
        total_base_anchor_count += base_anchor_count
        positive = budget_scores > 0.0
        per_budget.append(
            {
                "budget_ratio": float(ratio_value),
                "budget_weight": float(budget_weight),
                "temporal_base_point_count": int(base_anchor_count),
                "hit_group_count": int(budget_hit_group_count),
                "selected_anchor_count": int(budget_selected_anchor_count),
                "selected_residual_count": int(budget_selected_residual_count),
                "positive_label_count": int(positive.sum().item()),
                "positive_label_mass": (
                    float(budget_scores[positive].sum().item()) if bool(positive.any().item()) else 0.0
                ),
            }
        )

    if used_weight > 1e-12:
        target = target / float(used_weight)
    positive = target > 0.0
    diagnostics = {
        "query_residual_range_query_count": int(range_query_count),
        "query_residual_used_budget_count": int(used_budget_count),
        "query_residual_budget_weights": list(budget_weights),
        "query_residual_budget_weight_power": float(
            getattr(model_config, "range_target_budget_weight_power", 0.0) or 0.0
        ),
        "query_residual_multiplier": float(multiplier),
        "query_residual_mass_mode": mass_mode,
        "query_residual_hit_group_count": int(total_hit_group_count),
        "query_residual_temporal_base_point_count": int(total_base_anchor_count),
        "query_residual_selected_anchor_count": int(total_selected_anchor_count),
        "query_residual_selected_residual_count": int(total_selected_residual_count),
        "query_residual_source_positive_count": int(positive.sum().item()),
        "query_residual_source_positive_mass": (
            float(target[positive].sum().item()) if bool(positive.any().item()) else 0.0
        ),
        "query_residual_per_budget": per_budget,
    }
    return target, diagnostics


def range_query_residual_frequency_training_labels(
    labels: torch.Tensor,
    labelled_mask: torch.Tensor,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, object]],
    model_config: object,
    type_idx: int = QUERY_TYPE_ID_RANGE,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Build train-query residual-anchor frequency labels for a blind student."""
    if labels.ndim != 2 or labelled_mask.shape != labels.shape:
        raise ValueError("labels and labelled_mask must have matching shape [n_points, n_types].")
    if type_idx < 0 or type_idx >= labels.shape[1]:
        raise ValueError(f"type_idx={type_idx} is outside label shape {tuple(labels.shape)}.")
    if int(points.shape[0]) != int(labels.shape[0]):
        raise ValueError("points and labels must have the same point count.")

    ratios = _target_budget_ratios(model_config)
    target, residual_diagnostics = _range_query_residual_scores(
        points=points,
        boundaries=boundaries,
        typed_queries=typed_queries,
        model_config=model_config,
        ratios=ratios,
    )
    if not bool((target > 0.0).any().item()):
        raise ValueError("query_residual_frequency target found no positive source scores.")

    transformed = labels.clone()
    transformed[:, type_idx] = target.clamp(0.0, 1.0)
    transformed_mask = labelled_mask.clone()
    transformed_mask[:, type_idx] = True
    positive = transformed[:, type_idx] > 0.0
    positive_count = int(positive.sum().item())
    diagnostics: dict[str, object] = {
        "mode": "query_residual_frequency",
        "source": "range_query_residual_anchors",
        "budget_loss_ratios": list(ratios),
        "budget_weights": list(_target_budget_weights(model_config, ratios)),
        "budget_weight_power": float(getattr(model_config, "range_target_budget_weight_power", 0.0) or 0.0),
        "labelled_point_count": int(transformed.shape[0]),
        "positive_label_count": positive_count,
        "positive_label_fraction": float(positive_count / max(1, int(transformed.shape[0]))),
        "positive_label_mass": float(transformed[positive, type_idx].sum().item()) if positive_count > 0 else 0.0,
        "mlqds_temporal_fraction": float(getattr(model_config, "mlqds_temporal_fraction", 0.0)),
        "mlqds_hybrid_mode": str(getattr(model_config, "mlqds_hybrid_mode", "fill")),
    }
    diagnostics.update(residual_diagnostics)
    return transformed, transformed_mask, diagnostics


def _range_set_utility_candidates(
    *,
    points: torch.Tensor,
    labels: torch.Tensor,
    type_idx: int,
    boundaries: list[tuple[int, int]],
    query: dict[str, object],
    base_mask: torch.Tensor,
    limit: int,
) -> torch.Tensor:
    """Return bounded train-query candidates for marginal set-utility scoring."""
    params = query.get("params")
    if not isinstance(params, dict):
        return torch.empty((0,), dtype=torch.long, device=points.device)
    range_mask = points_in_range_box(points, params)
    in_box = torch.where(range_mask)[0].to(dtype=torch.long)
    crossing = segment_box_bracket_indices(points, boundaries, params).to(device=points.device, dtype=torch.long)
    if in_box.numel() == 0 and crossing.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=points.device)
    candidates = torch.unique(torch.cat([in_box, crossing]), sorted=True)
    candidates = candidates[~base_mask[candidates]]
    if candidates.numel() <= 0:
        return candidates

    candidate_limit = int(limit)
    if candidate_limit <= 0 or int(candidates.numel()) <= candidate_limit:
        return candidates

    label_scores = labels[candidates, type_idx].float().clamp(min=0.0)
    top_count = min(int(candidates.numel()), max(1, candidate_limit // 2))
    top_local = deterministic_topk_with_jitter(
        label_scores,
        keep_count=top_count,
        trajectory_id=int(candidates.numel()) + 104729,
    )
    spaced_count = max(0, candidate_limit - top_count)
    spaced_local = evenly_spaced_indices(int(candidates.numel()), spaced_count, points.device)
    limited = torch.unique(torch.cat([candidates[top_local], candidates[spaced_local]]), sorted=True)
    if int(limited.numel()) > candidate_limit:
        limited_scores = labels[limited, type_idx].float().clamp(min=0.0)
        keep_local = deterministic_topk_with_jitter(
            limited_scores,
            keep_count=candidate_limit,
            trajectory_id=int(limited.numel()) + 1299709,
        )
        limited = torch.sort(limited[keep_local]).values
    return limited


def _range_set_utility_scores(
    points: torch.Tensor,
    labels: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, object]],
    model_config: object,
    ratios: tuple[float, ...],
    type_idx: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Return one-step marginal RangeUseful-gain targets from train queries.

    This target scores candidate residual points by the actual train-query
    RangeUseful gain from adding that single point to the query-blind temporal
    base. It is still workload-blind at inference because only the resulting
    aggregate labels are used to train point-only scoring.
    """
    n_points = int(points.shape[0])
    target = torch.zeros((n_points,), dtype=torch.float32, device=points.device)
    temporal_fraction = float(getattr(model_config, "mlqds_temporal_fraction", 0.0))
    multiplier = max(0.0, float(getattr(model_config, "range_set_utility_multiplier", 1.0) or 0.0))
    if multiplier <= 0.0:
        raise ValueError("range_set_utility_multiplier must be positive for set_utility_frequency.")
    mass_mode = str(getattr(model_config, "range_set_utility_mass_mode", "gain")).lower()
    if mass_mode not in {"gain", "point", "query"}:
        raise ValueError("range_set_utility_mass_mode must be 'gain', 'point', or 'query'.")
    candidate_limit = int(getattr(model_config, "range_set_utility_candidate_limit", 128) or 0)

    range_queries = [
        query
        for query in typed_queries
        if str(query.get("type", "")).lower() == "range" and isinstance(query.get("params"), dict)
    ]
    range_query_count = len(range_queries)
    if range_query_count <= 0:
        return target, {
            "set_utility_range_query_count": 0,
            "set_utility_used_budget_count": 0,
            "set_utility_multiplier": float(multiplier),
            "set_utility_mass_mode": mass_mode,
            "set_utility_candidate_limit": int(candidate_limit),
        }

    used_budget_count = 0
    budget_weights = _target_budget_weights(model_config, ratios)
    used_weight = 0.0
    total_candidate_count = 0
    total_scored_candidate_count = 0
    total_selected_count = 0
    total_positive_gain_count = 0
    total_gain_mass = 0.0
    per_budget: list[dict[str, object]] = []

    for ratio, budget_weight in zip(ratios, budget_weights):
        ratio_value = min(1.0, max(0.0, float(ratio)))
        if ratio_value <= 0.0:
            continue
        used_budget_count += 1
        base_mask = _temporal_base_mask_for_ratio(
            n_points=n_points,
            boundaries=boundaries,
            ratio=ratio_value,
            temporal_fraction=temporal_fraction,
            device=points.device,
        )
        budget_scores = torch.zeros_like(target)
        budget_candidate_count = 0
        budget_scored_candidate_count = 0
        budget_selected_count = 0
        budget_positive_gain_count = 0
        budget_gain_mass = 0.0

        for query_index, query in enumerate(range_queries):
            params = query.get("params")
            if not isinstance(params, dict):
                continue
            range_mask = points_in_range_box(points, params)
            hit_count = int(range_mask.sum().item())
            if hit_count <= 0:
                continue
            desired_count = min(
                hit_count,
                max(1, int(math.ceil(multiplier * ratio_value * float(hit_count)))),
            )
            if hit_count >= 2:
                desired_count = max(2, desired_count)
            base_hit_count = int((base_mask & range_mask).sum().item())
            residual_needed = max(0, desired_count - base_hit_count)
            if residual_needed <= 0:
                continue

            candidates = _range_set_utility_candidates(
                points=points,
                labels=labels,
                type_idx=type_idx,
                boundaries=boundaries,
                query=query,
                base_mask=base_mask,
                limit=candidate_limit,
            )
            budget_candidate_count += int(candidates.numel())
            if candidates.numel() <= 0:
                continue

            query_list = [query]
            query_cache = EvaluationQueryCache.for_workload(points, boundaries, query_list)
            retained = base_mask.clone()
            base_score = float(
                score_range_usefulness(
                    points=points,
                    boundaries=boundaries,
                    retained_mask=retained,
                    typed_queries=query_list,
                    query_cache=query_cache,
                )["range_usefulness_score"]
            )
            gains = torch.zeros((int(candidates.numel()),), dtype=torch.float32, device=points.device)
            for candidate_pos, candidate_idx_tensor in enumerate(candidates):
                candidate_idx = int(candidate_idx_tensor.item())
                retained[candidate_idx] = True
                score = float(
                    score_range_usefulness(
                        points=points,
                        boundaries=boundaries,
                        retained_mask=retained,
                        typed_queries=query_list,
                        query_cache=query_cache,
                    )["range_usefulness_score"]
                )
                retained[candidate_idx] = False
                gains[candidate_pos] = max(0.0, score - base_score)

            positive_gain = gains > 1e-12
            positive_gain_count = int(positive_gain.sum().item())
            budget_scored_candidate_count += int(candidates.numel())
            budget_positive_gain_count += positive_gain_count
            if positive_gain_count <= 0:
                continue

            keep_count = min(int(residual_needed), positive_gain_count)
            positive_local = torch.where(positive_gain)[0]
            selected_local_in_positive = deterministic_topk_with_jitter(
                gains[positive_local],
                keep_count=keep_count,
                trajectory_id=(query_index + 1) * 1009 + int(ratio_value * 10000),
            )
            selected_local = positive_local[selected_local_in_positive]
            selected_indices = candidates[selected_local]
            selected_gains = gains[selected_local]
            selected_gain_mass = float(selected_gains.sum().item())
            budget_selected_count += int(selected_indices.numel())
            budget_gain_mass += selected_gain_mass

            if mass_mode == "gain":
                budget_scores[selected_indices] += selected_gains
            elif mass_mode == "point":
                budget_scores[selected_indices] += 1.0
            else:
                query_weight = 1.0 / float(max(1, int(selected_indices.numel())))
                budget_scores[selected_indices] += float(query_weight)

        budget_scores = budget_scores / float(range_query_count)
        target += float(budget_weight) * budget_scores
        used_weight += float(budget_weight)
        total_candidate_count += budget_candidate_count
        total_scored_candidate_count += budget_scored_candidate_count
        total_selected_count += budget_selected_count
        total_positive_gain_count += budget_positive_gain_count
        total_gain_mass += budget_gain_mass
        positive = budget_scores > 0.0
        per_budget.append(
            {
                "budget_ratio": float(ratio_value),
                "budget_weight": float(budget_weight),
                "temporal_base_point_count": int(base_mask.sum().item()),
                "candidate_count": int(budget_candidate_count),
                "scored_candidate_count": int(budget_scored_candidate_count),
                "positive_gain_candidate_count": int(budget_positive_gain_count),
                "selected_count": int(budget_selected_count),
                "selected_gain_mass": float(budget_gain_mass),
                "positive_label_count": int(positive.sum().item()),
                "positive_label_mass": (
                    float(budget_scores[positive].sum().item()) if bool(positive.any().item()) else 0.0
                ),
            }
        )

    if used_weight > 1e-12:
        target = target / float(used_weight)
    positive = target > 0.0
    diagnostics = {
        "set_utility_range_query_count": int(range_query_count),
        "set_utility_used_budget_count": int(used_budget_count),
        "set_utility_budget_weights": list(budget_weights),
        "set_utility_budget_weight_power": float(
            getattr(model_config, "range_target_budget_weight_power", 0.0) or 0.0
        ),
        "set_utility_multiplier": float(multiplier),
        "set_utility_mass_mode": mass_mode,
        "set_utility_candidate_limit": int(candidate_limit),
        "set_utility_candidate_count": int(total_candidate_count),
        "set_utility_scored_candidate_count": int(total_scored_candidate_count),
        "set_utility_positive_gain_candidate_count": int(total_positive_gain_count),
        "set_utility_selected_count": int(total_selected_count),
        "set_utility_selected_gain_mass": float(total_gain_mass),
        "set_utility_source_positive_count": int(positive.sum().item()),
        "set_utility_source_positive_mass": (
            float(target[positive].sum().item()) if bool(positive.any().item()) else 0.0
        ),
        "set_utility_per_budget": per_budget,
    }
    return target, diagnostics


def range_set_utility_frequency_training_labels(
    labels: torch.Tensor,
    labelled_mask: torch.Tensor,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, object]],
    model_config: object,
    type_idx: int = QUERY_TYPE_ID_RANGE,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Build one-step marginal RangeUseful-gain labels for a blind student."""
    if labels.ndim != 2 or labelled_mask.shape != labels.shape:
        raise ValueError("labels and labelled_mask must have matching shape [n_points, n_types].")
    if type_idx < 0 or type_idx >= labels.shape[1]:
        raise ValueError(f"type_idx={type_idx} is outside label shape {tuple(labels.shape)}.")
    if int(points.shape[0]) != int(labels.shape[0]):
        raise ValueError("points and labels must have the same point count.")

    ratios = _target_budget_ratios(model_config)
    target, utility_diagnostics = _range_set_utility_scores(
        points=points,
        labels=labels,
        boundaries=boundaries,
        typed_queries=typed_queries,
        model_config=model_config,
        ratios=ratios,
        type_idx=type_idx,
    )
    if not bool((target > 0.0).any().item()):
        raise ValueError("set_utility_frequency target found no positive source scores.")

    transformed = labels.clone()
    transformed[:, type_idx] = target.clamp(0.0, 1.0)
    transformed_mask = labelled_mask.clone()
    transformed_mask[:, type_idx] = True
    positive = transformed[:, type_idx] > 0.0
    positive_count = int(positive.sum().item())
    diagnostics: dict[str, object] = {
        "mode": "set_utility_frequency",
        "source": "range_train_query_marginal_usefulness_gain",
        "budget_loss_ratios": list(ratios),
        "budget_weights": list(_target_budget_weights(model_config, ratios)),
        "budget_weight_power": float(getattr(model_config, "range_target_budget_weight_power", 0.0) or 0.0),
        "labelled_point_count": int(transformed.shape[0]),
        "positive_label_count": positive_count,
        "positive_label_fraction": float(positive_count / max(1, int(transformed.shape[0]))),
        "positive_label_mass": float(transformed[positive, type_idx].sum().item()) if positive_count > 0 else 0.0,
        "mlqds_temporal_fraction": float(getattr(model_config, "mlqds_temporal_fraction", 0.0)),
        "mlqds_hybrid_mode": str(getattr(model_config, "mlqds_hybrid_mode", "fill")),
    }
    diagnostics.update(utility_diagnostics)
    return transformed, transformed_mask, diagnostics


def _local_swap_base_plan(
    *,
    n_points: int,
    boundaries: list[tuple[int, int]],
    ratio: float,
    temporal_fraction: float,
    device: torch.device,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, int]:
    """Return full temporal base plus removable points for local-swap targets."""
    base_mask = torch.zeros((int(n_points),), dtype=torch.bool, device=device)
    trajectory_ids = torch.full((int(n_points),), -1, dtype=torch.long, device=device)
    removable_by_trajectory: list[torch.Tensor] = []
    base_fraction = min(1.0, max(0.0, float(temporal_fraction)))
    total_capacity = 0

    for trajectory_id, (start, end) in enumerate(boundaries):
        point_count = int(end - start)
        trajectory_ids[start:end] = int(trajectory_id)
        if point_count <= 0:
            removable_by_trajectory.append(torch.empty((0,), dtype=torch.long, device=device))
            continue
        keep_count = min(point_count, max(2, int(math.ceil(float(ratio) * point_count))))
        base_indices = evenly_spaced_indices(point_count, keep_count, device)
        base_mask[start + base_indices] = True
        protected_count = min(keep_count, max(2, int(math.ceil(keep_count * base_fraction))))
        swap_count = min(keep_count - protected_count, point_count - keep_count)
        removable_local = base_indices[(base_indices != 0) & (base_indices != point_count - 1)]
        swap_count = min(max(0, int(swap_count)), int(removable_local.numel()))
        removable_global = start + removable_local
        if swap_count <= 0:
            removable_global = removable_global[:0]
        removable_by_trajectory.append(removable_global.to(dtype=torch.long))
        total_capacity += int(swap_count)

    return base_mask, removable_by_trajectory, trajectory_ids, int(total_capacity)


def _nearest_local_swap_removal(
    candidate_idx: int,
    *,
    trajectory_ids: torch.Tensor,
    removable_by_trajectory: list[torch.Tensor],
) -> int | None:
    """Return the nearest removable temporal-base point for one candidate."""
    if candidate_idx < 0 or candidate_idx >= int(trajectory_ids.numel()):
        return None
    trajectory_id = int(trajectory_ids[candidate_idx].item())
    if trajectory_id < 0 or trajectory_id >= len(removable_by_trajectory):
        return None
    removable = removable_by_trajectory[trajectory_id]
    if int(removable.numel()) <= 0:
        return None
    distances = torch.abs(removable.to(dtype=torch.long) - int(candidate_idx))
    best = torch.argmin(distances)
    return int(removable[best].item())


def _range_local_swap_utility_scores(
    points: torch.Tensor,
    labels: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, object]],
    model_config: object,
    ratios: tuple[float, ...],
    type_idx: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Return train-query utility labels for the actual local-swap action."""
    hybrid_mode = str(getattr(model_config, "mlqds_hybrid_mode", "fill")).lower()
    if hybrid_mode not in {"local_swap", "local_delta_swap"}:
        raise ValueError(
            "local_swap_utility_frequency requires mlqds_hybrid_mode='local_swap' or 'local_delta_swap'."
        )

    n_points = int(points.shape[0])
    target = torch.zeros((n_points,), dtype=torch.float32, device=points.device)
    temporal_fraction = float(getattr(model_config, "mlqds_temporal_fraction", 0.0))
    multiplier = max(0.0, float(getattr(model_config, "range_set_utility_multiplier", 1.0) or 0.0))
    if multiplier <= 0.0:
        raise ValueError("range_set_utility_multiplier must be positive for local_swap_utility_frequency.")
    mass_mode = str(getattr(model_config, "range_set_utility_mass_mode", "gain")).lower()
    if mass_mode not in {"gain", "point", "query"}:
        raise ValueError("range_set_utility_mass_mode must be 'gain', 'point', or 'query'.")
    candidate_limit = int(getattr(model_config, "range_set_utility_candidate_limit", 128) or 0)

    range_queries = [
        query
        for query in typed_queries
        if str(query.get("type", "")).lower() == "range" and isinstance(query.get("params"), dict)
    ]
    range_query_count = len(range_queries)
    if range_query_count <= 0:
        return target, {
            "local_swap_utility_range_query_count": 0,
            "local_swap_utility_used_budget_count": 0,
            "local_swap_utility_multiplier": float(multiplier),
            "local_swap_utility_mass_mode": mass_mode,
            "local_swap_utility_candidate_limit": int(candidate_limit),
        }

    used_budget_count = 0
    budget_weights = _target_budget_weights(model_config, ratios)
    used_weight = 0.0
    total_candidate_count = 0
    total_scored_candidate_count = 0
    total_selected_count = 0
    total_positive_gain_count = 0
    total_gain_mass = 0.0
    per_budget: list[dict[str, object]] = []

    for ratio, budget_weight in zip(ratios, budget_weights):
        ratio_value = min(1.0, max(0.0, float(ratio)))
        if ratio_value <= 0.0:
            continue
        base_mask, removable_by_trajectory, trajectory_ids, swap_capacity = _local_swap_base_plan(
            n_points=n_points,
            boundaries=boundaries,
            ratio=ratio_value,
            temporal_fraction=temporal_fraction,
            device=points.device,
        )
        if swap_capacity <= 0:
            per_budget.append(
                {
                    "budget_ratio": float(ratio_value),
                    "budget_weight": float(budget_weight),
                    "swap_capacity": 0,
                    "candidate_count": 0,
                    "scored_candidate_count": 0,
                    "positive_gain_candidate_count": 0,
                    "selected_count": 0,
                    "selected_gain_mass": 0.0,
                    "positive_label_count": 0,
                    "positive_label_mass": 0.0,
                }
            )
            continue
        used_budget_count += 1
        budget_scores = torch.zeros_like(target)
        budget_candidate_count = 0
        budget_scored_candidate_count = 0
        budget_selected_count = 0
        budget_positive_gain_count = 0
        budget_gain_mass = 0.0

        for query_index, query in enumerate(range_queries):
            params = query.get("params")
            if not isinstance(params, dict):
                continue
            range_mask = points_in_range_box(points, params)
            hit_count = int(range_mask.sum().item())
            if hit_count <= 0:
                continue
            candidates = _range_set_utility_candidates(
                points=points,
                labels=labels,
                type_idx=type_idx,
                boundaries=boundaries,
                query=query,
                base_mask=base_mask,
                limit=candidate_limit,
            )
            budget_candidate_count += int(candidates.numel())
            if int(candidates.numel()) <= 0:
                continue

            query_list = [query]
            query_cache = EvaluationQueryCache.for_workload(points, boundaries, query_list)
            base_score = float(
                score_range_usefulness(
                    points=points,
                    boundaries=boundaries,
                    retained_mask=base_mask,
                    typed_queries=query_list,
                    query_cache=query_cache,
                )["range_usefulness_score"]
            )
            scored_indices: list[int] = []
            gains: list[float] = []
            retained = base_mask.clone()
            for candidate_idx_tensor in candidates:
                candidate_idx = int(candidate_idx_tensor.item())
                remove_idx = _nearest_local_swap_removal(
                    candidate_idx,
                    trajectory_ids=trajectory_ids,
                    removable_by_trajectory=removable_by_trajectory,
                )
                if remove_idx is None or remove_idx == candidate_idx:
                    continue
                retained[remove_idx] = False
                retained[candidate_idx] = True
                score = float(
                    score_range_usefulness(
                        points=points,
                        boundaries=boundaries,
                        retained_mask=retained,
                        typed_queries=query_list,
                        query_cache=query_cache,
                    )["range_usefulness_score"]
                )
                retained[candidate_idx] = bool(base_mask[candidate_idx].item())
                retained[remove_idx] = True
                gain = max(0.0, score - base_score)
                scored_indices.append(candidate_idx)
                gains.append(gain)

            budget_scored_candidate_count += len(scored_indices)
            if not scored_indices:
                continue
            gain_tensor = torch.tensor(gains, dtype=torch.float32, device=points.device)
            positive_gain = gain_tensor > 1e-12
            positive_gain_count = int(positive_gain.sum().item())
            budget_positive_gain_count += positive_gain_count
            if positive_gain_count <= 0:
                continue
            desired_count = min(
                positive_gain_count,
                max(1, int(math.ceil(multiplier * ratio_value * float(hit_count)))),
                int(swap_capacity),
            )
            positive_local = torch.where(positive_gain)[0]
            selected_local_in_positive = deterministic_topk_with_jitter(
                gain_tensor[positive_local],
                keep_count=desired_count,
                trajectory_id=(query_index + 1) * 9176 + int(ratio_value * 10000),
            )
            selected_local = positive_local[selected_local_in_positive]
            selected_indices = torch.tensor(
                [scored_indices[int(local.item())] for local in selected_local],
                dtype=torch.long,
                device=points.device,
            )
            selected_gains = gain_tensor[selected_local]
            selected_gain_mass = float(selected_gains.sum().item())
            budget_selected_count += int(selected_indices.numel())
            budget_gain_mass += selected_gain_mass
            if mass_mode == "gain":
                budget_scores[selected_indices] += selected_gains
            elif mass_mode == "point":
                budget_scores[selected_indices] += 1.0
            else:
                budget_scores[selected_indices] += 1.0 / float(max(1, int(selected_indices.numel())))

        budget_scores = budget_scores / float(range_query_count)
        target += float(budget_weight) * budget_scores
        used_weight += float(budget_weight)
        total_candidate_count += budget_candidate_count
        total_scored_candidate_count += budget_scored_candidate_count
        total_selected_count += budget_selected_count
        total_positive_gain_count += budget_positive_gain_count
        total_gain_mass += budget_gain_mass
        positive = budget_scores > 0.0
        per_budget.append(
            {
                "budget_ratio": float(ratio_value),
                "budget_weight": float(budget_weight),
                "swap_capacity": int(swap_capacity),
                "candidate_count": int(budget_candidate_count),
                "scored_candidate_count": int(budget_scored_candidate_count),
                "positive_gain_candidate_count": int(budget_positive_gain_count),
                "selected_count": int(budget_selected_count),
                "selected_gain_mass": float(budget_gain_mass),
                "positive_label_count": int(positive.sum().item()),
                "positive_label_mass": (
                    float(budget_scores[positive].sum().item()) if bool(positive.any().item()) else 0.0
                ),
            }
        )

    if used_weight > 1e-12:
        target = target / float(used_weight)
    positive = target > 0.0
    diagnostics = {
        "local_swap_utility_range_query_count": int(range_query_count),
        "local_swap_utility_used_budget_count": int(used_budget_count),
        "local_swap_utility_budget_weights": list(budget_weights),
        "local_swap_utility_budget_weight_power": float(
            getattr(model_config, "range_target_budget_weight_power", 0.0) or 0.0
        ),
        "local_swap_utility_multiplier": float(multiplier),
        "local_swap_utility_mass_mode": mass_mode,
        "local_swap_utility_candidate_limit": int(candidate_limit),
        "local_swap_utility_candidate_count": int(total_candidate_count),
        "local_swap_utility_scored_candidate_count": int(total_scored_candidate_count),
        "local_swap_utility_positive_gain_candidate_count": int(total_positive_gain_count),
        "local_swap_utility_selected_count": int(total_selected_count),
        "local_swap_utility_selected_gain_mass": float(total_gain_mass),
        "local_swap_utility_source_positive_count": int(positive.sum().item()),
        "local_swap_utility_source_positive_mass": (
            float(target[positive].sum().item()) if bool(positive.any().item()) else 0.0
        ),
        "local_swap_utility_per_budget": per_budget,
    }
    return target, diagnostics


def range_local_swap_utility_frequency_training_labels(
    labels: torch.Tensor,
    labelled_mask: torch.Tensor,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, object]],
    model_config: object,
    type_idx: int = QUERY_TYPE_ID_RANGE,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Build train-query labels from positive local-swap RangeUseful gains."""
    if labels.ndim != 2 or labelled_mask.shape != labels.shape:
        raise ValueError("labels and labelled_mask must have matching shape [n_points, n_types].")
    if type_idx < 0 or type_idx >= labels.shape[1]:
        raise ValueError(f"type_idx={type_idx} is outside label shape {tuple(labels.shape)}.")
    if int(points.shape[0]) != int(labels.shape[0]):
        raise ValueError("points and labels must have the same point count.")

    ratios = _target_budget_ratios(model_config)
    target, utility_diagnostics = _range_local_swap_utility_scores(
        points=points,
        labels=labels,
        boundaries=boundaries,
        typed_queries=typed_queries,
        model_config=model_config,
        ratios=ratios,
        type_idx=type_idx,
    )
    if not bool((target > 0.0).any().item()):
        raise ValueError("local_swap_utility_frequency target found no positive source scores.")

    transformed = labels.clone()
    transformed[:, type_idx] = target.clamp(0.0, 1.0)
    transformed_mask = labelled_mask.clone()
    transformed_mask[:, type_idx] = True
    positive = transformed[:, type_idx] > 0.0
    positive_count = int(positive.sum().item())
    diagnostics: dict[str, object] = {
        "mode": "local_swap_utility_frequency",
        "source": "range_train_query_local_swap_usefulness_gain",
        "budget_loss_ratios": list(ratios),
        "budget_weights": list(_target_budget_weights(model_config, ratios)),
        "budget_weight_power": float(getattr(model_config, "range_target_budget_weight_power", 0.0) or 0.0),
        "labelled_point_count": int(transformed.shape[0]),
        "positive_label_count": positive_count,
        "positive_label_fraction": float(positive_count / max(1, int(transformed.shape[0]))),
        "positive_label_mass": float(transformed[positive, type_idx].sum().item()) if positive_count > 0 else 0.0,
        "mlqds_temporal_fraction": float(getattr(model_config, "mlqds_temporal_fraction", 0.0)),
        "mlqds_hybrid_mode": str(getattr(model_config, "mlqds_hybrid_mode", "fill")),
    }
    diagnostics.update(utility_diagnostics)
    return transformed, transformed_mask, diagnostics


def _range_local_swap_gain_cost_scores(
    points: torch.Tensor,
    labels: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, object]],
    model_config: object,
    ratios: tuple[float, ...],
    type_idx: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Return paired add-value/removal-cost labels for local-delta swaps.

    For a candidate paired with its nearest removable temporal anchor:

    - candidate value = score(base - anchor + candidate) - score(base - anchor)
    - anchor cost = score(base) - score(base - anchor)

    The local-delta selector accepts a replacement when candidate score exceeds
    the paired anchor score, so these labels directly encode the desired gate:
    candidate value > anchor cost iff the exact one-step replacement improves
    train-query RangeUseful.
    """
    hybrid_mode = str(getattr(model_config, "mlqds_hybrid_mode", "fill")).lower()
    if hybrid_mode != "local_delta_swap":
        raise ValueError("local_swap_gain_cost_frequency requires mlqds_hybrid_mode='local_delta_swap'.")

    n_points = int(points.shape[0])
    target = torch.zeros((n_points,), dtype=torch.float32, device=points.device)
    temporal_fraction = float(getattr(model_config, "mlqds_temporal_fraction", 0.0))
    multiplier = max(0.0, float(getattr(model_config, "range_set_utility_multiplier", 1.0) or 0.0))
    if multiplier <= 0.0:
        raise ValueError("range_set_utility_multiplier must be positive for local_swap_gain_cost_frequency.")
    mass_mode = str(getattr(model_config, "range_set_utility_mass_mode", "gain")).lower()
    if mass_mode not in {"gain", "point", "query"}:
        raise ValueError("range_set_utility_mass_mode must be 'gain', 'point', or 'query'.")
    candidate_limit = int(getattr(model_config, "range_set_utility_candidate_limit", 128) or 0)

    range_queries = [
        query
        for query in typed_queries
        if str(query.get("type", "")).lower() == "range" and isinstance(query.get("params"), dict)
    ]
    range_query_count = len(range_queries)
    if range_query_count <= 0:
        return target, {
            "local_swap_gain_cost_range_query_count": 0,
            "local_swap_gain_cost_used_budget_count": 0,
            "local_swap_gain_cost_multiplier": float(multiplier),
            "local_swap_gain_cost_mass_mode": mass_mode,
            "local_swap_gain_cost_candidate_limit": int(candidate_limit),
        }

    used_budget_count = 0
    budget_weights = _target_budget_weights(model_config, ratios)
    used_weight = 0.0
    total_candidate_count = 0
    total_scored_candidate_count = 0
    total_positive_net_gain_count = 0
    total_selected_count = 0
    total_candidate_value_mass = 0.0
    total_removal_cost_mass = 0.0
    per_budget: list[dict[str, object]] = []

    for ratio, budget_weight in zip(ratios, budget_weights):
        ratio_value = min(1.0, max(0.0, float(ratio)))
        if ratio_value <= 0.0:
            continue
        base_mask, removable_by_trajectory, trajectory_ids, swap_capacity = _local_swap_base_plan(
            n_points=n_points,
            boundaries=boundaries,
            ratio=ratio_value,
            temporal_fraction=temporal_fraction,
            device=points.device,
        )
        if swap_capacity <= 0:
            per_budget.append(
                {
                    "budget_ratio": float(ratio_value),
                    "budget_weight": float(budget_weight),
                    "swap_capacity": 0,
                    "candidate_count": 0,
                    "scored_candidate_count": 0,
                    "positive_net_gain_count": 0,
                    "selected_count": 0,
                    "selected_candidate_value_mass": 0.0,
                    "selected_removal_cost_mass": 0.0,
                    "positive_label_count": 0,
                    "positive_label_mass": 0.0,
                }
            )
            continue

        used_budget_count += 1
        budget_scores = torch.zeros_like(target)
        budget_candidate_count = 0
        budget_scored_candidate_count = 0
        budget_positive_net_gain_count = 0
        budget_selected_count = 0
        budget_candidate_value_mass = 0.0
        budget_removal_cost_mass = 0.0

        for query_index, query in enumerate(range_queries):
            params = query.get("params")
            if not isinstance(params, dict):
                continue
            range_mask = points_in_range_box(points, params)
            hit_count = int(range_mask.sum().item())
            if hit_count <= 0:
                continue
            candidates = _range_set_utility_candidates(
                points=points,
                labels=labels,
                type_idx=type_idx,
                boundaries=boundaries,
                query=query,
                base_mask=base_mask,
                limit=candidate_limit,
            )
            budget_candidate_count += int(candidates.numel())
            if int(candidates.numel()) <= 0:
                continue

            query_list = [query]
            query_cache = EvaluationQueryCache.for_workload(points, boundaries, query_list)
            base_score = float(
                score_range_usefulness(
                    points=points,
                    boundaries=boundaries,
                    retained_mask=base_mask,
                    typed_queries=query_list,
                    query_cache=query_cache,
                )["range_usefulness_score"]
            )
            removal_score_cache: dict[int, float] = {}
            removal_cost_cache: dict[int, float] = {}
            scored_records: list[tuple[int, int, float, float, float]] = []

            for candidate_idx_tensor in candidates:
                candidate_idx = int(candidate_idx_tensor.item())
                remove_idx = _nearest_local_swap_removal(
                    candidate_idx,
                    trajectory_ids=trajectory_ids,
                    removable_by_trajectory=removable_by_trajectory,
                )
                if remove_idx is None or remove_idx == candidate_idx:
                    continue
                if remove_idx not in removal_score_cache:
                    retained_without = base_mask.clone()
                    retained_without[remove_idx] = False
                    removal_score = float(
                        score_range_usefulness(
                            points=points,
                            boundaries=boundaries,
                            retained_mask=retained_without,
                            typed_queries=query_list,
                            query_cache=query_cache,
                        )["range_usefulness_score"]
                    )
                    removal_score_cache[remove_idx] = removal_score
                    removal_cost_cache[remove_idx] = max(0.0, base_score - removal_score)
                removal_score = removal_score_cache[remove_idx]
                retained_replacement = base_mask.clone()
                retained_replacement[remove_idx] = False
                retained_replacement[candidate_idx] = True
                replacement_score = float(
                    score_range_usefulness(
                        points=points,
                        boundaries=boundaries,
                        retained_mask=retained_replacement,
                        typed_queries=query_list,
                        query_cache=query_cache,
                    )["range_usefulness_score"]
                )
                candidate_value = max(0.0, replacement_score - removal_score)
                removal_cost = removal_cost_cache[remove_idx]
                net_gain = replacement_score - base_score
                scored_records.append((candidate_idx, remove_idx, net_gain, candidate_value, removal_cost))

            budget_scored_candidate_count += len(scored_records)
            if not scored_records:
                continue
            net_gains = torch.tensor(
                [record[2] for record in scored_records],
                dtype=torch.float32,
                device=points.device,
            )
            positive_net_gain = net_gains > 1e-12
            positive_net_gain_count = int(positive_net_gain.sum().item())
            budget_positive_net_gain_count += positive_net_gain_count
            if positive_net_gain_count <= 0:
                continue
            desired_count = min(
                positive_net_gain_count,
                max(1, int(math.ceil(multiplier * ratio_value * float(hit_count)))),
                int(swap_capacity),
            )
            positive_local = torch.where(positive_net_gain)[0]
            tie_positions = torch.arange(
                int(positive_local.numel()),
                dtype=torch.float32,
                device=points.device,
            )
            tie_jitter = 1e-6 * torch.sin(
                tie_positions * 12.9898 + float((query_index + 1) * 9176 + int(ratio_value * 10000))
            )
            ordered_positive = positive_local[torch.argsort(net_gains[positive_local] + tie_jitter, descending=True)]

            selected_records: list[tuple[int, int, float, float, float]] = []
            used_removals: set[int] = set()
            for local_idx_tensor in ordered_positive:
                record = scored_records[int(local_idx_tensor.item())]
                remove_idx = int(record[1])
                if remove_idx in used_removals:
                    continue
                selected_records.append(record)
                used_removals.add(remove_idx)
                if len(selected_records) >= desired_count:
                    break
            if not selected_records:
                continue

            selected_count = len(selected_records)
            budget_selected_count += selected_count
            selected_candidate_value_mass = sum(float(record[3]) for record in selected_records)
            selected_removal_cost_by_idx: dict[int, float] = {}
            for _candidate_idx, remove_idx, _net_gain, _candidate_value, removal_cost in selected_records:
                selected_removal_cost_by_idx[int(remove_idx)] = max(
                    selected_removal_cost_by_idx.get(int(remove_idx), 0.0),
                    float(removal_cost),
                )
            selected_removal_cost_mass = sum(selected_removal_cost_by_idx.values())
            budget_candidate_value_mass += selected_candidate_value_mass
            budget_removal_cost_mass += selected_removal_cost_mass

            if mass_mode == "gain":
                for candidate_idx, _remove_idx, _net_gain, candidate_value, _removal_cost in selected_records:
                    budget_scores[int(candidate_idx)] += float(candidate_value)
                for remove_idx, removal_cost in selected_removal_cost_by_idx.items():
                    budget_scores[int(remove_idx)] += float(removal_cost)
            elif mass_mode == "point":
                for candidate_idx, _remove_idx, _net_gain, _candidate_value, _removal_cost in selected_records:
                    budget_scores[int(candidate_idx)] += 1.0
                for remove_idx, removal_cost in selected_removal_cost_by_idx.items():
                    if removal_cost > 0.0:
                        budget_scores[int(remove_idx)] += 1.0
            else:
                candidate_mass = 1.0 / float(max(1, selected_count))
                removal_mass = 1.0 / float(max(1, len(selected_removal_cost_by_idx)))
                for candidate_idx, _remove_idx, _net_gain, _candidate_value, _removal_cost in selected_records:
                    budget_scores[int(candidate_idx)] += candidate_mass
                for remove_idx, removal_cost in selected_removal_cost_by_idx.items():
                    if removal_cost > 0.0:
                        budget_scores[int(remove_idx)] += removal_mass

        budget_scores = budget_scores / float(range_query_count)
        target += float(budget_weight) * budget_scores
        used_weight += float(budget_weight)
        total_candidate_count += budget_candidate_count
        total_scored_candidate_count += budget_scored_candidate_count
        total_positive_net_gain_count += budget_positive_net_gain_count
        total_selected_count += budget_selected_count
        total_candidate_value_mass += budget_candidate_value_mass
        total_removal_cost_mass += budget_removal_cost_mass
        positive = budget_scores > 0.0
        per_budget.append(
            {
                "budget_ratio": float(ratio_value),
                "budget_weight": float(budget_weight),
                "swap_capacity": int(swap_capacity),
                "candidate_count": int(budget_candidate_count),
                "scored_candidate_count": int(budget_scored_candidate_count),
                "positive_net_gain_count": int(budget_positive_net_gain_count),
                "selected_count": int(budget_selected_count),
                "selected_candidate_value_mass": float(budget_candidate_value_mass),
                "selected_removal_cost_mass": float(budget_removal_cost_mass),
                "positive_label_count": int(positive.sum().item()),
                "positive_label_mass": (
                    float(budget_scores[positive].sum().item()) if bool(positive.any().item()) else 0.0
                ),
            }
        )

    if used_weight > 1e-12:
        target = target / float(used_weight)
    positive = target > 0.0
    diagnostics = {
        "local_swap_gain_cost_range_query_count": int(range_query_count),
        "local_swap_gain_cost_used_budget_count": int(used_budget_count),
        "local_swap_gain_cost_budget_weights": list(budget_weights),
        "local_swap_gain_cost_budget_weight_power": float(
            getattr(model_config, "range_target_budget_weight_power", 0.0) or 0.0
        ),
        "local_swap_gain_cost_multiplier": float(multiplier),
        "local_swap_gain_cost_mass_mode": mass_mode,
        "local_swap_gain_cost_candidate_limit": int(candidate_limit),
        "local_swap_gain_cost_candidate_count": int(total_candidate_count),
        "local_swap_gain_cost_scored_candidate_count": int(total_scored_candidate_count),
        "local_swap_gain_cost_positive_net_gain_count": int(total_positive_net_gain_count),
        "local_swap_gain_cost_selected_count": int(total_selected_count),
        "local_swap_gain_cost_selected_candidate_value_mass": float(total_candidate_value_mass),
        "local_swap_gain_cost_selected_removal_cost_mass": float(total_removal_cost_mass),
        "local_swap_gain_cost_source_positive_count": int(positive.sum().item()),
        "local_swap_gain_cost_source_positive_mass": (
            float(target[positive].sum().item()) if bool(positive.any().item()) else 0.0
        ),
        "local_swap_gain_cost_per_budget": per_budget,
    }
    return target, diagnostics


def range_local_swap_gain_cost_frequency_training_labels(
    labels: torch.Tensor,
    labelled_mask: torch.Tensor,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, object]],
    model_config: object,
    type_idx: int = QUERY_TYPE_ID_RANGE,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Build train-query labels for local-delta candidate value and base cost."""
    if labels.ndim != 2 or labelled_mask.shape != labels.shape:
        raise ValueError("labels and labelled_mask must have matching shape [n_points, n_types].")
    if type_idx < 0 or type_idx >= labels.shape[1]:
        raise ValueError(f"type_idx={type_idx} is outside label shape {tuple(labels.shape)}.")
    if int(points.shape[0]) != int(labels.shape[0]):
        raise ValueError("points and labels must have the same point count.")

    ratios = _target_budget_ratios(model_config)
    target, gain_cost_diagnostics = _range_local_swap_gain_cost_scores(
        points=points,
        labels=labels,
        boundaries=boundaries,
        typed_queries=typed_queries,
        model_config=model_config,
        ratios=ratios,
        type_idx=type_idx,
    )
    if not bool((target > 0.0).any().item()):
        raise ValueError("local_swap_gain_cost_frequency target found no positive source scores.")

    transformed = labels.clone()
    transformed[:, type_idx] = target.clamp(0.0, 1.0)
    transformed_mask = labelled_mask.clone()
    transformed_mask[:, type_idx] = True
    positive = transformed[:, type_idx] > 0.0
    positive_count = int(positive.sum().item())
    diagnostics: dict[str, object] = {
        "mode": "local_swap_gain_cost_frequency",
        "source": "range_train_query_local_swap_candidate_value_and_removal_cost",
        "budget_loss_ratios": list(ratios),
        "budget_weights": list(_target_budget_weights(model_config, ratios)),
        "budget_weight_power": float(getattr(model_config, "range_target_budget_weight_power", 0.0) or 0.0),
        "labelled_point_count": int(transformed.shape[0]),
        "positive_label_count": positive_count,
        "positive_label_fraction": float(positive_count / max(1, int(transformed.shape[0]))),
        "positive_label_mass": float(transformed[positive, type_idx].sum().item()) if positive_count > 0 else 0.0,
        "mlqds_temporal_fraction": float(getattr(model_config, "mlqds_temporal_fraction", 0.0)),
        "mlqds_hybrid_mode": str(getattr(model_config, "mlqds_hybrid_mode", "fill")),
    }
    diagnostics.update(gain_cost_diagnostics)
    return transformed, transformed_mask, diagnostics


def range_component_retained_frequency_training_labels(
    labels: torch.Tensor,
    labelled_mask: torch.Tensor,
    component_labels: dict[str, torch.Tensor],
    boundaries: list[tuple[int, int]],
    model_config: object,
    type_idx: int = QUERY_TYPE_ID_RANGE,
    component_weights: dict[str, float] | None = None,
    mode: str = "component_retained_frequency",
    source: str = "range_component_training_labels",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Build retained-frequency targets independently per RangeUseful component.

    The regular retained-frequency target can be dominated by point-hit and ship
    presence labels. This variant gives temporal, gap, turn, and shape labels a
    direct path into the retained-set target while remaining query-blind at
    inference time.
    """
    if labels.ndim != 2 or labelled_mask.shape != labels.shape:
        raise ValueError("labels and labelled_mask must have matching shape [n_points, n_types].")
    if type_idx < 0 or type_idx >= labels.shape[1]:
        raise ValueError(f"type_idx={type_idx} is outside label shape {tuple(labels.shape)}.")

    ratios = _target_budget_ratios(model_config)
    budget_weights = _target_budget_weights(model_config, ratios)
    weights = component_weights or dict(RANGE_USEFULNESS_WEIGHTS)
    component_target = torch.zeros((labels.shape[0],), dtype=torch.float32, device=labels.device)
    available_weight = 0.0
    per_component: dict[str, dict[str, object]] = {}
    for component_name, component_weight in weights.items():
        component = component_labels.get(component_name)
        if component is None:
            continue
        if component.shape != labels.shape:
            raise ValueError(f"component {component_name!r} has shape {tuple(component.shape)}, expected {tuple(labels.shape)}.")
        source_scores = component[:, type_idx].float().clamp(min=0.0)
        source_positive = source_scores > 0.0
        source_mass = float(source_scores[source_positive].sum().item()) if bool(source_positive.any().item()) else 0.0
        if source_mass <= 1e-12:
            per_component[component_name] = {
                "source_positive_label_count": 0,
                "source_positive_label_mass": 0.0,
                "used": False,
            }
            continue
        retained_frequency, used = _retained_frequency_from_scores(
            source_scores=source_scores,
            boundaries=boundaries,
            model_config=model_config,
            ratios=ratios,
        )
        if used <= 0:
            continue
        weight = float(component_weight)
        component_target += weight * retained_frequency
        available_weight += weight
        target_positive = retained_frequency > 0.0
        per_component[component_name] = {
            "source_positive_label_count": int(source_positive.sum().item()),
            "source_positive_label_mass": source_mass,
            "target_positive_label_count": int(target_positive.sum().item()),
            "target_positive_label_mass": (
                float(retained_frequency[target_positive].sum().item()) if bool(target_positive.any().item()) else 0.0
            ),
            "weight": weight,
            "used": True,
        }

    if available_weight <= 1e-12:
        raise ValueError("component_retained_frequency target found no positive component label mass.")
    component_target = (component_target / float(available_weight)).clamp(0.0, 1.0)
    component_blend = max(0.0, min(1.0, float(getattr(model_config, "range_component_target_blend", 1.0))))
    base_retained_frequency = None
    if component_blend < 1.0:
        base_retained_frequency, _used = _retained_frequency_from_scores(
            source_scores=labels[:, type_idx].float().clamp(min=0.0),
            boundaries=boundaries,
            model_config=model_config,
            ratios=ratios,
        )
        component_target = (
            (1.0 - component_blend) * base_retained_frequency
            + component_blend * component_target
        ).clamp(0.0, 1.0)
    component_target, temporal_blend_diagnostics = _apply_temporal_target_blend(
        retained_frequency=component_target,
        boundaries=boundaries,
        model_config=model_config,
        ratios=ratios,
    )

    transformed = labels.clone()
    transformed[:, type_idx] = component_target.to(dtype=transformed.dtype)
    transformed_mask = labelled_mask.clone()
    transformed_mask[:, type_idx] = True
    positive = transformed[:, type_idx] > 0.0
    positive_count = int(positive.sum().item())
    diagnostics: dict[str, object] = {
        "mode": mode,
        "source": source,
        "budget_loss_ratios": list(ratios),
        "budget_weights": list(budget_weights),
        "budget_weight_power": float(getattr(model_config, "range_target_budget_weight_power", 0.0) or 0.0),
        "labelled_point_count": int(transformed.shape[0]),
        "positive_label_count": positive_count,
        "positive_label_fraction": float(positive_count / max(1, int(transformed.shape[0]))),
        "positive_label_mass": float(transformed[positive, type_idx].sum().item()) if positive_count > 0 else 0.0,
        "available_component_weight": float(available_weight),
        "component_target_blend": float(component_blend),
        "base_retained_frequency_positive_label_count": (
            int((base_retained_frequency > 0.0).sum().item()) if base_retained_frequency is not None else None
        ),
        "component_diagnostics": per_component,
        "mlqds_temporal_fraction": float(getattr(model_config, "mlqds_temporal_fraction", 0.0)),
        "mlqds_hybrid_mode": str(getattr(model_config, "mlqds_hybrid_mode", "fill")),
    }
    diagnostics.update(temporal_blend_diagnostics)
    return transformed, transformed_mask, diagnostics


def range_continuity_retained_frequency_training_labels(
    labels: torch.Tensor,
    labelled_mask: torch.Tensor,
    component_labels: dict[str, torch.Tensor],
    boundaries: list[tuple[int, int]],
    model_config: object,
    type_idx: int = QUERY_TYPE_ID_RANGE,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Build a retained-frequency target from continuity and boundary components only."""
    return range_component_retained_frequency_training_labels(
        labels=labels,
        labelled_mask=labelled_mask,
        component_labels=component_labels,
        boundaries=boundaries,
        model_config=model_config,
        type_idx=type_idx,
        component_weights=dict(RANGE_CONTINUITY_TARGET_WEIGHTS),
        mode="continuity_retained_frequency",
        source="range_continuity_component_training_labels",
    )


def aggregate_range_component_label_sets(
    label_sets: list[tuple[torch.Tensor, torch.Tensor]],
    component_label_sets: list[dict[str, torch.Tensor] | None],
    type_idx: int = QUERY_TYPE_ID_RANGE,
    aggregation: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, object]]:
    """Aggregate raw range labels and their component-specific label streams."""
    if len(component_label_sets) != len(label_sets):
        raise ValueError("component_label_sets must have the same length as label_sets.")
    if not component_label_sets or any(component_labels is None for component_labels in component_label_sets):
        raise ValueError("all component_label_sets entries must be present.")

    aggregated, aggregated_mask, diagnostics = aggregate_range_label_sets(
        label_sets,
        type_idx=type_idx,
        source="range_component_label_replicates",
        aggregation=aggregation,
    )
    aggregated_components: dict[str, torch.Tensor] = {}
    component_diagnostics: dict[str, object] = {}
    for component_name in RANGE_USEFULNESS_WEIGHTS:
        component_sets: list[tuple[torch.Tensor, torch.Tensor]] = []
        for (labels, labelled_mask), component_labels in zip(label_sets, component_label_sets):
            if component_labels is None:
                raise RuntimeError("all component_label_sets entries must be present.")
            component = component_labels.get(component_name)
            if component is None:
                raise ValueError(f"component_label_sets missing {component_name!r}.")
            if component.shape != labels.shape:
                raise ValueError(
                    f"component {component_name!r} has shape {tuple(component.shape)}, expected {tuple(labels.shape)}."
                )
            component_sets.append((component, labelled_mask))
        component_aggregated, _component_mask, component_diag = aggregate_range_label_sets(
            component_sets,
            type_idx=type_idx,
            source=f"range_component_{component_name}_replicates",
            aggregation=aggregation,
        )
        aggregated_components[component_name] = component_aggregated
        component_diagnostics[component_name] = component_diag

    diagnostics["component_aggregation"] = component_diagnostics
    return aggregated, aggregated_mask, aggregated_components, diagnostics


def aggregate_range_component_retained_frequency_training_labels(
    label_sets: list[tuple[torch.Tensor, torch.Tensor]],
    component_label_sets: list[dict[str, torch.Tensor] | None],
    boundaries: list[tuple[int, int]],
    model_config: object,
    type_idx: int = QUERY_TYPE_ID_RANGE,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Average component-retained targets over independent train workloads."""
    if len(component_label_sets) != len(label_sets):
        raise ValueError("component_label_sets must have the same length as label_sets.")
    if not component_label_sets or any(component_labels is None for component_labels in component_label_sets):
        raise ValueError("all component_label_sets entries must be present.")

    transformed_sets: list[tuple[torch.Tensor, torch.Tensor]] = []
    per_replicate: list[dict[str, object]] = []
    for (labels, labelled_mask), component_labels in zip(label_sets, component_label_sets):
        if component_labels is None:
            raise RuntimeError("all component_label_sets entries must be present.")
        transformed, transformed_mask, diagnostics = range_component_retained_frequency_training_labels(
            labels=labels,
            labelled_mask=labelled_mask,
            component_labels=component_labels,
            boundaries=boundaries,
            model_config=model_config,
            type_idx=type_idx,
        )
        transformed_sets.append((transformed, transformed_mask))
        per_replicate.append(diagnostics)

    aggregated, aggregated_mask, diagnostics = aggregate_range_label_sets(
        transformed_sets,
        type_idx=type_idx,
        source="range_component_retained_frequency_training_label_replicates",
    )
    diagnostics.update(
        {
            "mode": "component_retained_frequency",
            "budget_loss_ratios": list(_target_budget_ratios(model_config)),
            "budget_weights": list(_target_budget_weights(model_config, _target_budget_ratios(model_config))),
            "budget_weight_power": float(getattr(model_config, "range_target_budget_weight_power", 0.0) or 0.0),
            "component_target_blend": float(getattr(model_config, "range_component_target_blend", 1.0)),
            "mlqds_temporal_fraction": float(getattr(model_config, "mlqds_temporal_fraction", 0.0)),
            "mlqds_hybrid_mode": str(getattr(model_config, "mlqds_hybrid_mode", "fill")),
            "per_replicate": per_replicate,
        }
    )
    return aggregated, aggregated_mask, diagnostics


def aggregate_range_continuity_retained_frequency_training_labels(
    label_sets: list[tuple[torch.Tensor, torch.Tensor]],
    component_label_sets: list[dict[str, torch.Tensor] | None],
    boundaries: list[tuple[int, int]],
    model_config: object,
    type_idx: int = QUERY_TYPE_ID_RANGE,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Average continuity-retained targets over independent train workloads."""
    if len(component_label_sets) != len(label_sets):
        raise ValueError("component_label_sets must have the same length as label_sets.")
    if not component_label_sets or any(component_labels is None for component_labels in component_label_sets):
        raise ValueError("all component_label_sets entries must be present.")

    transformed_sets: list[tuple[torch.Tensor, torch.Tensor]] = []
    per_replicate: list[dict[str, object]] = []
    for (labels, labelled_mask), component_labels in zip(label_sets, component_label_sets):
        if component_labels is None:
            raise RuntimeError("all component_label_sets entries must be present.")
        transformed, transformed_mask, diagnostics = range_continuity_retained_frequency_training_labels(
            labels=labels,
            labelled_mask=labelled_mask,
            component_labels=component_labels,
            boundaries=boundaries,
            model_config=model_config,
            type_idx=type_idx,
        )
        transformed_sets.append((transformed, transformed_mask))
        per_replicate.append(diagnostics)

    aggregated, aggregated_mask, diagnostics = aggregate_range_label_sets(
        transformed_sets,
        type_idx=type_idx,
        source="range_continuity_retained_frequency_training_label_replicates",
    )
    diagnostics.update(
        {
            "mode": "continuity_retained_frequency",
            "budget_loss_ratios": list(_target_budget_ratios(model_config)),
            "budget_weights": list(_target_budget_weights(model_config, _target_budget_ratios(model_config))),
            "budget_weight_power": float(getattr(model_config, "range_target_budget_weight_power", 0.0) or 0.0),
            "component_target_blend": float(getattr(model_config, "range_component_target_blend", 1.0)),
            "continuity_component_weights": dict(RANGE_CONTINUITY_TARGET_WEIGHTS),
            "mlqds_temporal_fraction": float(getattr(model_config, "mlqds_temporal_fraction", 0.0)),
            "mlqds_hybrid_mode": str(getattr(model_config, "mlqds_hybrid_mode", "fill")),
            "per_replicate": per_replicate,
        }
    )
    return aggregated, aggregated_mask, diagnostics


def aggregate_range_structural_retained_frequency_training_labels(
    label_sets: list[tuple[torch.Tensor, torch.Tensor]],
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    model_config: object,
    type_idx: int = QUERY_TYPE_ID_RANGE,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Average structural-retained targets over independent train workloads."""
    transformed_sets: list[tuple[torch.Tensor, torch.Tensor]] = []
    per_replicate: list[dict[str, object]] = []
    for labels, labelled_mask in label_sets:
        transformed, transformed_mask, diagnostics = range_structural_retained_frequency_training_labels(
            labels=labels,
            labelled_mask=labelled_mask,
            points=points,
            boundaries=boundaries,
            model_config=model_config,
            type_idx=type_idx,
        )
        transformed_sets.append((transformed, transformed_mask))
        per_replicate.append(diagnostics)

    aggregated, aggregated_mask, diagnostics = aggregate_range_label_sets(
        transformed_sets,
        type_idx=type_idx,
        source="range_structural_retained_frequency_training_label_replicates",
    )
    diagnostics.update(
        {
            "mode": "structural_retained_frequency",
            "budget_loss_ratios": list(_target_budget_ratios(model_config)),
            "budget_weights": list(_target_budget_weights(model_config, _target_budget_ratios(model_config))),
            "budget_weight_power": float(getattr(model_config, "range_target_budget_weight_power", 0.0) or 0.0),
            "structural_target_blend": float(
                max(0.0, min(1.0, float(getattr(model_config, "range_structural_target_blend", 0.25) or 0.0)))
            ),
            "structural_target_source_mode": str(
                getattr(model_config, "range_structural_target_source_mode", "blend")
            ).lower(),
            "mlqds_temporal_fraction": float(getattr(model_config, "mlqds_temporal_fraction", 0.0)),
            "mlqds_hybrid_mode": str(getattr(model_config, "mlqds_hybrid_mode", "fill")),
            "per_replicate": per_replicate,
        }
    )
    return aggregated, aggregated_mask, diagnostics


def aggregate_range_retained_frequency_training_labels(
    label_sets: list[tuple[torch.Tensor, torch.Tensor]],
    boundaries: list[tuple[int, int]],
    model_config: object,
    type_idx: int = QUERY_TYPE_ID_RANGE,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Average retained-frequency targets over independent train workloads."""
    transformed_sets: list[tuple[torch.Tensor, torch.Tensor]] = []
    per_replicate: list[dict[str, object]] = []
    for labels, labelled_mask in label_sets:
        transformed, transformed_mask, diagnostics = range_retained_frequency_training_labels(
            labels=labels,
            labelled_mask=labelled_mask,
            boundaries=boundaries,
            model_config=model_config,
            type_idx=type_idx,
        )
        transformed_sets.append((transformed, transformed_mask))
        per_replicate.append(diagnostics)

    aggregated, aggregated_mask, diagnostics = aggregate_range_label_sets(
        transformed_sets,
        type_idx=type_idx,
        source="range_retained_frequency_training_label_replicates",
    )
    diagnostics.update(
        {
            "mode": "retained_frequency",
            "budget_loss_ratios": list(_target_budget_ratios(model_config)),
            "budget_weights": list(_target_budget_weights(model_config, _target_budget_ratios(model_config))),
            "budget_weight_power": float(getattr(model_config, "range_target_budget_weight_power", 0.0) or 0.0),
            "mlqds_temporal_fraction": float(getattr(model_config, "mlqds_temporal_fraction", 0.0)),
            "mlqds_hybrid_mode": str(getattr(model_config, "mlqds_hybrid_mode", "fill")),
            "per_replicate": per_replicate,
        }
    )
    return aggregated, aggregated_mask, diagnostics


def aggregate_range_global_budget_retained_frequency_training_labels(
    label_sets: list[tuple[torch.Tensor, torch.Tensor]],
    boundaries: list[tuple[int, int]],
    model_config: object,
    type_idx: int = QUERY_TYPE_ID_RANGE,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Average global-budget retained targets over independent train workloads."""
    transformed_sets: list[tuple[torch.Tensor, torch.Tensor]] = []
    per_replicate: list[dict[str, object]] = []
    for labels, labelled_mask in label_sets:
        transformed, transformed_mask, diagnostics = range_global_budget_retained_frequency_training_labels(
            labels=labels,
            labelled_mask=labelled_mask,
            boundaries=boundaries,
            model_config=model_config,
            type_idx=type_idx,
        )
        transformed_sets.append((transformed, transformed_mask))
        per_replicate.append(diagnostics)

    aggregated, aggregated_mask, diagnostics = aggregate_range_label_sets(
        transformed_sets,
        type_idx=type_idx,
        source="range_global_budget_retained_frequency_training_label_replicates",
    )
    diagnostics.update(
        {
            "mode": "global_budget_retained_frequency",
            "budget_loss_ratios": list(_target_budget_ratios(model_config)),
            "budget_weights": list(_target_budget_weights(model_config, _target_budget_ratios(model_config))),
            "budget_weight_power": float(getattr(model_config, "range_target_budget_weight_power", 0.0) or 0.0),
            "global_budget_min_points_per_trajectory": 2,
            "per_replicate": per_replicate,
        }
    )
    return aggregated, aggregated_mask, diagnostics


def aggregate_range_marginal_coverage_training_labels(
    label_sets: list[tuple[torch.Tensor, torch.Tensor]],
    boundaries: list[tuple[int, int]],
    model_config: object,
    type_idx: int = QUERY_TYPE_ID_RANGE,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Average marginal-coverage retained targets over independent train workloads."""
    transformed_sets: list[tuple[torch.Tensor, torch.Tensor]] = []
    per_replicate: list[dict[str, object]] = []
    for labels, labelled_mask in label_sets:
        transformed, transformed_mask, diagnostics = range_marginal_coverage_training_labels(
            labels=labels,
            labelled_mask=labelled_mask,
            boundaries=boundaries,
            model_config=model_config,
            type_idx=type_idx,
        )
        transformed_sets.append((transformed, transformed_mask))
        per_replicate.append(diagnostics)

    aggregated, aggregated_mask, diagnostics = aggregate_range_label_sets(
        transformed_sets,
        type_idx=type_idx,
        source="range_marginal_coverage_training_label_replicates",
    )
    diagnostics.update(
        {
            "mode": "marginal_coverage_frequency",
            "budget_loss_ratios": list(_target_budget_ratios(model_config)),
            "budget_weights": list(_target_budget_weights(model_config, _target_budget_ratios(model_config))),
            "budget_weight_power": float(getattr(model_config, "range_target_budget_weight_power", 0.0) or 0.0),
            "marginal_target_radius_scale": float(
                getattr(model_config, "range_marginal_target_radius_scale", 0.50) or 0.0
            ),
            "mlqds_temporal_fraction": float(getattr(model_config, "mlqds_temporal_fraction", 0.0)),
            "mlqds_hybrid_mode": str(getattr(model_config, "mlqds_hybrid_mode", "fill")),
            "per_replicate": per_replicate,
        }
    )
    return aggregated, aggregated_mask, diagnostics
