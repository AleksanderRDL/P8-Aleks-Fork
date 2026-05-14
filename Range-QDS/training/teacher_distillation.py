"""Teacher-student label construction for workload-blind range training."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from experiments.experiment_config import ModelConfig
from queries.query_types import NUM_QUERY_TYPES, QUERY_TYPE_ID_RANGE
from queries.workload import TypedQueryWorkload
from simplification.mlqds_scoring import pure_workload_scores
from simplification.simplify_trajectories import simplify_with_temporal_score_hybrid
from training.inference import windowed_predict
from training.model_features import build_model_point_features
from training.training_losses import _budget_loss_ratios, _safe_quantile
from training.training_outputs import TrainingOutputs

RANGE_TEACHER_DISTILLATION_MODES = ("none", "rank_percentile", "retained_frequency")


def range_teacher_distillation_enabled(model_config: ModelConfig) -> bool:
    """Return whether the run should train a query-aware range teacher."""
    mode = str(getattr(model_config, "range_teacher_distillation_mode", "none")).lower()
    return mode != "none"


def build_range_teacher_config(student_config: ModelConfig) -> ModelConfig:
    """Return the training config for the query-aware range teacher.

    The teacher is a supervision generator only. It is deliberately selected by
    training loss, not by final eval queries.
    """
    teacher_epochs = int(getattr(student_config, "range_teacher_epochs", 0) or student_config.epochs)
    return replace(
        student_config,
        model_type="range_aware",
        epochs=max(1, teacher_epochs),
        mlqds_range_geometry_blend=0.0,
        checkpoint_selection_metric="loss",
        validation_score_every=0,
        early_stopping_patience=0,
    )


def _teacher_predictions(
    *,
    teacher: TrainingOutputs,
    teacher_model_type: str,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    workload: TypedQueryWorkload,
    model_config: ModelConfig,
) -> torch.Tensor:
    """Score training points with the query-aware teacher workload."""
    teacher_points = build_model_point_features(points, workload, teacher_model_type)
    norm_points, norm_queries = teacher.scaler.transform(teacher_points, workload.query_features)
    return windowed_predict(
        model=teacher.model,
        norm_points=norm_points,
        boundaries=boundaries,
        queries=norm_queries,
        query_type_ids=workload.type_ids,
        window_length=model_config.window_length,
        window_stride=model_config.window_stride,
        batch_size=model_config.inference_batch_size,
        amp_mode=model_config.amp_mode,
    ).float()


def _rank_percentile_labels(
    predictions: torch.Tensor,
    boundaries: list[tuple[int, int]],
) -> torch.Tensor:
    """Convert teacher logits into per-trajectory rank-percentile labels."""
    return pure_workload_scores(
        predictions,
        boundaries,
        "range",
        score_mode="rank_tie",
    ).float()


def _retained_frequency_labels(
    predictions: torch.Tensor,
    boundaries: list[tuple[int, int]],
    model_config: ModelConfig,
) -> torch.Tensor:
    """Use teacher-retained membership frequency across configured budgets."""
    ratios = _budget_loss_ratios(model_config)
    if not ratios:
        ratios = (float(model_config.compression_ratio),)
    scores = pure_workload_scores(
        predictions,
        boundaries,
        "range",
        score_mode=str(getattr(model_config, "mlqds_score_mode", "rank")),
        score_temperature=float(getattr(model_config, "mlqds_score_temperature", 1.0)),
        rank_confidence_weight=float(getattr(model_config, "mlqds_rank_confidence_weight", 0.15)),
    ).float()
    frequency = torch.zeros_like(scores)
    used = 0
    for ratio in ratios:
        if float(ratio) <= 0.0:
            continue
        mask = simplify_with_temporal_score_hybrid(
            scores,
            boundaries,
            float(ratio),
            temporal_fraction=float(getattr(model_config, "mlqds_temporal_fraction", 0.0)),
            diversity_bonus=float(getattr(model_config, "mlqds_diversity_bonus", 0.0)),
            hybrid_mode=str(getattr(model_config, "mlqds_hybrid_mode", "fill")),
            stratified_center_weight=float(getattr(model_config, "mlqds_stratified_center_weight", 0.0)),
            min_learned_swaps=int(getattr(model_config, "mlqds_min_learned_swaps", 0)),
        )
        frequency += mask.to(dtype=frequency.dtype)
        used += 1
    if used <= 0:
        return frequency
    return frequency / float(used)


def _label_diagnostics(values: torch.Tensor, mode: str) -> dict[str, Any]:
    """Summarize distilled labels for experiment artifacts."""
    positive = values > 0.0
    positive_count = int(positive.sum().item())
    diag: dict[str, Any] = {
        "enabled": True,
        "mode": str(mode),
        "labelled_point_count": int(values.numel()),
        "positive_label_count": positive_count,
        "positive_label_fraction": float(positive_count / max(1, int(values.numel()))),
        "positive_label_mass": float(values[positive].sum().item()) if positive_count > 0 else 0.0,
        "label_min": float(values.min().item()) if values.numel() else 0.0,
        "label_max": float(values.max().item()) if values.numel() else 0.0,
        "label_mean": float(values.mean().item()) if values.numel() else 0.0,
    }
    if positive_count > 0:
        positive_values = values[positive]
        diag["positive_label_p50"] = float(_safe_quantile(positive_values, 0.50).item())
        diag["positive_label_p90"] = float(_safe_quantile(positive_values, 0.90).item())
        diag["positive_label_p99"] = float(_safe_quantile(positive_values, 0.99).item())
    else:
        diag["positive_label_p50"] = 0.0
        diag["positive_label_p90"] = 0.0
        diag["positive_label_p99"] = 0.0
    return diag


def distill_range_teacher_labels(
    *,
    teacher: TrainingOutputs,
    teacher_model_type: str,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    workload: TypedQueryWorkload,
    model_config: ModelConfig,
) -> tuple[tuple[torch.Tensor, torch.Tensor], dict[str, Any]]:
    """Build query-blind training labels from a query-aware range teacher."""
    mode = str(getattr(model_config, "range_teacher_distillation_mode", "none")).lower()
    if mode not in RANGE_TEACHER_DISTILLATION_MODES:
        raise ValueError(f"range_teacher_distillation_mode must be one of {RANGE_TEACHER_DISTILLATION_MODES}.")
    if mode == "none":
        raise ValueError("distill_range_teacher_labels requires an enabled distillation mode.")

    predictions = _teacher_predictions(
        teacher=teacher,
        teacher_model_type=teacher_model_type,
        points=points,
        boundaries=boundaries,
        workload=workload,
        model_config=model_config,
    )
    if mode == "rank_percentile":
        target_values = _rank_percentile_labels(predictions, boundaries)
    elif mode == "retained_frequency":
        target_values = _retained_frequency_labels(predictions, boundaries, model_config)
    else:
        raise ValueError(f"Unsupported range_teacher_distillation_mode={mode!r}.")

    labels = torch.zeros((points.shape[0], NUM_QUERY_TYPES), dtype=torch.float32, device=points.device)
    labelled_mask = torch.zeros_like(labels, dtype=torch.bool)
    labels[:, QUERY_TYPE_ID_RANGE] = target_values.to(device=points.device, dtype=torch.float32).clamp(0.0, 1.0)
    labelled_mask[:, QUERY_TYPE_ID_RANGE] = True
    diagnostics = _label_diagnostics(labels[:, QUERY_TYPE_ID_RANGE], mode)
    diagnostics["teacher_model_type"] = str(teacher_model_type)
    diagnostics["teacher_epochs_trained"] = int(teacher.epochs_trained)
    diagnostics["budget_loss_ratios"] = list(_budget_loss_ratios(model_config))
    diagnostics["mlqds_temporal_fraction"] = float(getattr(model_config, "mlqds_temporal_fraction", 0.0))
    diagnostics["mlqds_hybrid_mode"] = str(getattr(model_config, "mlqds_hybrid_mode", "fill"))
    return (labels, labelled_mask), diagnostics
