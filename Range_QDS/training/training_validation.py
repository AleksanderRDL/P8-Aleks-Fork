"""Validation scoring helpers for training-time checkpoint selection."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from evaluation.baselines import UniformTemporalMethod
from evaluation.evaluate_methods import score_range_usefulness, score_retained_mask
from experiments.experiment_config import ModelConfig
from experiments.torch_runtime import normalize_amp_mode, torch_autocast_context
from queries.query_types import single_workload_type
from queries.workload import TypedQueryWorkload
from simplification.mlqds_scoring import simplify_mlqds_predictions
from training.model_features import build_model_point_features_for_dim
from training.inference import _is_workload_blind_model, _model_point_dim
from training.scaler import FeatureScaler
from training.training_setup import _pure_query_type_id
from training.training_windows import _trajectory_batch_to_device
from training.trajectory_batching import batch_windows, build_trajectory_windows

PredictWorkloadLogits = Callable[..., torch.Tensor]


def _predict_workload_logits(
    model: torch.nn.Module,
    scaler: FeatureScaler,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    workload: TypedQueryWorkload,
    model_config: ModelConfig,
    device: torch.device,
) -> torch.Tensor:
    """Predict per-point pure-workload scores for exact validation-score diagnostics."""
    point_dim = _model_point_dim(model)
    model_points = build_model_point_features_for_dim(points, workload, point_dim)
    if _is_workload_blind_model(model):
        norm_points = scaler.transform_points(model_points)
        norm_queries = None
        type_ids_dev = None
    else:
        norm_points, norm_queries = scaler.transform(model_points, workload.query_features)
        type_ids_dev = workload.type_ids.to(device)
        _pure_query_type_id(workload.type_ids)
    norm_points_dev = norm_points.to(device)
    norm_queries_dev = None if norm_queries is None else norm_queries.to(device)
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
    model: torch.nn.Module,
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
    predict_logits_fn: PredictWorkloadLogits | None = None,
) -> tuple[float, dict[str, float], dict[str, float]]:
    """Evaluate a checkpoint and return selected score plus explicit validation metrics."""
    points = validation_points if validation_points is not None else torch.cat(trajectories, dim=0)
    predict = predict_logits_fn or _predict_workload_logits
    predictions = predict(
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
        stratified_center_weight=float(getattr(model_config, "mlqds_stratified_center_weight", 0.0)),
        min_learned_swaps=int(getattr(model_config, "mlqds_min_learned_swaps", 0)),
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
    model: torch.nn.Module,
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
    predict_logits_fn: PredictWorkloadLogits | None = None,
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
        predict_logits_fn=predict_logits_fn,
    )
    return score, per_type


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
