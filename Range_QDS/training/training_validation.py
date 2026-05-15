"""Validation scoring helpers for training-time checkpoint selection."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch

from evaluation.baselines import UniformTemporalMethod
from evaluation.evaluate_methods import score_range_usefulness, score_retained_mask
from evaluation.query_useful_v1 import query_useful_v1_from_range_audit
from experiments.experiment_config import ModelConfig
from experiments.torch_runtime import normalize_amp_mode
from queries.query_types import single_workload_type
from queries.workload import TypedQueryWorkload
from simplification.mlqds_scoring import simplify_mlqds_predictions
from training.query_useful_targets import QUERY_USEFUL_V1_HEAD_NAMES
from training.model_features import build_model_point_features_for_dim
from training.inference import _is_workload_blind_model, _model_point_dim
from training.inference import windowed_predict_with_heads
from training.scaler import FeatureScaler
from training.training_setup import _pure_query_type_id

PredictWorkloadLogits = Callable[..., torch.Tensor]


def _predict_workload_logits_with_heads(
    model: torch.nn.Module,
    scaler: FeatureScaler,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    workload: TypedQueryWorkload,
    model_config: ModelConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Predict per-point pure-workload scores and optional factorized head logits."""
    point_dim = _model_point_dim(model)
    model_points = build_model_point_features_for_dim(
        points,
        workload,
        point_dim,
        boundaries=boundaries,
        query_prior_field=getattr(model, "query_prior_field", None),
    )
    if _is_workload_blind_model(model):
        norm_points = scaler.transform_points(model_points)
        norm_queries = None
        type_ids_dev = None
    else:
        norm_points, norm_queries = scaler.transform(model_points, workload.query_features)
        type_ids_dev = workload.type_ids.to(device)
        _pure_query_type_id(workload.type_ids)
    inference_batch_size = max(1, int(getattr(model_config, "inference_batch_size", 16)))
    amp_mode = normalize_amp_mode(getattr(model_config, "amp_mode", "off"))
    scores, head_logits = windowed_predict_with_heads(
        model=model,
        norm_points=norm_points,
        boundaries=boundaries,
        queries=norm_queries,
        query_type_ids=type_ids_dev,
        window_length=model_config.window_length,
        window_stride=model_config.window_stride,
        batch_size=inference_batch_size,
        device=device,
        amp_mode=amp_mode,
    )
    return scores.detach().cpu(), None if head_logits is None else head_logits.detach().cpu()


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
    scores, _head_logits = _predict_workload_logits_with_heads(
        model=model,
        scaler=scaler,
        points=points,
        boundaries=boundaries,
        workload=workload,
        model_config=model_config,
        device=device,
    )
    return scores


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
    head_logits = None
    if predict_logits_fn is None:
        predictions, head_logits = _predict_workload_logits_with_heads(
            model=model,
            scaler=scaler,
            points=points,
            boundaries=boundaries,
            workload=workload,
            model_config=model_config,
            device=device,
        )
    else:
        predictions = predict_logits_fn(
            model=model,
            scaler=scaler,
            points=points,
            boundaries=boundaries,
            workload=workload,
            model_config=model_config,
            device=device,
        )
    segment_scores = None
    if head_logits is not None:
        try:
            segment_head_idx = tuple(QUERY_USEFUL_V1_HEAD_NAMES).index("segment_budget_target")
        except ValueError:
            segment_head_idx = -1
        if segment_head_idx >= 0 and int(head_logits.shape[-1]) > segment_head_idx:
            segment_scores = torch.sigmoid(head_logits[:, segment_head_idx].detach().cpu().float())
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
        selector_type=str(getattr(model_config, "selector_type", "temporal_hybrid")),
        segment_scores=segment_scores,
        points=points,
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
        query_useful = query_useful_v1_from_range_audit(range_audit)
        metrics.update(
            {
                "range_usefulness": float(range_audit["range_usefulness_score"]),
                "query_useful_v1": float(cast(Any, query_useful["query_useful_v1_score"])),
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
    if variant == "query_useful_v1":
        if range_audit is None:
            return float(answer_agg), answer_pt, metrics
        score = float(cast(Any, query_useful_v1_from_range_audit(range_audit)["query_useful_v1_score"]))
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
    if variant == "query_useful_v1":
        audit = score_range_usefulness(
            points=points,
            boundaries=boundaries,
            retained_mask=retained_mask,
            typed_queries=workload.typed_queries,
            query_cache=query_cache,
        )
        score = float(cast(Any, query_useful_v1_from_range_audit(audit)["query_useful_v1_score"]))
        return score, {"range": score}
    if variant == "combined":
        return combined_agg, combined_pt
    return answer_agg, answer_pt
