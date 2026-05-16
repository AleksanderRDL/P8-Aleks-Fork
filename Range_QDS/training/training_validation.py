"""Validation scoring helpers for training-time checkpoint selection."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch

from evaluation.baselines import UniformTemporalMethod
from evaluation.evaluate_methods import score_range_usefulness, score_retained_mask
from evaluation.metrics import compute_geometric_distortion, compute_length_preservation
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


def _validation_endpoint_sanity(retained_mask: torch.Tensor, boundaries: list[tuple[int, int]]) -> float:
    """Return fraction of eligible trajectories whose endpoints are retained."""
    retained = retained_mask.detach().cpu().bool()
    eligible = 0
    passing = 0
    for start, end in boundaries:
        if int(end) - int(start) < 2:
            continue
        local_count = int(retained[int(start) : int(end)].sum().item())
        if local_count < 2:
            continue
        eligible += 1
        if bool(retained[int(start)].item()) and bool(retained[int(end) - 1].item()):
            passing += 1
    if eligible <= 0:
        return 1.0
    return float(passing / eligible)


def _validation_sed_ratio_threshold(compression_ratio: float) -> float:
    """Return the same soft SED threshold used by final global sanity."""
    ratio = float(compression_ratio)
    if ratio <= 0.01 + 1e-12:
        return 2.00
    if ratio <= 0.02 + 1e-12:
        return 1.75
    return 1.50


def _validation_global_sanity_metrics(
    *,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    retained_mask: torch.Tensor,
    model_config: ModelConfig,
    uniform_retained_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Return geometry guardrail metrics used by validation checkpoint scoring."""
    uniform_mask = (
        uniform_retained_mask
        if uniform_retained_mask is not None
        else UniformTemporalMethod().simplify(
            points=points,
            boundaries=boundaries,
            compression_ratio=model_config.compression_ratio,
        )
    )
    geometric = compute_geometric_distortion(points, boundaries, retained_mask)
    uniform_geometric = compute_geometric_distortion(points, boundaries, uniform_mask)
    avg_sed = float(geometric.get("avg_sed_km", 0.0))
    uniform_avg_sed = float(uniform_geometric.get("avg_sed_km", 0.0))
    if uniform_avg_sed <= 1e-12:
        sed_ratio = 1.0 if avg_sed <= 1e-12 else float("inf")
    else:
        sed_ratio = float(avg_sed / uniform_avg_sed)
    return {
        "avg_length_preserved": float(compute_length_preservation(points, boundaries, retained_mask)),
        "endpoint_sanity": _validation_endpoint_sanity(retained_mask, boundaries),
        "avg_sed_km": avg_sed,
        "uniform_avg_sed_km": uniform_avg_sed,
        "avg_sed_ratio_vs_uniform": sed_ratio,
        "avg_sed_ratio_vs_uniform_max": _validation_sed_ratio_threshold(float(model_config.compression_ratio)),
    }


def _validation_query_useful_selection_score(
    raw_query_useful_v1: float,
    sanity: dict[str, float],
    model_config: ModelConfig,
) -> float:
    """Apply a light validation-only penalty for global sanity failures."""
    if not bool(getattr(model_config, "validation_global_sanity_penalty_enabled", True)):
        return float(raw_query_useful_v1)
    length_min = float(getattr(model_config, "validation_length_preservation_min", 0.80))
    length_penalty = max(0.0, length_min - float(sanity.get("avg_length_preserved", 1.0)))
    sed_penalty = max(
        0.0,
        float(sanity.get("avg_sed_ratio_vs_uniform", 1.0))
        - float(sanity.get("avg_sed_ratio_vs_uniform_max", 1.50)),
    )
    endpoint_penalty = max(0.0, 1.0 - float(sanity.get("endpoint_sanity", 1.0)))
    total_penalty = (
        float(getattr(model_config, "validation_global_sanity_penalty_weight", 0.10)) * length_penalty
        + float(getattr(model_config, "validation_sed_penalty_weight", 0.05)) * sed_penalty
        + float(getattr(model_config, "validation_endpoint_penalty_weight", 0.10)) * endpoint_penalty
    )
    return float(raw_query_useful_v1 - total_penalty)


def _validation_global_sanity_penalty(
    raw_query_useful_v1: float,
    sanity: dict[str, float],
    model_config: ModelConfig,
) -> float:
    """Return the validation-only global-sanity penalty magnitude."""
    return float(raw_query_useful_v1 - _validation_query_useful_selection_score(raw_query_useful_v1, sanity, model_config))


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
            segment_scores = head_logits[:, segment_head_idx].detach().cpu().float()
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
        learned_segment_geometry_gain_weight=float(getattr(model_config, "learned_segment_geometry_gain_weight", 0.12)),
        learned_segment_score_blend_weight=float(getattr(model_config, "learned_segment_score_blend_weight", 0.05)),
        learned_segment_fairness_preallocation=bool(getattr(model_config, "learned_segment_fairness_preallocation", True)),
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
        sanity = _validation_global_sanity_metrics(
            points=points,
            boundaries=boundaries,
            retained_mask=retained_mask,
            model_config=model_config,
        )
        query_useful = query_useful_v1_from_range_audit(
            range_audit,
            length_preservation=sanity["avg_length_preserved"],
            avg_sed_km=sanity["avg_sed_km"],
            endpoint_sanity=sanity["endpoint_sanity"],
        )
        raw_query_useful_score = float(cast(Any, query_useful["query_useful_v1_score"]))
        penalized_query_useful_score = _validation_query_useful_selection_score(
            raw_query_useful_score,
            sanity,
            model_config,
        )
        metrics.update(
            {
                "range_usefulness": float(range_audit["range_usefulness_score"]),
                "query_useful_v1": raw_query_useful_score,
                "query_useful_v1_selection_score": penalized_query_useful_score,
                "validation_global_sanity_penalty": _validation_global_sanity_penalty(
                    raw_query_useful_score,
                    sanity,
                    model_config,
                ),
                "validation_avg_length_preserved": sanity["avg_length_preserved"],
                "validation_endpoint_sanity": sanity["endpoint_sanity"],
                "validation_avg_sed_km": sanity["avg_sed_km"],
                "validation_uniform_avg_sed_km": sanity["uniform_avg_sed_km"],
                "validation_avg_sed_ratio_vs_uniform": sanity["avg_sed_ratio_vs_uniform"],
                "validation_avg_sed_ratio_vs_uniform_max": sanity["avg_sed_ratio_vs_uniform_max"],
                "range_ship_f1": float(range_audit["range_ship_f1"]),
                "range_ship_coverage": float(range_audit["range_ship_coverage"]),
                "range_entry_exit_f1": float(range_audit["range_entry_exit_f1"]),
                "range_crossing_f1": float(range_audit["range_crossing_f1"]),
                "range_temporal_coverage": float(range_audit["range_temporal_coverage"]),
                "range_gap_coverage": float(range_audit["range_gap_coverage"]),
                "range_turn_coverage": float(range_audit["range_turn_coverage"]),
                "range_shape_score": float(range_audit["range_shape_score"]),
                "range_query_local_interpolation_fidelity": float(
                    range_audit.get("range_query_local_interpolation_fidelity", 0.0)
                ),
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
        raw_score = float(metrics.get("query_useful_v1", 0.0))
        score = float(metrics.get("query_useful_v1_selection_score", raw_score))
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
        sanity = _validation_global_sanity_metrics(
            points=points,
            boundaries=boundaries,
            retained_mask=retained_mask,
            model_config=model_config,
            uniform_retained_mask=retained_mask,
        )
        query_useful = query_useful_v1_from_range_audit(
            audit,
            length_preservation=sanity["avg_length_preserved"],
            avg_sed_km=sanity["avg_sed_km"],
            endpoint_sanity=sanity["endpoint_sanity"],
        )
        raw_score = float(cast(Any, query_useful["query_useful_v1_score"]))
        score = _validation_query_useful_selection_score(raw_score, sanity, model_config)
        return score, {"range": score}
    if variant == "combined":
        return combined_agg, combined_pt
    return answer_agg, answer_pt
