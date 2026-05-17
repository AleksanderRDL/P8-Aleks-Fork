"""Validation scoring helpers for training-time checkpoint selection."""

from __future__ import annotations

import math
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
from simplification.learned_segment_budget import blend_segment_support_scores
from simplification.mlqds_scoring import simplify_mlqds_predictions
from training.inference import _is_workload_blind_model, _model_point_dim
from training.inference import windowed_predict_with_heads
from training.model_features import build_model_point_features_for_dim
from training.query_useful_targets import QUERY_USEFUL_V1_HEAD_NAMES, build_query_useful_v1_targets
from training.scaler import FeatureScaler
from training.training_diagnostics import _discriminative_sample, _kendall_tau
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


def _validation_segment_scores_from_head_logits(head_logits: torch.Tensor | None) -> torch.Tensor | None:
    """Return the segment-budget head scores used by learned-segment validation selection."""
    if head_logits is None:
        return None
    try:
        segment_head_idx = tuple(QUERY_USEFUL_V1_HEAD_NAMES).index("segment_budget_target")
    except ValueError:
        return None
    if int(head_logits.shape[-1]) <= segment_head_idx:
        return None
    return head_logits[:, segment_head_idx].detach().cpu().float()


def _validation_path_length_support_scores_from_head_logits(head_logits: torch.Tensor | None) -> torch.Tensor | None:
    """Return the path-length support head scores used by optional selector blending."""
    if head_logits is None:
        return None
    try:
        path_head_idx = tuple(QUERY_USEFUL_V1_HEAD_NAMES).index("path_length_support_target")
    except ValueError:
        return None
    if int(head_logits.shape[-1]) <= path_head_idx:
        return None
    return head_logits[:, path_head_idx].detach().cpu().float()


def _validation_factorized_target_fit_metrics(
    *,
    head_logits: torch.Tensor | None,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    workload: TypedQueryWorkload,
    segment_size: int = 32,
) -> dict[str, float]:
    """Return validation target-fit diagnostics for factorized heads without affecting selection."""
    metrics: dict[str, float] = {
        "factorized_target_fit_available": 0.0,
        "factorized_target_fit_used_for_checkpoint_selection": 0.0,
    }
    if head_logits is None:
        return metrics
    logits = head_logits.detach().cpu().float()
    if logits.ndim != 2 or int(logits.shape[1]) != len(QUERY_USEFUL_V1_HEAD_NAMES):
        return metrics
    targets = build_query_useful_v1_targets(
        points=points.detach().cpu().float(),
        boundaries=boundaries,
        typed_queries=workload.typed_queries,
        segment_size=segment_size,
    )
    if targets.head_targets.shape != logits.shape or targets.head_mask.shape != logits.shape:
        return metrics

    def top_fraction_count(values: torch.Tensor) -> int:
        return max(1, math.ceil(0.05 * int(values.numel())))

    generator = torch.Generator().manual_seed(17_203)
    probabilities = torch.sigmoid(logits)
    metrics["factorized_target_fit_available"] = 1.0
    for head_idx, head_name in enumerate(QUERY_USEFUL_V1_HEAD_NAMES):
        valid = targets.head_mask[:, head_idx].detach().cpu().bool()
        if not bool(valid.any().item()):
            metrics[f"head_{head_name}_target_fit_available"] = 0.0
            continue
        scores = probabilities[:, head_idx][valid].float()
        head_targets = targets.head_targets[:, head_idx].detach().cpu().float().clamp(0.0, 1.0)[valid]
        sampled_scores, sampled_targets = _discriminative_sample(
            scores,
            head_targets,
            n_each=200,
            generator=generator,
        )
        k = top_fraction_count(scores)
        selected = torch.topk(scores, k=k, largest=True).indices
        ideal = torch.topk(head_targets, k=k, largest=True).indices
        selected_mass = float(head_targets[selected].sum().item())
        ideal_mass = float(head_targets[ideal].sum().item())
        metrics[f"head_{head_name}_target_fit_available"] = 1.0
        metrics[f"head_{head_name}_tau"] = float(_kendall_tau(sampled_scores, sampled_targets))
        metrics[f"head_{head_name}_top5_mass_recall"] = float(selected_mass / max(ideal_mass, 1e-12))
        metrics[f"head_{head_name}_prediction_std"] = (
            float(scores.std(unbiased=False).item()) if int(scores.numel()) > 1 else 0.0
        )

    try:
        segment_head_idx = tuple(QUERY_USEFUL_V1_HEAD_NAMES).index("segment_budget_target")
    except ValueError:
        return metrics
    segment_scores: list[torch.Tensor] = []
    segment_targets: list[torch.Tensor] = []
    size = max(1, int(segment_size))
    for start, end in boundaries:
        for seg_start in range(int(start), int(end), size):
            seg_end = min(int(end), int(seg_start) + size)
            if seg_end <= seg_start:
                continue
            local_valid = targets.head_mask[seg_start:seg_end, segment_head_idx].detach().cpu().bool()
            if not bool(local_valid.any().item()):
                continue
            segment_scores.append(probabilities[seg_start:seg_end, segment_head_idx][local_valid].mean())
            segment_targets.append(
                targets.head_targets[seg_start:seg_end, segment_head_idx].detach().cpu().float()[local_valid].mean()
            )
    if segment_scores:
        pooled_scores = torch.stack(segment_scores).float()
        pooled_targets = torch.stack(segment_targets).float().clamp(0.0, 1.0)
        sampled_scores, sampled_targets = _discriminative_sample(
            pooled_scores,
            pooled_targets,
            n_each=200,
            generator=generator,
        )
        k = top_fraction_count(pooled_scores)
        selected = torch.topk(pooled_scores, k=k, largest=True).indices
        ideal = torch.topk(pooled_targets, k=k, largest=True).indices
        selected_mass = float(pooled_targets[selected].sum().item())
        ideal_mass = float(pooled_targets[ideal].sum().item())
        metrics["segment_budget_canonical_segment_fit_available"] = 1.0
        metrics["segment_budget_canonical_segment_tau"] = float(
            _kendall_tau(sampled_scores, sampled_targets)
        )
        metrics["segment_budget_canonical_segment_top5_mass_recall"] = float(
            selected_mass / max(ideal_mass, 1e-12)
        )
        metrics["segment_budget_canonical_segment_count"] = float(pooled_scores.numel())
    else:
        metrics["segment_budget_canonical_segment_fit_available"] = 0.0
    return metrics


def _validation_selector_segment_scores(
    *,
    segment_scores: torch.Tensor | None,
    path_length_support_scores: torch.Tensor | None,
    model_config: ModelConfig,
) -> torch.Tensor | None:
    """Return the segment-score tensor used by validation selection."""
    return blend_segment_support_scores(
        segment_scores=segment_scores,
        path_length_support_scores=path_length_support_scores,
        path_length_support_weight=float(getattr(model_config, "learned_segment_length_support_blend_weight", 0.0)),
    )


def _neutral_validation_segment_scores_for_ablation(segment_scores: torch.Tensor) -> torch.Tensor:
    """Return neutral segment scores for validation no-segment-head diagnostics."""
    return torch.zeros_like(segment_scores.detach().cpu().float())


def _validation_raw_predictions_without_factorized_head(
    *,
    model: torch.nn.Module,
    head_logits: torch.Tensor,
    disabled_head_name: str,
) -> torch.Tensor | None:
    """Return final raw predictions with one factorized head neutralized, if supported."""
    compose = getattr(model, "final_logit_from_head_logits", None)
    if not callable(compose):
        return None
    compose_fn = cast(Callable[..., torch.Tensor], compose)
    model_device = next(model.parameters()).device
    original_training = bool(model.training)
    try:
        model.eval()
        with torch.no_grad():
            logits = head_logits.detach().to(model_device).unsqueeze(0)
            return compose_fn(
                logits,
                disabled_head_names=(str(disabled_head_name),),
            ).reshape(-1).detach().cpu()
    finally:
        model.train(original_training)


def _validation_retained_mask_from_scores(
    *,
    predictions: torch.Tensor,
    boundaries: list[tuple[int, int]],
    workload_map: dict[str, float],
    model_config: ModelConfig,
    range_geometry_scores: torch.Tensor | None,
    segment_scores: torch.Tensor | None,
    segment_point_scores: torch.Tensor | None,
    points: torch.Tensor,
) -> torch.Tensor:
    """Apply the active validation selector to a score vector."""
    return simplify_mlqds_predictions(
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
        segment_point_scores=segment_point_scores,
        points=points,
        learned_segment_geometry_gain_weight=float(getattr(model_config, "learned_segment_geometry_gain_weight", 0.12)),
        learned_segment_score_blend_weight=float(getattr(model_config, "learned_segment_score_blend_weight", 0.05)),
        learned_segment_fairness_preallocation=bool(
            getattr(model_config, "learned_segment_fairness_preallocation", True)
        ),
        learned_segment_length_repair_fraction=float(
            getattr(model_config, "learned_segment_length_repair_fraction", 0.0)
        ),
    )


def _validation_query_useful_score_for_mask(
    *,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    retained_mask: torch.Tensor,
    workload: TypedQueryWorkload,
    query_cache: Any | None,
    model_config: ModelConfig,
) -> tuple[float, dict[str, Any], dict[str, float]]:
    """Return raw QueryUsefulV1 plus supporting audit/sanity payloads for a validation mask."""
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
    return float(cast(Any, query_useful["query_useful_v1_score"])), range_audit, sanity


def _validation_causality_ablation_metrics(
    *,
    model: torch.nn.Module,
    head_logits: torch.Tensor | None,
    primary_query_useful_score: float,
    predictions: torch.Tensor,
    boundaries: list[tuple[int, int]],
    workload: TypedQueryWorkload,
    workload_map: dict[str, float],
    model_config: ModelConfig,
    points: torch.Tensor,
    query_cache: Any | None,
    range_geometry_scores: torch.Tensor | None,
    segment_scores: torch.Tensor | None,
    segment_budget_scores: torch.Tensor | None,
    path_length_support_scores: torch.Tensor | None,
) -> dict[str, float]:
    """Return checkpoint-validation ablation deltas for factorized-head causality diagnostics."""
    if head_logits is None:
        return {"checkpoint_causality_ablation_available": 0.0}
    metrics: dict[str, float] = {"checkpoint_causality_ablation_available": 1.0}

    no_behavior_predictions = _validation_raw_predictions_without_factorized_head(
        model=model,
        head_logits=head_logits,
        disabled_head_name="conditional_behavior_utility",
    )
    if no_behavior_predictions is not None:
        no_behavior_mask = _validation_retained_mask_from_scores(
            predictions=no_behavior_predictions,
            boundaries=boundaries,
            workload_map=workload_map,
            model_config=model_config,
            range_geometry_scores=range_geometry_scores,
            segment_scores=segment_scores,
            segment_point_scores=segment_budget_scores,
            points=points,
        )
        no_behavior_score, _no_behavior_audit, _no_behavior_sanity = _validation_query_useful_score_for_mask(
            points=points,
            boundaries=boundaries,
            retained_mask=no_behavior_mask,
            workload=workload,
            query_cache=query_cache,
            model_config=model_config,
        )
        metrics["no_behavior_query_useful_v1"] = no_behavior_score
        metrics["no_behavior_query_useful_delta"] = float(primary_query_useful_score - no_behavior_score)

    if segment_budget_scores is not None:
        no_segment_scores = _validation_selector_segment_scores(
            segment_scores=_neutral_validation_segment_scores_for_ablation(segment_budget_scores),
            path_length_support_scores=path_length_support_scores,
            model_config=model_config,
        )
        no_segment_mask = _validation_retained_mask_from_scores(
            predictions=predictions,
            boundaries=boundaries,
            workload_map=workload_map,
            model_config=model_config,
            range_geometry_scores=range_geometry_scores,
            segment_scores=no_segment_scores,
            segment_point_scores=_neutral_validation_segment_scores_for_ablation(segment_budget_scores),
            points=points,
        )
        no_segment_score, _no_segment_audit, _no_segment_sanity = _validation_query_useful_score_for_mask(
            points=points,
            boundaries=boundaries,
            retained_mask=no_segment_mask,
            workload=workload,
            query_cache=query_cache,
            model_config=model_config,
        )
        metrics["no_segment_budget_query_useful_v1"] = no_segment_score
        metrics["no_segment_budget_query_useful_delta"] = float(primary_query_useful_score - no_segment_score)

    return metrics


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
    segment_budget_scores = _validation_segment_scores_from_head_logits(head_logits)
    path_length_support_scores = _validation_path_length_support_scores_from_head_logits(head_logits)
    validation_factorized_fit_metrics = _validation_factorized_target_fit_metrics(
        head_logits=head_logits,
        points=points,
        boundaries=boundaries,
        workload=workload,
    )
    segment_scores = _validation_selector_segment_scores(
        segment_scores=segment_budget_scores,
        path_length_support_scores=path_length_support_scores,
        model_config=model_config,
    )
    retained_mask = _validation_retained_mask_from_scores(
        predictions=predictions,
        boundaries=boundaries,
        workload_map=workload_map,
        model_config=model_config,
        range_geometry_scores=range_geometry_scores,
        segment_scores=segment_scores,
        segment_point_scores=segment_budget_scores,
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
        raw_query_useful_score, range_audit, sanity = _validation_query_useful_score_for_mask(
            points=points,
            boundaries=boundaries,
            retained_mask=retained_mask,
            workload=workload,
            query_cache=query_cache,
            model_config=model_config,
        )
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
        metrics.update(
            _validation_causality_ablation_metrics(
                model=model,
                head_logits=head_logits,
                primary_query_useful_score=raw_query_useful_score,
                predictions=predictions,
                boundaries=boundaries,
                workload=workload,
                workload_map=workload_map,
                model_config=model_config,
                points=points,
                query_cache=query_cache,
                range_geometry_scores=range_geometry_scores,
                segment_scores=segment_scores,
                segment_budget_scores=segment_budget_scores,
                path_length_support_scores=path_length_support_scores,
            )
        )
        metrics.update(validation_factorized_fit_metrics)
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
