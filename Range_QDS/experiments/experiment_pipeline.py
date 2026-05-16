"""Experiment orchestration helpers for training and evaluation runs. See experiments/README.md for details."""

from __future__ import annotations

import time
import copy
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, cast

import torch

from evaluation.baselines import (
    FrozenMaskMethod,
    Method,
    MLQDSMethod,
    OracleMethod,
)
from evaluation.evaluate_methods import evaluate_method
from evaluation.metrics import MethodEvaluation
from evaluation.query_cache import EvaluationQueryCache
from evaluation.range_usefulness import range_usefulness_weight_summary
from evaluation.tables import (
    print_geometric_distortion_table,
    print_method_comparison_table,
    print_range_usefulness_table,
    print_shift_table,
)
from experiments.experiment_config import ExperimentConfig, derive_seed_bundle
from experiments.experiment_data import build_experiment_datasets, prepare_experiment_split
from experiments.geojson_writers import report_trajectory_length_loss, write_queries_geojson, write_simplified_csv
from experiments.experiment_methods import (
    attach_range_geometry_scores,
    build_learned_fill_methods,
    build_primary_methods,
    evaluate_shift_pairs,
    prepare_eval_labels,
    prepare_eval_query_cache,
)
from experiments.experiment_outputs import ExperimentOutputs, write_experiment_results
from experiments.range_cache import (
    RangeRuntimeCache,
    prepare_range_label_cache,
    range_only_queries,
)
from experiments.range_diagnostics import (
    _evaluation_metrics_payload,
    _print_range_diagnostics_summary,
    _print_range_distribution_comparison,
    _range_audit_ratios,
    _range_learned_fill_summary,
    _range_workload_diagnostics,
    _range_workload_distribution_comparison,
)
from experiments.experiment_workloads import (
    generate_experiment_workloads,
    resolve_workload_maps,
    workload_name,
)
from queries.query_types import QUERY_TYPE_ID_RANGE, single_workload_type
from simplification.mlqds_scoring import mlqds_simplification_scores, workload_type_head
from simplification.learned_segment_budget import (
    learned_segment_budget_diagnostics,
    simplify_with_learned_segment_budget_v1,
    simplify_with_learned_segment_budget_v1_with_trace,
)
from simplification.simplify_trajectories import temporal_hybrid_selector_budget_diagnostics
from training.train_model import train_model
from training.checkpoints import ModelArtifacts, save_checkpoint
from training.model_features import is_workload_blind_model_type, model_type_metadata
from training.predictability_audit import query_prior_predictability_audit, query_prior_predictability_scores
from training.query_prior_fields import (
    QUERY_PRIOR_FIELD_NAMES,
    query_prior_field_metadata,
    sample_query_prior_fields,
    zero_query_prior_field_like,
)
from training.training_outputs import TrainingOutputs
from training.teacher_distillation import (
    build_range_teacher_config,
    distill_range_teacher_labels,
    range_teacher_distillation_enabled,
)
from training.training_targets import (
    aggregate_range_component_label_sets,
    aggregate_range_component_retained_frequency_training_labels,
    aggregate_range_continuity_retained_frequency_training_labels,
    aggregate_range_global_budget_retained_frequency_training_labels,
    aggregate_range_label_sets,
    aggregate_range_marginal_coverage_training_labels,
    aggregate_range_retained_frequency_training_labels,
    aggregate_range_structural_retained_frequency_training_labels,
    balance_range_training_target_by_trajectory,
    range_component_retained_frequency_training_labels,
    range_continuity_retained_frequency_training_labels,
    range_global_budget_retained_frequency_training_labels,
    range_historical_prior_retained_frequency_training_labels,
    range_local_swap_gain_cost_frequency_training_labels,
    range_local_swap_utility_frequency_training_labels,
    range_query_residual_frequency_training_labels,
    range_set_utility_frequency_training_labels,
    range_query_spine_frequency_training_labels,
    range_marginal_coverage_training_labels,
    range_retained_frequency_training_labels,
    range_structural_retained_frequency_training_labels,
)
from experiments.torch_runtime import (
    amp_runtime_snapshot,
    cuda_memory_snapshot,
    reset_cuda_peak_memory_stats,
    torch_runtime_snapshot,
)


@contextmanager
def _phase(name: str):
    """Log a named phase with wall-clock timing."""
    print(f"[{name}] starting...", flush=True)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        print(f"[{name}] done in {dt:.2f}s", flush=True)


def _learned_slot_summary(
    selector_budget_diagnostics: dict[str, Any],
    compression_ratio: float,
    selector_trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return learned-slot accounting without pretending budget rows are proof."""
    eval_diagnostics = selector_budget_diagnostics.get("eval")
    if not isinstance(eval_diagnostics, dict):
        return {
            "learned_controlled_retained_slots": None,
            "learned_controlled_retained_slot_fraction": None,
            "learned_slot_accounting_status": "missing_selector_diagnostics",
        }
    rows = eval_diagnostics.get("budget_rows")
    if not isinstance(rows, list) or not rows:
        return {
            "learned_controlled_retained_slots": None,
            "learned_controlled_retained_slot_fraction": None,
            "learned_slot_accounting_status": "missing_budget_rows",
        }
    selected_row = None
    for row in rows:
        if isinstance(row, dict) and abs(float(row.get("compression_ratio", -1.0)) - float(compression_ratio)) <= 1e-9:
            selected_row = row
            break
    if selected_row is None:
        selected_row = rows[0] if isinstance(rows[0], dict) else None
    if selected_row is None:
        return {
            "learned_controlled_retained_slots": None,
            "learned_controlled_retained_slot_fraction": None,
            "learned_slot_accounting_status": "invalid_budget_row",
        }
    planned_slots = int(selected_row.get("learned_slot_count", 0))
    planned_fraction = float(selected_row.get("learned_slot_fraction_of_budget", 0.0))
    summary = {
        "learned_controlled_retained_slots": planned_slots,
        "learned_controlled_retained_slot_fraction": planned_fraction,
        "total_retained_slot_budget": int(selected_row.get("total_budget_count", 0)),
        "minimal_skeleton_slot_cap": int(selected_row.get("minimal_skeleton_slot_cap", 0)),
        "no_fixed_85_percent_temporal_scaffold": bool(
            selected_row.get("no_fixed_85_percent_temporal_scaffold", False)
        ),
        "planned_learned_controlled_retained_slots": planned_slots,
        "planned_learned_controlled_retained_slot_fraction": planned_fraction,
        "learned_slot_accounting_status": "budget_level_accounting_only",
        "learned_slot_accounting_note": (
            "Counts planned learned-controlled selector budget. "
            "Per-retained-point skeleton-vs-learned attribution is not yet recorded."
        ),
    }
    if not isinstance(selector_trace, dict) or not selector_trace.get("point_attribution_available"):
        return summary

    actual_slots = int(selector_trace.get("learned_controlled_retained_slots", 0))
    actual_fraction = float(selector_trace.get("learned_controlled_retained_slot_fraction", 0.0))
    summary.update(
        {
            "learned_controlled_retained_slots": actual_slots,
            "learned_controlled_retained_slot_fraction": actual_fraction,
            "actual_learned_controlled_retained_slots": actual_slots,
            "actual_learned_controlled_retained_slot_fraction": actual_fraction,
            "skeleton_retained_count": int(selector_trace.get("skeleton_retained_count", 0)),
            "fallback_retained_count": int(selector_trace.get("fallback_retained_count", 0)),
            "unattributed_retained_count": int(selector_trace.get("unattributed_retained_count", 0)),
            "trajectories_with_at_least_one_learned_decision": int(
                selector_trace.get("trajectories_with_at_least_one_learned_decision", 0)
            ),
            "trajectories_with_zero_learned_decisions": int(
                selector_trace.get("trajectories_with_zero_learned_decisions", 0)
            ),
            "segment_budget_entropy": float(selector_trace.get("segment_budget_entropy", 0.0)),
            "segment_budget_entropy_normalized": float(
                selector_trace.get("segment_budget_entropy_normalized", 0.0)
            ),
            "segments_with_learned_budget": int(selector_trace.get("segments_with_learned_budget", 0)),
            "learned_slot_accounting_status": "point_attribution_available",
            "learned_slot_accounting_note": (
                "Counts actual retained points attributed to skeleton, learned segment allocation, "
                "or fallback fill after masks were frozen."
            ),
        }
    )
    if "retained_mask_matches_frozen_primary" in selector_trace:
        summary["selector_trace_retained_mask_matches_primary"] = bool(
            selector_trace.get("retained_mask_matches_frozen_primary")
        )
    return summary


def _learned_segment_frozen_method(
    *,
    name: str,
    scores: torch.Tensor,
    boundaries: list[tuple[int, int]],
    compression_ratio: float,
    segment_scores: torch.Tensor | None = None,
    points: torch.Tensor | None = None,
    learned_segment_geometry_gain_weight: float = 0.12,
    learned_segment_score_blend_weight: float = 0.05,
    learned_segment_fairness_preallocation: bool = True,
) -> FrozenMaskMethod:
    """Freeze a score-based learned-segment diagnostic mask before query scoring."""
    retained_mask = simplify_with_learned_segment_budget_v1(
        scores.detach().cpu().float(),
        boundaries,
        compression_ratio,
        segment_scores=None if segment_scores is None else segment_scores.detach().cpu().float(),
        points=None if points is None else points.detach().cpu().float(),
        geometry_gain_weight=float(learned_segment_geometry_gain_weight),
        segment_score_point_blend_weight=float(learned_segment_score_blend_weight),
        fairness_preallocation_enabled=bool(learned_segment_fairness_preallocation),
    )
    return FrozenMaskMethod(name=name, retained_mask=retained_mask.detach().cpu())


def _neutral_segment_scores_for_ablation(segment_scores: torch.Tensor) -> torch.Tensor:
    """Return neutral segment scores for the no-segment-budget-head ablation."""
    return torch.zeros_like(segment_scores.detach().cpu().float())


def _query_useful_delta(
    primary: MethodEvaluation,
    ablations: dict[str, MethodEvaluation],
    name: str,
) -> float | None:
    """Return primary minus ablation QueryUsefulV1 if the ablation exists."""
    ablation = ablations.get(name)
    if ablation is None:
        return None
    return float(primary.query_useful_v1_score - ablation.query_useful_v1_score)


LEARNING_CAUSALITY_MIN_MATERIAL_DELTA = 0.005
SHUFFLED_SCORE_DELTA_FRACTION_OF_UNIFORM_GAP_MIN = 0.60


def _learning_causality_delta_gate_config(
    *,
    primary: MethodEvaluation,
    uniform: MethodEvaluation | None,
) -> dict[str, Any]:
    """Return material QueryUsefulV1 delta thresholds for learning-causality checks."""
    min_delta = float(LEARNING_CAUSALITY_MIN_MATERIAL_DELTA)
    thresholds = {
        "shuffled_scores_should_lose": min_delta,
        "untrained_model_should_lose": min_delta,
        "shuffled_prior_fields_should_lose": min_delta,
        "without_query_prior_features_should_lose": min_delta,
        "without_behavior_utility_head_should_lose": min_delta,
        "without_segment_budget_head_should_lose": min_delta,
        "prior_field_only_should_not_match_trained": min_delta,
    }
    uniform_gap = None
    if uniform is not None:
        uniform_gap = float(primary.query_useful_v1_score - uniform.query_useful_v1_score)
        if uniform_gap > 0.0:
            thresholds["shuffled_scores_should_lose"] = max(
                min_delta,
                float(SHUFFLED_SCORE_DELTA_FRACTION_OF_UNIFORM_GAP_MIN) * uniform_gap,
            )
    return {
        "min_material_query_useful_delta": min_delta,
        "shuffled_score_delta_fraction_of_uniform_gap_min": float(
            SHUFFLED_SCORE_DELTA_FRACTION_OF_UNIFORM_GAP_MIN
        ),
        "mlqds_uniform_query_useful_gap": uniform_gap,
        "thresholds": thresholds,
    }


def _score_ablation_sensitivity(
    *,
    primary_scores: torch.Tensor | None,
    ablation_scores: torch.Tensor | None,
    primary_mask: torch.Tensor | None,
    ablation_mask: torch.Tensor | None,
) -> dict[str, Any]:
    """Return score- and mask-level sensitivity for a frozen ablation."""
    if primary_scores is None or ablation_scores is None:
        return {"available": False, "reason": "missing_scores"}
    primary = primary_scores.detach().cpu().float().flatten()
    ablation = ablation_scores.detach().cpu().float().flatten()
    if int(primary.numel()) == 0 or primary.shape != ablation.shape:
        return {
            "available": False,
            "reason": "score_shape_mismatch",
            "primary_score_count": int(primary.numel()),
            "ablation_score_count": int(ablation.numel()),
        }
    finite = torch.isfinite(primary) & torch.isfinite(ablation)
    if not bool(finite.any().item()):
        return {"available": False, "reason": "no_finite_scores"}
    primary_f = primary[finite]
    ablation_f = ablation[finite]
    delta = primary_f - ablation_f
    primary_std = float(primary_f.std(unbiased=False).item()) if int(primary_f.numel()) > 1 else 0.0
    ablation_std = float(ablation_f.std(unbiased=False).item()) if int(ablation_f.numel()) > 1 else 0.0

    retained_jaccard: float | None = None
    retained_hamming_fraction: float | None = None
    retained_changed: bool | None = None
    topk_jaccard: float | None = None
    retained_count: int | None = None
    if primary_mask is not None and ablation_mask is not None:
        primary_bool = primary_mask.detach().cpu().bool().flatten()
        ablation_bool = ablation_mask.detach().cpu().bool().flatten()
        if primary_bool.shape == ablation_bool.shape == primary.shape:
            intersection = int((primary_bool & ablation_bool).sum().item())
            union = int((primary_bool | ablation_bool).sum().item())
            retained_jaccard = float(intersection / max(1, union))
            retained_hamming_fraction = float((primary_bool != ablation_bool).float().mean().item())
            retained_changed = bool(not torch.equal(primary_bool, ablation_bool))
            retained_count = int(primary_bool.sum().item())
            if retained_count > 0:
                k = min(retained_count, int(primary.numel()))
                primary_top = torch.zeros_like(primary_bool)
                ablation_top = torch.zeros_like(ablation_bool)
                primary_top[torch.topk(primary, k=k, largest=True).indices] = True
                ablation_top[torch.topk(ablation, k=k, largest=True).indices] = True
                top_intersection = int((primary_top & ablation_top).sum().item())
                top_union = int((primary_top | ablation_top).sum().item())
                topk_jaccard = float(top_intersection / max(1, top_union))

    return {
        "available": True,
        "score_count": int(primary.numel()),
        "finite_score_count": int(finite.sum().item()),
        "mean_abs_score_delta": float(delta.abs().mean().item()),
        "max_abs_score_delta": float(delta.abs().max().item()),
        "mean_signed_score_delta": float(delta.mean().item()),
        "primary_score_std": primary_std,
        "ablation_score_std": ablation_std,
        "retained_count": retained_count,
        "retained_mask_changed": retained_changed,
        "retained_mask_jaccard": retained_jaccard,
        "retained_mask_hamming_fraction": retained_hamming_fraction,
        "score_topk_jaccard_at_retained_count": topk_jaccard,
    }


def _prior_feature_sample_sensitivity(
    *,
    points: torch.Tensor,
    primary_prior_field: dict[str, Any] | None,
    ablation_prior_field: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return sampled query-prior feature sensitivity at eval-compression points."""
    if primary_prior_field is None:
        return {"available": False, "reason": "missing_primary_prior_field"}
    primary = sample_query_prior_fields(points, primary_prior_field).detach().cpu().float()
    ablation = sample_query_prior_fields(points, ablation_prior_field).detach().cpu().float()
    if int(primary.numel()) == 0 or primary.shape != ablation.shape:
        return {
            "available": False,
            "reason": "sample_shape_mismatch",
            "primary_shape": list(primary.shape),
            "ablation_shape": list(ablation.shape),
        }
    finite = torch.isfinite(primary) & torch.isfinite(ablation)
    if not bool(finite.any().item()):
        return {"available": False, "reason": "no_finite_sampled_features"}
    delta = primary - ablation
    finite_delta = delta[finite]
    per_feature: dict[str, dict[str, float | int]] = {}
    feature_names = list(QUERY_PRIOR_FIELD_NAMES)
    for idx in range(int(primary.shape[1])):
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        primary_col = primary[:, idx]
        ablation_col = ablation[:, idx]
        col_finite = torch.isfinite(primary_col) & torch.isfinite(ablation_col)
        if not bool(col_finite.any().item()):
            per_feature[name] = {"finite_count": 0}
            continue
        primary_f = primary_col[col_finite]
        ablation_f = ablation_col[col_finite]
        col_delta = primary_f - ablation_f
        per_feature[name] = {
            "finite_count": int(col_finite.sum().item()),
            "mean_abs_delta": float(col_delta.abs().mean().item()),
            "max_abs_delta": float(col_delta.abs().max().item()),
            "primary_mean": float(primary_f.mean().item()),
            "ablation_mean": float(ablation_f.mean().item()),
            "primary_std": float(primary_f.std(unbiased=False).item()) if int(primary_f.numel()) > 1 else 0.0,
            "ablation_std": float(ablation_f.std(unbiased=False).item()) if int(ablation_f.numel()) > 1 else 0.0,
            "primary_nonzero_fraction": float((primary_f.abs() > 1e-12).float().mean().item()),
            "ablation_nonzero_fraction": float((ablation_f.abs() > 1e-12).float().mean().item()),
        }
    primary_flat = primary[torch.isfinite(primary)]
    ablation_flat = ablation[torch.isfinite(ablation)]
    outside_extent_fraction: float | None = None
    extent = primary_prior_field.get("extent")
    if isinstance(extent, dict) and int(points.shape[0]) > 0:
        lat = points[:, 1].detach().cpu().float()
        lon = points[:, 2].detach().cpu().float()
        outside = (
            (lat < float(extent.get("lat_min", -float("inf"))))
            | (lat > float(extent.get("lat_max", float("inf"))))
            | (lon < float(extent.get("lon_min", -float("inf"))))
            | (lon > float(extent.get("lon_max", float("inf"))))
        )
        outside_extent_fraction = float(outside.float().mean().item())
    return {
        "available": True,
        "point_count": int(primary.shape[0]),
        "feature_count": int(primary.shape[1]),
        "finite_value_count": int(finite.sum().item()),
        "sampled_inputs_changed": bool(float(finite_delta.abs().max().item()) > 1e-9),
        "mean_abs_feature_delta": float(finite_delta.abs().mean().item()),
        "max_abs_feature_delta": float(finite_delta.abs().max().item()),
        "mean_signed_feature_delta": float(finite_delta.mean().item()),
        "primary_feature_mean": float(primary_flat.mean().item()) if int(primary_flat.numel()) > 0 else 0.0,
        "ablation_feature_mean": float(ablation_flat.mean().item()) if int(ablation_flat.numel()) > 0 else 0.0,
        "primary_feature_std": (
            float(primary_flat.std(unbiased=False).item()) if int(primary_flat.numel()) > 1 else 0.0
        ),
        "ablation_feature_std": (
            float(ablation_flat.std(unbiased=False).item()) if int(ablation_flat.numel()) > 1 else 0.0
        ),
        "primary_nonzero_fraction": float((primary.abs() > 1e-12).float().mean().item()),
        "ablation_nonzero_fraction": float((ablation.abs() > 1e-12).float().mean().item()),
        "points_outside_prior_extent_fraction": outside_extent_fraction,
        "per_feature": per_feature,
    }


def _prior_sample_gate_failures(prior_sensitivity_diagnostics: dict[str, Any]) -> list[str]:
    """Return failures showing prior-feature ablations did not exercise useful inputs."""
    shuffled = prior_sensitivity_diagnostics.get("shuffled_prior_fields")
    if not isinstance(shuffled, dict):
        return []
    sampled = shuffled.get("sampled_prior_features")
    if not isinstance(sampled, dict) or not sampled.get("available"):
        return []
    failures: list[str] = []
    primary_nonzero = float(sampled.get("primary_nonzero_fraction") or 0.0)
    if primary_nonzero <= 1e-6:
        failures.append("sampled_query_prior_features_all_zero")
    if not bool(sampled.get("sampled_inputs_changed", False)):
        failures.append("shuffled_prior_fields_did_not_change_sampled_inputs")
    outside_fraction = sampled.get("points_outside_prior_extent_fraction")
    if outside_fraction is not None and float(outside_fraction) > 0.50:
        failures.append("eval_points_mostly_outside_query_prior_extent")
    return failures


def _points_outside_prior_extent_fraction(points: torch.Tensor, extent: dict[str, Any] | None) -> float | None:
    """Return the fraction of points outside a train-prior spatial extent."""
    if not isinstance(extent, dict) or int(points.shape[0]) <= 0:
        return None
    lat = points[:, 1].detach().cpu().float()
    lon = points[:, 2].detach().cpu().float()
    outside = (
        (lat < float(extent.get("lat_min", -float("inf"))))
        | (lat > float(extent.get("lat_max", float("inf"))))
        | (lon < float(extent.get("lon_min", -float("inf"))))
        | (lon > float(extent.get("lon_max", float("inf"))))
    )
    return float(outside.float().mean().item())


def _spatial_extent_intersection_fraction(train_points: torch.Tensor, eval_points: torch.Tensor) -> float | None:
    """Return train/eval lat-lon extent intersection as a fraction of eval extent area."""
    if int(train_points.shape[0]) <= 0 or int(eval_points.shape[0]) <= 0:
        return None
    train_lat_min = float(train_points[:, 1].min().item())
    train_lat_max = float(train_points[:, 1].max().item())
    train_lon_min = float(train_points[:, 2].min().item())
    train_lon_max = float(train_points[:, 2].max().item())
    eval_lat_min = float(eval_points[:, 1].min().item())
    eval_lat_max = float(eval_points[:, 1].max().item())
    eval_lon_min = float(eval_points[:, 2].min().item())
    eval_lon_max = float(eval_points[:, 2].max().item())
    eval_lat_span = eval_lat_max - eval_lat_min
    eval_lon_span = eval_lon_max - eval_lon_min
    if eval_lat_span <= 1e-12 or eval_lon_span <= 1e-12:
        inside = (
            eval_lat_min >= train_lat_min - 1e-12
            and eval_lat_max <= train_lat_max + 1e-12
            and eval_lon_min >= train_lon_min - 1e-12
            and eval_lon_max <= train_lon_max + 1e-12
        )
        return 1.0 if inside else 0.0
    lat_overlap = max(0.0, min(train_lat_max, eval_lat_max) - max(train_lat_min, eval_lat_min))
    lon_overlap = max(0.0, min(train_lon_max, eval_lon_max) - max(train_lon_min, eval_lon_min))
    eval_area = max(1e-12, eval_lat_span * eval_lon_span)
    return float(max(0.0, min(1.0, (lat_overlap * lon_overlap) / eval_area)))


def _support_overlap_gate(
    *,
    train_points: torch.Tensor,
    eval_points: torch.Tensor,
    query_prior_field: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return train/eval support-overlap evidence for final query-driven claims."""
    thresholds = {
        "eval_points_outside_train_prior_extent_fraction_max": 0.10,
        "sampled_prior_nonzero_fraction_min": 0.50,
        "primary_sampled_prior_nonzero_fraction_min": 0.30,
        "route_density_overlap_min": 0.25,
        "query_prior_support_overlap_min": 0.25,
    }
    if query_prior_field is None:
        return {
            "schema_version": 1,
            "gate_pass": False,
            "failed_checks": ["query_prior_field_missing"],
            **thresholds,
            "eval_points_outside_train_prior_extent_fraction": None,
            "sampled_prior_nonzero_fraction": 0.0,
            "primary_sampled_prior_nonzero_fraction": 0.0,
            "route_density_overlap": 0.0,
            "query_prior_support_overlap": 0.0,
            "train_eval_spatial_extent_intersection_fraction": _spatial_extent_intersection_fraction(
                train_points,
                eval_points,
            ),
        }
    sampled = sample_query_prior_fields(eval_points, query_prior_field).detach().cpu().float()
    if int(sampled.numel()) == 0:
        sampled_any = torch.zeros((int(eval_points.shape[0]),), dtype=torch.bool)
        primary = sampled_any
        route = sampled_any
        query_support = sampled_any
    else:
        sampled_any = (sampled.abs() > 1e-12).any(dim=1)
        feature_names: tuple[str, ...] = tuple(QUERY_PRIOR_FIELD_NAMES)

        def col(name: str) -> torch.Tensor:
            try:
                idx = feature_names.index(name)
            except ValueError:
                return torch.zeros((int(sampled.shape[0]),), dtype=torch.bool)
            if idx >= int(sampled.shape[1]):
                return torch.zeros((int(sampled.shape[0]),), dtype=torch.bool)
            return sampled[:, idx].abs() > 1e-12

        primary = col("spatial_query_hit_probability")
        spatiotemporal = col("spatiotemporal_query_hit_probability")
        route = col("route_density_prior")
        query_support = primary | spatiotemporal
    point_count = max(1, int(eval_points.shape[0]))
    outside = _points_outside_prior_extent_fraction(eval_points, query_prior_field.get("extent"))
    sampled_fraction = float(sampled_any.float().sum().item() / point_count)
    primary_fraction = float(primary.float().sum().item() / point_count)
    route_fraction = float(route.float().sum().item() / point_count)
    query_support_fraction = float(query_support.float().sum().item() / point_count)
    failed_checks: list[str] = []
    if outside is None:
        failed_checks.append("train_prior_extent_missing")
    elif outside > thresholds["eval_points_outside_train_prior_extent_fraction_max"] + 1e-12:
        failed_checks.append("eval_points_outside_train_prior_extent_too_high")
    if sampled_fraction + 1e-12 < thresholds["sampled_prior_nonzero_fraction_min"]:
        failed_checks.append("sampled_prior_nonzero_fraction_too_low")
    if primary_fraction + 1e-12 < thresholds["primary_sampled_prior_nonzero_fraction_min"]:
        failed_checks.append("primary_sampled_prior_nonzero_fraction_too_low")
    if route_fraction + 1e-12 < thresholds["route_density_overlap_min"]:
        failed_checks.append("route_density_overlap_too_low")
    if query_support_fraction + 1e-12 < thresholds["query_prior_support_overlap_min"]:
        failed_checks.append("query_prior_support_overlap_too_low")
    return {
        "schema_version": 1,
        "gate_pass": not failed_checks,
        "failed_checks": failed_checks,
        **thresholds,
        "eval_points_outside_train_prior_extent_fraction": outside,
        "sampled_prior_nonzero_fraction": sampled_fraction,
        "primary_sampled_prior_nonzero_fraction": primary_fraction,
        "route_density_overlap": route_fraction,
        "query_prior_support_overlap": query_support_fraction,
        "train_eval_spatial_extent_intersection_fraction": _spatial_extent_intersection_fraction(
            train_points,
            eval_points,
        ),
    }


def _reset_module_parameters(module: torch.nn.Module, seed: int) -> torch.nn.Module:
    """Return a deepcopy with reset trainable parameters for untrained-model ablations."""
    clone = copy.deepcopy(module)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        for child in clone.modules():
            reset_parameters = getattr(child, "reset_parameters", None)
            if callable(reset_parameters):
                reset_parameters()
    return clone


def _shuffled_query_prior_field(prior_field: dict[str, Any], seed: int) -> dict[str, Any]:
    """Return a copy of query-prior fields with spatial associations broken."""
    shuffled: dict[str, Any] = {}
    generator = torch.Generator().manual_seed(int(seed))
    for key, value in prior_field.items():
        if isinstance(value, torch.Tensor) and value.numel() > 1:
            flat = value.detach().cpu().flatten()
            order = torch.randperm(int(flat.numel()), generator=generator)
            shuffled[key] = flat[order].reshape(value.shape).to(dtype=value.dtype)
        else:
            shuffled[key] = copy.deepcopy(value)
    shuffled["ablation"] = "shuffled_prior_fields"
    shuffled["contains_eval_queries"] = False
    shuffled["contains_validation_queries"] = False
    return shuffled


def _scores_without_factorized_head(
    *,
    model: torch.nn.Module,
    head_logits: torch.Tensor,
    disabled_head_name: str,
    boundaries: list[tuple[int, int]],
    workload_type: str,
    score_mode: str,
    score_temperature: float,
    rank_confidence_weight: float,
) -> torch.Tensor:
    """Return simplification scores with one factorized head neutralized."""
    compose = getattr(model, "final_logit_from_head_logits", None)
    if not callable(compose):
        raise RuntimeError(f"{type(model).__name__} does not expose final_logit_from_head_logits.")
    compose_fn = cast(Callable[..., torch.Tensor], compose)
    model_device = next(model.parameters()).device
    original_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            logits = head_logits.detach().to(model_device).unsqueeze(0)
            pred = compose_fn(
                logits,
                disabled_head_names=(str(disabled_head_name),),
            ).reshape(-1).detach().cpu()
    finally:
        model.train(original_training)
    return mlqds_simplification_scores(
        pred,
        boundaries,
        workload_type,
        score_mode=score_mode,
        score_temperature=score_temperature,
        rank_confidence_weight=rank_confidence_weight,
    )


def _normalize_fraction_for_gate(value: Any) -> float | None:
    """Normalize optional fraction/percent values for gate checks."""
    if not isinstance(value, (int, float)):
        return None
    out = float(value)
    if out > 1.0 and out <= 100.0:
        out /= 100.0
    return out


def _optional_float_for_gate(value: Any) -> float | None:
    """Coerce optional gate evidence to float without percent normalization."""
    if not isinstance(value, (int, float)):
        return None
    return float(value)


def _workload_stability_gate(
    *,
    config: ExperimentConfig,
    train_label_workloads: list[Any],
    eval_workload: Any,
    selection_workload: Any | None,
) -> dict[str, Any]:
    """Return final-candidate gate evidence for statistically stable workloads."""
    allowed_coverage_targets = (0.05, 0.10, 0.15, 0.30)
    min_train_replicates = 4
    min_queries_per_workload = 8
    gate_mode = str(getattr(config.query, "workload_stability_gate_mode", "final")).lower()
    required_profile_id = "range_workload_v1"
    coverage_tolerance = 1e-6
    failed_checks: list[str] = []

    configured_target = _normalize_fraction_for_gate(getattr(config.query, "target_coverage", None))
    configured_target_in_grid = (
        configured_target is not None
        and any(abs(configured_target - target) <= 1e-9 for target in allowed_coverage_targets)
    )
    if not configured_target_in_grid:
        failed_checks.append("coverage_target_not_in_final_grid")
    if len(train_label_workloads) < min_train_replicates:
        failed_checks.append("too_few_train_workload_replicates")

    overshoot = _normalize_fraction_for_gate(getattr(config.query, "range_max_coverage_overshoot", None))
    max_allowed_overshoot = _coverage_overshoot_tolerance_for_target(configured_target)
    if max_allowed_overshoot is None or overshoot is None or overshoot > max_allowed_overshoot + 1e-12:
        failed_checks.append("coverage_overshoot_tolerance_too_loose")
    workload_rows: list[dict[str, Any]] = []
    workloads: list[tuple[str, Any]] = [
        *[(f"train_r{idx}", workload) for idx, workload in enumerate(train_label_workloads)],
        ("eval", eval_workload),
    ]
    if selection_workload is not None:
        workloads.append(("selection", selection_workload))

    for label, workload in workloads:
        generation = (getattr(workload, "generation_diagnostics", None) or {}).get("query_generation", {})
        if not isinstance(generation, dict):
            generation = {}
        profile_id = str(generation.get("workload_profile_id", ""))
        mode = str(generation.get("mode", ""))
        query_count = len(getattr(workload, "typed_queries", []) or [])
        query_count_mode = str(generation.get("query_count_mode", ""))
        target = _normalize_fraction_for_gate(generation.get("target_coverage"))
        final_coverage = _normalize_fraction_for_gate(getattr(workload, "coverage_fraction", None))
        coverage_mode = str(generation.get("coverage_calibration_mode", ""))
        coverage_guard_enabled = bool(generation.get("coverage_guard_enabled", False))
        stop_reason = str(generation.get("stop_reason", ""))
        is_calibrated_query_count_mode = query_count_mode == "calibrated_to_coverage"
        row_min_queries_per_workload = 1 if gate_mode == "smoke" and is_calibrated_query_count_mode else min_queries_per_workload
        acceptance = (getattr(workload, "generation_diagnostics", None) or {}).get("range_acceptance", {})
        if not isinstance(acceptance, dict):
            acceptance = {}
        row_failed: list[str] = []
        if profile_id != required_profile_id:
            row_failed.append("wrong_workload_profile")
        if mode != "target_coverage":
            row_failed.append("not_target_coverage_generation")
        if coverage_mode != "profile_sampled_query_count":
            row_failed.append("coverage_calibration_not_profile_sampled")
        if query_count < row_min_queries_per_workload:
            row_failed.append("too_few_queries")
        if gate_mode != "smoke":
            if (
                bool(acceptance.get("exhausted", False))
                or stop_reason == "range_acceptance_exhausted"
                or stop_reason == "range_coverage_guard_exhausted"
            ):
                row_failed.append("range_acceptance_or_coverage_guard_exhausted")
            attempts = int(acceptance.get("attempts", 0) or 0)
            rejected = int(acceptance.get("rejected", 0) or 0)
            accepted = int(acceptance.get("accepted", 0) or 0)
            rejection_rate = float(rejected / max(1, attempts))
            if attempts > 0 and rejection_rate > 0.85:
                row_failed.append("range_generation_rejection_rate_too_high")
            coverage_rejections = int((acceptance.get("rejection_reasons", {}) or {}).get("coverage_overshoot", 0))
            if accepted > 0 and coverage_rejections / max(1, accepted) > 2.0:
                row_failed.append("coverage_guard_rejection_pressure_too_high")
        if not coverage_guard_enabled:
            row_failed.append("coverage_guard_disabled")
        coverage_target_satisfied = False
        if configured_target is not None and target is not None and abs(target - configured_target) > 1e-9:
            row_failed.append("target_coverage_mismatch")
        if target is not None and final_coverage is not None:
            if final_coverage + coverage_tolerance < target:
                row_failed.append("coverage_below_target")
            if overshoot is not None and final_coverage > min(1.0, target + overshoot) + coverage_tolerance:
                row_failed.append("coverage_above_guard")
            coverage_target_satisfied = (
                final_coverage + coverage_tolerance >= target
                and (
                    overshoot is None
                    or final_coverage <= min(1.0, target + overshoot) + coverage_tolerance
                )
            )
        else:
            row_failed.append("missing_coverage_fields")
        if stop_reason != "target_coverage_reached" and not coverage_target_satisfied:
            row_failed.append("target_coverage_not_reached")
        failed_checks.extend(f"{label}:{check}" for check in row_failed)
        workload_rows.append(
            {
                "label": label,
                "profile_id": profile_id,
                "mode": mode,
                "coverage_calibration_mode": coverage_mode,
                "query_count_mode": query_count_mode,
                "query_count": int(query_count),
                "target_coverage": target,
                "final_coverage": final_coverage,
                "coverage_guard_enabled": coverage_guard_enabled,
                "stop_reason": stop_reason,
                "range_acceptance": acceptance,
                "coverage_target_satisfied": bool(coverage_target_satisfied),
                "failed_checks": row_failed,
            }
        )

    return {
        "schema_version": 1,
        "gate_pass": not failed_checks,
        "failed_checks": failed_checks,
        "configured_target_coverage": configured_target,
        "allowed_coverage_targets": list(allowed_coverage_targets),
        "configured_target_in_grid": bool(configured_target_in_grid),
        "gate_mode": gate_mode,
        "train_workload_replicate_count": int(len(train_label_workloads)),
        "min_train_workload_replicates": int(min_train_replicates),
        "min_queries_per_workload": int(min_queries_per_workload),
        "range_max_coverage_overshoot": overshoot,
        "max_allowed_coverage_overshoot": max_allowed_overshoot,
        "required_profile_id": required_profile_id,
        "workloads": workload_rows,
    }


def _coverage_overshoot_tolerance_for_target(target: float | None) -> float | None:
    """Return guide-recommended absolute coverage overshoot tolerance."""
    if target is None:
        return None
    if target <= 0.05:
        return 0.005
    if target <= 0.10:
        return 0.0075
    if target <= 0.15:
        return 0.010
    return 0.020


def _global_sanity_gate(
    *,
    primary: MethodEvaluation,
    uniform: MethodEvaluation | None,
    compression_ratio: float,
) -> dict[str, Any]:
    """Return final-candidate geometry sanity gate evidence."""
    failed_checks: list[str] = []
    length_min = 0.80
    length_max = 1.20
    sed_ratio_threshold = (
        2.00
        if compression_ratio <= 0.01 + 1e-12
        else 1.75
        if compression_ratio <= 0.02 + 1e-12
        else 1.50
    )

    length_preserved = float(primary.avg_length_preserved)
    if length_preserved < length_min or length_preserved > length_max:
        failed_checks.append("length_preservation_outside_range")

    endpoint_sanity_raw = (
        primary.range_audit.get("endpoint_sanity") if isinstance(primary.range_audit, dict) else None
    )
    endpoint_sanity = _normalize_fraction_for_gate(endpoint_sanity_raw)
    if endpoint_sanity is None:
        failed_checks.append("endpoint_sanity_missing")
    elif endpoint_sanity < 1.0 - 1e-12:
        failed_checks.append("endpoints_not_retained_for_all_eligible_trajectories")

    primary_avg_sed = _optional_float_for_gate(primary.geometric_distortion.get("avg_sed_km"))
    uniform_avg_sed = (
        _optional_float_for_gate(uniform.geometric_distortion.get("avg_sed_km"))
        if uniform is not None
        else None
    )
    if primary_avg_sed is None or uniform_avg_sed is None:
        sed_ratio = None
        failed_checks.append("avg_sed_ratio_missing")
    elif uniform_avg_sed <= 1e-12:
        sed_ratio = 1.0 if primary_avg_sed <= 1e-12 else float("inf")
    else:
        sed_ratio = float(primary_avg_sed / uniform_avg_sed)
    if sed_ratio is not None and sed_ratio > sed_ratio_threshold + 1e-12:
        failed_checks.append("avg_sed_ratio_vs_uniform_too_high")

    return {
        "schema_version": 1,
        "gate_pass": not failed_checks,
        "failed_checks": failed_checks,
        "compression_ratio": float(compression_ratio),
        "endpoint_sanity": endpoint_sanity,
        "endpoint_sanity_required": 1.0,
        "avg_length_preserved": length_preserved,
        "length_preservation_min": length_min,
        "length_preservation_max": length_max,
        "avg_sed_km": primary_avg_sed,
        "uniform_avg_sed_km": uniform_avg_sed,
        "avg_sed_ratio_vs_uniform": sed_ratio,
        "avg_sed_ratio_vs_uniform_max": sed_ratio_threshold,
        "catastrophic_geometry_outlier_fraction": None,
        "catastrophic_geometry_outlier_fraction_max": 0.05,
        "catastrophic_geometry_outlier_status": "not_available_report_only",
    }


def _target_diffusion_gate(target_diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Return gate evidence for labels that are too diffuse for low-budget ranking."""
    max_support_fraction = 0.50
    min_top5_mass_fraction = 0.10
    final_support_threshold_key = "gt_0.01"
    default_head_support_threshold_key = "gt_0.01"
    head_support_threshold_keys = {
        "boundary_event_utility": "gt_0.05",
        "conditional_behavior_utility": "gt_0.01",
        "replacement_representative_value": "gt_0.05",
    }
    blocking_heads = frozenset(head_support_threshold_keys)
    low_budget_key = "0.05"
    failed_checks: list[str] = []

    factorized = target_diagnostics.get("query_useful_v1_factorized")
    if not isinstance(factorized, dict):
        return {
            "schema_version": 1,
            "gate_pass": False,
            "failed_checks": ["query_useful_v1_factorized_diagnostics_missing"],
            "final_support_threshold_key": final_support_threshold_key,
            "max_support_fraction": max_support_fraction,
            "min_top5_label_mass_fraction": min_top5_mass_fraction,
            "blocking_heads": sorted(blocking_heads),
            "head_rows": [],
        }

    final_supports = factorized.get("final_label_support_fraction_by_threshold")
    final_support = None
    if isinstance(final_supports, dict):
        final_support = _optional_float_for_gate(final_supports.get(final_support_threshold_key))
    if final_support is None:
        final_support = _optional_float_for_gate(factorized.get("final_label_positive_fraction"))
    if final_support is None:
        failed_checks.append("final_label_support_fraction_missing")
    elif final_support > max_support_fraction + 1e-12:
        failed_checks.append("final_label_support_fraction_above_max")

    support_by_head = factorized.get("support_fraction_by_threshold_by_head")
    if not isinstance(support_by_head, dict):
        support_by_head = {}
    positive_by_head = factorized.get("positive_fraction_by_head")
    if not isinstance(positive_by_head, dict):
        positive_by_head = {}
    topk_by_head = factorized.get("topk_label_mass_budget_grid")
    if not isinstance(topk_by_head, dict):
        topk_by_head = {}

    head_names = sorted(set(support_by_head) | set(positive_by_head) | set(topk_by_head))
    head_rows: list[dict[str, Any]] = []
    if not head_names:
        failed_checks.append("head_diffusion_diagnostics_missing")
    for head_name in head_names:
        blocking = head_name in blocking_heads
        support_threshold_key = head_support_threshold_keys.get(head_name, default_head_support_threshold_key)
        head_supports = support_by_head.get(head_name)
        support_fraction = None
        if isinstance(head_supports, dict):
            support_fraction = _optional_float_for_gate(head_supports.get(support_threshold_key))
        if support_fraction is None:
            support_fraction = _optional_float_for_gate(positive_by_head.get(head_name))

        topk_grid = topk_by_head.get(head_name)
        top5_mass = _optional_float_for_gate(topk_grid.get(low_budget_key)) if isinstance(topk_grid, dict) else None
        head_failed: list[str] = []
        if support_fraction is None:
            head_failed.append("support_fraction_missing")
        elif support_fraction > max_support_fraction + 1e-12:
            head_failed.append("support_fraction_above_max")
        if top5_mass is None:
            head_failed.append("top5_label_mass_missing")
        elif top5_mass < min_top5_mass_fraction - 1e-12:
            head_failed.append("top5_label_mass_below_min")
        if blocking:
            failed_checks.extend(f"{head_name}:{check}" for check in head_failed)
        head_rows.append(
            {
                "head": str(head_name),
                "blocking": bool(blocking),
                "support_threshold_key": support_threshold_key,
                "support_fraction": support_fraction,
                "top5_label_mass_fraction": top5_mass,
                "failed_checks": head_failed,
            }
        )

    return {
        "schema_version": 1,
        "gate_pass": not failed_checks,
        "failed_checks": failed_checks,
        "final_support_threshold_key": final_support_threshold_key,
        "default_head_support_threshold_key": default_head_support_threshold_key,
        "head_support_threshold_keys": head_support_threshold_keys,
        "blocking_heads": sorted(blocking_heads),
        "max_support_fraction": max_support_fraction,
        "min_top5_label_mass_fraction": min_top5_mass_fraction,
        "final_label_support_fraction": final_support,
        "head_rows": head_rows,
    }


def run_experiment_pipeline(
    config: ExperimentConfig,
    trajectories: list[torch.Tensor],
    results_dir: str,
    save_model: str | None = None,
    save_queries_dir: str | None = None,
    save_simplified_dir: str | None = None,
    trajectory_mmsis: list[int] | None = None,
    validation_trajectories: list[torch.Tensor] | None = None,
    eval_trajectories: list[torch.Tensor] | None = None,
    eval_trajectory_mmsis: list[int] | None = None,
    trajectory_source_ids: list[int] | None = None,
    data_audit: dict[str, Any] | None = None,
) -> ExperimentOutputs:
    """Run training, matched evaluation, and shifted evaluation tables. See experiments/README.md for details."""
    pipeline_t0 = time.perf_counter()
    train_workload_map, eval_workload_map = resolve_workload_maps(config.query.workload)
    if eval_trajectories is None:
        print(
            f"[pipeline] {len(trajectories)} trajectories, workload={workload_name(eval_workload_map)}",
            flush=True,
        )
    else:
        validation_part = (
            f", validation={len(validation_trajectories)} trajectories"
            if validation_trajectories is not None
            else ""
        )
        print(
            f"[pipeline] train={len(trajectories)} trajectories{validation_part}, "
            f"eval={len(eval_trajectories)} trajectories, "
            f"workload={workload_name(eval_workload_map)}",
            flush=True,
        )

    seeds = derive_seed_bundle(config.data.seed)
    selection_metric = str(getattr(config.model, "checkpoint_selection_metric", "score")).lower()
    validation_score_every = int(getattr(config.model, "validation_score_every", 0) or 0)
    needs_validation_score = selection_metric in {"score", "uniform_gap"} or validation_score_every > 0
    with _phase("split"):
        data_split = prepare_experiment_split(
            config=config,
            seeds=seeds,
            trajectories=trajectories,
            needs_validation_score=needs_validation_score,
            trajectory_mmsis=trajectory_mmsis,
            validation_trajectories=validation_trajectories,
            eval_trajectories=eval_trajectories,
            eval_trajectory_mmsis=eval_trajectory_mmsis,
            trajectory_source_ids=trajectory_source_ids,
        )
        train_traj = data_split.train_traj
        test_traj = data_split.test_traj
        selection_traj = data_split.selection_traj
        train_mmsis = data_split.train_mmsis
        test_mmsis = data_split.test_mmsis
        train_source_ids = data_split.train_source_ids

    with _phase("build-datasets"):
        datasets = build_experiment_datasets(data_split)
        train_points = datasets.train_points
        test_points = datasets.test_points
        selection_points = datasets.selection_points
        train_boundaries = datasets.train_boundaries
        test_boundaries = datasets.test_boundaries
        selection_boundaries = datasets.selection_boundaries

    with _phase("generate-workloads"):
        workloads = generate_experiment_workloads(
            config=config,
            seeds=seeds,
            train_traj=train_traj,
            test_traj=test_traj,
            selection_traj=selection_traj,
            train_points=train_points,
            test_points=test_points,
            selection_points=selection_points,
            train_boundaries=train_boundaries,
            test_boundaries=test_boundaries,
            selection_boundaries=selection_boundaries,
            train_workload_map=train_workload_map,
            eval_workload_map=eval_workload_map,
        )
        train_workload = workloads.train_workload
        train_label_workloads = workloads.train_label_workloads
        train_label_workload_seeds = workloads.train_label_workload_seeds
        eval_workload = workloads.eval_workload
        selection_workload = workloads.selection_workload

    range_diagnostics_summary: dict[str, Any] = {}
    range_diagnostics_rows: list[dict[str, Any]] = []
    range_runtime_caches = {
        "train": RangeRuntimeCache(),
        "eval": RangeRuntimeCache(),
        "selection": RangeRuntimeCache(),
    }
    workload_distribution_comparison: dict[str, Any] = {"deltas_vs_eval": {}}

    if save_queries_dir:
        with _phase("write-queries-geojson"):
            write_queries_geojson(save_queries_dir, eval_workload.typed_queries)

    reset_cuda_peak_memory_stats()
    train_labels: tuple[torch.Tensor, torch.Tensor] | None = None
    range_training_target_mode = str(getattr(config.model, "range_training_target_mode", "point_value")).lower()
    range_replicate_target_aggregation = str(
        getattr(config.model, "range_replicate_target_aggregation", "label_mean")
    ).lower()
    if range_replicate_target_aggregation not in {"label_mean", "label_max", "frequency_mean"}:
        raise ValueError(
            "range_replicate_target_aggregation must be 'label_mean', 'label_max', or 'frequency_mean'."
        )
    if len(train_label_workloads) > 1 and not is_workload_blind_model_type(config.model.model_type):
        raise RuntimeError("range_train_workload_replicates > 1 is only valid for workload-blind model types.")
    range_training_target_transform: dict[str, Any] = {
        "mode": range_training_target_mode,
        "enabled": False,
    }
    range_target_balance_diagnostics: dict[str, Any] = {
        "enabled": False,
        "mode": str(getattr(config.model, "range_target_balance_mode", "none")).lower(),
    }
    range_training_label_aggregation: dict[str, Any] = {
        "enabled": False,
        "replicate_count": int(len(train_label_workloads)),
        "seeds": [int(seed) for seed in train_label_workload_seeds],
    }
    teacher_distillation_diagnostics: dict[str, Any] = {
        "enabled": False,
        "mode": str(getattr(config.model, "range_teacher_distillation_mode", "none")),
    }
    if range_training_target_mode == "query_useful_v1_factorized":
        range_training_target_transform.update(
            {
                "enabled": True,
                "target_family": "QueryUsefulV1Factorized",
                "final_success_allowed": True,
            }
        )
    selection_query_cache: EvaluationQueryCache | None = None
    selection_geometry_scores: torch.Tensor | None = None
    mlqds_range_geometry_blend = max(0.0, min(1.0, float(getattr(config.model, "mlqds_range_geometry_blend", 0.0))))
    with _phase("range-training-prep"):
        train_label_sets: list[tuple[torch.Tensor, torch.Tensor]] = []
        train_component_label_sets: list[dict[str, torch.Tensor] | None] = []
        if range_training_target_mode != "query_useful_v1_factorized" or range_teacher_distillation_enabled(config.model):
            for replicate_index, label_workload in enumerate(train_label_workloads):
                label_cache_name = "train" if replicate_index == 0 else f"train_r{replicate_index}"
                runtime_cache = range_runtime_caches["train"] if replicate_index == 0 else RangeRuntimeCache()
                label_result = prepare_range_label_cache(
                    cache_label=label_cache_name,
                    points=train_points,
                    boundaries=train_boundaries,
                    workload=label_workload,
                    workload_map=train_workload_map,
                    config=config,
                    seed=train_label_workload_seeds[replicate_index],
                    runtime_cache=runtime_cache,
                    range_boundary_prior_weight=float(getattr(config.model, "range_boundary_prior_weight", 0.0)),
                )
                if label_result is not None:
                    train_label_sets.append(label_result)
                    train_component_label_sets.append(runtime_cache.component_labels)
        if train_label_sets:
            train_labels = train_label_sets[0]
            if (
                len(train_label_sets) > 1
                and range_training_target_mode == "point_value"
                and not range_teacher_distillation_enabled(config.model)
            ):
                if range_replicate_target_aggregation == "frequency_mean":
                    raise ValueError("range_replicate_target_aggregation='frequency_mean' requires a frequency target.")
                labels, labelled_mask, aggregation_diagnostics = aggregate_range_label_sets(
                    train_label_sets,
                    aggregation="max" if range_replicate_target_aggregation == "label_max" else "mean",
                )
                train_labels = (labels, labelled_mask)
                range_training_label_aggregation.update(aggregation_diagnostics)
                range_training_label_aggregation["enabled"] = True
                range_training_label_aggregation["replicate_target_aggregation"] = (
                    range_replicate_target_aggregation
                )
        if (
            selection_workload is not None
            and selection_points is not None
            and selection_boundaries is not None
            and len(range_only_queries(selection_workload.typed_queries)) == len(selection_workload.typed_queries)
        ):
            selection_query_cache = EvaluationQueryCache.for_workload(
                selection_points,
                selection_boundaries,
                selection_workload.typed_queries,
            )
            range_runtime_caches["selection"].query_cache = selection_query_cache
            if mlqds_range_geometry_blend > 0.0:
                selection_labels = prepare_range_label_cache(
                    cache_label="selection",
                    points=selection_points,
                    boundaries=selection_boundaries,
                    workload=selection_workload,
                    workload_map=eval_workload_map,
                    config=config,
                    seed=seeds.eval_query_seed + 17,
                    runtime_cache=range_runtime_caches["selection"],
                    range_boundary_prior_weight=float(getattr(config.model, "range_boundary_prior_weight", 0.0)),
                )
                if selection_labels is not None:
                    labels, _labelled_mask = selection_labels
                    _, selection_type_id = workload_type_head(single_workload_type(eval_workload_map))
                    selection_geometry_scores = labels[:, selection_type_id].float()
    if (
        train_labels is not None
        and len(range_only_queries(train_workload.typed_queries)) == len(train_workload.typed_queries)
    ):
        print("  prepared train range labels for precomputed training target", flush=True)
    if selection_query_cache is not None:
        print("  prepared checkpoint-validation range query cache", flush=True)
    if range_teacher_distillation_enabled(config.model):
        if not is_workload_blind_model_type(config.model.model_type):
            raise RuntimeError("range teacher distillation is only valid for workload-blind model types.")
        if train_labels is None:
            raise RuntimeError("range teacher distillation requires precomputed range training labels.")
        for label_workload in train_label_workloads:
            if len(range_only_queries(label_workload.typed_queries)) != len(label_workload.typed_queries):
                raise RuntimeError("range teacher distillation requires pure range training workloads.")
        teacher_config = build_range_teacher_config(config.model)
        print(
            f"  range teacher distillation enabled: mode={config.model.range_teacher_distillation_mode} "
            f"teacher_epochs={teacher_config.epochs} "
            f"replicates={len(train_label_workloads)}",
            flush=True,
        )
        distilled_label_sets: list[tuple[torch.Tensor, torch.Tensor]] = []
        per_teacher: list[dict[str, Any]] = []
        for replicate_index, label_workload in enumerate(train_label_workloads):
            with _phase(f"train-range-teacher-r{replicate_index} ({teacher_config.epochs} epochs)"):
                teacher_trained = train_model(
                    train_trajectories=train_traj,
                    train_boundaries=train_boundaries,
                    workload=label_workload,
                    model_config=teacher_config,
                    seed=seeds.torch_seed + 31 + replicate_index,
                    train_workload_map=train_workload_map,
                    precomputed_labels=train_label_sets[replicate_index],
                    train_trajectory_source_ids=train_source_ids,
                    train_trajectory_mmsis=train_mmsis,
                )
            with _phase(f"distill-range-teacher-r{replicate_index}-labels"):
                distilled_labels, replicate_diagnostics = distill_range_teacher_labels(
                    teacher=teacher_trained,
                    teacher_model_type=teacher_config.model_type,
                    points=train_points,
                    boundaries=train_boundaries,
                    workload=label_workload,
                    model_config=config.model,
                )
            replicate_diagnostics["replicate_index"] = int(replicate_index)
            replicate_diagnostics["seed"] = int(train_label_workload_seeds[replicate_index])
            per_teacher.append(replicate_diagnostics)
            distilled_label_sets.append(distilled_labels)
        if len(distilled_label_sets) == 1:
            train_labels = distilled_label_sets[0]
            teacher_distillation_diagnostics = dict(per_teacher[0])
        else:
            teacher_aggregation_mode = "max" if range_replicate_target_aggregation == "label_max" else "mean"
            labels, labelled_mask, aggregation_diagnostics = aggregate_range_label_sets(
                distilled_label_sets,
                source="range_teacher_distillation_replicates",
                aggregation=teacher_aggregation_mode,
            )
            train_labels = (labels, labelled_mask)
            positive = labelled_mask[:, QUERY_TYPE_ID_RANGE] & (labels[:, QUERY_TYPE_ID_RANGE] > 0.0)
            teacher_distillation_diagnostics = {
                "enabled": True,
                "mode": str(getattr(config.model, "range_teacher_distillation_mode", "none")),
                "teacher_model_type": str(teacher_config.model_type),
                "teacher_epochs": int(teacher_config.epochs),
                "replicate_count": int(len(distilled_label_sets)),
                "replicate_target_aggregation": range_replicate_target_aggregation,
                "aggregation": aggregation_diagnostics,
                "per_replicate": per_teacher,
                "labelled_point_count": int(labelled_mask[:, QUERY_TYPE_ID_RANGE].sum().item()),
                "positive_label_count": int(positive.sum().item()),
                "positive_label_fraction": float(positive.sum().item() / max(1, int(labels.shape[0]))),
                "positive_label_mass": (
                    float(labels[positive, QUERY_TYPE_ID_RANGE].sum().item()) if bool(positive.any().item()) else 0.0
                ),
                "budget_loss_ratios": list(getattr(config.model, "budget_loss_ratios", [])),
                "mlqds_temporal_fraction": float(getattr(config.model, "mlqds_temporal_fraction", 0.0)),
                "mlqds_hybrid_mode": str(getattr(config.model, "mlqds_hybrid_mode", "fill")),
            }
            range_training_label_aggregation.update(aggregation_diagnostics)
            range_training_label_aggregation["enabled"] = True
            range_training_label_aggregation["target_mode"] = "teacher_distillation"
            range_training_label_aggregation["replicate_target_aggregation"] = range_replicate_target_aggregation
            print(
                f"  distilled range labels: replicate_count={len(distilled_label_sets)} "
                f"positives={teacher_distillation_diagnostics['positive_label_count']} "
                f"fraction={teacher_distillation_diagnostics['positive_label_fraction']:.4f} "
                f"mass={teacher_distillation_diagnostics['positive_label_mass']:.4f}",
                flush=True,
            )
    elif range_training_target_mode in {
        "query_spine_frequency",
        "query_residual_frequency",
        "set_utility_frequency",
        "local_swap_utility_frequency",
        "local_swap_gain_cost_frequency",
    }:
        if train_labels is None:
            raise RuntimeError(f"{range_training_target_mode} target mode requires precomputed range training labels.")
        if len(train_label_sets) > 1:
            raise RuntimeError(f"{range_training_target_mode} does not yet support multiple train workload replicates.")
        target_phase = range_training_target_mode.replace("_", "-")
        with _phase(f"range-{target_phase}-target"):
            labels, labelled_mask = train_labels
            target_fn = (
                range_local_swap_gain_cost_frequency_training_labels
                if range_training_target_mode == "local_swap_gain_cost_frequency"
                else (
                    range_local_swap_utility_frequency_training_labels
                    if range_training_target_mode == "local_swap_utility_frequency"
                    else (
                        range_set_utility_frequency_training_labels
                        if range_training_target_mode == "set_utility_frequency"
                        else (
                            range_query_residual_frequency_training_labels
                            if range_training_target_mode == "query_residual_frequency"
                            else range_query_spine_frequency_training_labels
                        )
                    )
                )
            )
            labels, labelled_mask, range_training_target_transform = target_fn(
                labels=labels,
                labelled_mask=labelled_mask,
                points=train_points,
                boundaries=train_boundaries,
                typed_queries=train_workload.typed_queries,
                model_config=config.model,
            )
            range_training_target_transform["enabled"] = True
            range_training_target_transform["replicate_count"] = len(train_label_sets)
            train_labels = (labels, labelled_mask)
            print(
                f"  {target_phase} target: "
                f"positives={range_training_target_transform['positive_label_count']} "
                f"fraction={range_training_target_transform['positive_label_fraction']:.4f} "
                f"mass={range_training_target_transform['positive_label_mass']:.4f}",
                flush=True,
            )
    elif range_training_target_mode in {
        "retained_frequency",
        "global_budget_retained_frequency",
        "marginal_coverage_frequency",
        "historical_prior_retained_frequency",
        "structural_retained_frequency",
    }:
        if train_labels is None:
            raise RuntimeError(
                f"{range_training_target_mode} target mode requires precomputed range training labels."
            )
        target_fn = (
            range_marginal_coverage_training_labels
            if range_training_target_mode == "marginal_coverage_frequency"
            else range_global_budget_retained_frequency_training_labels
            if range_training_target_mode == "global_budget_retained_frequency"
            else range_structural_retained_frequency_training_labels
            if range_training_target_mode == "structural_retained_frequency"
            else range_historical_prior_retained_frequency_training_labels
            if range_training_target_mode == "historical_prior_retained_frequency"
            else range_retained_frequency_training_labels
        )
        aggregate_target_fn = (
            aggregate_range_marginal_coverage_training_labels
            if range_training_target_mode == "marginal_coverage_frequency"
            else aggregate_range_global_budget_retained_frequency_training_labels
            if range_training_target_mode == "global_budget_retained_frequency"
            else aggregate_range_structural_retained_frequency_training_labels
            if range_training_target_mode == "structural_retained_frequency"
            else aggregate_range_retained_frequency_training_labels
        )
        target_phase = range_training_target_mode.replace("_", "-")
        with _phase(f"range-{target_phase}-target"):
            if len(train_label_sets) > 1:
                if range_replicate_target_aggregation == "frequency_mean":
                    if range_training_target_mode == "historical_prior_retained_frequency":
                        raise RuntimeError(
                            "historical_prior_retained_frequency does not support "
                            "range_replicate_target_aggregation='frequency_mean'; use label_mean or label_max."
                        )
                    aggregate_target_kwargs = {
                        "label_sets": train_label_sets,
                        "boundaries": train_boundaries,
                        "model_config": config.model,
                    }
                    if range_training_target_mode == "structural_retained_frequency":
                        aggregate_target_kwargs["points"] = train_points
                    labels, labelled_mask, range_training_target_transform = aggregate_target_fn(
                        **aggregate_target_kwargs
                    )
                    range_training_label_aggregation["enabled"] = True
                    range_training_label_aggregation["target_mode"] = range_training_target_mode
                    range_training_label_aggregation["replicate_target_aggregation"] = "frequency_mean"
                else:
                    labels, labelled_mask, aggregation_diagnostics = aggregate_range_label_sets(
                        label_sets=train_label_sets,
                        source=(
                            f"range_label_{'max' if range_replicate_target_aggregation == 'label_max' else 'mean'}"
                            f"_before_{range_training_target_mode}"
                        ),
                        aggregation="max" if range_replicate_target_aggregation == "label_max" else "mean",
                    )
                    range_training_label_aggregation.update(aggregation_diagnostics)
                    range_training_label_aggregation["enabled"] = True
                    range_training_label_aggregation["target_mode"] = range_training_target_mode
                    range_training_label_aggregation["replicate_target_aggregation"] = (
                        range_replicate_target_aggregation
                    )
                    target_kwargs = {
                        "labels": labels,
                        "labelled_mask": labelled_mask,
                        "boundaries": train_boundaries,
                        "model_config": config.model,
                    }
                    if range_training_target_mode in {
                        "historical_prior_retained_frequency",
                        "structural_retained_frequency",
                    }:
                        target_kwargs["points"] = train_points
                    labels, labelled_mask, range_training_target_transform = target_fn(**target_kwargs)
                    range_training_target_transform["label_aggregation"] = aggregation_diagnostics
                range_training_target_transform["replicate_target_aggregation"] = (
                    range_replicate_target_aggregation
                )
            else:
                labels, labelled_mask = train_labels
                target_kwargs = {
                    "labels": labels,
                    "labelled_mask": labelled_mask,
                    "boundaries": train_boundaries,
                    "model_config": config.model,
                }
                if range_training_target_mode in {
                    "historical_prior_retained_frequency",
                    "structural_retained_frequency",
                }:
                    target_kwargs["points"] = train_points
                labels, labelled_mask, range_training_target_transform = target_fn(**target_kwargs)
            range_training_target_transform["enabled"] = True
            range_training_target_transform["replicate_count"] = len(train_label_sets)
            train_labels = (labels, labelled_mask)
            print(
                f"  {target_phase} target: positives={range_training_target_transform['positive_label_count']} "
                f"fraction={range_training_target_transform['positive_label_fraction']:.4f} "
                f"mass={range_training_target_transform['positive_label_mass']:.4f}",
                flush=True,
            )
    elif range_training_target_mode not in {"point_value", "query_useful_v1_factorized"}:
        if range_training_target_mode in {"component_retained_frequency", "continuity_retained_frequency"}:
            if train_labels is None:
                raise RuntimeError(
                    f"{range_training_target_mode} target mode requires precomputed range training labels."
                )
            if not train_component_label_sets or any(component_labels is None for component_labels in train_component_label_sets):
                raise RuntimeError(
                    f"{range_training_target_mode} requires range component labels; use range_label_mode=usefulness."
                )
            target_fn = (
                range_continuity_retained_frequency_training_labels
                if range_training_target_mode == "continuity_retained_frequency"
                else range_component_retained_frequency_training_labels
            )
            aggregate_target_fn = (
                aggregate_range_continuity_retained_frequency_training_labels
                if range_training_target_mode == "continuity_retained_frequency"
                else aggregate_range_component_retained_frequency_training_labels
            )
            target_phase = range_training_target_mode.replace("_", "-")
            with _phase(f"range-{target_phase}-target"):
                if len(train_label_sets) > 1:
                    if range_replicate_target_aggregation == "frequency_mean":
                        labels, labelled_mask, range_training_target_transform = (
                            aggregate_target_fn(
                                label_sets=train_label_sets,
                                component_label_sets=train_component_label_sets,
                                boundaries=train_boundaries,
                                model_config=config.model,
                            )
                        )
                        range_training_label_aggregation["replicate_target_aggregation"] = "frequency_mean"
                    else:
                        aggregation_mode = "max" if range_replicate_target_aggregation == "label_max" else "mean"
                        labels, labelled_mask, component_labels, aggregation_diagnostics = (
                            aggregate_range_component_label_sets(
                                label_sets=train_label_sets,
                                component_label_sets=train_component_label_sets,
                                aggregation=aggregation_mode,
                            )
                        )
                        range_training_label_aggregation.update(aggregation_diagnostics)
                        range_training_label_aggregation["replicate_target_aggregation"] = (
                            range_replicate_target_aggregation
                        )
                        labels, labelled_mask, range_training_target_transform = (
                            target_fn(
                                labels=labels,
                                labelled_mask=labelled_mask,
                                component_labels=component_labels,
                                boundaries=train_boundaries,
                                model_config=config.model,
                            )
                        )
                        range_training_target_transform["label_aggregation"] = aggregation_diagnostics
                    range_training_label_aggregation["enabled"] = True
                    range_training_label_aggregation["target_mode"] = range_training_target_mode
                else:
                    labels, labelled_mask = train_labels
                    component_labels = train_component_label_sets[0]
                    if component_labels is None:
                        raise RuntimeError("component_retained_frequency requires component labels.")
                    labels, labelled_mask, range_training_target_transform = (
                        target_fn(
                            labels=labels,
                            labelled_mask=labelled_mask,
                            component_labels=component_labels,
                            boundaries=train_boundaries,
                            model_config=config.model,
                        )
                    )
                range_training_target_transform["enabled"] = True
                range_training_target_transform["replicate_count"] = len(train_label_sets)
                range_training_target_transform["replicate_target_aggregation"] = range_replicate_target_aggregation
                train_labels = (labels, labelled_mask)
                print(
                    f"  {target_phase} target: "
                    f"positives={range_training_target_transform['positive_label_count']} "
                    f"fraction={range_training_target_transform['positive_label_fraction']:.4f} "
                    f"mass={range_training_target_transform['positive_label_mass']:.4f}",
                    flush=True,
                )
        else:
            raise RuntimeError(
                "range_training_target_mode must be 'point_value', 'retained_frequency', "
                "'global_budget_retained_frequency', 'historical_prior_retained_frequency', "
                "'marginal_coverage_frequency', 'query_spine_frequency', "
                "'query_residual_frequency', 'set_utility_frequency', 'local_swap_utility_frequency', "
                "'local_swap_gain_cost_frequency', 'structural_retained_frequency', "
                "'component_retained_frequency', or "
                "'continuity_retained_frequency', or 'query_useful_v1_factorized'."
            )
    range_target_balance_mode = str(getattr(config.model, "range_target_balance_mode", "none")).lower()
    if range_target_balance_mode != "none":
        if train_labels is None:
            raise RuntimeError("range_target_balance_mode requires precomputed range training labels.")
        with _phase("range-target-balance"):
            labels, labelled_mask = train_labels
            labels, labelled_mask, range_target_balance_diagnostics = balance_range_training_target_by_trajectory(
                labels=labels,
                labelled_mask=labelled_mask,
                boundaries=train_boundaries,
                mode=range_target_balance_mode,
            )
            train_labels = (labels, labelled_mask)
            print(
                f"  target balance={range_target_balance_diagnostics['mode']} "
                f"positives={range_target_balance_diagnostics['positive_label_count']} "
                f"mass={range_target_balance_diagnostics['positive_label_mass']:.4f} "
                f"trajectories={range_target_balance_diagnostics['balanced_trajectory_count']}",
                flush=True,
            )
    if range_training_target_mode != "query_useful_v1_factorized":
        range_training_target_transform.setdefault("target_family", "legacy_range_useful_scalar")
        range_training_target_transform.setdefault("final_success_allowed", False)
        range_training_target_transform.setdefault(
            "legacy_reason",
            "Old RangeUseful/scalar-target diagnostic path. "
            "Not valid for query-driven rework acceptance.",
        )
    with _phase(f"train-model ({config.model.epochs} epochs)"):
        trained = train_model(
            train_trajectories=train_traj,
            train_boundaries=train_boundaries,
            workload=train_workload,
            model_config=config.model,
            seed=seeds.torch_seed,
            train_workload_map=train_workload_map,
            validation_trajectories=selection_traj,
            validation_boundaries=selection_boundaries,
            validation_workload=selection_workload,
            validation_workload_map=eval_workload_map if selection_workload is not None else None,
            precomputed_labels=train_labels,
            validation_points=selection_points,
            precomputed_validation_query_cache=selection_query_cache,
            precomputed_validation_geometry_scores=selection_geometry_scores,
            train_trajectory_source_ids=train_source_ids,
            train_trajectory_mmsis=train_mmsis,
            query_prior_workloads=train_label_workloads,
            query_prior_workload_seeds=train_label_workload_seeds,
        )
    training_cuda_memory = cuda_memory_snapshot()
    if training_cuda_memory.get("available"):
        print(
            f"  train_cuda_peak_allocated={training_cuda_memory['max_allocated_mb']:.1f} MiB  "
            f"peak_reserved={training_cuda_memory['max_reserved_mb']:.1f} MiB",
            flush=True,
        )

    if save_model:
        with _phase("save-model"):
            artifacts = ModelArtifacts(
                model=trained.model,
                scaler=trained.scaler,
                config=config,
                epochs_trained=trained.epochs_trained,
                workload_type=single_workload_type(eval_workload_map),
                query_prior_field=trained.feature_context.get("query_prior_field"),
            )
            save_checkpoint(save_model, artifacts)
            print(
                f"  saved checkpoint to {save_model}  "
                f"(epochs_trained={trained.epochs_trained}, "
                f"best_epoch={trained.best_epoch}, best_loss={trained.best_loss:.8f}, "
                f"workload={workload_name(eval_workload_map)})",
                flush=True,
            )
    methods = build_primary_methods(
        trained=trained,
        eval_workload=eval_workload,
        eval_workload_map=eval_workload_map,
        config=config,
        trajectory_mmsis=test_mmsis,
    )
    retention_methods = list(methods)
    workload_blind_eval = is_workload_blind_model_type(config.model.model_type)
    audit_ratios = _range_audit_ratios(config)
    selector_budget_ratios = tuple(
        sorted({float(config.model.compression_ratio), *(float(ratio) for ratio in audit_ratios)})
    )
    if str(getattr(config.model, "selector_type", "temporal_hybrid")).lower() == "learned_segment_budget_v1":
        selector_budget_diagnostics = {
            "train": learned_segment_budget_diagnostics(train_boundaries, selector_budget_ratios),
            "eval": learned_segment_budget_diagnostics(test_boundaries, selector_budget_ratios),
        }
    else:
        selector_budget_diagnostics = {
            "train": temporal_hybrid_selector_budget_diagnostics(
                train_boundaries,
                selector_budget_ratios,
                temporal_fraction=float(config.model.mlqds_temporal_fraction),
                hybrid_mode=str(config.model.mlqds_hybrid_mode),
                min_learned_swaps=int(config.model.mlqds_min_learned_swaps),
            ),
            "eval": temporal_hybrid_selector_budget_diagnostics(
                test_boundaries,
                selector_budget_ratios,
                temporal_fraction=float(config.model.mlqds_temporal_fraction),
                hybrid_mode=str(config.model.mlqds_hybrid_mode),
                min_learned_swaps=int(config.model.mlqds_min_learned_swaps),
            ),
        }
    frozen_primary_masks: dict[str, torch.Tensor] = {}
    frozen_audit_methods_by_ratio: dict[str, list[Method]] = {}
    frozen_primary_scores: dict[str, torch.Tensor] = {}
    frozen_primary_raw_preds: dict[str, torch.Tensor] = {}
    frozen_primary_head_logits: dict[str, torch.Tensor] = {}
    frozen_primary_segment_scores: dict[str, torch.Tensor] = {}
    primary_selector_trace: dict[str, Any] | None = None
    causality_ablation_methods: list[FrozenMaskMethod] = []
    causal_ablation_freeze_failures: dict[str, str] = {}
    prior_sensitivity_diagnostics: dict[str, Any] = {}
    segment_budget_head_ablation_mode: str | None = None
    if workload_blind_eval:
        with _phase("freeze-retained-masks"):
            for method in methods:
                with _phase(f"  freeze {method.name}"):
                    freeze_t0 = time.perf_counter()
                    frozen_primary_masks[method.name] = method.simplify(
                        test_points,
                        test_boundaries,
                        config.model.compression_ratio,
                    ).detach().cpu()
                    setattr(method, "latency_ms", float((time.perf_counter() - freeze_t0) * 1000.0))
                    score_cache = getattr(method, "_score_cache", None)
                    if isinstance(score_cache, torch.Tensor):
                        frozen_primary_scores[method.name] = score_cache.detach().cpu().float()
                    raw_pred_cache = getattr(method, "_raw_pred_cache", None)
                    if isinstance(raw_pred_cache, torch.Tensor):
                        frozen_primary_raw_preds[method.name] = raw_pred_cache.detach().cpu().float()
                    head_logit_cache = getattr(method, "_head_logit_cache", None)
                    if isinstance(head_logit_cache, torch.Tensor):
                        frozen_primary_head_logits[method.name] = head_logit_cache.detach().cpu().float()
                    segment_score_cache = getattr(method, "_segment_score_cache", None)
                    if isinstance(segment_score_cache, torch.Tensor):
                        frozen_primary_segment_scores[method.name] = segment_score_cache.detach().cpu().float()
            primary_scores = frozen_primary_scores.get("MLQDS")
            primary_raw_preds = frozen_primary_raw_preds.get("MLQDS")
            if primary_scores is not None and str(getattr(config.model, "selector_type", "")).lower() == "learned_segment_budget_v1":
                primary_segment_scores = frozen_primary_segment_scores.get("MLQDS")
                trace_mask, trace = simplify_with_learned_segment_budget_v1_with_trace(
                    primary_scores,
                    test_boundaries,
                    float(config.model.compression_ratio),
                    segment_scores=primary_segment_scores,
                    points=test_points.detach().cpu().float(),
                    geometry_gain_weight=float(config.model.learned_segment_geometry_gain_weight),
                    segment_score_point_blend_weight=float(config.model.learned_segment_score_blend_weight),
                    fairness_preallocation_enabled=bool(config.model.learned_segment_fairness_preallocation),
                )
                frozen_mlqds_mask = frozen_primary_masks.get("MLQDS")
                if isinstance(frozen_mlqds_mask, torch.Tensor):
                    trace["retained_mask_matches_frozen_primary"] = bool(
                        torch.equal(trace_mask.detach().cpu(), frozen_mlqds_mask.detach().cpu())
                    )
                    trace["frozen_primary_retained_count"] = int(frozen_mlqds_mask.sum().item())
                primary_selector_trace = trace
                generator = torch.Generator().manual_seed(int(seeds.eval_query_seed) + 91_337)
                shuffled_order = torch.randperm(int(primary_scores.numel()), generator=generator)
                shuffled_scores = primary_scores[shuffled_order]
                shuffled_segment_scores = (
                    primary_segment_scores[shuffled_order] if primary_segment_scores is not None else None
                )
                causality_ablation_methods.append(
                    _learned_segment_frozen_method(
                        name="MLQDS_shuffled_scores",
                        scores=shuffled_scores,
                        boundaries=test_boundaries,
                        compression_ratio=float(config.model.compression_ratio),
                        segment_scores=shuffled_segment_scores,
                        points=test_points,
                        learned_segment_geometry_gain_weight=float(config.model.learned_segment_geometry_gain_weight),
                        learned_segment_score_blend_weight=float(config.model.learned_segment_score_blend_weight),
                        learned_segment_fairness_preallocation=bool(config.model.learned_segment_fairness_preallocation),
                    )
                )
                if primary_segment_scores is not None:
                    neutral_segment_scores = _neutral_segment_scores_for_ablation(primary_segment_scores)
                    segment_budget_head_ablation_mode = "neutral_constant_segment_scores"
                    causality_ablation_methods.append(
                        _learned_segment_frozen_method(
                            name="MLQDS_without_segment_budget_head",
                            scores=primary_scores,
                            boundaries=test_boundaries,
                            compression_ratio=float(config.model.compression_ratio),
                            segment_scores=neutral_segment_scores,
                            points=test_points,
                            learned_segment_geometry_gain_weight=float(config.model.learned_segment_geometry_gain_weight),
                            learned_segment_score_blend_weight=float(config.model.learned_segment_score_blend_weight),
                            learned_segment_fairness_preallocation=bool(config.model.learned_segment_fairness_preallocation),
                        )
                    )
                    if bool(config.model.learned_segment_fairness_preallocation):
                        causality_ablation_methods.append(
                            _learned_segment_frozen_method(
                                name="MLQDS_without_trajectory_fairness_preallocation",
                                scores=primary_scores,
                                boundaries=test_boundaries,
                                compression_ratio=float(config.model.compression_ratio),
                                segment_scores=primary_segment_scores,
                                points=test_points,
                                learned_segment_geometry_gain_weight=float(config.model.learned_segment_geometry_gain_weight),
                                learned_segment_score_blend_weight=float(config.model.learned_segment_score_blend_weight),
                                learned_segment_fairness_preallocation=False,
                            )
                        )
                primary_head_logits = frozen_primary_head_logits.get("MLQDS")
                if primary_head_logits is not None:
                    try:
                        behavior_scores = _scores_without_factorized_head(
                            model=trained.model,
                            head_logits=primary_head_logits,
                            disabled_head_name="conditional_behavior_utility",
                            boundaries=test_boundaries,
                            workload_type=single_workload_type(eval_workload_map),
                            score_mode=config.model.mlqds_score_mode,
                            score_temperature=float(config.model.mlqds_score_temperature),
                            rank_confidence_weight=float(config.model.mlqds_rank_confidence_weight),
                        )
                        causality_ablation_methods.append(
                            _learned_segment_frozen_method(
                                name="MLQDS_without_behavior_utility_head",
                                scores=behavior_scores,
                                boundaries=test_boundaries,
                                compression_ratio=float(config.model.compression_ratio),
                                segment_scores=primary_segment_scores,
                                points=test_points,
                                learned_segment_geometry_gain_weight=float(config.model.learned_segment_geometry_gain_weight),
                                learned_segment_score_blend_weight=float(config.model.learned_segment_score_blend_weight),
                                learned_segment_fairness_preallocation=bool(config.model.learned_segment_fairness_preallocation),
                            )
                        )
                    except Exception as exc:  # pragma: no cover - diagnostic should not break final eval.
                        causal_ablation_freeze_failures["MLQDS_without_behavior_utility_head"] = str(exc)
                try:
                    untrained_model = _reset_module_parameters(
                        trained.model,
                        seed=int(seeds.torch_seed) + 44_021,
                    )
                    untrained_outputs = TrainingOutputs(
                        model=untrained_model,
                        scaler=trained.scaler,
                        labels=trained.labels,
                        labelled_mask=trained.labelled_mask,
                        history=[],
                        feature_context=dict(trained.feature_context),
                    )
                    untrained_method = MLQDSMethod(
                        name="MLQDS_untrained_model",
                        trained=untrained_outputs,
                        workload=eval_workload,
                        workload_type=single_workload_type(eval_workload_map),
                        score_mode=config.model.mlqds_score_mode,
                        score_temperature=config.model.mlqds_score_temperature,
                        rank_confidence_weight=config.model.mlqds_rank_confidence_weight,
                        temporal_fraction=config.model.mlqds_temporal_fraction,
                        diversity_bonus=config.model.mlqds_diversity_bonus,
                        hybrid_mode=config.model.mlqds_hybrid_mode,
                        stratified_center_weight=config.model.mlqds_stratified_center_weight,
                        min_learned_swaps=config.model.mlqds_min_learned_swaps,
                        selector_type=config.model.selector_type,
                        trajectory_mmsis=test_mmsis,
                        inference_device=None,
                        amp_mode=config.model.amp_mode,
                        inference_batch_size=config.model.inference_batch_size,
                        learned_segment_geometry_gain_weight=config.model.learned_segment_geometry_gain_weight,
                        learned_segment_score_blend_weight=config.model.learned_segment_score_blend_weight,
                        learned_segment_fairness_preallocation=config.model.learned_segment_fairness_preallocation,
                    )
                    untrained_mask = untrained_method.simplify(
                        test_points,
                        test_boundaries,
                        float(config.model.compression_ratio),
                    )
                    causality_ablation_methods.append(
                        FrozenMaskMethod(
                            name="MLQDS_untrained_model",
                            retained_mask=untrained_mask.detach().cpu(),
                        )
                    )
                except Exception as exc:  # pragma: no cover - diagnostic should not break final eval.
                    causal_ablation_freeze_failures["MLQDS_untrained_model"] = str(exc)
                query_prior_field = trained.feature_context.get("query_prior_field")
                if isinstance(query_prior_field, dict):
                    prior_scores = query_prior_predictability_scores(test_points, query_prior_field).detach().cpu()
                    causality_ablation_methods.append(
                        _learned_segment_frozen_method(
                            name="MLQDS_prior_field_only_score",
                            scores=prior_scores,
                            boundaries=test_boundaries,
                            compression_ratio=float(config.model.compression_ratio),
                            points=test_points,
                            learned_segment_geometry_gain_weight=float(config.model.learned_segment_geometry_gain_weight),
                            learned_segment_score_blend_weight=float(config.model.learned_segment_score_blend_weight),
                            learned_segment_fairness_preallocation=bool(config.model.learned_segment_fairness_preallocation),
                        )
                    )
                    try:
                        shuffled_prior_field = _shuffled_query_prior_field(
                            query_prior_field,
                            seed=int(seeds.eval_query_seed) + 71_003,
                        )
                        shuffled_prior_feature_sensitivity = _prior_feature_sample_sensitivity(
                            points=test_points,
                            primary_prior_field=query_prior_field,
                            ablation_prior_field=shuffled_prior_field,
                        )
                        shuffled_prior_trained = TrainingOutputs(
                            model=trained.model,
                            scaler=trained.scaler,
                            labels=trained.labels,
                            labelled_mask=trained.labelled_mask,
                            history=trained.history,
                            epochs_trained=trained.epochs_trained,
                            best_epoch=trained.best_epoch,
                            best_loss=trained.best_loss,
                            best_selection_score=trained.best_selection_score,
                            target_diagnostics=trained.target_diagnostics,
                            fit_diagnostics=trained.fit_diagnostics,
                            feature_context={
                                **trained.feature_context,
                                "query_prior_field": shuffled_prior_field,
                            },
                        )
                        shuffled_prior_method = MLQDSMethod(
                            name="MLQDS_shuffled_prior_fields",
                            trained=shuffled_prior_trained,
                            workload=eval_workload,
                            workload_type=single_workload_type(eval_workload_map),
                            score_mode=config.model.mlqds_score_mode,
                            score_temperature=config.model.mlqds_score_temperature,
                            rank_confidence_weight=config.model.mlqds_rank_confidence_weight,
                            temporal_fraction=config.model.mlqds_temporal_fraction,
                            diversity_bonus=config.model.mlqds_diversity_bonus,
                            hybrid_mode=config.model.mlqds_hybrid_mode,
                            stratified_center_weight=config.model.mlqds_stratified_center_weight,
                            min_learned_swaps=config.model.mlqds_min_learned_swaps,
                            selector_type=config.model.selector_type,
                            trajectory_mmsis=test_mmsis,
                            inference_device=None,
                            amp_mode=config.model.amp_mode,
                            inference_batch_size=config.model.inference_batch_size,
                            learned_segment_geometry_gain_weight=config.model.learned_segment_geometry_gain_weight,
                            learned_segment_score_blend_weight=config.model.learned_segment_score_blend_weight,
                            learned_segment_fairness_preallocation=config.model.learned_segment_fairness_preallocation,
                        )
                        shuffled_prior_mask = shuffled_prior_method.simplify(
                            test_points,
                            test_boundaries,
                            float(config.model.compression_ratio),
                        )
                        shuffled_prior_scores = getattr(shuffled_prior_method, "_score_cache", None)
                        shuffled_prior_raw_preds = getattr(shuffled_prior_method, "_raw_pred_cache", None)
                        score_sensitivity = _score_ablation_sensitivity(
                            primary_scores=primary_scores,
                            ablation_scores=shuffled_prior_scores if isinstance(shuffled_prior_scores, torch.Tensor) else None,
                            primary_mask=frozen_primary_masks.get("MLQDS"),
                            ablation_mask=shuffled_prior_mask,
                        )
                        raw_sensitivity = _score_ablation_sensitivity(
                            primary_scores=primary_raw_preds,
                            ablation_scores=(
                                shuffled_prior_raw_preds if isinstance(shuffled_prior_raw_preds, torch.Tensor) else None
                            ),
                            primary_mask=frozen_primary_masks.get("MLQDS"),
                            ablation_mask=shuffled_prior_mask,
                        )
                        prior_sensitivity_diagnostics["shuffled_prior_fields"] = {
                            "sampled_prior_features": shuffled_prior_feature_sensitivity,
                            "selector_score": score_sensitivity,
                            "raw_prediction": raw_sensitivity,
                        }
                        causality_ablation_methods.append(
                            FrozenMaskMethod(
                                name="MLQDS_shuffled_prior_fields",
                                retained_mask=shuffled_prior_mask.detach().cpu(),
                            )
                        )
                    except Exception as exc:  # pragma: no cover - diagnostic should not break final eval.
                        causal_ablation_freeze_failures["MLQDS_shuffled_prior_fields"] = str(exc)
                    try:
                        zero_prior_field = zero_query_prior_field_like(query_prior_field)
                        zero_prior_feature_sensitivity = _prior_feature_sample_sensitivity(
                            points=test_points,
                            primary_prior_field=query_prior_field,
                            ablation_prior_field=zero_prior_field,
                        )
                        zero_prior_trained = TrainingOutputs(
                            model=trained.model,
                            scaler=trained.scaler,
                            labels=trained.labels,
                            labelled_mask=trained.labelled_mask,
                            history=trained.history,
                            epochs_trained=trained.epochs_trained,
                            best_epoch=trained.best_epoch,
                            best_loss=trained.best_loss,
                            best_selection_score=trained.best_selection_score,
                            target_diagnostics=trained.target_diagnostics,
                            fit_diagnostics=trained.fit_diagnostics,
                            feature_context={
                                **trained.feature_context,
                                "query_prior_field": zero_prior_field,
                                "query_prior_field_metadata": query_prior_field_metadata(zero_prior_field),
                            },
                        )
                        zero_prior_method = MLQDSMethod(
                            name="MLQDS_without_query_prior_features",
                            trained=zero_prior_trained,
                            workload=eval_workload,
                            workload_type=single_workload_type(eval_workload_map),
                            score_mode=config.model.mlqds_score_mode,
                            score_temperature=config.model.mlqds_score_temperature,
                            rank_confidence_weight=config.model.mlqds_rank_confidence_weight,
                            temporal_fraction=config.model.mlqds_temporal_fraction,
                            diversity_bonus=config.model.mlqds_diversity_bonus,
                            hybrid_mode=config.model.mlqds_hybrid_mode,
                            stratified_center_weight=config.model.mlqds_stratified_center_weight,
                            min_learned_swaps=config.model.mlqds_min_learned_swaps,
                            selector_type=config.model.selector_type,
                            trajectory_mmsis=test_mmsis,
                            inference_device=None,
                            amp_mode=config.model.amp_mode,
                            inference_batch_size=config.model.inference_batch_size,
                            learned_segment_geometry_gain_weight=config.model.learned_segment_geometry_gain_weight,
                            learned_segment_score_blend_weight=config.model.learned_segment_score_blend_weight,
                            learned_segment_fairness_preallocation=config.model.learned_segment_fairness_preallocation,
                        )
                        zero_prior_mask = zero_prior_method.simplify(
                            test_points,
                            test_boundaries,
                            float(config.model.compression_ratio),
                        )
                        zero_prior_scores = getattr(zero_prior_method, "_score_cache", None)
                        zero_prior_raw_preds = getattr(zero_prior_method, "_raw_pred_cache", None)
                        score_sensitivity = _score_ablation_sensitivity(
                            primary_scores=primary_scores,
                            ablation_scores=zero_prior_scores if isinstance(zero_prior_scores, torch.Tensor) else None,
                            primary_mask=frozen_primary_masks.get("MLQDS"),
                            ablation_mask=zero_prior_mask,
                        )
                        raw_sensitivity = _score_ablation_sensitivity(
                            primary_scores=primary_raw_preds,
                            ablation_scores=zero_prior_raw_preds if isinstance(zero_prior_raw_preds, torch.Tensor) else None,
                            primary_mask=frozen_primary_masks.get("MLQDS"),
                            ablation_mask=zero_prior_mask,
                        )
                        prior_sensitivity_diagnostics["without_query_prior_features"] = {
                            "sampled_prior_features": zero_prior_feature_sensitivity,
                            "selector_score": score_sensitivity,
                            "raw_prediction": raw_sensitivity,
                        }
                        causality_ablation_methods.append(
                            FrozenMaskMethod(
                                name="MLQDS_without_query_prior_features",
                                retained_mask=zero_prior_mask.detach().cpu(),
                            )
                        )
                    except Exception as exc:  # pragma: no cover - diagnostic should not break final eval.
                        causal_ablation_freeze_failures["MLQDS_without_query_prior_features"] = str(exc)
        methods = [
            FrozenMaskMethod(
                name=method.name,
                retained_mask=frozen_primary_masks[method.name],
                latency_ms=float(getattr(method, "latency_ms", 0.0)),
            )
            for method in methods
        ]
        print(
            "  workload_blind_protocol=enabled: primary retained masks frozen before eval query scoring",
            flush=True,
        )
        if audit_ratios:
            with _phase("freeze-audit-retained-masks"):
                for ratio in audit_ratios:
                    if abs(float(ratio) - float(config.model.compression_ratio)) <= 1e-9:
                        continue
                    ratio_key = f"{float(ratio):.4f}"
                    frozen_ratio_methods: list[Method] = []
                    for method in retention_methods:
                        with _phase(f"  freeze audit {method.name} ratio={ratio:.4f}"):
                            freeze_t0 = time.perf_counter()
                            retained_mask = method.simplify(
                                test_points,
                                test_boundaries,
                                float(ratio),
                            ).detach().cpu()
                            frozen_ratio_methods.append(
                                FrozenMaskMethod(
                                    name=method.name,
                                    retained_mask=retained_mask,
                                    latency_ms=float((time.perf_counter() - freeze_t0) * 1000.0),
                                )
                            )
                    frozen_audit_methods_by_ratio[ratio_key] = frozen_ratio_methods
            print(
                "  workload_blind_protocol=enabled: audit retained masks frozen before eval query scoring",
                flush=True,
            )

    matched: dict[str, MethodEvaluation] = {}
    oracle_method: OracleMethod | None = None
    eval_labels: torch.Tensor | None = None
    save_masks = bool(save_simplified_dir)
    eval_is_range_only = len(range_only_queries(eval_workload.typed_queries)) == len(eval_workload.typed_queries)
    final_metrics_mode = str(getattr(config.baselines, "final_metrics_mode", "diagnostic")).lower()
    if final_metrics_mode not in {"diagnostic", "core"}:
        raise ValueError("final_metrics_mode must be either 'diagnostic' or 'core'.")
    run_final_diagnostics = final_metrics_mode == "diagnostic"
    run_oracle_baseline = bool(config.baselines.include_oracle and run_final_diagnostics)
    run_learned_fill_diagnostics = bool(eval_is_range_only and run_final_diagnostics)
    with _phase("eval-query-cache-prep"):
        eval_query_cache = prepare_eval_query_cache(
            test_points=test_points,
            test_boundaries=test_boundaries,
            eval_workload=eval_workload,
            eval_is_range_only=eval_is_range_only,
            runtime_cache=range_runtime_caches["eval"],
        )
    if run_oracle_baseline or run_learned_fill_diagnostics or mlqds_range_geometry_blend > 0.0:
        with _phase("eval-label-prep"):
            eval_labels = prepare_eval_labels(
                test_points=test_points,
                test_boundaries=test_boundaries,
                eval_workload=eval_workload,
                eval_workload_map=eval_workload_map,
                config=config,
                seeds=seeds,
                eval_is_range_only=eval_is_range_only,
                run_oracle_baseline=run_oracle_baseline,
                runtime_cache=range_runtime_caches["eval"],
            )
    if mlqds_range_geometry_blend > 0.0:
        if eval_labels is None:
            raise RuntimeError("MLQDS range geometry blend requested but eval labels were not prepared.")
        attach_range_geometry_scores(
            methods=methods,
            eval_labels=eval_labels,
            eval_workload_map=eval_workload_map,
        )
    with _phase("evaluate-matched"):
        for method in methods:
            with _phase(f"  eval {method.name}"):
                matched[method.name] = evaluate_method(
                    method=method,
                    points=test_points,
                    boundaries=test_boundaries,
                    typed_queries=eval_workload.typed_queries,
                    workload_map=eval_workload_map,
                    compression_ratio=config.model.compression_ratio,
                    return_mask=method.name == "MLQDS" or save_masks,
                    query_cache=eval_query_cache,
                )

        if run_oracle_baseline:
            if eval_labels is None:
                raise RuntimeError("Oracle baseline requested but eval labels were not prepared.")
            oracle_method = OracleMethod(labels=eval_labels, workload_type=single_workload_type(eval_workload_map))
            with _phase(f"  eval {oracle_method.name}"):
                matched[oracle_method.name] = evaluate_method(
                    method=oracle_method,
                    points=test_points,
                    boundaries=test_boundaries,
                    typed_queries=eval_workload.typed_queries,
                    workload_map=eval_workload_map,
                    compression_ratio=config.model.compression_ratio,
                    query_cache=eval_query_cache,
                )

    causality_ablation_evaluations: dict[str, MethodEvaluation] = {}
    if causality_ablation_methods:
        with _phase("learning-causality-ablations"):
            for method in causality_ablation_methods:
                with _phase(f"  ablation {method.name}"):
                    causality_ablation_evaluations[method.name] = evaluate_method(
                        method=method,
                        points=test_points,
                        boundaries=test_boundaries,
                        typed_queries=eval_workload.typed_queries,
                        workload_map=eval_workload_map,
                        compression_ratio=config.model.compression_ratio,
                        query_cache=eval_query_cache,
                    )

    learned_fill_diagnostics: dict[str, MethodEvaluation] = {"MLQDS": matched["MLQDS"]}
    learned_fill_table = ""
    diagnostic_methods: list[Method] = []
    if run_learned_fill_diagnostics:
        if eval_labels is None:
            raise RuntimeError("Learned-fill diagnostics requested but eval labels were not prepared.")
        diagnostic_methods = build_learned_fill_methods(
            test_points=test_points,
            eval_labels=eval_labels,
            eval_workload_map=eval_workload_map,
            config=config,
            seeds=seeds,
        )
        with _phase("learned-fill-diagnostics"):
            for method in diagnostic_methods:
                with _phase(f"  fill {method.name}"):
                    learned_fill_diagnostics[method.name] = evaluate_method(
                        method=method,
                        points=test_points,
                        boundaries=test_boundaries,
                        typed_queries=eval_workload.typed_queries,
                        workload_map=eval_workload_map,
                        compression_ratio=config.model.compression_ratio,
                        query_cache=eval_query_cache,
                    )
        learned_fill_table = print_range_usefulness_table(learned_fill_diagnostics)

    matched_table = print_method_comparison_table(matched)
    geometric_table = print_geometric_distortion_table(matched)
    range_usefulness_table = print_range_usefulness_table(matched)
    range_compression_audit: dict[str, dict[str, Any]] = {}
    range_compression_audit_table = ""
    if audit_ratios:
        audit_methods = [*(retention_methods if workload_blind_eval else methods), *diagnostic_methods]
        if oracle_method is not None:
            audit_methods.append(oracle_method)
        audit_sections: list[str] = []
        with _phase("range-compression-audit"):
            for ratio in audit_ratios:
                if abs(float(ratio) - float(config.model.compression_ratio)) <= 1e-9:
                    ratio_results = {
                        **matched,
                        **{
                            name: metrics
                            for name, metrics in learned_fill_diagnostics.items()
                            if name not in matched
                        },
                    }
                else:
                    ratio_results: dict[str, MethodEvaluation] = {}
                    ratio_key = f"{float(ratio):.4f}"
                    ratio_audit_methods = audit_methods
                    if workload_blind_eval and ratio_key in frozen_audit_methods_by_ratio:
                        ratio_audit_methods = [*frozen_audit_methods_by_ratio[ratio_key], *diagnostic_methods]
                        if oracle_method is not None:
                            ratio_audit_methods.append(oracle_method)
                    for method in ratio_audit_methods:
                        with _phase(f"  audit {method.name} ratio={ratio:.4f}"):
                            ratio_results[method.name] = evaluate_method(
                                method=method,
                                points=test_points,
                                boundaries=test_boundaries,
                                typed_queries=eval_workload.typed_queries,
                                workload_map=eval_workload_map,
                                compression_ratio=float(ratio),
                                query_cache=eval_query_cache,
                            )
                ratio_key = f"{float(ratio):.4f}"
                range_compression_audit[ratio_key] = {
                    name: _evaluation_metrics_payload(metrics) for name, metrics in ratio_results.items()
                }
                audit_sections.append(f"compression_ratio={ratio_key}\n{print_range_usefulness_table(ratio_results)}")
        range_compression_audit_table = "\n\n".join(audit_sections)

    with _phase("evaluate-shift"):
        shift_pairs = evaluate_shift_pairs(
            matched_mlqds_score=float(matched["MLQDS"].aggregate_f1),
            trained=trained,
            train_workload=train_workload,
            train_workload_map=train_workload_map,
            eval_workload_map=eval_workload_map,
            config=config,
        test_points=test_points,
        test_boundaries=test_boundaries,
        test_mmsis=test_mmsis,
    )
    shift_table = print_shift_table(shift_pairs)

    with _phase("range-diagnostics"):
        train_summary, train_rows = _range_workload_diagnostics(
            "train",
            train_points,
            train_boundaries,
            train_workload,
            train_workload_map,
            config,
            seeds.train_query_seed,
            range_runtime_caches["train"],
        )
        eval_summary, eval_rows = _range_workload_diagnostics(
            "eval",
            test_points,
            test_boundaries,
            eval_workload,
            eval_workload_map,
            config,
            seeds.eval_query_seed,
            range_runtime_caches["eval"],
        )
        range_diagnostics_summary["train"] = train_summary
        range_diagnostics_summary["eval"] = eval_summary
        range_diagnostics_rows.extend(train_rows)
        range_diagnostics_rows.extend(eval_rows)
        for replicate_index, replicate_workload in enumerate(train_label_workloads[1:], start=1):
            replicate_label = f"train_r{replicate_index}"
            replicate_summary, replicate_rows = _range_workload_diagnostics(
                replicate_label,
                train_points,
                train_boundaries,
                replicate_workload,
                train_workload_map,
                config,
                train_label_workload_seeds[replicate_index],
                RangeRuntimeCache(),
            )
            range_diagnostics_summary[replicate_label] = replicate_summary
            range_diagnostics_rows.extend(replicate_rows)
        if selection_workload is not None and selection_points is not None and selection_boundaries is not None:
            selection_summary, selection_rows = _range_workload_diagnostics(
                "selection",
                selection_points,
                selection_boundaries,
                selection_workload,
                eval_workload_map,
                config,
                seeds.eval_query_seed + 17,
                range_runtime_caches["selection"],
            )
            range_diagnostics_summary["selection"] = selection_summary
            range_diagnostics_rows.extend(selection_rows)
        _print_range_diagnostics_summary(range_diagnostics_summary)
        workload_distribution_comparison = _range_workload_distribution_comparison(range_diagnostics_summary)
        _print_range_distribution_comparison(workload_distribution_comparison)

    range_learned_fill_summary = _range_learned_fill_summary(
        learned_fill_diagnostics=learned_fill_diagnostics,
        training_target_diagnostics=trained.target_diagnostics,
        range_diagnostics_summary=range_diagnostics_summary,
        compression_ratio=float(config.model.compression_ratio),
    )
    predictability_audit = query_prior_predictability_audit(
        points=test_points,
        boundaries=test_boundaries,
        eval_typed_queries=eval_workload.typed_queries,
        query_prior_field=trained.feature_context.get("query_prior_field"),
    )
    uniform_eval = matched.get("uniform")
    douglas_peucker_eval = matched.get("DouglasPeucker")
    workload_signature_gate = workload_distribution_comparison.get("workload_signature_gate", {})
    predictability_gate_pass = bool(predictability_audit.get("gate_pass", False))
    prior_predictive_alignment_gate = predictability_audit.get("prior_predictive_alignment_gate", {})
    prior_predictive_alignment_gate_pass = bool(
        isinstance(prior_predictive_alignment_gate, dict) and prior_predictive_alignment_gate.get("gate_pass", False)
    )
    signature_gate_pass = bool(
        isinstance(workload_signature_gate, dict)
        and workload_signature_gate.get("all_available")
        and workload_signature_gate.get("all_pass")
    )
    workload_stability_gate = _workload_stability_gate(
        config=config,
        train_label_workloads=train_label_workloads,
        eval_workload=eval_workload,
        selection_workload=selection_workload,
    )
    workload_stability_gate_pass = bool(workload_stability_gate.get("gate_pass", False))
    support_overlap_gate = _support_overlap_gate(
        train_points=train_points,
        eval_points=test_points,
        query_prior_field=trained.feature_context.get("query_prior_field"),
    )
    support_overlap_gate_pass = bool(support_overlap_gate.get("gate_pass", False))
    target_diffusion_gate = _target_diffusion_gate(trained.target_diagnostics)
    target_diffusion_gate_pass = bool(target_diffusion_gate.get("gate_pass", False))
    final_candidate = (
        str(config.query.workload_profile_id or "").lower() == "range_workload_v1"
        and str(config.model.model_type).lower() == "workload_blind_range_v2"
        and str(config.model.range_training_target_mode).lower() == "query_useful_v1_factorized"
        and str(getattr(config.model, "selector_type", "")).lower() == "learned_segment_budget_v1"
    )
    legacy_range_useful_summary = {
        "metric": "RangeUsefulLegacy",
        "schema": "range_usefulness_schema_version",
        "diagnostic_only": True,
        "mlqds_score": matched["MLQDS"].range_usefulness_score,
        "uniform_score": uniform_eval.range_usefulness_score if uniform_eval is not None else None,
        "douglas_peucker_score": (
            douglas_peucker_eval.range_usefulness_score
            if douglas_peucker_eval is not None
            else None
        ),
    }
    learned_slot_summary = _learned_slot_summary(
        selector_budget_diagnostics,
        float(config.model.compression_ratio),
        primary_selector_trace,
    )
    primary_eval = matched["MLQDS"]
    shuffled_delta = _query_useful_delta(primary_eval, causality_ablation_evaluations, "MLQDS_shuffled_scores")
    prior_only_delta = _query_useful_delta(
        primary_eval,
        causality_ablation_evaluations,
        "MLQDS_prior_field_only_score",
    )
    untrained_delta = _query_useful_delta(primary_eval, causality_ablation_evaluations, "MLQDS_untrained_model")
    shuffled_prior_delta = _query_useful_delta(
        primary_eval,
        causality_ablation_evaluations,
        "MLQDS_shuffled_prior_fields",
    )
    no_query_prior_delta = _query_useful_delta(
        primary_eval,
        causality_ablation_evaluations,
        "MLQDS_without_query_prior_features",
    )
    no_behavior_head_delta = _query_useful_delta(
        primary_eval,
        causality_ablation_evaluations,
        "MLQDS_without_behavior_utility_head",
    )
    no_segment_budget_head_delta = _query_useful_delta(
        primary_eval,
        causality_ablation_evaluations,
        "MLQDS_without_segment_budget_head",
    )
    no_fairness_preallocation_delta = _query_useful_delta(
        primary_eval,
        causality_ablation_evaluations,
        "MLQDS_without_trajectory_fairness_preallocation",
    )
    required_causality_ablation_names = (
        "MLQDS_shuffled_scores",
        "MLQDS_untrained_model",
        "MLQDS_shuffled_prior_fields",
        "MLQDS_without_query_prior_features",
        "MLQDS_without_behavior_utility_head",
        "MLQDS_without_segment_budget_head",
    )
    missing_causality_ablations = [
        name for name in required_causality_ablation_names if name not in causality_ablation_evaluations
    ]
    failed_causality_checks: list[str] = []
    delta_checks = {
        "shuffled_scores_should_lose": shuffled_delta,
        "untrained_model_should_lose": untrained_delta,
        "shuffled_prior_fields_should_lose": shuffled_prior_delta,
        "without_query_prior_features_should_lose": no_query_prior_delta,
        "without_behavior_utility_head_should_lose": no_behavior_head_delta,
        "without_segment_budget_head_should_lose": no_segment_budget_head_delta,
        "prior_field_only_should_not_match_trained": prior_only_delta,
    }
    delta_gate_config = _learning_causality_delta_gate_config(
        primary=primary_eval,
        uniform=uniform_eval,
    )
    delta_thresholds = delta_gate_config.get("thresholds", {})
    for check_name, delta in delta_checks.items():
        threshold = float(delta_thresholds.get(check_name, LEARNING_CAUSALITY_MIN_MATERIAL_DELTA))
        if delta is not None and float(delta) + 1e-12 < threshold:
            failed_causality_checks.append(check_name)
    prior_sample_failures = _prior_sample_gate_failures(prior_sensitivity_diagnostics)
    failed_causality_checks.extend(prior_sample_failures)
    learned_slot_fraction = float(learned_slot_summary.get("learned_controlled_retained_slot_fraction") or 0.0)
    learned_slot_fraction_min = 0.0
    if float(config.model.compression_ratio) >= 0.10:
        learned_slot_fraction_min = 0.35
    elif float(config.model.compression_ratio) >= 0.05:
        learned_slot_fraction_min = 0.25
    if learned_slot_fraction_min > 0.0 and learned_slot_fraction < learned_slot_fraction_min:
        failed_causality_checks.append("learned_controlled_slot_fraction_below_minimum")
    ablation_status = "not_run"
    if causality_ablation_evaluations or causal_ablation_freeze_failures:
        ablation_status = "complete" if not missing_causality_ablations and not causal_ablation_freeze_failures else "partial"
    learning_causality_gate_pass = (
        ablation_status == "complete" and not failed_causality_checks and not missing_causality_ablations
    )
    learning_causality_summary = {
        "selector_diagnostics_present": bool(selector_budget_diagnostics),
        "training_fit_diagnostics_present": bool(trained.fit_diagnostics),
        "selector_type": str(getattr(config.model, "selector_type", "temporal_hybrid")),
        "legacy_temporal_hybrid_selector": str(getattr(config.model, "selector_type", "temporal_hybrid")) != "learned_segment_budget_v1",
        "query_prior_field_available": bool(trained.feature_context.get("query_prior_field")),
        **learned_slot_summary,
        "shuffled_score_ablation_delta": shuffled_delta,
        "untrained_score_ablation_delta": untrained_delta,
        "shuffled_prior_field_ablation_delta": shuffled_prior_delta,
        "no_query_prior_field_ablation_delta": no_query_prior_delta,
        "no_behavior_head_ablation_delta": no_behavior_head_delta,
        "no_segment_budget_head_ablation_delta": no_segment_budget_head_delta,
        "no_trajectory_fairness_preallocation_ablation_delta": no_fairness_preallocation_delta,
        "segment_budget_head_ablation_mode": segment_budget_head_ablation_mode,
        "learned_segment_selector_config": {
            "geometry_gain_weight": float(config.model.learned_segment_geometry_gain_weight),
            "segment_score_blend_weight": float(config.model.learned_segment_score_blend_weight),
            "fairness_preallocation_enabled": bool(config.model.learned_segment_fairness_preallocation),
        },
        "prior_field_only_score_ablation_delta": prior_only_delta,
        "without_query_prior_features_delta": no_query_prior_delta,
        "learning_causality_delta_gate": delta_gate_config,
        "prior_sensitivity_diagnostics": prior_sensitivity_diagnostics,
        "prior_sample_gate_pass": not prior_sample_failures,
        "prior_sample_gate_failures": prior_sample_failures,
        "causality_ablation_scores": {
            name: metrics.query_useful_v1_score for name, metrics in causality_ablation_evaluations.items()
        },
        "causality_ablation_freeze_failures": causal_ablation_freeze_failures,
        "causality_ablation_missing": missing_causality_ablations,
        "learning_causality_gate_pass": learning_causality_gate_pass,
        "learning_causality_failed_checks": failed_causality_checks,
        "learned_controlled_slot_fraction_min": learned_slot_fraction_min,
        "learning_causality_ablation_status": ablation_status,
        "predictability_gate_pass": predictability_gate_pass,
        "prior_predictive_alignment_gate_pass": prior_predictive_alignment_gate_pass,
        "workload_signature_gate_pass": signature_gate_pass,
        "support_overlap_gate_pass": support_overlap_gate_pass,
    }
    global_sanity_gate = _global_sanity_gate(
        primary=matched["MLQDS"],
        uniform=uniform_eval,
        compression_ratio=float(config.model.compression_ratio),
    )
    global_sanity_gate_pass = bool(global_sanity_gate.get("gate_pass", False))
    blocking_gates: list[str] = []
    if final_candidate:
        if not workload_stability_gate_pass:
            blocking_gates.append("workload_stability_gate")
        if not support_overlap_gate_pass:
            blocking_gates.append("support_overlap_gate")
        if not predictability_gate_pass:
            blocking_gates.append("predictability_gate")
        if not prior_predictive_alignment_gate_pass:
            blocking_gates.append("prior_predictive_alignment_gate")
        if not target_diffusion_gate_pass:
            blocking_gates.append("target_diffusion_gate")
        if not signature_gate_pass:
            blocking_gates.append("workload_signature_gate")
        if not learning_causality_gate_pass:
            blocking_gates.append("learning_causality_ablations")
        if not global_sanity_gate_pass:
            blocking_gates.append("global_sanity_gates")
        blocking_gates.append("full_coverage_compression_grid")
        final_claim_summary = {
            "primary_metric": "QueryUsefulV1",
            "status": "candidate_blocked_by_required_gates" if blocking_gates else "candidate_ready_for_final_claim",
            "final_success_allowed": not blocking_gates,
            "blocking_gates": blocking_gates,
            "workload_stability_gate_pass": workload_stability_gate_pass,
            "support_overlap_gate_pass": support_overlap_gate_pass,
            "predictability_gate_pass": predictability_gate_pass,
            "prior_predictive_alignment_gate_pass": prior_predictive_alignment_gate_pass,
            "target_diffusion_gate_pass": target_diffusion_gate_pass,
            "workload_signature_gate_pass": signature_gate_pass,
            "learning_causality_gate_pass": learning_causality_gate_pass,
            "global_sanity_gate_pass": global_sanity_gate_pass,
            "mlqds_score": matched["MLQDS"].query_useful_v1_score,
            "uniform_score": uniform_eval.query_useful_v1_score if uniform_eval is not None else None,
            "douglas_peucker_score": (
                douglas_peucker_eval.query_useful_v1_score
                if douglas_peucker_eval is not None
                else None
            ),
            "reason": (
                "Single runs remain blocked until the benchmark-level full coverage/compression "
                "grid validates the numeric success bars."
            ),
        }
    else:
        final_claim_summary = {
            "primary_metric": None,
            "status": "not_final_query_driven_candidate",
            "final_success_allowed": False,
            "reason": "Requires range_workload_v1, QueryUsefulV1 factorized target, workload_blind_range_v2, and learned_segment_budget_v1.",
        }
    learning_causality_summary["final_success_allowed"] = bool(final_candidate and not blocking_gates)

    dump = {
        "config": config.to_dict(),
        "final_claim_summary": final_claim_summary,
        "diagnostic_summary": {
            "legacy_range_useful_available": True,
            "query_useful_v1_available": True,
            "range_component_diagnostics_available": True,
            "workload_blind_protocol_available": True,
            "predictability_audit_available": bool(predictability_audit.get("available", False)),
            "prior_predictive_alignment_gate_available": isinstance(prior_predictive_alignment_gate, dict),
            "workload_stability_gate_available": bool(workload_stability_gate),
            "support_overlap_gate_available": bool(support_overlap_gate),
            "global_sanity_gate_available": bool(global_sanity_gate),
            "target_diffusion_gate_available": bool(target_diffusion_gate),
            "workload_signature_gate_available": bool(
                isinstance(workload_signature_gate, dict) and workload_signature_gate.get("all_available")
            ),
        },
        "legacy_range_useful_summary": legacy_range_useful_summary,
        "learning_causality_summary": learning_causality_summary,
        "support_overlap_gate": support_overlap_gate,
        "global_sanity_gate": global_sanity_gate,
        "target_diffusion_gate": target_diffusion_gate,
        "workload": single_workload_type(eval_workload_map),
        "train_query_count": len(train_workload.typed_queries),
        "train_label_workload_count": len(train_label_workloads),
        "train_label_workload_query_counts": [len(workload.typed_queries) for workload in train_label_workloads],
        "eval_query_count": len(eval_workload.typed_queries),
        "selection_query_count": len(selection_workload.typed_queries) if selection_workload is not None else None,
        "train_query_coverage": train_workload.coverage_fraction,
        "train_label_workload_coverages": [workload.coverage_fraction for workload in train_label_workloads],
        "eval_query_coverage": eval_workload.coverage_fraction,
        "selection_query_coverage": selection_workload.coverage_fraction if selection_workload is not None else None,
        "query_generation_diagnostics": {
            "train": train_workload.generation_diagnostics,
            "train_label_workloads": [workload.generation_diagnostics for workload in train_label_workloads],
            "eval": eval_workload.generation_diagnostics,
            "selection": selection_workload.generation_diagnostics if selection_workload is not None else None,
        },
        "data_split_diagnostics": data_split.split_diagnostics,
        "selector_budget_diagnostics": selector_budget_diagnostics,
        "selector_trace_diagnostics": {
            "eval_primary": primary_selector_trace if primary_selector_trace is not None else {"available": False}
        },
        "matched": {name: _evaluation_metrics_payload(m) for name, m in matched.items()},
        "learning_causality_ablations": {
            name: _evaluation_metrics_payload(metrics)
            for name, metrics in causality_ablation_evaluations.items()
        },
        "learned_fill_diagnostics": {
            name: _evaluation_metrics_payload(metrics) for name, metrics in learned_fill_diagnostics.items()
        },
        "range_learned_fill_summary": range_learned_fill_summary,
        "predictability_audit": predictability_audit,
        "workload_stability_gate": workload_stability_gate,
        "range_compression_audit": range_compression_audit,
        "shift": shift_pairs,
        "training_history": trained.history,
        "training_target_diagnostics": trained.target_diagnostics,
        "training_fit_diagnostics": trained.fit_diagnostics,
        "range_training_target_transform": range_training_target_transform,
        "model_metadata": model_type_metadata(config.model.model_type),
        "query_prior_field": trained.feature_context.get("query_prior_field_metadata", {"available": False}),
        "range_target_balance": range_target_balance_diagnostics,
        "range_training_label_aggregation": range_training_label_aggregation,
        "teacher_distillation": teacher_distillation_diagnostics,
        "best_epoch": trained.best_epoch,
        "best_loss": trained.best_loss,
        "best_selection_score": trained.best_selection_score,
        "checkpoint_selection_metric": selection_metric,
        "checkpoint_selection_metric_requested": config.model.checkpoint_selection_metric,
        "checkpoint_score_variant": config.model.checkpoint_score_variant,
        "final_metrics_mode": config.baselines.final_metrics_mode,
        "workload_blind_protocol": {
            "enabled": bool(workload_blind_eval),
            "model_type": config.model.model_type,
            "masks_frozen_before_eval_query_scoring": bool(workload_blind_eval),
            "eval_queries_seen_by_model": False,
            "eval_queries_seen_by_feature_builder": False,
            "eval_queries_seen_by_selector": False,
            "checkpoint_selected_on_eval_queries": False,
            "query_conditioned_range_aware_used_for_product_acceptance": False,
            "primary_masks_frozen_before_eval_query_scoring": bool(workload_blind_eval),
            "audit_masks_frozen_before_eval_query_scoring": bool(
                workload_blind_eval and bool(frozen_audit_methods_by_ratio)
            ),
            "frozen_audit_ratio_count": int(len(frozen_audit_methods_by_ratio)),
            "frozen_method_names": sorted(frozen_primary_masks),
            "frozen_audit_ratios": sorted(frozen_audit_methods_by_ratio),
            "eval_geometry_blend_allowed": not bool(workload_blind_eval),
        },
        "range_usefulness_weight_summary": range_usefulness_weight_summary(),
        "checkpoint_smoothing_window": config.model.checkpoint_smoothing_window,
        "mlqds_score_mode": config.model.mlqds_score_mode,
        "mlqds_score_temperature": config.model.mlqds_score_temperature,
        "mlqds_rank_confidence_weight": config.model.mlqds_rank_confidence_weight,
        "mlqds_range_geometry_blend": config.model.mlqds_range_geometry_blend,
        "mlqds_hybrid_mode": config.model.mlqds_hybrid_mode,
        "mlqds_stratified_center_weight": config.model.mlqds_stratified_center_weight,
        "mlqds_min_learned_swaps": config.model.mlqds_min_learned_swaps,
        "oracle_diagnostic": {
            "kind": "additive_label_greedy",
            "enabled": run_oracle_baseline,
            "exact_optimum": False,
            "retained_mask_constructor": "per_trajectory_topk_with_endpoints",
            "purpose": "diagnostic label-greedy reference, not exact retained-set RangeUseful optimum",
        },
        "range_label_mode": config.model.range_label_mode,
        "range_boundary_prior_weight": config.model.range_boundary_prior_weight,
        "range_boundary_prior_enabled": config.model.range_boundary_prior_weight > 0.0,
        "data_audit": data_audit,
        "workload_diagnostics": range_diagnostics_summary,
        "workload_distribution_comparison": workload_distribution_comparison,
        "torch_runtime": {
            **torch_runtime_snapshot(),
            "amp": amp_runtime_snapshot(config.model.amp_mode),
        },
        "cuda_memory": {
            "training": training_cuda_memory,
        },
    }

    with _phase("write-results"):
        out_dir = write_experiment_results(
            results_dir=results_dir,
            matched_table=matched_table,
            shift_table=shift_table,
            geometric_table=geometric_table,
            range_usefulness_table=range_usefulness_table,
            learned_fill_table=learned_fill_table,
            learned_fill_diagnostics=learned_fill_diagnostics,
            range_learned_fill_summary=range_learned_fill_summary,
            range_compression_audit=range_compression_audit,
            range_compression_audit_table=range_compression_audit_table,
            range_diagnostics_summary=range_diagnostics_summary,
            workload_distribution_comparison=workload_distribution_comparison,
            range_diagnostics_rows=range_diagnostics_rows,
            dump=dump,
        )
        print(f"  wrote results to {out_dir}", flush=True)

    if save_simplified_dir:
        with _phase("write-simplified-csv"):
            out_dir = Path(save_simplified_dir)
            eval_mask = matched["MLQDS"].retained_mask
            if eval_mask is None:
                eval_mlqds = MLQDSMethod(
                    name="MLQDS",
                    trained=trained,
                    workload=eval_workload,
                    workload_type=single_workload_type(eval_workload_map),
                    score_mode=config.model.mlqds_score_mode,
                    score_temperature=config.model.mlqds_score_temperature,
                    rank_confidence_weight=config.model.mlqds_rank_confidence_weight,
                    temporal_fraction=config.model.mlqds_temporal_fraction,
                    diversity_bonus=config.model.mlqds_diversity_bonus,
                    hybrid_mode=config.model.mlqds_hybrid_mode,
                    selector_type=config.model.selector_type,
                    stratified_center_weight=config.model.mlqds_stratified_center_weight,
                    min_learned_swaps=config.model.mlqds_min_learned_swaps,
                    trajectory_mmsis=test_mmsis,
                    inference_batch_size=config.model.inference_batch_size,
                    amp_mode=config.model.amp_mode,
                )
                eval_mask = eval_mlqds.simplify(test_points, test_boundaries, config.model.compression_ratio)
            write_simplified_csv(
                str(out_dir / "ML_simplified_eval.csv"),
                test_points,
                test_boundaries,
                eval_mask,
                trajectory_mmsis=test_mmsis,
            )
            for ref_name, csv_name in (("uniform", "uniform_simplified_eval.csv"),
                                       ("DouglasPeucker", "DP_simplified_eval.csv")):
                ref_eval = matched.get(ref_name)
                ref_mask = ref_eval.retained_mask if ref_eval is not None else None
                if ref_mask is not None:
                    write_simplified_csv(
                        str(out_dir / csv_name),
                        test_points,
                        test_boundaries,
                        ref_mask,
                        trajectory_mmsis=test_mmsis,
                    )

        with _phase("trajectory-length-loss"):
            report_trajectory_length_loss(
                test_points,
                test_boundaries,
                eval_mask,
                top_k=25,
                trajectory_mmsis=test_mmsis,
            )

    print(f"[pipeline] total runtime {time.perf_counter() - pipeline_t0:.2f}s", flush=True)
    return ExperimentOutputs(
        matched_table=matched_table,
        shift_table=shift_table,
        metrics_dump=dump,
        geometric_table=geometric_table,
        range_usefulness_table=range_usefulness_table,
        range_compression_audit_table=range_compression_audit_table,
    )
