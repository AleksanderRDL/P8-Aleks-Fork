"""Focused tests for the query-driven Range-QDS rework path."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from data.ais_loader import generate_synthetic_ais_data
from evaluation.evaluate_methods import score_range_usefulness
from evaluation.metrics import MethodEvaluation, compute_length_preservation
from evaluation.query_useful_v1 import query_useful_v1_from_range_audit
from experiments.experiment_pipeline import (
    _causality_ablation_diagnostics_payload,
    _causality_ablation_tradeoff_summary,
    _factorized_head_probability_sources_from_logits,
    _global_sanity_gate,
    _head_ablation_sensitivity,
    _learned_segment_frozen_method,
    _learned_slot_summary,
    _learning_causality_delta_gate_config,
    _neutral_segment_scores_for_ablation,
    _prior_feature_sample_sensitivity,
    _prior_sample_gate_failures,
    _query_useful_component_delta_summary,
    _pre_repair_frozen_method_from_trace,
    _retained_mask_comparison,
    _reset_module_parameters,
    _max_length_required_mask,
    _score_protected_length_feasibility,
    _score_protected_length_frontier,
    _segment_score_quantile_bands_for_ablation,
    _segment_score_top_band_for_ablation,
    _segment_oracle_allocation_audit,
    _score_ablation_sensitivity,
    _support_overlap_gate,
    _target_diffusion_gate,
    _target_segment_oracle_alignment_audit,
    _workload_stability_gate,
)
from experiments.range_diagnostics import _range_workload_distribution_comparison
from models.workload_blind_range_v2 import WorkloadBlindRangeV2Model
from queries.query_generator import (
    _anchor_weights_for_family,
    _make_range_query,
    _profile_query_plan,
    _profile_query_settings,
    _weighted_choice_with_deterministic_key,
    generate_typed_query_workload,
)
from queries.query_types import QUERY_TYPE_ID_RANGE
from queries.workload_profiles import range_workload_profile
from simplification.learned_segment_budget import (
    blend_segment_support_scores,
    learned_segment_budget_diagnostics,
    simplify_with_learned_segment_budget_v1,
    simplify_with_learned_segment_budget_v1_with_trace,
)
from simplification.mlqds_scoring import simplify_mlqds_predictions
from training.model_features import (
    WORKLOAD_BLIND_RANGE_V2_MODEL_DISABLED_PRIOR_FIELDS,
    WORKLOAD_BLIND_RANGE_V2_POINT_DIM,
    build_workload_blind_range_v2_point_features,
)
from training.query_prior_fields import (
    QUERY_PRIOR_FIELD_NAMES,
    build_train_query_prior_fields,
    query_prior_field_metadata,
    sample_query_prior_fields,
    zero_query_prior_field_channels,
    zero_query_prior_field_like,
)
from training.query_useful_targets import (
    QUERY_USEFUL_V1_HEAD_NAMES,
    build_query_useful_v1_targets,
)
from training.predictability_audit import (
    _prior_channel_scores,
    _rankdata,
    query_prior_predictability_audit,
    query_prior_predictability_scores,
)
from training.train_model import (
    _factorized_final_score_composition_diagnostics,
    _factorized_head_fit_diagnostics,
    _fit_scaler_for_model,
    _initialize_factorized_head_output_biases_from_targets,
    _scalar_training_target_for_mode,
)
from training.training_diagnostics import _training_target_diagnostics
from training.training_epoch import _factorized_query_useful_loss, _segment_budget_head_segment_level_loss
from training.training_validation import (
    _validation_factorized_target_fit_metrics,
    _validation_query_useful_selection_score,
)


def _boundaries(trajectories: list[torch.Tensor]) -> list[tuple[int, int]]:
    cursor = 0
    out = []
    for trajectory in trajectories:
        end = cursor + int(trajectory.shape[0])
        out.append((cursor, end))
        cursor = end
    return out


def test_profile_query_plan_preserves_weighted_family_quotas() -> None:
    profile = range_workload_profile("range_workload_v1")

    plan = _profile_query_plan(profile, requested_queries=20, workload_seed=123)

    assert plan["enabled"] is True
    assert len(plan["anchor_family_sequence"]) == 20
    assert len(plan["footprint_family_sequence"]) == 20
    assert plan["anchor_family_planned_counts"] == {
        "density_route": 8,
        "boundary_entry_exit": 4,
        "crossing_turn_change": 3,
        "port_or_approach_zone": 3,
        "sparse_background_control": 2,
    }
    assert plan["footprint_family_planned_counts"] == {
        "small_local": 5,
        "medium_operational": 9,
        "large_context": 4,
        "route_corridor_like": 2,
    }


def test_profile_query_plan_prefixes_preserve_family_mix_when_workloads_expand() -> None:
    profile = range_workload_profile("range_workload_v1")

    plan = _profile_query_plan(profile, requested_queries=256, workload_seed=123)
    anchor_prefix = plan["anchor_family_sequence"][:48]
    footprint_prefix = plan["footprint_family_sequence"][:48]

    assert {family: anchor_prefix.count(family) for family in set(anchor_prefix)} == {
        "density_route": 19,
        "boundary_entry_exit": 10,
        "crossing_turn_change": 7,
        "port_or_approach_zone": 7,
        "sparse_background_control": 5,
    }
    assert {family: footprint_prefix.count(family) for family in set(footprint_prefix)} == {
        "small_local": 12,
        "medium_operational": 22,
        "large_context": 9,
        "route_corridor_like": 5,
    }


def test_range_workload_v1_footprints_match_rework_guide_defaults() -> None:
    profile = range_workload_profile("range_workload_v1")

    assert profile.footprint_families == {
        "small_local": {
            "spatial_radius_km": 1.1,
            "time_half_window_hours": 2.5,
            "elongation_allowed": False,
        },
        "medium_operational": {
            "spatial_radius_km": 2.2,
            "time_half_window_hours": 5.0,
            "elongation_allowed": False,
        },
        "large_context": {
            "spatial_radius_km": 4.0,
            "time_half_window_hours": 8.0,
            "elongation_allowed": False,
        },
        "route_corridor_like": {
            "spatial_radius_km": 2.2,
            "time_half_window_hours": 5.0,
            "elongation_allowed": True,
        },
    }


def test_range_workload_v1_target_coverage_keeps_requested_query_count() -> None:
    trajectories = generate_synthetic_ais_data(n_ships=5, n_points_per_ship=48, seed=87, route_families=1)
    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=8,
        workload_map={"range": 1.0},
        seed=14,
        target_coverage=0.05,
        max_queries=64,
        workload_profile_id="range_workload_v1",
        coverage_calibration_mode="profile_sampled_query_count",
        range_max_point_hit_fraction=1.0,
        range_duplicate_iou_threshold=1.0,
    )

    generation = (workload.generation_diagnostics or {})["query_generation"]
    assert generation["query_count_mode"] == "calibrated_to_coverage"
    assert generation["coverage_calibration_mode"] == "profile_sampled_query_count"
    assert generation["final_query_count"] >= 8


def test_range_workload_v1_records_profile_signature() -> None:
    trajectories = generate_synthetic_ais_data(n_ships=5, n_points_per_ship=48, seed=81)
    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=6,
        workload_map={"range": 1.0},
        seed=9,
        workload_profile_id="range_workload_v1",
        range_max_point_hit_fraction=1.0,
        range_duplicate_iou_threshold=1.0,
    )

    diagnostics = workload.generation_diagnostics or {}
    signature = diagnostics["workload_signature"]
    profile = diagnostics["workload_profile"]
    generation = diagnostics["query_generation"]

    assert profile["profile_id"] == "range_workload_v1"
    assert generation["range_time_domain_mode"] == "anchor_day"
    assert signature["profile_id"] == "range_workload_v1"
    assert sum(signature["anchor_family_counts"].values()) == len(workload.typed_queries)
    assert sum(signature["footprint_family_counts"].values()) == len(workload.typed_queries)
    assert signature["query_count"] == len(workload.typed_queries)
    assert signature["total_points"] == sum(int(trajectory.shape[0]) for trajectory in trajectories)
    assert signature["total_trajectories"] == len(trajectories)
    assert len(signature["point_hit_counts_per_query"]) == len(workload.typed_queries)
    assert len(signature["point_hit_fractions_per_query"]) == len(workload.typed_queries)
    assert len(signature["ship_hit_counts_per_query"]) == len(workload.typed_queries)
    assert len(signature["ship_hit_fractions_per_query"]) == len(workload.typed_queries)


def test_deterministic_profile_sampling_does_not_advance_generator() -> None:
    profile = range_workload_profile("range_workload_v1")
    gen = torch.Generator().manual_seed(12345)
    before = gen.get_state()

    chosen_anchor = _weighted_choice_with_deterministic_key(
        profile.anchor_family_weights,
        gen,
        fallback="density_route",
        deterministic_value=0.33,
    )
    chosen_footprint = _weighted_choice_with_deterministic_key(
        profile.footprint_family_weights,
        gen,
        fallback="medium_operational",
        deterministic_value=0.77,
    )

    settings = _profile_query_settings(profile, torch.Generator().manual_seed(1), query_index=7, workload_seed=19)
    settings_deterministic = _profile_query_settings(profile, torch.Generator().manual_seed(1), query_index=7, workload_seed=19)
    assert settings == settings_deterministic

    after = gen.get_state()
    assert chosen_anchor in profile.anchor_family_weights
    assert chosen_footprint in profile.footprint_family_weights
    assert torch.equal(before, after)

    baseline = torch.Generator().manual_seed(12345)
    expected_seq = torch.randint(0, 999, (3,), generator=baseline).tolist()
    observed_seq = torch.randint(0, 999, (3,), generator=gen).tolist()
    assert observed_seq == expected_seq


def test_synthetic_route_families_create_same_support_trajectories() -> None:
    trajectories = generate_synthetic_ais_data(n_ships=5, n_points_per_ship=32, seed=86, route_families=1)
    points = torch.cat(trajectories, dim=0)

    assert float(points[:, 1].max().item() - points[:, 1].min().item()) < 0.50
    assert float(points[:, 2].max().item() - points[:, 2].min().item()) < 0.50


def test_query_useful_v1_prioritizes_query_local_components() -> None:
    weak = {
        "range_point_f1": 0.1,
        "range_ship_coverage": 0.1,
        "range_ship_f1": 0.1,
        "range_turn_coverage": 0.1,
        "range_shape_score": 0.1,
        "range_entry_exit_f1": 0.1,
        "range_crossing_f1": 0.1,
    }
    strong = dict(weak)
    strong.update(
        {
            "range_point_f1": 0.7,
            "range_ship_coverage": 0.6,
            "range_turn_coverage": 0.8,
            "range_shape_score": 0.7,
        }
    )

    strong_score = float(cast(Any, query_useful_v1_from_range_audit(strong)["query_useful_v1_score"]))
    weak_score = float(cast(Any, query_useful_v1_from_range_audit(weak)["query_useful_v1_score"]))
    assert strong_score > weak_score


def test_query_useful_v1_has_true_query_local_interpolation_component() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 2.0],
            [3.0, 0.0, 3.0],
            [4.0, 0.0, 4.0],
        ],
        dtype=torch.float32,
    )
    retained = torch.tensor([True, False, False, False, True])
    query = {
        "type": "range",
        "params": {
            "t_start": 0.5,
            "t_end": 3.5,
            "lat_min": -1.0,
            "lat_max": 1.0,
            "lon_min": 0.5,
            "lon_max": 3.5,
        },
    }

    audit = score_range_usefulness(
        points=points,
        boundaries=[(0, 5)],
        retained_mask=retained,
        typed_queries=[query],
    )
    useful = query_useful_v1_from_range_audit(audit)
    components = cast(dict[str, float], useful["query_useful_v1_components"])

    assert audit["range_shape_score"] == 0.0
    assert audit["range_query_local_interpolation_fidelity"] == 0.0
    assert components["query_local_interpolation_fidelity"] == 0.0
    assert useful["query_useful_v1_metric_maturity"] == "bridge_with_true_query_local_interpolation_component"


def test_validation_query_useful_penalizes_bad_global_sanity() -> None:
    cfg = SimpleNamespace(
        validation_global_sanity_penalty_enabled=True,
        validation_global_sanity_penalty_weight=0.10,
        validation_sed_penalty_weight=0.05,
        validation_endpoint_penalty_weight=0.10,
        validation_length_preservation_min=0.80,
    )
    good = {
        "avg_length_preserved": 0.90,
        "avg_sed_ratio_vs_uniform": 1.00,
        "avg_sed_ratio_vs_uniform_max": 1.50,
        "endpoint_sanity": 1.00,
    }
    bad = {
        "avg_length_preserved": 0.40,
        "avg_sed_ratio_vs_uniform": 2.50,
        "avg_sed_ratio_vs_uniform_max": 1.50,
        "endpoint_sanity": 0.00,
    }

    assert _validation_query_useful_selection_score(0.50, bad, cast(Any, cfg)) < (
        _validation_query_useful_selection_score(0.50, good, cast(Any, cfg)) - 0.10
    )


def test_validation_factorized_target_fit_metrics_are_diagnostic_only() -> None:
    points = torch.zeros((8, 8), dtype=torch.float32)
    points[:, 0] = torch.arange(8, dtype=torch.float32)
    points[:, 1] = torch.linspace(0.0, 0.7, steps=8)
    points[:, 2] = torch.linspace(0.0, 0.7, steps=8)
    points[:, 7] = torch.tensor([0.0, 0.2, 1.0, 0.3, 0.0, 0.0, 0.0, 0.0])
    query = {
        "type": "range",
        "params": {
            "t_start": -0.5,
            "t_end": 3.5,
            "lat_min": -0.1,
            "lat_max": 0.35,
            "lon_min": -0.1,
            "lon_max": 0.35,
        },
    }
    workload = SimpleNamespace(typed_queries=[query])
    targets = build_query_useful_v1_targets(
        points=points,
        boundaries=[(0, 8)],
        typed_queries=[query],
        segment_size=4,
    )
    head_logits = torch.logit(targets.head_targets.clamp(1e-4, 1.0 - 1e-4))

    metrics = _validation_factorized_target_fit_metrics(
        head_logits=head_logits,
        points=points,
        boundaries=[(0, 8)],
        workload=cast(Any, workload),
        segment_size=4,
    )

    assert metrics["factorized_target_fit_available"] == 1.0
    assert metrics["factorized_target_fit_used_for_checkpoint_selection"] == 0.0
    assert metrics["head_segment_budget_target_target_fit_available"] == 1.0
    assert metrics["head_segment_budget_target_top5_mass_recall"] > 0.99
    assert metrics["segment_budget_canonical_segment_fit_available"] == 1.0
    assert metrics["segment_budget_canonical_segment_top5_mass_recall"] > 0.99


def test_factorized_targets_and_prior_fields_are_train_query_derived() -> None:
    trajectories = generate_synthetic_ais_data(n_ships=4, n_points_per_ship=32, seed=82)
    points = torch.cat(trajectories, dim=0)
    boundaries = _boundaries(trajectories)
    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=5,
        workload_map={"range": 1.0},
        seed=12,
        workload_profile_id="range_workload_v1",
        range_max_point_hit_fraction=1.0,
        range_duplicate_iou_threshold=1.0,
    )

    targets = build_query_useful_v1_targets(
        points=points,
        boundaries=boundaries,
        typed_queries=workload.typed_queries,
    )
    prior = build_train_query_prior_fields(
        points=points,
        boundaries=boundaries,
        typed_queries=workload.typed_queries,
        labels=targets.labels,
        workload_profile_id="range_workload_v1",
        train_workload_seed=12,
    )

    assert targets.head_targets.shape == (points.shape[0], len(QUERY_USEFUL_V1_HEAD_NAMES))
    assert targets.labels.shape[0] == points.shape[0]
    assert targets.diagnostics["target_family"] == "QueryUsefulV1Factorized"
    assert "support_fraction_by_threshold_by_head" in targets.diagnostics
    assert "final_label_support_fraction_by_threshold" in targets.diagnostics
    assert prior["built_from_split"] == "train_only"
    assert prior["contains_eval_queries"] is False
    assert prior["contains_validation_queries"] is False


def test_factorized_replacement_target_is_query_local_and_final_label_keeps_query_mass() -> None:
    points = torch.zeros((10, 8), dtype=torch.float32)
    points[:, 0] = torch.arange(10, dtype=torch.float32)
    points[:, 1] = torch.linspace(0.0, 1.0, steps=10)
    points[:, 2] = torch.linspace(0.0, 1.0, steps=10)
    points[:, 5] = 0.0
    points[:, 6] = 0.0
    points[0, 5] = 1.0
    points[-1, 6] = 1.0
    points[:, 7] = 1.0
    query = {
        "type": "range",
        "params": {
            "t_start": -1.0,
            "t_end": 10.0,
            "lat_min": -1.0,
            "lat_max": 2.0,
            "lon_min": -1.0,
            "lon_max": 2.0,
        },
    }

    targets = build_query_useful_v1_targets(
        points=points,
        boundaries=[(0, 10)],
        typed_queries=[query],
    )

    replacement = targets.head_targets[:, tuple(QUERY_USEFUL_V1_HEAD_NAMES).index("replacement_representative_value")]
    final_score = targets.labels[:, QUERY_TYPE_ID_RANGE]
    assert int((replacement > 0.0).sum().item()) == 5
    assert int((final_score > 0.0).sum().item()) == 10
    assert targets.diagnostics["replacement_representative_value_normalization"] == "conditional_on_query_hit"
    assert (
        targets.diagnostics["final_label_formula"]
        == "query_hit_times_behavior_with_conditional_replacement_modulation_plus_boundary"
    )


def test_conditional_behavior_target_is_masked_to_query_hits() -> None:
    points = torch.zeros((6, 8), dtype=torch.float32)
    points[:, 0] = torch.arange(6, dtype=torch.float32)
    points[:, 1] = torch.arange(6, dtype=torch.float32)
    points[:, 2] = 0.0
    points[0, 5] = 1.0
    points[-1, 6] = 1.0
    points[2, 7] = 1.0
    query = {
        "type": "range",
        "params": {
            "t_start": 1.0,
            "t_end": 3.0,
            "lat_min": 1.0,
            "lat_max": 3.0,
            "lon_min": -1.0,
            "lon_max": 1.0,
        },
    }

    targets = build_query_useful_v1_targets(
        points=points,
        boundaries=[(0, 6)],
        typed_queries=[query],
    )
    query_hit_idx = tuple(QUERY_USEFUL_V1_HEAD_NAMES).index("query_hit_probability")
    behavior_idx = tuple(QUERY_USEFUL_V1_HEAD_NAMES).index("conditional_behavior_utility")

    hit_mask = targets.head_targets[:, query_hit_idx] > 0.0

    assert torch.equal(targets.head_mask[:, behavior_idx], hit_mask)
    assert targets.head_mask[:, query_hit_idx].all()
    assert targets.diagnostics["conditional_behavior_utility_training"] == "masked_to_query_hit_points"


def test_path_length_support_target_is_query_free_segment_geometry() -> None:
    points = torch.zeros((8, 8), dtype=torch.float32)
    points[:, 0] = torch.arange(8, dtype=torch.float32)
    points[:, 1] = torch.arange(8, dtype=torch.float32)
    points[:, 2] = torch.tensor([0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    first_query = {
        "type": "range",
        "params": {
            "t_start": 0.0,
            "t_end": 7.0,
            "lat_min": 0.0,
            "lat_max": 3.5,
            "lon_min": -1.0,
            "lon_max": 4.0,
        },
    }
    second_query = {
        "type": "range",
        "params": {
            "t_start": 3.0,
            "t_end": 7.0,
            "lat_min": 3.0,
            "lat_max": 7.5,
            "lon_min": -1.0,
            "lon_max": 1.0,
        },
    }

    first_targets = build_query_useful_v1_targets(
        points=points,
        boundaries=[(0, 8)],
        typed_queries=[first_query],
        segment_size=2,
    )
    second_targets = build_query_useful_v1_targets(
        points=points,
        boundaries=[(0, 8)],
        typed_queries=[second_query],
        segment_size=2,
    )
    path_idx = tuple(QUERY_USEFUL_V1_HEAD_NAMES).index("path_length_support_target")
    first_path = first_targets.head_targets[:, path_idx]
    second_path = second_targets.head_targets[:, path_idx]

    assert torch.allclose(first_path, second_path)
    assert float(first_path.sum().item()) > 0.0
    assert int((first_path > 0.0).sum().item()) < int(first_path.numel())
    assert first_targets.head_mask[:, path_idx].all()
    assert first_targets.diagnostics["path_length_support_target_query_free"] is True
    assert first_targets.diagnostics["path_length_support_target_highpass_quantile"] == 0.50
    assert (
        first_targets.diagnostics["path_length_support_target_base_source"]
        == "per_point_path_length_removal_loss_segment_highpass_mass"
    )


def test_conditional_behavior_target_alignment_diagnostics_report_final_mass_recall() -> None:
    points = torch.zeros((10, 8), dtype=torch.float32)
    points[:, 0] = torch.arange(10, dtype=torch.float32)
    points[:, 1] = torch.arange(10, dtype=torch.float32)
    points[:, 2] = 0.0
    points[:, 7] = torch.linspace(0.0, 1.0, steps=10)
    query = {
        "type": "range",
        "params": {
            "t_start": -1.0,
            "t_end": 10.0,
            "lat_min": -1.0,
            "lat_max": 10.0,
            "lon_min": -1.0,
            "lon_max": 1.0,
        },
    }

    targets = build_query_useful_v1_targets(
        points=points,
        boundaries=[(0, 5), (5, 10)],
        typed_queries=[query],
    )
    alignment = targets.diagnostics["conditional_behavior_target_alignment"]

    assert alignment["valid_point_count"] == 10
    assert alignment["topk_ratio"] == 0.05
    assert alignment["spearman_with_final_score"] is not None
    assert alignment["spearman_with_final_score"] > 0.0
    assert alignment["topk_final_score_mass_recall_ranked_by_behavior"] > 0.0
    assert "spearman_with_ship_query_evidence" in alignment
    assert "topk_ship_query_evidence_mass_recall_ranked_by_behavior" in alignment
    assert "spearman_with_replacement_representative_value" in alignment
    assert "topk_overlap_with_segment_budget_target" in alignment
    candidates = targets.diagnostics["conditional_behavior_candidate_alignment"]
    current = candidates["current_local_behavior"]
    replacement_gated = candidates["replacement_gated_local_behavior"]

    assert set(candidates) == {
        "current_local_behavior",
        "replacement_gated_local_behavior",
        "segment_gated_local_behavior",
        "replacement_support_only_local_behavior",
        "replacement_segment_gated_local_behavior",
    }
    assert current["valid_point_count"] == 10
    assert replacement_gated["valid_point_count"] == 10
    assert "support_fraction_by_threshold" in replacement_gated
    assert "topk_final_score_mass_recall_ranked_by_behavior" in replacement_gated
    assert current["ship_query_pair_count"] == 2
    assert current["ship_query_topk_selected_point_count"] == 1
    assert current["ship_query_pair_coverage_at_topk"] == 0.5
    assert "topk_ship_query_evidence_mass_recall_ranked_by_behavior" in replacement_gated


def test_target_diffusion_gate_blocks_broad_low_budget_labels() -> None:
    diagnostics = {
        "query_useful_v1_factorized": {
            "final_label_support_fraction_by_threshold": {"gt_0.01": 0.80},
            "support_fraction_by_threshold_by_head": {
                "query_hit_probability": {"gt_0.01": 0.70},
                "conditional_behavior_utility": {"gt_0.01": 0.70},
                "replacement_representative_value": {"gt_0.05": 0.20},
            },
            "topk_label_mass_budget_grid": {
                "query_hit_probability": {"0.05": 0.08},
                "conditional_behavior_utility": {"0.05": 0.08},
                "replacement_representative_value": {"0.05": 0.35},
            },
        }
    }

    gate = _target_diffusion_gate(diagnostics)

    assert gate["gate_pass"] is False
    assert "final_label_support_fraction_above_max" in gate["failed_checks"]
    assert "conditional_behavior_utility:support_fraction_above_max" in gate["failed_checks"]
    assert "conditional_behavior_utility:top5_label_mass_below_min" in gate["failed_checks"]
    assert "query_hit_probability:support_fraction_above_max" not in gate["failed_checks"]


def test_target_diffusion_gate_accepts_concentrated_factorized_labels() -> None:
    diagnostics = {
        "query_useful_v1_factorized": {
            "final_label_support_fraction_by_threshold": {"gt_0.01": 0.30},
            "support_fraction_by_threshold_by_head": {
                "query_hit_probability": {"gt_0.01": 0.35},
                "conditional_behavior_utility": {"gt_0.01": 0.20},
                "replacement_representative_value": {"gt_0.05": 0.20},
            },
            "topk_label_mass_budget_grid": {
                "query_hit_probability": {"0.05": 0.25},
                "conditional_behavior_utility": {"0.05": 0.35},
                "replacement_representative_value": {"0.05": 0.35},
            },
        }
    }

    gate = _target_diffusion_gate(diagnostics)

    assert gate["gate_pass"] is True
    assert gate["failed_checks"] == []


def test_prior_behavior_field_uses_behavior_values_not_hit_probability() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 2.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    behavior_values = torch.tensor([0.9, 0.2, 0.7], dtype=torch.float32)
    labels = torch.zeros((3, QUERY_TYPE_ID_RANGE + 1), dtype=torch.float32)
    labels[:, QUERY_TYPE_ID_RANGE] = 1.0
    query = {
        "type": "range",
        "params": {
            "t_start": -1.0,
            "t_end": 3.0,
            "lat_min": -1.0,
            "lat_max": 3.0,
            "lon_min": -1.0,
            "lon_max": 1.0,
        },
    }

    prior = build_train_query_prior_fields(
        points=points,
        boundaries=[(0, 3)],
        typed_queries=[query],
        labels=labels,
        behavior_values=behavior_values,
        workload_profile_id="range_workload_v1",
        grid_bins=4,
        smoothing_passes=0,
    )
    features = build_workload_blind_range_v2_point_features(points, prior)

    spatial_query_hit_probability = features[:, -6]
    behavior_utility_prior = features[:, -2]
    assert torch.allclose(spatial_query_hit_probability, torch.ones_like(spatial_query_hit_probability))
    assert torch.allclose(behavior_utility_prior, behavior_values)
    assert not torch.allclose(behavior_utility_prior, spatial_query_hit_probability)


def test_range_v2_scaler_preserves_semantic_zero_for_prior_ablation() -> None:
    points = torch.zeros((3, WORKLOAD_BLIND_RANGE_V2_POINT_DIM), dtype=torch.float32)
    points[:, -6:] = torch.tensor(
        [
            [0.20, 0.10, 0.01, 0.02, 0.30, 0.50],
            [0.25, 0.20, 0.02, 0.03, 0.40, 0.60],
            [0.30, 0.30, 0.03, 0.04, 0.50, 0.70],
        ],
        dtype=torch.float32,
    )
    queries = torch.zeros((1, 12), dtype=torch.float32)

    scaler = _fit_scaler_for_model(points, queries, "workload_blind_range_v2")
    zero_prior_points = points.clone()
    zero_prior_points[:, -6:] = 0.0
    transformed = scaler.transform_points(zero_prior_points)

    assert torch.allclose(scaler.point_min[-6:], torch.zeros(6))
    assert torch.allclose(scaler.point_max[-6:], torch.ones(6))
    assert torch.allclose(transformed[:, -6:], torch.zeros((3, 6)))


def test_query_prior_field_rasterizes_query_boxes_not_only_hit_points() -> None:
    train_points = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    eval_points = torch.tensor([[0.5, 2.0, 2.0], [0.5, 4.0, 4.0]], dtype=torch.float32)
    query = {
        "type": "range",
        "params": {
            "t_start": 0.0,
            "t_end": 1.0,
            "lat_min": 1.5,
            "lat_max": 2.5,
            "lon_min": 1.5,
            "lon_max": 2.5,
        },
    }

    prior = build_train_query_prior_fields(
        points=train_points,
        boundaries=[(0, 1)],
        typed_queries=[query],
        workload_profile_id="range_workload_v1",
        grid_bins=8,
        time_bins=4,
        smoothing_passes=0,
    )
    sampled = sample_query_prior_fields(eval_points, prior)

    assert prior["spatial_query_field_source"] == "train_query_box_density"
    assert prior["out_of_extent_sampling"] == "zero"
    assert prior["diagnostics"]["raw_nonzero_point_hit_cells"] == 0
    assert prior["diagnostics"]["raw_nonzero_spatial_query_cells"] > 0
    assert float(sampled[0, 0].item()) > 0.0
    assert float(sampled[0, 1].item()) > 0.0
    assert torch.allclose(sampled[1], torch.zeros_like(sampled[1]))


def test_sample_query_prior_fields_nearest_mode_clamps_out_of_extent_points() -> None:
    train_points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
    eval_points = torch.tensor(
        [[0.5, 5.0, 5.0], [0.25, -2.0, -3.0]],
        dtype=torch.float32,
    )
    query = {
        "type": "range",
        "params": {
            "t_start": 0.0,
            "t_end": 1.0,
            "lat_min": 0.0,
            "lat_max": 1.0,
            "lon_min": 0.0,
            "lon_max": 1.0,
        },
    }
    prior = build_train_query_prior_fields(
        points=train_points,
        boundaries=[(0, 2)],
        typed_queries=[query],
        workload_profile_id="range_workload_v1",
        grid_bins=8,
        time_bins=2,
        smoothing_passes=0,
        out_of_extent_sampling="nearest",
    )
    sampled_nearest = sample_query_prior_fields(eval_points, prior)
    sampled_zero = sample_query_prior_fields(
        eval_points,
        dict(prior, out_of_extent_sampling="zero"),
    )

    assert prior["out_of_extent_sampling"] == "nearest"
    assert bool((sampled_nearest.abs().sum(dim=1) > 0.0).all().item())
    assert torch.allclose(sampled_zero, torch.zeros_like(sampled_zero))


def test_zero_prior_field_like_preserves_metadata_and_shape() -> None:
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
    query = {
        "type": "range",
        "params": {
            "t_start": -1.0,
            "t_end": 2.0,
            "lat_min": -1.0,
            "lat_max": 2.0,
            "lon_min": -1.0,
            "lon_max": 2.0,
        },
    }
    prior = build_train_query_prior_fields(
        points=points,
        boundaries=[(0, 2)],
        typed_queries=[query],
        workload_profile_id="range_workload_v1",
        grid_bins=4,
        time_bins=2,
        smoothing_passes=0,
    )

    zeroed = zero_query_prior_field_like(prior)

    assert zeroed["extent"] == prior["extent"]
    assert zeroed["grid_bins"] == prior["grid_bins"]
    assert zeroed["time_bins"] == prior["time_bins"]
    assert zeroed["ablation"] == "zero_query_prior_features"
    assert query_prior_field_metadata(zeroed)["contains_eval_queries"] is False
    for name in zeroed["field_names"]:
        assert zeroed[name].shape == prior[name].shape
        assert torch.count_nonzero(zeroed[name]).item() == 0


def test_zero_query_prior_field_channels_only_zeros_requested_channels() -> None:
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
    query = {
        "type": "range",
        "params": {
            "t_start": -1.0,
            "t_end": 2.0,
            "lat_min": -1.0,
            "lat_max": 2.0,
            "lon_min": -1.0,
            "lon_max": 2.0,
        },
    }
    prior = build_train_query_prior_fields(
        points=points,
        boundaries=[(0, 2)],
        typed_queries=[query],
        workload_profile_id="range_workload_v1",
        grid_bins=4,
        time_bins=2,
        smoothing_passes=0,
    )

    zeroed = zero_query_prior_field_channels(prior, ["route_density_prior"])

    assert zeroed["extent"] == prior["extent"]
    assert zeroed["ablation"] == "zero_query_prior_channels"
    assert zeroed["zeroed_prior_channels"] == ["route_density_prior"]
    assert query_prior_field_metadata(zeroed)["contains_eval_queries"] is False
    assert torch.count_nonzero(zeroed["route_density_prior"]).item() == 0
    for name in zeroed["field_names"]:
        if name == "route_density_prior":
            continue
        assert torch.equal(zeroed[name], prior[name])


def test_no_query_prior_ablation_preserves_train_extent() -> None:
    train_points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 10.0, 10.0]], dtype=torch.float32)
    eval_points = train_points.clone()
    query = {
        "type": "range",
        "params": {
            "t_start": -1.0,
            "t_end": 2.0,
            "lat_min": -1.0,
            "lat_max": 11.0,
            "lon_min": -1.0,
            "lon_max": 11.0,
        },
    }
    prior = build_train_query_prior_fields(
        points=train_points,
        boundaries=[(0, 2)],
        typed_queries=[query],
        workload_profile_id="range_workload_v1",
        grid_bins=4,
        time_bins=2,
        smoothing_passes=0,
    )
    zeroed = zero_query_prior_field_like(prior)

    with_prior = build_workload_blind_range_v2_point_features(eval_points, prior)
    no_prior = build_workload_blind_range_v2_point_features(eval_points, zeroed)
    without_field = build_workload_blind_range_v2_point_features(eval_points, None)

    assert with_prior.shape == no_prior.shape == without_field.shape
    assert torch.allclose(with_prior[:, :-6], no_prior[:, :-6])
    assert not torch.allclose(no_prior[:, :-6], without_field[:, :-6])
    assert torch.count_nonzero(no_prior[:, -6:]).item() == 0


def test_workload_blind_range_v2_excludes_route_density_from_model_features() -> None:
    train_points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 10.0, 10.0]], dtype=torch.float32)
    eval_points = train_points.clone()
    query = {
        "type": "range",
        "params": {
            "t_start": -1.0,
            "t_end": 2.0,
            "lat_min": -1.0,
            "lat_max": 11.0,
            "lon_min": -1.0,
            "lon_max": 11.0,
        },
    }
    prior = build_train_query_prior_fields(
        points=train_points,
        boundaries=[(0, 2)],
        typed_queries=[query],
        workload_profile_id="range_workload_v1",
        grid_bins=4,
        time_bins=2,
        smoothing_passes=0,
    )
    route_idx = list(QUERY_PRIOR_FIELD_NAMES).index("route_density_prior")

    sampled = sample_query_prior_fields(eval_points, prior)
    features = build_workload_blind_range_v2_point_features(eval_points, prior)

    assert WORKLOAD_BLIND_RANGE_V2_MODEL_DISABLED_PRIOR_FIELDS == ("route_density_prior",)
    assert torch.count_nonzero(sampled[:, route_idx]).item() > 0
    prior_start = -len(QUERY_PRIOR_FIELD_NAMES)
    assert torch.count_nonzero(features[:, prior_start + route_idx]).item() == 0
    for idx, name in enumerate(QUERY_PRIOR_FIELD_NAMES):
        if name == "route_density_prior":
            continue
        assert torch.equal(features[:, prior_start + idx], sampled[:, idx])


def test_support_overlap_gate_passes_same_support_eval_points() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.5, 0.5],
            [2.0, 1.0, 1.0],
            [3.0, 0.25, 0.75],
        ],
        dtype=torch.float32,
    )
    query = {
        "type": "range",
        "params": {
            "t_start": -1.0,
            "t_end": 4.0,
            "lat_min": -0.5,
            "lat_max": 1.5,
            "lon_min": -0.5,
            "lon_max": 1.5,
        },
    }
    prior = build_train_query_prior_fields(
        points=points,
        boundaries=[(0, 4)],
        typed_queries=[query],
        workload_profile_id="range_workload_v1",
        grid_bins=4,
        time_bins=2,
        smoothing_passes=0,
    )

    gate = _support_overlap_gate(train_points=points, eval_points=points, query_prior_field=prior)

    assert gate["gate_pass"] is True
    assert gate["failed_checks"] == []
    assert gate["eval_points_outside_train_prior_extent_fraction"] == 0.0
    assert gate["sampled_prior_nonzero_fraction"] >= 0.50


def test_support_overlap_gate_blocks_out_of_extent_eval_points() -> None:
    train_points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
    eval_points = torch.tensor([[0.0, 10.0, 10.0], [1.0, 11.0, 11.0]], dtype=torch.float32)
    query = {
        "type": "range",
        "params": {
            "t_start": -1.0,
            "t_end": 2.0,
            "lat_min": -0.5,
            "lat_max": 1.5,
            "lon_min": -0.5,
            "lon_max": 1.5,
        },
    }
    prior = build_train_query_prior_fields(
        points=train_points,
        boundaries=[(0, 2)],
        typed_queries=[query],
        workload_profile_id="range_workload_v1",
        grid_bins=4,
        time_bins=2,
        smoothing_passes=0,
    )

    gate = _support_overlap_gate(train_points=train_points, eval_points=eval_points, query_prior_field=prior)

    assert gate["gate_pass"] is False
    assert "eval_points_outside_train_prior_extent_too_high" in gate["failed_checks"]
    assert "sampled_prior_nonzero_fraction_too_low" in gate["failed_checks"]


def test_workload_signature_gate_rejects_profile_mismatch_and_tiny_query_counts() -> None:
    summaries = {
        "train": {
            "range": {"range_query_count": 4},
            "range_signal": {},
            "generation": {
                "workload_signature": {
                    "profile_id": "legacy_generator",
                    "query_count": 4,
                    "anchor_family_counts": {"density_route": 4},
                    "footprint_family_counts": {"medium_operational": 4},
                    "point_hit_counts_per_query": [1, 2, 3, 4],
                    "ship_hit_counts_per_query": [1, 1, 2, 2],
                    "near_duplicate_rate": 0.0,
                    "broad_query_rate": 0.0,
                }
            },
        },
        "eval": {
            "range": {"range_query_count": 4},
            "range_signal": {},
            "generation": {
                "workload_signature": {
                    "profile_id": "range_workload_v1",
                    "query_count": 4,
                    "anchor_family_counts": {"density_route": 4},
                    "footprint_family_counts": {"medium_operational": 4},
                    "point_hit_counts_per_query": [1, 2, 3, 4],
                    "ship_hit_counts_per_query": [1, 1, 2, 2],
                    "near_duplicate_rate": 0.0,
                    "broad_query_rate": 0.0,
                }
            },
        },
    }

    gate = _range_workload_distribution_comparison(summaries)["workload_signature_gate"]["pairs"]["train"]

    assert gate["gate_pass"] is False
    assert "profile_id_mismatch" in gate["failed_checks"]
    assert "train_signature_query_count_below_min" in gate["failed_checks"]
    assert "eval_signature_query_count_below_min" in gate["failed_checks"]


def test_workload_signature_gate_rejects_query_count_mismatch() -> None:
    def signature(query_count: int) -> dict[str, Any]:
        return {
            "profile_id": "range_workload_v1",
            "query_count": query_count,
            "anchor_family_counts": {"density_route": query_count},
            "footprint_family_counts": {"medium_operational": query_count},
            "point_hit_counts_per_query": [3 for _ in range(query_count)],
            "ship_hit_counts_per_query": [1 for _ in range(query_count)],
            "near_duplicate_rate": 0.0,
            "broad_query_rate": 0.0,
        }

    summaries = {
        "train": {
            "range": {"range_query_count": 8},
            "range_signal": {},
            "generation": {"workload_signature": signature(8)},
        },
        "eval": {
            "range": {"range_query_count": 12},
            "range_signal": {},
            "generation": {"workload_signature": signature(12)},
        },
    }

    gate = _range_workload_distribution_comparison(summaries)["workload_signature_gate"]["pairs"]["train"]

    assert gate["gate_pass"] is False
    assert gate["metrics"]["query_count_delta"] == 4
    assert gate["metrics"]["query_count_relative_delta"] == pytest.approx(4 / 12)
    assert gate["thresholds"]["query_count_relative_delta_max"] == 0.15
    assert "query_count_mismatch" in gate["failed_checks"]


def test_workload_signature_gate_allows_small_calibrated_query_count_drift() -> None:
    def signature(query_count: int) -> dict[str, Any]:
        return {
            "profile_id": "range_workload_v1",
            "query_count": query_count,
            "anchor_family_counts": {"density_route": query_count},
            "footprint_family_counts": {"medium_operational": query_count},
            "point_hit_counts_per_query": [3 for _ in range(query_count)],
            "ship_hit_counts_per_query": [1 for _ in range(query_count)],
            "near_duplicate_rate": 0.0,
            "broad_query_rate": 0.0,
        }

    summaries = {
        "train": {
            "range": {"range_query_count": 8},
            "range_signal": {},
            "generation": {"workload_signature": signature(8)},
        },
        "eval": {
            "range": {"range_query_count": 9},
            "range_signal": {},
            "generation": {"workload_signature": signature(9)},
        },
    }

    gate = _range_workload_distribution_comparison(summaries)["workload_signature_gate"]["pairs"]["train"]

    assert gate["gate_pass"] is True
    assert gate["metrics"]["query_count_delta"] == 1
    assert gate["metrics"]["query_count_relative_delta"] == pytest.approx(1 / 9)


def test_workload_signature_gate_reports_normalized_hit_distribution_diagnostics() -> None:
    def signature(total_points: int, total_trajectories: int) -> dict[str, Any]:
        return {
            "profile_id": "range_workload_v1",
            "query_count": 8,
            "total_points": total_points,
            "total_trajectories": total_trajectories,
            "anchor_family_counts": {"density_route": 8},
            "footprint_family_counts": {"medium_operational": 8},
            "point_hit_counts_per_query": [10, 20, 10, 20, 10, 20, 10, 20],
            "point_hit_fractions_per_query": [0.10, 0.20, 0.10, 0.20, 0.10, 0.20, 0.10, 0.20],
            "ship_hit_counts_per_query": [1, 2, 1, 2, 1, 2, 1, 2],
            "ship_hit_fractions_per_query": [0.10, 0.20, 0.10, 0.20, 0.10, 0.20, 0.10, 0.20],
            "near_duplicate_rate": 0.0,
            "broad_query_rate": 0.0,
        }

    summaries = {
        "train": {
            "range": {"range_query_count": 8},
            "range_signal": {},
            "generation": {"workload_signature": signature(100, 10)},
        },
        "eval": {
            "range": {"range_query_count": 8},
            "range_signal": {},
            "generation": {"workload_signature": signature(50, 5)},
        },
    }

    gate = _range_workload_distribution_comparison(summaries)["workload_signature_gate"]["pairs"]["train"]
    metrics = gate["metrics"]

    assert metrics["point_hit_fraction_distribution_ks"] == 0.0
    assert metrics["ship_hit_fraction_distribution_ks"] == 0.0
    assert metrics["train_total_points"] == 100
    assert metrics["eval_total_points"] == 50
    assert metrics["train_total_trajectories"] == 10
    assert metrics["eval_total_trajectories"] == 5


def test_workload_blind_range_v2_features_and_selector_are_query_free() -> None:
    trajectories = generate_synthetic_ais_data(n_ships=3, n_points_per_ship=40, seed=83)
    points = torch.cat(trajectories, dim=0)
    boundaries = _boundaries(trajectories)
    features = build_workload_blind_range_v2_point_features(points)
    model = WorkloadBlindRangeV2Model(
        point_dim=WORKLOAD_BLIND_RANGE_V2_POINT_DIM,
        query_dim=12,
        embed_dim=32,
        num_heads=2,
        num_layers=1,
    )

    pred, head_logits = model.forward_with_heads(features.unsqueeze(0), padding_mask=None)
    no_behavior_pred = model.final_logit_from_head_logits(
        head_logits,
        disabled_head_names=("conditional_behavior_utility",),
    )
    segment_scores = head_logits.squeeze(0)[:, 4]
    retained = simplify_with_learned_segment_budget_v1(
        pred.squeeze(0),
        boundaries,
        compression_ratio=0.10,
        segment_scores=segment_scores,
    )
    retained_with_trace, trace = simplify_with_learned_segment_budget_v1_with_trace(
        pred.squeeze(0),
        boundaries,
        compression_ratio=0.10,
        segment_scores=segment_scores,
    )
    diagnostics = learned_segment_budget_diagnostics(boundaries, (0.05, 0.10))

    assert pred.shape == (1, points.shape[0])
    assert head_logits.shape == (1, points.shape[0], len(QUERY_USEFUL_V1_HEAD_NAMES))
    assert no_behavior_pred.shape == pred.shape
    assert torch.isfinite(pred).all()
    assert retained.dtype == torch.bool
    assert torch.equal(retained, retained_with_trace)
    assert int(retained.sum().item()) > 0
    assert trace["point_attribution_available"] is True
    assert trace["skeleton_retained_count"] + trace["learned_controlled_retained_slots"] + trace[
        "fallback_retained_count"
    ] == int(retained.sum().item())
    assert trace["trajectories_with_at_least_one_learned_decision"] >= 0
    assert 0.0 <= trace["segment_budget_entropy_normalized"] <= 1.0
    assert trace["segment_score_source"] == "segment_budget_head_mean"
    assert diagnostics["selector_type"] == "learned_segment_budget_v1"
    assert diagnostics["budget_rows"][0]["no_fixed_85_percent_temporal_scaffold"] is True


def test_workload_blind_range_v2_has_dedicated_prior_feature_encoder() -> None:
    torch.manual_seed(17)
    model = WorkloadBlindRangeV2Model(
        point_dim=WORKLOAD_BLIND_RANGE_V2_POINT_DIM,
        query_dim=12,
        embed_dim=32,
        num_heads=2,
        num_layers=0,
        dropout=0.0,
    )
    base = torch.zeros((1, 4, WORKLOAD_BLIND_RANGE_V2_POINT_DIM), dtype=torch.float32)
    with_prior = base.clone()
    with_prior[..., -6:] = torch.tensor([1.0, 0.5, 0.25, 0.0, 0.75, 1.0], dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        base_score, base_heads = model.forward_with_heads(base)
        prior_score, prior_heads = model.forward_with_heads(with_prior)

    assert model.prior_feature_dim == 6
    assert isinstance(model.prior_feature_encoder[0], torch.nn.Linear)
    prior_layer = cast(torch.nn.Linear, model.prior_feature_encoder[0])
    assert tuple(prior_layer.weight.shape) == (32, 6)
    prior_output = cast(torch.nn.Linear, model.prior_feature_encoder[-1])
    assert float(prior_output.weight.detach().std(unbiased=False).item()) > 0.05
    assert abs(float(model.prior_feature_scale.detach().item()) - 0.25) < 1e-6
    assert not torch.allclose(base_score, prior_score)
    assert not torch.allclose(base_heads, prior_heads)
    assert float((base_heads - prior_heads).abs().mean().item()) > 1e-4


def test_range_v2_untrained_reset_restores_standalone_parameters() -> None:
    model = WorkloadBlindRangeV2Model(
        point_dim=WORKLOAD_BLIND_RANGE_V2_POINT_DIM,
        query_dim=12,
        embed_dim=32,
        num_heads=2,
        num_layers=0,
        dropout=0.0,
    )
    model.prior_feature_scale.data.fill_(3.0)

    reset_model = cast(WorkloadBlindRangeV2Model, _reset_module_parameters(model, seed=101))

    assert torch.allclose(reset_model.prior_feature_scale.detach(), torch.tensor(0.25))
    assert torch.allclose(model.prior_feature_scale.detach(), torch.tensor(3.0))


def test_factorized_head_ablation_uses_neutral_multiplicative_heads() -> None:
    model = WorkloadBlindRangeV2Model(
        point_dim=WORKLOAD_BLIND_RANGE_V2_POINT_DIM,
        query_dim=12,
        embed_dim=32,
        num_heads=2,
        num_layers=0,
        dropout=0.0,
    )
    head_logits = torch.zeros((1, 1, len(QUERY_USEFUL_V1_HEAD_NAMES)), dtype=torch.float32)
    head_logits[..., 1] = -10.0

    disabled = model.final_logit_from_head_logits(
        head_logits,
        disabled_head_names=("conditional_behavior_utility",),
    )
    for parameter in model.calibration_head.parameters():
        parameter.data.fill_(100.0)
    disabled_with_large_calibration = model.final_logit_from_head_logits(
        head_logits,
        disabled_head_names=("conditional_behavior_utility",),
    )
    path_idx = tuple(QUERY_USEFUL_V1_HEAD_NAMES).index("path_length_support_target")
    low_path = head_logits.clone()
    high_path = head_logits.clone()
    low_path[..., path_idx] = -10.0
    high_path[..., path_idx] = 10.0
    expected_score = torch.tensor(0.34375, dtype=torch.float32)

    assert torch.allclose(disabled.squeeze(), torch.logit(expected_score), atol=1e-6)
    assert torch.allclose(disabled, disabled_with_large_calibration)
    assert torch.allclose(model.final_logit_from_head_logits(low_path), model.final_logit_from_head_logits(high_path))
    assert all(not parameter.requires_grad for parameter in model.calibration_head.parameters())


def test_range_v2_final_score_composition_matches_query_useful_target_formula() -> None:
    model = WorkloadBlindRangeV2Model(
        point_dim=WORKLOAD_BLIND_RANGE_V2_POINT_DIM,
        query_dim=12,
        embed_dim=32,
        num_heads=2,
        num_layers=0,
        dropout=0.0,
    )
    head_probabilities = torch.tensor(
        [[[0.8, 0.4, 0.2, 0.1, 0.9, 0.1], [0.8, 0.4, 0.2, 0.9, 0.1, 0.9]]],
        dtype=torch.float32,
    )
    head_logits = torch.logit(head_probabilities.clamp(1e-4, 1.0 - 1e-4))

    final_probabilities = torch.sigmoid(model.final_logit_from_head_logits(head_logits))

    expected = (
        head_probabilities[..., 0]
        * (0.5 + head_probabilities[..., 1])
        * (0.75 + 0.25 * head_probabilities[..., 3])
        + 0.25 * head_probabilities[..., 2]
    )
    assert torch.allclose(final_probabilities, expected, atol=1e-5)
    assert final_probabilities[0, 0] > 0.5
    assert final_probabilities[0, 1] > final_probabilities[0, 0]


def test_learned_segment_budget_trace_exposes_fallback_dominance_regression() -> None:
    scores = torch.linspace(0.0, 1.0, steps=32)
    boundaries = [(0, 32)]

    retained, trace = simplify_with_learned_segment_budget_v1_with_trace(
        scores,
        boundaries,
        compression_ratio=0.20,
    )

    assert int(retained.sum().item()) == 7
    assert trace["minimal_skeleton_slot_cap"] == 1
    assert trace["skeleton_retained_count"] == 2
    assert trace["skeleton_cap_exceeded_for_endpoint_sanity"] is True
    assert bool(retained[0].item()) is True
    assert bool(retained[-1].item()) is True
    assert trace["learned_controlled_retained_slots"] == 5
    assert trace["fallback_retained_count"] == 0


def test_learned_segment_budget_uses_geometry_gain_within_learned_budget() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.25],
            [2.0, 1.0, 0.50],
            [3.0, 0.0, 0.75],
            [4.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    scores = torch.ones((5,), dtype=torch.float32)
    boundaries = [(0, 5)]

    retained = simplify_with_learned_segment_budget_v1(
        scores,
        boundaries,
        compression_ratio=0.60,
        points=points,
    )

    endpoint_only = torch.tensor([True, False, False, False, True])
    assert retained.tolist() == [True, False, True, False, True]
    assert compute_length_preservation(points, boundaries, retained) > compute_length_preservation(
        points,
        boundaries,
        endpoint_only,
    )


def test_no_geometry_tie_breaker_ablation_freezes_same_scores_without_geometry_gain() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.25],
            [2.0, 1.0, 0.50],
            [3.0, 0.0, 0.75],
            [4.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    scores = torch.ones((5,), dtype=torch.float32)
    boundaries = [(0, 5)]

    geometry_method = _learned_segment_frozen_method(
        name="MLQDS",
        scores=scores,
        boundaries=boundaries,
        compression_ratio=0.60,
        points=points,
        learned_segment_geometry_gain_weight=1.0,
    )
    no_geometry_method = _learned_segment_frozen_method(
        name="MLQDS_without_geometry_tie_breaker",
        scores=scores,
        boundaries=boundaries,
        compression_ratio=0.60,
        points=points,
        learned_segment_geometry_gain_weight=0.0,
    )

    assert geometry_method.retained_mask.tolist() == [True, False, True, False, True]
    assert no_geometry_method.retained_mask.tolist() == [True, True, False, False, True]
    assert not torch.equal(geometry_method.retained_mask, no_geometry_method.retained_mask)


def test_point_score_allocation_diagnostic_uses_point_score_segments() -> None:
    scores = torch.zeros((64,), dtype=torch.float32)
    scores[8:16] = 10.0
    scores[40:48] = 1.0
    bad_segment_scores = torch.zeros_like(scores)
    bad_segment_scores[32:64] = 10.0
    boundaries = [(0, 64)]

    point_allocation_method = _learned_segment_frozen_method(
        name="MLQDS_point_score_allocation_diagnostic",
        scores=scores,
        boundaries=boundaries,
        compression_ratio=0.125,
        segment_scores=None,
        learned_segment_geometry_gain_weight=0.0,
        learned_segment_length_repair_fraction=0.0,
    )
    bad_segment_method = _learned_segment_frozen_method(
        name="MLQDS_bad_segment_allocation",
        scores=scores,
        boundaries=boundaries,
        compression_ratio=0.125,
        segment_scores=bad_segment_scores,
        learned_segment_geometry_gain_weight=0.0,
        learned_segment_length_repair_fraction=0.0,
    )

    assert int(point_allocation_method.retained_mask[:32].sum().item()) > int(
        bad_segment_method.retained_mask[:32].sum().item()
    )
    assert int(point_allocation_method.retained_mask[32:].sum().item()) < int(
        bad_segment_method.retained_mask[32:].sum().item()
    )


def test_segment_allocation_authority_bands_coarsen_segment_scores() -> None:
    segment_scores = torch.zeros((16,), dtype=torch.float32)
    segment_scores[0:4] = 0.1
    segment_scores[4:8] = 0.2
    segment_scores[8:12] = 0.9
    segment_scores[12:16] = 0.8
    boundaries = [(0, 16)]

    top_half = _segment_score_top_band_for_ablation(
        segment_scores,
        boundaries,
        segment_size=4,
        top_fraction=0.50,
    )
    quartiles = _segment_score_quantile_bands_for_ablation(
        segment_scores,
        boundaries,
        segment_size=4,
        band_count=4,
    )

    assert top_half.tolist() == [0.0] * 8 + [1.0] * 8
    assert quartiles[0:4].unique().tolist() == [0.0]
    assert quartiles[4:8].unique().tolist() == [1.0]
    assert quartiles[8:12].unique().tolist() == [3.0]
    assert quartiles[12:16].unique().tolist() == [2.0]


def test_max_length_required_mask_keeps_required_points_and_improves_path_length() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.2],
            [2.0, 1.0, 1.0],
            [3.0, 2.0, 0.2],
            [4.0, 2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    required = torch.tensor([True, True, False, False, True])

    retained = _max_length_required_mask(points, required, keep_count=4)

    assert retained.tolist() == [True, True, True, False, True]
    assert bool(retained[1].item()) is True
    assert compute_length_preservation(points, [(0, 5)], retained) > compute_length_preservation(
        points,
        [(0, 5)],
        required,
    )


def test_score_protected_length_feasibility_reports_protected_score_upper_bound() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.2],
            [2.0, 1.0, 1.0],
            [3.0, 2.0, 0.2],
            [4.0, 2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.0, 10.0, 1.0, 0.0, 0.0], dtype=torch.float32)

    diagnostic = _score_protected_length_feasibility(
        scores=scores,
        points=points,
        boundaries=[(0, 5)],
        compression_ratio=0.80,
        learned_slot_fraction_min=0.25,
    )

    assert diagnostic["available"] is True
    assert diagnostic["diagnostic_only"] is True
    assert diagnostic["retained_count"] == 4
    assert diagnostic["protected_score_point_count"] == 1
    assert diagnostic["protected_score_point_fraction_of_budget"] == pytest.approx(0.25)
    assert diagnostic["length_preservation"] > compute_length_preservation(
        points,
        [(0, 5)],
        torch.tensor([True, True, False, False, True]),
    )


def test_score_protected_length_frontier_reports_materiality_floor() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.2],
            [2.0, 1.0, 1.0],
            [3.0, 2.0, 0.2],
            [4.0, 2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.0, 10.0, 1.0, 0.0, 0.0], dtype=torch.float32)

    frontier = _score_protected_length_frontier(
        scores=scores,
        points=points,
        boundaries=[(0, 5)],
        compression_ratio=0.80,
        learned_slot_fraction_min=0.25,
        protected_fractions=(0.0, 0.25, 0.50),
    )

    assert frontier["available"] is True
    assert frontier["diagnostic_only"] is True
    assert frontier["learned_slot_fraction_min"] == pytest.approx(0.25)
    assert len(frontier["rows"]) == 3
    assert frontier["materiality_floor_length_preservation"] == pytest.approx(
        frontier["rows"][1]["length_preservation"]
    )
    assert frontier["materiality_floor_length_gate_would_pass"] == frontier["rows"][1]["length_gate_would_pass"]
    assert frontier["rows"][0]["protected_score_point_count"] == 0
    assert frontier["rows"][1]["protected_score_point_count"] == 1


def test_learned_segment_budget_trace_reports_geometry_diagnostics_without_changing_mask() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.25],
            [2.0, 1.0, 0.50],
            [3.0, 0.0, 0.75],
            [4.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    scores = torch.ones((5,), dtype=torch.float32)
    boundaries = [(0, 5)]

    retained, trace = simplify_with_learned_segment_budget_v1_with_trace(
        scores,
        boundaries,
        compression_ratio=0.60,
        points=points,
    )
    without_trace = simplify_with_learned_segment_budget_v1(
        scores,
        boundaries,
        compression_ratio=0.60,
        points=points,
    )
    endpoint_only = torch.tensor([True, False, False, False, True])
    geometry = trace["geometry_diagnostics"]

    assert torch.equal(retained, without_trace)
    assert geometry["available"] is True
    assert geometry["trajectory_count"] == 1
    assert geometry["retained_length_preservation"] == pytest.approx(
        compute_length_preservation(points, boundaries, retained)
    )
    assert geometry["skeleton_length_preservation"] == pytest.approx(
        compute_length_preservation(points, boundaries, endpoint_only)
    )
    assert geometry["learned_length_gain_over_skeleton"] > 0.0
    assert geometry["trajectory_length_preservation_below_0_8_count"] in {0, 1}
    assert geometry["worst_trajectories"][0]["trajectory_id"] == 0


def test_learned_segment_budget_trace_separates_allocation_from_point_selection() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.2],
            [2.0, 0.0, 0.4],
            [3.0, 1.5, 0.6],
            [4.0, -1.5, 0.8],
            [5.0, 0.0, 1.0],
            [6.0, 0.0, 1.2],
            [7.0, 0.0, 1.4],
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.0, 10.0, 9.0, 0.1, 0.1, 8.0, 7.0, 0.0], dtype=torch.float32)
    boundaries = [(0, 8)]

    retained, trace = simplify_with_learned_segment_budget_v1_with_trace(
        scores,
        boundaries,
        compression_ratio=0.625,
        segment_size=8,
        points=points,
        geometry_gain_weight=0.0,
    )
    without_trace = simplify_with_learned_segment_budget_v1(
        scores,
        boundaries,
        compression_ratio=0.625,
        segment_size=8,
        points=points,
        geometry_gain_weight=0.0,
    )
    diagnostic = trace["allocation_point_selection_diagnostics"]

    assert torch.equal(retained, without_trace)
    assert diagnostic["available"] is True
    assert diagnostic["diagnostic_only"] is True
    assert diagnostic["primary_retained_stage"] == "pre_length_repair"
    assert diagnostic["primary_length_preservation"] == pytest.approx(
        compute_length_preservation(points, boundaries, retained)
    )
    assert diagnostic["same_allocation_length_only_point_selection_preservation"] > (
        diagnostic["primary_length_preservation"]
    )
    assert diagnostic["counterfactual_retained_count"] == diagnostic["total_budget_count"]


def test_learned_segment_budget_length_repair_swaps_learned_slots_for_path_gain() -> None:
    steps = torch.arange(0, 32, dtype=torch.float32)
    points = torch.stack(
        [
            steps,
            torch.where((steps.long() % 2) == 0, torch.zeros_like(steps), torch.ones_like(steps)),
            steps * 0.10,
        ],
        dim=1,
    )
    scores = torch.zeros((32,), dtype=torch.float32)
    scores[14:19] = torch.tensor([5.0, 6.0, 7.0, 6.0, 5.0])
    boundaries = [(0, 32)]

    retained, trace = simplify_with_learned_segment_budget_v1_with_trace(
        scores,
        boundaries,
        compression_ratio=0.20,
        points=points,
        geometry_gain_weight=0.0,
        length_repair_fraction=0.0,
    )
    repaired, repaired_trace = simplify_with_learned_segment_budget_v1_with_trace(
        scores,
        boundaries,
        compression_ratio=0.20,
        points=points,
        geometry_gain_weight=0.0,
        length_repair_fraction=1.0,
    )

    assert int(repaired.sum().item()) == int(retained.sum().item())
    assert repaired_trace["length_repair_swap_count"] > 0
    assert repaired_trace["length_repair_retained_count"] > 0
    assert repaired_trace["learned_controlled_retained_slots"] < trace["learned_controlled_retained_slots"]
    assert compute_length_preservation(points, boundaries, repaired) > compute_length_preservation(
        points,
        boundaries,
        retained,
    )


def test_segment_support_score_blend_uses_length_head_at_full_weight() -> None:
    segment_scores = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
    path_scores = torch.tensor([3.0, 4.0, 5.0], dtype=torch.float32)

    half = blend_segment_support_scores(
        segment_scores=segment_scores,
        path_length_support_scores=path_scores,
        path_length_support_weight=0.5,
    )
    full = blend_segment_support_scores(
        segment_scores=segment_scores,
        path_length_support_scores=path_scores,
        path_length_support_weight=1.0,
    )

    assert torch.allclose(cast(torch.Tensor, half), torch.tensor([1.5, 2.5, 3.5]))
    assert torch.allclose(cast(torch.Tensor, full), path_scores)


def test_learned_segment_budget_trace_accepts_explicit_segment_score_source_label() -> None:
    scores = torch.linspace(0.0, 1.0, steps=16)
    segment_scores = torch.linspace(1.0, 0.0, steps=16)

    _retained, trace = simplify_with_learned_segment_budget_v1_with_trace(
        scores,
        [(0, 16)],
        compression_ratio=0.25,
        segment_scores=segment_scores,
        segment_score_source_label="path_length_support_head_mean",
    )

    assert trace["segment_score_source"] == "path_length_support_head_mean"


def test_learned_segment_budget_geometry_gain_uses_trajectory_retained_anchors() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [2.0, 5.0, 2.0],
            [3.0, 0.0, 3.0],
            [4.0, 0.0, 4.0],
            [5.0, 0.0, 5.0],
            [6.0, 0.0, 6.0],
        ],
        dtype=torch.float32,
    )
    scores = torch.ones((7,), dtype=torch.float32)
    segment_scores = torch.zeros((7,), dtype=torch.float32)
    segment_scores[2:4] = 10.0
    boundaries = [(0, 7)]

    retained, trace = simplify_with_learned_segment_budget_v1_with_trace(
        scores,
        boundaries,
        compression_ratio=0.40,
        segment_size=2,
        segment_scores=segment_scores,
        points=points,
    )

    assert retained.tolist() == [True, False, True, False, False, False, True]
    assert trace["learned_controlled_retained_slots"] == 1
    assert trace["fallback_retained_count"] == 0


def test_no_segment_budget_head_ablation_uses_neutral_segment_scores() -> None:
    scores = torch.linspace(0.0, 1.0, steps=32)
    scores[27] = 10.0
    boundaries = [(0, 32)]
    learned_segment_scores = torch.zeros_like(scores)
    learned_segment_scores[24:32] = 5.0

    neutral_segment_scores = _neutral_segment_scores_for_ablation(learned_segment_scores)
    learned_retained, learned_trace = simplify_with_learned_segment_budget_v1_with_trace(
        scores,
        boundaries,
        compression_ratio=0.15,
        segment_size=8,
        segment_scores=learned_segment_scores,
    )
    ablated_retained, ablated_trace = simplify_with_learned_segment_budget_v1_with_trace(
        scores,
        boundaries,
        compression_ratio=0.15,
        segment_size=8,
        segment_scores=neutral_segment_scores,
    )

    assert torch.count_nonzero(neutral_segment_scores).item() == 0
    assert learned_trace["segment_score_source"] == "segment_budget_head_mean"
    assert ablated_trace["segment_score_source"] == "segment_budget_head_mean"
    assert bool(learned_retained[27].item()) is True
    assert bool(ablated_retained[27].item()) is False
    assert not torch.equal(learned_retained, ablated_retained)


def test_learned_segment_budget_can_split_allocation_and_point_segment_scores() -> None:
    scores = torch.zeros((8,), dtype=torch.float32)
    allocation_scores = torch.zeros_like(scores)
    allocation_scores[4:8] = 10.0
    point_segment_scores = torch.zeros_like(scores)
    point_segment_scores[5] = 10.0

    retained, trace = simplify_with_learned_segment_budget_v1_with_trace(
        scores,
        [(0, 8)],
        compression_ratio=0.375,
        segment_size=4,
        segment_scores=allocation_scores,
        segment_point_scores=point_segment_scores,
        geometry_gain_weight=0.0,
        segment_score_point_blend_weight=1.0,
        min_temporal_spacing_fraction_within_segment=0.0,
    )

    assert retained.tolist() == [True, False, False, False, False, True, False, True]
    assert trace["segment_budget_allocation_count"] == 1
    assert trace["learned_controlled_retained_slots"] == 1
    assert trace["fallback_retained_count"] == 0


def test_mlqds_scoring_passes_segment_point_scores_to_learned_selector() -> None:
    predictions = torch.zeros((64,), dtype=torch.float32)
    allocation_scores = torch.zeros_like(predictions)
    allocation_scores[32:64] = 10.0
    point_segment_scores = torch.zeros_like(predictions)
    point_segment_scores[40] = 10.0

    retained = simplify_mlqds_predictions(
        predictions,
        [(0, 64)],
        workload_type="range",
        compression_ratio=0.0625,
        temporal_fraction=0.0,
        diversity_bonus=0.0,
        selector_type="learned_segment_budget_v1",
        score_mode="raw",
        segment_scores=allocation_scores,
        segment_point_scores=point_segment_scores,
        learned_segment_geometry_gain_weight=0.0,
        learned_segment_score_blend_weight=1.0,
        learned_segment_length_repair_fraction=0.0,
    )

    assert bool(retained[40].item()) is True
    assert int(retained[:32].sum().item()) == 1
    assert int(retained[32:].sum().item()) == 3


def test_segment_oracle_allocation_audit_reports_ranking_alignment_after_freeze() -> None:
    point_scores = torch.tensor([0.9, 0.8, 0.1, 0.0, 0.1, 0.0, 0.7, 0.6], dtype=torch.float32)
    segment_scores = torch.tensor([0.9, 0.8, 0.1, 0.0, 0.95, 0.9, 0.2, 0.1], dtype=torch.float32)
    selector_scores = torch.tensor([0.9, 0.8, 0.1, 0.0, 0.95, 0.9, 0.2, 0.1], dtype=torch.float32)
    head_logits = torch.zeros((8, len(QUERY_USEFUL_V1_HEAD_NAMES)), dtype=torch.float32)
    query_hit_idx = tuple(QUERY_USEFUL_V1_HEAD_NAMES).index("query_hit_probability")
    clamped_point_scores = point_scores.clamp(1e-4, 1.0 - 1e-4)
    head_logits[:, query_hit_idx] = torch.logit(clamped_point_scores)
    labels = torch.zeros((8, 4), dtype=torch.float32)
    labels[0:2, QUERY_TYPE_ID_RANGE] = 1.0
    labels[6:8, QUERY_TYPE_ID_RANGE] = 0.5
    retained_mask = torch.tensor([True, False, False, False, False, False, True, False])
    head_sources = _factorized_head_probability_sources_from_logits(head_logits)

    audit = _segment_oracle_allocation_audit(
        point_scores=point_scores,
        segment_budget_scores=segment_scores,
        selector_segment_scores=selector_scores,
        eval_labels=labels,
        boundaries=[(0, 8)],
        workload_type="range",
        head_scores_by_name=head_sources,
        retained_mask=retained_mask,
        segment_size=2,
        paired_row_limit=2,
    )

    assert audit["available"] is True
    assert audit["diagnostic_only"] is True
    assert audit["uses_eval_labels_after_mask_freeze"] is True
    alignment = audit["source_alignment"]
    assert alignment["segment_budget_head_top20_mean"]["spearman_vs_oracle_mass"] < 1.0
    assert alignment["point_score_top20_mean"]["spearman_vs_oracle_mass"] == pytest.approx(1.0)
    assert alignment["head_query_hit_probability_sigmoid_top20_mean"]["spearman_vs_oracle_mass"] == pytest.approx(1.0)
    assert audit["best_source_by_top25_oracle_mass_recall"] == "point_score_top20_mean"
    assert "head_segment_budget_target_sigmoid_top20_mean" in audit["score_source_names"]
    transfer_rows = audit["paired_segment_transfer_rows"]
    assert transfer_rows["available"] is True
    assert transfer_rows["row_limit_per_source"] == 2
    assert transfer_rows["retained_segment_summary"]["available"] is True
    assert transfer_rows["retained_segment_summary"]["frozen_primary_retained_count_total"] == 2
    assert transfer_rows["retained_segment_summary"]["segments_with_any_frozen_primary_retained_point"] == 2
    first_row = transfer_rows["rows"][0]
    assert {
        "segment_index",
        "trajectory_id",
        "oracle_mass",
        "oracle_mass_rank",
        "point_score_top20_mean_score",
        "point_score_top20_mean_rank",
        "segment_budget_head_top20_mean_score",
        "segment_budget_head_top20_mean_rank",
        "head_query_hit_probability_sigmoid_top20_mean_score",
        "head_query_hit_probability_sigmoid_top20_mean_rank",
        "frozen_primary_retained_count",
        "frozen_primary_retained_count_rank",
    }.issubset(first_row)
    assert first_row["oracle_mass_rank"] == 1
    all_rows = audit["all_segment_transfer_rows"]
    assert all_rows["available"] is True
    assert all_rows["diagnostic_only"] is True
    assert all_rows["uses_eval_labels_after_mask_freeze"] is True
    assert all_rows["row_scope"] == "all_segments"
    assert all_rows["row_count"] == 4
    assert len(all_rows["rows"]) == 4
    assert all_rows["rows"][0]["segment_index"] == 0
    assert all_rows["rows"][0]["oracle_mass_rank"] == 1
    assert all_rows["rows"][0]["canonical_order_rank"] == 1
    assert all_rows["rows"][0]["neutral_allocation_order_rank"] == 1
    assert all_rows["rows"][3]["segment_index"] == 3
    assert all_rows["rows"][3]["oracle_mass_rank"] == 2


def test_target_segment_oracle_alignment_audit_reports_eval_target_sources_after_freeze() -> None:
    points = torch.zeros((8, 8), dtype=torch.float32)
    points[:, 0] = torch.arange(8, dtype=torch.float32)
    points[:, 1] = torch.linspace(0.0, 0.7, steps=8)
    points[:, 2] = torch.linspace(0.0, 0.7, steps=8)
    points[:, 7] = 1.0
    query = {
        "type": "range",
        "params": {
            "t_start": -0.5,
            "t_end": 3.5,
            "lat_min": -0.1,
            "lat_max": 0.35,
            "lon_min": -0.1,
            "lon_max": 0.35,
        },
    }
    labels = torch.zeros((8, 4), dtype=torch.float32)
    labels[0:4, QUERY_TYPE_ID_RANGE] = 1.0
    retained_mask = torch.tensor([True, False, True, False, False, False, False, True])

    audit = _target_segment_oracle_alignment_audit(
        points=points,
        boundaries=[(0, 8)],
        typed_queries=[query],
        eval_labels=labels,
        workload_type="range",
        retained_mask=retained_mask,
        segment_size=2,
        paired_row_limit=2,
    )

    assert audit["available"] is True
    assert audit["diagnostic_only"] is True
    assert audit["uses_eval_labels_after_mask_freeze"] is True
    assert audit["target_alignment_attempted"] is True
    assert audit["source_semantics"]["point_score_top20_mean"] == "eval_query_useful_v1_final_target_top20_mean"
    assert (
        audit["source_semantics"]["target_head_segment_budget_target_top20_mean"]
        == "eval_query_useful_v1_factorized_target_head:segment_budget_target"
    )
    assert "target_head_query_hit_probability_top20_mean" in audit["score_source_names"]
    assert "target_head_segment_budget_target_top20_mean" in audit["source_alignment"]
    rows = audit["all_segment_transfer_rows"]["rows"]
    assert len(rows) == 4
    assert "target_head_query_hit_probability_top20_mean_rank" in rows[0]
    assert audit["target_diagnostics_summary"]["segment_budget_target_base_source"] == "query_useful_v1_final_score"


def test_learned_segment_allocation_guarantees_one_slot_per_trajectory_when_possible() -> None:
    scores = torch.ones((24,), dtype=torch.float32)
    # Favor trajectory 0 strongly in segment scores and keep trajectory 1 low.
    segment_scores = torch.zeros((24,), dtype=torch.float32)
    segment_scores[0:12] = 10.0
    segment_scores[12:] = 0.1

    boundaries = [(0, 12), (12, 24)]
    retained = simplify_with_learned_segment_budget_v1(
        scores,
        boundaries,
        compression_ratio=0.30,
        segment_size=4,
        segment_scores=segment_scores,
    )
    _, trace = simplify_with_learned_segment_budget_v1_with_trace(
        scores,
        boundaries,
        compression_ratio=0.30,
        segment_size=4,
        segment_scores=segment_scores,
    )

    learned_counts = trace["trajectory_learned_decision_counts"]
    assert len(learned_counts) == 2
    assert int(learned_counts[0]) >= 1
    assert int(learned_counts[1]) >= 1
    assert bool(retained[0].item()) is True
    assert bool(retained[11].item()) is True
    assert bool(retained[12].item()) is True
    assert bool(retained[23].item()) is True
    assert trace["trajectories_with_at_least_one_learned_decision"] >= 2


def test_learned_segment_trace_reports_query_free_segment_source_attribution() -> None:
    scores = torch.linspace(0.0, 1.0, steps=24, dtype=torch.float32)
    segment_scores = torch.zeros((24,), dtype=torch.float32)
    segment_scores[8:12] = 5.0
    segment_scores[20:24] = 4.0

    retained, trace = simplify_with_learned_segment_budget_v1_with_trace(
        scores,
        [(0, 12), (12, 24)],
        compression_ratio=0.30,
        segment_size=4,
        segment_scores=segment_scores,
    )

    attribution = trace["segment_source_attribution"]
    pre_repair = trace["pre_repair_segment_source_attribution"]
    assert attribution["available"] is True
    assert attribution["diagnostic_only"] is True
    assert attribution["query_free"] is True
    assert pre_repair["available"] is True
    assert pre_repair["diagnostic_only"] is True
    assert pre_repair["query_free"] is True
    assert attribution["segment_count"] == trace["segments_considered_count"]
    summary = attribution["summary"]
    assert summary["retained_count_total"] == int(retained.sum().item())
    assert summary["skeleton_count_total"] == trace["skeleton_retained_count"]
    assert summary["learned_count_total"] == trace["learned_controlled_retained_slots"]
    assert summary["fallback_count_total"] == trace["fallback_retained_count"]
    assert summary["length_repair_count_total"] == trace["length_repair_retained_count"]
    assert summary["segment_allocation_count_total"] == trace["segment_budget_allocation_count"]
    first_row = attribution["rows"][0]
    assert {
        "segment_index",
        "allocation_order_index",
        "trajectory_id",
        "segment_score",
        "segment_score_rank",
        "segment_allocation_count",
        "retained_count",
        "skeleton_count",
        "learned_count",
        "fallback_count",
        "length_repair_count",
        "unattributed_count",
    }.issubset(first_row)


def test_learned_segment_trace_reports_pre_repair_source_attribution() -> None:
    scores = torch.zeros((32,), dtype=torch.float32)
    scores[8:24] = torch.linspace(1.0, 2.0, steps=16, dtype=torch.float32)
    points = torch.zeros((32, 5), dtype=torch.float32)
    points[:, 0] = torch.arange(32, dtype=torch.float32)
    points[:, 1] = torch.linspace(0.0, 0.1, steps=32)
    points[:, 2] = torch.sin(torch.linspace(0.0, 12.56, steps=32)) * 0.05

    _, trace = simplify_with_learned_segment_budget_v1_with_trace(
        scores,
        [(0, 32)],
        compression_ratio=0.25,
        segment_size=4,
        points=points,
        geometry_gain_weight=0.0,
        length_repair_fraction=1.0,
    )

    pre_summary = trace["pre_repair_segment_source_attribution"]["summary"]
    final_summary = trace["segment_source_attribution"]["summary"]
    pre_mask_payload = trace["pre_repair_retained_mask"]
    assert trace["length_repair_swap_count"] > 0
    assert pre_summary["retained_count_total"] == final_summary["retained_count_total"]
    assert pre_summary["length_repair_count_total"] == 0
    assert final_summary["length_repair_count_total"] == trace["length_repair_retained_count"]
    assert pre_summary["learned_count_total"] == trace["segment_budget_allocation_count"]
    assert final_summary["learned_count_total"] < pre_summary["learned_count_total"]
    assert pre_mask_payload["available"] is True
    assert pre_mask_payload["diagnostic_only"] is True
    assert pre_mask_payload["query_free"] is True
    assert pre_mask_payload["retained_count"] == pre_summary["retained_count_total"]
    assert pre_mask_payload["indices"] == sorted(set(pre_mask_payload["indices"]))

    pre_repair_method = _pre_repair_frozen_method_from_trace(
        name="MLQDS_pre_repair_allocation_diagnostic",
        selector_trace=trace,
        point_count=int(scores.numel()),
    )
    assert pre_repair_method.retained_mask.dtype == torch.bool
    assert int(pre_repair_method.retained_mask.sum().item()) == pre_summary["retained_count_total"]
    assert torch.equal(
        torch.where(pre_repair_method.retained_mask)[0],
        torch.tensor(pre_mask_payload["indices"], dtype=torch.long),
    )


def test_segment_source_attribution_uses_canonical_segment_index_after_score_sort() -> None:
    scores = torch.linspace(0.0, 1.0, steps=16, dtype=torch.float32)
    segment_scores = torch.zeros((16,), dtype=torch.float32)
    segment_scores[8:12] = 10.0
    segment_scores[0:4] = 1.0
    segment_scores[4:8] = 2.0
    segment_scores[12:16] = 3.0

    _, trace = simplify_with_learned_segment_budget_v1_with_trace(
        scores,
        [(0, 16)],
        compression_ratio=0.50,
        segment_size=4,
        segment_scores=segment_scores,
    )

    rows = trace["segment_source_attribution"]["rows"]
    by_bounds = {(row["start"], row["end"]): row for row in rows}
    assert by_bounds[(0, 4)]["segment_index"] == 0
    assert by_bounds[(4, 8)]["segment_index"] == 1
    assert by_bounds[(8, 12)]["segment_index"] == 2
    assert by_bounds[(12, 16)]["segment_index"] == 3
    assert by_bounds[(8, 12)]["allocation_order_index"] == 0
    assert by_bounds[(8, 12)]["segment_score_rank"] == 1


def test_segment_budget_head_has_segment_level_loss() -> None:
    head_targets = torch.zeros((1, 8, len(QUERY_USEFUL_V1_HEAD_NAMES)), dtype=torch.float32)
    head_mask = torch.ones_like(head_targets, dtype=torch.bool)
    segment_idx = tuple(QUERY_USEFUL_V1_HEAD_NAMES).index("segment_budget_target")
    head_targets[:, :4, segment_idx] = 1.0
    aligned = torch.zeros_like(head_targets)
    reversed_logits = torch.zeros_like(head_targets)
    aligned[:, :4, segment_idx] = 4.0
    aligned[:, 4:, segment_idx] = -4.0
    reversed_logits[:, :4, segment_idx] = -4.0
    reversed_logits[:, 4:, segment_idx] = 4.0

    aligned_loss = _segment_budget_head_segment_level_loss(
        head_logits=aligned,
        head_targets=head_targets,
        head_mask=head_mask,
        segment_size=4,
    )
    reversed_loss = _segment_budget_head_segment_level_loss(
        head_logits=reversed_logits,
        head_targets=head_targets,
        head_mask=head_mask,
        segment_size=4,
    )

    assert float(aligned_loss.item()) < float(reversed_loss.item())


def test_factorized_query_useful_loss_exposes_segment_budget_weights() -> None:
    head_targets = torch.zeros((1, 8, len(QUERY_USEFUL_V1_HEAD_NAMES)), dtype=torch.float32)
    head_mask = torch.zeros_like(head_targets, dtype=torch.bool)
    segment_idx = tuple(QUERY_USEFUL_V1_HEAD_NAMES).index("segment_budget_target")
    head_targets[:, :4, segment_idx] = 1.0
    head_mask[:, :, segment_idx] = True
    reversed_logits = torch.zeros_like(head_targets)
    reversed_logits[:, :4, segment_idx] = -4.0
    reversed_logits[:, 4:, segment_idx] = 4.0

    implicit_default = _factorized_query_useful_loss(
        head_logits=reversed_logits,
        head_targets=head_targets,
        head_mask=head_mask,
        segment_size=4,
    )
    explicit_default = _factorized_query_useful_loss(
        head_logits=reversed_logits,
        head_targets=head_targets,
        head_mask=head_mask,
        segment_size=4,
        segment_budget_head_weight=0.10,
        segment_level_loss_weight=0.25,
    )
    stronger_segment_pressure = _factorized_query_useful_loss(
        head_logits=reversed_logits,
        head_targets=head_targets,
        head_mask=head_mask,
        segment_size=4,
        segment_budget_head_weight=0.40,
        segment_level_loss_weight=1.0,
    )
    point_only = _factorized_query_useful_loss(
        head_logits=reversed_logits,
        head_targets=head_targets,
        head_mask=head_mask,
        segment_size=4,
        segment_budget_head_weight=0.10,
        segment_level_loss_weight=0.0,
    )

    assert torch.allclose(implicit_default, explicit_default)
    assert float(stronger_segment_pressure.item()) > float(implicit_default.item())
    assert float(implicit_default.item()) > float(point_only.item())


def test_factorized_head_fit_diagnostics_reports_each_head() -> None:
    head_targets = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.1, 0.0, 1.0],
            [0.2, 0.1, 0.0, 0.2, 0.2, 0.8],
            [0.4, 0.3, 0.2, 0.4, 0.4, 0.6],
            [0.6, 0.6, 0.4, 0.6, 0.6, 0.4],
            [0.8, 0.8, 0.6, 0.8, 0.8, 0.2],
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    head_logits = torch.logit(head_targets.clamp(1e-4, 1.0 - 1e-4))
    head_mask = torch.ones_like(head_targets, dtype=torch.bool)

    diagnostics = _factorized_head_fit_diagnostics(
        head_logits=head_logits,
        factorized_targets=head_targets,
        factorized_mask=head_mask,
        seed=19,
    )
    behavior = diagnostics["factorized_head_fit"]["conditional_behavior_utility"]

    assert diagnostics["factorized_head_fit_diagnostics_available"] is True
    assert set(diagnostics["factorized_head_fit"]) == set(QUERY_USEFUL_V1_HEAD_NAMES)
    assert behavior["available"] is True
    assert behavior["valid_point_count"] == 6
    assert behavior["positive_target_count"] == 5
    assert behavior["kendall_tau"] > 0.99
    assert behavior["topk_mass_recall_at_5_percent"] == 1.0
    assert diagnostics["conditional_behavior_utility_head_tau"] == behavior["kendall_tau"]


def test_factorized_final_score_composition_diagnostics_match_scalar_target() -> None:
    point_count = 8
    head_targets = torch.zeros((point_count, len(QUERY_USEFUL_V1_HEAD_NAMES)), dtype=torch.float32)
    head_targets[:, 0] = torch.linspace(0.20, 0.90, steps=point_count)
    head_targets[:, 1] = torch.linspace(0.10, 0.80, steps=point_count)
    head_targets[:, 2] = torch.linspace(0.00, 0.35, steps=point_count)
    head_targets[:, 3] = torch.linspace(0.10, 0.90, steps=point_count)
    head_targets[:, 4] = torch.linspace(0.00, 1.00, steps=point_count)
    head_targets[:, 5] = torch.linspace(1.00, 0.00, steps=point_count)
    scalar_target = (
        head_targets[:, 0]
        * (0.5 + head_targets[:, 1])
        * (0.75 + 0.25 * head_targets[:, 3])
        + 0.25 * head_targets[:, 2]
    ).clamp(0.0, 1.0)
    head_logits = torch.logit(head_targets.clamp(1e-4, 1.0 - 1e-4))

    diagnostics = _factorized_final_score_composition_diagnostics(
        head_logits=head_logits,
        factorized_targets=head_targets,
        scalar_target=scalar_target,
        scalar_mask=torch.ones((point_count,), dtype=torch.bool),
        seed=23,
    )

    assert diagnostics["factorized_final_score_composition_available"] is True
    assert diagnostics["factorized_final_score_tau"] > 0.99
    assert diagnostics["factorized_final_score_topk_mass_recall_at_5_percent"] == pytest.approx(1.0)
    assert diagnostics["factorized_final_score_prediction_std_to_target_std"] == pytest.approx(1.0, abs=1e-4)
    assert diagnostics["factorized_target_formula_label_mae"] < 1e-5
    assert diagnostics["factorized_target_formula_topk_mass_recall_at_5_percent"] == pytest.approx(1.0)
    assert diagnostics["factorized_replacement_multiplier_mean"] > 0.75


def test_factorized_scalar_training_target_keeps_raw_query_useful_scale() -> None:
    labels = torch.tensor([[0.0], [0.01], [0.02], [0.10]], dtype=torch.float32)
    labelled_mask = torch.ones_like(labels, dtype=torch.bool)

    factorized_target, factorized_basis = _scalar_training_target_for_mode(
        labels=labels,
        labelled_mask=labelled_mask,
        workload_type_id=0,
        range_training_target_mode="query_useful_v1_factorized",
    )
    legacy_target, legacy_basis = _scalar_training_target_for_mode(
        labels=labels,
        labelled_mask=labelled_mask,
        workload_type_id=0,
        range_training_target_mode="point_value",
    )

    assert factorized_basis == "raw_query_useful_v1_final_label_for_loss"
    assert legacy_basis == "scaled_training_target_for_loss"
    assert torch.allclose(factorized_target, labels[:, 0])
    assert float(legacy_target[-1].item()) == pytest.approx(1.0)
    assert not torch.allclose(legacy_target, factorized_target)


def test_factorized_head_bias_initialization_uses_training_base_rates() -> None:
    model = WorkloadBlindRangeV2Model(
        point_dim=WORKLOAD_BLIND_RANGE_V2_POINT_DIM,
        query_dim=0,
        embed_dim=16,
        num_heads=2,
        num_layers=0,
        dropout=0.0,
    )
    head_targets = torch.tensor(
        [
            [0.0010, 0.10, 0.0000, 0.20, 0.05, 0.30],
            [0.0020, 0.20, 0.0001, 0.10, 0.15, 0.10],
            [0.0000, 0.00, 0.0000, 0.00, 0.10, 0.40],
            [0.0010, 0.30, 0.0000, 0.30, 0.20, 0.20],
        ],
        dtype=torch.float32,
    )
    head_mask = torch.ones_like(head_targets, dtype=torch.bool)
    behavior_idx = tuple(QUERY_USEFUL_V1_HEAD_NAMES).index("conditional_behavior_utility")
    head_mask[2, behavior_idx] = False

    diagnostics = _initialize_factorized_head_output_biases_from_targets(
        model,
        head_targets=head_targets,
        head_mask=head_mask,
    )

    assert diagnostics["available"] is True
    for head_idx, head_name in enumerate(QUERY_USEFUL_V1_HEAD_NAMES):
        valid = head_mask[:, head_idx]
        target_mean = float(head_targets[:, head_idx][valid].mean().item())
        probability = max(1e-4, min(1.0 - 1e-4, target_mean))
        expected_bias = float(torch.logit(torch.tensor(probability)).item())
        head_module = cast(torch.nn.Sequential, model.heads[head_name])
        final_linear = cast(torch.nn.Linear, head_module[-1])

        assert diagnostics["heads"][head_name]["target_mean"] == pytest.approx(target_mean)
        assert diagnostics["heads"][head_name]["bias"] == pytest.approx(expected_bias)
        assert final_linear.bias is not None
        assert float(final_linear.bias.item()) == pytest.approx(expected_bias)


def test_factorized_training_diagnostics_do_not_claim_legacy_scalar_target() -> None:
    labels = torch.tensor([[1.0], [0.0], [0.5]], dtype=torch.float32)
    labelled_mask = torch.ones_like(labels, dtype=torch.bool)

    diagnostics = _training_target_diagnostics(
        labels=labels,
        labelled_mask=labelled_mask,
        workload_type_id=0,
        configured_budget_ratios=(0.1,),
        effective_budget_ratios=(0.1,),
        temporal_residual_budget_masks=(),
        temporal_residual_label_mode="none",
        loss_objective="budget_topk",
        temporal_fraction=0.0,
        range_training_target_mode="query_useful_v1_factorized",
    )

    assert diagnostics["target_family"] == "QueryUsefulV1Factorized"
    assert diagnostics["final_success_allowed"] is True
    assert "legacy_reason" not in diagnostics


def test_learning_causality_summary_reports_learned_slot_budget_without_ablation_claims() -> None:
    selector_diagnostics = {
        "eval": {
            "budget_rows": [
                {
                    "compression_ratio": 0.10,
                    "total_budget_count": 20,
                    "minimal_skeleton_slot_cap": 4,
                    "learned_slot_count": 16,
                    "learned_slot_fraction_of_budget": 0.80,
                    "no_fixed_85_percent_temporal_scaffold": True,
                }
            ]
        }
    }

    summary = _learned_slot_summary(selector_diagnostics, 0.10)

    assert summary["learned_controlled_retained_slots"] == 16
    assert summary["learned_controlled_retained_slot_fraction"] == 0.80
    assert summary["learned_slot_accounting_status"] == "budget_level_accounting_only"


def test_learning_causality_summary_prefers_point_attribution_when_available() -> None:
    selector_diagnostics = {
        "eval": {
            "budget_rows": [
                {
                    "compression_ratio": 0.10,
                    "total_budget_count": 20,
                    "minimal_skeleton_slot_cap": 4,
                    "learned_slot_count": 16,
                    "learned_slot_fraction_of_budget": 0.80,
                    "no_fixed_85_percent_temporal_scaffold": True,
                }
            ]
        }
    }
    trace = {
        "point_attribution_available": True,
        "learned_controlled_retained_slots": 12,
        "learned_controlled_retained_slot_fraction": 0.60,
        "skeleton_retained_count": 4,
        "fallback_retained_count": 4,
        "unattributed_retained_count": 0,
        "trajectories_with_at_least_one_learned_decision": 3,
        "trajectories_with_zero_learned_decisions": 1,
        "segment_budget_entropy": 1.2,
        "segment_budget_entropy_normalized": 0.8,
        "segments_with_learned_budget": 5,
        "retained_mask_matches_frozen_primary": True,
    }

    summary = _learned_slot_summary(selector_diagnostics, 0.10, trace)

    assert summary["learned_controlled_retained_slots"] == 12
    assert summary["planned_learned_controlled_retained_slots"] == 16
    assert summary["actual_learned_controlled_retained_slot_fraction"] == 0.60
    assert summary["trajectories_with_at_least_one_learned_decision"] == 3
    assert summary["selector_trace_retained_mask_matches_primary"] is True
    assert summary["learned_slot_accounting_status"] == "point_attribution_available"


def test_learning_causality_delta_gate_requires_material_ablation_loss() -> None:
    primary = MethodEvaluation(
        aggregate_f1=0.0,
        per_type_f1={},
        query_useful_v1_score=0.30,
    )
    uniform = MethodEvaluation(
        aggregate_f1=0.0,
        per_type_f1={},
        query_useful_v1_score=0.25,
    )

    gate = _learning_causality_delta_gate_config(primary=primary, uniform=uniform)
    thresholds = gate["thresholds"]

    assert gate["min_material_query_useful_delta"] == 0.005
    assert abs(gate["mlqds_uniform_query_useful_gap"] - 0.05) < 1e-12
    assert abs(thresholds["shuffled_scores_should_lose"] - 0.03) < 1e-12
    assert thresholds["without_segment_budget_head_should_lose"] == 0.005
    assert thresholds["prior_field_only_should_not_match_trained"] == 0.005


def test_query_useful_component_delta_summary_reports_weighted_tradeoffs() -> None:
    primary = MethodEvaluation(
        aggregate_f1=0.0,
        per_type_f1={},
        query_useful_v1_score=0.108,
        query_useful_v1_components={
            "query_balanced_point_recall": 0.60,
            "ship_f1": 0.40,
            "length_preservation_guardrail": 0.80,
        },
    )
    ablation = MethodEvaluation(
        aggregate_f1=0.0,
        per_type_f1={},
        query_useful_v1_score=0.092,
        query_useful_v1_components={
            "query_balanced_point_recall": 0.50,
            "ship_f1": 0.50,
            "length_preservation_guardrail": 0.70,
        },
    )

    summary = _query_useful_component_delta_summary(
        primary=primary,
        ablations={"MLQDS_without_behavior_utility_head": ablation},
        top_k=2,
    )
    row = summary["MLQDS_without_behavior_utility_head"]

    assert row["available"] is True
    assert row["query_useful_v1_delta"] == pytest.approx(0.016)
    assert row["component_deltas"]["query_balanced_point_recall"] == pytest.approx(0.10)
    assert row["weighted_component_deltas"]["query_balanced_point_recall"] == pytest.approx(0.010)
    assert row["weighted_component_deltas"]["ship_f1"] == pytest.approx(-0.008)
    assert row["component_weighted_delta_sum"] == pytest.approx(0.003)
    assert row["component_delta_residual"] == pytest.approx(0.013)
    assert row["top_positive_weighted_component_deltas"][0]["component"] == "query_balanced_point_recall"
    assert row["top_negative_weighted_component_deltas"][0]["component"] == "ship_f1"


def test_causality_ablation_tradeoff_summary_connects_mask_and_component_changes() -> None:
    primary = MethodEvaluation(
        aggregate_f1=0.0,
        per_type_f1={},
        query_useful_v1_score=0.108,
        query_useful_v1_components={
            "query_balanced_point_recall": 0.60,
            "ship_f1": 0.40,
            "length_preservation_guardrail": 0.80,
        },
    )
    ablation = MethodEvaluation(
        aggregate_f1=0.0,
        per_type_f1={},
        query_useful_v1_score=0.092,
        query_useful_v1_components={
            "query_balanced_point_recall": 0.50,
            "ship_f1": 0.50,
            "length_preservation_guardrail": 0.70,
        },
    )
    primary_mask = torch.tensor([True, True, False, False])
    ablation_mask = torch.tensor([False, True, True, False])

    component_deltas = _query_useful_component_delta_summary(
        primary=primary,
        ablations={"MLQDS_without_behavior_utility_head": ablation},
        top_k=2,
    )
    mask_diagnostics = {
        "MLQDS_without_behavior_utility_head": _retained_mask_comparison(
            primary_mask=primary_mask,
            ablation_mask=ablation_mask,
            expected_shape=primary_mask.shape,
        )
    }

    summary = _causality_ablation_tradeoff_summary(
        component_deltas=component_deltas,
        mask_diagnostics=mask_diagnostics,
    )
    row = summary["MLQDS_without_behavior_utility_head"]

    assert row["available"] is True
    assert row["tradeoff_status"] == "mask_change_helped_primary_metric"
    assert row["retained_symmetric_difference_count"] == 2.0
    assert row["query_useful_v1_delta_per_changed_retained_decision"] == pytest.approx(0.008)
    assert row["positive_weighted_component_delta_sum"] == pytest.approx(0.011)
    assert row["negative_weighted_component_delta_sum"] == pytest.approx(-0.008)
    assert row["dominant_positive_weighted_component_delta"]["component"] == "query_balanced_point_recall"
    assert row["dominant_negative_weighted_component_delta"]["component"] == "ship_f1"


def test_causality_ablation_diagnostics_payload_reuses_component_and_mask_tradeoffs() -> None:
    primary = MethodEvaluation(
        aggregate_f1=0.0,
        per_type_f1={},
        query_useful_v1_score=0.108,
        query_useful_v1_components={
            "query_balanced_point_recall": 0.60,
            "ship_f1": 0.40,
        },
    )
    ablation = MethodEvaluation(
        aggregate_f1=0.0,
        per_type_f1={},
        query_useful_v1_score=0.092,
        query_useful_v1_components={
            "query_balanced_point_recall": 0.50,
            "ship_f1": 0.50,
        },
    )
    mask_diagnostics = {
        "MLQDS_without_behavior_utility_head": {
            "available": True,
            "retained_symmetric_difference_count": 4,
            "retained_mask_changed": True,
            "retained_mask_jaccard": 0.5,
            "retained_mask_hamming_fraction": 0.25,
        }
    }

    payload = _causality_ablation_diagnostics_payload(
        primary=primary,
        ablations={"MLQDS_without_behavior_utility_head": ablation},
        mask_diagnostics=mask_diagnostics,
    )
    row = payload["tradeoff_diagnostics"]["MLQDS_without_behavior_utility_head"]

    assert payload["available"] is True
    assert payload["primary_query_useful_v1_score"] == pytest.approx(0.108)
    assert payload["ablation_scores"]["MLQDS_without_behavior_utility_head"] == pytest.approx(0.092)
    assert payload["ablation_query_useful_deltas"]["MLQDS_without_behavior_utility_head"] == pytest.approx(0.016)
    assert row["retained_symmetric_difference_count"] == 4.0
    assert row["dominant_negative_weighted_component_delta"]["component"] == "ship_f1"


def test_score_ablation_sensitivity_reports_score_and_mask_changes() -> None:
    primary_scores = torch.tensor([0.9, 0.8, 0.1, 0.0], dtype=torch.float32)
    ablation_scores = torch.tensor([0.1, 0.8, 0.9, 0.0], dtype=torch.float32)
    primary_mask = torch.tensor([True, True, False, False])
    ablation_mask = torch.tensor([False, True, True, False])

    diagnostics = _score_ablation_sensitivity(
        primary_scores=primary_scores,
        ablation_scores=ablation_scores,
        primary_mask=primary_mask,
        ablation_mask=ablation_mask,
    )

    assert diagnostics["available"] is True
    assert diagnostics["mean_abs_score_delta"] > 0.0
    assert diagnostics["retained_mask_changed"] is True
    assert diagnostics["retained_mask_jaccard"] == 1.0 / 3.0
    assert diagnostics["score_topk_jaccard_at_retained_count"] == 1.0 / 3.0


def test_head_ablation_sensitivity_reports_selector_raw_and_segment_channels() -> None:
    primary_scores = torch.tensor([0.9, 0.8, 0.1, 0.0], dtype=torch.float32)
    ablation_scores = torch.tensor([0.1, 0.8, 0.9, 0.0], dtype=torch.float32)
    primary_raw_predictions = torch.tensor([3.0, 2.0, 1.0, 0.0], dtype=torch.float32)
    ablation_raw_predictions = torch.tensor([2.0, 2.0, 2.0, 0.0], dtype=torch.float32)
    primary_segment_scores = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32)
    ablation_segment_scores = torch.zeros(4, dtype=torch.float32)
    primary_mask = torch.tensor([True, True, False, False])
    ablation_mask = torch.tensor([False, True, True, False])

    diagnostics = _head_ablation_sensitivity(
        primary_scores=primary_scores,
        ablation_scores=ablation_scores,
        primary_raw_predictions=primary_raw_predictions,
        ablation_raw_predictions=ablation_raw_predictions,
        primary_segment_scores=primary_segment_scores,
        ablation_segment_scores=ablation_segment_scores,
        primary_mask=primary_mask,
        ablation_mask=ablation_mask,
    )

    assert diagnostics["selector_score"]["available"] is True
    assert diagnostics["selector_score"]["retained_mask_changed"] is True
    assert diagnostics["raw_prediction"]["mean_abs_score_delta"] > 0.0
    assert diagnostics["segment_score"]["mean_abs_score_delta"] > 0.0


def test_retained_mask_comparison_reports_ablation_overlap() -> None:
    primary_mask = torch.tensor([True, True, False, False])
    ablation_mask = torch.tensor([False, True, True, False])

    diagnostics = _retained_mask_comparison(
        primary_mask=primary_mask,
        ablation_mask=ablation_mask,
        expected_shape=primary_mask.shape,
    )

    assert diagnostics["available"] is True
    assert diagnostics["primary_retained_count"] == 2
    assert diagnostics["ablation_retained_count"] == 2
    assert diagnostics["retained_intersection_count"] == 1
    assert diagnostics["retained_union_count"] == 3
    assert diagnostics["retained_symmetric_difference_count"] == 2
    assert diagnostics["retained_mask_changed"] is True
    assert diagnostics["retained_mask_jaccard"] == 1.0 / 3.0
    assert diagnostics["retained_mask_hamming_fraction"] == 0.5


def test_prior_feature_sample_sensitivity_reports_input_level_changes() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ],
        dtype=torch.float32,
    )
    query = {
        "type": "range",
        "params": {
            "t_start": -1.0,
            "t_end": 3.0,
            "lat_min": -1.0,
            "lat_max": 3.0,
            "lon_min": -1.0,
            "lon_max": 3.0,
        },
    }
    prior = build_train_query_prior_fields(
        points=points,
        boundaries=[(0, 3)],
        typed_queries=[query],
        workload_profile_id="range_workload_v1",
        grid_bins=4,
        time_bins=2,
        smoothing_passes=0,
    )

    diagnostics = _prior_feature_sample_sensitivity(
        points=points,
        primary_prior_field=prior,
        ablation_prior_field=None,
    )

    assert diagnostics["available"] is True
    assert diagnostics["point_count"] == 3
    assert diagnostics["feature_count"] == 6
    assert diagnostics["sampled_inputs_changed"] is True
    assert diagnostics["mean_abs_feature_delta"] > 0.0
    assert diagnostics["ablation_nonzero_fraction"] == 0.0
    assert diagnostics["per_feature"]["spatial_query_hit_probability"]["mean_abs_delta"] > 0.0


def test_prior_sample_gate_failures_explain_empty_or_out_of_extent_priors() -> None:
    diagnostics = {
        "shuffled_prior_fields": {
            "sampled_prior_features": {
                "available": True,
                "primary_nonzero_fraction": 0.0,
                "sampled_inputs_changed": False,
                "points_outside_prior_extent_fraction": 1.0,
            }
        }
    }

    failures = _prior_sample_gate_failures(diagnostics)

    assert "sampled_query_prior_features_all_zero" in failures
    assert "shuffled_prior_fields_did_not_change_sampled_inputs" in failures
    assert "eval_points_mostly_outside_query_prior_extent" in failures


def test_workload_signature_gate_reports_pass_for_matching_profiles() -> None:
    signature = {
        "profile_id": "range_workload_v1",
        "query_count": 8,
        "anchor_family_counts": {"density_route": 6, "boundary_entry_exit": 2},
        "footprint_family_counts": {"medium_operational": 8},
        "point_hits_per_query": {"p10": 3.0, "p50": 5.0, "p90": 8.0},
        "ship_hits_per_query": {"p10": 1.0, "p50": 2.0, "p90": 3.0},
        "near_duplicate_rate": 0.0,
        "broad_query_rate": 0.0,
    }
    summaries = {
        "train": {"range": {}, "range_signal": {}, "generation": {"workload_signature": signature}},
        "eval": {"range": {}, "range_signal": {}, "generation": {"workload_signature": dict(signature)}},
    }

    comparison = _range_workload_distribution_comparison(summaries)
    gate = comparison["workload_signature_gate"]

    assert gate["all_available"] is True
    assert gate["all_pass"] is True
    assert gate["pairs"]["train"]["gate_pass"] is True


def test_predictability_audit_is_diagnostic_only_and_reports_gate_fields() -> None:
    trajectories = generate_synthetic_ais_data(n_ships=4, n_points_per_ship=32, seed=84)
    points = torch.cat(trajectories, dim=0)
    boundaries = _boundaries(trajectories)
    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=5,
        workload_map={"range": 1.0},
        seed=13,
        workload_profile_id="range_workload_v1",
        range_max_point_hit_fraction=1.0,
        range_duplicate_iou_threshold=1.0,
    )
    targets = build_query_useful_v1_targets(
        points=points,
        boundaries=boundaries,
        typed_queries=workload.typed_queries,
    )
    prior = build_train_query_prior_fields(
        points=points,
        boundaries=boundaries,
        typed_queries=workload.typed_queries,
        labels=targets.labels,
        workload_profile_id="range_workload_v1",
        train_workload_seed=13,
    )

    audit = query_prior_predictability_audit(
        points=points,
        boundaries=boundaries,
        eval_typed_queries=workload.typed_queries,
        query_prior_field=prior,
    )

    assert audit["available"] is True
    assert audit["used_for_training"] is False
    assert audit["used_for_checkpoint_selection"] is False
    assert audit["used_for_retained_mask_decision"] is False
    assert "spearman" in audit["metrics"]
    assert audit["metrics"]["rank_correlation_method"] == "average_tie_ranks"
    assert "score_decile_calibration" in audit["metrics"]
    assert "lift_at_5_percent" in audit["metrics"]
    assert "lift_at_5_percent" in audit["gate_checks"]
    channel_by_head = audit["prior_channel_by_head_predictability"]
    assert "query_hit_probability" in channel_by_head
    assert "spatiotemporal_query_hit_probability" in channel_by_head["query_hit_probability"]
    assert "lift_at_5_percent" in channel_by_head["query_hit_probability"]["spatiotemporal_query_hit_probability"]
    assert (
        channel_by_head["query_hit_probability"]["spatiotemporal_query_hit_probability"]["rank_correlation_method"]
        == "average_tie_ranks"
    )
    assert "score_decile_calibration" in channel_by_head["query_hit_probability"]["spatiotemporal_query_hit_probability"]
    best_by_head = audit["best_prior_channel_by_head"]
    assert "query_hit_probability" in best_by_head
    assert "channel" in best_by_head["query_hit_probability"]["best_lift_at_5_percent"]


def test_predictability_rankdata_uses_average_tie_ranks() -> None:
    values = torch.tensor([2.0, 2.0, 5.0, 5.0, 9.0])

    ranks = _rankdata(values)

    assert torch.allclose(ranks, torch.tensor([0.5, 0.5, 2.5, 2.5, 4.0]))


def test_segment_budget_prior_does_not_blend_raw_route_density() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    prior = {
        "grid_bins": 1,
        "time_bins": 1,
        "extent": {
            "t_min": 0.0,
            "t_max": 1.0,
            "lat_min": 0.0,
            "lat_max": 1.0,
            "lon_min": 0.0,
            "lon_max": 1.0,
        },
        "out_of_extent_sampling": "nearest",
        "spatial_query_hit_probability": torch.tensor([0.20]),
        "spatiotemporal_query_hit_probability": torch.tensor([[0.10]]),
        "boundary_entry_exit_likelihood": torch.tensor([0.30]),
        "crossing_likelihood": torch.tensor([0.40]),
        "behavior_utility_prior": torch.tensor([0.80]),
        "route_density_prior": torch.tensor([1.00]),
    }

    scores = _prior_channel_scores(points, prior)

    assert torch.allclose(scores["segment_budget_prior"], scores["replacement_representative_prior"])
    assert not torch.allclose(scores["segment_budget_prior"], scores["route_density_prior"])


def test_query_prior_predictability_score_gates_behavior_utility_by_query_mass() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    prior = {
        "grid_bins": 1,
        "time_bins": 1,
        "extent": {
            "t_min": 0.0,
            "t_max": 1.0,
            "lat_min": 0.0,
            "lat_max": 1.0,
            "lon_min": 0.0,
            "lon_max": 1.0,
        },
        "out_of_extent_sampling": "nearest",
        "spatial_query_hit_probability": torch.tensor([0.01]),
        "spatiotemporal_query_hit_probability": torch.tensor([[0.01]]),
        "boundary_entry_exit_likelihood": torch.tensor([0.0]),
        "crossing_likelihood": torch.tensor([0.0]),
        "behavior_utility_prior": torch.tensor([0.90]),
        "route_density_prior": torch.tensor([0.0]),
    }

    score = query_prior_predictability_scores(points, prior)

    assert torch.allclose(score.cpu(), torch.full((2,), 0.014))


def test_route_corridor_family_has_actual_corridor_semantics_or_is_not_final() -> None:
    profile = range_workload_profile("range_workload_v1")
    assert profile.final_success_allowed is True
    assert profile.footprint_families["route_corridor_like"]["elongation_allowed"] is True
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0, 90.0],
            [1.0, 0.0, 1.0, 1.0, 90.0],
            [2.0, 0.0, 2.0, 1.0, 90.0],
        ],
        dtype=torch.float32,
    )
    bounds = {
        "t_min": 0.0,
        "t_max": 2.0,
        "lat_min": -5.0,
        "lat_max": 5.0,
        "lon_min": -5.0,
        "lon_max": 5.0,
    }
    query = _make_range_query(
        points,
        bounds,
        torch.Generator().manual_seed(3),
        range_spatial_km=10.0,
        range_time_hours=1.0,
        range_footprint_jitter=0.0,
        elongation_allowed=True,
        metadata={"footprint_family": "route_corridor_like"},
    )
    params = query["params"]
    metadata = query["_metadata"]
    assert metadata["corridor_axis"] == "east_west"
    assert float(params["lon_max"] - params["lon_min"]) > float(params["lat_max"] - params["lat_min"])


def test_port_or_approach_zone_anchor_family_is_distinct_from_density_route() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.1, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.1, 0.1, 4.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 0.2, 0.2, 5.0, 0.0, 0.0, 0.0, 0.0],
            [3.0, 1.0, 1.0, 0.2, 0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    density, _density_prob = _anchor_weights_for_family(points, "density_route")
    port, _port_prob = _anchor_weights_for_family(points, "port_or_approach_zone")

    assert density is not None
    assert port is not None
    assert not torch.allclose(density, port)
    assert float(port[0].item() + port[-1].item()) > float(density[0].item() + density[-1].item())


def test_final_profile_does_not_chase_uncovered_points_unless_declared() -> None:
    trajectories = generate_synthetic_ais_data(n_ships=4, n_points_per_ship=32, seed=91, route_families=1)
    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=4,
        workload_map={"range": 1.0},
        seed=22,
        target_coverage=0.30,
        max_queries=8,
        workload_profile_id="range_workload_v1",
        range_max_point_hit_fraction=1.0,
        range_duplicate_iou_threshold=1.0,
    )
    generation = (workload.generation_diagnostics or {})["query_generation"]
    legacy = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=4,
        workload_map={"range": 1.0},
        seed=22,
        target_coverage=0.30,
        max_queries=8,
        workload_profile_id="range_workload_v1",
        coverage_calibration_mode="uncovered_anchor_chasing",
        range_max_point_hit_fraction=1.0,
        range_duplicate_iou_threshold=1.0,
    )
    legacy_generation = (legacy.generation_diagnostics or {})["query_generation"]

    assert generation["coverage_calibration_mode"] == "profile_sampled_query_count"
    assert legacy_generation["coverage_calibration_mode"] == "uncovered_anchor_chasing"


def test_workload_stability_gate_rejects_tiny_fixed_count_workloads() -> None:
    config = SimpleNamespace(
        query=SimpleNamespace(target_coverage=None, range_max_coverage_overshoot=None)
    )
    workload = SimpleNamespace(
        typed_queries=[{} for _ in range(4)],
        coverage_fraction=0.25,
        generation_diagnostics={
            "query_generation": {
                "mode": "fixed_count",
                "workload_profile_id": "range_workload_v1",
                "coverage_calibration_mode": "profile_sampled_query_count",
                "coverage_guard_enabled": False,
                "stop_reason": "fixed_count_completed",
            }
        },
    )

    gate = _workload_stability_gate(
        config=cast(Any, config),
        train_label_workloads=[workload],
        eval_workload=workload,
        selection_workload=None,
    )

    assert gate["gate_pass"] is False
    assert "coverage_target_not_in_final_grid" in gate["failed_checks"]
    assert "too_few_train_workload_replicates" in gate["failed_checks"]
    assert "train_r0:not_target_coverage_generation" in gate["failed_checks"]
    assert "eval:too_few_queries" in gate["failed_checks"]


def test_workload_stability_gate_accepts_coverage_calibrated_replicates() -> None:
    config = SimpleNamespace(
        query=SimpleNamespace(target_coverage=0.10, range_max_coverage_overshoot=0.0075)
    )

    def workload() -> SimpleNamespace:
        return SimpleNamespace(
            typed_queries=[{} for _ in range(8)],
            coverage_fraction=0.105,
            generation_diagnostics={
            "query_generation": {
                "mode": "target_coverage",
                "workload_profile_id": "range_workload_v1",
                "coverage_calibration_mode": "profile_sampled_query_count",
                "query_count_mode": "calibrated_to_coverage",
                "target_coverage": 0.10,
                "coverage_guard_enabled": True,
                "stop_reason": "target_coverage_reached",
            }
        },
        )

    gate = _workload_stability_gate(
        config=cast(Any, config),
        train_label_workloads=[workload(), workload(), workload(), workload()],
        eval_workload=workload(),
        selection_workload=None,
    )

    assert gate["gate_pass"] is True
    assert gate["failed_checks"] == []
    assert gate["train_workload_replicate_count"] == 4


def test_workload_stability_gate_rejects_exhausted_generation_after_coverage_satisfied() -> None:
    config = SimpleNamespace(
        query=SimpleNamespace(
            target_coverage=0.10,
            range_max_coverage_overshoot=0.0075,
            workload_stability_gate_mode="final",
        )
    )
    workload = SimpleNamespace(
        typed_queries=[{} for _ in range(12)],
        coverage_fraction=0.105,
        generation_diagnostics={
            "query_generation": {
                "mode": "target_coverage",
                "workload_profile_id": "range_workload_v1",
                "coverage_calibration_mode": "profile_sampled_query_count",
                "query_count_mode": "calibrated_to_coverage",
                "target_coverage": 0.10,
                "coverage_guard_enabled": True,
                "stop_reason": "range_acceptance_exhausted",
            }
        ,
            "range_acceptance": {
                "enabled": True,
                "attempts": 6000,
                "accepted": 12,
                "rejected": 5988,
                "exhausted": True,
            },
        },
    )

    gate = _workload_stability_gate(
        config=cast(Any, config),
        train_label_workloads=[workload, workload, workload, workload],
        eval_workload=workload,
        selection_workload=None,
    )

    assert gate["gate_pass"] is False
    assert "train_r0:range_acceptance_or_coverage_guard_exhausted" in gate["failed_checks"]
    assert "eval:range_acceptance_or_coverage_guard_exhausted" in gate["failed_checks"]
    assert gate["workloads"][0]["coverage_target_satisfied"] is True


def test_workload_stability_gate_rejects_calibrated_low_query_count_in_final_mode() -> None:
    config = SimpleNamespace(
        query=SimpleNamespace(
            target_coverage=0.05,
            range_max_coverage_overshoot=0.005,
            workload_stability_gate_mode="final",
        )
    )
    workload = SimpleNamespace(
        typed_queries=[{} for _ in range(7)],
        coverage_fraction=0.054,
        generation_diagnostics={
            "query_generation": {
                "mode": "target_coverage",
                "workload_profile_id": "range_workload_v1",
                "coverage_calibration_mode": "profile_sampled_query_count",
                "query_count_mode": "calibrated_to_coverage",
                "target_coverage": 0.05,
                "coverage_guard_enabled": True,
                "stop_reason": "target_coverage_reached",
            }
        },
    )

    gate = _workload_stability_gate(
        config=cast(Any, config),
        train_label_workloads=[workload, workload, workload, workload],
        eval_workload=workload,
        selection_workload=None,
    )

    assert gate["gate_pass"] is False
    assert "train_r0:too_few_queries" in gate["failed_checks"]
    assert "eval:too_few_queries" in gate["failed_checks"]


def test_workload_stability_gate_smoke_mode_allows_calibrated_low_query_count() -> None:
    config = SimpleNamespace(
        query=SimpleNamespace(
            target_coverage=0.05,
            range_max_coverage_overshoot=0.005,
            workload_stability_gate_mode="smoke",
        )
    )
    workload = SimpleNamespace(
        typed_queries=[{} for _ in range(7)],
        coverage_fraction=0.054,
        generation_diagnostics={
            "query_generation": {
                "mode": "target_coverage",
                "workload_profile_id": "range_workload_v1",
                "coverage_calibration_mode": "profile_sampled_query_count",
                "query_count_mode": "calibrated_to_coverage",
                "target_coverage": 0.05,
                "coverage_guard_enabled": True,
                "stop_reason": "target_coverage_reached",
            }
        },
    )

    gate = _workload_stability_gate(
        config=cast(Any, config),
        train_label_workloads=[workload, workload, workload, workload],
        eval_workload=workload,
        selection_workload=None,
    )

    assert gate["gate_pass"] is True
    assert gate["failed_checks"] == []
    assert gate["gate_mode"] == "smoke"


def test_global_sanity_gate_enforces_endpoint_length_and_sed_ratio() -> None:
    primary = MethodEvaluation(
        aggregate_f1=0.0,
        per_type_f1={},
        avg_length_preserved=0.90,
        geometric_distortion={"avg_sed_km": 0.90},
        range_audit={"endpoint_sanity": 1.0},
    )
    uniform = MethodEvaluation(
        aggregate_f1=0.0,
        per_type_f1={},
        avg_length_preserved=0.95,
        geometric_distortion={"avg_sed_km": 0.60},
        range_audit={"endpoint_sanity": 1.0},
    )

    gate = _global_sanity_gate(primary=primary, uniform=uniform, compression_ratio=0.05)

    assert gate["gate_pass"] is True
    assert gate["avg_sed_ratio_vs_uniform"] == 1.5
    assert gate["catastrophic_geometry_outlier_status"] == "not_available_report_only"

    primary.avg_length_preserved = 0.70
    primary.range_audit["endpoint_sanity"] = 0.5
    primary.geometric_distortion["avg_sed_km"] = 1.20
    gate = _global_sanity_gate(primary=primary, uniform=uniform, compression_ratio=0.05)

    assert gate["gate_pass"] is False
    assert "length_preservation_outside_range" in gate["failed_checks"]
    assert "endpoints_not_retained_for_all_eligible_trajectories" in gate["failed_checks"]
    assert "avg_sed_ratio_vs_uniform_too_high" in gate["failed_checks"]
