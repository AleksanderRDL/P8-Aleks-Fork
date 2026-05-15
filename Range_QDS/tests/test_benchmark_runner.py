"""Tests for range benchmark run helpers."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

import experiments.benchmark_runner as benchmark_runner
from experiments.benchmark_profiles import (
    BLIND_EXPECTED_USEFULNESS_PROFILE,
    BLIND_RETAINED_FREQUENCY_PROFILE,
    BLIND_TEACHER_DISTILL_PROFILE,
    RANGE_BLIND_COVERAGE_MIN_QUERY_FLOOR,
    benchmark_profile_args,
    benchmark_profile_settings,
)
from experiments.benchmark_runner import (
    DEFAULT_WORKLOADS,
    DEFAULT_PROFILE,
    BenchmarkDataSources,
    PURE_WORKLOADS,
    _child_run_dir,
    _coverage_label_suffix,
    _format_report_table,
    _parse_coverage_targets,
    _parse_name_list,
    _profile_args,
    _resolve_data_sources,
    _row_from_run,
    _run_config,
    _run_capture_streaming,
    _runner_environment_metadata,
)
from experiments.benchmark_process import BenchmarkChildResult
from experiments.benchmark_report import (
    MIN_MATCHED_LEARNED_SLOT_FRACTION_FOR_BLIND_CLAIM,
    _query_floor_fields,
    query_driven_final_grid_summary,
)
from experiments.benchmark_artifacts import index_entry, write_family_indexes
from experiments.experiment_config import build_experiment_config
from experiments.experiment_workloads import resolve_workload_maps, validation_query_count


def _profile_core_args() -> list[str]:
    """Expected workload-aware diagnostic child args without data-source/cap flags."""
    return [
        "--n_queries",
        "80",
        "--query_coverage",
        "0.20",
        "--range_spatial_fraction",
        "0.0165",
        "--range_time_fraction",
        "0.033",
        "--range_spatial_km",
        "2.2",
        "--range_time_hours",
        "5.0",
        "--range_footprint_jitter",
        "0.0",
        "--range_max_coverage_overshoot",
        "0.02",
        "--range_time_domain_mode",
        "anchor_day",
        "--range_anchor_mode",
        "mixed_density",
        "--range_diagnostics_mode",
        "cached",
        "--final_metrics_mode",
        "diagnostic",
        "--float32_matmul_precision",
        "high",
        "--allow_tf32",
        "--amp_mode",
        "bf16",
        "--query_chunk_size",
        "2048",
        "--train_batch_size",
        "64",
        "--inference_batch_size",
        "64",
        "--model_type",
        "range_aware",
        "--max_queries",
        "2048",
        "--compression_ratio",
        "0.05",
        "--epochs",
        "8",
        "--early_stopping_patience",
        "5",
        "--checkpoint_smoothing_window",
        "1",
        "--checkpoint_full_score_every",
        "4",
        "--checkpoint_candidate_pool_size",
        "2",
        "--loss_objective",
        "budget_topk",
        "--budget_loss_ratios",
        "0.05,0.10",
        "--range_audit_compression_ratios",
        "0.01,0.02,0.05,0.10,0.15,0.20,0.30",
        "--budget_loss_temperature",
        "0.25",
        "--temporal_distribution_loss_weight",
        "0.000",
        "--mlqds_temporal_fraction",
        "0.25",
        "--mlqds_score_mode",
        "rank",
        "--mlqds_score_temperature",
        "1.00",
        "--mlqds_rank_confidence_weight",
        "0.15",
        "--mlqds_range_geometry_blend",
        "0.00",
        "--mlqds_diversity_bonus",
        "0.00",
        "--mlqds_hybrid_mode",
        "fill",
        "--selector_type",
        "temporal_hybrid",
        "--mlqds_stratified_center_weight",
        "0.00",
        "--temporal_residual_label_mode",
        "none",
        "--range_label_mode",
        "usefulness",
        "--range_training_target_mode",
        "point_value",
        "--range_temporal_target_blend",
        "0.000",
        "--range_target_budget_weight_power",
        "0.00",
        "--range_marginal_target_radius_scale",
        "0.50",
        "--range_query_spine_fraction",
        "0.10",
        "--range_query_spine_mass_mode",
        "hit_group",
        "--range_query_residual_multiplier",
        "1.00",
        "--range_query_residual_mass_mode",
        "query",
        "--range_set_utility_multiplier",
        "1.00",
        "--range_set_utility_candidate_limit",
        "128",
        "--range_set_utility_mass_mode",
        "gain",
        "--range_boundary_prior_weight",
        "0.0",
        "--range_teacher_distillation_mode",
        "none",
        "--range_teacher_epochs",
        "4",
        "--checkpoint_selection_metric",
        "uniform_gap",
        "--checkpoint_score_variant",
        "range_usefulness",
    ]


def test_benchmark_name_list_rejects_mixed_workloads() -> None:
    assert _parse_name_list("range", allowed=PURE_WORKLOADS, arg_name="--workloads") == ["range"]
    assert _parse_name_list(None, allowed=DEFAULT_WORKLOADS, arg_name="--workloads") == ["range"]

    with pytest.raises(ValueError, match="unknown"):
        _parse_name_list("range,legacy", allowed=PURE_WORKLOADS, arg_name="--workloads")


def test_benchmark_coverage_target_parser_accepts_fractions_and_percents() -> None:
    assert _parse_coverage_targets("0.05,10,0.3") == [0.05, 0.10, 0.30]
    assert _coverage_label_suffix(0.05) == "c05"
    assert _coverage_label_suffix(0.30) == "c30"

    with pytest.raises(ValueError, match="coverage targets"):
        _parse_coverage_targets("101")


def test_experiment_workload_resolution_is_pure_only() -> None:
    assert resolve_workload_maps("range") == ({"range": 1.0}, {"range": 1.0})

    with pytest.raises(ValueError, match="no longer supported"):
        resolve_workload_maps("legacy")


def test_profile_args_own_runtime_and_checkpoint_defaults() -> None:
    args = _profile_core_args()

    assert args[args.index("--float32_matmul_precision") + 1] == "high"
    assert "--allow_tf32" in args
    assert args[args.index("--amp_mode") + 1] == "bf16"


def test_benchmark_environment_metadata_is_scoped_to_parent_process() -> None:
    environment = _runner_environment_metadata()

    assert environment["scope"] == "benchmark_runner_parent_process"
    assert "rows[*].child_torch_runtime" in environment["note"]


def test_run_config_records_profile_checkpoint_selection_metric(tmp_path) -> None:
    args = argparse.Namespace(
        profile=DEFAULT_PROFILE,
        seed=42,
        cache_dir=None,
        refresh_cache=False,
        no_cache_warmup=False,
        min_points_per_segment=4,
        max_points_per_segment=None,
        max_time_gap_seconds=3600.0,
        max_segments=None,
        max_trajectories=None,
        validation_score_every=1,
        extra_args=None,
        coverage_targets="0.05,0.30",
        continue_on_failure=False,
    )
    data_sources = BenchmarkDataSources(
        train_csv_path="train.csv",
        validation_csv_path="validation.csv",
        eval_csv_path="eval.csv",
        selected_cleaned_csv_files=("train.csv", "validation.csv", "eval.csv"),
    )

    payload = _run_config(
        args=args,
        run_id="run",
        workloads=["range"],
        run_label="label",
        data_sources=data_sources,
        results_dir=tmp_path,
        extra_args=[],
    )

    assert payload["checkpoint_selection_metric"] == "uniform_gap"
    assert payload["coverage_targets"] == [0.05, 0.30]
    assert payload["profile_settings"]["profile_role"] == "workload_aware_diagnostic"
    assert payload["profile_settings"]["final_product_claim"] is False
    assert payload["profile_settings"]["workload_blind"] is False
    assert payload["profile_settings"]["mlqds_effective_diversity_bonus"] == 0.0
    assert payload["profile_settings"]["range_coverage_sweep_targets"] == [0.05, 0.10, 0.15, 0.30]
    assert payload["profile_settings"]["range_compression_sweep_ratios"] == [
        0.01,
        0.02,
        0.05,
        0.10,
        0.15,
        0.20,
        0.30,
    ]


def test_blind_profiles_use_small_query_floor_for_coverage_control() -> None:
    """Blind profiles should not force dense duplicate workloads after target coverage is met."""
    for profile in (
        BLIND_EXPECTED_USEFULNESS_PROFILE,
        BLIND_RETAINED_FREQUENCY_PROFILE,
        BLIND_TEACHER_DISTILL_PROFILE,
    ):
        args = benchmark_profile_args(profile)
        settings = benchmark_profile_settings(profile)

        assert args[args.index("--n_queries") + 1] == str(RANGE_BLIND_COVERAGE_MIN_QUERY_FLOOR)
        assert settings["n_queries"] == RANGE_BLIND_COVERAGE_MIN_QUERY_FLOOR
        assert settings["workload_blind"] is True
        assert settings["profile_legacy_diagnostic"] is True
        assert settings["final_success_allowed"] is False
        assert settings["final_product_candidate"] is False
        assert settings["final_product_claim"] is False
        assert "QueryUsefulV1" in str(settings["final_product_claim_gate"])


def test_benchmark_row_records_effective_child_torch_runtime(tmp_path) -> None:
    run_json = {
        "config": {
            "query": {
                "n_queries": 24,
                "target_coverage": 0.30,
                "max_queries": 512,
                "range_spatial_km": 2.2,
                "range_time_hours": 5.0,
                "range_train_workload_replicates": 3,
                "range_time_domain_mode": "anchor_day",
                "range_anchor_mode": "sparse",
                "range_train_anchor_modes": ["mixed_density", "sparse"],
                "range_train_footprints": ["1.1:2.5", "2.2:5.0"],
                "range_max_coverage_overshoot": 0.02,
            },
            "model": {
                "model_type": "range_aware",
                "historical_prior_k": 7,
                "historical_prior_mmsi_weight": 2.5,
                "historical_prior_source_aggregation": "mean",
                "mlqds_temporal_fraction": 0.25,
                "mlqds_diversity_bonus": 0.05,
                "mlqds_hybrid_mode": "swap",
                "mlqds_stratified_center_weight": 0.45,
                "mlqds_min_learned_swaps": 1,
                "mlqds_score_mode": "rank",
                "mlqds_score_temperature": 1.0,
                "mlqds_rank_confidence_weight": 0.15,
                "mlqds_range_geometry_blend": 0.25,
                "temporal_residual_label_mode": "temporal",
                "range_label_mode": "usefulness",
                "range_replicate_target_aggregation": "label_mean",
                "range_component_target_blend": 0.75,
                "range_temporal_target_blend": 0.15,
                "range_target_budget_weight_power": 0.50,
                "range_marginal_target_radius_scale": 0.65,
                "range_query_spine_fraction": 0.30,
                "range_query_spine_mass_mode": "query",
                "range_query_residual_multiplier": 1.25,
                "range_query_residual_mass_mode": "point",
                "range_set_utility_multiplier": 1.75,
                "range_set_utility_candidate_limit": 64,
                "range_set_utility_mass_mode": "query",
                "range_boundary_prior_weight": 0.0,
                "loss_objective": "budget_topk",
                "budget_loss_ratios": [0.01, 0.02, 0.05, 0.10],
                "budget_loss_temperature": 0.10,
                "temporal_distribution_loss_weight": 0.05,
                "checkpoint_full_score_every": 3,
                "checkpoint_candidate_pool_size": 2,
                "checkpoint_score_variant": "range_usefulness",
                "float32_matmul_precision": "high",
                "allow_tf32": True,
                "amp_mode": "bf16",
                "compression_ratio": 0.05,
            }
        },
        "oracle_diagnostic": {
            "kind": "additive_label_greedy",
            "exact_optimum": False,
        },
        "workload_blind_protocol": {
            "enabled": False,
            "primary_masks_frozen_before_eval_query_scoring": False,
            "audit_masks_frozen_before_eval_query_scoring": False,
            "eval_geometry_blend_allowed": True,
        },
        "final_claim_summary": {
            "primary_metric": "QueryUsefulV1",
            "status": "candidate_blocked_by_required_gates",
            "final_success_allowed": False,
            "blocking_gates": ["predictability_gate"],
        },
        "predictability_audit": {
            "gate_pass": False,
            "metrics": {
                "spearman": 0.12,
                "kendall_tau": 0.08,
                "lift_at_1_percent": 1.05,
                "lift_at_2_percent": 1.10,
                "lift_at_5_percent": 1.18,
                "pr_auc_lift_over_base_rate": 1.20,
            },
        },
        "workload_stability_gate": {
            "gate_pass": False,
            "failed_checks": ["train_r0:not_target_coverage_generation"],
            "train_workload_replicate_count": 1,
            "configured_target_coverage": 0.10,
        },
        "support_overlap_gate": {
            "gate_pass": True,
            "failed_checks": [],
            "eval_points_outside_train_prior_extent_fraction": 0.02,
            "sampled_prior_nonzero_fraction": 0.80,
            "primary_sampled_prior_nonzero_fraction": 0.60,
            "route_density_overlap": 0.70,
            "query_prior_support_overlap": 0.65,
            "train_eval_spatial_extent_intersection_fraction": 0.90,
        },
        "global_sanity_gate": {
            "gate_pass": True,
            "failed_checks": [],
            "endpoint_sanity": 1.0,
            "avg_sed_ratio_vs_uniform": 1.09,
            "avg_sed_ratio_vs_uniform_max": 1.50,
            "avg_length_preserved": 0.88,
        },
        "learning_causality_summary": {
            "learning_causality_ablation_status": "partial",
            "learning_causality_gate_pass": False,
            "learning_causality_failed_checks": ["shuffled_scores_should_lose"],
            "causality_ablation_missing": ["MLQDS_without_segment_budget_head"],
            "learned_controlled_retained_slot_fraction": 0.72,
            "planned_learned_controlled_retained_slot_fraction": 0.80,
            "actual_learned_controlled_retained_slot_fraction": 0.72,
            "trajectories_with_at_least_one_learned_decision": 5,
            "trajectories_with_zero_learned_decisions": 1,
            "segment_budget_entropy": 1.4,
            "segment_budget_entropy_normalized": 0.7,
            "selector_trace_retained_mask_matches_primary": True,
            "shuffled_score_ablation_delta": 0.04,
            "untrained_score_ablation_delta": 0.06,
            "shuffled_prior_field_ablation_delta": 0.05,
            "prior_field_only_score_ablation_delta": 0.02,
            "no_query_prior_field_ablation_delta": 0.03,
            "no_behavior_head_ablation_delta": 0.07,
            "no_segment_budget_head_ablation_delta": 0.08,
            "learning_causality_delta_gate": {
                "min_material_query_useful_delta": 0.005,
                "shuffled_score_delta_fraction_of_uniform_gap_min": 0.60,
                "mlqds_uniform_query_useful_gap": 0.05,
                "thresholds": {
                    "shuffled_scores_should_lose": 0.03,
                    "untrained_model_should_lose": 0.005,
                    "without_segment_budget_head_should_lose": 0.005,
                },
            },
            "segment_budget_head_ablation_mode": "neutral_constant_segment_scores",
            "prior_sample_gate_pass": False,
            "prior_sample_gate_failures": ["sampled_query_prior_features_all_zero"],
            "prior_sensitivity_diagnostics": {
                "shuffled_prior_fields": {
                    "sampled_prior_features": {
                        "sampled_inputs_changed": False,
                        "primary_nonzero_fraction": 0.0,
                        "ablation_nonzero_fraction": 0.01,
                        "mean_abs_feature_delta": 0.002,
                        "max_abs_feature_delta": 0.10,
                        "points_outside_prior_extent_fraction": 0.75,
                    }
                },
                "without_query_prior_features": {
                    "sampled_prior_features": {
                        "primary_nonzero_fraction": 0.0,
                        "mean_abs_feature_delta": 0.0,
                        "points_outside_prior_extent_fraction": 0.75,
                    }
                },
            },
        },
        "query_generation_diagnostics": {
            "train": {
                "query_generation": {
                    "mode": "target_coverage",
                    "target_coverage": 0.20,
                    "final_coverage": 0.21,
                    "minimum_queries": 8,
                    "max_queries": 2048,
                    "final_query_count": 28,
                    "target_reached_query_count": 24,
                    "coverage_at_target_reached": 0.20,
                    "extra_queries_after_target_reached": 4,
                    "coverage_guard_enabled": True,
                    "max_allowed_coverage": 0.22,
                    "stop_reason": "target_coverage_reached",
                },
            },
            "eval": {
                "query_generation": {
                    "mode": "target_coverage",
                    "target_coverage": 0.20,
                    "final_coverage": 0.215,
                    "minimum_queries": 160,
                    "max_queries": 2048,
                    "final_query_count": 160,
                    "target_reached_query_count": 31,
                    "coverage_at_target_reached": 0.201,
                    "extra_queries_after_target_reached": 129,
                    "coverage_guard_enabled": True,
                    "max_allowed_coverage": 0.22,
                    "stop_reason": "target_coverage_reached",
                },
            },
            "selection": {
                "query_generation": {
                    "mode": "target_coverage",
                    "target_coverage": 0.20,
                    "final_coverage": 0.205,
                    "minimum_queries": 8,
                    "max_queries": 2048,
                    "final_query_count": 21,
                    "target_reached_query_count": 21,
                    "coverage_at_target_reached": 0.205,
                    "extra_queries_after_target_reached": 0,
                    "coverage_guard_enabled": True,
                    "max_allowed_coverage": 0.22,
                    "stop_reason": "target_coverage_reached",
                },
            },
        },
        "teacher_distillation": {
            "enabled": True,
            "mode": "retained_frequency",
            "teacher_model_type": "range_aware",
            "replicate_count": 4,
            "positive_label_count": 120,
            "positive_label_fraction": 0.25,
            "positive_label_mass": 16.0,
        },
        "workload_distribution_comparison": {
            "workload_signature_gate": {
                "all_available": True,
                "all_pass": False,
                "pairs": {
                    "train": {
                        "metrics": {
                            "anchor_family_l1_distance": 0.16,
                            "footprint_family_l1_distance": 0.05,
                            "point_hit_distribution_ks": 0.22,
                            "ship_hit_distribution_ks": 0.10,
                            "point_hit_distribution_used_quantile_proxy": False,
                            "ship_hit_distribution_used_quantile_proxy": False,
                        }
                    }
                },
            },
            "summaries": {
                "train": {
                    "range_query_count": 28,
                    "coverage_fraction": 0.21,
                    "near_duplicate_query_rate": 0.10,
                    "point_hit_count_p50": 30.0,
                    "trajectory_hit_count_p50": 2.0,
                    "oracle_gap_over_best_baseline": 0.20,
                    "best_baseline": "uniform",
                },
                "eval": {
                    "range_query_count": 160,
                    "coverage_fraction": 0.215,
                    "empty_query_rate": 0.02,
                    "too_broad_query_rate": 0.03,
                    "near_duplicate_query_rate": 0.40,
                    "point_hit_count_p50": 24.0,
                    "trajectory_hit_count_p50": 3.0,
                    "oracle_gap_over_best_baseline": 0.25,
                    "best_baseline": "DouglasPeucker",
                },
                "selection": {
                    "range_query_count": 21,
                    "coverage_fraction": 0.205,
                    "near_duplicate_query_rate": 0.05,
                    "point_hit_count_p50": 18.0,
                    "trajectory_hit_count_p50": 2.0,
                    "oracle_gap_over_best_baseline": 0.22,
                    "best_baseline": "uniform",
                },
            },
        },
        "workload_diagnostics": {
            "train": {
                "range_signal": {
                    "labels": {
                        "positive_label_mass": 12.5,
                        "component_label_mass_basis": "pre_clamp_component_contributions",
                        "component_positive_label_mass_fraction": {
                            "range_point_f1": 0.22,
                            "range_ship_f1": 0.13,
                            "range_ship_coverage": 0.12,
                            "range_entry_exit_f1": 0.09,
                            "range_crossing_f1": 0.04,
                            "range_temporal_coverage": 0.11,
                            "range_gap_coverage": 0.10,
                            "range_turn_coverage": 0.08,
                            "range_shape_score": 0.11,
                        },
                    }
                }
            }
        },
        "training_target_diagnostics": {
            "positive_label_mass": 11.0,
            "historical_prior_source_count": 4,
            "historical_prior_stored_support_count": 1234,
            "budget_rows": [
                {
                    "total_budget_ratio": 0.01,
                    "effective_fill_budget_ratio": 0.008,
                    "temporal_base_label_mass_fraction": 0.20,
                    "residual_label_mass_fraction": 0.80,
                    "residual_positive_label_fraction": 0.10,
                },
                {
                    "total_budget_ratio": 0.05,
                    "effective_fill_budget_ratio": 0.041,
                    "temporal_base_label_mass_fraction": 0.35,
                    "residual_label_mass_fraction": 0.65,
                    "residual_positive_label_fraction": 0.20,
                },
            ],
        },
        "training_fit_diagnostics": {
            "score_target_kendall_tau": 0.31,
            "matched_mlqds_target_recall": 0.74,
            "matched_uniform_target_recall": 0.62,
            "matched_mlqds_vs_uniform_target_recall": 0.12,
            "low_budget_mean_mlqds_vs_uniform_target_recall": -0.04,
        },
        "range_training_target_transform": {
            "mode": "local_swap_utility_frequency",
            "positive_label_count": 17,
            "positive_label_fraction": 0.25,
            "positive_label_mass": 3.5,
            "local_swap_utility_scored_candidate_count": 40,
            "local_swap_utility_positive_gain_candidate_count": 11,
            "local_swap_utility_selected_count": 7,
            "local_swap_utility_selected_gain_mass": 1.25,
            "local_swap_utility_source_positive_mass": 3.5,
            "local_swap_gain_cost_scored_candidate_count": 44,
            "local_swap_gain_cost_positive_net_gain_count": 12,
            "local_swap_gain_cost_selected_count": 8,
            "local_swap_gain_cost_selected_candidate_value_mass": 1.50,
            "local_swap_gain_cost_selected_removal_cost_mass": 0.40,
            "local_swap_gain_cost_source_positive_mass": 3.75,
        },
        "matched": {
            "MLQDS": {
                "aggregate_f1": 0.40,
                "range_point_f1": 0.40,
                "range_usefulness_score": 0.42,
                "query_useful_v1_score": 0.46,
                "range_ship_f1": 0.60,
                "range_ship_coverage": 0.64,
                "range_entry_exit_f1": 0.25,
                "range_crossing_f1": 0.48,
                "range_temporal_coverage": 0.58,
                "range_gap_coverage": 0.31,
                "range_gap_time_coverage": 0.41,
                "range_gap_distance_coverage": 0.36,
                "range_gap_min_coverage": 0.36,
                "range_turn_coverage": 0.52,
                "range_shape_score": 0.44,
                "range_usefulness_schema_version": 7,
                "range_usefulness_gap_time_score": 0.429,
                "range_usefulness_gap_distance_score": 0.4245,
                "range_usefulness_gap_min_score": 0.4245,
                "range_usefulness_gap_ablation_version": 1,
                "geometric_distortion": {
                    "avg_sed_km": 0.60,
                    "max_sed_km": 4.0,
                    "avg_ped_km": 0.20,
                    "max_ped_km": 2.0,
                    "removed_points": 90,
                },
                "avg_length_preserved": 0.88,
                "latency_ms": 8.0,
            },
            "uniform": {
                "aggregate_f1": 0.35,
                "range_point_f1": 0.35,
                "range_usefulness_score": 0.37,
                "query_useful_v1_score": 0.39,
                "range_ship_f1": 0.50,
                "range_ship_coverage": 0.60,
                "range_entry_exit_f1": 0.30,
                "range_crossing_f1": 0.47,
                "range_temporal_coverage": 0.62,
                "range_gap_coverage": 0.40,
                "range_gap_time_coverage": 0.50,
                "range_gap_distance_coverage": 0.48,
                "range_gap_min_coverage": 0.48,
                "range_turn_coverage": 0.51,
                "range_shape_score": 0.50,
                "range_usefulness_gap_time_score": 0.379,
                "range_usefulness_gap_distance_score": 0.3772,
                "range_usefulness_gap_min_score": 0.3772,
                "geometric_distortion": {
                    "avg_sed_km": 0.55,
                    "max_sed_km": 3.5,
                    "avg_ped_km": 0.18,
                    "max_ped_km": 1.7,
                    "removed_points": 90,
                },
                "avg_length_preserved": 0.91,
                "latency_ms": 2.0,
            },
            "DouglasPeucker": {
                "aggregate_f1": 0.36,
                "range_point_f1": 0.36,
                "range_usefulness_score": 0.39,
                "query_useful_v1_score": 0.41,
                "range_ship_f1": 0.48,
                "range_ship_coverage": 0.55,
                "range_entry_exit_f1": 0.28,
                "range_crossing_f1": 0.42,
                "range_temporal_coverage": 0.51,
                "range_gap_coverage": 0.22,
                "range_gap_time_coverage": 0.30,
                "range_gap_distance_coverage": 0.26,
                "range_gap_min_coverage": 0.26,
                "range_turn_coverage": 0.49,
                "range_shape_score": 0.35,
                "range_usefulness_gap_time_score": 0.3972,
                "range_usefulness_gap_distance_score": 0.3936,
                "range_usefulness_gap_min_score": 0.3936,
                "geometric_distortion": {
                    "avg_sed_km": 0.45,
                    "max_sed_km": 3.0,
                    "avg_ped_km": 0.15,
                    "max_ped_km": 1.2,
                    "removed_points": 90,
                },
                "avg_length_preserved": 0.93,
                "latency_ms": 3.0,
            },
        },
        "range_compression_audit": {
            "0.0100": {
                "MLQDS": {
                    "range_usefulness_score": 0.10,
                    "query_useful_v1_score": 0.11,
                    "range_usefulness_gap_time_score": 0.11,
                },
                "uniform": {
                    "range_usefulness_score": 0.12,
                    "query_useful_v1_score": 0.13,
                    "range_usefulness_gap_time_score": 0.10,
                },
                "DouglasPeucker": {"range_usefulness_score": 0.09, "query_useful_v1_score": 0.10},
                "TemporalRandomFill": {"range_usefulness_score": 0.11},
            },
            "0.0500": {
                "MLQDS": {
                    "range_usefulness_score": 0.42,
                    "query_useful_v1_score": 0.46,
                    "range_usefulness_gap_time_score": 0.43,
                },
                "uniform": {
                    "range_usefulness_score": 0.37,
                    "query_useful_v1_score": 0.39,
                    "range_usefulness_gap_time_score": 0.38,
                },
                "DouglasPeucker": {"range_usefulness_score": 0.39, "query_useful_v1_score": 0.41},
                "TemporalRandomFill": {"range_usefulness_score": 0.41},
            },
            "0.1000": {
                "MLQDS": {"range_usefulness_score": 0.50, "query_useful_v1_score": 0.52},
                "uniform": {"range_usefulness_score": 0.48, "query_useful_v1_score": 0.49},
                "DouglasPeucker": {"range_usefulness_score": 0.47, "query_useful_v1_score": 0.48},
                "TemporalRandomFill": {"range_usefulness_score": 0.52},
            },
        },
        "learned_fill_diagnostics": {
            "TemporalRandomFill": {
                "range_point_f1": 0.38,
                "range_usefulness_score": 0.41,
            },
            "TemporalOracleFill": {
                "range_point_f1": 0.55,
                "range_usefulness_score": 0.70,
            },
        },
        "best_epoch": 2,
        "best_selection_score": 0.42,
        "training_history": [
            {
                "epoch": 0.0,
                "loss": 1.0,
                "pred_std": 0.0,
                "kendall_tau_t0": 0.1,
                "collapse_warning": 1.0,
                "epoch_forward_seconds": 0.2,
                "epoch_loss_seconds": 0.4,
                "epoch_backward_seconds": 0.3,
                "epoch_diagnostic_seconds": 0.1,
                "epoch_validation_score_seconds": 0.0,
            },
            {
                "epoch": 1.0,
                "loss": 0.8,
                "pred_std": 0.2,
                "kendall_tau_t0": 0.2,
                "epoch_forward_seconds": 0.4,
                "epoch_loss_seconds": 0.6,
                "epoch_backward_seconds": 0.5,
                "epoch_diagnostic_seconds": 0.3,
                "epoch_validation_score_seconds": 0.7,
            },
        ],
        "torch_runtime": {
            "float32_matmul_precision": "high",
            "tf32_matmul_allowed": True,
            "tf32_cudnn_allowed": True,
            "amp": {"enabled": True, "dtype": "bfloat16"},
        }
    }

    row = _row_from_run(
        workload="range",
        run_label="custom_runtime",
        command=["python", "-m", "experiments.run_ais_experiment"],
        returncode=0,
        elapsed_seconds=10.0,
        run_dir=tmp_path,
        stdout_path=tmp_path / "stdout.log",
        run_json_path=tmp_path / "example_run.json",
        timings={
            "phase_timings": [
                {"name": "train-model", "seconds": 6.0},
                {"name": "evaluate-matched", "seconds": 2.0},
            ],
            "epoch_timings": [],
            "inference_step_timings": [],
        },
        run_json=run_json,
    )

    assert row["float32_matmul_precision"] == "high"
    assert row["allow_tf32"] is True
    assert row["amp_mode"] == "bf16"
    assert row["child_float32_matmul_precision"] == "high"
    assert row["child_tf32_matmul_allowed"] is True
    assert row["child_amp_enabled"] is True
    assert row["child_amp_dtype"] == "bfloat16"
    assert row["model_type"] == "range_aware"
    assert row["historical_prior_k"] == 7
    assert row["historical_prior_mmsi_weight"] == 2.5
    assert row["historical_prior_source_aggregation"] == "mean"
    assert row["historical_prior_source_count"] == 4
    assert row["historical_prior_stored_support_count"] == 1234
    assert row["compression_ratio"] == 0.05
    assert row["n_queries"] == 24
    assert row["max_queries"] == 512
    assert row["query_target_coverage"] == 0.30
    assert row["range_spatial_km"] == 2.2
    assert row["range_time_hours"] == 5.0
    assert row["mlqds_temporal_fraction"] == 0.25
    assert row["mlqds_diversity_bonus"] == 0.05
    assert row["mlqds_effective_diversity_bonus"] == 0.05
    assert row["mlqds_hybrid_mode"] == "swap"
    assert row["mlqds_stratified_center_weight"] == 0.45
    assert row["mlqds_min_learned_swaps"] == 1
    assert row["mlqds_score_mode"] == "rank"
    assert row["mlqds_score_temperature"] == 1.0
    assert row["mlqds_rank_confidence_weight"] == 0.15
    assert row["mlqds_range_geometry_blend"] == 0.25
    assert row["temporal_residual_label_mode"] == "temporal"
    assert row["range_label_mode"] == "usefulness"
    assert row["range_replicate_target_aggregation"] == "label_mean"
    assert row["range_component_target_blend"] == 0.75
    assert row["range_temporal_target_blend"] == 0.15
    assert row["range_target_budget_weight_power"] == 0.50
    assert row["range_marginal_target_radius_scale"] == 0.65
    assert row["range_query_spine_fraction"] == 0.30
    assert row["range_query_spine_mass_mode"] == "query"
    assert row["range_query_residual_multiplier"] == 1.25
    assert row["range_query_residual_mass_mode"] == "point"
    assert row["range_set_utility_multiplier"] == 1.75
    assert row["range_set_utility_candidate_limit"] == 64
    assert row["range_set_utility_mass_mode"] == "query"
    assert row["loss_objective"] == "budget_topk"
    assert row["budget_loss_ratios"] == [0.01, 0.02, 0.05, 0.10]
    assert row["budget_loss_temperature"] == 0.10
    assert row["temporal_distribution_loss_weight"] == 0.05
    assert row["range_train_workload_replicates"] == 3
    assert row["range_time_domain_mode"] == "anchor_day"
    assert row["range_anchor_mode"] == "sparse"
    assert row["range_train_anchor_modes"] == ["mixed_density", "sparse"]
    assert row["range_train_footprints"] == ["1.1:2.5", "2.2:5.0"]
    assert row["range_max_coverage_overshoot"] == 0.02
    assert row["train_query_final_count"] == 28
    assert row["train_query_final_coverage"] == 0.21
    assert row["train_query_target_reached"] is True
    assert row["train_query_target_shortfall"] == 0.0
    assert row["train_query_target_overshoot"] == pytest.approx(0.01)
    assert row["train_query_target_missed_by_max_queries"] is False
    assert row["train_query_extra_after_target_reached"] == 4
    assert row["eval_query_final_count"] == 160
    assert row["eval_query_final_coverage"] == 0.215
    assert row["eval_query_target_reached"] is True
    assert row["eval_query_target_shortfall"] == 0.0
    assert row["eval_query_target_overshoot"] == pytest.approx(0.015)
    assert row["eval_query_target_missed_by_max_queries"] is False
    assert row["eval_query_target_reached_count"] == 31
    assert row["eval_query_extra_after_target_reached"] == 129
    assert row["eval_query_extra_after_target_fraction"] == pytest.approx(129 / 160)
    assert row["eval_query_floor_dominated"] is True
    assert row["eval_query_generation_stop_reason"] == "target_coverage_reached"
    assert row["eval_workload_near_duplicate_query_rate"] == 0.40
    assert row["eval_workload_best_baseline"] == "DouglasPeucker"
    assert row["selection_query_final_count"] == 21
    assert row["selection_query_target_reached"] is True
    assert row["selection_query_target_shortfall"] == 0.0
    assert row["selection_query_target_overshoot"] == pytest.approx(0.005)
    assert row["selection_query_target_missed_by_max_queries"] is False
    assert row["selection_query_extra_after_target_reached"] == 0
    assert row["selection_query_floor_dominated"] is False
    assert row["checkpoint_full_score_every"] == 3
    assert row["checkpoint_candidate_pool_size"] == 2
    assert row["best_selection_score"] == 0.42
    assert row["final_loss"] == 0.8
    assert row["final_kendall_tau_t0"] == 0.2
    assert row["final_pred_std"] == 0.2
    assert row["epoch_forward_mean_seconds"] == pytest.approx(0.3)
    assert row["epoch_loss_mean_seconds"] == pytest.approx(0.5)
    assert row["epoch_validation_score_mean_seconds"] == pytest.approx(0.35)
    assert row["runtime_bottleneck_phase"] == "train-model"
    assert row["runtime_bottleneck_seconds"] == 6.0
    assert row["runtime_bottleneck_fraction"] == pytest.approx(0.6)
    assert row["single_cell_range_status"] == "diagnostic_upper_bound"
    assert row["final_claim_status"] == "candidate_blocked_by_required_gates"
    assert row["final_success_allowed"] is False
    assert row["final_claim_blocking_gates"] == ["predictability_gate"]
    assert row["workload_stability_gate_pass"] is False
    assert row["workload_stability_failed_checks"] == ["train_r0:not_target_coverage_generation"]
    assert row["workload_stability_train_replicates"] == 1
    assert row["workload_stability_configured_target_coverage"] == 0.10
    assert row["support_overlap_gate_pass"] is True
    assert row["support_overlap_failed_checks"] == []
    assert row["support_eval_points_outside_train_prior_extent_fraction"] == 0.02
    assert row["support_sampled_prior_nonzero_fraction"] == 0.80
    assert row["support_primary_sampled_prior_nonzero_fraction"] == 0.60
    assert row["support_route_density_overlap"] == 0.70
    assert row["support_query_prior_support_overlap"] == 0.65
    assert row["support_train_eval_spatial_extent_intersection_fraction"] == 0.90
    assert row["global_sanity_gate_pass"] is True
    assert row["global_sanity_failed_checks"] == []
    assert row["global_sanity_endpoint_sanity"] == 1.0
    assert row["global_sanity_avg_sed_ratio_vs_uniform"] == pytest.approx(1.09)
    assert row["global_sanity_avg_length_preserved"] == pytest.approx(0.88)
    assert row["predictability_gate_pass"] is False
    assert row["predictability_spearman"] == 0.12
    assert row["predictability_lift_at_5_percent"] == 1.18
    assert row["predictability_pr_auc_lift_over_base_rate"] == 1.20
    assert row["workload_signature_gate_available"] is True
    assert row["workload_signature_gate_pass"] is False
    assert row["workload_signature_pair_count"] == 1
    assert row["workload_signature_failed_pairs"] == ["train"]
    assert row["train_eval_anchor_family_l1_distance"] == 0.16
    assert row["train_eval_point_hit_distribution_ks"] == 0.22
    assert row["train_eval_point_hit_distribution_ks_proxy"] == 0.22
    assert row["train_eval_point_hit_distribution_used_quantile_proxy"] is False
    assert row["learning_causality_ablation_status"] == "partial"
    assert row["learning_causality_gate_pass"] is False
    assert row["learning_causality_failed_checks"] == ["shuffled_scores_should_lose"]
    assert row["causality_ablation_missing"] == ["MLQDS_without_segment_budget_head"]
    assert row["learned_controlled_retained_slot_fraction"] == 0.72
    assert row["planned_learned_controlled_retained_slot_fraction"] == 0.80
    assert row["actual_learned_controlled_retained_slot_fraction"] == 0.72
    assert row["trajectories_with_at_least_one_learned_decision"] == 5
    assert row["trajectories_with_zero_learned_decisions"] == 1
    assert row["segment_budget_entropy"] == 1.4
    assert row["segment_budget_entropy_normalized"] == 0.7
    assert row["selector_trace_retained_mask_matches_primary"] is True
    assert row["shuffled_score_ablation_delta"] == 0.04
    assert row["untrained_score_ablation_delta"] == 0.06
    assert row["shuffled_prior_field_ablation_delta"] == 0.05
    assert row["no_query_prior_field_ablation_delta"] == 0.03
    assert row["no_behavior_head_ablation_delta"] == 0.07
    assert row["no_segment_budget_head_ablation_delta"] == 0.08
    assert row["learning_causality_min_material_delta"] == 0.005
    assert row["learning_causality_shuffled_fraction_of_uniform_gap_min"] == 0.60
    assert row["learning_causality_mlqds_uniform_gap"] == 0.05
    assert row["learning_causality_delta_thresholds"]["shuffled_scores_should_lose"] == 0.03
    assert row["segment_budget_head_ablation_mode"] == "neutral_constant_segment_scores"
    assert row["prior_sample_gate_pass"] is False
    assert row["prior_sample_gate_failures"] == ["sampled_query_prior_features_all_zero"]
    assert row["shuffled_prior_sampled_inputs_changed"] is False
    assert row["shuffled_prior_sampled_primary_nonzero_fraction"] == 0.0
    assert row["shuffled_prior_sampled_ablation_nonzero_fraction"] == 0.01
    assert row["shuffled_prior_sampled_mean_abs_feature_delta"] == 0.002
    assert row["shuffled_prior_sampled_max_abs_feature_delta"] == 0.10
    assert row["shuffled_prior_sampled_outside_extent_fraction"] == 0.75
    assert row["no_prior_sampled_primary_nonzero_fraction"] == 0.0
    assert row["no_prior_sampled_mean_abs_feature_delta"] == 0.0
    assert row["no_prior_sampled_outside_extent_fraction"] == 0.75
    assert row["workload_blind_candidate"] is False
    assert row["workload_blind_protocol_enabled"] is False
    assert row["primary_masks_frozen_before_eval_query_scoring"] is False
    assert row["audit_masks_frozen_before_eval_query_scoring"] is False
    assert row["eval_geometry_blend_allowed"] is True
    assert row["beats_uniform_range_usefulness"] is True
    assert row["beats_douglas_peucker_range_usefulness"] is True
    assert row["beats_temporal_random_fill_range_usefulness"] is True
    assert row["audit_compression_ratio_count"] == 3
    assert row["audit_low_compression_ratio_count"] == 2
    assert row["audit_beats_uniform_range_usefulness_count"] == 2
    assert row["audit_beats_douglas_peucker_range_usefulness_count"] == 3
    assert row["audit_beats_temporal_random_fill_range_usefulness_count"] == 1
    assert row["audit_beats_both_range_usefulness_count"] == 2
    assert row["audit_low_beats_uniform_range_usefulness_count"] == 1
    assert row["audit_low_beats_temporal_random_fill_range_usefulness_count"] == 1
    assert row["audit_low_beats_both_range_usefulness_count"] == 1
    assert row["audit_beats_uniform_query_useful_v1_count"] == 2
    assert row["audit_beats_douglas_peucker_query_useful_v1_count"] == 3
    assert row["audit_low_beats_uniform_query_useful_v1_count"] == 1
    assert row["audit_min_low_vs_uniform_range_usefulness"] == pytest.approx(-0.02)
    assert row["audit_min_low_vs_uniform_query_useful_v1"] == pytest.approx(-0.02)
    assert row["audit_mean_low_vs_uniform_range_usefulness"] == pytest.approx(0.015)
    assert row["audit_mean_low_vs_uniform_query_useful_v1"] == pytest.approx(0.025)
    assert row["audit_min_low_vs_temporal_random_fill_range_usefulness"] == pytest.approx(-0.01)
    assert row["audit_mean_vs_temporal_random_fill_range_usefulness"] == pytest.approx(-0.02 / 3.0)
    assert row["audit_mean_low_vs_temporal_random_fill_range_usefulness"] == pytest.approx(0.0)
    assert row["audit_ratio_0p0100_compression_ratio"] == pytest.approx(0.01)
    assert row["audit_ratio_0p0100_mlqds_range_usefulness"] == pytest.approx(0.10)
    assert row["audit_ratio_0p0100_uniform_range_usefulness"] == pytest.approx(0.12)
    assert row["audit_ratio_0p0100_mlqds_query_useful_v1"] == pytest.approx(0.11)
    assert row["audit_ratio_0p0100_uniform_query_useful_v1"] == pytest.approx(0.13)
    assert row["audit_ratio_0p0100_mlqds_vs_uniform_query_useful_v1"] == pytest.approx(-0.02)
    assert row["audit_ratio_0p0100_douglas_peucker_range_usefulness"] == pytest.approx(0.09)
    assert row["audit_ratio_0p0100_temporal_random_fill_range_usefulness"] == pytest.approx(0.11)
    assert row["audit_ratio_0p0100_mlqds_vs_uniform_range_usefulness"] == pytest.approx(-0.02)
    assert row["audit_ratio_0p0100_mlqds_vs_douglas_peucker_range_usefulness"] == pytest.approx(0.01)
    assert row["audit_ratio_0p0100_mlqds_vs_temporal_random_fill_range_usefulness"] == pytest.approx(-0.01)
    assert row["audit_ratio_0p0500_mlqds_vs_uniform_range_usefulness"] == pytest.approx(0.05)
    assert row["audit_ratio_0p1000_mlqds_vs_temporal_random_fill_range_usefulness"] == pytest.approx(-0.02)
    assert row["range_boundary_prior_weight"] == 0.0
    assert row["range_boundary_prior_enabled"] is False
    assert row["teacher_distillation_enabled"] is True
    assert row["teacher_distillation_mode"] == "retained_frequency"
    assert row["teacher_model_type"] == "range_aware"
    assert row["teacher_replicate_count"] == 4
    assert row["teacher_positive_label_fraction"] == 0.25
    assert row["teacher_positive_label_mass"] == 16.0
    assert row["train_positive_label_mass"] == 12.5
    assert row["train_label_mass_basis"] == "pre_clamp_component_contributions"
    assert row["train_label_mass_range_point_f1"] == 0.22
    assert row["train_label_mass_range_ship_f1"] == 0.13
    assert row["train_label_mass_range_ship_coverage"] == 0.12
    assert row["train_label_mass_range_entry_exit_f1"] == 0.09
    assert row["train_label_mass_range_crossing_f1"] == 0.04
    assert row["train_label_mass_range_temporal_coverage"] == 0.11
    assert row["train_label_mass_range_gap_coverage"] == 0.10
    assert row["train_label_mass_range_turn_coverage"] == 0.08
    assert row["train_label_mass_range_shape_score"] == 0.11
    assert row["train_target_positive_label_mass"] == 11.0
    assert row["train_target_budget_ratio"] == 0.05
    assert row["train_target_effective_fill_budget_ratio"] == 0.041
    assert row["train_target_temporal_base_label_mass_fraction"] == 0.35
    assert row["train_target_residual_label_mass_fraction"] == 0.65
    assert row["train_target_residual_positive_label_fraction"] == 0.20
    assert row["train_fit_score_target_kendall_tau"] == 0.31
    assert row["train_fit_matched_mlqds_target_recall"] == 0.74
    assert row["train_fit_matched_uniform_target_recall"] == 0.62
    assert row["train_fit_matched_mlqds_vs_uniform_target_recall"] == 0.12
    assert row["train_fit_low_budget_mean_mlqds_vs_uniform_target_recall"] == -0.04
    assert row["range_target_transform_mode"] == "local_swap_utility_frequency"
    assert row["range_target_transform_positive_label_count"] == 17
    assert row["range_target_transform_positive_label_mass"] == 3.5
    assert row["local_swap_utility_scored_candidate_count"] == 40
    assert row["local_swap_utility_positive_gain_candidate_count"] == 11
    assert row["local_swap_utility_selected_count"] == 7
    assert row["local_swap_utility_selected_gain_mass"] == 1.25
    assert row["local_swap_utility_source_positive_mass"] == 3.5
    assert row["local_swap_gain_cost_scored_candidate_count"] == 44
    assert row["local_swap_gain_cost_positive_net_gain_count"] == 12
    assert row["local_swap_gain_cost_selected_count"] == 8
    assert row["local_swap_gain_cost_selected_candidate_value_mass"] == 1.50
    assert row["local_swap_gain_cost_selected_removal_cost_mass"] == 0.40
    assert row["local_swap_gain_cost_source_positive_mass"] == 3.75
    assert row["mlqds_primary_metric"] == "query_useful_v1"
    assert row["mlqds_primary_score"] == 0.46
    assert row["mlqds_aggregate_f1"] == 0.40
    assert row["mlqds_range_point_f1"] == 0.40
    assert row["mlqds_range_usefulness"] == 0.42
    assert row["mlqds_range_usefulness_score"] == 0.42
    assert row["mlqds_query_useful_v1_score"] == 0.46
    assert row["mlqds_range_usefulness_gap_time_score"] == 0.429
    assert row["mlqds_range_usefulness_gap_distance_score"] == 0.4245
    assert row["mlqds_range_usefulness_gap_min_score"] == 0.4245
    assert row["uniform_range_point_f1"] == 0.35
    assert row["uniform_range_usefulness"] == 0.37
    assert row["uniform_query_useful_v1_score"] == 0.39
    assert row["uniform_range_usefulness_gap_time_score"] == 0.379
    assert row["douglas_peucker_range_point_f1"] == 0.36
    assert row["douglas_peucker_range_usefulness"] == 0.39
    assert row["douglas_peucker_query_useful_v1_score"] == 0.41
    assert row["douglas_peucker_range_usefulness_gap_min_score"] == 0.3936
    assert row["mlqds_avg_sed_km"] == 0.60
    assert row["uniform_avg_sed_km"] == 0.55
    assert row["douglas_peucker_avg_sed_km"] == 0.45
    assert row["mlqds_avg_length_preserved"] == 0.88
    assert row["uniform_avg_length_preserved"] == 0.91
    assert row["douglas_peucker_avg_length_preserved"] == 0.93
    assert row["mlqds_vs_uniform_avg_sed_km"] == pytest.approx(0.05)
    assert row["mlqds_vs_uniform_avg_ped_km"] == pytest.approx(0.02)
    assert row["mlqds_vs_uniform_avg_length_preserved"] == pytest.approx(-0.03)
    assert row["mlqds_range_ship_coverage"] == 0.64
    assert row["mlqds_range_entry_exit_f1"] == 0.25
    assert row["mlqds_range_crossing_f1"] == 0.48
    assert row["mlqds_range_gap_coverage"] == 0.31
    assert row["mlqds_range_gap_time_coverage"] == 0.41
    assert row["mlqds_range_gap_distance_coverage"] == 0.36
    assert row["mlqds_range_gap_min_coverage"] == 0.36
    assert row["mlqds_range_turn_coverage"] == 0.52
    assert row["mlqds_vs_uniform_range_entry_exit_f1"] == pytest.approx(-0.05)
    assert row["mlqds_vs_uniform_range_temporal_coverage"] == pytest.approx(-0.04)
    assert row["mlqds_vs_uniform_range_gap_coverage"] == pytest.approx(-0.09)
    assert row["mlqds_vs_uniform_range_shape_score"] == pytest.approx(-0.06)
    assert row["worst_uniform_component_delta_metric"] == "mlqds_vs_uniform_range_gap_coverage"
    assert row["worst_uniform_component_delta"] == pytest.approx(-0.09)
    assert row["range_usefulness_schema_version"] == 7
    assert row["range_usefulness_gap_ablation_version"] == 1
    assert row["temporal_random_fill_range_point_f1"] == 0.38
    assert row["temporal_random_fill_range_usefulness_score"] == 0.41
    assert row["temporal_oracle_fill_range_point_f1"] == 0.55
    assert row["temporal_oracle_fill_range_usefulness_score"] == 0.70
    assert row["mlqds_vs_temporal_random_fill_range_usefulness"] == pytest.approx(0.01)
    assert row["temporal_oracle_fill_gap_range_usefulness"] == pytest.approx(0.28)
    assert row["collapse_warning_any"] is True
    assert row["collapse_warning_count"] == 1
    assert row["best_epoch_collapse_warning"] is False
    assert row["min_pred_std"] == 0.0
    assert row["best_epoch_pred_std"] == 0.2
    assert row["oracle_kind"] == "additive_label_greedy"
    assert row["oracle_exact_optimum"] is False
    assert row["mlqds_vs_uniform_range_point_f1"] == pytest.approx(0.05)
    assert row["mlqds_vs_douglas_peucker_range_point_f1"] == pytest.approx(0.04)
    assert row["mlqds_vs_uniform_range_usefulness"] == pytest.approx(0.05)
    assert row["mlqds_vs_uniform_query_useful_v1"] == pytest.approx(0.07)
    assert row["mlqds_vs_douglas_peucker_range_usefulness"] == pytest.approx(0.03)
    assert row["mlqds_vs_douglas_peucker_query_useful_v1"] == pytest.approx(0.05)
    assert row["mlqds_vs_uniform_range_usefulness_gap_time"] == pytest.approx(0.05)
    assert row["mlqds_vs_uniform_range_usefulness_gap_distance"] == pytest.approx(0.0473)
    assert row["mlqds_vs_uniform_range_usefulness_gap_min"] == pytest.approx(0.0473)
    assert row["mlqds_vs_douglas_peucker_range_usefulness_gap_time"] == pytest.approx(0.0318)
    assert row["mlqds_vs_douglas_peucker_range_usefulness_gap_distance"] == pytest.approx(0.0309)
    assert row["mlqds_vs_douglas_peucker_range_usefulness_gap_min"] == pytest.approx(0.0309)
    assert row["audit_beats_uniform_range_usefulness_gap_time_count"] == 2
    assert row["audit_low_beats_uniform_range_usefulness_gap_time_count"] == 2


def _final_grid_row(coverage: float, *, mlqds_delta: float = 0.05) -> dict[str, object]:
    row: dict[str, object] = {
        "workload": "range",
        "run_label": f"c{int(coverage * 100):02d}",
        "returncode": 0,
        "query_target_coverage": coverage,
        "compression_ratio": 0.05,
        "mlqds_query_useful_v1_score": 0.55,
        "uniform_query_useful_v1_score": 0.50,
        "douglas_peucker_query_useful_v1_score": 0.49,
        "workload_stability_gate_pass": True,
        "predictability_gate_pass": True,
        "target_diffusion_gate_pass": True,
        "workload_signature_gate_pass": True,
        "learning_causality_gate_pass": True,
        "prior_sample_gate_pass": True,
        "global_sanity_gate_pass": True,
        "support_overlap_gate_pass": True,
        "mlqds_primary_metric": "query_useful_v1",
    }
    for ratio in (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30):
        prefix = f"audit_ratio_{ratio:.4f}".replace(".", "p")
        row[f"{prefix}_mlqds_query_useful_v1"] = 0.50 + mlqds_delta
        row[f"{prefix}_uniform_query_useful_v1"] = 0.50
        row[f"{prefix}_douglas_peucker_query_useful_v1"] = 0.49
    return row


def test_query_driven_final_grid_summary_accepts_complete_passing_grid() -> None:
    rows = [_final_grid_row(coverage) for coverage in (0.05, 0.10, 0.15, 0.30)]
    run_config = {
        "profile_settings": {
            "final_product_candidate": True,
            "range_coverage_sweep_targets": [0.05, 0.10, 0.15, 0.30],
            "range_compression_sweep_ratios": [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30],
        }
    }

    summary = query_driven_final_grid_summary(rows, run_config)

    assert summary["status"] == "final_grid_pass"
    assert summary["final_success_allowed"] is True
    assert summary["grid_complete"] is True
    assert summary["observed_cell_count"] == 28
    assert summary["beats_uniform_queryuseful_cells"] == 28
    assert summary["beats_douglas_peucker_queryuseful_cells"] == 28
    assert summary["low_budget_beats_uniform_queryuseful_cells"] == 12
    assert summary["matched_5_percent_coverage_cells_uniform"] == 4
    assert summary["failed_checks"] == []


def test_query_driven_final_grid_summary_blocks_missing_or_failed_evidence() -> None:
    rows = [_final_grid_row(0.05), _final_grid_row(0.10, mlqds_delta=-0.02)]
    rows[0]["predictability_gate_pass"] = False

    summary = query_driven_final_grid_summary(
        rows,
        {"profile_settings": {"final_product_candidate": True}},
    )

    assert summary["status"] == "final_grid_blocked"
    assert summary["final_success_allowed"] is False
    assert "coverage_grid_incomplete" in summary["failed_checks"]
    assert "compression_grid_incomplete" in summary["failed_checks"]
    assert "too_few_uniform_queryuseful_wins" in summary["failed_checks"]
    assert "required_single_run_gates_failed" in summary["failed_checks"]
    assert summary["child_gate_failures"][0]["failed_gates"] == ["predictability_gate_pass"]


def test_query_floor_fields_flags_coverage_target_miss() -> None:
    fields = _query_floor_fields(
        "eval",
        {
            "mode": "target_coverage",
            "target_coverage": 0.30,
            "final_coverage": 0.251,
            "minimum_queries": 8,
            "max_queries": 128,
            "final_query_count": 128,
            "target_reached_query_count": None,
            "coverage_at_target_reached": None,
            "extra_queries_after_target_reached": None,
            "stop_reason": "max_queries_reached",
        },
    )

    assert fields["eval_query_target_reached"] is False
    assert fields["eval_query_target_shortfall"] == pytest.approx(0.049)
    assert fields["eval_query_target_overshoot"] == 0.0
    assert fields["eval_query_target_missed_by_max_queries"] is True


def test_benchmark_row_reports_zero_effective_diversity_for_stratified(tmp_path) -> None:
    """Configured diversity should not imply an effect in stratified mode."""
    run_json = {
        "config": {
            "query": {},
            "model": {
                "model_type": "range_prior",
                "mlqds_diversity_bonus": 0.05,
                "mlqds_hybrid_mode": "stratified",
                "compression_ratio": 0.05,
            },
        },
        "workload_blind_protocol": {
            "enabled": True,
            "primary_masks_frozen_before_eval_query_scoring": True,
            "audit_masks_frozen_before_eval_query_scoring": True,
        },
        "matched": {
            "MLQDS": {"range_usefulness_score": 0.2},
            "uniform": {"range_usefulness_score": 0.1},
            "DouglasPeucker": {"range_usefulness_score": 0.1},
        },
    }

    row = _row_from_run(
        workload="range",
        run_label="run",
        command=[],
        returncode=0,
        elapsed_seconds=1.0,
        run_dir=tmp_path,
        stdout_path=tmp_path / "stdout.log",
        run_json_path=tmp_path / "example_run.json",
        timings={"phase_timings": [], "epoch_timings": [], "inference_step_timings": []},
        run_json=run_json,
    )

    assert row["mlqds_diversity_bonus"] == 0.05
    assert row["mlqds_effective_diversity_bonus"] == 0.0


def test_benchmark_row_marks_blind_protocol_fail_even_with_good_score(tmp_path) -> None:
    run_json = {
        "config": {"model": {"model_type": "range_prior"}},
        "workload_blind_protocol": {
            "enabled": False,
            "primary_masks_frozen_before_eval_query_scoring": False,
            "audit_masks_frozen_before_eval_query_scoring": False,
        },
        "matched": {
            "MLQDS": {"range_usefulness_score": 0.50, "range_point_f1": 0.40},
            "uniform": {"range_usefulness_score": 0.40, "range_point_f1": 0.30},
            "DouglasPeucker": {"range_usefulness_score": 0.30, "range_point_f1": 0.20},
        },
    }

    row = _row_from_run(
        workload="range",
        run_label="leaky_blind",
        command=[],
        returncode=0,
        elapsed_seconds=1.0,
        run_dir=tmp_path,
        stdout_path=tmp_path / "stdout.log",
        run_json_path=tmp_path / "example_run.json",
        timings={"phase_timings": [], "epoch_timings": [], "inference_step_timings": []},
        run_json=run_json,
    )

    assert row["workload_blind_candidate"] is True
    assert row["beats_uniform_range_usefulness"] is True
    assert row["beats_douglas_peucker_range_usefulness"] is True
    assert row["single_cell_range_status"] == "protocol_fail"


def test_benchmark_row_blocks_blind_success_when_selector_scaffold_dominated(tmp_path) -> None:
    run_json = {
        "config": {"model": {"model_type": "range_prior", "compression_ratio": 0.05}},
        "workload_blind_protocol": {
            "enabled": True,
            "primary_masks_frozen_before_eval_query_scoring": True,
            "audit_masks_frozen_before_eval_query_scoring": True,
        },
        "matched": {
            "MLQDS": {"range_usefulness_score": 0.50},
            "uniform": {"range_usefulness_score": 0.40},
            "DouglasPeucker": {"range_usefulness_score": 0.30},
        },
        "selector_budget_diagnostics": {
            "eval": {
                "budget_rows": [
                    {
                        "compression_ratio": 0.05,
                        "learned_slot_fraction_of_budget": 0.10,
                    }
                ]
            }
        },
    }

    row = _row_from_run(
        workload="range",
        run_label="scaffold_dominated",
        command=[],
        returncode=0,
        elapsed_seconds=1.0,
        run_dir=tmp_path,
        stdout_path=tmp_path / "stdout.log",
        run_json_path=tmp_path / "example_run.json",
        timings={"phase_timings": [], "epoch_timings": [], "inference_step_timings": []},
        run_json=run_json,
    )

    assert row["beats_uniform_range_usefulness"] is True
    assert row["beats_douglas_peucker_range_usefulness"] is True
    assert row["selector_claim_status"] == "selector_scaffold_dominated"
    assert row["selector_claim_has_material_learned_budget"] is False
    assert row["selector_claim_min_learned_slot_fraction"] == (
        MIN_MATCHED_LEARNED_SLOT_FRACTION_FOR_BLIND_CLAIM
    )
    assert row["single_cell_range_status"] == "selector_scaffold_dominated"


def test_benchmark_row_allows_blind_success_with_material_learned_budget(tmp_path) -> None:
    run_json = {
        "config": {"model": {"model_type": "range_prior", "compression_ratio": 0.05}},
        "workload_blind_protocol": {
            "enabled": True,
            "primary_masks_frozen_before_eval_query_scoring": True,
            "audit_masks_frozen_before_eval_query_scoring": True,
        },
        "matched": {
            "MLQDS": {"range_usefulness_score": 0.50},
            "uniform": {"range_usefulness_score": 0.40},
            "DouglasPeucker": {"range_usefulness_score": 0.30},
        },
        "selector_budget_diagnostics": {
            "eval": {
                "budget_rows": [
                    {
                        "compression_ratio": 0.05,
                        "learned_slot_fraction_of_budget": 0.35,
                    }
                ]
            }
        },
    }

    row = _row_from_run(
        workload="range",
        run_label="material_learned_budget",
        command=[],
        returncode=0,
        elapsed_seconds=1.0,
        run_dir=tmp_path,
        stdout_path=tmp_path / "stdout.log",
        run_json_path=tmp_path / "example_run.json",
        timings={"phase_timings": [], "epoch_timings": [], "inference_step_timings": []},
        run_json=run_json,
    )

    assert row["selector_claim_status"] == "model_has_material_budget"
    assert row["selector_claim_has_material_learned_budget"] is True
    assert row["single_cell_range_status"] == "beats_uniform_and_douglas_peucker"


def test_benchmark_row_records_data_source_metadata(tmp_path) -> None:
    run_json = {
        "config": {"model": {"model_type": "historical_prior", "compression_ratio": 0.05}},
        "workload_blind_protocol": {
            "enabled": True,
            "primary_masks_frozen_before_eval_query_scoring": True,
            "audit_masks_frozen_before_eval_query_scoring": True,
        },
        "matched": {
            "MLQDS": {"range_usefulness_score": 0.50},
            "uniform": {"range_usefulness_score": 0.40},
            "DouglasPeucker": {"range_usefulness_score": 0.30},
        },
        "selector_budget_diagnostics": {
            "eval": {
                "budget_rows": [
                    {
                        "compression_ratio": 0.01,
                        "learned_slot_count": 0,
                        "learned_slot_fraction_of_budget": 0.0,
                    },
                    {
                        "compression_ratio": 0.05,
                        "learned_slot_count": 12,
                        "learned_slot_fraction_of_budget": 0.20,
                        "zero_learned_slot_trajectory_fraction": 0.10,
                        "endpoint_only_trajectory_fraction": 0.0,
                    },
                ]
            }
        },
    }

    row = _row_from_run(
        workload="range",
        run_label="multi_day",
        command=[],
        returncode=0,
        elapsed_seconds=1.0,
        run_dir=tmp_path,
        stdout_path=tmp_path / "stdout.log",
        run_json_path=tmp_path / "example_run.json",
        timings={"phase_timings": [], "epoch_timings": [], "inference_step_timings": []},
        run_json=run_json,
        data_sources={
            "csv_path": None,
            "train_csv_path": "train_a.csv,train_b.csv",
            "validation_csv_path": "validation_a.csv,validation_b.csv",
            "eval_csv_path": "eval_a.csv,eval_b.csv",
            "selected_cleaned_csv_files": [
                "train_a.csv",
                "train_b.csv",
                "validation_a.csv",
                "validation_b.csv",
                "eval_a.csv",
                "eval_b.csv",
            ],
        },
    )

    assert row["train_csv_path"] == "train_a.csv,train_b.csv"
    assert row["validation_csv_path"] == "validation_a.csv,validation_b.csv"
    assert row["eval_csv_path"] == "eval_a.csv,eval_b.csv"
    assert row["train_csv_file_count"] == 2
    assert row["validation_csv_file_count"] == 2
    assert row["eval_csv_file_count"] == 2
    assert row["selected_cleaned_csv_file_count"] == 6
    assert row["selected_cleaned_csv_files"] == (
        "train_a.csv;train_b.csv;validation_a.csv;validation_b.csv;eval_a.csv;eval_b.csv"
    )
    assert row["eval_selector_matched_learned_slot_fraction"] == 0.20
    assert row["eval_selector_matched_zero_learned_trajectory_fraction"] == 0.10
    assert row["eval_selector_low_budget_zero_learned_ratio_count"] == 1
    assert row["eval_selector_low_budget_min_learned_slot_fraction"] == 0.0


def test_profile_args_use_csv_when_provided() -> None:
    args = argparse.Namespace(
        csv_path="../AISDATA/cleaned/day.csv",
        train_csv_path=None,
        validation_csv_path=None,
        eval_csv_path=None,
        cache_dir="artifacts/cache/benchmark",
        refresh_cache=True,
        min_points_per_segment=4,
        max_points_per_segment=128,
        max_time_gap_seconds=3600.0,
        max_segments=16,
        max_trajectories=8,
    )
    data_sources = BenchmarkDataSources(csv_path="../AISDATA/cleaned/day.csv")

    assert _profile_args(DEFAULT_PROFILE, args, data_sources) == [
        "--csv_path",
        "../AISDATA/cleaned/day.csv",
        *_profile_core_args(),
        "--min_points_per_segment",
        "4",
        "--max_time_gap_seconds",
        "3600.0",
        "--max_points_per_segment",
        "128",
        "--max_segments",
        "16",
        "--max_trajectories",
        "8",
        "--cache_dir",
        "artifacts/cache/benchmark",
        "--refresh_cache",
    ]


def test_profile_args_use_three_day_train_validation_eval_sources() -> None:
    args = argparse.Namespace(
        csv_path=None,
        train_csv_path="../AISDATA/cleaned/day1.csv",
        validation_csv_path="../AISDATA/cleaned/day2.csv",
        eval_csv_path="../AISDATA/cleaned/day3.csv",
        cache_dir="artifacts/cache/benchmark",
        refresh_cache=True,
        min_points_per_segment=4,
        max_points_per_segment=None,
        max_time_gap_seconds=3600.0,
        max_segments=None,
        max_trajectories=None,
    )
    data_sources = BenchmarkDataSources(
        train_csv_path="../AISDATA/cleaned/day1.csv",
        validation_csv_path="../AISDATA/cleaned/day2.csv",
        eval_csv_path="../AISDATA/cleaned/day3.csv",
    )

    assert _profile_args(DEFAULT_PROFILE, args, data_sources, include_refresh_cache=False) == [
        "--train_csv_path",
        "../AISDATA/cleaned/day1.csv",
        "--validation_csv_path",
        "../AISDATA/cleaned/day2.csv",
        "--eval_csv_path",
        "../AISDATA/cleaned/day3.csv",
        *_profile_core_args(),
        "--min_points_per_segment",
        "4",
        "--max_time_gap_seconds",
        "3600.0",
        "--cache_dir",
        "artifacts/cache/benchmark",
    ]


def test_profile_args_support_workload_blind_profiles() -> None:
    args = argparse.Namespace(
        csv_path=None,
        train_csv_path="../AISDATA/cleaned/day1.csv",
        validation_csv_path="../AISDATA/cleaned/day2.csv",
        eval_csv_path="../AISDATA/cleaned/day3.csv",
        cache_dir="artifacts/cache/benchmark",
        refresh_cache=False,
        min_points_per_segment=4,
        max_points_per_segment=256,
        max_time_gap_seconds=3600.0,
        max_segments=120,
        max_trajectories=None,
    )
    data_sources = BenchmarkDataSources(
        train_csv_path="../AISDATA/cleaned/day1.csv",
        validation_csv_path="../AISDATA/cleaned/day2.csv",
        eval_csv_path="../AISDATA/cleaned/day3.csv",
    )

    profile_args = _profile_args(BLIND_RETAINED_FREQUENCY_PROFILE, args, data_sources, include_refresh_cache=False)

    assert profile_args[:6] == [
        "--train_csv_path",
        "../AISDATA/cleaned/day1.csv",
        "--validation_csv_path",
        "../AISDATA/cleaned/day2.csv",
        "--eval_csv_path",
        "../AISDATA/cleaned/day3.csv",
    ]
    assert profile_args[profile_args.index("--model_type") + 1] == "workload_blind_range"
    assert profile_args[profile_args.index("--range_training_target_mode") + 1] == "retained_frequency"
    assert profile_args[profile_args.index("--range_max_coverage_overshoot") + 1] == "0.02"
    assert profile_args[profile_args.index("--range_audit_compression_ratios") + 1] == (
        "0.01,0.02,0.05,0.10,0.15,0.20,0.30"
    )
    assert "--max_points_per_segment" in profile_args
    assert profile_args[profile_args.index("--max_segments") + 1] == "120"


def test_workload_aware_diagnostic_profile_uses_requested_training_shape() -> None:
    args = argparse.Namespace(
        csv_path=None,
        train_csv_path="../AISDATA/cleaned/day1.csv",
        validation_csv_path="../AISDATA/cleaned/day2.csv",
        eval_csv_path="../AISDATA/cleaned/day3.csv",
        cache_dir="artifacts/cache/benchmark",
        refresh_cache=False,
        min_points_per_segment=4,
        max_points_per_segment=None,
        max_time_gap_seconds=3600.0,
        max_segments=None,
        max_trajectories=None,
    )
    data_sources = BenchmarkDataSources(
        train_csv_path="../AISDATA/cleaned/day1.csv",
        validation_csv_path="../AISDATA/cleaned/day2.csv",
        eval_csv_path="../AISDATA/cleaned/day3.csv",
    )

    profile_args = _profile_args(DEFAULT_PROFILE, args, data_sources, include_refresh_cache=False)

    assert profile_args[6 : 6 + len(_profile_core_args())] == _profile_core_args()
    assert "--max_points_per_segment" not in profile_args
    assert "--max_segments" not in profile_args
    assert "--max_trajectories" not in profile_args


def test_selection_query_count_matches_eval_query_count() -> None:
    cfg = build_experiment_config(n_queries=80, query_coverage=0.20, max_queries=None)

    assert validation_query_count(cfg) == 80


def test_csv_config_suppresses_inactive_synthetic_metadata() -> None:
    cfg = build_experiment_config(
        n_ships=24,
        n_points=200,
        train_csv_path="../AISDATA/cleaned/day1.csv",
        validation_csv_path="../AISDATA/cleaned/day2.csv",
        eval_csv_path="../AISDATA/cleaned/day3.csv",
    )

    assert cfg.data.n_ships is None
    assert cfg.data.n_points_per_ship is None


def test_child_run_dir_uses_readable_layout(tmp_path) -> None:
    run_label = "custom_run"

    assert _child_run_dir(tmp_path, "range", run_label, 1) == tmp_path / "custom_run"
    assert _child_run_dir(tmp_path, "range", run_label, 2) == tmp_path / "range" / "custom_run"


def test_family_index_upserts_current_status_and_appends_events(tmp_path) -> None:
    args = argparse.Namespace(
        profile=DEFAULT_PROFILE,
        seed=42,
        max_points_per_segment=3000,
        max_segments=None,
        max_trajectories=None,
    )
    run_label = "custom_run"
    sources = BenchmarkDataSources(train_csv_path="day1.csv", validation_csv_path="day2.csv", eval_csv_path="day3.csv")
    git = {"commit": "abc123", "dirty": False}
    running_status = {
        "status": "running",
        "started_at_utc": "2026-05-10T00:00:00+00:00",
        "finished_at_utc": None,
        "exit_status": None,
        "failures": None,
    }
    completed_status = {
        **running_status,
        "status": "completed",
        "finished_at_utc": "2026-05-10T00:01:00+00:00",
        "exit_status": 0,
        "failures": 0,
    }

    write_family_indexes(
        tmp_path,
        index_entry(
            run_id="run-a",
            status_payload=running_status,
            args=args,
            workloads=["range"],
            run_label=run_label,
            data_sources=sources,
            results_dir=tmp_path / "runs" / "run-a",
            rows=[],
            git=git,
        ),
    )
    write_family_indexes(
        tmp_path,
        index_entry(
            run_id="run-a",
            status_payload=completed_status,
            args=args,
            workloads=["range"],
            run_label=run_label,
            data_sources=sources,
            results_dir=tmp_path / "runs" / "run-a",
            rows=[
                {
                    "run_label": "custom_run",
                    "mlqds_primary_metric": "range_usefulness",
                    "mlqds_primary_score": 0.42,
                    "mlqds_aggregate_f1": 0.4,
                    "mlqds_range_point_f1": 0.4,
                    "mlqds_range_usefulness": 0.42,
                }
            ],
            git=git,
        ),
    )

    with open(tmp_path / "runs_index.csv", encoding="utf-8", newline="") as f:
        index_rows = list(csv.DictReader(f))
    events_text = (tmp_path / "runs_index_events.jsonl").read_text(encoding="utf-8")
    assert len(index_rows) == 1
    assert index_rows[0]["run_id"] == "run-a"
    assert index_rows[0]["status"] == "completed"
    assert index_rows[0]["run_label"] == "custom_run"
    assert index_rows[0]["best_mlqds_primary_metric"] == "range_usefulness"
    assert index_rows[0]["best_mlqds_primary_score"] == "0.42"
    assert index_rows[0]["best_mlqds_aggregate_f1"] == "0.4"
    assert index_rows[0]["best_mlqds_range_point_f1"] == "0.4"
    assert index_rows[0]["best_mlqds_range_usefulness"] == "0.42"
    assert index_rows[0]["best_mlqds_run_label"] == "custom_run"
    assert events_text.count('"run_id": "run-a"') == 2


def test_benchmark_report_records_concrete_family_root(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    family = tmp_path / "range_family"
    results_dir = family / "runs" / "artifact-test"
    train_csv = tmp_path / "train.csv"
    validation_csv = tmp_path / "validation.csv"
    eval_csv = tmp_path / "eval.csv"
    for path in (train_csv, validation_csv, eval_csv):
        path.write_text("mmsi,timestamp,lat,lon\n", encoding="utf-8")

    def fake_run_capture_streaming(
        command: list[str],
        cwd: Path,
        stdout_path: Path,
        *,
        max_stdout_chars: int = 1_000_000,
    ) -> BenchmarkChildResult:
        run_dir = stdout_path.parent
        run_dir.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("[train-model] done in 1.00s\n", encoding="utf-8")
        (run_dir / "example_run.json").write_text(
            json.dumps(
                {
                    "config": {"model": {"model_type": "range_aware", "compression_ratio": 0.05}},
                    "matched": {"MLQDS": {"range_point_f1": 0.4, "range_usefulness_score": 0.5}},
                }
            ),
            encoding="utf-8",
        )
        return BenchmarkChildResult(
            returncode=0,
            stdout="",
            stdout_truncated=False,
            timings={"phase_timings": [], "epoch_timings": [], "inference_step_timings": []},
            elapsed_seconds=1.0,
        )

    monkeypatch.setattr(benchmark_runner, "_run_capture_streaming", fake_run_capture_streaming)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_runner",
            "--results_dir",
            str(results_dir),
            "--run_id",
            "artifact-test",
            "--run_label",
            "unit",
            "--coverage_targets",
            "0.05,0.10",
            "--workloads",
            "range",
            "--train_csv_path",
            str(train_csv),
            "--validation_csv_path",
            str(validation_csv),
            "--eval_csv_path",
            str(eval_csv),
            "--no_cache_warmup",
        ],
    )

    benchmark_runner.main()

    artifact = json.loads((results_dir / "benchmark_report.json").read_text(encoding="utf-8"))
    assert artifact["family_root"] == str(family)
    assert "<function" not in artifact["family_root"]
    assert artifact["run_config"]["coverage_targets"] == [0.05, 0.10]
    assert [row["run_label"] for row in artifact["rows"]] == ["unit_c05", "unit_c10"]
    assert artifact["rows"][0]["command"][-2:] == ["--query_coverage", "0.05"]
    assert artifact["rows"][1]["command"][-2:] == ["--query_coverage", "0.1"]
    assert artifact["rows"][0]["train_csv_path"] == str(train_csv)
    assert artifact["rows"][0]["validation_csv_path"] == str(validation_csv)
    assert artifact["rows"][0]["eval_csv_path"] == str(eval_csv)
    assert artifact["rows"][0]["selected_cleaned_csv_file_count"] == 3


def test_resolve_data_sources_selects_three_cleaned_days(tmp_path) -> None:
    (tmp_path / "aisdk-2026-02-02_cleaned.csv").write_text("x\n", encoding="utf-8")
    (tmp_path / "aisdk-2026-02-03_cleaned.csv").write_text("x\n", encoding="utf-8")
    (tmp_path / "aisdk-2026-02-04_cleaned.csv").write_text("x\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("ignore\n", encoding="utf-8")
    args = argparse.Namespace(csv_path=str(tmp_path), train_csv_path=None, validation_csv_path=None, eval_csv_path=None)

    sources = _resolve_data_sources(args)

    assert sources.csv_path is None
    assert sources.train_csv_path == str(tmp_path / "aisdk-2026-02-02_cleaned.csv")
    assert sources.validation_csv_path == str(tmp_path / "aisdk-2026-02-03_cleaned.csv")
    assert sources.eval_csv_path == str(tmp_path / "aisdk-2026-02-04_cleaned.csv")
    assert sources.csv_sources == (sources.train_csv_path, sources.validation_csv_path, sources.eval_csv_path)


def test_resolve_data_sources_requires_paired_train_eval() -> None:
    args = argparse.Namespace(csv_path=None, train_csv_path="train.csv", validation_csv_path=None, eval_csv_path=None)

    with pytest.raises(ValueError, match="supplied together"):
        _resolve_data_sources(args)


def test_resolve_data_sources_rejects_duplicate_explicit_splits(tmp_path) -> None:
    day = tmp_path / "day.csv"
    day.write_text("x\n", encoding="utf-8")
    args = argparse.Namespace(
        csv_path=None,
        train_csv_path=str(day),
        validation_csv_path=None,
        eval_csv_path=str(day),
    )

    with pytest.raises(ValueError, match="must be distinct"):
        _resolve_data_sources(args)


def test_resolve_data_sources_accepts_multiple_train_csvs(tmp_path) -> None:
    train_a = tmp_path / "train_a.csv"
    train_b = tmp_path / "train_b.csv"
    validation = tmp_path / "validation.csv"
    eval_day = tmp_path / "eval.csv"
    for path in (train_a, train_b, validation, eval_day):
        path.write_text("x\n", encoding="utf-8")
    args = argparse.Namespace(
        csv_path=None,
        train_csv_path=f"{train_a},{train_b}",
        validation_csv_path=str(validation),
        eval_csv_path=str(eval_day),
    )

    sources = _resolve_data_sources(args)

    assert sources.train_csv_path == f"{train_a},{train_b}"
    assert sources.selected_cleaned_csv_files == (
        str(train_a),
        str(train_b),
        str(validation),
        str(eval_day),
    )
    assert sources.csv_sources == sources.selected_cleaned_csv_files


def test_resolve_data_sources_accepts_multi_validation_and_eval_csvs(tmp_path) -> None:
    train_a = tmp_path / "train_a.csv"
    train_b = tmp_path / "train_b.csv"
    validation_a = tmp_path / "validation_a.csv"
    validation_b = tmp_path / "validation_b.csv"
    eval_a = tmp_path / "eval_a.csv"
    eval_b = tmp_path / "eval_b.csv"
    for path in (train_a, train_b, validation_a, validation_b, eval_a, eval_b):
        path.write_text("x\n", encoding="utf-8")
    args = argparse.Namespace(
        csv_path=None,
        train_csv_path=f"{train_a},{train_b}",
        validation_csv_path=f"{validation_a},{validation_b}",
        eval_csv_path=f"{eval_a},{eval_b}",
    )

    sources = _resolve_data_sources(args)

    assert sources.train_csv_path == f"{train_a},{train_b}"
    assert sources.validation_csv_path == f"{validation_a},{validation_b}"
    assert sources.eval_csv_path == f"{eval_a},{eval_b}"
    assert sources.selected_cleaned_csv_files == (
        str(train_a),
        str(train_b),
        str(validation_a),
        str(validation_b),
        str(eval_a),
        str(eval_b),
    )
    assert sources.csv_sources == sources.selected_cleaned_csv_files


def test_resolve_data_sources_rejects_duplicate_multi_train_csv(tmp_path) -> None:
    train = tmp_path / "train.csv"
    eval_day = tmp_path / "eval.csv"
    train.write_text("x\n", encoding="utf-8")
    eval_day.write_text("x\n", encoding="utf-8")
    args = argparse.Namespace(
        csv_path=None,
        train_csv_path=f"{train},{train}",
        validation_csv_path=None,
        eval_csv_path=str(eval_day),
    )

    with pytest.raises(ValueError, match="must be distinct"):
        _resolve_data_sources(args)


def test_resolve_data_sources_rejects_duplicate_multi_eval_csv(tmp_path) -> None:
    train = tmp_path / "train.csv"
    eval_day = tmp_path / "eval.csv"
    train.write_text("x\n", encoding="utf-8")
    eval_day.write_text("x\n", encoding="utf-8")
    args = argparse.Namespace(
        csv_path=None,
        train_csv_path=str(train),
        validation_csv_path=None,
        eval_csv_path=f"{eval_day},{eval_day}",
    )

    with pytest.raises(ValueError, match="must be distinct"):
        _resolve_data_sources(args)


def test_benchmark_markdown_table_is_compact() -> None:
    table = _format_report_table(
        [
            {
                "workload": "range",
                "run_label": "custom",
                "returncode": 0,
                "elapsed_seconds": 12.34567,
                "epoch_mean_seconds": 1.25,
                "peak_allocated_mb": 123.0,
                "best_selection_score": 0.5,
                "single_cell_range_status": "fails_uniform",
                "audit_low_beats_uniform_range_usefulness_count": 0,
                "worst_uniform_component_delta_metric": "mlqds_vs_uniform_range_gap_coverage",
                "runtime_bottleneck_phase": "train-model",
                "eval_query_extra_after_target_reached": 100,
                "eval_query_floor_dominated": True,
                "mlqds_primary_metric": "range_usefulness",
                "mlqds_primary_score": 0.41,
                "mlqds_aggregate_f1": 0.4,
                "mlqds_range_point_f1": 0.4,
                "mlqds_range_usefulness": 0.41,
                "uniform_range_point_f1": 0.3,
                "uniform_range_usefulness": 0.32,
                "douglas_peucker_range_point_f1": 0.2,
                "douglas_peucker_range_usefulness": 0.22,
                "mlqds_vs_uniform_range_point_f1": 0.1,
                "mlqds_vs_uniform_range_usefulness": 0.09,
                "mlqds_vs_douglas_peucker_range_point_f1": 0.2,
                "mlqds_vs_douglas_peucker_range_usefulness": 0.19,
                "mlqds_latency_ms": 10.0,
                "collapse_warning": False,
            }
        ]
    )

    assert "| workload | run_label |" in table
    assert "train_label_mass_range_point_f1" in table
    assert "single_cell_range_status" in table
    assert "audit_low_beats_uniform_range_usefulness_count" in table
    assert "runtime_bottleneck_phase" in table
    assert "eval_query_extra_after_target_reached" in table
    assert "eval_query_floor_dominated" in table
    assert "mlqds_avg_sed_km" in table
    assert "mlqds_primary_score" in table
    assert "mlqds_range_usefulness" in table
    assert "| range | custom | 0 | 12.3457 |" in table


def test_run_capture_streaming_writes_log_and_console(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    stdout_path = tmp_path / "child" / "stdout.log"

    result = _run_capture_streaming(
        [sys.executable, "-c", "print('alpha', flush=True); print('beta', flush=True)"],
        cwd=tmp_path,
        stdout_path=stdout_path,
    )

    assert result.returncode == 0
    assert result.stdout == "alpha\nbeta\n"
    assert stdout_path.read_text(encoding="utf-8") == "alpha\nbeta\n"
    assert "alpha\nbeta\n" in capsys.readouterr().out


def test_run_capture_streaming_retains_bounded_tail_but_keeps_timings(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    stdout_path = tmp_path / "child" / "stdout.log"
    command = (
        "print('[train-model] done in 1.23s', flush=True)\n"
        "for i in range(20):\n"
        "    print(f'filler-{i:03d}-' + 'x' * 40, flush=True)\n"
    )

    result = _run_capture_streaming(
        [sys.executable, "-c", command],
        cwd=tmp_path,
        stdout_path=stdout_path,
        max_stdout_chars=64,
    )

    full_log = stdout_path.read_text(encoding="utf-8")
    assert result.returncode == 0
    assert result.stdout_truncated is True
    assert len(result.stdout) <= 64
    assert full_log.startswith("[train-model] done in 1.23s\n")
    assert len(full_log) > len(result.stdout)
    assert result.timings["phase_timings"] == [{"name": "train-model", "seconds": 1.23}]
    assert "[train-model] done in 1.23s\n" in capsys.readouterr().out


def test_mark_benchmark_failed_updates_stale_running_status_and_family_index(tmp_path) -> None:
    family = tmp_path / "range_family"
    results_dir = family / "runs" / "stale-run"
    results_dir.mkdir(parents=True)
    status_file = results_dir / "run_status.json"
    status_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "stale-run",
                "status": "running",
                "started_at_utc": "2026-05-10T00:00:00+00:00",
                "finished_at_utc": None,
                "exit_status": None,
                "failures": None,
                "message": "benchmark run started",
                "results_dir": str(results_dir),
            }
        ),
        encoding="utf-8",
    )
    with open(family / "runs_index.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run_id", "status", "finished_at_utc", "exit_status", "failures", "results_dir"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "run_id": "stale-run",
                "status": "running",
                "finished_at_utc": "",
                "exit_status": "",
                "failures": "",
                "results_dir": str(results_dir),
            }
        )

    script = Path(__file__).resolve().parents[1] / "scripts" / "mark_benchmark_failed.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--status-file",
            str(status_file),
            "--exit-status",
            "-9",
            "--message",
            "killed",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(status_file.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["exit_status"] == -9
    assert payload["failures"] == 1
    assert payload["message"] == "killed"

    with open(family / "runs_index.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["status"] == "failed"
    assert rows[0]["exit_status"] == "-9"
    assert rows[0]["failures"] == "1"
    assert '"run_id": "stale-run"' in (family / "runs_index_events.jsonl").read_text(encoding="utf-8")
