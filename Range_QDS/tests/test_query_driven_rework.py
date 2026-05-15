"""Focused tests for the query-driven Range-QDS rework path."""

from __future__ import annotations

import torch
from types import SimpleNamespace
from typing import Any, cast

from data.ais_loader import generate_synthetic_ais_data
from evaluation.evaluate_methods import score_range_usefulness
from evaluation.metrics import MethodEvaluation, compute_length_preservation
from evaluation.query_useful_v1 import query_useful_v1_from_range_audit
from experiments.experiment_pipeline import (
    _global_sanity_gate,
    _learned_slot_summary,
    _learning_causality_delta_gate_config,
    _neutral_segment_scores_for_ablation,
    _prior_feature_sample_sensitivity,
    _prior_sample_gate_failures,
    _score_ablation_sensitivity,
    _support_overlap_gate,
    _target_diffusion_gate,
    _workload_stability_gate,
)
from experiments.range_diagnostics import _range_workload_distribution_comparison
from models.workload_blind_range_v2 import WorkloadBlindRangeV2Model
from queries.query_generator import _anchor_weights_for_family, _make_range_query, generate_typed_query_workload
from queries.query_types import QUERY_TYPE_ID_RANGE
from queries.workload_profiles import range_workload_profile
from simplification.learned_segment_budget import (
    learned_segment_budget_diagnostics,
    simplify_with_learned_segment_budget_v1,
    simplify_with_learned_segment_budget_v1_with_trace,
)
from training.model_features import (
    WORKLOAD_BLIND_RANGE_V2_POINT_DIM,
    build_workload_blind_range_v2_point_features,
)
from training.query_prior_fields import (
    build_train_query_prior_fields,
    query_prior_field_metadata,
    sample_query_prior_fields,
    zero_query_prior_field_like,
)
from training.query_useful_targets import QUERY_USEFUL_V1_HEAD_NAMES, build_query_useful_v1_targets
from training.predictability_audit import query_prior_predictability_audit
from training.training_diagnostics import _training_target_diagnostics
from training.training_epoch import _segment_budget_head_segment_level_loss
from training.training_validation import _validation_query_useful_selection_score


def _boundaries(trajectories: list[torch.Tensor]) -> list[tuple[int, int]]:
    cursor = 0
    out = []
    for trajectory in trajectories:
        end = cursor + int(trajectory.shape[0])
        out.append((cursor, end))
        cursor = end
    return out


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
    assert audit["range_query_local_interpolation_fidelity"] > 0.99
    assert components["query_local_interpolation_fidelity"] > 0.99
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


def test_factorized_replacement_target_is_query_local_and_sparse() -> None:
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
    assert int((final_score > 0.0).sum().item()) == 5


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


def test_no_query_prior_ablation_preserves_train_extent() -> None:
    train_points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 10.0, 10.0]], dtype=torch.float32)
    eval_points = torch.tensor([[0.5, 5.0, 5.0], [0.75, 6.0, 6.0]], dtype=torch.float32)
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
    segment_scores = torch.sigmoid(head_logits.squeeze(0)[:, 4])
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
    assert head_logits.shape == (1, points.shape[0], 5)
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
    assert abs(float(model.prior_feature_scale.detach().item()) - 1.0) < 1e-6
    assert not torch.allclose(base_score, prior_score)
    assert not torch.allclose(base_heads, prior_heads)


def test_factorized_head_ablation_uses_neutral_multiplicative_heads() -> None:
    model = WorkloadBlindRangeV2Model(
        point_dim=WORKLOAD_BLIND_RANGE_V2_POINT_DIM,
        query_dim=12,
        embed_dim=32,
        num_heads=2,
        num_layers=0,
        dropout=0.0,
    )
    for parameter in model.calibration_head.parameters():
        parameter.data.zero_()
    head_logits = torch.zeros((1, 1, len(QUERY_USEFUL_V1_HEAD_NAMES)), dtype=torch.float32)
    head_logits[..., 1] = -10.0

    disabled = model.final_logit_from_head_logits(
        head_logits,
        disabled_head_names=("conditional_behavior_utility",),
    )
    expected_score = torch.tensor(0.375, dtype=torch.float32)

    assert torch.allclose(disabled.squeeze(), torch.logit(expected_score), atol=1e-6)


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
        "anchor_family_counts": {"density_route": 3, "boundary_entry_exit": 1},
        "footprint_family_counts": {"medium_operational": 4},
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
    assert "lift_at_5_percent" in audit["metrics"]
    assert "lift_at_5_percent" in audit["gate_checks"]


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


def test_workload_stability_gate_accepts_exhausted_stop_after_coverage_satisfied() -> None:
    config = SimpleNamespace(
        query=SimpleNamespace(target_coverage=0.10, range_max_coverage_overshoot=0.0075)
    )
    workload = SimpleNamespace(
        typed_queries=[{} for _ in range(12)],
        coverage_fraction=0.105,
        generation_diagnostics={
            "query_generation": {
                "mode": "target_coverage",
                "workload_profile_id": "range_workload_v1",
                "coverage_calibration_mode": "profile_sampled_query_count",
                "target_coverage": 0.10,
                "coverage_guard_enabled": True,
                "stop_reason": "range_acceptance_exhausted",
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
    assert gate["workloads"][0]["coverage_target_satisfied"] is True


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
