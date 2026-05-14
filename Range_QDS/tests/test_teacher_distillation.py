"""Tests for range teacher-student distillation targets."""

from __future__ import annotations

import torch
import pytest

from experiments.experiment_config import ModelConfig
from models.trajectory_qds_model import TrajectoryQDSModel
from queries.query_types import pad_query_features
from queries.workload import TypedQueryWorkload
from training.scaler import FeatureScaler
from training.teacher_distillation import build_range_teacher_config, distill_range_teacher_labels
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
from training.training_outputs import TrainingOutputs


def _toy_points() -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 55.00, 12.00, 1.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 55.01, 12.01, 1.0, 5.0, 0.0, 0.0, 0.1],
            [2.0, 55.02, 12.02, 1.0, 10.0, 0.0, 0.0, 0.2],
            [3.0, 55.03, 12.03, 1.0, 15.0, 0.0, 0.0, 0.3],
            [4.0, 55.04, 12.04, 1.0, 20.0, 0.0, 0.0, 0.2],
            [5.0, 55.05, 12.05, 1.0, 25.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )


def _toy_workload() -> TypedQueryWorkload:
    queries = [
        {
            "type": "range",
            "params": {
                "lat_min": 54.99,
                "lat_max": 55.06,
                "lon_min": 11.99,
                "lon_max": 12.06,
                "t_start": 0.0,
                "t_end": 6.0,
            },
        }
    ]
    query_features, type_ids = pad_query_features(queries)
    return TypedQueryWorkload(query_features=query_features, typed_queries=queries, type_ids=type_ids)


def _diag_number(diagnostics: dict[str, object], key: str) -> float:
    value = diagnostics[key]
    assert isinstance(value, (int, float))
    return float(value)


def _diag_dict(diagnostics: dict[str, object], key: str) -> dict[str, object]:
    value = diagnostics[key]
    assert isinstance(value, dict)
    return value


def test_build_range_teacher_config_uses_query_aware_loss_selected_teacher() -> None:
    student = ModelConfig(
        model_type="workload_blind_range",
        range_teacher_distillation_mode="retained_frequency",
        range_teacher_epochs=3,
        checkpoint_selection_metric="uniform_gap",
        mlqds_range_geometry_blend=0.4,
    )

    teacher = build_range_teacher_config(student)

    assert teacher.model_type == "range_aware"
    assert teacher.epochs == 3
    assert teacher.checkpoint_selection_metric == "loss"
    assert teacher.validation_score_every == 0
    assert teacher.mlqds_range_geometry_blend == 0.0


def test_distill_range_teacher_labels_retained_frequency_shape_and_bounds() -> None:
    torch.manual_seed(7)
    points = _toy_points()
    workload = _toy_workload()
    model = TrajectoryQDSModel(
        point_dim=7,
        query_dim=workload.query_features.shape[1],
        embed_dim=16,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
    )
    scaler = FeatureScaler.fit(points[:, :7], workload.query_features)
    teacher = TrainingOutputs(
        model=model,
        scaler=scaler,
        labels=torch.zeros((points.shape[0], 1), dtype=torch.float32),
        labelled_mask=torch.zeros((points.shape[0], 1), dtype=torch.bool),
        history=[],
        epochs_trained=2,
    )
    config = ModelConfig(
        range_teacher_distillation_mode="retained_frequency",
        budget_loss_ratios=[0.33, 0.50],
        window_length=8,
        window_stride=4,
        inference_batch_size=2,
        mlqds_temporal_fraction=0.0,
    )

    (labels, labelled_mask), diagnostics = distill_range_teacher_labels(
        teacher=teacher,
        teacher_model_type="baseline",
        points=points,
        boundaries=[(0, points.shape[0])],
        workload=workload,
        model_config=config,
    )

    assert labels.shape == (points.shape[0], 1)
    assert labelled_mask.shape == labels.shape
    assert labelled_mask[:, 0].all()
    assert float(labels.min().item()) >= 0.0
    assert float(labels.max().item()) <= 1.0
    assert float(labels[:, 0].sum().item()) > 0.0
    assert diagnostics["enabled"] is True
    assert diagnostics["mode"] == "retained_frequency"
    assert diagnostics["teacher_epochs_trained"] == 2


def test_range_retained_frequency_training_labels_builds_budget_frequency_target() -> None:
    points = _toy_points()
    labels = torch.zeros((points.shape[0], 1), dtype=torch.float32)
    labelled_mask = torch.ones_like(labels, dtype=torch.bool)
    labels[:, 0] = torch.tensor([0.1, 0.2, 0.9, 0.8, 0.3, 0.4])
    config = ModelConfig(
        budget_loss_ratios=[0.33, 0.50],
        mlqds_temporal_fraction=0.0,
    )

    transformed, transformed_mask, diagnostics = range_retained_frequency_training_labels(
        labels=labels,
        labelled_mask=labelled_mask,
        boundaries=[(0, points.shape[0])],
        model_config=config,
    )

    assert transformed.shape == labels.shape
    assert transformed_mask[:, 0].all()
    assert diagnostics["mode"] == "retained_frequency"
    assert diagnostics["positive_label_count"] == 3
    assert torch.allclose(transformed[:, 0], torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.5]))


def test_retained_frequency_can_weight_low_budget_targets() -> None:
    points = _toy_points()
    labels = torch.zeros((points.shape[0], 1), dtype=torch.float32)
    labelled_mask = torch.ones_like(labels, dtype=torch.bool)
    labels[:, 0] = torch.tensor([0.1, 0.2, 0.9, 0.8, 0.3, 0.4])
    config = ModelConfig(
        budget_loss_ratios=[0.25, 0.50],
        range_target_budget_weight_power=1.0,
        mlqds_temporal_fraction=0.0,
    )

    transformed, _transformed_mask, diagnostics = range_retained_frequency_training_labels(
        labels=labels,
        labelled_mask=labelled_mask,
        boundaries=[(0, points.shape[0])],
        model_config=config,
    )

    assert diagnostics["budget_weight_power"] == 1.0
    assert diagnostics["budget_weights"] == pytest.approx([2.0 / 3.0, 1.0 / 3.0])
    assert torch.allclose(
        transformed[:, 0],
        torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 1.0 / 3.0]),
    )


def test_global_budget_retained_frequency_uses_database_level_competition() -> None:
    labels = torch.zeros((8, 1), dtype=torch.float32)
    labelled_mask = torch.ones_like(labels, dtype=torch.bool)
    labels[:, 0] = torch.tensor([0.1, 0.2, 0.3, 0.0, 0.0, 0.8, 0.7, 0.0])
    config = ModelConfig(
        budget_loss_ratios=[0.75],
        mlqds_temporal_fraction=0.0,
    )

    transformed, transformed_mask, diagnostics = range_global_budget_retained_frequency_training_labels(
        labels=labels,
        labelled_mask=labelled_mask,
        boundaries=[(0, 4), (4, 8)],
        model_config=config,
    )

    assert transformed_mask[:, 0].all()
    assert diagnostics["mode"] == "global_budget_retained_frequency"
    assert diagnostics["positive_label_count"] == 3
    assert diagnostics["global_budget_frequency_budget_count"] == 1
    assert torch.allclose(
        transformed[:, 0],
        torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0]),
    )


def test_global_budget_retained_frequency_aggregates_replicates_after_selection() -> None:
    labelled_mask = torch.ones((8, 1), dtype=torch.bool)
    first = torch.zeros((8, 1), dtype=torch.float32)
    second = torch.zeros((8, 1), dtype=torch.float32)
    first[:, 0] = torch.tensor([0.1, 0.2, 0.3, 0.0, 0.0, 0.8, 0.7, 0.0])
    second[:, 0] = torch.tensor([0.0, 0.1, 0.9, 0.0, 0.0, 0.2, 0.7, 0.0])
    config = ModelConfig(
        budget_loss_ratios=[0.75],
        mlqds_temporal_fraction=0.0,
    )

    transformed, transformed_mask, diagnostics = aggregate_range_global_budget_retained_frequency_training_labels(
        label_sets=[(first, labelled_mask), (second, labelled_mask)],
        boundaries=[(0, 4), (4, 8)],
        model_config=config,
    )

    assert transformed_mask[:, 0].all()
    assert diagnostics["mode"] == "global_budget_retained_frequency"
    assert diagnostics["replicate_count"] == 2
    assert torch.allclose(
        transformed[:, 0],
        torch.tensor([0.5, 0.0, 0.5, 0.0, 0.0, 0.5, 1.0, 0.0]),
    )


def test_retained_frequency_can_blend_temporal_anchor_target() -> None:
    points = _toy_points()
    labels = torch.zeros((points.shape[0], 1), dtype=torch.float32)
    labelled_mask = torch.ones_like(labels, dtype=torch.bool)
    labels[:, 0] = torch.tensor([0.1, 0.2, 0.9, 0.8, 0.7, 0.3])
    config = ModelConfig(
        budget_loss_ratios=[0.50],
        mlqds_temporal_fraction=0.0,
        range_temporal_target_blend=0.5,
    )

    transformed, _transformed_mask, diagnostics = range_retained_frequency_training_labels(
        labels=labels,
        labelled_mask=labelled_mask,
        boundaries=[(0, points.shape[0])],
        model_config=config,
    )

    assert diagnostics["temporal_target_blend"] == 0.5
    assert diagnostics["temporal_target_positive_label_count"] == 3
    assert torch.allclose(transformed[:, 0], torch.tensor([0.5, 0.0, 1.0, 0.5, 0.5, 0.5]))


def test_structural_retained_frequency_blends_query_free_shape_scores() -> None:
    points = _toy_points()
    bent = points.clone()
    bent[2, 1] += 0.08
    labels = torch.zeros((bent.shape[0], 1), dtype=torch.float32)
    labelled_mask = torch.ones_like(labels, dtype=torch.bool)
    labels[:, 0] = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    config = ModelConfig(
        budget_loss_ratios=[0.50],
        range_structural_target_blend=1.0,
        mlqds_temporal_fraction=0.0,
    )

    transformed, transformed_mask, diagnostics = range_structural_retained_frequency_training_labels(
        labels=labels,
        labelled_mask=labelled_mask,
        points=bent,
        boundaries=[(0, bent.shape[0])],
        model_config=config,
    )

    assert transformed_mask[:, 0].all()
    assert diagnostics["mode"] == "structural_retained_frequency"
    assert diagnostics["structural_target_blend"] == 1.0
    assert _diag_number(diagnostics, "structural_score_positive_mass") > 0.0
    assert transformed[-1, 0] > 0.0
    assert transformed[2, 0] > 0.0
    assert transformed[3, 0] > 0.0


def test_structural_retained_frequency_boost_preserves_label_support() -> None:
    points = _toy_points()
    bent = points.clone()
    bent[2, 1] += 0.08
    labels = torch.zeros((bent.shape[0], 1), dtype=torch.float32)
    labelled_mask = torch.ones_like(labels, dtype=torch.bool)
    labels[:, 0] = torch.tensor([0.9, 0.0, 0.0, 0.0, 0.0, 0.8])
    config = ModelConfig(
        budget_loss_ratios=[0.50],
        range_structural_target_blend=1.0,
        range_structural_target_source_mode="boost",
        mlqds_temporal_fraction=0.0,
    )

    transformed, _transformed_mask, diagnostics = range_structural_retained_frequency_training_labels(
        labels=labels,
        labelled_mask=labelled_mask,
        points=bent,
        boundaries=[(0, bent.shape[0])],
        model_config=config,
    )

    assert diagnostics["structural_target_source_mode"] == "boost"
    assert diagnostics["source_positive_label_count"] == 2
    assert transformed[0, 0] > 0.0
    assert transformed[-1, 0] > 0.0
    assert transformed[2, 0].item() == 0.0


def test_aggregate_structural_retained_frequency_averages_replicates() -> None:
    points = _toy_points()
    labels_a = torch.zeros((points.shape[0], 1), dtype=torch.float32)
    labels_b = torch.zeros((points.shape[0], 1), dtype=torch.float32)
    mask = torch.ones_like(labels_a, dtype=torch.bool)
    labels_a[:, 0] = torch.tensor([0.9, 0.1, 0.1, 0.1, 0.1, 0.1])
    labels_b[:, 0] = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.9])
    config = ModelConfig(
        budget_loss_ratios=[0.33],
        range_structural_target_blend=0.5,
        mlqds_temporal_fraction=0.0,
    )

    transformed, transformed_mask, diagnostics = aggregate_range_structural_retained_frequency_training_labels(
        label_sets=[(labels_a, mask), (labels_b, mask)],
        points=points,
        boundaries=[(0, points.shape[0])],
        model_config=config,
    )

    assert transformed_mask[:, 0].all()
    assert diagnostics["mode"] == "structural_retained_frequency"
    assert diagnostics["replicate_count"] == 2
    assert diagnostics["structural_target_blend"] == 0.5
    assert transformed[:, 0].sum().item() > 0.0


def test_historical_prior_retained_frequency_distills_sparse_teacher_target() -> None:
    points = _toy_points()
    labels = torch.zeros((points.shape[0], 1), dtype=torch.float32)
    labelled_mask = torch.ones_like(labels, dtype=torch.bool)
    labels[:, 0] = torch.tensor([0.1, 0.2, 0.9, 0.8, 0.3, 0.4])
    config = ModelConfig(
        budget_loss_ratios=[0.50],
        mlqds_temporal_fraction=0.0,
        historical_prior_k=2,
    )

    transformed, transformed_mask, diagnostics = range_historical_prior_retained_frequency_training_labels(
        labels=labels,
        labelled_mask=labelled_mask,
        points=points,
        boundaries=[(0, points.shape[0])],
        model_config=config,
    )

    assert transformed.shape == labels.shape
    assert transformed_mask[:, 0].all()
    assert diagnostics["mode"] == "historical_prior_retained_frequency"
    assert diagnostics["historical_prior_teacher_leave_one_out"] is True
    assert diagnostics["historical_prior_stored_support_count"] == points.shape[0]
    assert diagnostics["base_retained_frequency_positive_label_count"] == 3
    assert diagnostics["positive_label_count"] == 3
    assert transformed[:, 0].sum().item() == pytest.approx(3.0)


def test_balance_range_training_target_by_trajectory_equalizes_positive_mass() -> None:
    labels = torch.zeros((6, 1), dtype=torch.float32)
    labels[:, 0] = torch.tensor([0.2, 0.6, 0.0, 0.1, 0.1, 0.0])
    labelled_mask = torch.ones_like(labels, dtype=torch.bool)

    balanced, balanced_mask, diagnostics = balance_range_training_target_by_trajectory(
        labels=labels,
        labelled_mask=labelled_mask,
        boundaries=[(0, 3), (3, 6)],
        mode="trajectory_unit_mass",
    )

    assert balanced_mask.equal(labelled_mask)
    assert diagnostics["enabled"] is True
    assert diagnostics["balanced_trajectory_count"] == 2
    assert diagnostics["positive_label_mass_before_balance"] == pytest.approx(1.0)
    assert diagnostics["positive_label_mass"] == pytest.approx(2.0)
    assert torch.allclose(balanced[:3, 0], torch.tensor([0.25, 0.75, 0.0]))
    assert torch.allclose(balanced[3:, 0], torch.tensor([0.5, 0.5, 0.0]))


def test_marginal_coverage_target_spreads_redundant_hotspots() -> None:
    labels = torch.zeros((8, 1), dtype=torch.float32)
    labelled_mask = torch.ones_like(labels, dtype=torch.bool)
    labels[:, 0] = torch.tensor([0.0, 0.9, 1.0, 0.8, 0.0, 0.0, 0.7, 0.6])
    config = ModelConfig(
        budget_loss_ratios=[0.25],
        mlqds_temporal_fraction=0.0,
        range_marginal_target_radius_scale=0.25,
    )

    transformed, transformed_mask, diagnostics = range_marginal_coverage_training_labels(
        labels=labels,
        labelled_mask=labelled_mask,
        boundaries=[(0, 8)],
        model_config=config,
    )

    assert transformed_mask[:, 0].all()
    assert diagnostics["mode"] == "marginal_coverage_frequency"
    assert diagnostics["marginal_target_radius_scale"] == 0.25
    assert transformed[2, 0] == 1.0
    assert transformed[6, 0] == 1.0
    assert float(transformed[:, 0].sum().item()) == 2.0


def test_aggregate_marginal_coverage_labels_averages_workload_targets() -> None:
    labels_a = torch.zeros((8, 1), dtype=torch.float32)
    labels_b = torch.zeros((8, 1), dtype=torch.float32)
    labels_a[:, 0] = torch.tensor([0.0, 0.9, 1.0, 0.8, 0.0, 0.0, 0.7, 0.6])
    labels_b[:, 0] = torch.tensor([0.8, 0.7, 0.0, 0.0, 0.0, 0.9, 1.0, 0.8])
    mask = torch.ones_like(labels_a, dtype=torch.bool)
    config = ModelConfig(
        budget_loss_ratios=[0.25],
        mlqds_temporal_fraction=0.0,
        range_marginal_target_radius_scale=0.25,
    )

    labels, labelled_mask, diagnostics = aggregate_range_marginal_coverage_training_labels(
        label_sets=[(labels_a, mask), (labels_b, mask)],
        boundaries=[(0, 8)],
        model_config=config,
    )

    assert labelled_mask[:, 0].all()
    assert diagnostics["mode"] == "marginal_coverage_frequency"
    assert diagnostics["replicate_count"] == 2
    assert labels[6, 0] == 1.0
    assert float(labels[:, 0].sum().item()) == 2.0


def test_aggregate_range_label_sets_averages_training_workloads() -> None:
    labels_a = torch.tensor([[1.0], [0.0], [0.0], [0.0]])
    labels_b = torch.tensor([[0.0], [0.0], [0.5], [0.0]])
    mask = torch.ones_like(labels_a, dtype=torch.bool)

    labels, labelled_mask, diagnostics = aggregate_range_label_sets(
        [(labels_a, mask), (labels_b, mask)],
    )

    assert labelled_mask[:, 0].all()
    assert diagnostics["replicate_count"] == 2
    assert torch.allclose(labels[:, 0], torch.tensor([0.5, 0.0, 0.25, 0.0]))


def test_aggregate_range_label_sets_can_take_max_training_workload_signal() -> None:
    labels_a = torch.tensor([[1.0], [0.2], [0.0], [0.0]])
    labels_b = torch.tensor([[0.0], [0.6], [0.5], [0.0]])
    mask = torch.ones_like(labels_a, dtype=torch.bool)

    labels, labelled_mask, diagnostics = aggregate_range_label_sets(
        [(labels_a, mask), (labels_b, mask)],
        aggregation="max",
    )

    assert labelled_mask[:, 0].all()
    assert diagnostics["aggregation"] == "max"
    assert diagnostics["replicate_count"] == 2
    assert torch.allclose(labels[:, 0], torch.tensor([1.0, 0.6, 0.5, 0.0]))


def test_aggregate_retained_frequency_labels_averages_workload_topk_frequency() -> None:
    labels_a = torch.zeros((4, 1), dtype=torch.float32)
    labels_b = torch.zeros((4, 1), dtype=torch.float32)
    labels_a[:, 0] = torch.tensor([1.0, 0.9, 0.0, 0.0])
    labels_b[:, 0] = torch.tensor([0.0, 0.0, 0.8, 0.7])
    mask = torch.ones_like(labels_a, dtype=torch.bool)
    config = ModelConfig(
        budget_loss_ratios=[0.50],
        mlqds_temporal_fraction=0.0,
    )

    labels, labelled_mask, diagnostics = aggregate_range_retained_frequency_training_labels(
        label_sets=[(labels_a, mask), (labels_b, mask)],
        boundaries=[(0, 4)],
        model_config=config,
    )

    assert labelled_mask[:, 0].all()
    assert diagnostics["mode"] == "retained_frequency"
    assert diagnostics["replicate_count"] == 2
    assert torch.allclose(labels[:, 0], torch.tensor([0.5, 0.5, 0.5, 0.5]))


def test_component_retained_frequency_keeps_component_specific_targets() -> None:
    labels = torch.zeros((6, 1), dtype=torch.float32)
    labelled_mask = torch.ones_like(labels, dtype=torch.bool)
    component_labels = {
        "range_point_f1": torch.tensor([[1.0], [0.8], [0.0], [0.0], [0.0], [0.0]]),
        "range_gap_coverage": torch.tensor([[0.0], [0.0], [0.0], [0.0], [0.9], [0.7]]),
    }
    config = ModelConfig(
        budget_loss_ratios=[0.33],
        mlqds_temporal_fraction=0.0,
    )

    transformed, transformed_mask, diagnostics = range_component_retained_frequency_training_labels(
        labels=labels,
        labelled_mask=labelled_mask,
        component_labels=component_labels,
        boundaries=[(0, 6)],
        model_config=config,
    )

    assert transformed_mask[:, 0].all()
    assert diagnostics["mode"] == "component_retained_frequency"
    assert transformed[0, 0] > 0.0
    assert transformed[1, 0] > 0.0
    assert transformed[4, 0] > 0.0
    assert transformed[5, 0] > 0.0
    assert torch.allclose(transformed[2:4, 0], torch.zeros((2,)))


def test_continuity_retained_frequency_ignores_point_only_components() -> None:
    labels = torch.zeros((6, 1), dtype=torch.float32)
    labelled_mask = torch.ones_like(labels, dtype=torch.bool)
    component_labels = {
        "range_point_f1": torch.tensor([[1.0], [0.9], [0.0], [0.0], [0.0], [0.0]]),
        "range_gap_coverage": torch.tensor([[0.0], [0.0], [0.0], [0.0], [0.9], [0.7]]),
        "range_entry_exit_f1": torch.tensor([[0.0], [0.0], [0.8], [0.0], [0.0], [0.0]]),
    }
    config = ModelConfig(
        budget_loss_ratios=[0.33],
        mlqds_temporal_fraction=0.0,
        range_component_target_blend=1.0,
    )

    transformed, transformed_mask, diagnostics = range_continuity_retained_frequency_training_labels(
        labels=labels,
        labelled_mask=labelled_mask,
        component_labels=component_labels,
        boundaries=[(0, 6)],
        model_config=config,
    )

    assert transformed_mask[:, 0].all()
    assert diagnostics["mode"] == "continuity_retained_frequency"
    assert "range_point_f1" not in _diag_dict(diagnostics, "component_diagnostics")
    assert transformed[4, 0] > 0.0
    assert transformed[5, 0] > 0.0
    assert transformed[0, 0].item() == 0.0
    assert transformed[1, 0].item() == 0.0


def _component_label_dict(values: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        name: values.clone()
        for name in (
            "range_point_f1",
            "range_ship_f1",
            "range_ship_coverage",
            "range_entry_exit_f1",
            "range_crossing_f1",
            "range_temporal_coverage",
            "range_gap_coverage",
            "range_turn_coverage",
            "range_shape_score",
        )
    }


def test_aggregate_component_label_sets_preserves_component_streams() -> None:
    labels_a = torch.tensor([[1.0], [0.2], [0.0], [0.0]])
    labels_b = torch.tensor([[0.0], [0.6], [0.5], [0.0]])
    mask = torch.ones_like(labels_a, dtype=torch.bool)
    component_a = _component_label_dict(torch.zeros_like(labels_a))
    component_b = _component_label_dict(torch.zeros_like(labels_a))
    component_a["range_crossing_f1"][:, 0] = torch.tensor([0.1, 0.0, 0.0, 0.0])
    component_b["range_crossing_f1"][:, 0] = torch.tensor([0.0, 0.0, 0.7, 0.0])

    labels, labelled_mask, components, diagnostics = aggregate_range_component_label_sets(
        [(labels_a, mask), (labels_b, mask)],
        [component_a, component_b],
        aggregation="max",
    )

    assert labelled_mask[:, 0].all()
    assert diagnostics["aggregation"] == "max"
    assert torch.allclose(labels[:, 0], torch.tensor([1.0, 0.6, 0.5, 0.0]))
    assert torch.allclose(components["range_crossing_f1"][:, 0], torch.tensor([0.1, 0.0, 0.7, 0.0]))


def test_aggregate_component_retained_frequency_averages_replicate_targets() -> None:
    labels_a = torch.zeros((4, 1), dtype=torch.float32)
    labels_b = torch.zeros((4, 1), dtype=torch.float32)
    labels_a[:, 0] = torch.tensor([1.0, 0.9, 0.0, 0.0])
    labels_b[:, 0] = torch.tensor([0.0, 0.0, 0.8, 0.7])
    mask = torch.ones_like(labels_a, dtype=torch.bool)
    config = ModelConfig(
        budget_loss_ratios=[0.50],
        mlqds_temporal_fraction=0.0,
        range_component_target_blend=1.0,
    )

    labels, labelled_mask, diagnostics = aggregate_range_component_retained_frequency_training_labels(
        label_sets=[(labels_a, mask), (labels_b, mask)],
        component_label_sets=[_component_label_dict(labels_a), _component_label_dict(labels_b)],
        boundaries=[(0, 4)],
        model_config=config,
    )

    assert labelled_mask[:, 0].all()
    assert diagnostics["mode"] == "component_retained_frequency"
    assert diagnostics["replicate_count"] == 2
    assert torch.allclose(labels[:, 0], torch.tensor([0.5, 0.5, 0.5, 0.5]))


def test_aggregate_continuity_retained_frequency_averages_replicate_targets() -> None:
    labels_a = torch.zeros((4, 1), dtype=torch.float32)
    labels_b = torch.zeros((4, 1), dtype=torch.float32)
    mask = torch.ones_like(labels_a, dtype=torch.bool)
    component_a = _component_label_dict(torch.zeros_like(labels_a))
    component_b = _component_label_dict(torch.zeros_like(labels_b))
    component_a["range_gap_coverage"][:, 0] = torch.tensor([1.0, 0.9, 0.0, 0.0])
    component_b["range_temporal_coverage"][:, 0] = torch.tensor([0.0, 0.0, 0.8, 0.7])
    config = ModelConfig(
        budget_loss_ratios=[0.50],
        mlqds_temporal_fraction=0.0,
        range_component_target_blend=1.0,
    )

    labels, labelled_mask, diagnostics = aggregate_range_continuity_retained_frequency_training_labels(
        label_sets=[(labels_a, mask), (labels_b, mask)],
        component_label_sets=[component_a, component_b],
        boundaries=[(0, 4)],
        model_config=config,
    )

    assert labelled_mask[:, 0].all()
    assert diagnostics["mode"] == "continuity_retained_frequency"
    assert diagnostics["replicate_count"] == 2
    assert torch.allclose(labels[:, 0], torch.tensor([0.5, 0.5, 0.5, 0.5]))


def test_query_spine_frequency_builds_workload_blind_spine_target() -> None:
    points = _toy_points()
    workload = _toy_workload()
    labels = torch.zeros((points.shape[0], 1), dtype=torch.float32)
    labelled_mask = torch.zeros_like(labels, dtype=torch.bool)
    config = ModelConfig(
        budget_loss_ratios=[0.33, 0.50],
        mlqds_temporal_fraction=0.0,
        range_query_spine_fraction=0.34,
    )

    transformed, transformed_mask, diagnostics = range_query_spine_frequency_training_labels(
        labels=labels,
        labelled_mask=labelled_mask,
        points=points,
        boundaries=[(0, points.shape[0])],
        typed_queries=workload.typed_queries,
        model_config=config,
    )

    assert transformed.shape == labels.shape
    assert transformed_mask[:, 0].all()
    assert diagnostics["mode"] == "query_spine_frequency"
    assert diagnostics["source"] == "range_query_temporal_spines"
    assert diagnostics["query_spine_range_query_count"] == 1
    assert diagnostics["query_spine_hit_group_count"] == 1
    assert _diag_number(diagnostics, "query_spine_source_positive_count") >= 3
    assert _diag_number(diagnostics, "positive_label_count") >= 2
    assert float(transformed[:, 0].sum().item()) > 0.0
    assert float(transformed[:, 0].max().item()) <= 1.0


def test_query_spine_frequency_can_normalize_mass_per_query() -> None:
    points = _toy_points()
    workload = _toy_workload()
    labels = torch.zeros((points.shape[0], 1), dtype=torch.float32)
    labelled_mask = torch.zeros_like(labels, dtype=torch.bool)
    boundaries = [(0, 3), (3, 6)]

    hit_group_config = ModelConfig(
        budget_loss_ratios=[0.50],
        mlqds_temporal_fraction=0.0,
        range_query_spine_fraction=0.67,
        range_query_spine_mass_mode="hit_group",
    )
    query_config = ModelConfig(
        budget_loss_ratios=[0.50],
        mlqds_temporal_fraction=0.0,
        range_query_spine_fraction=0.67,
        range_query_spine_mass_mode="query",
    )

    _hit_labels, _hit_mask, hit_diagnostics = range_query_spine_frequency_training_labels(
        labels=labels,
        labelled_mask=labelled_mask,
        points=points,
        boundaries=boundaries,
        typed_queries=workload.typed_queries,
        model_config=hit_group_config,
    )
    _query_labels, _query_mask, query_diagnostics = range_query_spine_frequency_training_labels(
        labels=labels,
        labelled_mask=labelled_mask,
        points=points,
        boundaries=boundaries,
        typed_queries=workload.typed_queries,
        model_config=query_config,
    )

    assert hit_diagnostics["query_spine_mass_mode"] == "hit_group"
    assert query_diagnostics["query_spine_mass_mode"] == "query"
    assert hit_diagnostics["query_spine_hit_group_count"] == 2
    assert query_diagnostics["query_spine_hit_group_count"] == 2
    assert hit_diagnostics["query_spine_source_positive_mass"] == pytest.approx(2.0)
    assert query_diagnostics["query_spine_source_positive_mass"] == pytest.approx(1.0)


def test_query_residual_frequency_labels_train_query_fill_anchors_only() -> None:
    points = _toy_points()
    workload = _toy_workload()
    labels = torch.zeros((points.shape[0], 1), dtype=torch.float32)
    labelled_mask = torch.zeros_like(labels, dtype=torch.bool)
    config = ModelConfig(
        budget_loss_ratios=[0.50],
        mlqds_temporal_fraction=0.50,
        range_query_residual_multiplier=1.0,
    )

    transformed, transformed_mask, diagnostics = range_query_residual_frequency_training_labels(
        labels=labels,
        labelled_mask=labelled_mask,
        points=points,
        boundaries=[(0, points.shape[0])],
        typed_queries=workload.typed_queries,
        model_config=config,
    )

    assert transformed.shape == labels.shape
    assert transformed_mask[:, 0].all()
    assert diagnostics["mode"] == "query_residual_frequency"
    assert diagnostics["source"] == "range_query_residual_anchors"
    assert diagnostics["query_residual_range_query_count"] == 1
    assert diagnostics["query_residual_used_budget_count"] == 1
    assert diagnostics["query_residual_mass_mode"] == "query"
    assert _diag_number(diagnostics, "query_residual_selected_residual_count") > 0
    assert 0 < _diag_number(diagnostics, "positive_label_count") < points.shape[0]
    assert float(transformed[:, 0].sum().item()) > 0.0
    assert float(transformed[:, 0].max().item()) <= 1.0


def test_query_residual_point_mass_mode_keeps_anchor_frequency_mass() -> None:
    points = _toy_points()
    workload = _toy_workload()
    labels = torch.zeros((points.shape[0], 1), dtype=torch.float32)
    labelled_mask = torch.zeros_like(labels, dtype=torch.bool)
    config = ModelConfig(
        budget_loss_ratios=[0.50],
        mlqds_temporal_fraction=0.50,
        range_query_residual_multiplier=1.0,
        range_query_residual_mass_mode="point",
    )

    transformed, _transformed_mask, diagnostics = range_query_residual_frequency_training_labels(
        labels=labels,
        labelled_mask=labelled_mask,
        points=points,
        boundaries=[(0, points.shape[0])],
        typed_queries=workload.typed_queries,
        model_config=config,
    )

    assert diagnostics["query_residual_mass_mode"] == "point"
    assert float(transformed[:, 0].sum().item()) == pytest.approx(
        _diag_number(diagnostics, "query_residual_selected_residual_count")
    )


def test_set_utility_frequency_labels_marginal_range_usefulness_gain() -> None:
    points = _toy_points()
    workload = _toy_workload()
    labels = torch.zeros((points.shape[0], 1), dtype=torch.float32)
    labelled_mask = torch.zeros_like(labels, dtype=torch.bool)
    config = ModelConfig(
        budget_loss_ratios=[0.50],
        mlqds_temporal_fraction=0.50,
        range_set_utility_multiplier=1.0,
        range_set_utility_candidate_limit=0,
        range_set_utility_mass_mode="gain",
    )

    transformed, transformed_mask, diagnostics = range_set_utility_frequency_training_labels(
        labels=labels,
        labelled_mask=labelled_mask,
        points=points,
        boundaries=[(0, points.shape[0])],
        typed_queries=workload.typed_queries,
        model_config=config,
    )

    assert transformed.shape == labels.shape
    assert transformed_mask[:, 0].all()
    assert diagnostics["mode"] == "set_utility_frequency"
    assert diagnostics["source"] == "range_train_query_marginal_usefulness_gain"
    assert diagnostics["set_utility_range_query_count"] == 1
    assert diagnostics["set_utility_used_budget_count"] == 1
    assert _diag_number(diagnostics, "set_utility_scored_candidate_count") > 0
    assert _diag_number(diagnostics, "set_utility_selected_count") > 0
    assert _diag_number(diagnostics, "set_utility_selected_gain_mass") > 0.0
    assert 0 < _diag_number(diagnostics, "positive_label_count") < points.shape[0]
    assert float(transformed[:, 0].sum().item()) > 0.0
    assert float(transformed[:, 0].max().item()) <= 1.0


def test_local_swap_utility_frequency_labels_replacement_gain() -> None:
    points = _toy_points()
    queries = [
        {
            "type": "range",
            "params": {
                "lat_min": 55.025,
                "lat_max": 55.055,
                "lon_min": 12.025,
                "lon_max": 12.055,
                "t_start": 2.5,
                "t_end": 5.5,
            },
        }
    ]
    query_features, type_ids = pad_query_features(queries)
    workload = TypedQueryWorkload(query_features=query_features, typed_queries=queries, type_ids=type_ids)
    labels = torch.zeros((points.shape[0], 1), dtype=torch.float32)
    labelled_mask = torch.zeros_like(labels, dtype=torch.bool)
    config = ModelConfig(
        budget_loss_ratios=[0.50],
        mlqds_temporal_fraction=0.50,
        mlqds_hybrid_mode="local_swap",
        range_set_utility_multiplier=1.0,
        range_set_utility_candidate_limit=0,
        range_set_utility_mass_mode="point",
    )

    transformed, transformed_mask, diagnostics = range_local_swap_utility_frequency_training_labels(
        labels=labels,
        labelled_mask=labelled_mask,
        points=points,
        boundaries=[(0, points.shape[0])],
        typed_queries=workload.typed_queries,
        model_config=config,
    )

    assert transformed.shape == labels.shape
    assert transformed_mask[:, 0].all()
    assert diagnostics["mode"] == "local_swap_utility_frequency"
    assert diagnostics["source"] == "range_train_query_local_swap_usefulness_gain"
    assert diagnostics["local_swap_utility_range_query_count"] == 1
    assert diagnostics["local_swap_utility_used_budget_count"] == 1
    assert _diag_number(diagnostics, "local_swap_utility_scored_candidate_count") > 0
    assert _diag_number(diagnostics, "local_swap_utility_positive_gain_candidate_count") > 0
    assert _diag_number(diagnostics, "local_swap_utility_selected_count") > 0
    assert _diag_number(diagnostics, "local_swap_utility_selected_gain_mass") > 0.0
    assert 0 < _diag_number(diagnostics, "positive_label_count") < points.shape[0]
    assert float(transformed[:, 0].sum().item()) > 0.0
    assert float(transformed[:, 0].max().item()) <= 1.0


def test_local_swap_gain_cost_frequency_labels_candidate_value_and_base_cost() -> None:
    points = _toy_points()
    queries = [
        {
            "type": "range",
            "params": {
                "lat_min": 55.025,
                "lat_max": 55.055,
                "lon_min": 12.025,
                "lon_max": 12.055,
                "t_start": 2.5,
                "t_end": 5.5,
            },
        }
    ]
    query_features, type_ids = pad_query_features(queries)
    workload = TypedQueryWorkload(query_features=query_features, typed_queries=queries, type_ids=type_ids)
    labels = torch.zeros((points.shape[0], 1), dtype=torch.float32)
    labelled_mask = torch.zeros_like(labels, dtype=torch.bool)
    config = ModelConfig(
        budget_loss_ratios=[0.50],
        mlqds_temporal_fraction=0.50,
        mlqds_hybrid_mode="local_delta_swap",
        range_set_utility_multiplier=1.0,
        range_set_utility_candidate_limit=0,
        range_set_utility_mass_mode="gain",
    )

    transformed, transformed_mask, diagnostics = range_local_swap_gain_cost_frequency_training_labels(
        labels=labels,
        labelled_mask=labelled_mask,
        points=points,
        boundaries=[(0, points.shape[0])],
        typed_queries=workload.typed_queries,
        model_config=config,
    )

    assert transformed.shape == labels.shape
    assert transformed_mask[:, 0].all()
    assert diagnostics["mode"] == "local_swap_gain_cost_frequency"
    assert diagnostics["source"] == "range_train_query_local_swap_candidate_value_and_removal_cost"
    assert diagnostics["local_swap_gain_cost_range_query_count"] == 1
    assert diagnostics["local_swap_gain_cost_used_budget_count"] == 1
    assert _diag_number(diagnostics, "local_swap_gain_cost_scored_candidate_count") > 0
    assert _diag_number(diagnostics, "local_swap_gain_cost_positive_net_gain_count") > 0
    assert _diag_number(diagnostics, "local_swap_gain_cost_selected_count") > 0
    assert _diag_number(diagnostics, "local_swap_gain_cost_selected_candidate_value_mass") > 0.0
    assert _diag_number(diagnostics, "local_swap_gain_cost_selected_removal_cost_mass") > 0.0
    assert transformed[2, 0] > 0.0
    assert torch.max(transformed[3:5, 0]) > transformed[2, 0]
    assert float(transformed[:, 0].max().item()) <= 1.0
