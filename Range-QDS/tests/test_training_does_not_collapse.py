"""Tests that short training keeps non-collapsed typed predictions. See training/README.md for details."""

from __future__ import annotations

import pytest
import torch

from data.ais_loader import generate_synthetic_ais_data
from data.trajectory_dataset import TrajectoryDataset
from evaluation.baselines import MLQDSMethod
from evaluation.evaluate_methods import score_range_usefulness, score_retained_mask
from experiments.experiment_config import build_experiment_config
from models.historical_prior_qds_model import HistoricalPriorRangeQDSModel
from models.trajectory_qds_model import TrajectoryQDSModel
from queries.query_generator import generate_typed_query_workload
from queries.query_types import NUM_QUERY_TYPES, QUERY_TYPE_ID_RANGE
from training.checkpoint_selection import (
    validation_score_selection_score as _validation_score_selection_score,
    selection_score as _selection_score,
    uniform_gap_selection_score as _uniform_gap_selection_score,
)
from training.importance_labels import compute_typed_importance_labels
from training.scaler import FeatureScaler
from training.train_model import train_model
from training.training_losses import (
    _balanced_pointwise_loss,
    _balanced_pointwise_loss_rows,
    _budget_topk_recall_loss,
    _budget_topk_recall_loss_rows,
    _budget_topk_temporal_residual_loss,
    _budget_topk_temporal_residual_loss_rows,
    _effective_budget_loss_ratios,
    _ranking_loss_for_type,
)
from training.training_diagnostics import train_target_fit_diagnostics
from training.training_outputs import TrainingOutputs
from training.training_setup import _single_active_type_id
from training.training_targets import _apply_temporal_residual_labels
from training.training_validation import _validation_query_score
from training.training_windows import _filter_supervised_windows
from training.trajectory_batching import build_trajectory_windows


def test_selection_score_penalizes_collapsed_predictions() -> None:
    """Assert model selection does not prefer collapsed output solely because tau is nonnegative."""
    assert _selection_score(avg_tau=0.0, pred_std=0.0) < _selection_score(avg_tau=-0.05, pred_std=0.01)


def test_temporal_residual_budget_ratios_match_learned_fill_budget() -> None:
    cfg = build_experiment_config(
        compression_ratio=0.05,
        budget_loss_ratios=[0.01, 0.02, 0.05, 0.10],
        mlqds_temporal_fraction=0.75,
    )

    ratios = _effective_budget_loss_ratios(cfg.model, "temporal")

    assert ratios == pytest.approx(
        (
            0.0025188917,
            0.0050761421,
            0.0129870130,
            0.0270270270,
        )
    )
    assert _effective_budget_loss_ratios(cfg.model, "none") == pytest.approx((0.01, 0.02, 0.05, 0.10))

    stratified_cfg = build_experiment_config(
        compression_ratio=0.05,
        budget_loss_ratios=[0.01, 0.02, 0.05, 0.10],
        mlqds_hybrid_mode="stratified",
        mlqds_temporal_fraction=0.75,
    )
    assert _effective_budget_loss_ratios(stratified_cfg.model, "temporal") == pytest.approx(
        (0.01, 0.02, 0.05, 0.10)
    )
    global_cfg = build_experiment_config(
        compression_ratio=0.05,
        budget_loss_ratios=[0.01, 0.02, 0.05, 0.10],
        mlqds_hybrid_mode="global_budget",
        mlqds_temporal_fraction=0.75,
    )
    assert _effective_budget_loss_ratios(global_cfg.model, "temporal") == pytest.approx(
        (0.01, 0.02, 0.05, 0.10)
    )


def test_selection_score_uses_loss_before_tau_proxy() -> None:
    """Assert checkpoint selection does not restore a worse-loss epoch solely from noisy tau."""
    proxy_best = _selection_score(avg_tau=0.9, pred_std=0.1, loss=0.20)
    lower_loss = _selection_score(avg_tau=-0.1, pred_std=0.1, loss=0.10)

    assert lower_loss > proxy_best


def test_train_target_fit_diagnostics_reports_budget_target_recall() -> None:
    """Assert train-target fit diagnostics compare learned masks to uniform masks."""
    target = torch.tensor([0.0, 0.1, 1.0, 0.9, 0.2, 0.0], dtype=torch.float32)
    predictions = target.clone()
    labelled_mask = torch.ones_like(target, dtype=torch.bool)
    cfg = build_experiment_config(
        compression_ratio=0.5,
        budget_loss_ratios=[0.5],
        workload="range",
        mlqds_hybrid_mode="fill",
        mlqds_temporal_fraction=0.0,
    )

    diagnostics = train_target_fit_diagnostics(
        predictions=predictions,
        target=target,
        labelled_mask=labelled_mask,
        boundaries=[(0, int(target.numel()))],
        model_config=cfg.model,
        workload_type="range",
        seed=12,
    )

    assert diagnostics["enabled"] is True
    assert diagnostics["score_target_kendall_tau"] > 0.9
    assert diagnostics["matched_mlqds_target_recall"] == pytest.approx(1.0)
    assert diagnostics["matched_mlqds_vs_uniform_target_recall"] > 0.0
    assert diagnostics["budget_rows"][0]["mlqds_target_mass"] > diagnostics["budget_rows"][0]["uniform_target_mass"]


def test_validation_score_selection_score_penalizes_collapsed_predictions() -> None:
    assert _validation_score_selection_score(
        validation_score=0.8,
        pred_std=0.0,
    ) < _validation_score_selection_score(validation_score=0.2, pred_std=0.01)


def test_uniform_gap_selection_penalizes_active_type_deficit() -> None:
    workload_map = {"range": 1.0}
    uniform_per_type = {"range": 0.50}
    range_deficit = _uniform_gap_selection_score(
        validation_score=0.55,
        per_type_score={"range": 0.45},
        uniform_score=0.50,
        uniform_per_type_score=uniform_per_type,
        workload_map=workload_map,
        pred_std=0.1,
    )
    balanced = _uniform_gap_selection_score(
        validation_score=0.54,
        per_type_score={"range": 0.54},
        uniform_score=0.50,
        uniform_per_type_score=uniform_per_type,
        workload_map=workload_map,
        pred_std=0.1,
    )

    assert balanced > range_deficit


def test_balanced_pointwise_loss_pushes_constant_scores_apart() -> None:
    """Assert the anti-collapse BCE term has useful gradients from constant predictions."""
    pred = torch.zeros((8,), requires_grad=True)
    target = torch.tensor([1.0, 0.8, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0])
    valid_mask = torch.ones((8,), dtype=torch.bool)

    loss = _balanced_pointwise_loss(
        pred=pred,
        target=target,
        valid_mask=valid_mask,
        generator=torch.Generator().manual_seed(123),
        negatives_per_positive=2,
    )
    loss.backward()

    assert float(loss.item()) > 0.0
    assert pred.grad is not None
    assert float(pred.grad[:3].sum().item()) < 0.0
    assert float(pred.grad[3:].sum().item()) > 0.0


def test_balanced_pointwise_loss_rows_matches_scalar_when_all_zeros_selected() -> None:
    pred = torch.tensor(
        [
            [0.0, 0.1, -0.2, 0.3],
            [0.2, -0.1, 0.4, -0.3],
        ],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [
            [1.0, 0.5, 0.0, 0.0],
            [0.0, 0.7, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    valid_mask = torch.ones_like(target, dtype=torch.bool)
    generator = torch.Generator().manual_seed(123)

    row_loss, active_rows = _balanced_pointwise_loss_rows(
        pred=pred,
        target=target,
        valid_mask=valid_mask,
        generator=generator,
        negatives_per_positive=8,
    )
    scalar0 = _balanced_pointwise_loss(
        pred=pred[0],
        target=target[0],
        valid_mask=valid_mask[0],
        generator=torch.Generator().manual_seed(123),
        negatives_per_positive=8,
    )
    scalar1 = _balanced_pointwise_loss(
        pred=pred[1],
        target=target[1],
        valid_mask=valid_mask[1],
        generator=torch.Generator().manual_seed(123),
        negatives_per_positive=8,
    )

    assert active_rows.tolist() == [True, True]
    assert torch.allclose(row_loss, torch.stack([scalar0, scalar1]))


def test_balanced_pointwise_loss_rows_has_useful_gradients() -> None:
    pred = torch.zeros((2, 6), dtype=torch.float32, requires_grad=True)
    target = torch.tensor(
        [
            [1.0, 0.8, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.7, 0.4, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    valid_mask = torch.ones_like(target, dtype=torch.bool)

    row_loss, active_rows = _balanced_pointwise_loss_rows(
        pred=pred,
        target=target,
        valid_mask=valid_mask,
        generator=torch.Generator().manual_seed(123),
        negatives_per_positive=2,
    )
    row_loss[active_rows].mean().backward()

    assert pred.grad is not None
    assert float(pred.grad[target > 0].sum().item()) < 0.0
    assert float(pred.grad[target == 0].sum().item()) > 0.0


def test_ranking_pair_sampler_returns_finite_loss() -> None:
    pred = torch.linspace(0.1, 0.8, steps=8)
    target = torch.tensor([1.0, 0.9, 0.7, 0.4, 0.2, 0.1, 0.0, 0.0])
    valid_mask = torch.ones((8,), dtype=torch.bool)

    loss, pair_count = _ranking_loss_for_type(
        pred=pred,
        target=target,
        valid_mask=valid_mask,
        pairs_per_type=16,
        top_quantile=0.5,
        margin=0.05,
        generator=torch.Generator().manual_seed(123),
    )

    assert 0 < pair_count <= 16
    assert bool(torch.isfinite(loss).item())


def test_budget_topk_recall_loss_prefers_budget_aligned_scores() -> None:
    target = torch.tensor([1.0, 0.9, 0.1, 0.0, 0.0, 0.0], dtype=torch.float32)
    valid_mask = torch.ones((6,), dtype=torch.bool)
    good_pred = torch.tensor([3.0, 2.0, 0.5, -1.0, -2.0, -3.0], dtype=torch.float32)
    bad_pred = torch.flip(good_pred, dims=(0,))

    good_loss = _budget_topk_recall_loss(
        pred=good_pred,
        target=target,
        valid_mask=valid_mask,
        budget_ratios=(0.20, 0.40),
        temperature=0.10,
    )
    bad_loss = _budget_topk_recall_loss(
        pred=bad_pred,
        target=target,
        valid_mask=valid_mask,
        budget_ratios=(0.20, 0.40),
        temperature=0.10,
    )

    assert bool(torch.isfinite(good_loss).item())
    assert good_loss.item() < bad_loss.item()


def test_budget_topk_recall_loss_has_useful_gradients() -> None:
    pred = torch.zeros((6,), dtype=torch.float32, requires_grad=True)
    target = torch.tensor([1.0, 0.8, 0.2, 0.0, 0.0, 0.0], dtype=torch.float32)
    valid_mask = torch.ones((6,), dtype=torch.bool)

    loss = _budget_topk_recall_loss(
        pred=pred,
        target=target,
        valid_mask=valid_mask,
        budget_ratios=(0.20, 0.40),
        temperature=0.10,
    )
    loss.backward()

    assert pred.grad is not None
    assert float(pred.grad[:2].sum().item()) < 0.0
    assert float(pred.grad[3:].sum().item()) > 0.0


def test_budget_topk_recall_loss_rows_match_scalar_helper() -> None:
    pred = torch.tensor(
        [
            [3.0, 2.0, 0.5, -1.0, -2.0, -3.0],
            [-2.0, 1.5, 0.2, 0.1, -0.5, -9.0],
        ],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [
            [1.0, 0.9, 0.1, 0.0, 0.0, 0.0],
            [0.0, 0.7, 0.4, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    valid_mask = torch.tensor(
        [
            [True, True, True, True, True, True],
            [True, True, True, True, True, False],
        ]
    )

    row_loss, active_rows = _budget_topk_recall_loss_rows(
        pred=pred,
        target=target,
        valid_mask=valid_mask,
        budget_ratios=(0.20, 0.40),
        temperature=0.10,
    )

    assert active_rows.tolist() == [True, True]
    for row in range(pred.shape[0]):
        scalar = _budget_topk_recall_loss(
            pred=pred[row],
            target=target[row],
            valid_mask=valid_mask[row],
            budget_ratios=(0.20, 0.40),
            temperature=0.10,
        )
        assert row_loss[row].item() == pytest.approx(scalar.item(), abs=1e-6)


def test_budget_topk_temporal_residual_loss_rows_match_scalar_helper() -> None:
    pred = torch.tensor(
        [
            [3.0, 2.0, 0.5, -1.0, -2.0, -3.0],
            [-2.0, 1.5, 0.2, 0.1, -0.5, -9.0],
        ],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [
            [1.0, 0.9, 0.1, 0.0, 0.0, 0.0],
            [0.0, 0.7, 0.4, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    valid_mask = torch.tensor(
        [
            [True, True, True, True, True, True],
            [True, True, True, True, True, False],
        ]
    )
    global_idx = torch.tensor([[0, 1, 2, 3, 4, 5], [5, 6, 7, 8, 9, -1]])
    base_mask_a = torch.zeros((10,), dtype=torch.bool)
    base_mask_a[[0, 5, 9]] = True
    base_mask_b = torch.zeros((10,), dtype=torch.bool)
    base_mask_b[[0, 1, 5, 9]] = True
    temporal_base_masks = (
        (0.05, 0.02, base_mask_a),
        (0.10, 0.05, base_mask_b),
    )

    row_loss, active_rows = _budget_topk_temporal_residual_loss_rows(
        pred=pred,
        target=target,
        valid_mask=valid_mask,
        global_idx=global_idx.clamp(min=0),
        temporal_base_masks=temporal_base_masks,
        temperature=0.10,
    )

    assert active_rows.tolist() == [True, True]
    for row in range(pred.shape[0]):
        scalar = _budget_topk_temporal_residual_loss(
            pred=pred[row][valid_mask[row]],
            target=target[row][valid_mask[row]],
            valid_mask=torch.ones((int(valid_mask[row].sum().item()),), dtype=torch.bool),
            global_idx=global_idx[row][valid_mask[row]],
            temporal_base_masks=temporal_base_masks,
            temperature=0.10,
        )
        assert row_loss[row].item() == pytest.approx(scalar.item(), abs=1e-6)


def test_temporal_residual_labels_drop_base_points() -> None:
    labels = torch.ones((10, 4), dtype=torch.float32)
    labelled_mask = torch.ones((10, 4), dtype=torch.bool)

    residual_labels, residual_mask = _apply_temporal_residual_labels(
        labels=labels,
        labelled_mask=labelled_mask,
        boundaries=[(0, 10)],
        compression_ratio=0.3,
        temporal_fraction=0.5,
    )

    assert torch.where(~residual_mask[:, 0])[0].tolist() == [0, 9]
    assert residual_labels[[0, 9]].sum().item() == pytest.approx(0.0)
    assert bool(residual_mask[5].all().item())


def test_temporal_residual_labels_keep_all_labels_when_base_disabled() -> None:
    labels = torch.ones((10, 4), dtype=torch.float32)
    labelled_mask = torch.ones((10, 4), dtype=torch.bool)

    residual_labels, residual_mask = _apply_temporal_residual_labels(
        labels=labels,
        labelled_mask=labelled_mask,
        boundaries=[(0, 10)],
        compression_ratio=0.3,
        temporal_fraction=0.0,
    )

    assert bool(residual_mask.all().item())
    assert float(residual_labels.sum().item()) == pytest.approx(float(labels.sum().item()))


def test_single_active_type_rejects_mixed_training_weights() -> None:
    assert _single_active_type_id(torch.tensor([1.0, 0.0, 0.0, 0.0])) == 0

    with pytest.raises(ValueError, match="Pure-workload"):
        _single_active_type_id(torch.tensor([0.5, 0.5, 0.0, 0.0]))


def test_filter_supervised_windows_removes_zero_positive_training_windows() -> None:
    points = torch.arange(24, dtype=torch.float32).reshape(12, 2)
    windows = build_trajectory_windows(points, boundaries=[(0, 4), (4, 12)], window_length=4, stride=4)
    targets = torch.zeros((12, 4), dtype=torch.float32)
    labelled_mask = torch.ones((12, 4), dtype=torch.bool)
    targets[5, 0] = 1.0

    kept, filtered = _filter_supervised_windows(
        windows=windows,
        training_target=targets[:, 0],
        labelled_mask=labelled_mask[:, 0],
        active_type_id=0,
    )

    assert len(windows) == 3
    assert len(kept) == 1
    assert int(filtered[0].item()) == 2


def test_filter_supervised_windows_can_keep_zero_labelled_windows_for_pointwise_objective() -> None:
    points = torch.arange(24, dtype=torch.float32).reshape(12, 2)
    windows = build_trajectory_windows(points, boundaries=[(0, 4), (4, 12)], window_length=4, stride=4)
    targets = torch.zeros((12,), dtype=torch.float32)
    labelled_mask = torch.ones((12,), dtype=torch.bool)

    kept, filtered = _filter_supervised_windows(
        windows=windows,
        training_target=targets,
        labelled_mask=labelled_mask,
        active_type_id=0,
        require_positive=False,
    )

    assert len(windows) == 3
    assert len(kept) == 3
    assert int(filtered[0].item()) == 0


def test_training_records_validation_selection_score() -> None:
    trajectories = generate_synthetic_ais_data(n_ships=5, n_points_per_ship=24, seed=444)
    train_trajectories = trajectories[:4]
    validation_trajectories = trajectories[4:]
    train_ds = TrajectoryDataset(train_trajectories)
    validation_ds = TrajectoryDataset(validation_trajectories)
    train_boundaries = train_ds.get_trajectory_boundaries()
    validation_boundaries = validation_ds.get_trajectory_boundaries()

    cfg = build_experiment_config(
        epochs=8,
        n_queries=4,
        workload="range",
        checkpoint_selection_metric="score",
        validation_score_every=2,
        compression_ratio=0.5,
    )
    cfg.model.embed_dim = 16
    cfg.model.num_heads = 2
    cfg.model.num_layers = 1
    cfg.model.query_chunk_size = 8
    cfg.model.window_length = 16
    cfg.model.window_stride = 8
    cfg.model.ranking_pairs_per_type = 8
    cfg.model.train_batch_size = 4
    cfg.model.diagnostic_window_fraction = 1.0

    train_workload = generate_typed_query_workload(
        trajectories=train_trajectories,
        n_queries=4,
        workload_map={"range": 1.0},
        seed=101,
    )
    validation_workload = generate_typed_query_workload(
        trajectories=validation_trajectories,
        n_queries=4,
        workload_map={"range": 1.0},
        seed=202,
    )

    out = train_model(
        train_trajectories=train_trajectories,
        train_boundaries=train_boundaries,
        workload=train_workload,
        model_config=cfg.model,
        seed=303,
        validation_trajectories=validation_trajectories,
        validation_boundaries=validation_boundaries,
        validation_workload=validation_workload,
        validation_workload_map={"range": 1.0},
    )

    score_rows = [row for row in out.history if "val_selection_score" in row]
    assert score_rows
    assert [int(row["epoch"]) for row in score_rows] == [0, 1, 3, 5, 7]
    assert all(0.0 <= row["val_selection_score"] <= 1.0 for row in score_rows)
    assert all("val_range_point_f1" in row for row in score_rows)
    assert all("val_range_usefulness" in row for row in score_rows)
    assert all("val_query_f1" not in row for row in score_rows)
    assert all("selection_score" not in row for row in out.history if "val_selection_score" not in row)
    assert out.best_selection_score == pytest.approx(max(row["val_selection_score"] for row in score_rows))


def test_checkpoint_candidate_pool_defers_full_validation() -> None:
    trajectories = generate_synthetic_ais_data(n_ships=5, n_points_per_ship=24, seed=445)
    train_trajectories = trajectories[:4]
    validation_trajectories = trajectories[4:]
    train_boundaries = TrajectoryDataset(train_trajectories).get_trajectory_boundaries()
    validation_boundaries = TrajectoryDataset(validation_trajectories).get_trajectory_boundaries()

    cfg = build_experiment_config(
        epochs=5,
        n_queries=4,
        workload="range",
        checkpoint_selection_metric="score",
        validation_score_every=1,
        checkpoint_full_score_every=3,
        checkpoint_candidate_pool_size=1,
        compression_ratio=0.5,
    )
    cfg.model.embed_dim = 16
    cfg.model.num_heads = 2
    cfg.model.num_layers = 1
    cfg.model.query_chunk_size = 8
    cfg.model.window_length = 16
    cfg.model.window_stride = 8
    cfg.model.ranking_pairs_per_type = 8
    cfg.model.train_batch_size = 4
    cfg.model.diagnostic_window_fraction = 1.0

    train_workload = generate_typed_query_workload(
        trajectories=train_trajectories,
        n_queries=4,
        workload_map={"range": 1.0},
        seed=111,
    )
    validation_workload = generate_typed_query_workload(
        trajectories=validation_trajectories,
        n_queries=4,
        workload_map={"range": 1.0},
        seed=222,
    )

    out = train_model(
        train_trajectories=train_trajectories,
        train_boundaries=train_boundaries,
        workload=train_workload,
        model_config=cfg.model,
        seed=333,
        validation_trajectories=validation_trajectories,
        validation_boundaries=validation_boundaries,
        validation_workload=validation_workload,
        validation_workload_map={"range": 1.0},
    )

    candidate_rows = [row for row in out.history if row.get("checkpoint_score_candidate") == 1.0]
    evaluated_rows = [row for row in out.history if row.get("checkpoint_candidate_evaluated") == 1.0]

    assert len(candidate_rows) == 5
    assert 1 <= len(evaluated_rows) <= 3
    assert all("val_selection_score" in row for row in evaluated_rows)
    assert all("selection_score" in row for row in evaluated_rows)
    assert all(0.0 <= row["val_selection_score"] <= 1.0 for row in evaluated_rows)
    assert all("selection_score" not in row for row in candidate_rows if row.get("checkpoint_candidate_evaluated") != 1.0)
    assert out.best_selection_score == pytest.approx(
        max(row["val_selection_score"] for row in evaluated_rows)
    )


@pytest.mark.parametrize(
    "score_mode",
    ["rank", "rank_tie", "raw", "sigmoid", "zscore_sigmoid", "rank_confidence", "temperature_sigmoid"],
)
def test_validation_query_score_matches_final_mlqds_scoring(score_mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    trajectories = generate_synthetic_ais_data(n_ships=2, n_points_per_ship=12, seed=515)
    ds = TrajectoryDataset(trajectories)
    points = ds.get_all_points()
    boundaries = ds.get_trajectory_boundaries()
    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=6,
        workload_map={"range": 1.0},
        seed=516,
        range_spatial_fraction=0.40,
        range_time_fraction=0.40,
    )
    cfg = build_experiment_config(
        compression_ratio=0.40,
        workload="range",
        mlqds_temporal_fraction=0.25,
        mlqds_diversity_bonus=0.0,
        mlqds_hybrid_mode="swap",
        mlqds_score_mode=score_mode,
        mlqds_score_temperature=0.75,
        mlqds_rank_confidence_weight=0.35,
        checkpoint_score_variant="answer",
    )
    predictions = torch.linspace(-1.0, 1.0, steps=points.shape[0])

    model = TrajectoryQDSModel(
        point_dim=7,
        query_dim=int(workload.query_features.shape[1]),
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        query_chunk_size=8,
    )
    scaler = FeatureScaler.fit(points[:, :7], workload.query_features)
    trained = TrainingOutputs(
        model=model,
        scaler=scaler,
        labels=torch.zeros((points.shape[0], 4), dtype=torch.float32),
        labelled_mask=torch.ones((points.shape[0], 4), dtype=torch.bool),
        history=[],
    )

    monkeypatch.setattr(
        "evaluation.baselines.windowed_predict",
        lambda **_kwargs: predictions.clone(),
    )

    validation_score, validation_per_type = _validation_query_score(
        model=model,
        scaler=scaler,
        trajectories=trajectories,
        boundaries=boundaries,
        workload=workload,
        workload_map={"range": 1.0},
        model_config=cfg.model,
        device=torch.device("cpu"),
        validation_points=points,
        predict_logits_fn=lambda **_kwargs: predictions.clone(),
    )
    retained = MLQDSMethod(
        name="MLQDS",
        trained=trained,
        workload=workload,
        workload_type="range",
        score_mode=cfg.model.mlqds_score_mode,
        score_temperature=cfg.model.mlqds_score_temperature,
        rank_confidence_weight=cfg.model.mlqds_rank_confidence_weight,
        temporal_fraction=cfg.model.mlqds_temporal_fraction,
        diversity_bonus=cfg.model.mlqds_diversity_bonus,
        hybrid_mode=cfg.model.mlqds_hybrid_mode,
        inference_device="cpu",
        inference_batch_size=cfg.model.inference_batch_size,
    ).simplify(points, boundaries, cfg.model.compression_ratio)
    final_f1, final_per_type, _combined_f1, _combined_per_type = score_retained_mask(
        points=points,
        boundaries=boundaries,
        retained_mask=retained,
        typed_queries=workload.typed_queries,
        workload_map={"range": 1.0},
    )

    assert validation_score == pytest.approx(final_f1)
    assert validation_per_type["range"] == pytest.approx(final_per_type["range"])


def test_validation_range_usefulness_matches_final_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    trajectories = generate_synthetic_ais_data(n_ships=2, n_points_per_ship=12, seed=615)
    ds = TrajectoryDataset(trajectories)
    points = ds.get_all_points()
    boundaries = ds.get_trajectory_boundaries()
    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=6,
        workload_map={"range": 1.0},
        seed=616,
        range_spatial_fraction=0.40,
        range_time_fraction=0.40,
    )
    cfg = build_experiment_config(
        compression_ratio=0.40,
        workload="range",
        mlqds_temporal_fraction=0.25,
        mlqds_diversity_bonus=0.0,
        mlqds_hybrid_mode="swap",
        mlqds_score_mode="rank",
        checkpoint_score_variant="range_usefulness",
    )
    predictions = torch.linspace(-1.0, 1.0, steps=points.shape[0])

    model = TrajectoryQDSModel(
        point_dim=7,
        query_dim=int(workload.query_features.shape[1]),
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        query_chunk_size=8,
    )
    scaler = FeatureScaler.fit(points[:, :7], workload.query_features)
    trained = TrainingOutputs(
        model=model,
        scaler=scaler,
        labels=torch.zeros((points.shape[0], 4), dtype=torch.float32),
        labelled_mask=torch.ones((points.shape[0], 4), dtype=torch.bool),
        history=[],
    )

    monkeypatch.setattr(
        "evaluation.baselines.windowed_predict",
        lambda **_kwargs: predictions.clone(),
    )

    validation_score, validation_per_type = _validation_query_score(
        model=model,
        scaler=scaler,
        trajectories=trajectories,
        boundaries=boundaries,
        workload=workload,
        workload_map={"range": 1.0},
        model_config=cfg.model,
        device=torch.device("cpu"),
        validation_points=points,
        predict_logits_fn=lambda **_kwargs: predictions.clone(),
    )
    retained = MLQDSMethod(
        name="MLQDS",
        trained=trained,
        workload=workload,
        workload_type="range",
        score_mode=cfg.model.mlqds_score_mode,
        temporal_fraction=cfg.model.mlqds_temporal_fraction,
        diversity_bonus=cfg.model.mlqds_diversity_bonus,
        hybrid_mode=cfg.model.mlqds_hybrid_mode,
        inference_device="cpu",
        inference_batch_size=cfg.model.inference_batch_size,
    ).simplify(points, boundaries, cfg.model.compression_ratio)
    audit = score_range_usefulness(
        points=points,
        boundaries=boundaries,
        retained_mask=retained,
        typed_queries=workload.typed_queries,
    )

    assert validation_score == pytest.approx(audit["range_usefulness_score"])
    assert validation_per_type["range"] == pytest.approx(audit["range_usefulness_score"])


def test_training_accepts_precomputed_importance_labels() -> None:
    trajectories = generate_synthetic_ais_data(n_ships=3, n_points_per_ship=12, seed=818)
    ds = TrajectoryDataset(trajectories)
    boundaries = ds.get_trajectory_boundaries()
    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=4,
        workload_map={"range": 1.0},
        seed=819,
    )
    labels, labelled_mask = compute_typed_importance_labels(
        points=ds.get_all_points(),
        boundaries=boundaries,
        typed_queries=workload.typed_queries,
    )

    cfg = build_experiment_config(
        epochs=1,
        n_queries=4,
        workload="range",
        compression_ratio=0.5,
    )
    cfg.model.embed_dim = 16
    cfg.model.num_heads = 2
    cfg.model.num_layers = 1
    cfg.model.query_chunk_size = 8
    cfg.model.window_length = 8
    cfg.model.window_stride = 4
    cfg.model.ranking_pairs_per_type = 4
    cfg.model.train_batch_size = 2
    cfg.model.diagnostic_window_fraction = 1.0

    out = train_model(
        train_trajectories=trajectories,
        train_boundaries=boundaries,
        workload=workload,
        model_config=cfg.model,
        seed=821,
        precomputed_labels=(labels, labelled_mask),
    )

    assert out.history
    assert out.epochs_trained == 1


def test_historical_prior_training_returns_fitted_prior_and_diagnostics() -> None:
    trajectories = generate_synthetic_ais_data(n_ships=3, n_points_per_ship=12, seed=822)
    ds = TrajectoryDataset(trajectories)
    boundaries = ds.get_trajectory_boundaries()
    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=4,
        workload_map={"range": 1.0},
        seed=823,
    )
    cfg = build_experiment_config(
        epochs=3,
        n_queries=4,
        workload="range",
        model_type="historical_prior",
        historical_prior_k=2,
        compression_ratio=0.5,
    )

    out = train_model(
        train_trajectories=trajectories,
        train_boundaries=boundaries,
        workload=workload,
        model_config=cfg.model,
        seed=824,
    )

    assert isinstance(out.model, HistoricalPriorRangeQDSModel)
    assert out.epochs_trained == 0
    assert out.best_epoch == 0
    assert out.history
    assert out.target_diagnostics["workload_type_id"] == 0
    assert "positive_fraction_t0" in out.history[0]
    assert out.fit_diagnostics["enabled"] is True
    assert out.fit_diagnostics["model_fits_stored_train_support"] is True
    assert out.fit_diagnostics["matched_mlqds_target_recall"] is not None
    assert out.model.historical_features.shape[0] == ds.get_all_points().shape[0]
    scores = out.model(
        out.model.historical_features[:4].unsqueeze(0),
        queries=None,
        query_type_ids=None,
    )
    assert torch.isfinite(scores).all()


def test_historical_prior_training_caps_support_per_trajectory() -> None:
    trajectories = generate_synthetic_ais_data(n_ships=3, n_points_per_ship=12, seed=825)
    ds = TrajectoryDataset(trajectories)
    boundaries = ds.get_trajectory_boundaries()
    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=4,
        workload_map={"range": 1.0},
        seed=826,
    )
    points = ds.get_all_points()
    labels = torch.zeros((points.shape[0], NUM_QUERY_TYPES), dtype=torch.float32)
    labelled_mask = torch.zeros_like(labels, dtype=torch.bool)
    labelled_mask[:, QUERY_TYPE_ID_RANGE] = True
    for start, end in boundaries:
        labels[start:end, QUERY_TYPE_ID_RANGE] = torch.linspace(0.0, 1.0, steps=end - start)
    cfg = build_experiment_config(
        epochs=3,
        n_queries=4,
        workload="range",
        model_type="historical_prior",
        historical_prior_k=2,
        historical_prior_support_ratio=0.25,
        compression_ratio=0.5,
    )

    out = train_model(
        train_trajectories=trajectories,
        train_boundaries=boundaries,
        workload=workload,
        model_config=cfg.model,
        seed=827,
        precomputed_labels=(labels, labelled_mask),
    )

    assert isinstance(out.model, HistoricalPriorRangeQDSModel)
    assert out.model.historical_features.shape[0] == 9
    assert out.target_diagnostics["historical_prior_support_pre_min_count"] == 9
    assert out.target_diagnostics["historical_prior_stored_support_count"] == 9


def test_historical_prior_training_preserves_train_source_ids() -> None:
    trajectories = generate_synthetic_ais_data(n_ships=3, n_points_per_ship=12, seed=828)
    ds = TrajectoryDataset(trajectories)
    boundaries = ds.get_trajectory_boundaries()
    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=4,
        workload_map={"range": 1.0},
        seed=829,
    )
    cfg = build_experiment_config(
        epochs=1,
        n_queries=4,
        workload="range",
        model_type="historical_prior",
        historical_prior_k=2,
        historical_prior_source_aggregation="mean",
        compression_ratio=0.5,
    )
    source_ids = [0, 0, 1]

    out = train_model(
        train_trajectories=trajectories,
        train_boundaries=boundaries,
        workload=workload,
        model_config=cfg.model,
        seed=830,
        train_trajectory_source_ids=source_ids,
    )

    assert isinstance(out.model, HistoricalPriorRangeQDSModel)
    expected = torch.cat([
        torch.full((end - start,), source_id, dtype=torch.long)
        for source_id, (start, end) in zip(source_ids, boundaries, strict=True)
    ])
    assert torch.equal(out.model.historical_source_ids, expected)
    assert out.target_diagnostics["historical_prior_source_aggregation"] == "mean"
    assert out.target_diagnostics["historical_prior_source_count"] == 2


def test_ranking_bce_objective_keeps_rank_signal(synthetic_dataset) -> None:
    """Assert the ranking_bce objective preserves its rank-signal invariant."""
    trajectories, _ = synthetic_dataset
    ds = TrajectoryDataset(trajectories)
    boundaries = ds.get_trajectory_boundaries()

    cfg = build_experiment_config(epochs=4, n_queries=80, workload="range", loss_objective="ranking_bce")
    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=80,
        workload_map={"range": 1.0},
        seed=77,
    )
    out = train_model(
        train_trajectories=trajectories,
        train_boundaries=boundaries,
        workload=workload,
        model_config=cfg.model,
        seed=77,
    )

    diagnostics = [row for row in out.history if "pred_std" in row]
    last = diagnostics[-1]
    assert last["pred_std"] > 0.02

    best_range_tau = max(row["kendall_tau_t0"] for row in diagnostics)
    assert best_range_tau > 0.15


def test_pointwise_bce_objective_trains_on_range_labels(synthetic_dataset) -> None:
    """Assert the direct pointwise objective is accepted by the trainer."""
    trajectories, _ = synthetic_dataset
    ds = TrajectoryDataset(trajectories)
    boundaries = ds.get_trajectory_boundaries()

    cfg = build_experiment_config(epochs=1, n_queries=24, workload="range", loss_objective="pointwise_bce")
    cfg.model.embed_dim = 16
    cfg.model.num_heads = 2
    cfg.model.num_layers = 1
    cfg.model.query_chunk_size = 8
    cfg.model.window_length = 8
    cfg.model.window_stride = 4
    cfg.model.train_batch_size = 2
    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=24,
        workload_map={"range": 1.0},
        seed=78,
    )
    out = train_model(
        train_trajectories=trajectories,
        train_boundaries=boundaries,
        workload=workload,
        model_config=cfg.model,
        seed=78,
    )

    assert out.history
    assert out.history[-1]["loss"] > 0.0
    assert out.fit_diagnostics["enabled"] is True
    assert out.fit_diagnostics["matched_mlqds_target_recall"] is not None


def test_range_coverage_training_keeps_score_spread(synthetic_dataset) -> None:
    """Assert coverage-targeted range training does not converge to constant scores."""
    trajectories, _ = synthetic_dataset
    ds = TrajectoryDataset(trajectories)
    boundaries = ds.get_trajectory_boundaries()

    cfg = build_experiment_config(
        epochs=4,
        n_queries=60,
        query_coverage=0.30,
        max_queries=160,
        workload="range",
        lr=1e-3,
    )
    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=60,
        target_coverage=0.30,
        max_queries=160,
        workload_map={"range": 1.0},
        range_spatial_fraction=0.02,
        range_time_fraction=0.04,
        seed=91,
    )
    out = train_model(
        train_trajectories=trajectories,
        train_boundaries=boundaries,
        workload=workload,
        model_config=cfg.model,
        seed=91,
    )

    diagnostics = [row for row in out.history if "pred_std" in row]
    assert max(row["pred_std"] for row in diagnostics) > 0.01
    assert diagnostics[-1]["pred_std"] > 1e-3
