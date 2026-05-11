"""Tests that short training keeps non-collapsed typed predictions. See src/training/README.md for details."""

from __future__ import annotations

import pytest
import torch

from src.data.ais_loader import generate_synthetic_ais_data
from src.data.trajectory_dataset import TrajectoryDataset
from src.evaluation.baselines import MLQDSMethod
from src.evaluation.evaluate_methods import score_retained_mask
from src.experiments.experiment_config import build_experiment_config
from src.models.trajectory_qds_model import TrajectoryQDSModel
from src.queries.query_generator import generate_typed_query_workload
from src.training.importance_labels import compute_typed_importance_labels
from src.training.scaler import FeatureScaler
from src.training.train_model import (
    TrainingOutputs,
    _apply_temporal_residual_labels,
    _balanced_pointwise_loss,
    _filter_supervised_windows,
    _f1_selection_score,
    _ranking_loss_for_type,
    _selection_score,
    _single_active_type_id,
    _uniform_gap_selection_score,
    _validation_query_f1,
    train_model,
)
from src.training.trajectory_batching import build_trajectory_windows


def test_selection_score_penalizes_collapsed_predictions() -> None:
    """Assert model selection does not prefer collapsed output solely because tau is nonnegative."""
    assert _selection_score(avg_tau=0.0, pred_std=0.0) < _selection_score(avg_tau=-0.05, pred_std=0.01)


def test_selection_score_uses_loss_before_tau_proxy() -> None:
    """Assert checkpoint selection does not restore a worse-loss epoch solely from noisy tau."""
    proxy_best = _selection_score(avg_tau=0.9, pred_std=0.1, loss=0.20)
    lower_loss = _selection_score(avg_tau=-0.1, pred_std=0.1, loss=0.10)

    assert lower_loss > proxy_best


def test_f1_selection_score_penalizes_collapsed_predictions() -> None:
    assert _f1_selection_score(query_f1=0.8, pred_std=0.0) < _f1_selection_score(query_f1=0.2, pred_std=0.01)


def test_uniform_gap_selection_penalizes_active_type_deficit() -> None:
    workload_map = {"range": 1.0}
    uniform_per_type = {"range": 0.50}
    range_deficit = _uniform_gap_selection_score(
        query_f1=0.55,
        per_type_f1={"range": 0.45},
        uniform_f1=0.50,
        uniform_per_type=uniform_per_type,
        workload_map=workload_map,
        pred_std=0.1,
    )
    balanced = _uniform_gap_selection_score(
        query_f1=0.54,
        per_type_f1={"range": 0.54},
        uniform_f1=0.50,
        uniform_per_type=uniform_per_type,
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


def test_training_records_validation_query_f1() -> None:
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
        checkpoint_selection_metric="f1",
        f1_diagnostic_every=2,
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

    f1_rows = [row for row in out.history if "val_query_f1" in row]
    assert f1_rows
    assert [int(row["epoch"]) for row in f1_rows] == [0, 1, 3, 5, 7]
    assert all(0.0 <= row["val_query_f1"] <= 1.0 for row in f1_rows)
    assert all("selection_score" not in row for row in out.history if "val_query_f1" not in row)
    assert out.best_f1 == pytest.approx(max(row["val_query_f1"] for row in f1_rows))


@pytest.mark.parametrize(
    "score_mode",
    ["rank", "rank_tie", "raw", "sigmoid", "zscore_sigmoid", "rank_confidence", "temperature_sigmoid"],
)
def test_validation_query_f1_matches_final_mlqds_scoring(score_mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
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
        mlqds_score_mode=score_mode,
        mlqds_score_temperature=0.75,
        mlqds_rank_confidence_weight=0.35,
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
        "src.training.train_model._predict_workload_logits",
        lambda **_kwargs: predictions.clone(),
    )
    monkeypatch.setattr(
        "src.evaluation.baselines.windowed_predict",
        lambda **_kwargs: predictions.clone(),
    )

    validation_f1, validation_per_type = _validation_query_f1(
        model=model,
        scaler=scaler,
        trajectories=trajectories,
        boundaries=boundaries,
        workload=workload,
        workload_map={"range": 1.0},
        model_config=cfg.model,
        device=torch.device("cpu"),
        validation_points=points,
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

    assert validation_f1 == pytest.approx(final_f1)
    assert validation_per_type["range"] == pytest.approx(final_per_type["range"])


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
        seed=820,
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


def test_range_training_does_not_collapse(synthetic_dataset) -> None:
    """Assert range-focused training keeps prediction spread and rank signal healthy."""
    trajectories, _ = synthetic_dataset
    ds = TrajectoryDataset(trajectories)
    boundaries = ds.get_trajectory_boundaries()

    cfg = build_experiment_config(epochs=4, n_queries=80, workload="range")
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
