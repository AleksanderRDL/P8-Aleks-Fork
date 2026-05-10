"""Tests scaler and model persistence keeps predictions identical. See src/training/README.md for details."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.experiments.experiment_config import build_experiment_config
from src.models.trajectory_qds_model import TrajectoryQDSModel
from src.training.scaler import FeatureScaler
from src.training.training_pipeline import ModelArtifacts, forward_predict, load_checkpoint, save_checkpoint, windowed_predict


def test_scaler_persisted(tmp_path: Path) -> None:
    """Assert checkpoint reload yields identical predictions. See src/training/README.md for details."""
    model = TrajectoryQDSModel(point_dim=7, query_dim=12)
    model.eval()
    points = torch.randn(128, 7)
    queries = torch.randn(32, 12)
    q_ids = torch.zeros((32,), dtype=torch.long)

    scaler = FeatureScaler.fit(points, queries)
    cfg = build_experiment_config()
    art = ModelArtifacts(
        model=model,
        scaler=scaler,
        config=cfg,
        epochs_trained=3,
        train_workload_mix={"range": 1.0},
        eval_workload_mix={"knn": 1.0},
    )

    ckpt = tmp_path / "model.pt"
    save_checkpoint(str(ckpt), art)
    loaded = load_checkpoint(str(ckpt))

    p1 = forward_predict(art, points, queries, q_ids)
    p2 = forward_predict(loaded, points, queries, q_ids)
    assert torch.allclose(p1, p2, atol=1e-7)
    assert loaded.epochs_trained == 3
    assert loaded.train_workload_mix == {"range": 1.0}
    assert loaded.eval_workload_mix == {"knn": 1.0}


def test_windowed_predict_batching_matches_single_window_loop() -> None:
    """Assert batched inference preserves the previous per-window predictions."""
    model = TrajectoryQDSModel(point_dim=7, query_dim=12)
    model.eval()
    points = torch.randn(30, 7)
    queries = torch.randn(4, 12)
    q_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    boundaries = [(0, 10), (10, 20), (20, 30)]

    pred_single = windowed_predict(
        model=model,
        norm_points=points,
        boundaries=boundaries,
        queries=queries,
        query_type_ids=q_ids,
        window_length=8,
        window_stride=4,
        batch_size=1,
    )
    pred_batched = windowed_predict(
        model=model,
        norm_points=points,
        boundaries=boundaries,
        queries=queries,
        query_type_ids=q_ids,
        window_length=8,
        window_stride=4,
        batch_size=3,
    )

    assert torch.allclose(pred_single, pred_batched, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_windowed_predict_cuda_matches_cpu() -> None:
    """Assert CUDA inference keeps predictions numerically aligned with CPU output."""
    torch.manual_seed(222)
    model = TrajectoryQDSModel(point_dim=7, query_dim=12, embed_dim=16, num_heads=2, num_layers=1)
    model.eval()
    points = torch.randn(30, 7)
    queries = torch.randn(4, 12)
    q_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    boundaries = [(0, 10), (10, 20), (20, 30)]

    pred_cpu = windowed_predict(
        model=model,
        norm_points=points,
        boundaries=boundaries,
        queries=queries,
        query_type_ids=q_ids,
        window_length=8,
        window_stride=4,
        batch_size=3,
        device="cpu",
    )
    pred_cuda = windowed_predict(
        model=model,
        norm_points=points,
        boundaries=boundaries,
        queries=queries,
        query_type_ids=q_ids,
        window_length=8,
        window_stride=4,
        batch_size=3,
        device="cuda",
    )
    pred_cuda_input = windowed_predict(
        model=model,
        norm_points=points.cuda(),
        boundaries=boundaries,
        queries=queries.cuda(),
        query_type_ids=q_ids.cuda(),
        window_length=8,
        window_stride=4,
        batch_size=3,
        device="cuda",
    )

    assert pred_cuda.device == points.device
    assert pred_cuda_input.device.type == "cuda"
    assert next(model.parameters()).device.type == "cpu"
    assert torch.allclose(pred_cpu, pred_cuda, atol=1e-4, rtol=1e-4)
    assert torch.allclose(pred_cpu, pred_cuda_input.cpu(), atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_forward_predict_cuda_matches_cpu() -> None:
    """Assert persisted-artifact prediction can run on CUDA without changing outputs."""
    torch.manual_seed(333)
    model = TrajectoryQDSModel(point_dim=7, query_dim=12, embed_dim=16, num_heads=2, num_layers=1)
    model.eval()
    points = torch.randn(24, 7)
    queries = torch.randn(5, 12)
    q_ids = torch.tensor([0, 1, 2, 3, 0], dtype=torch.long)
    boundaries = [(0, 12), (12, 24)]
    scaler = FeatureScaler.fit(points, queries)
    art = ModelArtifacts(
        model=model,
        scaler=scaler,
        config=build_experiment_config(),
    )

    pred_cpu = forward_predict(art, points, queries, q_ids, boundaries=boundaries, device="cpu")
    pred_cuda = forward_predict(art, points, queries, q_ids, boundaries=boundaries, device="cuda")

    assert pred_cuda.device == points.device
    assert next(art.model.parameters()).device.type == "cpu"
    assert torch.allclose(pred_cpu, pred_cuda, atol=1e-4, rtol=1e-4)
