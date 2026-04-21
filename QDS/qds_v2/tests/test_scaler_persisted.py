"""Tests scaler and model persistence keeps predictions identical. See src/training/README.md for details."""

from __future__ import annotations

from pathlib import Path

import torch

from src.experiments.experiment_config import build_experiment_config
from src.models.trajectory_qds_model import TrajectoryQDSModel
from src.training.scaler import FeatureScaler
from src.training.training_pipeline import ModelArtifacts, forward_predict, load_checkpoint, save_checkpoint


def test_scaler_persisted(tmp_path: Path) -> None:
    """Assert checkpoint reload yields identical predictions. See src/training/README.md for details."""
    model = TrajectoryQDSModel(point_dim=7, query_dim=12)
    model.eval()
    points = torch.randn(128, 7)
    queries = torch.randn(32, 12)
    q_ids = torch.zeros((32,), dtype=torch.long)

    scaler = FeatureScaler.fit(points, queries)
    cfg = build_experiment_config()
    art = ModelArtifacts(model=model, scaler=scaler, config=cfg)

    ckpt = tmp_path / "model.pt"
    save_checkpoint(str(ckpt), art)
    loaded = load_checkpoint(str(ckpt))

    p1 = forward_predict(art, points, queries, q_ids)
    p2 = forward_predict(loaded, points, queries, q_ids)
    assert torch.allclose(p1, p2, atol=1e-7)
