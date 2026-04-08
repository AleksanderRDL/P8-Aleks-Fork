"""Tests for training helpers and model training flow."""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.trajectory_qds_model import TrajectoryQDSModel
from src.models.turn_aware_qds_model import TurnAwareQDSModel
from src.training.importance_labels import compute_importance
from src.training.train_model import train_model


def _make_trajectories(
    *,
    n_ships: int = 2,
    n_points: int = 5,
    n_features: int = 8,
) -> list[torch.Tensor]:
    trajectories: list[torch.Tensor] = []
    for ship in range(n_ships):
        traj = torch.zeros(n_points, n_features, dtype=torch.float32)
        traj[:, 0] = torch.arange(n_points, dtype=torch.float32) + ship * 100.0
        traj[:, 1] = 50.0 + torch.linspace(0.0, 0.4, n_points)
        traj[:, 2] = 10.0 + torch.linspace(0.0, 0.4, n_points)
        traj[:, 3] = torch.linspace(5.0, 9.0, n_points)
        traj[:, 4] = torch.linspace(0.0, 90.0, n_points)
        if n_features >= 6:
            traj[0, 5] = 1.0
        if n_features >= 7:
            traj[-1, 6] = 1.0
        if n_features >= 8 and n_points > 2:
            traj[1:-1, 7] = torch.linspace(0.1, 0.9, n_points - 2)
        trajectories.append(traj)
    return trajectories


def _make_queries() -> torch.Tensor:
    return torch.tensor(
        [
            [49.5, 51.0, 9.5, 11.0, 0.0, 200.0],
            [50.1, 50.6, 10.1, 10.6, 0.0, 200.0],
        ],
        dtype=torch.float32,
    )


class TestComputeImportance:
    def test_empty_queries_returns_zero_importance(self):
        points = torch.rand(6, 5)
        queries = torch.zeros((0, 6), dtype=torch.float32)
        importance = compute_importance(points, queries)
        assert importance.shape == (6,)
        assert torch.allclose(importance, torch.zeros(6))

    def test_importance_is_normalized_and_ordered_for_speed(self):
        points = torch.tensor(
            [
                [0.0, 50.0, 10.0, 1.0, 0.0],
                [1.0, 50.1, 10.1, 2.0, 0.0],
                [2.0, 50.2, 10.2, 3.0, 0.0],
            ],
            dtype=torch.float32,
        )
        queries = torch.tensor([[49.0, 51.0, 9.0, 11.0, 0.0, 3.0]], dtype=torch.float32)
        importance = compute_importance(points, queries, chunk_size=2)
        assert importance.shape == (3,)
        assert importance.min().item() >= 0.0
        assert importance.max().item() <= 1.0
        assert importance[2].item() > importance[1].item() > importance[0].item()


@pytest.mark.integration
@pytest.mark.slow
class TestTrainModel:
    def test_importance_length_mismatch_raises(self):
        trajectories = _make_trajectories(n_features=8)
        queries = _make_queries()
        n_points = sum(t.shape[0] for t in trajectories)
        bad_importance = torch.zeros(n_points - 1, dtype=torch.float32)
        with pytest.raises(ValueError, match="same length"):
            train_model(
                trajectories=trajectories,
                queries=queries,
                epochs=1,
                importance=bad_importance,
                point_batch_size=None,
            )

    def test_trains_baseline_and_saves_weights(self, tmp_path):
        trajectories = _make_trajectories(n_features=8)
        queries = _make_queries()
        n_points = sum(t.shape[0] for t in trajectories)
        importance = torch.linspace(0.0, 1.0, n_points)
        save_path = tmp_path / "baseline_model.pt"

        model = train_model(
            trajectories=trajectories,
            queries=queries,
            epochs=1,
            importance=importance,
            point_batch_size=None,
            save_path=str(save_path),
            model_type="baseline",
        )
        assert isinstance(model, TrajectoryQDSModel)
        assert save_path.exists()

    def test_turn_aware_model_pads_7_feature_points(self):
        trajectories = _make_trajectories(n_features=7)
        queries = _make_queries()
        n_points = sum(t.shape[0] for t in trajectories)
        importance = torch.linspace(0.0, 1.0, n_points)

        model = train_model(
            trajectories=trajectories,
            queries=queries,
            epochs=1,
            importance=importance,
            point_batch_size=None,
            model_type="turn_aware",
        )
        assert isinstance(model, TurnAwareQDSModel)
        assert model.point_encoder[0].in_features == 8
