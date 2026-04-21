"""Tests that short training keeps non-collapsed typed predictions. See src/training/README.md for details."""

from __future__ import annotations

from src.data.trajectory_dataset import TrajectoryDataset
from src.experiments.experiment_config import build_experiment_config
from src.queries.query_generator import generate_typed_query_workload
from src.training.train_model import train_model


def test_training_does_not_collapse(synthetic_dataset) -> None:
    """Assert prediction spread and typewise rank correlation stay healthy. See src/training/README.md for details."""
    trajectories, _ = synthetic_dataset
    ds = TrajectoryDataset(trajectories)
    boundaries = ds.get_trajectory_boundaries()

    cfg = build_experiment_config(epochs=4, n_queries=80)
    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=80,
        workload_mix={"range": 0.25, "knn": 0.25, "similarity": 0.25, "clustering": 0.25},
        seed=77,
    )
    out = train_model(
        train_trajectories=trajectories,
        train_boundaries=boundaries,
        workload=workload,
        model_config=cfg.model,
        seed=77,
    )

    last = out.history[-1]
    assert last["pred_std"] > 0.02
    for t in range(4):
        assert last[f"kendall_tau_t{t}"] > 0.3
