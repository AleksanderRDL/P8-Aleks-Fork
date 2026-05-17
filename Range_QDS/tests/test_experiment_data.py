"""Tests for experiment data splitting."""

from __future__ import annotations

import pytest
import torch

from experiments.experiment_config import SeedBundle, build_experiment_config
from experiments.experiment_data import prepare_experiment_split


def _trajectory(value: float) -> torch.Tensor:
    """Build a tiny valid trajectory with an identifiable value."""
    return torch.tensor(
        [
            [value, 0.0, 0.0],
            [value + 0.1, 0.1, 1.0],
        ],
        dtype=torch.float32,
    )


def _seeds() -> SeedBundle:
    return SeedBundle(split_seed=123, train_query_seed=0, eval_query_seed=0, torch_seed=0)


def test_source_stratified_validation_holds_out_each_train_source() -> None:
    cfg = build_experiment_config(validation_split_mode="source_stratified")
    train = [_trajectory(float(value)) for value in (0, 1, 2, 10, 11, 12)]
    source_ids = [0, 0, 0, 1, 1, 1]

    split = prepare_experiment_split(
        config=cfg,
        seeds=_seeds(),
        trajectories=train,
        needs_validation_score=True,
        eval_trajectories=[_trajectory(100.0)],
        trajectory_source_ids=source_ids,
    )

    selection_values = {int(trajectory[0, 0].item()) for trajectory in split.selection_traj or []}
    assert len(selection_values) == 2
    assert any(value < 10 for value in selection_values)
    assert any(value >= 10 for value in selection_values)
    assert split.split_diagnostics["validation_split_mode_effective"] == "source_stratified"
    assert split.split_diagnostics["fallback_validation_source_counts"] == {"0": 1, "1": 1}
    assert split.split_diagnostics["train_source_counts"] == {"0": 2, "1": 2}


def test_source_stratified_validation_requires_source_ids() -> None:
    cfg = build_experiment_config(validation_split_mode="source_stratified")

    with pytest.raises(ValueError, match="requires train trajectory source ids"):
        prepare_experiment_split(
            config=cfg,
            seeds=_seeds(),
            trajectories=[_trajectory(0.0), _trajectory(1.0)],
            needs_validation_score=True,
            eval_trajectories=[_trajectory(100.0)],
        )


def test_single_dataset_split_respects_configured_fractions() -> None:
    cfg = build_experiment_config(train_fraction=0.34, val_fraction=0.33)
    trajectories = [_trajectory(float(value)) for value in range(12)]

    split = prepare_experiment_split(
        config=cfg,
        seeds=_seeds(),
        trajectories=trajectories,
        needs_validation_score=True,
    )

    assert len(split.train_traj) == 4
    assert len(split.selection_traj or []) == 3
    assert len(split.test_traj) == 5
    assert split.split_diagnostics["train_trajectory_count"] == 4
    assert split.split_diagnostics["selection_trajectory_count"] == 3
    assert split.split_diagnostics["eval_trajectory_count"] == 5


def test_single_dataset_split_rejects_fractions_without_eval_holdout() -> None:
    cfg = build_experiment_config(train_fraction=0.80, val_fraction=0.20)

    with pytest.raises(ValueError, match="less than 1.0"):
        prepare_experiment_split(
            config=cfg,
            seeds=_seeds(),
            trajectories=[_trajectory(float(value)) for value in range(12)],
            needs_validation_score=True,
        )
