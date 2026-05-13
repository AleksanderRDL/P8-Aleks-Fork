"""Data splitting and dataset construction for experiment runs."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from data.trajectory_dataset import TrajectoryDataset
from experiments.experiment_config import ExperimentConfig, SeedBundle


@dataclass
class ExperimentDataSplit:
    """Resolved train, selection-validation, and evaluation trajectory splits."""

    train_traj: list[torch.Tensor]
    test_traj: list[torch.Tensor]
    selection_traj: list[torch.Tensor] | None
    train_mmsis: list[int] | None
    test_mmsis: list[int] | None


@dataclass
class ExperimentDatasets:
    """Flattened datasets and trajectory boundaries for one experiment run."""

    train_points: torch.Tensor
    test_points: torch.Tensor
    selection_points: torch.Tensor | None
    train_boundaries: list[tuple[int, int]]
    test_boundaries: list[tuple[int, int]]
    selection_boundaries: list[tuple[int, int]] | None


def prepare_experiment_split(
    *,
    config: ExperimentConfig,
    seeds: SeedBundle,
    trajectories: list[torch.Tensor],
    needs_validation_score: bool,
    trajectory_mmsis: list[int] | None = None,
    validation_trajectories: list[torch.Tensor] | None = None,
    eval_trajectories: list[torch.Tensor] | None = None,
    eval_trajectory_mmsis: list[int] | None = None,
) -> ExperimentDataSplit:
    """Resolve train/eval/selection split for single-dataset and separate-CSV modes."""
    selection_traj: list[torch.Tensor] | None = None
    if eval_trajectories is None:
        trajectory_count = len(trajectories)
        generator = torch.Generator().manual_seed(int(seeds.split_seed))
        permutation = torch.randperm(trajectory_count, generator=generator).tolist()
        train_count = max(1, int(trajectory_count * config.data.train_fraction))
        val_count = (
            max(1, int(trajectory_count * config.data.val_fraction))
            if trajectory_count - train_count > 1
            else 0
        )
        train_traj = [trajectories[i] for i in permutation[:train_count]]
        val_traj = [trajectories[i] for i in permutation[train_count : train_count + val_count]]
        test_traj = [trajectories[i] for i in permutation[train_count + val_count :]]
        if not test_traj:
            test_traj = val_traj if val_traj else train_traj
        selection_traj = val_traj if needs_validation_score and val_traj else None
        if trajectory_mmsis is not None and len(trajectory_mmsis) == trajectory_count:
            train_mmsis = [trajectory_mmsis[i] for i in permutation[:train_count]]
            test_mmsis = [trajectory_mmsis[i] for i in permutation[train_count + val_count :]]
            if not test_mmsis:
                test_mmsis = (
                    [trajectory_mmsis[i] for i in permutation[train_count : train_count + val_count]]
                    or [trajectory_mmsis[i] for i in permutation[:train_count]]
                )
        else:
            train_mmsis = None
            test_mmsis = None
        print(f"  split mode=single dataset  train={len(train_traj)}  test={len(test_traj)}", flush=True)
    else:
        train_traj = trajectories
        test_traj = eval_trajectories
        train_mmsis = trajectory_mmsis
        test_mmsis = eval_trajectory_mmsis
        if validation_trajectories is not None:
            selection_traj = validation_trajectories if needs_validation_score else None
        elif needs_validation_score:
            train_count = len(train_traj)
            generator = torch.Generator().manual_seed(int(seeds.split_seed))
            permutation = torch.randperm(train_count, generator=generator).tolist()
            val_count = max(1, int(train_count * config.data.val_fraction)) if train_count > 1 else 0
            val_indices = set(permutation[:val_count])
            selection_traj = [
                trajectory
                for trajectory_idx, trajectory in enumerate(train_traj)
                if trajectory_idx in val_indices
            ]
            train_traj = [
                trajectory
                for trajectory_idx, trajectory in enumerate(train_traj)
                if trajectory_idx not in val_indices
            ]
            if train_mmsis is not None and len(train_mmsis) == train_count:
                train_mmsis = [
                    mmsi
                    for trajectory_idx, mmsi in enumerate(train_mmsis)
                    if trajectory_idx not in val_indices
                ]
        print(f"  split mode=separate CSVs  train={len(train_traj)}  eval={len(test_traj)}", flush=True)
    if selection_traj:
        print(f"  checkpoint-selection validation={len(selection_traj)} trajectories", flush=True)

    return ExperimentDataSplit(
        train_traj=train_traj,
        test_traj=test_traj,
        selection_traj=selection_traj,
        train_mmsis=train_mmsis,
        test_mmsis=test_mmsis,
    )


def build_experiment_datasets(data_split: ExperimentDataSplit) -> ExperimentDatasets:
    """Build flattened trajectory datasets for train, eval, and optional selection split."""
    train_ds = TrajectoryDataset(data_split.train_traj)
    test_ds = TrajectoryDataset(data_split.test_traj)
    selection_ds = TrajectoryDataset(data_split.selection_traj) if data_split.selection_traj else None
    return ExperimentDatasets(
        train_points=train_ds.get_all_points(),
        test_points=test_ds.get_all_points(),
        selection_points=selection_ds.get_all_points() if selection_ds is not None else None,
        train_boundaries=train_ds.get_trajectory_boundaries(),
        test_boundaries=test_ds.get_trajectory_boundaries(),
        selection_boundaries=selection_ds.get_trajectory_boundaries() if selection_ds is not None else None,
    )
