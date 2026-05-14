"""Data splitting and dataset construction for experiment runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    train_source_ids: list[int] | None
    split_diagnostics: dict[str, Any]


@dataclass
class ExperimentDatasets:
    """Flattened datasets and trajectory boundaries for one experiment run."""

    train_points: torch.Tensor
    test_points: torch.Tensor
    selection_points: torch.Tensor | None
    train_boundaries: list[tuple[int, int]]
    test_boundaries: list[tuple[int, int]]
    selection_boundaries: list[tuple[int, int]] | None


def _source_counts(source_ids: list[int] | None) -> dict[str, int]:
    """Return stable source-id counts for split diagnostics."""
    counts: dict[int, int] = {}
    for source_id in source_ids or []:
        counts[int(source_id)] = counts.get(int(source_id), 0) + 1
    return {str(source_id): counts[source_id] for source_id in sorted(counts)}


def _random_validation_indices(
    *,
    train_count: int,
    val_fraction: float,
    generator: torch.Generator,
) -> set[int]:
    """Return random validation indices while preserving at least one train trajectory."""
    val_count = max(1, int(train_count * val_fraction)) if train_count > 1 else 0
    val_count = min(val_count, max(0, train_count - 1))
    permutation = torch.randperm(train_count, generator=generator).tolist()
    return set(permutation[:val_count])


def _source_stratified_validation_indices(
    *,
    source_ids: list[int] | None,
    val_fraction: float,
    generator: torch.Generator,
) -> set[int]:
    """Hold out validation trajectories from each train source without emptying a source."""
    if source_ids is None:
        raise ValueError("validation_split_mode='source_stratified' requires train trajectory source ids.")
    grouped: dict[int, list[int]] = {}
    for trajectory_idx, source_id in enumerate(source_ids):
        grouped.setdefault(int(source_id), []).append(trajectory_idx)
    val_indices: set[int] = set()
    for source_id in sorted(grouped):
        indices = grouped[source_id]
        if len(indices) <= 1:
            continue
        source_val_count = max(1, int(len(indices) * val_fraction))
        source_val_count = min(source_val_count, len(indices) - 1)
        permutation = torch.randperm(len(indices), generator=generator).tolist()
        val_indices.update(indices[position] for position in permutation[:source_val_count])
    if not val_indices and len(source_ids) > 1:
        raise ValueError(
            "validation_split_mode='source_stratified' could not hold out validation trajectories "
            "without emptying a train source."
        )
    return val_indices


def _fallback_validation_indices(
    *,
    mode: str,
    train_source_ids: list[int] | None,
    train_count: int,
    val_fraction: float,
    generator: torch.Generator,
) -> set[int]:
    """Resolve fallback validation indices for separate train/eval CSV mode."""
    if mode == "random":
        return _random_validation_indices(
            train_count=train_count,
            val_fraction=val_fraction,
            generator=generator,
        )
    if mode == "source_stratified":
        return _source_stratified_validation_indices(
            source_ids=train_source_ids,
            val_fraction=val_fraction,
            generator=generator,
        )
    raise ValueError(
        "validation_split_mode must be 'random' or 'source_stratified'; "
        f"got {mode!r}."
    )


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
    trajectory_source_ids: list[int] | None = None,
) -> ExperimentDataSplit:
    """Resolve train/eval/selection split for single-dataset and separate-CSV modes."""
    if trajectory_source_ids is not None and len(trajectory_source_ids) != len(trajectories):
        raise ValueError(
            "trajectory_source_ids must match the input trajectories length: "
            f"got {len(trajectory_source_ids)} ids for {len(trajectories)} trajectories."
        )
    selection_traj: list[torch.Tensor] | None = None
    split_diagnostics: dict[str, Any] = {
        "mode": None,
        "validation_split_mode": config.data.validation_split_mode,
        "selection_source": "none",
    }
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
        train_source_ids = (
            [int(trajectory_source_ids[i]) for i in permutation[:train_count]]
            if trajectory_source_ids is not None
            else None
        )
        split_diagnostics.update(
            {
                "mode": "single_dataset",
                "selection_source": "single_dataset_fraction" if selection_traj else "none",
                "train_trajectory_count": len(train_traj),
                "selection_trajectory_count": len(selection_traj or []),
                "eval_trajectory_count": len(test_traj),
                "train_source_counts": _source_counts(train_source_ids),
            }
        )
        print(f"  split mode=single dataset  train={len(train_traj)}  test={len(test_traj)}", flush=True)
    else:
        train_traj = trajectories
        test_traj = eval_trajectories
        train_mmsis = trajectory_mmsis
        test_mmsis = eval_trajectory_mmsis
        train_source_ids = (
            [int(source_id) for source_id in trajectory_source_ids]
            if trajectory_source_ids is not None
            else None
        )
        if validation_trajectories is not None:
            selection_traj = validation_trajectories if needs_validation_score else None
            split_diagnostics.update(
                {
                    "selection_source": "validation_csv" if selection_traj else "none",
                    "validation_split_mode_effective": "explicit_validation_csv",
                }
            )
        elif needs_validation_score:
            train_count = len(train_traj)
            generator = torch.Generator().manual_seed(int(seeds.split_seed))
            val_indices = _fallback_validation_indices(
                mode=config.data.validation_split_mode,
                train_source_ids=train_source_ids,
                train_count=train_count,
                val_fraction=config.data.val_fraction,
                generator=generator,
            )
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
            if train_source_ids is not None and len(train_source_ids) == train_count:
                train_source_ids = [
                    source_id
                    for trajectory_idx, source_id in enumerate(train_source_ids)
                    if trajectory_idx not in val_indices
                ]
            split_diagnostics.update(
                {
                    "selection_source": "train_fallback_split" if selection_traj else "none",
                    "validation_split_mode_effective": config.data.validation_split_mode,
                    "fallback_validation_index_count": len(val_indices),
                    "fallback_validation_source_counts": _source_counts(
                        [
                            int(trajectory_source_ids[idx])
                            for idx in sorted(val_indices)
                        ]
                        if trajectory_source_ids is not None
                        else None
                    ),
                }
            )
        split_diagnostics.update(
            {
                "mode": "separate_csvs",
                "train_trajectory_count": len(train_traj),
                "selection_trajectory_count": len(selection_traj or []),
                "eval_trajectory_count": len(test_traj),
                "train_source_counts": _source_counts(train_source_ids),
            }
        )
        print(f"  split mode=separate CSVs  train={len(train_traj)}  eval={len(test_traj)}", flush=True)
    if selection_traj:
        print(f"  checkpoint-selection validation={len(selection_traj)} trajectories", flush=True)

    return ExperimentDataSplit(
        train_traj=train_traj,
        test_traj=test_traj,
        selection_traj=selection_traj,
        train_mmsis=train_mmsis,
        test_mmsis=test_mmsis,
        train_source_ids=train_source_ids,
        split_diagnostics=split_diagnostics,
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
