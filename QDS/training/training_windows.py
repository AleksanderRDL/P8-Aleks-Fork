"""Trajectory-window helpers for training and validation loops."""

from __future__ import annotations

import torch

from queries.query_types import NUM_QUERY_TYPES
from training.trajectory_batching import TrajectoryBatch


def _window_has_positive_supervision(
    window: TrajectoryBatch,
    training_target: torch.Tensor,
    labelled_mask: torch.Tensor,
) -> bool:
    """Return whether a pure-workload window has positive supervision."""
    global_indices = window.global_indices.reshape(-1)
    valid_points = global_indices >= 0
    if not bool(valid_points.any().item()):
        return False
    valid_indices = global_indices[valid_points].to(device=training_target.device, dtype=torch.long)
    return bool((labelled_mask[valid_indices] & (training_target[valid_indices] > 0)).any().item())


def _filter_supervised_windows(
    windows: list[TrajectoryBatch],
    training_target: torch.Tensor,
    labelled_mask: torch.Tensor,
    active_type_id: int,
) -> tuple[list[TrajectoryBatch], torch.Tensor]:
    """Drop windows that cannot contribute loss for the active pure workload."""
    if not windows:
        return windows, torch.zeros((NUM_QUERY_TYPES,), dtype=torch.long)

    kept: list[TrajectoryBatch] = []
    filtered_zero_windows = torch.zeros((NUM_QUERY_TYPES,), dtype=torch.long)
    for window in windows:
        if _window_has_positive_supervision(window, training_target, labelled_mask):
            kept.append(window)
            continue
        filtered_zero_windows[active_type_id] += 1

    if not kept:
        return windows, torch.zeros((NUM_QUERY_TYPES,), dtype=torch.long)
    return kept, filtered_zero_windows


def _trajectory_batch_to_device(batch: TrajectoryBatch, device: torch.device) -> TrajectoryBatch:
    """Move one already-batched trajectory window group to the model device."""
    return TrajectoryBatch(
        points=batch.points.to(device=device, non_blocking=True),
        padding_mask=batch.padding_mask.to(device=device, non_blocking=True),
        trajectory_ids=batch.trajectory_ids,
        global_indices=batch.global_indices.to(device=device, non_blocking=True),
    )
