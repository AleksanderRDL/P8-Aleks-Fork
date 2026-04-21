"""Trajectory-window batching utilities for ranking training. See src/training/README.md for details."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TrajectoryBatch:
    """Padded trajectory window batch container. See src/training/README.md for details."""

    points: torch.Tensor
    padding_mask: torch.Tensor
    trajectory_ids: torch.Tensor
    global_indices: torch.Tensor


def build_trajectory_windows(
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    window_length: int,
    stride: int,
) -> list[TrajectoryBatch]:
    """Build trajectory-local windows with no cross-trajectory attention. See src/training/README.md for details."""
    windows: list[TrajectoryBatch] = []
    for tid, (start, end) in enumerate(boundaries):
        traj = points[start:end]
        n = traj.shape[0]
        if n <= window_length:
            pad = window_length - n
            p = torch.cat([traj, torch.zeros((pad, traj.shape[1]), dtype=traj.dtype)], dim=0)
            mask = torch.zeros((window_length,), dtype=torch.bool)
            if pad > 0:
                mask[n:] = True
            idx = torch.cat([torch.arange(start, end), torch.full((pad,), -1, dtype=torch.long)])
            windows.append(
                TrajectoryBatch(
                    points=p.unsqueeze(0),
                    padding_mask=mask.unsqueeze(0),
                    trajectory_ids=torch.tensor([tid], dtype=torch.long),
                    global_indices=idx.unsqueeze(0),
                )
            )
            continue

        for w_start in range(0, n, stride):
            w_end = min(n, w_start + window_length)
            chunk = traj[w_start:w_end]
            if chunk.shape[0] < window_length:
                pad = window_length - chunk.shape[0]
                chunk = torch.cat([chunk, torch.zeros((pad, traj.shape[1]), dtype=traj.dtype)], dim=0)
                mask = torch.zeros((window_length,), dtype=torch.bool)
                mask[window_length - pad :] = True
                idx = torch.cat(
                    [
                        torch.arange(start + w_start, start + w_end),
                        torch.full((pad,), -1, dtype=torch.long),
                    ]
                )
            else:
                mask = torch.zeros((window_length,), dtype=torch.bool)
                idx = torch.arange(start + w_start, start + w_end)
            windows.append(
                TrajectoryBatch(
                    points=chunk.unsqueeze(0),
                    padding_mask=mask.unsqueeze(0),
                    trajectory_ids=torch.tensor([tid], dtype=torch.long),
                    global_indices=idx.unsqueeze(0),
                )
            )
            if w_end == n:
                break
    return windows
