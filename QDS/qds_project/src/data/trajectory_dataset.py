"""
trajectory_dataset.py

PyTorch Dataset wrapper for AIS trajectory tensors.

Each trajectory is a tensor of shape [T, 5] with columns:
[time, lat, lon, speed, heading].
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """PyTorch Dataset for AIS vessel trajectories.

    Wraps a list of variable-length trajectory tensors and provides
    utilities for accessing individual trajectories or the full flattened
    point cloud.

    Args:
        trajectories: List of tensors, each of shape [T, 5] with columns
                      [time, lat, lon, speed, heading].
    """

    def __init__(self, trajectories: List[Tensor]) -> None:
        self.trajectories = trajectories

    def __len__(self) -> int:
        """Return the number of trajectories in the dataset."""
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Tensor:
        """Return the trajectory tensor at index ``idx``.

        Args:
            idx: Trajectory index.

        Returns:
            Tensor of shape [T, 5].
        """
        return self.trajectories[idx]

    def get_all_points(self) -> Tensor:
        """Flatten all trajectories into a single point cloud tensor.

        Returns:
            Tensor of shape [N, 5] where N is the total number of points
            across all trajectories.
        """
        return torch.cat(self.trajectories, dim=0)

    def get_trajectory_boundaries(self) -> List[Tuple[int, int]]:
        """Return the start and end indices of each trajectory in the flattened point cloud.

        Returns:
            List of (start, end) index pairs (exclusive end) — one per trajectory.
        """
        boundaries: List[Tuple[int, int]] = []
        offset = 0
        for traj in self.trajectories:
            length = traj.shape[0]
            boundaries.append((offset, offset + length))
            offset += length
        return boundaries
