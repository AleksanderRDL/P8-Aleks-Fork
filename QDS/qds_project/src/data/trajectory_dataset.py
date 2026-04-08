"""PyTorch Dataset wrapper for AIS trajectory tensors. See src/data/README.md."""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """PyTorch Dataset for AIS vessel trajectories."""

    def __init__(self, trajectories: List[Tensor]) -> None:
        self.trajectories = trajectories

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Tensor:
        """Return the trajectory tensor at index ``idx``."""
        return self.trajectories[idx]

    def get_all_points(self) -> Tensor:
        """Flatten all trajectories into a single [N, 8] point cloud tensor."""
        return torch.cat(self.trajectories, dim=0)

    def get_trajectory_boundaries(self) -> List[Tuple[int, int]]:
        """Return (start, end) index pairs for each trajectory in the flattened cloud."""
        boundaries: List[Tuple[int, int]] = []
        offset = 0
        for traj in self.trajectories:
            length = traj.shape[0]
            boundaries.append((offset, offset + length))
            offset += length
        return boundaries
