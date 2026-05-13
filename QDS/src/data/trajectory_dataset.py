"""Trajectory dataset helper for flattened and boundary-based access. See src/data/README.md for details."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.data.trajectory_index import boundaries_from_trajectories


@dataclass
class TrajectoryDataset:
    """Container around a list of trajectory tensors. See src/data/README.md for details."""

    trajectories: list[torch.Tensor]

    def get_all_points(self) -> torch.Tensor:
        """Return all trajectory points in one tensor. See src/data/README.md for details."""
        if not self.trajectories:
            return torch.zeros((0, 8), dtype=torch.float32)
        return torch.cat(self.trajectories, dim=0)

    def get_trajectory_boundaries(self) -> list[tuple[int, int]]:
        """Return start/end index ranges per trajectory in flattened order. See src/data/README.md for details."""
        return boundaries_from_trajectories(self.trajectories)
