"""Pytest fixtures for the AIS-QDS test suite. See experiments/README.md for details."""

from __future__ import annotations

import pytest

from data.ais_loader import generate_synthetic_ais_data
from data.trajectory_dataset import TrajectoryDataset


@pytest.fixture
def synthetic_dataset() -> tuple[list, object]:
    """Create synthetic trajectories and dataset helper. See data/README.md for details."""
    traj = generate_synthetic_ais_data(n_ships=12, n_points_per_ship=120, seed=123)
    ds = TrajectoryDataset(traj)
    return traj, ds
