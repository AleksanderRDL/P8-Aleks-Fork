"""Tests for AIS data loading and synthetic trajectory generation."""
import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.ais_loader import generate_synthetic_ais_data
from src.data.trajectory_dataset import TrajectoryDataset

class TestGenerateSyntheticAISData:
    def test_returns_list(self):
        trajs = generate_synthetic_ais_data(n_ships=3, n_points_per_ship=20)
        assert isinstance(trajs, list)
        assert len(trajs) == 3

    def test_trajectory_shape(self):
        trajs = generate_synthetic_ais_data(n_ships=5, n_points_per_ship=50)
        for traj in trajs:
            assert traj.shape == (50, 7), f"Expected (50, 7), got {traj.shape}"

    def test_trajectory_dtype(self):
        trajs = generate_synthetic_ais_data(n_ships=2, n_points_per_ship=10)
        for traj in trajs:
            assert traj.dtype == torch.float32

    def test_trajectory_columns_range(self):
        """lat starts in [50,60], lon starts in [0,20], speed>=0, heading in [0,360]."""
        trajs = generate_synthetic_ais_data(n_ships=5, n_points_per_ship=50)
        for traj in trajs:
            speed = traj[:, 3]
            heading = traj[:, 4]
            assert (speed >= 0.0).all()
            assert (heading >= 0.0).all() and (heading <= 360.0).all()

    def test_single_ship(self):
        trajs = generate_synthetic_ais_data(n_ships=1, n_points_per_ship=10)
        assert len(trajs) == 1
        assert trajs[0].shape == (10, 7)

    def test_endpoint_flags_set(self):
        """is_start (col 5) and is_end (col 6) must be 1 at first/last point."""
        trajs = generate_synthetic_ais_data(n_ships=3, n_points_per_ship=20)
        for traj in trajs:
            assert traj[0, 5] == 1.0,  "First point must have is_start=1"
            assert traj[-1, 6] == 1.0, "Last point must have is_end=1"
            assert traj[1:-1, 5].sum() == 0.0, "Middle points must have is_start=0"
            assert traj[1:-1, 6].sum() == 0.0, "Middle points must have is_end=0"

    def test_endpoint_flags_single_point(self):
        """A single-point trajectory must have both flags set."""
        trajs = generate_synthetic_ais_data(n_ships=1, n_points_per_ship=1)
        traj = trajs[0]
        assert traj[0, 5] == 1.0, "is_start must be 1 for single-point trajectory"
        assert traj[0, 6] == 1.0, "is_end must be 1 for single-point trajectory"


class TestTrajectoryDataset:
    def test_len(self):
        trajs = generate_synthetic_ais_data(n_ships=4, n_points_per_ship=10)
        ds = TrajectoryDataset(trajs)
        assert len(ds) == 4

    def test_getitem_shape(self):
        trajs = generate_synthetic_ais_data(n_ships=3, n_points_per_ship=15)
        ds = TrajectoryDataset(trajs)
        item = ds[0]
        assert item.shape == (15, 7)

    def test_get_all_points(self):
        trajs = generate_synthetic_ais_data(n_ships=3, n_points_per_ship=20)
        ds = TrajectoryDataset(trajs)
        all_pts = ds.get_all_points()
        assert all_pts.shape == (60, 7)

    def test_get_trajectory_boundaries(self):
        trajs = generate_synthetic_ais_data(n_ships=3, n_points_per_ship=20)
        ds = TrajectoryDataset(trajs)
        boundaries = ds.get_trajectory_boundaries()
        assert len(boundaries) == 3
        starts = [b[0] for b in boundaries]
        ends = [b[1] for b in boundaries]
        assert starts == [0, 20, 40]
        assert ends == [20, 40, 60]
