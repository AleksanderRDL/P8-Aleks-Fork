"""Tests for AIS data loading and synthetic trajectory generation."""
import numpy as np
import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.ais_loader import generate_synthetic_ais_data, _compute_turn_scores
from src.data.ais_loader import load_ais_csv
from src.data.trajectory_dataset import TrajectoryDataset

class TestGenerateSyntheticAISData:
    def test_returns_list(self):
        trajs = generate_synthetic_ais_data(n_ships=3, n_points_per_ship=20)
        assert isinstance(trajs, list)
        assert len(trajs) == 3

    def test_trajectory_shape(self):
        trajs = generate_synthetic_ais_data(n_ships=5, n_points_per_ship=50)
        for traj in trajs:
            assert traj.shape == (50, 8), f"Expected (50, 8), got {traj.shape}"

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
        assert trajs[0].shape == (10, 8)

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

    def test_turn_score_column_present(self):
        """turn_score column (index 7) must be present in trajectory tensors."""
        trajs = generate_synthetic_ais_data(n_ships=3, n_points_per_ship=20)
        for traj in trajs:
            assert traj.shape[1] == 8, "Trajectory must have 8 columns including turn_score"

    def test_turn_score_range(self):
        """turn_score values must be in [0, 1]."""
        trajs = generate_synthetic_ais_data(n_ships=5, n_points_per_ship=30)
        for traj in trajs:
            turn = traj[:, 7]
            assert (turn >= 0.0).all(), "turn_score must be >= 0"
            assert (turn <= 1.0).all(), "turn_score must be <= 1"

    def test_turn_score_endpoints_zero(self):
        """Endpoints must have turn_score == 0 (no direction change for first/last point)."""
        trajs = generate_synthetic_ais_data(n_ships=5, n_points_per_ship=20)
        for traj in trajs:
            assert traj[0, 7] == 0.0, "First point must have turn_score=0"
            assert traj[-1, 7] == 0.0, "Last point must have turn_score=0"

    def test_turn_score_short_trajectory(self):
        """Trajectories with fewer than 3 points must have all turn scores = 0."""
        trajs = generate_synthetic_ais_data(n_ships=2, n_points_per_ship=2)
        for traj in trajs:
            assert (traj[:, 7] == 0.0).all(), "All turn scores must be 0 for 2-point trajectory"

    def test_turn_score_default_method_is_heading(self):
        """Default synthetic generation must match explicit heading mode."""
        default_traj = generate_synthetic_ais_data(n_ships=1, n_points_per_ship=30)[0]
        heading_traj = generate_synthetic_ais_data(
            n_ships=1,
            n_points_per_ship=30,
            turn_score_method="heading",
        )[0]
        assert torch.allclose(default_traj, heading_traj)

    def test_turn_score_invalid_method_raises(self):
        """Unknown turn-score methods must raise a clear ValueError."""
        with pytest.raises(ValueError, match="Unknown turn_score_method"):
            generate_synthetic_ais_data(
                n_ships=1,
                n_points_per_ship=5,
                turn_score_method="bad_method",
            )


class TestTurnScoreComputationMethods:
    def test_vectorized_geometry_turn_score_right_angle(self):
        """A 90-degree turn should map to a 0.5 geometry turn score."""
        lat_lon = np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            dtype=np.float32,
        )
        scores = _compute_turn_scores(lat_lon, method="geometry")
        assert scores.shape == (3, 1)
        assert scores[0, 0] == 0.0
        assert scores[-1, 0] == 0.0
        assert float(scores[1, 0]) == pytest.approx(0.5, rel=1e-6)

    def test_vectorized_heading_turn_score_wraparound(self):
        """Heading delta should use shortest circular distance across 360/0."""
        lat_lon = np.array(
            [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]],
            dtype=np.float32,
        )
        heading = np.array([350.0, 0.0, 10.0], dtype=np.float32)
        scores = _compute_turn_scores(lat_lon, heading=heading, method="heading")
        assert scores.shape == (3, 1)
        assert scores[0, 0] == 0.0
        assert scores[-1, 0] == 0.0
        assert float(scores[1, 0]) == pytest.approx(20.0 / 180.0, rel=1e-6)

    def test_heading_method_requires_heading_input(self):
        """Heading method must reject calls without heading values."""
        lat_lon = np.zeros((3, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="require a heading"):
            _compute_turn_scores(lat_lon, method="heading")


class TestTrajectoryDataset:
    def test_len(self):
        trajs = generate_synthetic_ais_data(n_ships=4, n_points_per_ship=10)
        ds = TrajectoryDataset(trajs)
        assert len(ds) == 4

    def test_getitem_shape(self):
        trajs = generate_synthetic_ais_data(n_ships=3, n_points_per_ship=15)
        ds = TrajectoryDataset(trajs)
        item = ds[0]
        assert item.shape == (15, 8)

    def test_get_all_points(self):
        trajs = generate_synthetic_ais_data(n_ships=3, n_points_per_ship=20)
        ds = TrajectoryDataset(trajs)
        all_pts = ds.get_all_points()
        assert all_pts.shape == (60, 8)

    def test_get_trajectory_boundaries(self):
        trajs = generate_synthetic_ais_data(n_ships=3, n_points_per_ship=20)
        ds = TrajectoryDataset(trajs)
        boundaries = ds.get_trajectory_boundaries()
        assert len(boundaries) == 3
        starts = [b[0] for b in boundaries]
        ends = [b[1] for b in boundaries]
        assert starts == [0, 20, 40]
        assert ends == [20, 40, 60]


@pytest.mark.integration
class TestLoadAISCSV:
    def test_missing_required_column_raises(self, tmp_path):
        csv_path = tmp_path / "missing_speed.csv"
        csv_path.write_text(
            "mmsi,lat,lon,heading,timestamp\n"
            "111000111,55.0,10.0,90.0,1000\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="missing required columns"):
            load_ais_csv(str(csv_path))

    def test_synthesizes_heading_and_timestamp_when_missing(self, tmp_path):
        csv_path = tmp_path / "minimal.csv"
        csv_path.write_text(
            "mmsi,lat,lon,speed\n"
            "111000111,55.0,10.0,12.0\n"
            "111000111,55.1,10.1,12.5\n"
            "222000222,56.0,11.0,9.0\n"
            "222000222,56.1,11.1,9.5\n",
            encoding="utf-8",
        )

        trajectories = load_ais_csv(str(csv_path))
        assert len(trajectories) == 2

        for traj in trajectories:
            assert traj.shape == (2, 8)
            assert torch.allclose(traj[:, 4], torch.zeros(2, dtype=traj.dtype))
            assert traj[0, 5].item() == pytest.approx(1.0)
            assert traj[-1, 6].item() == pytest.approx(1.0)
            time_delta = traj[1:, 0] - traj[:-1, 0]
            assert torch.allclose(time_delta, torch.full_like(time_delta, 60.0))

    def test_column_aliases_and_partial_timestamps(self, tmp_path):
        csv_path = tmp_path / "aliases.csv"
        csv_path.write_text(
            "MMSI,Latitude,Longitude,SOG,COG,Time\n"
            "123456789,55.00,10.00,12.0,90.0,1000\n"
            "123456789,55.10,10.10,12.5,92.0,\n"
            "123456789,55.20,10.20,13.0,94.0,1120\n",
            encoding="utf-8",
        )

        trajectories = load_ais_csv(str(csv_path))
        assert len(trajectories) == 1
        traj = trajectories[0]

        assert traj.shape == (3, 8)
        assert traj[0, 0].item() == pytest.approx(1000.0)
        assert traj[1, 0].item() == pytest.approx(1060.0)
        assert traj[2, 0].item() == pytest.approx(1120.0)
