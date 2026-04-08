"""Tests for the AIS query generator (uniform, density-biased, and mixed)."""

import torch
import pytest

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.queries.query_generator import (
    generate_uniform_queries,
    generate_density_biased_queries,
    generate_mixed_queries,
    generate_spatiotemporal_queries,
)


def _make_trajectories(n_ships: int = 3, n_points: int = 20) -> list[torch.Tensor]:
    """Create a small set of synthetic trajectory tensors for testing.

    Each trajectory tensor has shape [T, 5] with columns:
    [time, lat, lon, speed, heading].
    """
    torch.manual_seed(42)
    trajectories = []
    for i in range(n_ships):
        t = torch.linspace(0, 100, n_points).unsqueeze(1)  # [T, 1]
        lat = 50.0 + torch.rand(n_points, 1) * 5.0
        lon = -5.0 + torch.rand(n_points, 1) * 10.0
        speed = torch.rand(n_points, 1) * 20.0
        heading = torch.rand(n_points, 1) * 360.0
        trajectories.append(torch.cat([t, lat, lon, speed, heading], dim=1))
    return trajectories


class TestGenerateUniformQueries:
    def test_output_shape(self):
        trajs = _make_trajectories()
        queries = generate_uniform_queries(trajs, n_queries=50)
        assert queries.shape == (50, 6), f"Expected (50, 6), got {queries.shape}"

    def test_min_less_than_max(self):
        trajs = _make_trajectories()
        queries = generate_uniform_queries(trajs, n_queries=100)
        assert (queries[:, 0] < queries[:, 1]).all(), "lat_min must be < lat_max"
        assert (queries[:, 2] < queries[:, 3]).all(), "lon_min must be < lon_max"
        assert (queries[:, 4] < queries[:, 5]).all(), "time_start must be < time_end"

    def test_within_data_bounds(self):
        trajs = _make_trajectories()
        all_points = torch.cat(trajs, dim=0)
        lat_min, lat_max = float(all_points[:, 1].min()), float(all_points[:, 1].max())
        lon_min, lon_max = float(all_points[:, 2].min()), float(all_points[:, 2].max())
        time_min, time_max = float(all_points[:, 0].min()), float(all_points[:, 0].max())

        queries = generate_uniform_queries(trajs, n_queries=100)

        assert (queries[:, 0] >= lat_min - 1e-4).all()
        assert (queries[:, 1] <= lat_max + 1e-4).all()
        assert (queries[:, 2] >= lon_min - 1e-4).all()
        assert (queries[:, 3] <= lon_max + 1e-4).all()
        assert (queries[:, 4] >= time_min - 1e-4).all()
        assert (queries[:, 5] <= time_max + 1e-4).all()

    def test_single_query(self):
        trajs = _make_trajectories()
        queries = generate_uniform_queries(trajs, n_queries=1)
        assert queries.shape == (1, 6)

    def test_default_n_queries(self):
        trajs = _make_trajectories()
        queries = generate_uniform_queries(trajs)
        assert queries.shape[0] == 100


class TestGenerateDensityBiasedQueries:
    def test_output_shape(self):
        trajs = _make_trajectories()
        queries = generate_density_biased_queries(trajs, n_queries=50)
        assert queries.shape == (50, 6)

    def test_min_less_than_max(self):
        trajs = _make_trajectories()
        queries = generate_density_biased_queries(trajs, n_queries=100)
        assert (queries[:, 0] < queries[:, 1]).all(), "lat_min must be < lat_max"
        assert (queries[:, 2] < queries[:, 3]).all(), "lon_min must be < lon_max"
        assert (queries[:, 4] < queries[:, 5]).all(), "time_start must be < time_end"

    def test_within_data_bounds(self):
        trajs = _make_trajectories()
        all_points = torch.cat(trajs, dim=0)
        lat_min, lat_max = float(all_points[:, 1].min()), float(all_points[:, 1].max())
        lon_min, lon_max = float(all_points[:, 2].min()), float(all_points[:, 2].max())
        time_min, time_max = float(all_points[:, 0].min()), float(all_points[:, 0].max())

        queries = generate_density_biased_queries(trajs, n_queries=100)

        assert (queries[:, 0] >= lat_min - 1e-4).all()
        assert (queries[:, 1] <= lat_max + 1e-4).all()
        assert (queries[:, 2] >= lon_min - 1e-4).all()
        assert (queries[:, 3] <= lon_max + 1e-4).all()
        assert (queries[:, 4] >= time_min - 1e-4).all()
        assert (queries[:, 5] <= time_max + 1e-4).all()

    def test_default_n_queries(self):
        trajs = _make_trajectories()
        queries = generate_density_biased_queries(trajs)
        assert queries.shape[0] == 100


class TestGenerateMixedQueries:
    def test_output_shape(self):
        trajs = _make_trajectories()
        queries = generate_mixed_queries(trajs, total_queries=80)
        assert queries.shape == (80, 6)

    def test_density_ratio_one(self):
        """density_ratio=1.0 → all queries from density-biased generator."""
        trajs = _make_trajectories()
        queries = generate_mixed_queries(trajs, total_queries=50, density_ratio=1.0)
        assert queries.shape == (50, 6)

    def test_density_ratio_zero(self):
        """density_ratio=0.0 → all queries from uniform generator."""
        trajs = _make_trajectories()
        queries = generate_mixed_queries(trajs, total_queries=50, density_ratio=0.0)
        assert queries.shape == (50, 6)

    def test_mixed_split(self):
        """density_ratio=0.7 with 10 queries → 7 density + 3 uniform = 10 total."""
        trajs = _make_trajectories()
        queries = generate_mixed_queries(trajs, total_queries=10, density_ratio=0.7)
        assert queries.shape == (10, 6)

    def test_min_less_than_max(self):
        trajs = _make_trajectories()
        queries = generate_mixed_queries(trajs, total_queries=100, density_ratio=0.5)
        assert (queries[:, 0] < queries[:, 1]).all()
        assert (queries[:, 2] < queries[:, 3]).all()
        assert (queries[:, 4] < queries[:, 5]).all()

    def test_invalid_density_ratio(self):
        trajs = _make_trajectories()
        with pytest.raises(ValueError):
            generate_mixed_queries(trajs, total_queries=10, density_ratio=1.5)

    def test_default_args(self):
        trajs = _make_trajectories()
        queries = generate_mixed_queries(trajs)
        assert queries.shape == (100, 6)


class TestGenerateSpatiotemporalQueriesBackwardCompat:
    """Verify the legacy function still works and delegates correctly."""

    def test_anchor_to_data_true(self):
        trajs = _make_trajectories()
        queries = generate_spatiotemporal_queries(trajs, n_queries=20, anchor_to_data=True)
        assert queries.shape == (20, 6)

    def test_anchor_to_data_false(self):
        trajs = _make_trajectories()
        queries = generate_spatiotemporal_queries(trajs, n_queries=20, anchor_to_data=False)
        assert queries.shape == (20, 6)

    def test_min_less_than_max(self):
        trajs = _make_trajectories()
        queries = generate_spatiotemporal_queries(trajs, n_queries=50)
        assert (queries[:, 0] < queries[:, 1]).all()
        assert (queries[:, 2] < queries[:, 3]).all()
        assert (queries[:, 4] < queries[:, 5]).all()
