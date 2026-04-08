"""Tests for building structured experiment configuration from flat args."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.experiments.experiment_config import build_experiment_config


class TestBuildExperimentConfig:
    def test_maps_data_and_query_aliases(self):
        cfg = build_experiment_config(
            n_ships=7,
            n_points=123,
            query_spatial_fraction=0.12,
            query_temporal_fraction=0.34,
            query_spatial_lower_quantile=0.05,
            query_spatial_upper_quantile=0.95,
        )
        assert cfg.data.n_ships == 7
        assert cfg.data.n_points_per_ship == 123
        assert cfg.query.spatial_fraction == pytest.approx(0.12)
        assert cfg.query.temporal_fraction == pytest.approx(0.34)
        assert cfg.query.spatial_lower_quantile == pytest.approx(0.05)
        assert cfg.query.spatial_upper_quantile == pytest.approx(0.95)

    def test_ignores_unknown_flat_args(self):
        cfg = build_experiment_config(n_ships=3, unknown_flag="ignored")
        assert cfg.data.n_ships == 3
