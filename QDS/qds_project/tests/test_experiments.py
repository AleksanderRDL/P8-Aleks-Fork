"""Tests for experiment-layer edge-case behavior."""

import os
import tempfile
import torch
import pytest
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.experiments.experiment_config import DataConfig, ExperimentConfig, ModelConfig
from src.experiments.experiment_pipeline_helpers import _simplify_with_model
from src.experiments.workload_runner import _load_workload_data
from src.models.trajectory_qds_model import TrajectoryQDSModel


def _make_two_trajectory_points() -> torch.Tensor:
    points = torch.zeros(20, 8)
    points[:, 0] = torch.arange(20, dtype=torch.float32)  # time
    points[:, 1] = 50.0 + torch.linspace(0.0, 1.0, 20)  # lat
    points[:, 2] = torch.linspace(0.0, 1.0, 20)  # lon
    points[:, 3] = 10.0  # speed

    # Trajectory boundaries: [0, 10), [10, 20)
    points[0, 5] = 1.0
    points[9, 6] = 1.0
    points[10, 5] = 1.0
    points[19, 6] = 1.0
    return points


class TestTargetRatioSimplification:
    def test_target_ratio_keeps_endpoints_and_minimum_per_trajectory(self):
        points = _make_two_trajectory_points()
        boundaries = [(0, 10), (10, 20)]
        queries = torch.tensor([[49.0, 52.0, -1.0, 2.0, -1.0, 25.0]], dtype=torch.float32)

        # Deliberately make non-endpoint points highest, so unconstrained top-k
        # would not include endpoints.
        importance = torch.zeros(20, dtype=torch.float32)
        importance[4] = 1.0
        importance[14] = 0.9

        model_cfg = ModelConfig(
            target_ratio=0.10,
            compression_ratio=None,
            min_points_per_trajectory=2,
            model_max_points=0,  # force query-score path for deterministic behavior
        )
        model = TrajectoryQDSModel()

        result = _simplify_with_model(
            points=points,
            queries=queries,
            importance=importance,
            trained_model=model,
            model_variant="baseline",
            model_cfg=model_cfg,
            trajectory_boundaries=boundaries,
        )

        mask = result.retained_mask
        assert mask[0].item() and mask[9].item()
        assert mask[10].item() and mask[19].item()
        for start, end in boundaries:
            assert mask[start:end].sum().item() >= 2


class TestLoadWorkloadData:
    def test_missing_csv_path_raises(self):
        cfg = ExperimentConfig(data=DataConfig(csv_path="/tmp/this_file_should_not_exist.csv"))
        with pytest.raises(FileNotFoundError, match="--csv_path"):
            _load_workload_data(cfg, torch.device("cpu"))

    def test_empty_csv_raises_clear_error(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
            tmp.write("mmsi,lat,lon,speed,heading,timestamp\n")
            csv_path = tmp.name

        try:
            cfg = ExperimentConfig(data=DataConfig(csv_path=csv_path))
            with pytest.raises(ValueError, match="No trajectories available"):
                _load_workload_data(cfg, torch.device("cpu"))
        finally:
            os.remove(csv_path)
