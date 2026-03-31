"""Tests for pure/helper utilities in the experiment pipeline."""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.experiments.experiment_config import (
    BaselineConfig,
    ModelConfig,
    ModelSimplificationResult,
    QueryConfig,
)
from src.experiments.experiment_pipeline_helpers import (
    _build_methods_for_evaluation,
    _clean_csv_output_path,
    _generate_queries_for_workload,
    _resolve_effective_max_train_points,
    _resolve_model_variants,
)


def _make_points(n: int = 20) -> torch.Tensor:
    points = torch.zeros(n, 5, dtype=torch.float32)
    points[:, 0] = torch.arange(n, dtype=torch.float32)  # time
    points[:, 1] = 50.0 + torch.linspace(0.0, 1.0, n)  # lat
    points[:, 2] = 10.0 + torch.linspace(0.0, 1.0, n)  # lon
    points[:, 3] = torch.linspace(1.0, 2.0, n)  # speed
    points[:, 4] = 90.0  # heading
    return points


class TestHelperResolution:
    def test_explicit_max_train_points_takes_priority(self):
        model_cfg = ModelConfig(max_train_points=123, model_max_points=50)
        effective, used_auto = _resolve_effective_max_train_points(1_000, model_cfg)
        assert effective == 123
        assert used_auto is False

    def test_auto_cap_uses_model_max_points(self):
        model_cfg = ModelConfig(max_train_points=None, model_max_points=80)
        effective, used_auto = _resolve_effective_max_train_points(500, model_cfg)
        assert effective == 80
        assert used_auto is True

    def test_model_variant_resolution(self):
        assert _resolve_model_variants("baseline") == ["baseline"]
        assert _resolve_model_variants("turn_aware") == ["turn_aware"]
        assert _resolve_model_variants("all") == ["baseline", "turn_aware"]

    def test_clean_csv_output_path(self):
        out = _clean_csv_output_path("/tmp/ais.csv")
        assert out == "/tmp/MLClean-ais.csv"


class TestWorkloadAndMethods:
    def test_unknown_workload_raises(self):
        with pytest.raises(ValueError, match="Unknown workload"):
            _generate_queries_for_workload([], QueryConfig(workload="unknown"))

    def test_skip_baselines_keeps_only_ml_methods(self):
        points = _make_points()
        ml_points = points[:5]
        ml_results = {
            "baseline": ModelSimplificationResult(
                simplified_points=ml_points,
                retained_mask=torch.ones(points.shape[0], dtype=torch.bool),
                scores=torch.ones(points.shape[0], dtype=torch.float32),
                label="ML QDS",
            )
        }
        methods = _build_methods_for_evaluation(
            points=points,
            ml_ratio=0.25,
            ml_results_by_type=ml_results,
            baseline_cfg=BaselineConfig(skip_baselines=True),
        )
        assert list(methods.keys()) == ["ML QDS"]
        assert torch.equal(methods["ML QDS"], ml_points)

    def test_dp_baseline_is_skipped_when_points_exceed_cap(self):
        points = _make_points(n=30)
        ml_results = {
            "baseline": ModelSimplificationResult(
                simplified_points=points[:10],
                retained_mask=torch.ones(points.shape[0], dtype=torch.bool),
                scores=torch.ones(points.shape[0], dtype=torch.float32),
                label="ML QDS",
            )
        }
        methods = _build_methods_for_evaluation(
            points=points,
            ml_ratio=0.33,
            ml_results_by_type=ml_results,
            baseline_cfg=BaselineConfig(skip_baselines=False, dp_max_points=10),
        )
        assert "Random" in methods
        assert "Temporal" in methods
        assert "Douglas-Peucker" not in methods
