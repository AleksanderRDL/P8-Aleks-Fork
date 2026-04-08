"""Tests for experiment CLI parsing and validation."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.experiments.experiment_cli import parse_and_validate_experiment_args


class TestParseAndValidateExperimentArgs:
    def test_default_args_parse(self):
        kwargs = parse_and_validate_experiment_args([])
        assert kwargs["n_ships"] == 10
        assert kwargs["n_points"] == 100
        assert kwargs["compression_ratio"] == pytest.approx(0.2)

    def test_zero_compression_ratio_maps_to_none(self):
        kwargs = parse_and_validate_experiment_args(["--compression_ratio", "0"])
        assert kwargs["compression_ratio"] is None

    def test_save_csv_requires_csv_path(self):
        with pytest.raises(SystemExit):
            parse_and_validate_experiment_args(["--save_csv"])

    def test_target_ratio_requires_global_threshold_mode(self):
        with pytest.raises(SystemExit):
            parse_and_validate_experiment_args(["--target_ratio", "0.5"])

    def test_query_quantile_order_must_be_strict(self):
        with pytest.raises(SystemExit):
            parse_and_validate_experiment_args(
                [
                    "--query_spatial_lower_quantile",
                    "0.7",
                    "--query_spatial_upper_quantile",
                    "0.7",
                ]
            )

    def test_optional_positive_int_flags_reject_zero(self):
        with pytest.raises(SystemExit):
            parse_and_validate_experiment_args(["--model_max_points", "0"])

    def test_target_ratio_allowed_when_global_mode_enabled(self):
        kwargs = parse_and_validate_experiment_args(
            ["--target_ratio", "0.5", "--compression_ratio", "0"]
        )
        assert kwargs["target_ratio"] == pytest.approx(0.5)
        assert kwargs["compression_ratio"] is None

    def test_density_ratio_out_of_range_rejected(self):
        with pytest.raises(SystemExit):
            parse_and_validate_experiment_args(["--density_ratio", "1.1"])

    def test_query_spatial_fraction_must_be_strictly_positive(self):
        with pytest.raises(SystemExit):
            parse_and_validate_experiment_args(["--query_spatial_fraction", "0"])

    def test_positive_int_flags_reject_zero(self):
        with pytest.raises(SystemExit):
            parse_and_validate_experiment_args(["--n_queries", "0"])
