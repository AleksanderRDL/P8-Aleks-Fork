"""Guardrails for the pre-rework Range_QDS cleanup checkpoint."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.benchmark_profiles import (
    BLIND_EXPECTED_USEFULNESS_PROFILE,
    BLIND_RETAINED_FREQUENCY_PROFILE,
    BLIND_TEACHER_DISTILL_PROFILE,
    DEFAULT_PROFILE,
    RANGE_WORKLOAD_V1_WORKLOAD_BLIND_V2_PROFILE,
    benchmark_profile,
    benchmark_profile_args,
    benchmark_profile_settings,
)
from experiments.benchmark_report import _row_from_run
from training.model_features import model_type_metadata
from training.training_targets import LEGACY_RANGE_TARGET_MODES, QUERY_USEFUL_V1_TARGET_MODES


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_root_makefile_points_to_range_qds() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    range_dir_var = "RANGE_" + "QDS" + "_DIR"

    assert f"{range_dir_var} := $(REPO_ROOT)/Range_QDS" in makefile
    assert "QDS" + "_DIR := $(REPO_ROOT)/" + "QDS" not in makefile
    assert f"-C $({range_dir_var}) test" in makefile
    assert f"-C $({range_dir_var}) lint" in makefile
    assert f"-C $({range_dir_var}) typecheck" in makefile


def test_pyproject_uses_range_qds_paths() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'where = [".", "Range_QDS"]' in pyproject
    assert 'testpaths = ["Range_QDS/tests"]' in pyproject
    assert 'pythonpath = [".", "Range_QDS"]' in pyproject
    assert 'src = [".", "Range_QDS"]' in pyproject
    assert '"QDS"' not in pyproject


@pytest.mark.parametrize(
    "profile_name",
    [
        DEFAULT_PROFILE,
        BLIND_EXPECTED_USEFULNESS_PROFILE,
        BLIND_RETAINED_FREQUENCY_PROFILE,
        BLIND_TEACHER_DISTILL_PROFILE,
    ],
)
def test_legacy_profiles_final_success_allowed_false(profile_name: str) -> None:
    profile = benchmark_profile(profile_name)
    settings = benchmark_profile_settings(profile_name)

    assert profile.final_success_allowed is False
    assert settings["profile_diagnostic_only"] is True
    assert settings["primary_metric_family"] == "RangeUsefulLegacy"
    assert settings["final_success_allowed"] is False
    assert settings["final_product_candidate"] is False


def test_query_driven_v2_profile_is_final_candidate() -> None:
    profile = benchmark_profile(RANGE_WORKLOAD_V1_WORKLOAD_BLIND_V2_PROFILE)
    settings = benchmark_profile_settings(RANGE_WORKLOAD_V1_WORKLOAD_BLIND_V2_PROFILE)
    args = benchmark_profile_args(RANGE_WORKLOAD_V1_WORKLOAD_BLIND_V2_PROFILE)

    assert profile.model_type == "workload_blind_range_v2"
    assert profile.range_training_target_mode == "query_useful_v1_factorized"
    assert profile.workload_profile_id == "range_workload_v1"
    assert profile.selector_type == "learned_segment_budget_v1"
    assert profile.range_train_workload_replicates == 4
    assert profile.range_max_coverage_overshoot == 0.0075
    assert settings["primary_metric_family"] == "QueryUsefulV1"
    assert settings["final_success_allowed"] is True
    assert settings["range_train_workload_replicates"] == 4
    assert args[args.index("--range_train_workload_replicates") + 1] == "4"


def test_legacy_and_query_useful_target_modes_are_separated() -> None:
    assert "retained_frequency" in LEGACY_RANGE_TARGET_MODES
    assert "historical_prior_retained_frequency" in LEGACY_RANGE_TARGET_MODES
    assert "query_useful_v1_factorized" not in LEGACY_RANGE_TARGET_MODES
    assert QUERY_USEFUL_V1_TARGET_MODES == frozenset({"query_useful_v1_factorized"})


def test_historical_prior_metadata_blocks_success() -> None:
    assert model_type_metadata("historical_prior") == {
        "model_family": "historical_prior_knn",
        "trainable_final_candidate": False,
        "final_success_allowed": False,
    }
    student_metadata = model_type_metadata("historical_prior_student")
    assert student_metadata["requires_ablation_against_standalone_knn"] is True
    assert student_metadata["final_success_allowed"] is False


def test_benchmark_row_separates_final_claim_from_legacy_range_useful(tmp_path: Path) -> None:
    row = _row_from_run(
        workload="range",
        run_label="legacy",
        command=["python", "-m", "experiments.run_ais_experiment"],
        returncode=0,
        elapsed_seconds=1.0,
        run_dir=tmp_path,
        stdout_path=tmp_path / "stdout.txt",
        run_json_path=tmp_path / "example_run.json",
        timings={},
        run_json={
            "final_claim_summary": {
                "primary_metric": None,
                "status": "not_available_until_query_useful_v1",
                "final_success_allowed": False,
            },
            "legacy_range_useful_summary": {
                "metric": "RangeUsefulLegacy",
                "diagnostic_only": True,
            },
            "matched": {
                "MLQDS": {"range_usefulness_score": 0.7, "range_point_f1": 0.6},
                "uniform": {"range_usefulness_score": 0.5, "range_point_f1": 0.4},
                "DouglasPeucker": {"range_usefulness_score": 0.4, "range_point_f1": 0.3},
            },
            "config": {
                "model": {
                    "model_type": "historical_prior",
                    "compression_ratio": 0.05,
                    "range_training_target_mode": "retained_frequency",
                },
                "query": {},
                "data": {},
                "baselines": {},
            },
            "workload_blind_protocol": {
                "enabled": True,
                "primary_masks_frozen_before_eval_query_scoring": True,
                "audit_masks_frozen_before_eval_query_scoring": True,
            },
        },
    )

    assert row["final_claim_status"] == "not_available_until_query_useful_v1"
    assert row["final_success_allowed"] is False
    assert row["legacy_range_useful_diagnostic_only"] is True
    assert row["model_metadata_model_family"] == "historical_prior_knn"
