"""Tests for pure-workload benchmark matrix helpers."""

from __future__ import annotations

import argparse

import pytest

from src.experiments.benchmark_matrix import (
    DEFAULT_VARIANTS,
    PURE_WORKLOADS,
    _format_markdown_table,
    _parse_name_list,
    _profile_args,
    _selected_variants,
)
from src.experiments.experiment_pipeline_helpers import resolve_workload_mixes


def test_matrix_name_list_rejects_mixed_workloads() -> None:
    assert _parse_name_list("range,knn", allowed=PURE_WORKLOADS, arg_name="--workloads") == ["range", "knn"]

    with pytest.raises(ValueError, match="unknown"):
        _parse_name_list("range,mixed", allowed=PURE_WORKLOADS, arg_name="--workloads")


def test_experiment_workload_resolution_is_pure_only() -> None:
    assert resolve_workload_mixes(None, None, workload_keyword="range") == ({"range": 1.0}, {"range": 1.0})

    with pytest.raises(ValueError, match="no longer supported"):
        resolve_workload_mixes(None, None, workload_keyword="mixed")

    with pytest.raises(ValueError, match="exactly one query type"):
        resolve_workload_mixes("range=0.5,knn=0.5", None, workload_keyword="range")


def test_selected_variants_defaults_and_combined_variant() -> None:
    variants = _selected_variants(None)

    assert [variant.name for variant in variants] == list(DEFAULT_VARIANTS)
    assert variants[-1].checkpoint_f1_variant == "combined"
    assert variants[-1].amp_mode == "bf16"


def test_profile_args_use_csv_when_provided() -> None:
    args = argparse.Namespace(
        csv_path="../AISDATA/cleaned/day.csv",
        cache_dir="artifacts/cache/matrix",
        refresh_cache=True,
        max_points_per_segment=128,
        max_segments=16,
    )

    assert _profile_args("medium", args) == [
        "--csv_path",
        "../AISDATA/cleaned/day.csv",
        "--n_queries",
        "64",
        "--epochs",
        "8",
        "--max_points_per_segment",
        "128",
        "--max_segments",
        "16",
        "--cache_dir",
        "artifacts/cache/matrix",
        "--refresh_cache",
    ]


def test_matrix_markdown_table_is_compact() -> None:
    table = _format_markdown_table(
        [
            {
                "workload": "range",
                "variant": "tf32",
                "returncode": 0,
                "elapsed_seconds": 12.34567,
                "epoch_mean_seconds": 1.25,
                "peak_allocated_mb": 123.0,
                "best_f1": 0.5,
                "mlqds_f1": 0.4,
                "uniform_f1": 0.3,
                "douglas_peucker_f1": 0.2,
                "mlqds_latency_ms": 10.0,
                "collapse_warning": False,
            }
        ]
    )

    assert "| workload | variant |" in table
    assert "| range | tf32 | 0 | 12.3457 |" in table
