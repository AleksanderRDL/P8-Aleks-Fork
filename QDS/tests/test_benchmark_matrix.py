"""Tests for range benchmark matrix helpers."""

from __future__ import annotations

import argparse

import pytest

from src.experiments.benchmark_matrix import (
    DEFAULT_VARIANTS,
    DEFAULT_WORKLOADS,
    MatrixDataSources,
    PURE_WORKLOADS,
    _format_markdown_table,
    _parse_name_list,
    _profile_args,
    _resolve_data_sources,
    _selected_variants,
)
from src.experiments.experiment_pipeline_helpers import resolve_workload_mixes


def test_matrix_name_list_rejects_mixed_workloads() -> None:
    assert _parse_name_list("range,knn", allowed=PURE_WORKLOADS, arg_name="--workloads") == ["range", "knn"]
    assert _parse_name_list(None, allowed=DEFAULT_WORKLOADS, arg_name="--workloads") == ["range"]

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
        train_csv_path=None,
        eval_csv_path=None,
        cache_dir="artifacts/cache/matrix",
        refresh_cache=True,
        min_points_per_segment=4,
        max_points_per_segment=128,
        max_time_gap_seconds=3600.0,
        max_segments=16,
    )
    data_sources = MatrixDataSources(csv_path="../AISDATA/cleaned/day.csv")

    assert _profile_args("medium", args, data_sources) == [
        "--csv_path",
        "../AISDATA/cleaned/day.csv",
        "--n_queries",
        "64",
        "--epochs",
        "8",
        "--min_points_per_segment",
        "4",
        "--max_time_gap_seconds",
        "3600.0",
        "--max_points_per_segment",
        "128",
        "--max_segments",
        "16",
        "--cache_dir",
        "artifacts/cache/matrix",
        "--refresh_cache",
    ]


def test_profile_args_use_two_day_train_eval_sources() -> None:
    args = argparse.Namespace(
        csv_path=None,
        train_csv_path="../AISDATA/cleaned/day1.csv",
        eval_csv_path="../AISDATA/cleaned/day2.csv",
        cache_dir="artifacts/cache/matrix",
        refresh_cache=True,
        min_points_per_segment=4,
        max_points_per_segment=500,
        max_time_gap_seconds=3600.0,
        max_segments=512,
    )
    data_sources = MatrixDataSources(
        train_csv_path="../AISDATA/cleaned/day1.csv",
        eval_csv_path="../AISDATA/cleaned/day2.csv",
    )

    assert _profile_args("medium", args, data_sources, include_refresh_cache=False) == [
        "--train_csv_path",
        "../AISDATA/cleaned/day1.csv",
        "--eval_csv_path",
        "../AISDATA/cleaned/day2.csv",
        "--n_queries",
        "64",
        "--epochs",
        "8",
        "--min_points_per_segment",
        "4",
        "--max_time_gap_seconds",
        "3600.0",
        "--max_points_per_segment",
        "500",
        "--max_segments",
        "512",
        "--cache_dir",
        "artifacts/cache/matrix",
    ]


def test_resolve_data_sources_selects_two_cleaned_days(tmp_path) -> None:
    (tmp_path / "aisdk-2026-02-02_cleaned.csv").write_text("x\n", encoding="utf-8")
    (tmp_path / "aisdk-2026-02-03_cleaned.csv").write_text("x\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("ignore\n", encoding="utf-8")
    args = argparse.Namespace(csv_path=str(tmp_path), train_csv_path=None, eval_csv_path=None)

    sources = _resolve_data_sources(args)

    assert sources.csv_path is None
    assert sources.train_csv_path == str(tmp_path / "aisdk-2026-02-02_cleaned.csv")
    assert sources.eval_csv_path == str(tmp_path / "aisdk-2026-02-03_cleaned.csv")
    assert sources.csv_sources == (sources.train_csv_path, sources.eval_csv_path)


def test_resolve_data_sources_requires_paired_train_eval() -> None:
    args = argparse.Namespace(csv_path=None, train_csv_path="train.csv", eval_csv_path=None)

    with pytest.raises(ValueError, match="supplied together"):
        _resolve_data_sources(args)


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
