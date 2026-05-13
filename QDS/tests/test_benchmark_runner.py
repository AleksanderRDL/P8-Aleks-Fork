"""Tests for range benchmark run helpers."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.experiments.benchmark_runner import (
    DEFAULT_WORKLOADS,
    DEFAULT_PROFILE,
    BenchmarkDataSources,
    PURE_WORKLOADS,
    _child_run_dir,
    _format_report_table,
    _index_entry,
    _parse_name_list,
    _profile_args,
    _resolve_data_sources,
    _row_from_run,
    _run_capture_streaming,
    _runner_environment_metadata,
    _write_family_indexes,
)
from src.experiments.experiment_config import build_experiment_config
from src.experiments.experiment_pipeline_helpers import _validation_query_count, resolve_workload_maps


def _profile_core_args() -> list[str]:
    """Expected range_testing_baseline child args without data-source/cap flags."""
    return [
        "--n_queries",
        "80",
        "--query_coverage",
        "0.20",
        "--range_spatial_fraction",
        "0.0165",
        "--range_time_fraction",
        "0.033",
        "--range_spatial_km",
        "2.2",
        "--range_time_hours",
        "5.0",
        "--range_footprint_jitter",
        "0.0",
        "--range_diagnostics_mode",
        "cached",
        "--float32_matmul_precision",
        "high",
        "--allow_tf32",
        "--amp_mode",
        "bf16",
        "--query_chunk_size",
        "2048",
        "--train_batch_size",
        "64",
        "--inference_batch_size",
        "64",
        "--max_queries",
        "2048",
        "--compression_ratio",
        "0.05",
        "--epochs",
        "20",
        "--early_stopping_patience",
        "5",
        "--checkpoint_smoothing_window",
        "1",
        "--checkpoint_full_f1_every",
        "2",
        "--checkpoint_candidate_pool_size",
        "2",
        "--loss_objective",
        "budget_topk",
        "--budget_loss_ratios",
        "0.01,0.02,0.05,0.10",
        "--budget_loss_temperature",
        "0.10",
        "--mlqds_temporal_fraction",
        "0.25",
        "--mlqds_score_mode",
        "rank",
        "--mlqds_score_temperature",
        "1.00",
        "--mlqds_rank_confidence_weight",
        "0.15",
        "--mlqds_diversity_bonus",
        "0.00",
        "--residual_label_mode",
        "temporal",
        "--range_label_mode",
        "usefulness",
        "--range_boundary_prior_weight",
        "0.0",
        "--checkpoint_selection_metric",
        "f1",
        "--checkpoint_f1_variant",
        "range_usefulness",
    ]


def test_benchmark_name_list_rejects_mixed_workloads() -> None:
    assert _parse_name_list("range,knn", allowed=PURE_WORKLOADS, arg_name="--workloads") == ["range", "knn"]
    assert _parse_name_list(None, allowed=DEFAULT_WORKLOADS, arg_name="--workloads") == ["range"]

    with pytest.raises(ValueError, match="unknown"):
        _parse_name_list("range,mixed", allowed=PURE_WORKLOADS, arg_name="--workloads")


def test_experiment_workload_resolution_is_pure_only() -> None:
    assert resolve_workload_maps("range") == ({"range": 1.0}, {"range": 1.0})
    assert resolve_workload_maps("knn") == ({"knn": 1.0}, {"knn": 1.0})

    with pytest.raises(ValueError, match="no longer supported"):
        resolve_workload_maps("mixed")


def test_profile_args_own_runtime_and_checkpoint_defaults() -> None:
    args = _profile_core_args()

    assert args[args.index("--float32_matmul_precision") + 1] == "high"
    assert "--allow_tf32" in args
    assert args[args.index("--amp_mode") + 1] == "bf16"


def test_benchmark_environment_metadata_is_scoped_to_parent_process() -> None:
    environment = _runner_environment_metadata()

    assert environment["scope"] == "benchmark_runner_parent_process"
    assert "rows[*].child_torch_runtime" in environment["note"]


def test_benchmark_row_records_effective_child_torch_runtime(tmp_path) -> None:
    run_json = {
        "config": {
            "model": {
                "mlqds_temporal_fraction": 0.25,
                "mlqds_diversity_bonus": 0.05,
                "mlqds_score_mode": "rank",
                "mlqds_score_temperature": 1.0,
                "mlqds_rank_confidence_weight": 0.15,
                "residual_label_mode": "temporal",
                "range_label_mode": "usefulness",
                "range_boundary_prior_weight": 0.0,
                "loss_objective": "budget_topk",
                "budget_loss_ratios": [0.01, 0.02, 0.05, 0.10],
                "budget_loss_temperature": 0.10,
                "checkpoint_full_f1_every": 3,
                "checkpoint_candidate_pool_size": 2,
                "checkpoint_f1_variant": "range_usefulness",
                "float32_matmul_precision": "high",
                "allow_tf32": True,
                "amp_mode": "bf16",
                "compression_ratio": 0.05,
            }
        },
        "oracle_diagnostic": {
            "kind": "additive_label_greedy",
            "exact_optimum": False,
        },
        "workload_diagnostics": {
            "train": {
                "range_signal": {
                    "labels": {
                        "positive_label_mass": 12.5,
                        "component_label_mass_basis": "pre_clamp_component_contributions",
                        "component_positive_label_mass_fraction": {
                            "range_point_f1": 0.22,
                            "range_ship_f1": 0.13,
                            "range_ship_coverage": 0.12,
                            "range_entry_exit_f1": 0.09,
                            "range_crossing_f1": 0.04,
                            "range_temporal_coverage": 0.11,
                            "range_gap_coverage": 0.10,
                            "range_turn_coverage": 0.08,
                            "range_shape_score": 0.11,
                        },
                    }
                }
            }
        },
        "training_target_diagnostics": {
            "positive_label_mass": 11.0,
            "budget_rows": [
                {
                    "total_budget_ratio": 0.01,
                    "effective_fill_budget_ratio": 0.008,
                    "temporal_base_label_mass_fraction": 0.20,
                    "residual_label_mass_fraction": 0.80,
                    "residual_positive_label_fraction": 0.10,
                },
                {
                    "total_budget_ratio": 0.05,
                    "effective_fill_budget_ratio": 0.041,
                    "temporal_base_label_mass_fraction": 0.35,
                    "residual_label_mass_fraction": 0.65,
                    "residual_positive_label_fraction": 0.20,
                },
            ],
        },
        "matched": {
            "MLQDS": {
                "aggregate_f1": 0.40,
                "range_point_f1": 0.40,
                "range_usefulness_score": 0.42,
                "range_ship_coverage": 0.64,
                "range_entry_exit_f1": 0.25,
                "range_crossing_f1": 0.48,
                "range_gap_coverage": 0.31,
                "range_turn_coverage": 0.52,
                "range_usefulness_schema_version": 7,
            },
            "uniform": {"aggregate_f1": 0.35},
            "DouglasPeucker": {"aggregate_f1": 0.36},
        },
        "learned_fill_diagnostics": {
            "TemporalRandomFill": {
                "range_point_f1": 0.38,
                "range_usefulness_score": 0.41,
            },
            "TemporalOracleFill": {
                "range_point_f1": 0.55,
                "range_usefulness_score": 0.70,
            },
        },
        "best_epoch": 2,
        "best_selection_score": 0.42,
        "best_f1": 0.42,
        "training_history": [
            {"epoch": 0.0, "pred_std": 0.0, "collapse_warning": 1.0},
            {"epoch": 1.0, "pred_std": 0.2},
        ],
        "torch_runtime": {
            "float32_matmul_precision": "high",
            "tf32_matmul_allowed": True,
            "tf32_cudnn_allowed": True,
            "amp": {"enabled": True, "dtype": "bfloat16"},
        }
    }

    row = _row_from_run(
        workload="range",
        run_label="custom_runtime",
        command=["python", "-m", "src.experiments.run_ais_experiment"],
        returncode=0,
        elapsed_seconds=1.0,
        run_dir=tmp_path,
        stdout_path=tmp_path / "stdout.log",
        run_json_path=tmp_path / "example_run.json",
        timings={"phase_timings": [], "epoch_timings": [], "inference_step_timings": []},
        run_json=run_json,
    )

    assert row["float32_matmul_precision"] == "high"
    assert row["allow_tf32"] is True
    assert row["amp_mode"] == "bf16"
    assert row["child_float32_matmul_precision"] == "high"
    assert row["child_tf32_matmul_allowed"] is True
    assert row["child_amp_enabled"] is True
    assert row["child_amp_dtype"] == "bfloat16"
    assert row["mlqds_temporal_fraction"] == 0.25
    assert row["mlqds_diversity_bonus"] == 0.05
    assert row["mlqds_score_mode"] == "rank"
    assert row["mlqds_score_temperature"] == 1.0
    assert row["mlqds_rank_confidence_weight"] == 0.15
    assert row["residual_label_mode"] == "temporal"
    assert row["range_label_mode"] == "usefulness"
    assert row["loss_objective"] == "budget_topk"
    assert row["budget_loss_ratios"] == [0.01, 0.02, 0.05, 0.10]
    assert row["budget_loss_temperature"] == 0.10
    assert row["checkpoint_full_f1_every"] == 3
    assert row["checkpoint_candidate_pool_size"] == 2
    assert row["best_selection_score"] == 0.42
    assert row["best_f1"] == 0.42
    assert row["range_boundary_prior_weight"] == 0.0
    assert row["range_boundary_prior_enabled"] is False
    assert row["train_positive_label_mass"] == 12.5
    assert row["train_label_mass_basis"] == "pre_clamp_component_contributions"
    assert row["train_label_mass_range_point_f1"] == 0.22
    assert row["train_label_mass_range_ship_f1"] == 0.13
    assert row["train_label_mass_range_ship_coverage"] == 0.12
    assert row["train_label_mass_range_entry_exit_f1"] == 0.09
    assert row["train_label_mass_range_crossing_f1"] == 0.04
    assert row["train_label_mass_range_temporal_coverage"] == 0.11
    assert row["train_label_mass_range_gap_coverage"] == 0.10
    assert row["train_label_mass_range_turn_coverage"] == 0.08
    assert row["train_label_mass_range_shape_score"] == 0.11
    assert row["train_target_positive_label_mass"] == 11.0
    assert row["train_target_budget_ratio"] == 0.05
    assert row["train_target_effective_fill_budget_ratio"] == 0.041
    assert row["train_target_temporal_base_label_mass_fraction"] == 0.35
    assert row["train_target_residual_label_mass_fraction"] == 0.65
    assert row["train_target_residual_positive_label_fraction"] == 0.20
    assert row["mlqds_range_point_f1"] == 0.40
    assert row["mlqds_range_usefulness_score"] == 0.42
    assert row["mlqds_range_ship_coverage"] == 0.64
    assert row["mlqds_range_entry_exit_f1"] == 0.25
    assert row["mlqds_range_crossing_f1"] == 0.48
    assert row["mlqds_range_gap_coverage"] == 0.31
    assert row["mlqds_range_turn_coverage"] == 0.52
    assert row["range_usefulness_schema_version"] == 7
    assert row["temporal_random_fill_range_point_f1"] == 0.38
    assert row["temporal_random_fill_range_usefulness_score"] == 0.41
    assert row["temporal_oracle_fill_range_point_f1"] == 0.55
    assert row["temporal_oracle_fill_range_usefulness_score"] == 0.70
    assert row["mlqds_vs_temporal_random_fill_range_usefulness"] == pytest.approx(0.01)
    assert row["temporal_oracle_fill_gap_range_usefulness"] == pytest.approx(0.28)
    assert row["collapse_warning_any"] is True
    assert row["collapse_warning_count"] == 1
    assert row["best_epoch_collapse_warning"] is False
    assert row["min_pred_std"] == 0.0
    assert row["best_epoch_pred_std"] == 0.2
    assert row["oracle_kind"] == "additive_label_greedy"
    assert row["oracle_exact_optimum"] is False
    assert row["mlqds_vs_uniform_f1"] == pytest.approx(0.05)
    assert row["mlqds_vs_douglas_peucker_f1"] == pytest.approx(0.04)


def test_profile_args_use_csv_when_provided() -> None:
    args = argparse.Namespace(
        csv_path="../AISDATA/cleaned/day.csv",
        train_csv_path=None,
        validation_csv_path=None,
        eval_csv_path=None,
        cache_dir="artifacts/cache/benchmark",
        refresh_cache=True,
        min_points_per_segment=4,
        max_points_per_segment=128,
        max_time_gap_seconds=3600.0,
        max_segments=16,
        max_trajectories=8,
    )
    data_sources = BenchmarkDataSources(csv_path="../AISDATA/cleaned/day.csv")

    assert _profile_args(DEFAULT_PROFILE, args, data_sources) == [
        "--csv_path",
        "../AISDATA/cleaned/day.csv",
        *_profile_core_args(),
        "--min_points_per_segment",
        "4",
        "--max_time_gap_seconds",
        "3600.0",
        "--max_points_per_segment",
        "128",
        "--max_segments",
        "16",
        "--max_trajectories",
        "8",
        "--cache_dir",
        "artifacts/cache/benchmark",
        "--refresh_cache",
    ]


def test_profile_args_use_three_day_train_validation_eval_sources() -> None:
    args = argparse.Namespace(
        csv_path=None,
        train_csv_path="../AISDATA/cleaned/day1.csv",
        validation_csv_path="../AISDATA/cleaned/day2.csv",
        eval_csv_path="../AISDATA/cleaned/day3.csv",
        cache_dir="artifacts/cache/benchmark",
        refresh_cache=True,
        min_points_per_segment=4,
        max_points_per_segment=None,
        max_time_gap_seconds=3600.0,
        max_segments=None,
        max_trajectories=None,
    )
    data_sources = BenchmarkDataSources(
        train_csv_path="../AISDATA/cleaned/day1.csv",
        validation_csv_path="../AISDATA/cleaned/day2.csv",
        eval_csv_path="../AISDATA/cleaned/day3.csv",
    )

    assert _profile_args(DEFAULT_PROFILE, args, data_sources, include_refresh_cache=False) == [
        "--train_csv_path",
        "../AISDATA/cleaned/day1.csv",
        "--validation_csv_path",
        "../AISDATA/cleaned/day2.csv",
        "--eval_csv_path",
        "../AISDATA/cleaned/day3.csv",
        *_profile_core_args(),
        "--min_points_per_segment",
        "4",
        "--max_time_gap_seconds",
        "3600.0",
        "--cache_dir",
        "artifacts/cache/benchmark",
    ]


def test_testing_baseline_profile_uses_requested_training_shape() -> None:
    args = argparse.Namespace(
        csv_path=None,
        train_csv_path="../AISDATA/cleaned/day1.csv",
        validation_csv_path="../AISDATA/cleaned/day2.csv",
        eval_csv_path="../AISDATA/cleaned/day3.csv",
        cache_dir="artifacts/cache/benchmark",
        refresh_cache=False,
        min_points_per_segment=4,
        max_points_per_segment=None,
        max_time_gap_seconds=3600.0,
        max_segments=None,
        max_trajectories=None,
    )
    data_sources = BenchmarkDataSources(
        train_csv_path="../AISDATA/cleaned/day1.csv",
        validation_csv_path="../AISDATA/cleaned/day2.csv",
        eval_csv_path="../AISDATA/cleaned/day3.csv",
    )

    profile_args = _profile_args(DEFAULT_PROFILE, args, data_sources, include_refresh_cache=False)

    assert profile_args[6 : 6 + len(_profile_core_args())] == _profile_core_args()
    assert "--max_points_per_segment" not in profile_args
    assert "--max_segments" not in profile_args
    assert "--max_trajectories" not in profile_args


def test_validation_query_count_matches_eval_minimum_query_count() -> None:
    cfg = build_experiment_config(n_queries=80, query_coverage=0.20, max_queries=None)

    assert _validation_query_count(cfg) == 80


def test_csv_config_suppresses_inactive_synthetic_metadata() -> None:
    cfg = build_experiment_config(
        n_ships=24,
        n_points=200,
        train_csv_path="../AISDATA/cleaned/day1.csv",
        validation_csv_path="../AISDATA/cleaned/day2.csv",
        eval_csv_path="../AISDATA/cleaned/day3.csv",
    )

    assert cfg.data.n_ships is None
    assert cfg.data.n_points_per_ship is None


def test_child_run_dir_uses_readable_layout(tmp_path) -> None:
    run_label = "custom_run"

    assert _child_run_dir(tmp_path, "range", run_label, 1) == tmp_path / "custom_run"
    assert _child_run_dir(tmp_path, "knn", run_label, 2) == tmp_path / "knn" / "custom_run"


def test_family_index_upserts_current_status_and_appends_events(tmp_path) -> None:
    args = argparse.Namespace(
        profile=DEFAULT_PROFILE,
        seed=42,
        max_points_per_segment=3000,
        max_segments=None,
        max_trajectories=None,
    )
    run_label = "custom_run"
    sources = BenchmarkDataSources(train_csv_path="day1.csv", validation_csv_path="day2.csv", eval_csv_path="day3.csv")
    git = {"commit": "abc123", "dirty": False}
    running_status = {
        "status": "running",
        "started_at_utc": "2026-05-10T00:00:00+00:00",
        "finished_at_utc": None,
        "exit_status": None,
        "failures": None,
    }
    completed_status = {
        **running_status,
        "status": "completed",
        "finished_at_utc": "2026-05-10T00:01:00+00:00",
        "exit_status": 0,
        "failures": 0,
    }

    _write_family_indexes(
        tmp_path,
        _index_entry(
            run_id="run-a",
            status_payload=running_status,
            args=args,
            workloads=["range"],
            run_label=run_label,
            data_sources=sources,
            results_dir=tmp_path / "runs" / "run-a",
            rows=[],
            git=git,
        ),
    )
    _write_family_indexes(
        tmp_path,
        _index_entry(
            run_id="run-a",
            status_payload=completed_status,
            args=args,
            workloads=["range"],
            run_label=run_label,
            data_sources=sources,
            results_dir=tmp_path / "runs" / "run-a",
            rows=[{"run_label": "custom_run", "mlqds_f1": 0.4}],
            git=git,
        ),
    )

    with open(tmp_path / "runs_index.csv", encoding="utf-8", newline="") as f:
        index_rows = list(csv.DictReader(f))
    events_text = (tmp_path / "runs_index_events.jsonl").read_text(encoding="utf-8")
    assert len(index_rows) == 1
    assert index_rows[0]["run_id"] == "run-a"
    assert index_rows[0]["status"] == "completed"
    assert index_rows[0]["run_label"] == "custom_run"
    assert index_rows[0]["best_mlqds_f1"] == "0.4"
    assert index_rows[0]["best_mlqds_run_label"] == "custom_run"
    assert events_text.count('"run_id": "run-a"') == 2


def test_resolve_data_sources_selects_three_cleaned_days(tmp_path) -> None:
    (tmp_path / "aisdk-2026-02-02_cleaned.csv").write_text("x\n", encoding="utf-8")
    (tmp_path / "aisdk-2026-02-03_cleaned.csv").write_text("x\n", encoding="utf-8")
    (tmp_path / "aisdk-2026-02-04_cleaned.csv").write_text("x\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("ignore\n", encoding="utf-8")
    args = argparse.Namespace(csv_path=str(tmp_path), train_csv_path=None, validation_csv_path=None, eval_csv_path=None)

    sources = _resolve_data_sources(args)

    assert sources.csv_path is None
    assert sources.train_csv_path == str(tmp_path / "aisdk-2026-02-02_cleaned.csv")
    assert sources.validation_csv_path == str(tmp_path / "aisdk-2026-02-03_cleaned.csv")
    assert sources.eval_csv_path == str(tmp_path / "aisdk-2026-02-04_cleaned.csv")
    assert sources.csv_sources == (sources.train_csv_path, sources.validation_csv_path, sources.eval_csv_path)


def test_resolve_data_sources_requires_paired_train_eval() -> None:
    args = argparse.Namespace(csv_path=None, train_csv_path="train.csv", validation_csv_path=None, eval_csv_path=None)

    with pytest.raises(ValueError, match="supplied together"):
        _resolve_data_sources(args)


def test_resolve_data_sources_rejects_duplicate_explicit_splits(tmp_path) -> None:
    day = tmp_path / "day.csv"
    day.write_text("x\n", encoding="utf-8")
    args = argparse.Namespace(
        csv_path=None,
        train_csv_path=str(day),
        validation_csv_path=None,
        eval_csv_path=str(day),
    )

    with pytest.raises(ValueError, match="must be distinct"):
        _resolve_data_sources(args)


def test_benchmark_markdown_table_is_compact() -> None:
    table = _format_report_table(
        [
            {
                "workload": "range",
                "run_label": "custom",
                "returncode": 0,
                "elapsed_seconds": 12.34567,
                "epoch_mean_seconds": 1.25,
                "peak_allocated_mb": 123.0,
                "best_f1": 0.5,
                "mlqds_f1": 0.4,
                "uniform_f1": 0.3,
                "douglas_peucker_f1": 0.2,
                "mlqds_vs_uniform_f1": 0.1,
                "mlqds_vs_douglas_peucker_f1": 0.2,
                "mlqds_latency_ms": 10.0,
                "collapse_warning": False,
            }
        ]
    )

    assert "| workload | run_label |" in table
    assert "train_label_mass_range_point_f1" in table
    assert "| range | custom | 0 | 12.3457 |" in table


def test_run_capture_streaming_writes_log_and_console(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    stdout_path = tmp_path / "child" / "stdout.log"

    result = _run_capture_streaming(
        [sys.executable, "-c", "print('alpha', flush=True); print('beta', flush=True)"],
        cwd=tmp_path,
        stdout_path=stdout_path,
    )

    assert result.returncode == 0
    assert result.stdout == "alpha\nbeta\n"
    assert stdout_path.read_text(encoding="utf-8") == "alpha\nbeta\n"
    assert "alpha\nbeta\n" in capsys.readouterr().out


def test_run_capture_streaming_retains_bounded_tail_but_keeps_timings(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    stdout_path = tmp_path / "child" / "stdout.log"
    command = (
        "print('[train-model] done in 1.23s', flush=True)\n"
        "for i in range(20):\n"
        "    print(f'filler-{i:03d}-' + 'x' * 40, flush=True)\n"
    )

    result = _run_capture_streaming(
        [sys.executable, "-c", command],
        cwd=tmp_path,
        stdout_path=stdout_path,
        max_stdout_chars=64,
    )

    full_log = stdout_path.read_text(encoding="utf-8")
    assert result.returncode == 0
    assert result.stdout_truncated is True
    assert len(result.stdout) <= 64
    assert full_log.startswith("[train-model] done in 1.23s\n")
    assert len(full_log) > len(result.stdout)
    assert result.timings["phase_timings"] == [{"name": "train-model", "seconds": 1.23}]
    assert "[train-model] done in 1.23s\n" in capsys.readouterr().out


def test_mark_benchmark_failed_updates_stale_running_status_and_family_index(tmp_path) -> None:
    family = tmp_path / "range_family"
    results_dir = family / "runs" / "stale-run"
    results_dir.mkdir(parents=True)
    status_file = results_dir / "run_status.json"
    status_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "stale-run",
                "status": "running",
                "started_at_utc": "2026-05-10T00:00:00+00:00",
                "finished_at_utc": None,
                "exit_status": None,
                "failures": None,
                "message": "benchmark run started",
                "results_dir": str(results_dir),
            }
        ),
        encoding="utf-8",
    )
    with open(family / "runs_index.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run_id", "status", "finished_at_utc", "exit_status", "failures", "results_dir"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "run_id": "stale-run",
                "status": "running",
                "finished_at_utc": "",
                "exit_status": "",
                "failures": "",
                "results_dir": str(results_dir),
            }
        )

    script = Path(__file__).resolve().parents[1] / "scripts" / "mark_benchmark_failed.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--status-file",
            str(status_file),
            "--exit-status",
            "-9",
            "--message",
            "killed",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(status_file.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["exit_status"] == -9
    assert payload["failures"] == 1
    assert payload["message"] == "killed"

    with open(family / "runs_index.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["status"] == "failed"
    assert rows[0]["exit_status"] == "-9"
    assert rows[0]["failures"] == "1"
    assert '"run_id": "stale-run"' in (family / "runs_index_events.jsonl").read_text(encoding="utf-8")
