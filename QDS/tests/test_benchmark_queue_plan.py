"""Tests for queued benchmark launch helpers."""

from __future__ import annotations

from pathlib import Path
import subprocess

from scripts.validate_benchmark_queue_plan import validate_plan


def test_queue_plan_validation_accepts_child_training_args(tmp_path: Path) -> None:
    plan = tmp_path / "queue_plan.tsv"
    plan.write_text(
        "# run_id\tseed\tchild_extra_args\n"
        "run_a\t42\t--ranking_pairs_per_type 64 --ranking_top_quantile 0.70\n"
        "run_b\t43\t--pointwise_loss_weight 0.50 --mlqds_temporal_fraction 0.25\n"
        "run_c\t44\t--mlqds_score_mode rank_confidence --mlqds_rank_confidence_weight 0.15\n",
        encoding="utf-8",
    )

    assert validate_plan(plan) == []


def test_queue_plan_validation_rejects_unknown_child_args(tmp_path: Path) -> None:
    plan = tmp_path / "queue_plan.tsv"
    plan.write_text("run_a\t42\t--definitely_not_a_real_arg 1\n", encoding="utf-8")

    errors = validate_plan(plan)

    assert len(errors) == 1
    assert "run_a" in errors[0]
    assert "--definitely_not_a_real_arg" in errors[0]


def test_tracked_range_coverage_compression_grid_plan_validates() -> None:
    plan = Path(__file__).resolve().parents[1] / "benchmark_plans" / "range_coverage_compression_grid.tsv"

    assert validate_plan(plan) == []


def test_queue_launcher_script_keeps_failure_marking_and_parses() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_benchmark_queue_tmux.sh"

    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text(encoding="utf-8")

    assert "status=${PIPESTATUS[0]}" in text
    assert "scripts/mark_benchmark_failed.py" in text
    assert "CONTINUE_ON_FAILURE" in text
