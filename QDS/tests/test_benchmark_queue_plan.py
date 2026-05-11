"""Tests for queued benchmark launch helpers."""

from __future__ import annotations

from pathlib import Path

from scripts.validate_benchmark_queue_plan import validate_plan


def test_queue_plan_validation_accepts_child_training_args(tmp_path: Path) -> None:
    plan = tmp_path / "queue_plan.tsv"
    plan.write_text(
        "# run_id\tseed\tchild_extra_args\n"
        "run_a\t42\t--ranking_pairs_per_type 64 --ranking_top_quantile 0.70\n"
        "run_b\t43\t--pointwise_loss_weight 0.50 --mlqds_temporal_fraction 0.25\n",
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
