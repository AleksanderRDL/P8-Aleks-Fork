"""Tests pipeline reporting of higher-is-better aggregate F1. See src/evaluation/README.md for details."""

from __future__ import annotations

from src.experiments.experiment_config import build_experiment_config
from src.experiments.experiment_pipeline_helpers import run_experiment_pipeline


def test_pipeline_reports_f1_scores(synthetic_dataset, tmp_path) -> None:
    """Assert matched-workload metrics use F1 fields and valid score polarity."""
    trajectories, _ = synthetic_dataset
    cfg = build_experiment_config(n_queries=64, epochs=3)

    out = run_experiment_pipeline(
        config=cfg,
        trajectories=trajectories,
        results_dir=str(tmp_path),
        save_simplified_dir=str(tmp_path / "simplified"),
    )

    ml = out.metrics_dump["matched"]["MLQDS"]["aggregate_f1"]
    uni = out.metrics_dump["matched"]["uniform"]["aggregate_f1"]
    assert 0.0 <= ml <= 1.0
    assert 0.0 <= uni <= 1.0
    assert "Random" not in out.metrics_dump["matched"]
    assert "TemporalRandomFill" not in out.metrics_dump["matched"]
    assert "TemporalRandomFill" in out.metrics_dump["learned_fill_diagnostics"]
    assert "TemporalOracleFill" in out.metrics_dump["learned_fill_diagnostics"]
    assert "training_target_diagnostics" in out.metrics_dump
    assert "range_residual_objective_summary" in out.metrics_dump
    assert "workload_distribution_comparison" in out.metrics_dump
    assert "RangePointF1" in out.matched_table
    assert "RangeUseful" in out.matched_table
    assert "AggregateErr" not in out.matched_table
    assert "aggregate_error" not in out.metrics_dump["matched"]["MLQDS"]
    assert "per_type_f1" in out.metrics_dump["matched"]["MLQDS"]
    assert "range_usefulness_score" in out.metrics_dump["matched"]["MLQDS"]
    assert out.metrics_dump["checkpoint_selection_metric"] == "f1"
    assert out.metrics_dump["checkpoint_f1_variant"] == "range_usefulness"
    assert abs(out.metrics_dump["range_usefulness_weight_summary"]["total_weight"] - 1.0) < 1e-9
    assert out.metrics_dump["checkpoint_smoothing_window"] == 1
    residual_summary = out.metrics_dump["range_residual_objective_summary"]
    assert residual_summary["summary_version"] == 1
    assert residual_summary["oracle_notes"]["exact_optimum"] is False
    assert "mlqds_vs_temporal_random_fill_range_usefulness" in residual_summary
    assert "target_residual_label_mass_fraction" in residual_summary
    assert "train_label_component_mass_fraction" in residual_summary
    assert out.metrics_dump["oracle_diagnostic"]["retained_mask_constructor"] == "per_trajectory_topk_with_endpoints"

    # F1 is higher-is-better, so callers should rank with max(), not min().
    scores = [float(metrics["aggregate_f1"]) for metrics in out.metrics_dump["matched"].values()]
    assert max(scores) >= min(scores)
    assert (tmp_path / "simplified" / "ML_simplified_eval.csv").exists()
    assert (tmp_path / "range_residual_objective_summary.json").exists()
    assert not (tmp_path / "simplified" / "ML_simplified_train.csv").exists()
    assert not (tmp_path / "simplified" / "ML_simplified.csv").exists()


def test_core_final_metrics_mode_skips_diagnostic_baselines(synthetic_dataset, tmp_path) -> None:
    trajectories, _ = synthetic_dataset
    cfg = build_experiment_config(n_queries=16, epochs=1, final_metrics_mode="core")

    out = run_experiment_pipeline(
        config=cfg,
        trajectories=trajectories,
        results_dir=str(tmp_path),
    )

    assert out.metrics_dump["final_metrics_mode"] == "core"
    assert "Oracle" not in out.metrics_dump["matched"]
    assert set(out.metrics_dump["learned_fill_diagnostics"]) == {"MLQDS"}
    assert out.metrics_dump["oracle_diagnostic"]["enabled"] is False
