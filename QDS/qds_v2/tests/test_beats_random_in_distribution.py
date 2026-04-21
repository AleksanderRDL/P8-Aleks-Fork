"""Tests ML method beats random baseline on in-distribution aggregate error. See src/evaluation/README.md for details."""

from __future__ import annotations

from src.experiments.experiment_config import build_experiment_config
from src.experiments.experiment_pipeline_helpers import run_experiment_pipeline


def test_beats_random_in_distribution(synthetic_dataset, tmp_path) -> None:
    """Assert query-aware ML beats random on matched workload. See src/evaluation/README.md for details."""
    trajectories, _ = synthetic_dataset
    cfg = build_experiment_config(n_queries=64, epochs=3)
    mix = {"range": 0.7, "knn": 0.3}

    out = run_experiment_pipeline(
        config=cfg,
        trajectories=trajectories,
        train_mix=mix,
        eval_mix=mix,
        results_dir=str(tmp_path),
    )

    ml = out.metrics_dump["matched"]["MLQDS"]["aggregate_error"]
    rand = out.metrics_dump["matched"]["Random"]["aggregate_error"]
    assert ml < rand
