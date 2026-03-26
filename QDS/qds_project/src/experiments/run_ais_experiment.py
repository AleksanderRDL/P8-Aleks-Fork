"""End-to-end AIS QDS experiment pipeline. See src/experiments/README.md for full details."""

from __future__ import annotations

import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from src.experiments.experiment_cli import parse_and_validate_experiment_args
from src.experiments.experiment_config import (
    ExperimentConfig,
    MethodMetrics,
    build_experiment_config,
)
from src.experiments.experiment_pipeline_helpers import (
    _print_workload_comparison_table,
    _workload_title,
)
from src.experiments.workload_runner import run_single_workload


def run_ais_experiment(
    n_ships: int = 10,
    n_points: int = 100,
    n_queries: int = 100,
    epochs: int = 50,
    threshold: float = 0.5,
    target_ratio: float | None = None,
    compression_ratio: float | None = 0.2,
    min_points_per_trajectory: int = 5,
    max_train_points: int | None = None,
    model_max_points: int | None = 300_000,
    point_batch_size: int = 50_000,
    importance_chunk_size: int = 200_000,
    dp_max_points: int = 200_000,
    skip_baselines: bool = False,
    skip_visualizations: bool = False,
    max_visualization_points: int = 200_000,
    max_visualization_ships: int = 200,
    max_points_per_ship_plot: int = 2_000,
    csv_path: str | None = None,
    save_csv: bool = False,
    workload: str = "density",
    density_ratio: float = 0.7,
    query_spatial_fraction: float = 0.03,
    query_temporal_fraction: float = 0.10,
    query_spatial_lower_quantile: float = 0.01,
    query_spatial_upper_quantile: float = 0.99,
    model_type: str = "baseline",
    turn_bias_weight: float = 0.1,
    turn_score_method: str = "heading",
) -> None:
    """Run the full AIS QDS experiment and print a results table."""
    config = build_experiment_config(**locals())

    if config.query.workload == "all":
        # Run each workload independently and collect results for a combined table
        all_results: dict[str, dict[str, MethodMetrics]] = {}
        for wl in ("uniform", "density", "mixed"):
            print(f"\n{'='*65}")
            print(f"Running workload: {_workload_title(wl)}")
            print(f"{'='*65}")
            wl_config = replace(config, query=replace(config.query, workload=wl))
            wl_results = run_single_workload(wl_config)
            all_results[wl] = wl_results

        # Print combined comparison table
        _print_workload_comparison_table(all_results)
        return

    run_single_workload(config)


def _run_single_workload(config: ExperimentConfig) -> dict[str, MethodMetrics]:
    """Backward-compatible wrapper for the extracted workload runner."""
    return run_single_workload(config)


def main() -> None:
    """Command-line entry point for the AIS QDS experiment."""
    run_ais_experiment(**parse_and_validate_experiment_args())


if __name__ == "__main__":
    main()
