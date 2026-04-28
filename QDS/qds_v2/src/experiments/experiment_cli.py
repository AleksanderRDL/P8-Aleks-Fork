"""CLI parsing helpers for the AIS-QDS v2 experiment runner. See src/experiments/README.md for details."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build experiment CLI parser. See src/experiments/README.md for details."""
    parser = argparse.ArgumentParser(description="Run AIS-QDS v2 experiment.")
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--train_csv_path", "--train_csv", dest="train_csv_path", type=str, default=None)
    parser.add_argument("--eval_csv_path", "--eval_csv", dest="eval_csv_path", type=str, default=None)
    parser.add_argument("--n_ships", type=int, default=24)
    parser.add_argument("--n_points", type=int, default=200)
    parser.add_argument("--n_queries", type=int, default=128)
    parser.add_argument(
        "--query_coverage",
        "--target_query_coverage",
        dest="query_coverage",
        type=float,
        default=None,
        help="Generate queries until this point-coverage target is reached. Accepts 0.30 or 30 for 30%%.",
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=None,
        help="Safety cap for dynamic query generation when --query_coverage is set. Default: max(n_queries, 1000).",
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--compression_ratio", type=float, default=0.2)
    parser.add_argument("--model_type", type=str, default="baseline", choices=["baseline", "turn_aware"])
    parser.add_argument("--workload", type=str, default="mixed")

    parser.add_argument("--train_workload_mix", type=str, default=None)
    parser.add_argument("--eval_workload_mix", type=str, default=None)
    parser.add_argument("--workload_mix_train", type=str, default=None)
    parser.add_argument("--workload_mix_eval", type=str, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help="Stop training if avg Kendall tau does not improve for this many epochs. 0 disables.",
    )
    parser.add_argument(
        "--save_model",
        type=str,
        default=None,
        help="Path to save trained model checkpoint (.pt). Disabled if not provided.",
    )
    parser.add_argument(
        "--save_queries_dir",
        type=str,
        default=None,
        help="Directory to save eval-workload queries as one GeoJSON per query type.",
    )
    parser.add_argument(
        "--save_simplified_dir",
        type=str,
        default=None,
        help="Directory to save MLQDS simplified trajectories as CSV.",
    )
    parser.add_argument(
        "--query_area_boost",
        type=float,
        default=1.0,
        help="Multiply MLQDS scores of points within --query_buffer_deg of any query by this factor. 1.0 disables.",
    )
    parser.add_argument(
        "--query_buffer_deg",
        type=float,
        default=0.20,
        help="Buffer (in degrees lat/lon) around each query geometry used by --query_area_boost.",
    )
    return parser
