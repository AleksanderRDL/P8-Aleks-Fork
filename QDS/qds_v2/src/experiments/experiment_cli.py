"""CLI parsing helpers for the AIS-QDS v2 experiment runner. See src/experiments/README.md for details."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build experiment CLI parser. See src/experiments/README.md for details."""
    parser = argparse.ArgumentParser(description="Run AIS-QDS v2 experiment.")
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--n_ships", type=int, default=24)
    parser.add_argument("--n_points", type=int, default=200)
    parser.add_argument("--n_queries", type=int, default=128)
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
    return parser
