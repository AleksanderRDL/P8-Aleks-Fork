"""AIS-QDS v2 end-to-end experiment entrypoint. See src/experiments/README.md for details."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from src.data.ais_loader import generate_synthetic_ais_data, load_ais_csv
from src.experiments.experiment_cli import build_parser
from src.experiments.experiment_config import build_experiment_config
from src.experiments.experiment_pipeline_helpers import resolve_workload_mixes, run_experiment_pipeline


def main() -> None:
    """Parse CLI args and run the AIS-QDS v2 experiment. See src/experiments/README.md for details."""
    parser = build_parser()
    args = parser.parse_args()

    train_arg = args.train_workload_mix or args.workload_mix_train
    eval_arg = args.eval_workload_mix or args.workload_mix_eval
    train_mix, eval_mix = resolve_workload_mixes(train_arg, eval_arg, workload_keyword=args.workload)

    config = build_experiment_config(
        n_ships=args.n_ships,
        n_points=args.n_points,
        n_queries=args.n_queries,
        epochs=args.epochs,
        compression_ratio=args.compression_ratio,
        csv_path=args.csv_path,
        model_type=args.model_type,
        workload=args.workload,
        train_workload_mix=train_mix,
        eval_workload_mix=eval_mix,
        seed=args.seed,
        early_stopping_patience=args.early_stopping_patience,
    )

    print(f"[config] model={args.model_type}  workload={args.workload}  epochs={args.epochs}  "
          f"n_queries={args.n_queries}  compression_ratio={args.compression_ratio}", flush=True)

    t0 = time.perf_counter()
    if args.csv_path:
        print(f"[load-data] reading CSV: {args.csv_path}", flush=True)
        trajectories = load_ais_csv(args.csv_path)
    else:
        print(f"[load-data] generating synthetic data "
              f"(n_ships={config.data.n_ships}, n_points={config.data.n_points_per_ship})", flush=True)
        trajectories = generate_synthetic_ais_data(
            n_ships=config.data.n_ships,
            n_points_per_ship=config.data.n_points_per_ship,
            seed=config.data.seed,
        )
    print(f"[load-data] {len(trajectories)} trajectories loaded in {time.perf_counter() - t0:.2f}s", flush=True)

    out = run_experiment_pipeline(
        config=config,
        trajectories=trajectories,
        train_mix=train_mix,
        eval_mix=eval_mix,
        results_dir=args.results_dir,
        save_model=args.save_model,
        save_queries_dir=args.save_queries_dir,
        save_simplified_dir=args.save_simplified_dir,
    )

    print("\nMatched-workload table")
    print(out.matched_table)
    print("\nDistribution-shift table")
    print(out.shift_table)


if __name__ == "__main__":
    main()
