"""AIS-QDS v2 end-to-end experiment entrypoint. See src/experiments/README.md for details."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from src.data.ais_loader import generate_synthetic_ais_data, load_ais_csv
from src.experiments.experiment_cli import build_parser
from src.experiments.experiment_config import build_experiment_config
from src.experiments.experiment_pipeline_helpers import resolve_workload_mixes, run_experiment_pipeline


def _cap_loaded_trajectories(
    trajectories,
    mmsis: list[int] | None,
    max_trajectories: int | None,
):
    """Cap loaded trajectories for smoke runs while keeping MMSIs aligned."""
    if max_trajectories is None:
        return trajectories, mmsis
    cap = int(max_trajectories)
    if cap <= 0:
        raise ValueError("--max_trajectories must be positive when provided.")
    if len(trajectories) <= cap:
        return trajectories, mmsis
    capped = trajectories[:cap]
    capped_mmsis = mmsis[:cap] if mmsis is not None else None
    print(f"[load-data] capped trajectories to {cap}", flush=True)
    return capped, capped_mmsis


def _project_root() -> Path:
    """Find the repository root so default AISDATA outputs land in the shared folder."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "AISDATA").is_dir():
            return parent
    return Path.cwd()


def _default_simplified_dir(args) -> str:
    """Build a default run directory in AISDATA/ML_processed_AIS_files for simplified CSV output."""
    eval_stem = Path(args.eval_csv_path).stem if args.eval_csv_path else "eval"
    cov = args.query_coverage
    cov_tag = f"cov{float(cov):g}" if cov is not None else f"q{int(args.n_queries)}"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_workload = str(args.workload).replace("/", "_").replace(" ", "_")
    run_name = f"run_{eval_stem}_{cov_tag}_{safe_workload}_seed{int(args.seed)}_{timestamp}"
    return str(_project_root() / "AISDATA" / "ML_processed_AIS_files" / run_name)


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
        max_points_per_ship=args.max_points_per_ship,
        max_trajectories=args.max_trajectories,
        n_queries=args.n_queries,
        query_coverage=args.query_coverage,
        max_queries=args.max_queries,
        range_spatial_fraction=args.range_spatial_fraction,
        range_time_fraction=args.range_time_fraction,
        knn_k=args.knn_k,
        epochs=args.epochs,
        lr=args.lr,
        pointwise_loss_weight=args.pointwise_loss_weight,
        gradient_clip_norm=args.gradient_clip_norm,
        compression_ratio=args.compression_ratio,
        csv_path=args.csv_path,
        train_csv_path=args.train_csv_path,
        eval_csv_path=args.eval_csv_path,
        model_type=args.model_type,
        workload=args.workload,
        train_workload_mix=train_mix,
        eval_workload_mix=eval_mix,
        seed=args.seed,
        early_stopping_patience=args.early_stopping_patience,
        diagnostic_every=args.diagnostic_every,
        diagnostic_window_fraction=args.diagnostic_window_fraction,
        checkpoint_selection_metric=args.checkpoint_selection_metric,
        f1_diagnostic_every=args.f1_diagnostic_every,
        checkpoint_uniform_gap_weight=args.checkpoint_uniform_gap_weight,
        checkpoint_type_penalty_weight=args.checkpoint_type_penalty_weight,
        mlqds_temporal_fraction=args.mlqds_temporal_fraction,
        mlqds_diversity_bonus=args.mlqds_diversity_bonus,
        residual_label_mode=args.residual_label_mode,
    )

    coverage_msg = (
        f"  query_coverage={args.query_coverage}  max_queries={args.max_queries}"
        if args.query_coverage is not None else ""
    )
    print(
        f"[config] model={args.model_type}  workload={args.workload}  epochs={args.epochs}  "
        f"n_queries={args.n_queries}{coverage_msg}  compression_ratio={args.compression_ratio}  "
        f"lr={args.lr}  pointwise_loss_weight={args.pointwise_loss_weight}  "
        f"gradient_clip_norm={args.gradient_clip_norm}  "
        f"diagnostic_every={args.diagnostic_every}  "
        f"checkpoint_selection_metric={args.checkpoint_selection_metric}  "
        f"f1_diagnostic_every={args.f1_diagnostic_every}  "
        f"uniform_gap_weight={args.checkpoint_uniform_gap_weight}  "
        f"type_penalty_weight={args.checkpoint_type_penalty_weight}  "
        f"range_spatial_fraction={args.range_spatial_fraction}  range_time_fraction={args.range_time_fraction}  "
        f"knn_k={args.knn_k}  mlqds_temporal_fraction={args.mlqds_temporal_fraction}  "
        f"residual_label_mode={args.residual_label_mode}",
        flush=True,
    )

    t0 = time.perf_counter()
    mmsis: list[int] | None = None
    eval_trajectories = None
    eval_mmsis: list[int] | None = None
    if args.train_csv_path or args.eval_csv_path:
        if not args.train_csv_path or not args.eval_csv_path:
            parser.error("--train_csv_path/--train_csv and --eval_csv_path/--eval_csv must be supplied together.")
        print(f"[load-data] reading train CSV: {args.train_csv_path}", flush=True)
        trajectories, mmsis = load_ais_csv(
            args.train_csv_path,
            max_points_per_ship=args.max_points_per_ship,
            return_mmsis=True,
        )
        trajectories, mmsis = _cap_loaded_trajectories(trajectories, mmsis, args.max_trajectories)
        print(f"[load-data] reading eval CSV: {args.eval_csv_path}", flush=True)
        eval_trajectories, eval_mmsis = load_ais_csv(
            args.eval_csv_path,
            max_points_per_ship=args.max_points_per_ship,
            return_mmsis=True,
        )
        eval_trajectories, eval_mmsis = _cap_loaded_trajectories(
            eval_trajectories,
            eval_mmsis,
            args.max_trajectories,
        )
    elif args.csv_path:
        print(f"[load-data] reading CSV: {args.csv_path}", flush=True)
        trajectories, mmsis = load_ais_csv(
            args.csv_path,
            max_points_per_ship=args.max_points_per_ship,
            return_mmsis=True,
        )
        trajectories, mmsis = _cap_loaded_trajectories(trajectories, mmsis, args.max_trajectories)
    else:
        print(f"[load-data] generating synthetic data "
              f"(n_ships={config.data.n_ships}, n_points={config.data.n_points_per_ship})", flush=True)
        trajectories = generate_synthetic_ais_data(
            n_ships=config.data.n_ships,
            n_points_per_ship=config.data.n_points_per_ship,
            seed=config.data.seed,
        )
    if eval_trajectories is None:
        print(f"[load-data] {len(trajectories)} trajectories loaded in {time.perf_counter() - t0:.2f}s", flush=True)
    else:
        print(
            f"[load-data] train={len(trajectories)} eval={len(eval_trajectories)} trajectories "
            f"loaded in {time.perf_counter() - t0:.2f}s",
            flush=True,
        )

    save_simplified_dir = args.save_simplified_dir
    if args.eval_csv_path and save_simplified_dir is None:
        save_simplified_dir = _default_simplified_dir(args)
        print(f"[config] auto-saving simplified eval CSV under {save_simplified_dir}", flush=True)

    out = run_experiment_pipeline(
        config=config,
        trajectories=trajectories,
        train_mix=train_mix,
        eval_mix=eval_mix,
        results_dir=args.results_dir,
        save_model=args.save_model,
        save_queries_dir=args.save_queries_dir,
        save_simplified_dir=save_simplified_dir,
        trajectory_mmsis=mmsis,
        eval_trajectories=eval_trajectories,
        eval_trajectory_mmsis=eval_mmsis,
    )

    print("\nMatched-workload table")
    print(out.matched_table)
    print("\nGeometric-distortion table (lower is better; SED = time-synchronous, PED = perpendicular, in km)")
    print(out.geometric_table)
    print("\nDistribution-shift table")
    print(out.shift_table)


if __name__ == "__main__":
    main()
