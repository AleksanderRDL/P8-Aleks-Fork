"""AIS-QDS v2 end-to-end experiment entrypoint. See src/experiments/README.md for details."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from src.data.ais_loader import generate_synthetic_ais_data, load_ais_csv
from src.data.trajectory_cache import load_or_build_ais_cache
from src.experiments.experiment_cli import build_parser
from src.experiments.experiment_config import build_experiment_config
from src.experiments.experiment_pipeline_helpers import resolve_workload_mixes, run_experiment_pipeline
from src.experiments.torch_runtime import apply_torch_runtime_settings


def _normalized_gap_arg(value: float | None) -> float | None:
    """Normalize CLI gap controls so <=0 disables time-gap segmentation."""
    if value is None:
        return None
    value = float(value)
    return None if value <= 0.0 else value


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


def _log_load_audit(label: str, audit) -> None:
    """Print a compact AIS load audit for repeatable run logs."""
    length = audit.segment_length_stats
    gaps = audit.time_gap_stats
    print(
        f"[load-data] {label} audit: rows={audit.rows_loaded} "
        f"dropped_invalid={audit.rows_dropped_invalid} "
        f"duplicates={audit.duplicate_timestamp_rows} "
        f"mmsis={audit.input_mmsi_count} "
        f"segments={audit.output_segment_count} "
        f"points={audit.output_point_count} "
        f"short_segments_dropped={audit.dropped_short_segments} "
        f"gap_splits={audit.time_gap_over_threshold_count} "
        f"segment_len_p50={length.get('p50', 0.0):.1f} "
        f"segment_len_p95={length.get('p95', 0.0):.1f} "
        f"max_gap_s={gaps.get('max', 0.0):.1f}",
        flush=True,
    )


def _load_csv_trajectories(label: str, csv_path: str, args, load_kwargs: dict) -> tuple:
    """Load one CSV either through the Parquet cache or directly from source."""
    if args.cache_dir:
        cache = load_or_build_ais_cache(
            csv_path,
            cache_dir=args.cache_dir,
            refresh_cache=bool(args.refresh_cache),
            **load_kwargs,
        )
        state = "hit" if cache.cache_hit else "built"
        print(f"[load-data] cache {state}: {cache.cache_dir}", flush=True)
        _log_load_audit(label, cache.audit)
        audit_payload = cache.audit.to_dict()
        audit_payload["cache"] = cache.cache_metadata()
        return cache.trajectories, cache.mmsis, cache.audit, audit_payload

    trajectories, mmsis, audit = load_ais_csv(
        csv_path,
        **load_kwargs,
        return_mmsis=True,
        return_audit=True,
    )
    _log_load_audit(label, audit)
    return trajectories, mmsis, audit, audit.to_dict()


def _default_simplified_dir(args) -> str:
    """Build a run-local default directory for simplified CSV output."""
    return str(Path(args.results_dir) / "simplified_eval")


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
        min_points_per_segment=args.min_points_per_segment,
        max_points_per_segment=args.max_points_per_segment,
        max_time_gap_seconds=_normalized_gap_arg(args.max_time_gap_seconds),
        max_segments=args.max_segments,
        max_trajectories=args.max_trajectories,
        cache_dir=args.cache_dir,
        refresh_cache=args.refresh_cache,
        n_queries=args.n_queries,
        query_coverage=args.query_coverage,
        max_queries=args.max_queries,
        range_spatial_fraction=args.range_spatial_fraction,
        range_time_fraction=args.range_time_fraction,
        range_min_point_hits=args.range_min_point_hits,
        range_max_point_hit_fraction=args.range_max_point_hit_fraction,
        range_min_trajectory_hits=args.range_min_trajectory_hits,
        range_max_trajectory_hit_fraction=args.range_max_trajectory_hit_fraction,
        range_max_box_volume_fraction=args.range_max_box_volume_fraction,
        range_duplicate_iou_threshold=args.range_duplicate_iou_threshold,
        range_acceptance_max_attempts=args.range_acceptance_max_attempts,
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
        train_batch_size=args.train_batch_size,
        inference_batch_size=args.inference_batch_size,
        diagnostic_every=args.diagnostic_every,
        diagnostic_window_fraction=args.diagnostic_window_fraction,
        checkpoint_selection_metric=args.checkpoint_selection_metric,
        f1_diagnostic_every=args.f1_diagnostic_every,
        checkpoint_uniform_gap_weight=args.checkpoint_uniform_gap_weight,
        checkpoint_type_penalty_weight=args.checkpoint_type_penalty_weight,
        checkpoint_smoothing_window=args.checkpoint_smoothing_window,
        checkpoint_f1_variant=args.checkpoint_f1_variant,
        mlqds_temporal_fraction=args.mlqds_temporal_fraction,
        mlqds_diversity_bonus=args.mlqds_diversity_bonus,
        residual_label_mode=args.residual_label_mode,
        float32_matmul_precision=args.float32_matmul_precision,
        allow_tf32=args.allow_tf32,
        amp_mode=args.amp_mode,
    )
    runtime_settings = apply_torch_runtime_settings(
        float32_matmul_precision=config.model.float32_matmul_precision,
        allow_tf32=config.model.allow_tf32,
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
        f"train_batch_size={args.train_batch_size}  "
        f"inference_batch_size={args.inference_batch_size}  "
        f"diagnostic_every={args.diagnostic_every}  "
        f"checkpoint_selection_metric={args.checkpoint_selection_metric}  "
        f"f1_diagnostic_every={args.f1_diagnostic_every}  "
        f"uniform_gap_weight={args.checkpoint_uniform_gap_weight}  "
        f"type_penalty_weight={args.checkpoint_type_penalty_weight}  "
        f"smoothing_window={args.checkpoint_smoothing_window}  "
        f"f1_variant={args.checkpoint_f1_variant}  "
        f"range_spatial_fraction={args.range_spatial_fraction}  range_time_fraction={args.range_time_fraction}  "
        f"range_min_point_hits={args.range_min_point_hits}  "
        f"range_max_point_hit_fraction={args.range_max_point_hit_fraction}  "
        f"range_min_trajectory_hits={args.range_min_trajectory_hits}  "
        f"range_max_trajectory_hit_fraction={args.range_max_trajectory_hit_fraction}  "
        f"range_max_box_volume_fraction={args.range_max_box_volume_fraction}  "
        f"range_duplicate_iou_threshold={args.range_duplicate_iou_threshold}  "
        f"range_acceptance_max_attempts={args.range_acceptance_max_attempts}  "
        f"knn_k={args.knn_k}  mlqds_temporal_fraction={args.mlqds_temporal_fraction}  "
        f"residual_label_mode={args.residual_label_mode}  "
        f"min_points_per_segment={args.min_points_per_segment}  "
        f"max_points_per_segment={args.max_points_per_segment}  "
        f"max_time_gap_seconds={_normalized_gap_arg(args.max_time_gap_seconds)}  "
        f"max_segments={args.max_segments}  cache_dir={args.cache_dir}  "
        f"refresh_cache={args.refresh_cache}  "
        f"float32_matmul_precision={runtime_settings['float32_matmul_precision']}  "
        f"allow_tf32={runtime_settings['tf32_matmul_allowed']}  "
        f"amp_mode={config.model.amp_mode}",
        flush=True,
    )

    t0 = time.perf_counter()
    mmsis: list[int] | None = None
    eval_trajectories = None
    eval_mmsis: list[int] | None = None
    data_audit = None
    load_kwargs = {
        "min_points_per_segment": args.min_points_per_segment,
        "max_points_per_segment": args.max_points_per_segment,
        "max_time_gap_seconds": _normalized_gap_arg(args.max_time_gap_seconds),
        "max_segments": args.max_segments,
    }
    if args.train_csv_path or args.eval_csv_path:
        if not args.train_csv_path or not args.eval_csv_path:
            parser.error("--train_csv_path/--train_csv and --eval_csv_path/--eval_csv must be supplied together.")
        print(f"[load-data] reading train CSV: {args.train_csv_path}", flush=True)
        trajectories, mmsis, _train_audit, train_audit_payload = _load_csv_trajectories(
            "train",
            args.train_csv_path,
            args,
            load_kwargs,
        )
        trajectories, mmsis = _cap_loaded_trajectories(trajectories, mmsis, args.max_trajectories)
        print(f"[load-data] reading eval CSV: {args.eval_csv_path}", flush=True)
        eval_trajectories, eval_mmsis, _eval_audit, eval_audit_payload = _load_csv_trajectories(
            "eval",
            args.eval_csv_path,
            args,
            load_kwargs,
        )
        eval_trajectories, eval_mmsis = _cap_loaded_trajectories(
            eval_trajectories,
            eval_mmsis,
            args.max_trajectories,
        )
        data_audit = {"train": train_audit_payload, "eval": eval_audit_payload}
    elif args.csv_path:
        print(f"[load-data] reading CSV: {args.csv_path}", flush=True)
        trajectories, mmsis, _audit, audit_payload = _load_csv_trajectories(
            "csv",
            args.csv_path,
            args,
            load_kwargs,
        )
        trajectories, mmsis = _cap_loaded_trajectories(trajectories, mmsis, args.max_trajectories)
        data_audit = {"csv": audit_payload}
    else:
        if config.data.n_ships is None or config.data.n_points_per_ship is None:
            raise ValueError("Synthetic data generation requires n_ships and n_points_per_ship.")
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
        data_audit=data_audit,
    )

    print("\nMatched-workload table")
    print(out.matched_table)
    print("\nGeometric-distortion table (lower is better; SED = time-synchronous, PED = perpendicular, in km)")
    print(out.geometric_table)
    print("\nDistribution-shift table")
    print(out.shift_table)


if __name__ == "__main__":
    main()
