"""Run a saved AIS-QDS model on a new day's preprocessed CSV (no training).

Loads a .pt checkpoint produced by run_ais_experiment.py (save_checkpoint) and
evaluates MLQDS against baselines on the supplied CSV. No gradient updates are
performed, so this is safe for a local CPU/GPU machine.

Example, from `QDS`:

    ../.venv/bin/python -m src.experiments.run_inference \
        --checkpoint artifacts/benchmarks/range_testing_baseline/runs/<run_id>/range_testing_baseline/benchmark_model.pt \
        --csv_path ../AISDATA/cleaned/<cleaned-ais-file.csv> \
        --n_queries 512 \
        --results_dir artifacts/benchmarks/inference_range_testing_baseline

Run this from ``QDS`` so the ``src`` package resolves.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from src.data.ais_loader import load_ais_csv
from src.data.trajectory_dataset import TrajectoryDataset
from src.data.trajectory_cache import load_or_build_ais_cache
from src.evaluation.baselines import (
    DouglasPeuckerMethod,
    MLQDSMethod,
    UniformTemporalMethod,
)
from src.evaluation.evaluate_methods import evaluate_method
from src.evaluation.query_cache import EvaluationQueryCache
from src.evaluation.tables import print_geometric_distortion_table, print_method_comparison_table, print_range_usefulness_table
from src.experiments.geojson_writers import report_trajectory_length_loss, write_queries_geojson, write_simplified_csv
from src.queries.query_generator import generate_typed_query_workload
from src.training.train_model import TrainingOutputs
from src.training.checkpoints import load_checkpoint
from src.experiments.torch_runtime import (
    AMP_MODE_CHOICES,
    FLOAT32_MATMUL_PRECISION_CHOICES,
    amp_runtime_snapshot,
    apply_torch_runtime_settings,
    normalize_amp_mode,
    torch_runtime_snapshot,
)


def _normalized_gap_arg(value: float | None) -> float | None:
    """Normalize CLI gap controls so <=0 disables time-gap segmentation."""
    if value is None:
        return None
    value = float(value)
    return None if value <= 0.0 else value


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a saved AIS-QDS model on a new CSV.")
    p.add_argument("--checkpoint", required=True, help="Path to saved .pt checkpoint.")
    p.add_argument("--csv_path", required=True, help="Preprocessed AIS CSV to evaluate on.")
    p.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional directory for segmented AIS Parquet caches keyed by source file and load config.",
    )
    p.add_argument(
        "--refresh_cache",
        action="store_true",
        help="Rebuild AIS cache entries even when a matching manifest exists.",
    )
    p.add_argument("--n_queries", type=int, default=100, help="Queries to generate for evaluation.")
    p.add_argument(
        "--query_coverage",
        type=float,
        default=None,
        help="Bias generated queries toward this point-coverage target while keeping --n_queries fixed. Accepts 0.30 or 30 for 30%%.",
    )
    p.add_argument(
        "--max_queries",
        type=int,
        default=None,
        help="Optional cap for coverage-targeted query generation when it may expand beyond --n_queries.",
    )
    p.add_argument(
        "--range_spatial_fraction",
        type=float,
        default=0.08,
        help="Range query half-width as a fraction of dataset lat/lon span. Ignored when --range_spatial_km is set.",
    )
    p.add_argument(
        "--range_time_fraction",
        type=float,
        default=0.15,
        help="Range query half-window as a fraction of dataset time span. Ignored when --range_time_hours is set.",
    )
    p.add_argument(
        "--range_spatial_km",
        type=float,
        default=None,
        help="Nominal range query spatial half-width in kilometers.",
    )
    p.add_argument(
        "--range_time_hours",
        type=float,
        default=None,
        help="Nominal range query temporal half-window in hours.",
    )
    p.add_argument(
        "--range_footprint_jitter",
        type=float,
        default=0.5,
        help="Random +/- fraction applied to range query spatial and temporal half-windows. 0.0 makes footprints fixed.",
    )
    p.add_argument("--knn_k", type=int, default=12, help="Number of nearest trajectories in generated kNN queries.")
    p.add_argument(
        "--workload",
        type=str,
        default=None,
        choices=["range", "knn", "similarity", "clustering"],
        help="Pure eval workload type. Default: use pure workload saved in checkpoint; else range.",
    )
    p.add_argument(
        "--compression_ratio",
        type=float,
        default=None,
        help="Override compression ratio. Default: use value saved in checkpoint.",
    )
    p.add_argument("--seed", type=int, default=42, help="Seed for query generation.")
    p.add_argument("--results_dir", type=str, default="results/inference", help="Where to write tables / metrics.")
    p.add_argument(
        "--save_queries_dir",
        type=str,
        default=None,
        help="If set, write eval-workload queries as GeoJSON to this directory.",
    )
    p.add_argument(
        "--save_simplified_dir",
        type=str,
        default=None,
        help="If set, write MLQDS-simplified trajectory CSV here.",
    )
    p.add_argument(
        "--min_points_per_segment",
        type=int,
        default=4,
        help="Minimum points required to keep an AIS trajectory segment.",
    )
    p.add_argument(
        "--max_points_per_segment",
        type=int,
        default=None,
        help="Optional AIS CSV downsampling cap per trajectory segment.",
    )
    p.add_argument(
        "--max_time_gap_seconds",
        type=float,
        default=3600.0,
        help="Split one vessel track into new trajectory segments when consecutive points exceed this time gap. Set <=0 to disable.",
    )
    p.add_argument(
        "--max_segments",
        type=int,
        default=None,
        help="Optional cap applied during CSV segmentation.",
    )
    p.add_argument(
        "--max_trajectories",
        type=int,
        default=None,
        help="Optional cap on trajectories loaded (useful for quick smoke tests on a laptop).",
    )
    p.add_argument(
        "--inference_device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for MLQDS model inference. 'auto' uses CUDA when available.",
    )
    p.add_argument(
        "--inference_batch_size",
        type=int,
        default=None,
        help="Number of trajectory windows per MLQDS inference batch. Default: use checkpoint config; else 16.",
    )
    p.add_argument(
        "--float32_matmul_precision",
        type=str,
        default=None,
        choices=FLOAT32_MATMUL_PRECISION_CHOICES,
        help="Torch float32 matmul precision. Default: use checkpoint config; else highest.",
    )
    p.add_argument(
        "--allow_tf32",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow TF32 for CUDA float32 matmul. Default: use checkpoint config; else off.",
    )
    p.add_argument(
        "--amp_mode",
        choices=AMP_MODE_CHOICES,
        default=None,
        help="Optional CUDA autocast mode. Default: use checkpoint config; else off.",
    )
    return p


def _resolve_eval_workload(workload: str | None, checkpoint_workload: str | None) -> str:
    if workload:
        return workload
    if checkpoint_workload:
        return checkpoint_workload
    return "range"


def main() -> None:
    args = _build_parser().parse_args()

    print(f"[load-checkpoint] {args.checkpoint}", flush=True)
    artifacts = load_checkpoint(args.checkpoint)
    saved_cfg = artifacts.config
    saved_workload = artifacts.workload_type or saved_cfg.query.workload
    precision = args.float32_matmul_precision or str(getattr(saved_cfg.model, "float32_matmul_precision", "highest"))
    allow_tf32 = (
        bool(args.allow_tf32)
        if args.allow_tf32 is not None
        else bool(getattr(saved_cfg.model, "allow_tf32", False))
    )
    amp_mode = normalize_amp_mode(args.amp_mode or str(getattr(saved_cfg.model, "amp_mode", "off")))
    inference_batch_size = max(
        1,
        int(
            args.inference_batch_size
            if args.inference_batch_size is not None
            else getattr(saved_cfg.model, "inference_batch_size", 16)
        ),
    )
    runtime_settings = apply_torch_runtime_settings(
        float32_matmul_precision=precision,
        allow_tf32=allow_tf32,
    )
    print(
        f"  model_type={saved_cfg.model.model_type}  "
        f"epochs_trained={artifacts.epochs_trained}  "
        f"workload={saved_workload}  "
        f"float32_matmul_precision={runtime_settings['float32_matmul_precision']}  "
        f"allow_tf32={runtime_settings['tf32_matmul_allowed']}  "
        f"amp_mode={amp_mode}  "
        f"inference_batch_size={inference_batch_size}",
        flush=True,
    )

    eval_workload_type = _resolve_eval_workload(args.workload, saved_workload)
    eval_workload_map = {eval_workload_type: 1.0}
    compression_ratio = (
        float(args.compression_ratio) if args.compression_ratio is not None else float(saved_cfg.model.compression_ratio)
    )
    print(f"[eval-config] workload={eval_workload_type}  compression_ratio={compression_ratio}", flush=True)

    t0 = time.perf_counter()
    print(f"[load-data] reading CSV: {args.csv_path}", flush=True)
    load_kwargs = {
        "min_points_per_segment": args.min_points_per_segment,
        "max_points_per_segment": args.max_points_per_segment,
        "max_time_gap_seconds": _normalized_gap_arg(args.max_time_gap_seconds),
        "max_segments": args.max_segments,
    }
    cache_payload = None
    if args.cache_dir:
        cached = load_or_build_ais_cache(
            args.csv_path,
            cache_dir=args.cache_dir,
            refresh_cache=bool(args.refresh_cache),
            **load_kwargs,
        )
        trajectories = cached.trajectories
        trajectory_mmsis = cached.mmsis
        data_audit = cached.audit
        cache_payload = cached.cache_metadata()
        state = "hit" if cached.cache_hit else "built"
        print(f"[load-data] cache {state}: {cached.cache_dir}", flush=True)
    else:
        trajectories, trajectory_mmsis, data_audit = load_ais_csv(
            args.csv_path,
            **load_kwargs,
            return_mmsis=True,
            return_audit=True,
        )
    print(
        f"[load-data] audit: rows={data_audit.rows_loaded} "
        f"dropped_invalid={data_audit.rows_dropped_invalid} "
        f"duplicates={data_audit.duplicate_timestamp_rows} "
        f"segments={data_audit.output_segment_count} "
        f"gap_splits={data_audit.time_gap_over_threshold_count}",
        flush=True,
    )
    if args.max_trajectories is not None and len(trajectories) > args.max_trajectories:
        trajectories = trajectories[: args.max_trajectories]
        trajectory_mmsis = trajectory_mmsis[: args.max_trajectories]
        print(f"[load-data] capped trajectories to {args.max_trajectories}", flush=True)
    print(f"[load-data] {len(trajectories)} trajectories in {time.perf_counter() - t0:.2f}s", flush=True)

    dataset = TrajectoryDataset(trajectories)
    points = dataset.get_all_points()
    boundaries = dataset.get_trajectory_boundaries()
    print(f"[dataset] points={points.shape[0]}  trajectories={len(boundaries)}", flush=True)

    t0 = time.perf_counter()
    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=int(args.n_queries),
        workload_map=eval_workload_map,
        seed=int(args.seed),
        target_coverage=args.query_coverage,
        max_queries=args.max_queries,
        range_spatial_fraction=args.range_spatial_fraction,
        range_time_fraction=args.range_time_fraction,
        range_spatial_km=args.range_spatial_km,
        range_time_hours=args.range_time_hours,
        range_footprint_jitter=args.range_footprint_jitter,
        knn_k=args.knn_k,
    )
    coverage_msg = ""
    if workload.coverage_fraction is not None:
        coverage_msg = (
            f"  coverage={100.0 * workload.coverage_fraction:.2f}% "
            f"({workload.covered_points}/{workload.total_points})"
        )
    print(
        f"[workload] generated {len(workload.typed_queries)} queries in "
        f"{time.perf_counter() - t0:.2f}s{coverage_msg}",
        flush=True,
    )

    if args.save_queries_dir:
        write_queries_geojson(args.save_queries_dir, workload.typed_queries)
        print(f"[workload] wrote queries GeoJSON to {args.save_queries_dir}", flush=True)

    # Adapt ModelArtifacts -> TrainingOutputs (MLQDSMethod only reads .model and .scaler).
    trained = TrainingOutputs(
        model=artifacts.model,
        scaler=artifacts.scaler,
        labels=torch.zeros((1, 4), dtype=torch.float32),
        labelled_mask=torch.zeros((1, 4), dtype=torch.bool),
        history=[],
        epochs_trained=int(artifacts.epochs_trained),
    )

    methods = [
        MLQDSMethod(
            name="MLQDS",
            trained=trained,
            workload=workload,
            workload_type=eval_workload_type,
            score_mode=str(getattr(saved_cfg.model, "mlqds_score_mode", "rank")),
            score_temperature=float(getattr(saved_cfg.model, "mlqds_score_temperature", 1.0)),
            rank_confidence_weight=float(getattr(saved_cfg.model, "mlqds_rank_confidence_weight", 0.15)),
            temporal_fraction=float(getattr(saved_cfg.model, "mlqds_temporal_fraction", 0.50)),
            diversity_bonus=float(getattr(saved_cfg.model, "mlqds_diversity_bonus", 0.0)),
            inference_device=None if args.inference_device == "auto" else args.inference_device,
            inference_batch_size=inference_batch_size,
            amp_mode=amp_mode,
        ),
        UniformTemporalMethod(),
        DouglasPeuckerMethod(),
    ]

    results: dict[str, Any] = {}
    save_masks = bool(args.save_simplified_dir)
    query_cache = EvaluationQueryCache.for_workload(points, boundaries, workload.typed_queries)
    for method in methods:
        t0 = time.perf_counter()
        print(f"[eval] {method.name} ...", flush=True)
        results[method.name] = evaluate_method(
            method=method,
            points=points,
            boundaries=boundaries,
            typed_queries=workload.typed_queries,
            workload_map=eval_workload_map,
            compression_ratio=compression_ratio,
            return_mask=method.name == "MLQDS" or save_masks,
            query_cache=query_cache,
        )
        print(f"[eval] {method.name} done in {time.perf_counter() - t0:.2f}s", flush=True)

    mlqds_mask = results["MLQDS"].retained_mask
    if mlqds_mask is None:
        raise RuntimeError("MLQDS retained mask was not captured during inference evaluation.")

    table = print_method_comparison_table(results)
    geometric_table = print_geometric_distortion_table(results)
    range_usefulness_table = print_range_usefulness_table(results)
    print("\nMatched-workload table (inference on new CSV)")
    print(table)
    print("\nGeometric-distortion table (lower is better; SED = time-synchronous, PED = perpendicular, in km)")
    print(geometric_table)
    print("\nRange-usefulness audit table")
    print(range_usefulness_table)

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "matched_table.txt").write_text(table + "\n", encoding="utf-8")
    (out_dir / "geometric_distortion_table.txt").write_text(geometric_table + "\n", encoding="utf-8")
    (out_dir / "range_usefulness_table.txt").write_text(range_usefulness_table + "\n", encoding="utf-8")

    dump = {
        "checkpoint": str(args.checkpoint),
        "csv_path": str(args.csv_path),
        "n_trajectories": len(trajectories),
        "n_points": int(points.shape[0]),
        "workload": eval_workload_type,
        "compression_ratio": compression_ratio,
        "inference_batch_size": inference_batch_size,
        "query_coverage": workload.coverage_fraction,
        "covered_points": workload.covered_points,
        "total_points": workload.total_points,
        "data_audit": {
            **data_audit.to_dict(),
            **({"cache": cache_payload} if cache_payload is not None else {}),
        },
        "torch_runtime": {
            **torch_runtime_snapshot(),
            "amp": amp_runtime_snapshot(amp_mode),
        },
        "matched": {
            name: {
                "aggregate_f1": m.aggregate_f1,
                "per_type_f1": m.per_type_f1,
                "compression_ratio": m.compression_ratio,
                "latency_ms": m.latency_ms,
                "avg_retained_point_gap": m.avg_retained_point_gap,
                "avg_retained_point_gap_norm": m.avg_retained_point_gap_norm,
                "max_retained_point_gap": m.max_retained_point_gap,
                "geometric_distortion": m.geometric_distortion,
                "avg_length_preserved": m.avg_length_preserved,
                "combined_query_shape_score": m.combined_query_shape_score,
                "range_point_f1": m.range_point_f1,
                "pure_range_f1": m.range_point_f1,
                "range_ship_f1": m.range_ship_f1,
                "range_ship_coverage": m.range_ship_coverage,
                "range_entry_exit_f1": m.range_entry_exit_f1,
                "range_crossing_f1": m.range_crossing_f1,
                "range_temporal_coverage": m.range_temporal_coverage,
                "range_gap_coverage": m.range_gap_coverage,
                "range_turn_coverage": m.range_turn_coverage,
                "range_shape_score": m.range_shape_score,
                "range_usefulness_score": m.range_usefulness_score,
                "range_usefulness_schema_version": m.range_usefulness_schema_version,
                "range_audit": m.range_audit,
            }
            for name, m in results.items()
        },
    }
    with open(out_dir / "inference_run.json", "w", encoding="utf-8") as f:
        json.dump(dump, f, indent=2)
    print(f"[write] results -> {out_dir}", flush=True)

    t0 = time.perf_counter()
    print("[trajectory-length-loss] starting...", flush=True)
    try:
        report_trajectory_length_loss(points, boundaries, mlqds_mask, top_k=25, trajectory_mmsis=trajectory_mmsis)
    finally:
        print(f"[trajectory-length-loss] done in {time.perf_counter() - t0:.2f}s", flush=True)

    if args.save_simplified_dir:
        out_dir_simp = Path(args.save_simplified_dir)
        out_dir_simp.mkdir(parents=True, exist_ok=True)
        write_simplified_csv(str(out_dir_simp / "ML_simplified.csv"), points, boundaries, mlqds_mask, trajectory_mmsis=trajectory_mmsis)
        print(f"[write] simplified CSV -> {out_dir_simp / 'ML_simplified.csv'}", flush=True)
        for ref_name, csv_name in (("uniform", "uniform_simplified.csv"),
                                   ("DouglasPeucker", "DP_simplified.csv")):
            ref_eval = results.get(ref_name)
            if ref_eval is not None and ref_eval.retained_mask is not None:
                write_simplified_csv(
                    str(out_dir_simp / csv_name),
                    points,
                    boundaries,
                    ref_eval.retained_mask,
                    trajectory_mmsis=trajectory_mmsis,
                )
                print(f"[write] simplified CSV -> {out_dir_simp / csv_name}", flush=True)


if __name__ == "__main__":
    main()
