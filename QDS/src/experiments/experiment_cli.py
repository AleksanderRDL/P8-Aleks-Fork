"""CLI parsing helpers for the AIS-QDS experiment runner. See src/experiments/README.md for details."""

from __future__ import annotations

import argparse

from src.experiments.torch_runtime import AMP_MODE_CHOICES, FLOAT32_MATMUL_PRECISION_CHOICES
from src.simplification.mlqds_scoring import MLQDS_SCORE_MODES
from src.training.importance_labels import RANGE_LABEL_MODES


def _compression_ratio_list(value: str) -> list[float]:
    """Parse comma-separated compression ratios for optional range audits."""
    ratios: list[float] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        ratio = float(item)
        if ratio <= 0.0 or ratio > 1.0:
            raise argparse.ArgumentTypeError("compression ratios must be in (0, 1].")
        ratios.append(ratio)
    if not ratios:
        raise argparse.ArgumentTypeError("provide at least one compression ratio.")
    return ratios


def build_parser() -> argparse.ArgumentParser:
    """Build experiment CLI parser. See src/experiments/README.md for details."""
    parser = argparse.ArgumentParser(description="Run AIS-QDS experiment.")
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--train_csv_path", "--train_csv", dest="train_csv_path", type=str, default=None)
    parser.add_argument(
        "--validation_csv_path",
        "--validation_csv",
        "--val_csv_path",
        "--val_csv",
        dest="validation_csv_path",
        type=str,
        default=None,
        help="Optional dedicated checkpoint-validation CSV. Requires --train_csv_path and --eval_csv_path.",
    )
    parser.add_argument("--eval_csv_path", "--eval_csv", dest="eval_csv_path", type=str, default=None)
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional directory for segmented AIS Parquet caches keyed by source file and load config.",
    )
    parser.add_argument(
        "--refresh_cache",
        action="store_true",
        help="Rebuild AIS cache entries even when a matching manifest exists.",
    )
    parser.add_argument(
        "--range_diagnostics_mode",
        type=str,
        default="full",
        choices=["full", "cached"],
        help="Use full range diagnostics or reuse persistent range-diagnostics caches when --cache_dir is set.",
    )
    parser.add_argument("--n_ships", type=int, default=24)
    parser.add_argument("--n_points", type=int, default=200)
    parser.add_argument(
        "--min_points_per_segment",
        type=int,
        default=4,
        help="Minimum points required to keep an AIS trajectory segment.",
    )
    parser.add_argument(
        "--max_points_per_segment",
        type=int,
        default=None,
        help="Optional AIS CSV downsampling cap per trajectory segment, useful for smoke runs.",
    )
    parser.add_argument(
        "--max_time_gap_seconds",
        type=float,
        default=3600.0,
        help="Split one vessel track into new trajectory segments when consecutive points exceed this time gap. Set <=0 to disable.",
    )
    parser.add_argument(
        "--max_segments",
        type=int,
        default=None,
        help="Optional cap applied during CSV segmentation, useful for smoke runs.",
    )
    parser.add_argument(
        "--max_trajectories",
        type=int,
        default=None,
        help="Optional cap on loaded AIS trajectories after CSV loading, useful for smoke runs.",
    )
    parser.add_argument("--n_queries", type=int, default=128)
    parser.add_argument(
        "--query_coverage",
        type=float,
        default=None,
        help="Bias generated queries toward this point-coverage target while keeping --n_queries fixed. Accepts 0.30 or 30 for 30%%.",
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=None,
        help="Optional cap for coverage-targeted query generation when it may expand beyond --n_queries.",
    )
    parser.add_argument(
        "--range_spatial_fraction",
        type=float,
        default=0.08,
        help="Range query half-width as a fraction of dataset lat/lon span. Ignored when --range_spatial_km is set.",
    )
    parser.add_argument(
        "--range_time_fraction",
        type=float,
        default=0.15,
        help="Range query half-window as a fraction of dataset time span. Ignored when --range_time_hours is set.",
    )
    parser.add_argument(
        "--range_spatial_km",
        type=float,
        default=None,
        help="Nominal range query spatial half-width in kilometers. Keeps workload scale stable across datasets.",
    )
    parser.add_argument(
        "--range_time_hours",
        type=float,
        default=None,
        help="Nominal range query temporal half-window in hours. Keeps workload scale stable across datasets.",
    )
    parser.add_argument(
        "--range_footprint_jitter",
        type=float,
        default=0.5,
        help="Random +/- fraction applied to range query spatial and temporal half-windows. 0.0 makes footprints fixed.",
    )
    parser.add_argument(
        "--range_min_point_hits",
        type=int,
        default=None,
        help="Optional range-query acceptance filter: reject boxes with fewer point hits.",
    )
    parser.add_argument(
        "--range_max_point_hit_fraction",
        type=float,
        default=None,
        help="Optional range-query acceptance filter: reject boxes hitting more than this point fraction.",
    )
    parser.add_argument(
        "--range_min_trajectory_hits",
        type=int,
        default=None,
        help="Optional range-query acceptance filter: reject boxes hitting fewer trajectories.",
    )
    parser.add_argument(
        "--range_max_trajectory_hit_fraction",
        type=float,
        default=None,
        help="Optional range-query acceptance filter: reject boxes hitting more than this trajectory fraction.",
    )
    parser.add_argument(
        "--range_max_box_volume_fraction",
        type=float,
        default=None,
        help="Optional range-query acceptance filter: reject boxes with larger normalized spatiotemporal volume.",
    )
    parser.add_argument(
        "--range_duplicate_iou_threshold",
        type=float,
        default=None,
        help="Optional range-query acceptance filter: reject boxes with IoU at or above this threshold versus accepted boxes.",
    )
    parser.add_argument(
        "--range_acceptance_max_attempts",
        type=int,
        default=None,
        help="Maximum candidate range boxes to try when acceptance filters are enabled.",
    )
    parser.add_argument(
        "--knn_k",
        type=int,
        default=12,
        help="Number of nearest trajectories returned by generated kNN queries.",
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--ranking_pairs_per_type",
        type=int,
        default=96,
        help="Number of positive/negative ranking pairs sampled per query type and training window.",
    )
    parser.add_argument(
        "--ranking_top_quantile",
        type=float,
        default=0.80,
        help="Label quantile used to define top-ranked positive candidates for ranking-pair sampling.",
    )
    parser.add_argument(
        "--pointwise_loss_weight",
        type=float,
        default=0.25,
        help="Weight for balanced pointwise BCE supervision alongside the active set/ranking loss.",
    )
    parser.add_argument(
        "--loss_objective",
        type=str,
        default="budget_topk",
        choices=["ranking_bce", "budget_topk"],
        help=(
            "Training objective. 'ranking_bce' is the legacy pairwise ranking plus BCE loss; "
            "'budget_topk' optimizes soft retained-budget target mass across budget ratios."
        ),
    )
    parser.add_argument(
        "--budget_loss_ratios",
        type=_compression_ratio_list,
        default=[0.01, 0.02, 0.05, 0.10],
        help="Comma-separated retained-point ratios used by --loss_objective budget_topk.",
    )
    parser.add_argument(
        "--budget_loss_temperature",
        type=float,
        default=0.10,
        help="Soft top-k temperature for --loss_objective budget_topk.",
    )
    parser.add_argument(
        "--gradient_clip_norm",
        type=float,
        default=1.0,
        help="Max gradient norm. Set <=0 to disable clipping.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Number of trajectory windows per training optimizer step.",
    )
    parser.add_argument(
        "--inference_batch_size",
        type=int,
        default=16,
        help="Number of trajectory windows per MLQDS inference or validation-F1 diagnostic batch.",
    )
    parser.add_argument(
        "--query_chunk_size",
        type=int,
        default=2048,
        help=(
            "Number of workload queries attended per cross-attention chunk. "
            "Set at least --n_queries to use one exact attention softmax for the full workload."
        ),
    )
    parser.add_argument("--compression_ratio", type=float, default=0.2)
    parser.add_argument("--model_type", type=str, default="baseline", choices=["baseline", "turn_aware"])
    parser.add_argument(
        "--workload",
        type=str,
        default="range",
        choices=["range", "knn", "similarity", "clustering"],
        help="Pure query workload type for this model run.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help=(
            "Stop training if the active checkpoint selection score does not improve for this many "
            "eligible diagnostic epochs. 0 disables."
        ),
    )
    parser.add_argument(
        "--diagnostic_every",
        type=int,
        default=1,
        help="Run training diagnostics every N epochs. Use 1 so every epoch can be selected as best.",
    )
    parser.add_argument(
        "--diagnostic_window_fraction",
        type=float,
        default=0.2,
        help="Fraction of trajectory windows used for each diagnostic pass.",
    )
    parser.add_argument(
        "--checkpoint_selection_metric",
        type=str,
        default="f1",
        choices=["loss", "f1", "uniform_gap"],
        help="Select restored checkpoints by held-out query F1, training loss, or F1 with fair-uniform gap penalties.",
    )
    parser.add_argument(
        "--f1_diagnostic_every",
        type=int,
        default=0,
        help="Run held-out query-F1 diagnostics every N epochs. 0 disables unless checkpoint selection metric is f1 or uniform_gap.",
    )
    parser.add_argument(
        "--checkpoint_uniform_gap_weight",
        type=float,
        default=0.5,
        help="When checkpoint_selection_metric=uniform_gap, bonus/penalty weight for aggregate gap versus uniform.",
    )
    parser.add_argument(
        "--checkpoint_type_penalty_weight",
        type=float,
        default=1.0,
        help="When checkpoint_selection_metric=uniform_gap, penalty weight for per-type F1 deficits versus uniform.",
    )
    parser.add_argument(
        "--checkpoint_smoothing_window",
        type=int,
        default=1,
        help="Pick checkpoints by rolling-mean selection score over the last K diagnostic epochs. Reduces selection bias from noisy single-epoch F1. 1 = original single-epoch behavior; 5 = average over 5 latest diagnostic epochs.",
    )
    parser.add_argument(
        "--checkpoint_full_f1_every",
        type=int,
        default=1,
        help="Run exact validation F1/usefulness every N F1-diagnostic epochs. 1 keeps exact validation every eligible epoch.",
    )
    parser.add_argument(
        "--checkpoint_candidate_pool_size",
        type=int,
        default=1,
        help="When checkpoint_full_f1_every > 1, keep this many cheap-diagnostic candidate snapshots for the next exact validation round.",
    )
    parser.add_argument(
        "--checkpoint_f1_variant",
        type=str,
        default="range_usefulness",
        choices=["answer", "combined", "range_usefulness"],
        help=(
            "Which validation score to use for checkpoint selection. "
            "'range_usefulness' = range-local audit score for range workloads (default), "
            "'answer' = legacy point/query F1, 'combined' = legacy answer_f1 * point_subset_f1."
        ),
    )
    parser.add_argument(
        "--mlqds_temporal_fraction",
        type=float,
        default=0.0,
        help="Fraction of the retained budget reserved for evenly spaced temporal base points before MLQDS score fill. Default 0.0 = pure learned scoring; raise to add a uniform spine.",
    )
    parser.add_argument(
        "--mlqds_diversity_bonus",
        type=float,
        default=0.0,
        help="Spacing bonus for MLQDS fill candidates away from temporal base points. Default 0.0 keeps learned score fill isolated.",
    )
    parser.add_argument(
        "--mlqds_score_mode",
        type=str,
        default="rank",
        choices=MLQDS_SCORE_MODES,
        help="Convert pure workload logits to simplification scores using per-trajectory ranks, sigmoid logits, or raw logits.",
    )
    parser.add_argument(
        "--mlqds_score_temperature",
        type=float,
        default=1.0,
        help="Temperature for temperature_sigmoid, zscore_sigmoid, and rank_confidence score modes.",
    )
    parser.add_argument(
        "--mlqds_rank_confidence_weight",
        type=float,
        default=0.15,
        help="Blend weight for rank_confidence score mode. 0.0=pure rank, 1.0=pure zscore sigmoid.",
    )
    parser.add_argument(
        "--residual_label_mode",
        type=str,
        default="temporal",
        choices=["none", "temporal"],
        help="Use labels directly, or train only on points not already kept by the temporal base.",
    )
    parser.add_argument(
        "--range_label_mode",
        type=str,
        default="usefulness",
        choices=RANGE_LABEL_MODES,
        help=(
            "Range label construction mode. 'point_f1' is the old in-box point proxy; "
            "'usefulness' adds ship, entry/exit, temporal-span, and local-shape label signal."
        ),
    )
    parser.add_argument(
        "--range_boundary_prior_weight",
        type=float,
        default=0.0,
        help=(
            "Optional range-label boundary prior. 0.0 keeps pure point-F1 labels; "
            "1.0 gives in-box boundary-crossing points 2x raw weight before normalization."
        ),
    )
    parser.add_argument(
        "--range_audit_compression_ratios",
        type=_compression_ratio_list,
        default=None,
        help=(
            "Optional comma-separated retained-point ratios for a multi-budget range usefulness audit, "
            "for example 0.01,0.02,0.05,0.10. Disabled by default because it reruns method evaluation."
        ),
    )
    parser.add_argument(
        "--float32_matmul_precision",
        type=str,
        default="highest",
        choices=FLOAT32_MATMUL_PRECISION_CHOICES,
        help="Torch float32 matmul precision. Use 'high' with --allow_tf32 for TF32 benchmarking.",
    )
    parser.add_argument(
        "--allow_tf32",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow TF32 for CUDA float32 matmul. Defaults off for baseline comparability.",
    )
    parser.add_argument(
        "--amp_mode",
        choices=AMP_MODE_CHOICES,
        default="off",
        help="Optional CUDA autocast mode for model forward passes. Losses and diagnostics stay in FP32.",
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
    return parser
