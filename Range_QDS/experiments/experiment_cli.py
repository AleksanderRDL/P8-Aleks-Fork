"""CLI parsing helpers for the AIS-QDS experiment runner. See experiments/README.md for details."""

from __future__ import annotations

import argparse

from experiments.experiment_config import (
    DEFAULT_BUDGET_LOSS_RATIOS,
    DEFAULT_BUDGET_LOSS_TEMPERATURE,
    VALIDATION_SPLIT_MODES,
)
from experiments.torch_runtime import AMP_MODE_CHOICES, FLOAT32_MATMUL_PRECISION_CHOICES
from queries.query_generator import RANGE_ANCHOR_MODES, RANGE_TIME_DOMAIN_MODES
from simplification.mlqds_scoring import MLQDS_SCORE_MODES
from training.importance_labels import RANGE_LABEL_MODES
from training.model_features import SUPPORTED_MODEL_TYPES
from training.teacher_distillation import RANGE_TEACHER_DISTILLATION_MODES
from training.training_targets import RANGE_TARGET_BALANCE_MODES, RANGE_TRAINING_TARGET_MODES


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


def _range_anchor_mode_list(value: str) -> list[str]:
    """Parse comma-separated train anchor-prior modes."""
    modes: list[str] = []
    for raw in value.split(","):
        mode = raw.strip().lower()
        if not mode:
            continue
        if mode not in RANGE_ANCHOR_MODES:
            raise argparse.ArgumentTypeError(
                f"range train anchor modes must be one of {RANGE_ANCHOR_MODES}."
            )
        modes.append(mode)
    if not modes:
        raise argparse.ArgumentTypeError("provide at least one range anchor mode.")
    return modes


def _range_train_footprint_list(value: str) -> list[str]:
    """Parse comma-separated train footprint families as spatial_km:time_hours."""
    footprints: list[str] = []
    for raw in value.split(","):
        item = raw.strip().lower().replace("x", ":")
        if not item:
            continue
        parts = [part.strip() for part in item.split(":")]
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                "range train footprints must use spatial_km:time_hours entries, "
                "for example 1.1:2.5,2.2:5.0."
            )
        try:
            spatial_km = float(parts[0])
            time_hours = float(parts[1])
        except ValueError as exc:
            raise argparse.ArgumentTypeError("range train footprint values must be numeric.") from exc
        if spatial_km <= 0.0 or time_hours <= 0.0:
            raise argparse.ArgumentTypeError("range train footprint values must be positive.")
        footprints.append(f"{spatial_km:g}:{time_hours:g}")
    if not footprints:
        raise argparse.ArgumentTypeError("provide at least one range train footprint.")
    return footprints


def build_parser() -> argparse.ArgumentParser:
    """Build experiment CLI parser. See experiments/README.md for details."""
    parser = argparse.ArgumentParser(description="Run AIS-QDS experiment.")
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument(
        "--train_csv_path",
        "--train_csv",
        dest="train_csv_path",
        type=str,
        default=None,
        help=(
            "Dedicated train CSV path. A comma-separated list trains on multiple historical "
            "CSV days while keeping validation/eval sources separate."
        ),
    )
    parser.add_argument(
        "--validation_csv_path",
        "--validation_csv",
        "--val_csv_path",
        "--val_csv",
        dest="validation_csv_path",
        type=str,
        default=None,
        help=(
            "Optional dedicated checkpoint-validation CSV path or comma-separated CSV list. "
            "Requires --train_csv_path and --eval_csv_path."
        ),
    )
    parser.add_argument(
        "--eval_csv_path",
        "--eval_csv",
        dest="eval_csv_path",
        type=str,
        default=None,
        help="Dedicated final-eval CSV path or comma-separated CSV list.",
    )
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
    parser.add_argument(
        "--validation_split_mode",
        type=str,
        default="random",
        choices=VALIDATION_SPLIT_MODES,
        help=(
            "Fallback checkpoint-validation split when separate train/eval CSVs are used without "
            "--validation_csv_path. 'random' samples from combined train trajectories; "
            "'source_stratified' holds out validation trajectories from each train CSV source."
        ),
    )
    parser.add_argument(
        "--final_metrics_mode",
        type=str,
        default="diagnostic",
        choices=["diagnostic", "core"],
        help=(
            "Final evaluation scope. 'diagnostic' keeps Oracle and learned-fill diagnostic baselines; "
            "'core' reports exact MLQDS/uniform/DouglasPeucker metrics only."
        ),
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
        "--train_max_segments",
        type=int,
        default=None,
        help="Optional train CSV segment cap. Defaults to --max_segments when unset.",
    )
    parser.add_argument(
        "--validation_max_segments",
        type=int,
        default=None,
        help="Optional validation CSV segment cap. Defaults to --max_segments when unset.",
    )
    parser.add_argument(
        "--eval_max_segments",
        type=int,
        default=None,
        help="Optional eval CSV segment cap. Defaults to --max_segments when unset.",
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
        "--range_time_domain_mode",
        type=str,
        default="dataset",
        choices=RANGE_TIME_DOMAIN_MODES,
        help=(
            "Temporal clamp domain for generated range queries. 'dataset' uses global time bounds; "
            "'anchor_day' clamps each query to the 24-hour source/calendar day containing its anchor."
        ),
    )
    parser.add_argument(
        "--range_anchor_mode",
        type=str,
        default="mixed_density",
        choices=RANGE_ANCHOR_MODES,
        help=(
            "Anchor sampling prior for generated range queries. 'mixed_density' keeps the historical "
            "70 percent density-biased / 30 percent uniform mix; 'dense', 'uniform', and 'sparse' expose held-out "
            "generator settings."
        ),
    )
    parser.add_argument(
        "--range_train_anchor_modes",
        type=_range_anchor_mode_list,
        default=[],
        help=(
            "Optional comma-separated anchor priors cycled across train workload replicates. "
            "Leave unset to use --range_anchor_mode for training. Eval and checkpoint selection "
            "continue using --range_anchor_mode."
        ),
    )
    parser.add_argument(
        "--range_train_footprints",
        type=_range_train_footprint_list,
        default=[],
        help=(
            "Optional comma-separated train-only range footprint families as spatial_km:time_hours, "
            "cycled across train workload replicates. Eval and checkpoint selection continue using "
            "--range_spatial_km/--range_time_hours."
        ),
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
        "--range_max_coverage_overshoot",
        type=float,
        default=None,
        help=(
            "Optional target-coverage guard: reject candidate range boxes that would push union "
            "point coverage above --query_coverage plus this absolute tolerance. Accepts fractions or percents."
        ),
    )
    parser.add_argument(
        "--range_train_workload_replicates",
        type=int,
        default=1,
        help=(
            "Number of independent train-workload seeds to aggregate into blind range supervision. "
            "Eval and checkpoint-selection workloads remain separate."
        ),
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--embed_dim", type=int, default=64, help="Transformer hidden dimension.")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of transformer attention heads.")
    parser.add_argument(
        "--num_layers",
        type=int,
        default=3,
        help="Number of transformer encoder layers. For workload-blind models, 0 uses an MLP-only scorer.",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability used by the model.")
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
        choices=["ranking_bce", "budget_topk", "stratified_budget_topk", "pointwise_bce"],
        help=(
            "Training objective. 'ranking_bce' is the pairwise ranking plus BCE ablation; "
            "'budget_topk' optimizes soft retained-budget target mass across budget ratios; "
            "'stratified_budget_topk' is a slower diagnostic that optimizes the "
            "stratified selector's per-stratum choices; "
            "'pointwise_bce' directly fits every valid soft label."
        ),
    )
    parser.add_argument(
        "--budget_loss_ratios",
        type=_compression_ratio_list,
        default=list(DEFAULT_BUDGET_LOSS_RATIOS),
        help="Comma-separated retained-point ratios used by budget-aware loss objectives.",
    )
    parser.add_argument(
        "--budget_loss_temperature",
        type=float,
        default=DEFAULT_BUDGET_LOSS_TEMPERATURE,
        help="Soft top-k temperature for --loss_objective budget_topk.",
    )
    parser.add_argument(
        "--temporal_distribution_loss_weight",
        type=float,
        default=0.0,
        help=(
            "Optional weight for a budget-aware temporal CDF regularizer. "
            "0.0 disables it; small values discourage clustered soft top-k selections."
        ),
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
        help="Number of trajectory windows per MLQDS inference or validation-score diagnostic batch.",
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
    parser.add_argument(
        "--model_type",
        type=str,
        default="baseline",
        choices=SUPPORTED_MODEL_TYPES,
    )
    parser.add_argument(
        "--historical_prior_k",
        type=int,
        default=32,
        help="Nearest-neighbor count for model_type=historical_prior.",
    )
    parser.add_argument(
        "--historical_prior_clock_weight",
        type=float,
        default=0.0,
        help="Distance weight for historical-prior circular clock-time features. 0.0 preserves density-only behavior.",
    )
    parser.add_argument(
        "--historical_prior_mmsi_weight",
        type=float,
        default=1.0,
        help=(
            "Distance weight for model_type=historical_prior_mmsi deterministic vessel-id hash features. "
            "0.0 ignores identity; larger values prefer same-MMSI historical support."
        ),
    )
    parser.add_argument(
        "--historical_prior_density_weight",
        type=float,
        default=1.0,
        help="Distance weight for historical-prior spatial density/sparsity features.",
    )
    parser.add_argument(
        "--historical_prior_min_target",
        type=float,
        default=0.0,
        help=(
            "Minimum retained-frequency target stored by model_type=historical_prior. "
            "0.0 stores all train points; larger values keep only stronger useful-point support."
        ),
    )
    parser.add_argument(
        "--historical_prior_support_ratio",
        type=float,
        default=1.0,
        help=(
            "Per-train-trajectory top-target support cap for model_type=historical_prior. "
            "1.0 stores all points that pass min-target filtering; lower values reduce dense-route dominance."
        ),
    )
    parser.add_argument(
        "--historical_prior_source_aggregation",
        type=str,
        default="none",
        choices=["none", "mean", "min", "median"],
        help=(
            "How to combine historical-prior KNN scores across explicit train CSV sources. "
            "'none' preserves the original pooled prior; 'mean', 'min', and 'median' require signal "
            "to transfer across train days."
        ),
    )
    parser.add_argument(
        "--workload",
        type=str,
        default="range",
        choices=["range"],
        help="Query workload type for this model run. Only range is supported.",
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
        default="score",
        choices=["loss", "score", "uniform_gap"],
        help=(
            "Select restored checkpoints by held-out validation score, training loss, or validation score "
            "with fair-uniform gap penalties."
        ),
    )
    parser.add_argument(
        "--validation_score_every",
        type=int,
        default=0,
        help="Run held-out validation scoring every N epochs. 0 disables unless checkpoint selection metric is score/uniform_gap.",
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
        help="When checkpoint_selection_metric=uniform_gap, penalty weight for per-type validation-score deficits versus uniform.",
    )
    parser.add_argument(
        "--checkpoint_smoothing_window",
        type=int,
        default=1,
        help="Pick checkpoints by rolling-mean selection score over the last K diagnostic epochs. Reduces selection bias from noisy single-epoch validation scores. 1 = original single-epoch behavior; 5 = average over 5 latest diagnostic epochs.",
    )
    parser.add_argument(
        "--checkpoint_full_score_every",
        type=int,
        default=1,
        help="Run exact validation scoring every N eligible validation-score epochs. 1 keeps exact validation every eligible epoch.",
    )
    parser.add_argument(
        "--checkpoint_candidate_pool_size",
        type=int,
        default=1,
        help="When checkpoint_full_score_every > 1, keep this many cheap-diagnostic candidate snapshots for the next exact validation round.",
    )
    parser.add_argument(
        "--checkpoint_score_variant",
        type=str,
        default="range_usefulness",
        choices=["answer", "combined", "range_usefulness"],
        help=(
            "Which validation score to use for checkpoint selection. "
            "'range_usefulness' = range-local audit score for range workloads (default), "
            "'answer' = point/query F1, 'combined' = answer_f1 * point_subset_f1."
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
        help=(
            "Spacing bonus for MLQDS fill/swap/local_swap/local_delta_swap candidates away from temporal base points. "
            "Ignored by mlqds_hybrid_mode=stratified or global_budget."
        ),
    )
    parser.add_argument(
        "--mlqds_hybrid_mode",
        type=str,
        default="fill",
        choices=["fill", "swap", "local_swap", "local_delta_swap", "stratified", "global_fill", "global_budget"],
        help=(
            "How temporal scaffolding and learned scores are combined. "
            "'fill' reserves part of the budget for a temporal spine, then fills the rest. "
            "'swap' starts from full uniform temporal sampling and replaces only the unprotected budget share. "
            "'local_swap' pairs each learned addition with the nearest unprotected temporal-base removal. "
            "'local_delta_swap' performs that local replacement only when the learned score improves over the paired base point. "
            "'stratified' selects the highest learned-score point inside each temporal/index stratum. "
            "'global_fill' keeps a temporal base per trajectory, then spends residual budget globally by learned score. "
            "'global_budget' keeps endpoint skeletons, then spends remaining budget globally by learned score."
        ),
    )
    parser.add_argument(
        "--mlqds_stratified_center_weight",
        type=float,
        default=0.0,
        help=(
            "Optional center-distance penalty inside stratified learned-score bins. "
            "0.0 keeps pure learned top-score selection within each bin."
        ),
    )
    parser.add_argument(
        "--mlqds_min_learned_swaps",
        type=int,
        default=0,
        help=(
            "Diagnostic lower bound on learned replacements per trajectory for swap/local_swap/local_delta_swap modes. "
            "Default 0 preserves temporal-fraction rounding exactly."
        ),
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
        "--mlqds_range_geometry_blend",
        type=float,
        default=0.0,
        help=(
            "Blend model scores with cached range usefulness labels before MLQDS retention. "
            "0.0 uses model scores only; 1.0 uses range-geometry labels only."
        ),
    )
    parser.add_argument(
        "--temporal_residual_label_mode",
        type=str,
        default="none",
        choices=["none", "temporal"],
        help=(
            "Use labels directly, or explicitly train only on points not already kept by the temporal base. "
            "Default 'none' avoids accidental residual-only training in direct manual runs."
        ),
    )
    parser.add_argument(
        "--range_label_mode",
        type=str,
        default="usefulness",
        choices=RANGE_LABEL_MODES,
        help=(
            "Range label construction mode. 'point_f1' is the old in-box point proxy; "
            "'usefulness' adds audit-proxy signal; 'usefulness_balanced' rescales component "
            "mass toward RangeUseful audit weights; 'usefulness_ship_balanced' reduces dense "
            "query-hit ship dominance in point, entry/exit, and crossing support labels."
        ),
    )
    parser.add_argument(
        "--range_training_target_mode",
        type=str,
        default="point_value",
        choices=RANGE_TRAINING_TARGET_MODES,
        help=(
            "Transform training range labels before fitting the model. 'point_value' uses the raw "
            "expected-usefulness values; 'retained_frequency' trains on oracle retained-set membership "
            "frequency across configured budgets; 'global_budget_retained_frequency' trains on "
            "training-only global-budget oracle membership; 'historical_prior_retained_frequency' "
            "distills that target through a leave-one-out query-free historical KNN teacher; "
            "'marginal_coverage_frequency' trains on set-aware "
            "neighborhood coverage targets; 'query_spine_frequency' trains on query-derived temporal "
            "support anchors; 'query_residual_frequency' trains on train-query residual fill anchors; "
            "'set_utility_frequency' trains on one-step train-query RangeUseful gain; "
            "'local_swap_utility_frequency' trains on one-step train-query local-swap RangeUseful gain; "
            "'local_swap_gain_cost_frequency' trains local-delta candidate value against removal cost; "
            "'structural_retained_frequency' blends train workload usefulness with query-free "
            "globality/uniqueness scores; "
            "'continuity_retained_frequency' trains from boundary, temporal, gap, turn, and shape components."
        ),
    )
    parser.add_argument(
        "--range_target_balance_mode",
        type=str,
        default="none",
        choices=RANGE_TARGET_BALANCE_MODES,
        help=(
            "Optional training-only range target mass rebalance. 'trajectory_unit_mass' rescales "
            "each train trajectory's positive range-target mass to one before fitting a blind prior."
        ),
    )
    parser.add_argument(
        "--range_replicate_target_aggregation",
        type=str,
        default="label_mean",
        choices=["label_mean", "label_max", "frequency_mean"],
        help=(
            "How multiple train range workloads become retained-frequency targets. "
            "'label_mean' averages raw usefulness labels before target selection; "
            "'label_max' takes the max raw usefulness label before target selection; "
            "'frequency_mean' averages per-workload retained-frequency targets."
        ),
    )
    parser.add_argument(
        "--range_component_target_blend",
        type=float,
        default=1.0,
        help=(
            "When range_training_target_mode is component_retained_frequency or continuity_retained_frequency, "
            "blend component-wise retained targets with the ordinary retained-frequency target. 1.0 uses components only."
        ),
    )
    parser.add_argument(
        "--range_temporal_target_blend",
        type=float,
        default=0.0,
        help=(
            "Blend a query-blind uniform temporal retained-frequency target into retained-frequency "
            "range supervision. This changes training labels only; inference temporal_fraction is separate."
        ),
    )
    parser.add_argument(
        "--range_structural_target_blend",
        type=float,
        default=0.25,
        help=(
            "Training-only blend weight for structural_retained_frequency. "
            "0.0 uses train workload usefulness only; 1.0 uses query-free structural scores only."
        ),
    )
    parser.add_argument(
        "--range_structural_target_source_mode",
        type=str,
        default="blend",
        choices=["blend", "boost"],
        help=(
            "Source-score mode for structural_retained_frequency. 'blend' adds structural score support; "
            "'boost' only re-ranks train-useful points by structural prominence."
        ),
    )
    parser.add_argument(
        "--range_target_budget_weight_power",
        type=float,
        default=0.0,
        help=(
            "Training-only weighting for retained-frequency target budgets. "
            "0.0 averages configured budgets uniformly; positive values weight smaller compression ratios "
            "as ratio ** -power before normalization."
        ),
    )
    parser.add_argument(
        "--range_marginal_target_radius_scale",
        type=float,
        default=0.50,
        help=(
            "Neighborhood radius, as a fraction of target point spacing, for "
            "range_training_target_mode=marginal_coverage_frequency."
        ),
    )
    parser.add_argument(
        "--range_query_spine_fraction",
        type=float,
        default=0.10,
        help=(
            "Fraction of each in-query trajectory slice used as temporal support anchors for "
            "range_training_target_mode=query_spine_frequency."
        ),
    )
    parser.add_argument(
        "--range_query_spine_mass_mode",
        type=str,
        default="hit_group",
        choices=["hit_group", "query"],
        help=(
            "Mass normalization for query_spine_frequency labels. 'hit_group' preserves the old behavior: "
            "each train query/trajectory-hit group gets unit mass before averaging queries. 'query' gives each "
            "train query unit mass split across its hit trajectories."
        ),
    )
    parser.add_argument(
        "--range_query_residual_multiplier",
        type=float,
        default=1.0,
        help=(
            "Multiplier applied to budget_ratio * in-query point count when building "
            "range_training_target_mode=query_residual_frequency labels."
        ),
    )
    parser.add_argument(
        "--range_query_residual_mass_mode",
        type=str,
        default="query",
        choices=["query", "point"],
        help=(
            "Mass normalization for query_residual_frequency labels. 'query' gives each train query "
            "unit mass; 'point' gives each selected residual anchor unit mass before averaging queries."
        ),
    )
    parser.add_argument(
        "--range_set_utility_multiplier",
        type=float,
        default=1.0,
        help=(
            "Multiplier applied to budget_ratio * train-query hit count when building "
            "range_training_target_mode=set_utility_frequency, local_swap_utility_frequency, "
            "or local_swap_gain_cost_frequency labels."
        ),
    )
    parser.add_argument(
        "--range_set_utility_candidate_limit",
        type=int,
        default=128,
        help=(
            "Maximum candidates per train query and budget for one-step marginal RangeUseful scoring. "
            "Used by set_utility_frequency, local_swap_utility_frequency, and "
            "local_swap_gain_cost_frequency. Use 0 to score every candidate."
        ),
    )
    parser.add_argument(
        "--range_set_utility_mass_mode",
        type=str,
        default="gain",
        choices=["gain", "point", "query"],
        help=(
            "Target mass for set_utility_frequency/local_swap_utility_frequency/"
            "local_swap_gain_cost_frequency labels: raw marginal gain/value, selected-point frequency, "
            "or query-equal selected-point frequency."
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
        "--range_teacher_distillation_mode",
        type=str,
        default="none",
        choices=RANGE_TEACHER_DISTILLATION_MODES,
        help=(
            "Train a query-aware range teacher and convert its train-workload signal into query-blind "
            "student labels. 'rank_percentile' uses teacher rank labels; 'retained_frequency' uses "
            "teacher retained-set membership across budget ratios."
        ),
    )
    parser.add_argument(
        "--range_teacher_epochs",
        type=int,
        default=4,
        help="Epochs for the query-aware range teacher when teacher distillation is enabled.",
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
