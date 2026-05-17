"""Tests for torch runtime precision controls."""

from __future__ import annotations

from argparse import Namespace

import pytest
import torch

from experiments.benchmark_runtime import (
    DEFAULT_PROFILE,
    _batch_size_sweep_summary,
    _extra_args_include_training_data_source,
    _parse_train_batch_sizes,
    _profile_train_args,
    _runtime_child_args,
)
from experiments.experiment_cli import build_parser
from experiments.experiment_config import (
    DEFAULT_BUDGET_LOSS_RATIOS,
    DEFAULT_BUDGET_LOSS_TEMPERATURE,
    ExperimentConfig,
    build_experiment_config,
)
from experiments.run_ais_experiment import _split_max_segments
from experiments.torch_runtime import (
    amp_runtime_snapshot,
    apply_torch_runtime_settings,
    normalize_amp_mode,
    torch_autocast_context,
)
from training.checkpoints import _checkpoint_config_payload
from training.model_features import SUPPORTED_MODEL_TYPES


def test_apply_torch_runtime_settings_sets_precision_and_tf32() -> None:
    old_precision = torch.get_float32_matmul_precision()
    old_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    try:
        snapshot = apply_torch_runtime_settings(float32_matmul_precision="high", allow_tf32=True)

        assert snapshot["float32_matmul_precision"] == "high"
        assert snapshot["tf32_matmul_allowed"] is True
        assert torch.get_float32_matmul_precision() == "high"
        assert bool(torch.backends.cuda.matmul.allow_tf32) is True
    finally:
        torch.set_float32_matmul_precision(old_precision)
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


def test_experiment_config_roundtrips_precision_controls() -> None:
    cfg = build_experiment_config(
        train_csv_path="train.csv",
        validation_csv_path="validation.csv",
        eval_csv_path="eval.csv",
        train_max_segments=48,
        validation_max_segments=32,
        eval_max_segments=24,
        float32_matmul_precision="high",
        allow_tf32=True,
        embed_dim=96,
        num_heads=8,
        num_layers=2,
        dropout=0.05,
        train_batch_size=64,
        inference_batch_size=32,
        query_chunk_size=512,
        amp_mode="bf16",
        model_type="historical_prior",
        historical_prior_k=7,
        historical_prior_clock_weight=0.25,
        historical_prior_mmsi_weight=2.5,
        historical_prior_density_weight=3.5,
        historical_prior_min_target=0.25,
        historical_prior_support_ratio=0.4,
        range_boundary_prior_weight=1.0,
        range_label_mode="point_f1",
        range_target_balance_mode="trajectory_unit_mass",
        range_replicate_target_aggregation="frequency_mean",
        range_component_target_blend=0.35,
        range_temporal_target_blend=0.20,
        range_target_budget_weight_power=0.75,
        range_marginal_target_radius_scale=0.75,
        range_query_spine_fraction=0.25,
        range_query_spine_mass_mode="query",
        range_query_residual_multiplier=1.50,
        range_query_residual_mass_mode="point",
        range_set_utility_multiplier=1.75,
        range_set_utility_candidate_limit=64,
        range_set_utility_mass_mode="point",
        loss_objective="ranking_bce",
        budget_loss_ratios=[0.02, 0.05],
        budget_loss_temperature=0.2,
        query_useful_aux_loss_weight=0.75,
        query_useful_segment_budget_head_weight=0.35,
        query_useful_segment_level_loss_weight=0.90,
        temporal_distribution_loss_weight=0.07,
        ranking_pairs_per_type=192,
        ranking_top_quantile=0.9,
        range_spatial_km=2.2,
        range_time_hours=6.0,
        range_max_coverage_overshoot=0.02,
        range_time_domain_mode="anchor_day",
        range_anchor_mode="sparse",
        range_train_anchor_modes=["mixed_density", "sparse"],
        range_train_footprints=["1.1:2.5", "2.2:5"],
        range_train_workload_replicates=3,
        mlqds_hybrid_mode="swap",
        mlqds_stratified_center_weight=0.45,
        mlqds_min_learned_swaps=1,
        mlqds_score_mode="rank_confidence",
        mlqds_score_temperature=0.5,
        mlqds_rank_confidence_weight=0.3,
        mlqds_range_geometry_blend=0.4,
        range_audit_compression_ratios=[0.01, 0.05],
        checkpoint_full_score_every=3,
        checkpoint_candidate_pool_size=2,
        range_diagnostics_mode="cached",
        validation_split_mode="source_stratified",
        train_fraction=0.34,
        val_fraction=0.33,
        final_metrics_mode="core",
    )
    restored = ExperimentConfig.from_dict(cfg.to_dict())

    assert restored.model.float32_matmul_precision == "high"
    assert restored.model.embed_dim == 96
    assert restored.model.num_heads == 8
    assert restored.model.num_layers == 2
    assert restored.model.dropout == 0.05
    assert restored.data.train_csv_path == "train.csv"
    assert restored.data.validation_csv_path == "validation.csv"
    assert restored.data.eval_csv_path == "eval.csv"
    assert restored.data.train_max_segments == 48
    assert restored.data.validation_max_segments == 32
    assert restored.data.eval_max_segments == 24
    assert restored.data.range_diagnostics_mode == "cached"
    assert restored.data.validation_split_mode == "source_stratified"
    assert restored.data.train_fraction == 0.34
    assert restored.data.val_fraction == 0.33
    assert restored.baselines.final_metrics_mode == "core"
    assert restored.model.allow_tf32 is True
    assert restored.model.train_batch_size == 64
    assert restored.model.inference_batch_size == 32
    assert restored.model.query_chunk_size == 512
    assert restored.model.amp_mode == "bf16"
    assert restored.model.model_type == "historical_prior"
    assert restored.model.historical_prior_k == 7
    assert restored.model.historical_prior_clock_weight == 0.25
    assert restored.model.historical_prior_mmsi_weight == 2.5
    assert restored.model.historical_prior_density_weight == 3.5
    assert restored.model.historical_prior_min_target == 0.25
    assert restored.model.historical_prior_support_ratio == 0.4
    assert restored.model.range_boundary_prior_weight == 1.0
    assert restored.model.range_label_mode == "point_f1"
    assert restored.model.range_target_balance_mode == "trajectory_unit_mass"
    assert restored.model.range_replicate_target_aggregation == "frequency_mean"
    assert restored.model.range_component_target_blend == 0.35
    assert restored.model.range_temporal_target_blend == 0.20
    assert restored.model.range_target_budget_weight_power == 0.75
    assert restored.model.range_marginal_target_radius_scale == 0.75
    assert restored.model.range_query_spine_fraction == 0.25
    assert restored.model.range_query_spine_mass_mode == "query"
    assert restored.model.range_query_residual_multiplier == 1.50
    assert restored.model.range_query_residual_mass_mode == "point"
    assert restored.model.range_set_utility_multiplier == 1.75
    assert restored.model.range_set_utility_candidate_limit == 64
    assert restored.model.range_set_utility_mass_mode == "point"
    assert restored.model.loss_objective == "ranking_bce"
    assert restored.model.budget_loss_ratios == [0.02, 0.05]
    assert restored.model.budget_loss_temperature == 0.2
    assert restored.model.query_useful_aux_loss_weight == 0.75
    assert restored.model.query_useful_segment_budget_head_weight == 0.35
    assert restored.model.query_useful_segment_level_loss_weight == 0.90
    assert restored.model.temporal_distribution_loss_weight == 0.07
    assert restored.model.ranking_pairs_per_type == 192
    assert restored.model.ranking_top_quantile == 0.9
    assert restored.query.range_spatial_km == 2.2
    assert restored.query.range_time_hours == 6.0
    assert restored.query.range_max_coverage_overshoot == 0.02
    assert restored.query.range_time_domain_mode == "anchor_day"
    assert restored.query.range_anchor_mode == "sparse"
    assert restored.query.range_train_anchor_modes == ["mixed_density", "sparse"]
    assert restored.query.range_train_footprints == ["1.1:2.5", "2.2:5"]
    assert restored.query.range_train_workload_replicates == 3
    assert restored.model.mlqds_hybrid_mode == "swap"
    assert restored.model.mlqds_stratified_center_weight == 0.45
    assert restored.model.mlqds_min_learned_swaps == 1
    assert restored.model.mlqds_score_mode == "rank_confidence"
    assert restored.model.mlqds_score_temperature == 0.5
    assert restored.model.mlqds_rank_confidence_weight == 0.3
    assert restored.model.mlqds_range_geometry_blend == 0.4
    assert restored.model.range_audit_compression_ratios == [0.01, 0.05]
    assert restored.model.checkpoint_selection_metric == "score"
    assert restored.model.checkpoint_full_score_every == 3
    assert restored.model.checkpoint_candidate_pool_size == 2


def test_cli_exposes_training_and_scoring_tuning_controls() -> None:
    args = build_parser().parse_args(
        [
            "--ranking_pairs_per_type",
            "64",
            "--ranking_top_quantile",
            "0.70",
            "--mlqds_score_mode",
            "rank_confidence",
            "--embed_dim",
            "96",
            "--num_heads",
            "8",
            "--num_layers",
            "2",
            "--dropout",
            "0.05",
            "--mlqds_hybrid_mode",
            "swap",
            "--mlqds_stratified_center_weight",
            "0.45",
            "--mlqds_min_learned_swaps",
            "1",
            "--mlqds_score_temperature",
            "0.50",
            "--mlqds_rank_confidence_weight",
            "0.30",
            "--mlqds_range_geometry_blend",
            "0.40",
            "--validation_split_mode",
            "source_stratified",
            "--train_fraction",
            "0.34",
            "--val_fraction",
            "0.33",
            "--range_audit_compression_ratios",
            "0.01,0.02,0.10",
            "--range_label_mode",
            "point_f1",
            "--range_target_balance_mode",
            "trajectory_unit_mass",
            "--range_train_workload_replicates",
            "4",
            "--range_time_domain_mode",
            "anchor_day",
            "--range_anchor_mode",
            "sparse",
            "--range_train_anchor_modes",
            "mixed_density,sparse",
            "--range_train_footprints",
            "1.1:2.5,2.2x5.0",
            "--range_max_coverage_overshoot",
            "0.02",
            "--range_replicate_target_aggregation",
            "frequency_mean",
            "--range_component_target_blend",
            "0.40",
            "--range_temporal_target_blend",
            "0.15",
            "--range_structural_target_blend",
            "0.35",
            "--range_structural_target_source_mode",
            "boost",
            "--range_target_budget_weight_power",
            "0.75",
            "--range_marginal_target_radius_scale",
            "0.65",
            "--range_query_spine_fraction",
            "0.20",
            "--range_query_spine_mass_mode",
            "query",
            "--range_query_residual_multiplier",
            "1.25",
            "--range_query_residual_mass_mode",
            "point",
            "--range_set_utility_multiplier",
            "1.75",
            "--range_set_utility_candidate_limit",
            "64",
            "--range_set_utility_mass_mode",
            "query",
            "--query_prior_grid_bins",
            "128",
            "--query_prior_smoothing_passes",
            "0",
            "--loss_objective",
            "budget_topk",
            "--budget_loss_ratios",
            "0.01,0.05",
            "--budget_loss_temperature",
            "0.20",
            "--query_useful_aux_loss_weight",
            "0.75",
            "--query_useful_segment_budget_head_weight",
            "0.35",
            "--query_useful_segment_level_loss_weight",
            "0.90",
            "--temporal_distribution_loss_weight",
            "0.20",
            "--checkpoint_full_score_every",
            "3",
            "--checkpoint_candidate_pool_size",
            "2",
            "--range_diagnostics_mode",
            "cached",
            "--final_metrics_mode",
            "core",
            "--model_type",
            "historical_prior",
            "--historical_prior_k",
            "7",
            "--historical_prior_clock_weight",
            "0.25",
            "--historical_prior_mmsi_weight",
            "2.5",
            "--historical_prior_density_weight",
            "3.5",
            "--historical_prior_min_target",
            "0.25",
            "--historical_prior_support_ratio",
            "0.40",
            "--historical_prior_source_aggregation",
            "mean",
            "--train_max_segments",
            "48",
            "--validation_max_segments",
            "32",
            "--eval_max_segments",
            "24",
            "--validation_csv_path",
            "validation.csv",
        ]
    )

    cfg = build_experiment_config(
        ranking_pairs_per_type=args.ranking_pairs_per_type,
        ranking_top_quantile=args.ranking_top_quantile,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        mlqds_hybrid_mode=args.mlqds_hybrid_mode,
        mlqds_stratified_center_weight=args.mlqds_stratified_center_weight,
        mlqds_min_learned_swaps=args.mlqds_min_learned_swaps,
        mlqds_score_mode=args.mlqds_score_mode,
        mlqds_score_temperature=args.mlqds_score_temperature,
        mlqds_rank_confidence_weight=args.mlqds_rank_confidence_weight,
        mlqds_range_geometry_blend=args.mlqds_range_geometry_blend,
        range_audit_compression_ratios=args.range_audit_compression_ratios,
        range_label_mode=args.range_label_mode,
        range_target_balance_mode=args.range_target_balance_mode,
        range_train_workload_replicates=args.range_train_workload_replicates,
        range_time_domain_mode=args.range_time_domain_mode,
        range_anchor_mode=args.range_anchor_mode,
        range_train_anchor_modes=args.range_train_anchor_modes,
        range_train_footprints=args.range_train_footprints,
        range_max_coverage_overshoot=args.range_max_coverage_overshoot,
        range_replicate_target_aggregation=args.range_replicate_target_aggregation,
        range_component_target_blend=args.range_component_target_blend,
        range_temporal_target_blend=args.range_temporal_target_blend,
        range_structural_target_blend=args.range_structural_target_blend,
        range_structural_target_source_mode=args.range_structural_target_source_mode,
        range_target_budget_weight_power=args.range_target_budget_weight_power,
        range_marginal_target_radius_scale=args.range_marginal_target_radius_scale,
        range_query_spine_fraction=args.range_query_spine_fraction,
        range_query_spine_mass_mode=args.range_query_spine_mass_mode,
        range_query_residual_multiplier=args.range_query_residual_multiplier,
        range_query_residual_mass_mode=args.range_query_residual_mass_mode,
        range_set_utility_multiplier=args.range_set_utility_multiplier,
        range_set_utility_candidate_limit=args.range_set_utility_candidate_limit,
        range_set_utility_mass_mode=args.range_set_utility_mass_mode,
        query_prior_grid_bins=args.query_prior_grid_bins,
        query_prior_smoothing_passes=args.query_prior_smoothing_passes,
        loss_objective=args.loss_objective,
        budget_loss_ratios=args.budget_loss_ratios,
        budget_loss_temperature=args.budget_loss_temperature,
        query_useful_aux_loss_weight=args.query_useful_aux_loss_weight,
        query_useful_segment_budget_head_weight=args.query_useful_segment_budget_head_weight,
        query_useful_segment_level_loss_weight=args.query_useful_segment_level_loss_weight,
        temporal_distribution_loss_weight=args.temporal_distribution_loss_weight,
        checkpoint_full_score_every=args.checkpoint_full_score_every,
        checkpoint_candidate_pool_size=args.checkpoint_candidate_pool_size,
        range_diagnostics_mode=args.range_diagnostics_mode,
        final_metrics_mode=args.final_metrics_mode,
        model_type=args.model_type,
        historical_prior_k=args.historical_prior_k,
        historical_prior_clock_weight=args.historical_prior_clock_weight,
        historical_prior_mmsi_weight=args.historical_prior_mmsi_weight,
        historical_prior_density_weight=args.historical_prior_density_weight,
        historical_prior_min_target=args.historical_prior_min_target,
        historical_prior_support_ratio=args.historical_prior_support_ratio,
        historical_prior_source_aggregation=args.historical_prior_source_aggregation,
        train_max_segments=args.train_max_segments,
        validation_max_segments=args.validation_max_segments,
        eval_max_segments=args.eval_max_segments,
        validation_csv_path=args.validation_csv_path,
    )

    assert args.ranking_pairs_per_type == 64
    assert args.ranking_top_quantile == 0.70
    assert args.embed_dim == 96
    assert args.num_heads == 8
    assert args.num_layers == 2
    assert args.dropout == 0.05
    assert args.mlqds_hybrid_mode == "swap"
    assert args.mlqds_stratified_center_weight == 0.45
    assert args.mlqds_min_learned_swaps == 1
    assert args.mlqds_score_mode == "rank_confidence"
    assert args.mlqds_score_temperature == 0.50
    assert args.mlqds_rank_confidence_weight == 0.30
    assert args.mlqds_range_geometry_blend == 0.40
    assert args.validation_split_mode == "source_stratified"
    assert args.train_fraction == 0.34
    assert args.val_fraction == 0.33
    assert args.range_audit_compression_ratios == [0.01, 0.02, 0.10]
    assert args.range_label_mode == "point_f1"
    assert args.range_target_balance_mode == "trajectory_unit_mass"
    assert args.range_train_workload_replicates == 4
    assert args.range_time_domain_mode == "anchor_day"
    assert args.range_anchor_mode == "sparse"
    assert args.range_train_anchor_modes == ["mixed_density", "sparse"]
    assert args.range_train_footprints == ["1.1:2.5", "2.2:5"]
    assert args.range_max_coverage_overshoot == 0.02
    assert args.range_replicate_target_aggregation == "frequency_mean"
    assert args.range_component_target_blend == 0.40
    assert args.range_temporal_target_blend == 0.15
    assert args.range_structural_target_blend == 0.35
    assert args.range_structural_target_source_mode == "boost"
    assert args.range_target_budget_weight_power == 0.75
    assert args.range_marginal_target_radius_scale == 0.65
    assert args.range_query_spine_fraction == 0.20
    assert args.range_query_spine_mass_mode == "query"
    assert args.range_query_residual_multiplier == 1.25
    assert args.range_query_residual_mass_mode == "point"
    assert args.range_set_utility_multiplier == 1.75
    assert args.range_set_utility_candidate_limit == 64
    assert args.range_set_utility_mass_mode == "query"
    assert args.query_prior_grid_bins == 128
    assert args.query_prior_smoothing_passes == 0
    assert args.loss_objective == "budget_topk"
    assert args.budget_loss_ratios == [0.01, 0.05]
    assert args.budget_loss_temperature == 0.20
    assert args.query_useful_aux_loss_weight == 0.75
    assert args.query_useful_segment_budget_head_weight == 0.35
    assert args.query_useful_segment_level_loss_weight == 0.90
    assert args.temporal_distribution_loss_weight == 0.20
    assert args.checkpoint_full_score_every == 3
    assert args.checkpoint_candidate_pool_size == 2
    assert args.range_diagnostics_mode == "cached"
    assert args.final_metrics_mode == "core"
    assert args.model_type == "historical_prior"
    assert args.historical_prior_k == 7
    assert args.historical_prior_clock_weight == 0.25
    assert args.historical_prior_mmsi_weight == 2.5
    assert args.historical_prior_density_weight == 3.5
    assert args.historical_prior_min_target == 0.25
    assert args.historical_prior_support_ratio == 0.40
    assert args.historical_prior_source_aggregation == "mean"
    assert args.train_max_segments == 48
    assert args.validation_max_segments == 32
    assert args.eval_max_segments == 24
    assert args.validation_csv_path == "validation.csv"
    assert cfg.model.ranking_pairs_per_type == 64
    assert cfg.model.ranking_top_quantile == 0.70
    assert cfg.model.embed_dim == 96
    assert cfg.model.num_heads == 8
    assert cfg.model.num_layers == 2
    assert cfg.model.dropout == 0.05
    assert cfg.model.mlqds_hybrid_mode == "swap"
    assert cfg.model.mlqds_stratified_center_weight == 0.45
    assert cfg.model.mlqds_min_learned_swaps == 1
    assert cfg.model.mlqds_score_mode == "rank_confidence"
    assert cfg.model.mlqds_score_temperature == 0.50
    assert cfg.model.mlqds_rank_confidence_weight == 0.30
    assert cfg.model.mlqds_range_geometry_blend == 0.40
    assert cfg.model.range_audit_compression_ratios == [0.01, 0.02, 0.10]
    assert cfg.model.range_label_mode == "point_f1"
    assert cfg.model.range_target_balance_mode == "trajectory_unit_mass"
    assert cfg.query.range_train_workload_replicates == 4
    assert cfg.query.range_time_domain_mode == "anchor_day"
    assert cfg.query.range_anchor_mode == "sparse"
    assert cfg.query.range_train_anchor_modes == ["mixed_density", "sparse"]
    assert cfg.query.range_train_footprints == ["1.1:2.5", "2.2:5"]
    assert cfg.query.range_max_coverage_overshoot == 0.02
    assert cfg.model.range_replicate_target_aggregation == "frequency_mean"
    assert cfg.model.range_component_target_blend == 0.40
    assert cfg.model.range_temporal_target_blend == 0.15
    assert cfg.model.range_structural_target_blend == 0.35
    assert cfg.model.range_structural_target_source_mode == "boost"
    assert cfg.model.range_target_budget_weight_power == 0.75
    assert cfg.model.range_marginal_target_radius_scale == 0.65
    assert cfg.model.range_query_spine_fraction == 0.20
    assert cfg.model.range_query_spine_mass_mode == "query"
    assert cfg.model.range_query_residual_multiplier == 1.25
    assert cfg.model.range_query_residual_mass_mode == "point"
    assert cfg.model.range_set_utility_multiplier == 1.75
    assert cfg.model.range_set_utility_candidate_limit == 64
    assert cfg.model.range_set_utility_mass_mode == "query"
    assert cfg.model.query_prior_grid_bins == 128
    assert cfg.model.query_prior_smoothing_passes == 0
    assert cfg.model.loss_objective == "budget_topk"
    assert cfg.model.budget_loss_ratios == [0.01, 0.05]
    assert cfg.model.budget_loss_temperature == 0.20
    assert cfg.model.query_useful_aux_loss_weight == 0.75
    assert cfg.model.query_useful_segment_budget_head_weight == 0.35
    assert cfg.model.query_useful_segment_level_loss_weight == 0.90
    assert cfg.model.temporal_distribution_loss_weight == 0.20
    assert cfg.model.checkpoint_full_score_every == 3
    assert cfg.model.checkpoint_candidate_pool_size == 2
    assert cfg.data.range_diagnostics_mode == "cached"
    assert cfg.baselines.final_metrics_mode == "core"
    assert cfg.model.model_type == "historical_prior"
    assert cfg.model.historical_prior_k == 7
    assert cfg.model.historical_prior_clock_weight == 0.25
    assert cfg.model.historical_prior_mmsi_weight == 2.5
    assert cfg.model.historical_prior_density_weight == 3.5
    assert cfg.model.historical_prior_min_target == 0.25
    assert cfg.model.historical_prior_support_ratio == 0.40
    assert cfg.model.historical_prior_source_aggregation == "mean"
    assert cfg.data.train_max_segments == 48
    assert cfg.data.validation_max_segments == 32
    assert cfg.data.eval_max_segments == 24
    assert cfg.data.validation_csv_path == "validation.csv"


def test_experiment_config_loads_missing_runtime_and_mlqds_defaults() -> None:
    payload = build_experiment_config().to_dict()
    payload["model"].pop("float32_matmul_precision")
    payload["model"].pop("allow_tf32")
    payload["model"].pop("inference_batch_size")
    payload["model"].pop("amp_mode")
    payload["model"].pop("historical_prior_clock_weight")
    payload["model"].pop("historical_prior_mmsi_weight")
    payload["model"].pop("historical_prior_density_weight")
    payload["model"].pop("historical_prior_min_target")
    payload["model"].pop("historical_prior_support_ratio")
    payload["model"].pop("historical_prior_source_aggregation")
    payload["model"].pop("range_boundary_prior_weight")
    payload["model"].pop("range_label_mode")
    payload["model"].pop("range_target_balance_mode")
    payload["model"].pop("range_replicate_target_aggregation")
    payload["model"].pop("range_component_target_blend")
    payload["model"].pop("range_temporal_target_blend")
    payload["model"].pop("range_structural_target_blend")
    payload["model"].pop("range_structural_target_source_mode")
    payload["model"].pop("range_target_budget_weight_power")
    payload["model"].pop("range_marginal_target_radius_scale")
    payload["model"].pop("range_query_spine_fraction")
    payload["model"].pop("range_query_spine_mass_mode")
    payload["model"].pop("range_query_residual_multiplier")
    payload["model"].pop("range_query_residual_mass_mode")
    payload["model"].pop("range_set_utility_multiplier")
    payload["model"].pop("range_set_utility_candidate_limit")
    payload["model"].pop("range_set_utility_mass_mode")
    payload["model"].pop("loss_objective")
    payload["model"].pop("budget_loss_ratios")
    payload["model"].pop("budget_loss_temperature")
    payload["model"].pop("temporal_distribution_loss_weight")
    payload["model"].pop("range_audit_compression_ratios")
    payload["model"].pop("mlqds_score_mode")
    payload["model"].pop("mlqds_score_temperature")
    payload["model"].pop("mlqds_rank_confidence_weight")
    payload["model"].pop("mlqds_stratified_center_weight")
    payload["model"].pop("mlqds_min_learned_swaps")
    payload["model"].pop("checkpoint_full_score_every")
    payload["model"].pop("checkpoint_candidate_pool_size")
    payload["model"].pop("query_prior_grid_bins")
    payload["model"].pop("query_prior_smoothing_passes")
    payload["query"].pop("range_train_workload_replicates")
    payload["query"].pop("range_time_domain_mode")
    payload["query"].pop("range_anchor_mode")
    payload["query"].pop("range_train_anchor_modes")
    payload["query"].pop("range_train_footprints")
    payload["query"].pop("range_max_coverage_overshoot")
    payload["data"].pop("train_max_segments")
    payload["data"].pop("validation_max_segments")
    payload["data"].pop("eval_max_segments")
    payload["data"].pop("range_diagnostics_mode")
    payload["baselines"].pop("final_metrics_mode")

    restored = ExperimentConfig.from_dict(payload)

    assert restored.model.float32_matmul_precision == "highest"
    assert restored.model.allow_tf32 is False
    assert restored.model.inference_batch_size == 16
    assert restored.model.amp_mode == "off"
    assert restored.model.historical_prior_clock_weight == 0.0
    assert restored.model.historical_prior_mmsi_weight == 1.0
    assert restored.model.historical_prior_density_weight == 1.0
    assert restored.model.historical_prior_min_target == 0.0
    assert restored.model.historical_prior_support_ratio == 1.0
    assert restored.model.historical_prior_source_aggregation == "none"
    assert restored.model.range_boundary_prior_weight == 0.0
    assert restored.model.range_label_mode == "usefulness"
    assert restored.model.range_target_balance_mode == "none"
    assert restored.model.range_replicate_target_aggregation == "label_mean"
    assert restored.model.range_component_target_blend == 1.0
    assert restored.model.range_temporal_target_blend == 0.0
    assert restored.model.range_structural_target_blend == 0.25
    assert restored.model.range_structural_target_source_mode == "blend"
    assert restored.model.range_target_budget_weight_power == 0.0
    assert restored.model.range_marginal_target_radius_scale == 0.50
    assert restored.model.range_query_spine_fraction == 0.10
    assert restored.model.range_query_spine_mass_mode == "hit_group"
    assert restored.model.range_query_residual_multiplier == 1.0
    assert restored.model.range_query_residual_mass_mode == "query"
    assert restored.model.range_set_utility_multiplier == 1.0
    assert restored.model.range_set_utility_candidate_limit == 128
    assert restored.model.range_set_utility_mass_mode == "gain"
    assert restored.model.loss_objective == "budget_topk"
    assert restored.model.budget_loss_ratios == DEFAULT_BUDGET_LOSS_RATIOS
    assert restored.model.budget_loss_temperature == DEFAULT_BUDGET_LOSS_TEMPERATURE
    assert restored.model.temporal_distribution_loss_weight == 0.0
    assert restored.model.range_audit_compression_ratios == []
    assert restored.model.mlqds_score_mode == "rank"
    assert restored.model.mlqds_score_temperature == 1.0
    assert restored.model.mlqds_rank_confidence_weight == 0.15
    assert restored.model.mlqds_stratified_center_weight == 0.0
    assert restored.model.mlqds_min_learned_swaps == 0
    assert restored.model.mlqds_range_geometry_blend == 0.0
    assert restored.model.checkpoint_selection_metric == "score"
    assert restored.model.checkpoint_score_variant == "range_usefulness"
    assert restored.model.checkpoint_full_score_every == 1
    assert restored.model.checkpoint_candidate_pool_size == 1
    assert restored.model.query_prior_grid_bins == 64
    assert restored.model.query_prior_smoothing_passes == 2
    assert restored.query.range_train_workload_replicates == 1
    assert restored.query.range_time_domain_mode == "dataset"
    assert restored.query.range_anchor_mode == "mixed_density"
    assert restored.query.range_train_anchor_modes == []
    assert restored.query.range_train_footprints == []
    assert restored.query.range_max_coverage_overshoot is None
    assert restored.data.train_max_segments is None
    assert restored.data.validation_max_segments is None
    assert restored.data.eval_max_segments is None
    assert restored.data.range_diagnostics_mode == "full"
    assert restored.baselines.final_metrics_mode == "diagnostic"


def test_split_max_segments_falls_back_to_global_cap() -> None:
    args = Namespace(
        max_segments=120,
        train_max_segments=240,
        validation_max_segments=None,
        eval_max_segments=80,
    )

    assert _split_max_segments(args, "train") == 240
    assert _split_max_segments(args, "validation") == 120
    assert _split_max_segments(args, "eval") == 80


def test_validation_score_config_uses_current_names() -> None:
    payload = build_experiment_config(
        validation_score_every=2,
        checkpoint_full_score_every=4,
        checkpoint_score_variant="answer",
        learned_segment_length_repair_fraction=0.25,
        learned_segment_length_support_blend_weight=0.75,
        query_prior_grid_bins=128,
        query_prior_smoothing_passes=0,
        temporal_residual_label_mode="none",
    ).to_dict()
    restored = ExperimentConfig.from_dict(payload)
    args = build_parser().parse_args(
        [
            "--validation_score_every",
            "3",
            "--checkpoint_full_score_every",
            "5",
            "--checkpoint_score_variant",
            "combined",
            "--learned_segment_length_repair_fraction",
            "0.5",
            "--learned_segment_length_support_blend_weight",
            "1.0",
            "--query_prior_grid_bins",
            "96",
            "--query_prior_smoothing_passes",
            "1",
            "--temporal_residual_label_mode",
            "none",
        ]
    )

    assert restored.model.validation_score_every == 2
    assert restored.model.checkpoint_full_score_every == 4
    assert restored.model.checkpoint_score_variant == "answer"
    assert restored.model.learned_segment_length_repair_fraction == pytest.approx(0.25)
    assert restored.model.learned_segment_length_support_blend_weight == pytest.approx(0.75)
    assert restored.model.query_prior_grid_bins == 128
    assert restored.model.query_prior_smoothing_passes == 0
    assert restored.model.temporal_residual_label_mode == "none"
    assert not hasattr(restored.model, "f1_diagnostic_every")
    assert not hasattr(restored.model, "residual_label_mode")
    assert args.validation_score_every == 3
    assert args.checkpoint_full_score_every == 5
    assert args.checkpoint_score_variant == "combined"
    assert args.learned_segment_length_repair_fraction == pytest.approx(0.5)
    assert args.learned_segment_length_support_blend_weight == pytest.approx(1.0)
    assert args.query_prior_grid_bins == 96
    assert args.query_prior_smoothing_passes == 1
    assert args.temporal_residual_label_mode == "none"


def test_direct_config_and_cli_default_to_non_residual_training() -> None:
    cfg = build_experiment_config()
    args = build_parser().parse_args([])

    assert cfg.model.temporal_residual_label_mode == "none"
    assert args.temporal_residual_label_mode == "none"


def test_cli_model_type_choices_use_supported_model_registry() -> None:
    parser = build_parser()
    model_action = next(action for action in parser._actions if action.dest == "model_type")

    assert tuple(model_action.choices or ()) == SUPPORTED_MODEL_TYPES


def test_parser_accepts_stratified_budget_loss_diagnostic() -> None:
    args = build_parser().parse_args(
        [
            "--loss_objective",
            "stratified_budget_topk",
            "--mlqds_hybrid_mode",
            "stratified",
        ]
    )

    assert args.loss_objective == "stratified_budget_topk"
    assert args.mlqds_hybrid_mode == "stratified"


def test_parser_accepts_local_swap_gain_cost_target() -> None:
    args = build_parser().parse_args(
        [
            "--range_training_target_mode",
            "local_swap_gain_cost_frequency",
            "--mlqds_hybrid_mode",
            "local_delta_swap",
        ]
    )

    assert args.range_training_target_mode == "local_swap_gain_cost_frequency"
    assert args.mlqds_hybrid_mode == "local_delta_swap"


def test_parser_accepts_global_budget_target_and_selector() -> None:
    args = build_parser().parse_args(
        [
            "--range_training_target_mode",
            "global_budget_retained_frequency",
            "--mlqds_hybrid_mode",
            "global_budget",
        ]
    )

    assert args.range_training_target_mode == "global_budget_retained_frequency"
    assert args.mlqds_hybrid_mode == "global_budget"


def test_parser_accepts_global_fill_selector() -> None:
    args = build_parser().parse_args(
        [
            "--mlqds_hybrid_mode",
            "global_fill",
        ]
    )

    assert args.mlqds_hybrid_mode == "global_fill"


def test_parser_accepts_structural_target_blend() -> None:
    args = build_parser().parse_args(
        [
            "--range_training_target_mode",
            "structural_retained_frequency",
            "--range_structural_target_blend",
            "0.40",
            "--range_structural_target_source_mode",
            "boost",
        ]
    )

    assert args.range_training_target_mode == "structural_retained_frequency"
    assert args.range_structural_target_blend == 0.40
    assert args.range_structural_target_source_mode == "boost"


def test_experiment_config_rejects_unknown_model_keys() -> None:
    payload = build_experiment_config().to_dict()
    payload["model"]["f1_diagnostic_every"] = 1

    with pytest.raises(TypeError, match="f1_diagnostic_every"):
        ExperimentConfig.from_dict(payload)


def test_checkpoint_config_payload_filters_stale_section_keys() -> None:
    payload = build_experiment_config().to_dict()
    payload["legacy_top_level"] = True
    payload["data"]["stale_data_key"] = 1
    payload["query"]["stale_query_key"] = 2
    payload["model"]["f1_diagnostic_every"] = 3
    payload["model"]["residual_label_mode"] = "temporal"
    payload["baselines"]["stale_baseline_key"] = 4

    restored = ExperimentConfig.from_dict(_checkpoint_config_payload(payload))

    assert not hasattr(restored.data, "stale_data_key")
    assert not hasattr(restored.query, "stale_query_key")
    assert not hasattr(restored.model, "f1_diagnostic_every")
    assert not hasattr(restored.model, "residual_label_mode")
    assert not hasattr(restored.baselines, "stale_baseline_key")


def test_checkpoint_config_payload_supplies_missing_sections() -> None:
    restored = ExperimentConfig.from_dict(_checkpoint_config_payload({"model": {"model_type": "baseline"}}))

    assert restored.data.min_points_per_segment == 4
    assert restored.query.workload == "range"
    assert restored.model.model_type == "baseline"
    assert restored.baselines.final_metrics_mode == "diagnostic"


def test_amp_helpers_default_to_cuda_only_autocast() -> None:
    assert normalize_amp_mode(None) == "off"
    assert normalize_amp_mode(" BF16 ") == "bf16"

    cpu_snapshot = amp_runtime_snapshot("bf16", device="cpu")

    assert cpu_snapshot == {
        "mode": "bf16",
        "enabled": False,
        "device_type": "cpu",
        "dtype": "bfloat16",
    }
    with torch_autocast_context("cpu", "bf16"):
        value = torch.ones((2,), dtype=torch.float32) + 1.0
    assert value.dtype == torch.float32


def test_runtime_child_args_forward_amp_mode() -> None:
    assert _runtime_child_args("high", True, "bf16") == [
        "--float32_matmul_precision",
        "high",
        "--allow_tf32",
        "--amp_mode",
        "bf16",
    ]


def test_parse_train_batch_sizes() -> None:
    assert _parse_train_batch_sizes("16, 32,64") == [16, 32, 64]
    assert _parse_train_batch_sizes(None) is None


def test_runtime_profile_uses_workload_aware_diagnostic_shape(tmp_path) -> None:
    args = _profile_train_args(DEFAULT_PROFILE, seed=42, results_dir=tmp_path / "run", checkpoint=tmp_path / "m.pt")

    assert "--n_queries" in args
    assert args[args.index("--n_queries") + 1] == "80"
    assert args[args.index("--max_queries") + 1] == "2048"
    assert args[args.index("--compression_ratio") + 1] == "0.05"
    assert args[args.index("--query_chunk_size") + 1] == "2048"
    assert args[args.index("--train_batch_size") + 1] == "64"
    assert args[args.index("--inference_batch_size") + 1] == "64"
    assert args[args.index("--model_type") + 1] == "range_aware"
    assert args[args.index("--query_coverage") + 1] == "0.20"
    assert args[args.index("--range_spatial_km") + 1] == "2.2"
    assert args[args.index("--range_time_hours") + 1] == "5.0"
    assert args[args.index("--range_footprint_jitter") + 1] == "0.0"
    assert args[args.index("--range_max_coverage_overshoot") + 1] == "0.02"
    assert args[args.index("--range_time_domain_mode") + 1] == "anchor_day"
    assert args[args.index("--range_anchor_mode") + 1] == "mixed_density"
    assert args[args.index("--range_diagnostics_mode") + 1] == "cached"
    assert args[args.index("--final_metrics_mode") + 1] == "diagnostic"
    assert args[args.index("--early_stopping_patience") + 1] == "5"
    assert args[args.index("--validation_score_every") + 1] == "1"
    assert args[args.index("--checkpoint_smoothing_window") + 1] == "1"
    assert args[args.index("--checkpoint_full_score_every") + 1] == "4"
    assert args[args.index("--checkpoint_candidate_pool_size") + 1] == "2"
    assert args[args.index("--loss_objective") + 1] == "budget_topk"
    assert args[args.index("--budget_loss_ratios") + 1] == "0.05,0.10"
    assert args[args.index("--range_audit_compression_ratios") + 1] == (
        "0.01,0.02,0.05,0.10,0.15,0.20,0.30"
    )
    assert args[args.index("--budget_loss_temperature") + 1] == "0.25"
    assert args[args.index("--temporal_distribution_loss_weight") + 1] == "0.000"
    assert args[args.index("--mlqds_temporal_fraction") + 1] == "0.25"
    assert args[args.index("--mlqds_score_mode") + 1] == "rank"
    assert args[args.index("--mlqds_score_temperature") + 1] == "1.00"
    assert args[args.index("--mlqds_rank_confidence_weight") + 1] == "0.15"
    assert args[args.index("--mlqds_range_geometry_blend") + 1] == "0.00"
    assert args[args.index("--mlqds_diversity_bonus") + 1] == "0.00"
    assert args[args.index("--mlqds_hybrid_mode") + 1] == "fill"
    assert args[args.index("--mlqds_stratified_center_weight") + 1] == "0.00"
    assert args[args.index("--temporal_residual_label_mode") + 1] == "none"
    assert args[args.index("--range_label_mode") + 1] == "usefulness"
    assert args[args.index("--range_temporal_target_blend") + 1] == "0.000"
    assert args[args.index("--range_target_budget_weight_power") + 1] == "0.00"
    assert args[args.index("--range_marginal_target_radius_scale") + 1] == "0.50"
    assert args[args.index("--range_query_spine_fraction") + 1] == "0.10"
    assert args[args.index("--range_query_spine_mass_mode") + 1] == "hit_group"
    assert args[args.index("--range_query_residual_multiplier") + 1] == "1.00"
    assert args[args.index("--range_query_residual_mass_mode") + 1] == "query"
    assert args[args.index("--range_set_utility_multiplier") + 1] == "1.00"
    assert args[args.index("--range_set_utility_candidate_limit") + 1] == "128"
    assert args[args.index("--range_set_utility_mass_mode") + 1] == "gain"
    assert args[args.index("--range_boundary_prior_weight") + 1] == "0.0"
    assert "--n_ships" not in args
    assert "--n_points" not in args


def test_runtime_profile_requires_real_training_data_source() -> None:
    assert _extra_args_include_training_data_source("--csv_path ../AISDATA/cleaned/day.csv")
    assert _extra_args_include_training_data_source(
        "--train_csv_path=train.csv --validation_csv_path validation.csv --eval_csv_path eval.csv"
    )
    assert not _extra_args_include_training_data_source("--max_segments 10")
    assert not _extra_args_include_training_data_source("--validation_csv_path validation.csv")


def test_batch_size_sweep_summary_extracts_timing_memory_and_score() -> None:
    rows = _batch_size_sweep_summary(
        [
            {
                "name": "train_bs32",
                "train_batch_size": 32,
                "returncode": 0,
                "elapsed_seconds": 12.5,
                "timings": {"epoch_timings": [{"seconds": 2.0}, {"seconds": 3.0}]},
                "metrics": {
                    "best_selection_score": 0.4,
                    "batch_size": {"train_batch_size": 32},
                    "cuda_memory": {
                        "training": {
                            "max_allocated_mb": 123.0,
                            "max_reserved_mb": 256.0,
                        }
                    },
                    "methods": {"MLQDS": {"aggregate_f1": 0.5}},
                },
            }
        ]
    )

    assert rows == [
        {
            "train_batch_size": 32,
            "returncode": 0,
            "elapsed_seconds": 12.5,
            "epoch_time_mean_seconds": 2.5,
            "epoch_time_min_seconds": 2.0,
            "epoch_time_max_seconds": 3.0,
            "peak_allocated_mb": 123.0,
            "peak_reserved_mb": 256.0,
            "best_selection_score": 0.4,
            "mlqds_aggregate_f1": 0.5,
            "mlqds_range_usefulness_score": None,
            "mlqds_range_ship_coverage": None,
            "mlqds_range_crossing_f1": None,
            "mlqds_range_gap_coverage": None,
            "mlqds_range_gap_time_coverage": None,
            "mlqds_range_gap_distance_coverage": None,
            "mlqds_range_turn_coverage": None,
        }
    ]
