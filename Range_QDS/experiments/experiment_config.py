"""Experiment configuration dataclasses for AIS-QDS. See experiments/README.md for details."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

LCG_MULTIPLIER = 6364136223846793005
DEFAULT_BUDGET_LOSS_RATIOS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
DEFAULT_BUDGET_LOSS_TEMPERATURE = 0.25
VALIDATION_SPLIT_MODES = ("random", "source_stratified")


@dataclass
class DataConfig:
    """Data loading and splitting configuration. See data/README.md for details."""

    n_ships: int | None = 24
    n_points_per_ship: int | None = 200
    synthetic_route_families: int = 0
    min_points_per_segment: int = 4
    max_points_per_segment: int | None = None
    max_time_gap_seconds: float | None = 3600.0
    max_segments: int | None = None
    train_max_segments: int | None = None
    validation_max_segments: int | None = None
    eval_max_segments: int | None = None
    max_trajectories: int | None = None
    csv_path: str | None = None
    train_csv_path: str | None = None
    validation_csv_path: str | None = None
    eval_csv_path: str | None = None
    cache_dir: str | None = None
    refresh_cache: bool = False
    range_diagnostics_mode: str = "full"
    seed: int = 42
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    validation_split_mode: str = "random"

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a dictionary. See experiments/README.md for details."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataConfig":
        """Deserialize config from a dictionary. See experiments/README.md for details."""
        return cls(**data)


@dataclass
class QueryConfig:
    """Query generation and pure workload configuration. See queries/README.md for details."""

    n_queries: int = 128
    target_coverage: float | None = None
    max_queries: int | None = None
    range_spatial_fraction: float = 0.08
    range_time_fraction: float = 0.15
    range_spatial_km: float | None = None
    range_time_hours: float | None = None
    range_footprint_jitter: float = 0.5
    range_time_domain_mode: str = "dataset"
    range_anchor_mode: str = "mixed_density"
    range_train_anchor_modes: list[str] = field(default_factory=list)
    range_train_footprints: list[str] = field(default_factory=list)
    workload: str = "range"
    range_min_point_hits: int | None = None
    range_max_point_hit_fraction: float | None = None
    range_min_trajectory_hits: int | None = None
    range_max_trajectory_hit_fraction: float | None = None
    range_max_box_volume_fraction: float | None = None
    range_duplicate_iou_threshold: float | None = None
    range_acceptance_max_attempts: int | None = None
    range_max_coverage_overshoot: float | None = None
    range_train_workload_replicates: int = 1
    workload_profile_id: str | None = None
    coverage_calibration_mode: str | None = None
    workload_stability_gate_mode: str = "final"

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a dictionary. See experiments/README.md for details."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QueryConfig":
        """Deserialize config from a dictionary. See experiments/README.md for details."""
        payload = dict(data)
        raw_train_anchor_modes = payload.get("range_train_anchor_modes", [])
        if isinstance(raw_train_anchor_modes, str):
            payload["range_train_anchor_modes"] = [
                item.strip() for item in raw_train_anchor_modes.split(",") if item.strip()
            ]
        elif raw_train_anchor_modes is None:
            payload["range_train_anchor_modes"] = []
        else:
            payload["range_train_anchor_modes"] = list(raw_train_anchor_modes)
        raw_train_footprints = payload.get("range_train_footprints", [])
        if isinstance(raw_train_footprints, str):
            payload["range_train_footprints"] = [
                item.strip() for item in raw_train_footprints.split(",") if item.strip()
            ]
        elif raw_train_footprints is None:
            payload["range_train_footprints"] = []
        else:
            payload["range_train_footprints"] = list(raw_train_footprints)
        return cls(**payload)


@dataclass
class ModelConfig:
    """Model architecture and training behavior config. See models/README.md for details."""

    embed_dim: int = 64
    num_heads: int = 4
    num_layers: int = 3
    type_embed_dim: int = 16
    query_chunk_size: int = 2048
    dropout: float = 0.1
    window_length: int = 512
    window_stride: int = 256
    epochs: int = 6
    lr: float = 5e-4
    compression_ratio: float = 0.2
    model_type: str = "baseline"
    historical_prior_k: int = 32
    historical_prior_clock_weight: float = 0.0
    historical_prior_mmsi_weight: float = 1.0
    historical_prior_density_weight: float = 1.0
    historical_prior_min_target: float = 0.0
    historical_prior_support_ratio: float = 1.0
    historical_prior_source_aggregation: str = "none"
    rank_margin: float = 0.05
    ranking_pairs_per_type: int = 96
    ranking_top_quantile: float = 0.80
    pointwise_loss_weight: float = 0.25
    loss_objective: str = "budget_topk"
    budget_loss_ratios: list[float] = field(default_factory=lambda: list(DEFAULT_BUDGET_LOSS_RATIOS))
    budget_loss_temperature: float = DEFAULT_BUDGET_LOSS_TEMPERATURE
    temporal_distribution_loss_weight: float = 0.0
    gradient_clip_norm: float = 1.0
    l2_score_weight: float = 1e-4
    early_stopping_patience: int = 0
    train_batch_size: int = 16
    inference_batch_size: int = 16
    diagnostic_every: int = 1
    diagnostic_window_fraction: float = 0.2
    checkpoint_selection_metric: str = "score"
    validation_score_every: int = 0
    checkpoint_uniform_gap_weight: float = 0.5
    checkpoint_type_penalty_weight: float = 1.0
    checkpoint_smoothing_window: int = 1
    checkpoint_full_score_every: int = 1
    checkpoint_candidate_pool_size: int = 1
    checkpoint_score_variant: str = "range_usefulness"
    validation_global_sanity_penalty_enabled: bool = True
    validation_global_sanity_penalty_weight: float = 0.35
    validation_sed_penalty_weight: float = 0.15
    validation_endpoint_penalty_weight: float = 0.10
    validation_length_preservation_min: float = 0.80
    mlqds_temporal_fraction: float = 0.0
    mlqds_diversity_bonus: float = 0.0
    mlqds_hybrid_mode: str = "fill"
    selector_type: str = "temporal_hybrid"
    learned_segment_geometry_gain_weight: float = 0.12
    learned_segment_score_blend_weight: float = 0.05
    learned_segment_fairness_preallocation: bool = True
    mlqds_stratified_center_weight: float = 0.0
    mlqds_min_learned_swaps: int = 0
    mlqds_score_mode: str = "rank"
    mlqds_score_temperature: float = 1.0
    mlqds_rank_confidence_weight: float = 0.15
    mlqds_range_geometry_blend: float = 0.0
    temporal_residual_label_mode: str = "none"
    range_label_mode: str = "usefulness"
    range_training_target_mode: str = "point_value"
    range_target_balance_mode: str = "none"
    range_replicate_target_aggregation: str = "label_mean"
    range_component_target_blend: float = 1.0
    range_temporal_target_blend: float = 0.0
    range_structural_target_blend: float = 0.25
    range_structural_target_source_mode: str = "blend"
    range_target_budget_weight_power: float = 0.0
    range_marginal_target_radius_scale: float = 0.50
    range_query_spine_fraction: float = 0.10
    range_query_spine_mass_mode: str = "hit_group"
    range_query_residual_multiplier: float = 1.0
    range_query_residual_mass_mode: str = "query"
    range_set_utility_multiplier: float = 1.0
    range_set_utility_candidate_limit: int = 128
    range_set_utility_mass_mode: str = "gain"
    range_boundary_prior_weight: float = 0.0
    range_teacher_distillation_mode: str = "none"
    range_teacher_epochs: int = 4
    range_audit_compression_ratios: list[float] = field(default_factory=list)
    float32_matmul_precision: str = "highest"
    allow_tf32: bool = False
    amp_mode: str = "off"

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a dictionary. See experiments/README.md for details."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        """Deserialize config from a dictionary. See experiments/README.md for details."""
        return cls(**data)


@dataclass
class BaselineConfig:
    """Baseline methods configuration. See evaluation/README.md for details."""

    include_oracle: bool = True
    final_metrics_mode: str = "diagnostic"

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a dictionary. See experiments/README.md for details."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaselineConfig":
        """Deserialize config from a dictionary. See experiments/README.md for details."""
        return cls(**data)


@dataclass
class ExperimentConfig:
    """Top-level experiment config container. See experiments/README.md for details."""

    data: DataConfig = field(default_factory=DataConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    baselines: BaselineConfig = field(default_factory=BaselineConfig)

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a dictionary. See experiments/README.md for details."""
        return {
            "data": self.data.to_dict(),
            "query": self.query.to_dict(),
            "model": self.model.to_dict(),
            "baselines": self.baselines.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        """Deserialize config from a dictionary. See experiments/README.md for details."""
        expected_keys = {"data", "query", "model", "baselines"}
        unknown_keys = set(data) - expected_keys
        if unknown_keys:
            raise TypeError(f"Unknown ExperimentConfig keys: {sorted(unknown_keys)}")
        return cls(
            data=DataConfig.from_dict(data["data"]),
            query=QueryConfig.from_dict(data["query"]),
            model=ModelConfig.from_dict(data["model"]),
            baselines=BaselineConfig.from_dict(data["baselines"]),
        )


@dataclass
class SeedBundle:
    """Derived deterministic sub-seeds for experiment stages. See experiments/README.md for details."""

    split_seed: int
    train_query_seed: int
    eval_query_seed: int
    torch_seed: int


def build_experiment_config(
    n_ships: int = 24,
    n_points: int = 200,
    synthetic_route_families: int = 0,
    min_points_per_segment: int = 4,
    max_points_per_segment: int | None = None,
    max_time_gap_seconds: float | None = 3600.0,
    max_segments: int | None = None,
    train_max_segments: int | None = None,
    validation_max_segments: int | None = None,
    eval_max_segments: int | None = None,
    max_trajectories: int | None = None,
    n_queries: int = 128,
    query_coverage: float | None = None,
    max_queries: int | None = None,
    range_spatial_fraction: float = 0.08,
    range_time_fraction: float = 0.15,
    range_spatial_km: float | None = None,
    range_time_hours: float | None = None,
    range_footprint_jitter: float = 0.5,
    range_time_domain_mode: str = "dataset",
    range_anchor_mode: str = "mixed_density",
    range_train_anchor_modes: list[str] | None = None,
    range_train_footprints: list[str] | None = None,
    range_min_point_hits: int | None = None,
    range_max_point_hit_fraction: float | None = None,
    range_min_trajectory_hits: int | None = None,
    range_max_trajectory_hit_fraction: float | None = None,
    range_max_box_volume_fraction: float | None = None,
    range_duplicate_iou_threshold: float | None = None,
    range_acceptance_max_attempts: int | None = None,
    range_max_coverage_overshoot: float | None = None,
    range_train_workload_replicates: int = 1,
    workload_profile_id: str | None = None,
    coverage_calibration_mode: str | None = None,
    workload_stability_gate_mode: str = "final",
    epochs: int = 6,
    lr: float = 5e-4,
    embed_dim: int = 64,
    num_heads: int = 4,
    num_layers: int = 3,
    dropout: float = 0.1,
    ranking_pairs_per_type: int = 96,
    ranking_top_quantile: float = 0.80,
    pointwise_loss_weight: float = 0.25,
    loss_objective: str = "budget_topk",
    budget_loss_ratios: list[float] | None = None,
    budget_loss_temperature: float = DEFAULT_BUDGET_LOSS_TEMPERATURE,
    temporal_distribution_loss_weight: float = 0.0,
    gradient_clip_norm: float = 1.0,
    compression_ratio: float = 0.2,
    csv_path: str | None = None,
    train_csv_path: str | None = None,
    validation_csv_path: str | None = None,
    eval_csv_path: str | None = None,
    cache_dir: str | None = None,
    refresh_cache: bool = False,
    range_diagnostics_mode: str = "full",
    validation_split_mode: str = "random",
    model_type: str = "baseline",
    historical_prior_k: int = 32,
    historical_prior_clock_weight: float = 0.0,
    historical_prior_mmsi_weight: float = 1.0,
    historical_prior_density_weight: float = 1.0,
    historical_prior_min_target: float = 0.0,
    historical_prior_support_ratio: float = 1.0,
    historical_prior_source_aggregation: str = "none",
    workload: str = "range",
    seed: int = 42,
    early_stopping_patience: int = 0,
    train_batch_size: int = 16,
    inference_batch_size: int = 16,
    query_chunk_size: int = 2048,
    diagnostic_every: int = 1,
    diagnostic_window_fraction: float = 0.2,
    checkpoint_selection_metric: str = "score",
    validation_score_every: int | None = None,
    checkpoint_uniform_gap_weight: float = 0.5,
    checkpoint_type_penalty_weight: float = 1.0,
    checkpoint_smoothing_window: int = 1,
    checkpoint_full_score_every: int | None = None,
    checkpoint_candidate_pool_size: int = 1,
    checkpoint_score_variant: str | None = None,
    validation_global_sanity_penalty_enabled: bool = True,
    validation_global_sanity_penalty_weight: float = 0.35,
    validation_sed_penalty_weight: float = 0.15,
    validation_endpoint_penalty_weight: float = 0.10,
    validation_length_preservation_min: float = 0.80,
    mlqds_temporal_fraction: float = 0.0,
    mlqds_diversity_bonus: float = 0.0,
    mlqds_hybrid_mode: str = "fill",
    selector_type: str = "temporal_hybrid",
    learned_segment_geometry_gain_weight: float = 0.12,
    learned_segment_score_blend_weight: float = 0.05,
    learned_segment_fairness_preallocation: bool = True,
    mlqds_stratified_center_weight: float = 0.0,
    mlqds_min_learned_swaps: int = 0,
    mlqds_score_mode: str = "rank",
    mlqds_score_temperature: float = 1.0,
    mlqds_rank_confidence_weight: float = 0.15,
    mlqds_range_geometry_blend: float = 0.0,
    temporal_residual_label_mode: str | None = None,
    range_label_mode: str = "usefulness",
    range_training_target_mode: str = "point_value",
    range_target_balance_mode: str = "none",
    range_replicate_target_aggregation: str = "label_mean",
    range_component_target_blend: float = 1.0,
    range_temporal_target_blend: float = 0.0,
    range_structural_target_blend: float = 0.25,
    range_structural_target_source_mode: str = "blend",
    range_target_budget_weight_power: float = 0.0,
    range_marginal_target_radius_scale: float = 0.50,
    range_query_spine_fraction: float = 0.10,
    range_query_spine_mass_mode: str = "hit_group",
    range_query_residual_multiplier: float = 1.0,
    range_query_residual_mass_mode: str = "query",
    range_set_utility_multiplier: float = 1.0,
    range_set_utility_candidate_limit: int = 128,
    range_set_utility_mass_mode: str = "gain",
    range_boundary_prior_weight: float = 0.0,
    range_teacher_distillation_mode: str = "none",
    range_teacher_epochs: int = 4,
    range_audit_compression_ratios: list[float] | None = None,
    final_metrics_mode: str = "diagnostic",
    float32_matmul_precision: str = "highest",
    allow_tf32: bool = False,
    amp_mode: str = "off",
) -> ExperimentConfig:
    """Build a structured experiment config from flat arguments. See experiments/README.md for details."""
    uses_csv = bool(csv_path or train_csv_path or validation_csv_path or eval_csv_path)
    return ExperimentConfig(
        data=DataConfig(
            n_ships=None if uses_csv else n_ships,
            n_points_per_ship=None if uses_csv else n_points,
            synthetic_route_families=0 if uses_csv else int(synthetic_route_families),
            min_points_per_segment=min_points_per_segment,
            max_points_per_segment=max_points_per_segment,
            max_time_gap_seconds=max_time_gap_seconds,
            max_segments=max_segments,
            train_max_segments=train_max_segments,
            validation_max_segments=validation_max_segments,
            eval_max_segments=eval_max_segments,
            max_trajectories=max_trajectories,
            csv_path=csv_path,
            train_csv_path=train_csv_path,
            validation_csv_path=validation_csv_path,
            eval_csv_path=eval_csv_path,
            cache_dir=cache_dir,
            refresh_cache=refresh_cache,
            range_diagnostics_mode=range_diagnostics_mode,
            validation_split_mode=validation_split_mode,
            seed=seed,
        ),
        query=QueryConfig(
            n_queries=n_queries,
            target_coverage=query_coverage,
            max_queries=max_queries,
            range_spatial_fraction=range_spatial_fraction,
            range_time_fraction=range_time_fraction,
            range_spatial_km=range_spatial_km,
            range_time_hours=range_time_hours,
            range_footprint_jitter=range_footprint_jitter,
            range_time_domain_mode=range_time_domain_mode,
            range_anchor_mode=range_anchor_mode,
            range_train_anchor_modes=list(range_train_anchor_modes or []),
            range_train_footprints=list(range_train_footprints or []),
            range_min_point_hits=range_min_point_hits,
            range_max_point_hit_fraction=range_max_point_hit_fraction,
            range_min_trajectory_hits=range_min_trajectory_hits,
            range_max_trajectory_hit_fraction=range_max_trajectory_hit_fraction,
            range_max_box_volume_fraction=range_max_box_volume_fraction,
            range_duplicate_iou_threshold=range_duplicate_iou_threshold,
            range_acceptance_max_attempts=range_acceptance_max_attempts,
            range_max_coverage_overshoot=range_max_coverage_overshoot,
            range_train_workload_replicates=range_train_workload_replicates,
            workload_profile_id=workload_profile_id,
            coverage_calibration_mode=coverage_calibration_mode,
            workload_stability_gate_mode=workload_stability_gate_mode,
            workload=workload,
        ),
        model=ModelConfig(
            epochs=epochs,
            lr=lr,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            ranking_pairs_per_type=ranking_pairs_per_type,
            ranking_top_quantile=ranking_top_quantile,
            pointwise_loss_weight=pointwise_loss_weight,
            loss_objective=loss_objective,
            budget_loss_ratios=list(budget_loss_ratios or DEFAULT_BUDGET_LOSS_RATIOS),
            budget_loss_temperature=budget_loss_temperature,
            temporal_distribution_loss_weight=temporal_distribution_loss_weight,
            gradient_clip_norm=gradient_clip_norm,
            compression_ratio=compression_ratio,
            model_type=model_type,
            historical_prior_k=historical_prior_k,
            historical_prior_clock_weight=historical_prior_clock_weight,
            historical_prior_mmsi_weight=historical_prior_mmsi_weight,
            historical_prior_density_weight=historical_prior_density_weight,
            historical_prior_min_target=historical_prior_min_target,
            historical_prior_support_ratio=historical_prior_support_ratio,
            historical_prior_source_aggregation=historical_prior_source_aggregation,
            early_stopping_patience=early_stopping_patience,
            train_batch_size=train_batch_size,
            inference_batch_size=inference_batch_size,
            query_chunk_size=query_chunk_size,
            diagnostic_every=diagnostic_every,
            diagnostic_window_fraction=diagnostic_window_fraction,
            checkpoint_selection_metric=checkpoint_selection_metric,
            validation_score_every=0 if validation_score_every is None else validation_score_every,
            checkpoint_uniform_gap_weight=checkpoint_uniform_gap_weight,
            checkpoint_type_penalty_weight=checkpoint_type_penalty_weight,
            checkpoint_smoothing_window=checkpoint_smoothing_window,
            checkpoint_full_score_every=1 if checkpoint_full_score_every is None else checkpoint_full_score_every,
            checkpoint_candidate_pool_size=checkpoint_candidate_pool_size,
            checkpoint_score_variant=checkpoint_score_variant or "range_usefulness",
            validation_global_sanity_penalty_enabled=validation_global_sanity_penalty_enabled,
            validation_global_sanity_penalty_weight=validation_global_sanity_penalty_weight,
            validation_sed_penalty_weight=validation_sed_penalty_weight,
            validation_endpoint_penalty_weight=validation_endpoint_penalty_weight,
            validation_length_preservation_min=validation_length_preservation_min,
            mlqds_temporal_fraction=mlqds_temporal_fraction,
            mlqds_diversity_bonus=mlqds_diversity_bonus,
            mlqds_hybrid_mode=mlqds_hybrid_mode,
            selector_type=selector_type,
            learned_segment_geometry_gain_weight=learned_segment_geometry_gain_weight,
            learned_segment_score_blend_weight=learned_segment_score_blend_weight,
            learned_segment_fairness_preallocation=learned_segment_fairness_preallocation,
            mlqds_stratified_center_weight=mlqds_stratified_center_weight,
            mlqds_min_learned_swaps=mlqds_min_learned_swaps,
            mlqds_score_mode=mlqds_score_mode,
            mlqds_score_temperature=mlqds_score_temperature,
            mlqds_rank_confidence_weight=mlqds_rank_confidence_weight,
            mlqds_range_geometry_blend=mlqds_range_geometry_blend,
            temporal_residual_label_mode=temporal_residual_label_mode or "none",
            range_label_mode=range_label_mode,
            range_training_target_mode=range_training_target_mode,
            range_target_balance_mode=range_target_balance_mode,
            range_replicate_target_aggregation=range_replicate_target_aggregation,
            range_component_target_blend=range_component_target_blend,
            range_temporal_target_blend=range_temporal_target_blend,
            range_structural_target_blend=range_structural_target_blend,
            range_structural_target_source_mode=range_structural_target_source_mode,
            range_target_budget_weight_power=range_target_budget_weight_power,
            range_marginal_target_radius_scale=range_marginal_target_radius_scale,
            range_query_spine_fraction=range_query_spine_fraction,
            range_query_spine_mass_mode=range_query_spine_mass_mode,
            range_query_residual_multiplier=range_query_residual_multiplier,
            range_query_residual_mass_mode=range_query_residual_mass_mode,
            range_set_utility_multiplier=range_set_utility_multiplier,
            range_set_utility_candidate_limit=range_set_utility_candidate_limit,
            range_set_utility_mass_mode=range_set_utility_mass_mode,
            range_boundary_prior_weight=range_boundary_prior_weight,
            range_teacher_distillation_mode=range_teacher_distillation_mode,
            range_teacher_epochs=range_teacher_epochs,
            range_audit_compression_ratios=list(range_audit_compression_ratios or []),
            float32_matmul_precision=float32_matmul_precision,
            allow_tf32=allow_tf32,
            amp_mode=amp_mode,
        ),
        baselines=BaselineConfig(
            final_metrics_mode=final_metrics_mode,
        ),
    )


def derive_seed_bundle(master_seed: int) -> SeedBundle:
    """Derive deterministic sub-seeds from a master seed. See experiments/README.md for details."""
    return SeedBundle(
        split_seed=(master_seed * LCG_MULTIPLIER + 1) & 0xFFFF_FFFF,
        train_query_seed=(master_seed * LCG_MULTIPLIER + 3) & 0xFFFF_FFFF,
        eval_query_seed=(master_seed * LCG_MULTIPLIER + 5) & 0xFFFF_FFFF,
        torch_seed=(master_seed * LCG_MULTIPLIER + 7) & 0xFFFF_FFFF,
    )
