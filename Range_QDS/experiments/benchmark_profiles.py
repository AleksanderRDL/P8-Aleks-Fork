"""Shared benchmark profile definitions for AIS-QDS experiment wrappers."""

from __future__ import annotations

from dataclasses import dataclass

from training.model_features import is_workload_blind_model_type


DEFAULT_PROFILE = "range_workload_aware_diagnostic"
BLIND_EXPECTED_USEFULNESS_PROFILE = "range_workload_blind_expected_usefulness"
BLIND_RETAINED_FREQUENCY_PROFILE = "range_workload_blind_retained_frequency"
BLIND_TEACHER_DISTILL_PROFILE = "range_workload_blind_teacher_distill"
LEGACY_DIAGNOSTIC_PROFILE_NOTE = (
    "Old RangeUseful/scalar-target diagnostic path. "
    "Not valid for query-driven rework acceptance."
)
RANGE_WORKLOAD_V1_WORKLOAD_BLIND_V2_PROFILE = "range_workload_v1_workload_blind_v2"
PROFILE_CHOICES = (
    DEFAULT_PROFILE,
    BLIND_EXPECTED_USEFULNESS_PROFILE,
    BLIND_RETAINED_FREQUENCY_PROFILE,
    BLIND_TEACHER_DISTILL_PROFILE,
    RANGE_WORKLOAD_V1_WORKLOAD_BLIND_V2_PROFILE,
)
ProfileSetting = int | float | str | bool | list[float] | list[str] | None
RANGE_COVERAGE_SWEEP_TARGETS = (0.05, 0.10, 0.15, 0.30)
RANGE_COMPRESSION_SWEEP_RATIOS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30)
RANGE_BLIND_COVERAGE_MIN_QUERY_FLOOR = 8


def effective_mlqds_diversity_bonus(mlqds_hybrid_mode: str, mlqds_diversity_bonus: float) -> float:
    """Return the diversity bonus consumed by the selector for this hybrid mode."""
    if str(mlqds_hybrid_mode).lower() in {"stratified", "global_budget"}:
        return 0.0
    return float(mlqds_diversity_bonus)


@dataclass(frozen=True)
class BenchmarkProfile:
    """Stable benchmark profile shape shared by benchmark entry points."""

    name: str
    n_queries: int
    query_coverage: float
    range_spatial_fraction: float
    range_time_fraction: float
    range_spatial_km: float | None
    range_time_hours: float | None
    range_footprint_jitter: float
    range_max_coverage_overshoot: float | None
    range_time_domain_mode: str
    range_anchor_mode: str
    range_train_anchor_modes: tuple[str, ...]
    range_diagnostics_mode: str
    final_metrics_mode: str
    max_queries: int
    query_chunk_size: int
    train_batch_size: int
    inference_batch_size: int
    model_type: str
    compression_ratio: float
    epochs: int
    early_stopping_patience: int
    checkpoint_smoothing_window: int
    checkpoint_full_score_every: int
    checkpoint_candidate_pool_size: int
    mlqds_temporal_fraction: float
    workload: str
    checkpoint_selection_metric: str
    checkpoint_score_variant: str
    float32_matmul_precision: str
    allow_tf32: bool
    amp_mode: str
    loss_objective: str
    budget_loss_ratios: tuple[float, ...]
    budget_loss_temperature: float
    temporal_distribution_loss_weight: float
    mlqds_score_mode: str
    mlqds_score_temperature: float
    mlqds_rank_confidence_weight: float
    mlqds_range_geometry_blend: float
    mlqds_diversity_bonus: float
    mlqds_hybrid_mode: str
    temporal_residual_label_mode: str
    validation_score_every: int
    range_label_mode: str
    range_training_target_mode: str
    range_temporal_target_blend: float
    range_target_budget_weight_power: float
    range_marginal_target_radius_scale: float
    range_query_spine_fraction: float
    range_query_spine_mass_mode: str
    range_query_residual_multiplier: float
    range_query_residual_mass_mode: str
    range_set_utility_multiplier: float
    range_set_utility_candidate_limit: int
    range_set_utility_mass_mode: str
    range_boundary_prior_weight: float
    range_teacher_distillation_mode: str
    range_teacher_epochs: int
    final_success_allowed: bool = False
    profile_note: str = LEGACY_DIAGNOSTIC_PROFILE_NOTE
    mlqds_stratified_center_weight: float = 0.0
    workload_profile_id: str | None = None
    selector_type: str = "temporal_hybrid"
    range_train_workload_replicates: int = 1


RANGE_WORKLOAD_AWARE_DIAGNOSTIC_PROFILE = BenchmarkProfile(
    name=DEFAULT_PROFILE,
    n_queries=80,
    query_coverage=0.20,
    range_spatial_fraction=0.0165,
    range_time_fraction=0.033,
    range_spatial_km=2.2,
    range_time_hours=5.0,
    range_footprint_jitter=0.0,
    range_max_coverage_overshoot=0.02,
    range_time_domain_mode="anchor_day",
    range_anchor_mode="mixed_density",
    range_train_anchor_modes=(),
    range_diagnostics_mode="cached",
    final_metrics_mode="diagnostic",
    max_queries=2048,
    query_chunk_size=2048,
    train_batch_size=64,
    inference_batch_size=64,
    model_type="range_aware",
    compression_ratio=0.05,
    epochs=8,
    early_stopping_patience=5,
    checkpoint_smoothing_window=1,
    checkpoint_full_score_every=4,
    checkpoint_candidate_pool_size=2,
    mlqds_temporal_fraction=0.25,
    workload="range",
    checkpoint_selection_metric="uniform_gap",
    checkpoint_score_variant="range_usefulness",
    float32_matmul_precision="high",
    allow_tf32=True,
    amp_mode="bf16",
    loss_objective="budget_topk",
    budget_loss_ratios=(0.05, 0.10),
    budget_loss_temperature=0.25,
    temporal_distribution_loss_weight=0.0,
    mlqds_score_mode="rank",
    mlqds_score_temperature=1.0,
    mlqds_rank_confidence_weight=0.15,
    mlqds_range_geometry_blend=0.0,
    mlqds_diversity_bonus=0.0,
    mlqds_hybrid_mode="fill",
    temporal_residual_label_mode="none",
    validation_score_every=1,
    range_label_mode="usefulness",
    range_training_target_mode="point_value",
    range_temporal_target_blend=0.0,
    range_target_budget_weight_power=0.0,
    range_marginal_target_radius_scale=0.50,
    range_query_spine_fraction=0.10,
    range_query_spine_mass_mode="hit_group",
    range_query_residual_multiplier=1.0,
    range_query_residual_mass_mode="query",
    range_set_utility_multiplier=1.0,
    range_set_utility_candidate_limit=128,
    range_set_utility_mass_mode="gain",
    range_boundary_prior_weight=0.0,
    range_teacher_distillation_mode="none",
    range_teacher_epochs=4,
)

RANGE_WORKLOAD_BLIND_EXPECTED_USEFULNESS_PROFILE = BenchmarkProfile(
    name=BLIND_EXPECTED_USEFULNESS_PROFILE,
    n_queries=RANGE_BLIND_COVERAGE_MIN_QUERY_FLOOR,
    query_coverage=0.20,
    range_spatial_fraction=0.0165,
    range_time_fraction=0.033,
    range_spatial_km=2.2,
    range_time_hours=5.0,
    range_footprint_jitter=0.0,
    range_max_coverage_overshoot=0.02,
    range_time_domain_mode="anchor_day",
    range_anchor_mode="mixed_density",
    range_train_anchor_modes=(),
    range_diagnostics_mode="cached",
    final_metrics_mode="diagnostic",
    max_queries=2048,
    query_chunk_size=2048,
    train_batch_size=64,
    inference_batch_size=64,
    model_type="workload_blind_range",
    compression_ratio=0.05,
    epochs=10,
    early_stopping_patience=5,
    checkpoint_smoothing_window=1,
    checkpoint_full_score_every=2,
    checkpoint_candidate_pool_size=2,
    mlqds_temporal_fraction=0.10,
    workload="range",
    checkpoint_selection_metric="uniform_gap",
    checkpoint_score_variant="range_usefulness",
    float32_matmul_precision="high",
    allow_tf32=True,
    amp_mode="bf16",
    loss_objective="budget_topk",
    budget_loss_ratios=RANGE_COMPRESSION_SWEEP_RATIOS,
    budget_loss_temperature=0.25,
    temporal_distribution_loss_weight=0.0,
    mlqds_score_mode="rank",
    mlqds_score_temperature=1.0,
    mlqds_rank_confidence_weight=0.15,
    mlqds_range_geometry_blend=0.0,
    mlqds_diversity_bonus=0.0,
    mlqds_hybrid_mode="fill",
    temporal_residual_label_mode="none",
    validation_score_every=1,
    range_label_mode="usefulness",
    range_training_target_mode="point_value",
    range_temporal_target_blend=0.0,
    range_target_budget_weight_power=0.0,
    range_marginal_target_radius_scale=0.50,
    range_query_spine_fraction=0.10,
    range_query_spine_mass_mode="hit_group",
    range_query_residual_multiplier=1.0,
    range_query_residual_mass_mode="query",
    range_set_utility_multiplier=1.0,
    range_set_utility_candidate_limit=128,
    range_set_utility_mass_mode="gain",
    range_boundary_prior_weight=0.0,
    range_teacher_distillation_mode="none",
    range_teacher_epochs=4,
)

RANGE_WORKLOAD_BLIND_RETAINED_FREQUENCY_PROFILE = BenchmarkProfile(
    name=BLIND_RETAINED_FREQUENCY_PROFILE,
    n_queries=RANGE_BLIND_COVERAGE_MIN_QUERY_FLOOR,
    query_coverage=0.20,
    range_spatial_fraction=0.0165,
    range_time_fraction=0.033,
    range_spatial_km=2.2,
    range_time_hours=5.0,
    range_footprint_jitter=0.0,
    range_max_coverage_overshoot=0.02,
    range_time_domain_mode="anchor_day",
    range_anchor_mode="mixed_density",
    range_train_anchor_modes=(),
    range_diagnostics_mode="cached",
    final_metrics_mode="diagnostic",
    max_queries=2048,
    query_chunk_size=2048,
    train_batch_size=64,
    inference_batch_size=64,
    model_type="workload_blind_range",
    compression_ratio=0.05,
    epochs=10,
    early_stopping_patience=5,
    checkpoint_smoothing_window=1,
    checkpoint_full_score_every=2,
    checkpoint_candidate_pool_size=2,
    mlqds_temporal_fraction=0.30,
    workload="range",
    checkpoint_selection_metric="uniform_gap",
    checkpoint_score_variant="range_usefulness",
    float32_matmul_precision="high",
    allow_tf32=True,
    amp_mode="bf16",
    loss_objective="budget_topk",
    budget_loss_ratios=RANGE_COMPRESSION_SWEEP_RATIOS,
    budget_loss_temperature=0.25,
    temporal_distribution_loss_weight=0.0,
    mlqds_score_mode="rank",
    mlqds_score_temperature=1.0,
    mlqds_rank_confidence_weight=0.15,
    mlqds_range_geometry_blend=0.0,
    mlqds_diversity_bonus=0.0,
    mlqds_hybrid_mode="fill",
    temporal_residual_label_mode="none",
    validation_score_every=1,
    range_label_mode="usefulness",
    range_training_target_mode="retained_frequency",
    range_temporal_target_blend=0.0,
    range_target_budget_weight_power=0.0,
    range_marginal_target_radius_scale=0.50,
    range_query_spine_fraction=0.10,
    range_query_spine_mass_mode="hit_group",
    range_query_residual_multiplier=1.0,
    range_query_residual_mass_mode="query",
    range_set_utility_multiplier=1.0,
    range_set_utility_candidate_limit=128,
    range_set_utility_mass_mode="gain",
    range_boundary_prior_weight=0.0,
    range_teacher_distillation_mode="none",
    range_teacher_epochs=4,
)

RANGE_WORKLOAD_BLIND_TEACHER_DISTILL_PROFILE = BenchmarkProfile(
    name=BLIND_TEACHER_DISTILL_PROFILE,
    n_queries=RANGE_BLIND_COVERAGE_MIN_QUERY_FLOOR,
    query_coverage=0.20,
    range_spatial_fraction=0.0165,
    range_time_fraction=0.033,
    range_spatial_km=2.2,
    range_time_hours=5.0,
    range_footprint_jitter=0.0,
    range_max_coverage_overshoot=0.02,
    range_time_domain_mode="anchor_day",
    range_anchor_mode="mixed_density",
    range_train_anchor_modes=(),
    range_diagnostics_mode="cached",
    final_metrics_mode="diagnostic",
    max_queries=2048,
    query_chunk_size=2048,
    train_batch_size=64,
    inference_batch_size=64,
    model_type="workload_blind_range",
    compression_ratio=0.05,
    epochs=10,
    early_stopping_patience=5,
    checkpoint_smoothing_window=1,
    checkpoint_full_score_every=2,
    checkpoint_candidate_pool_size=2,
    mlqds_temporal_fraction=0.25,
    workload="range",
    checkpoint_selection_metric="uniform_gap",
    checkpoint_score_variant="range_usefulness",
    float32_matmul_precision="high",
    allow_tf32=True,
    amp_mode="bf16",
    loss_objective="budget_topk",
    budget_loss_ratios=RANGE_COMPRESSION_SWEEP_RATIOS,
    budget_loss_temperature=0.25,
    temporal_distribution_loss_weight=0.0,
    mlqds_score_mode="rank",
    mlqds_score_temperature=1.0,
    mlqds_rank_confidence_weight=0.15,
    mlqds_range_geometry_blend=0.0,
    mlqds_diversity_bonus=0.0,
    mlqds_hybrid_mode="fill",
    temporal_residual_label_mode="temporal",
    validation_score_every=1,
    range_label_mode="usefulness",
    range_training_target_mode="point_value",
    range_temporal_target_blend=0.0,
    range_target_budget_weight_power=0.0,
    range_marginal_target_radius_scale=0.50,
    range_query_spine_fraction=0.10,
    range_query_spine_mass_mode="hit_group",
    range_query_residual_multiplier=1.0,
    range_query_residual_mass_mode="query",
    range_set_utility_multiplier=1.0,
    range_set_utility_candidate_limit=128,
    range_set_utility_mass_mode="gain",
    range_boundary_prior_weight=0.0,
    range_teacher_distillation_mode="retained_frequency",
    range_teacher_epochs=4,
)

RANGE_WORKLOAD_V1_WORKLOAD_BLIND_V2_BENCHMARK_PROFILE = BenchmarkProfile(
    name=RANGE_WORKLOAD_V1_WORKLOAD_BLIND_V2_PROFILE,
    n_queries=RANGE_BLIND_COVERAGE_MIN_QUERY_FLOOR,
    query_coverage=0.10,
    range_spatial_fraction=0.0165,
    range_time_fraction=0.033,
    range_spatial_km=None,
    range_time_hours=None,
    range_footprint_jitter=0.20,
    range_max_coverage_overshoot=0.0075,
    range_time_domain_mode="anchor_day",
    range_anchor_mode="mixed_density",
    range_train_anchor_modes=(),
    range_diagnostics_mode="cached",
    final_metrics_mode="diagnostic",
    max_queries=2048,
    query_chunk_size=2048,
    train_batch_size=64,
    inference_batch_size=64,
    model_type="workload_blind_range_v2",
    compression_ratio=0.05,
    epochs=10,
    early_stopping_patience=5,
    checkpoint_smoothing_window=1,
    checkpoint_full_score_every=2,
    checkpoint_candidate_pool_size=2,
    mlqds_temporal_fraction=0.0,
    workload="range",
    checkpoint_selection_metric="uniform_gap",
    checkpoint_score_variant="query_useful_v1",
    float32_matmul_precision="high",
    allow_tf32=True,
    amp_mode="bf16",
    loss_objective="budget_topk",
    budget_loss_ratios=RANGE_COMPRESSION_SWEEP_RATIOS,
    budget_loss_temperature=0.25,
    temporal_distribution_loss_weight=0.0,
    mlqds_score_mode="rank_confidence",
    mlqds_score_temperature=1.0,
    mlqds_rank_confidence_weight=0.20,
    mlqds_range_geometry_blend=0.0,
    mlqds_diversity_bonus=0.0,
    mlqds_hybrid_mode="global_budget",
    temporal_residual_label_mode="none",
    validation_score_every=1,
    range_label_mode="usefulness",
    range_training_target_mode="query_useful_v1_factorized",
    range_temporal_target_blend=0.0,
    range_target_budget_weight_power=0.0,
    range_marginal_target_radius_scale=0.50,
    range_query_spine_fraction=0.10,
    range_query_spine_mass_mode="hit_group",
    range_query_residual_multiplier=1.0,
    range_query_residual_mass_mode="query",
    range_set_utility_multiplier=1.0,
    range_set_utility_candidate_limit=128,
    range_set_utility_mass_mode="gain",
    range_boundary_prior_weight=0.0,
    range_teacher_distillation_mode="none",
    range_teacher_epochs=4,
    final_success_allowed=True,
    profile_note="QueryUsefulV1/range_workload_v1 final-candidate profile.",
    workload_profile_id="range_workload_v1",
    selector_type="learned_segment_budget_v1",
    range_train_workload_replicates=4,
)

_PROFILES = {
    RANGE_WORKLOAD_AWARE_DIAGNOSTIC_PROFILE.name: RANGE_WORKLOAD_AWARE_DIAGNOSTIC_PROFILE,
    RANGE_WORKLOAD_BLIND_EXPECTED_USEFULNESS_PROFILE.name: RANGE_WORKLOAD_BLIND_EXPECTED_USEFULNESS_PROFILE,
    RANGE_WORKLOAD_BLIND_RETAINED_FREQUENCY_PROFILE.name: RANGE_WORKLOAD_BLIND_RETAINED_FREQUENCY_PROFILE,
    RANGE_WORKLOAD_BLIND_TEACHER_DISTILL_PROFILE.name: RANGE_WORKLOAD_BLIND_TEACHER_DISTILL_PROFILE,
    RANGE_WORKLOAD_V1_WORKLOAD_BLIND_V2_BENCHMARK_PROFILE.name: RANGE_WORKLOAD_V1_WORKLOAD_BLIND_V2_BENCHMARK_PROFILE,
}


def benchmark_profile(name: str) -> BenchmarkProfile:
    """Return a known benchmark profile by name."""
    try:
        return _PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown benchmark profile: {name}") from exc


def benchmark_profile_args(
    name: str,
    *,
    include_workload: bool = False,
    include_checkpoint_selection: bool = False,
    include_validation_score_diagnostic: bool = False,
) -> list[str]:
    """Return shared child CLI args for a benchmark profile."""
    profile = benchmark_profile(name)
    args = [
        "--n_queries",
        str(profile.n_queries),
        "--query_coverage",
        f"{profile.query_coverage:.2f}",
        "--range_spatial_fraction",
        str(profile.range_spatial_fraction),
        "--range_time_fraction",
        str(profile.range_time_fraction),
    ]
    if profile.range_spatial_km is not None:
        args += ["--range_spatial_km", str(profile.range_spatial_km)]
    if profile.range_time_hours is not None:
        args += ["--range_time_hours", str(profile.range_time_hours)]
    args += [
        "--range_footprint_jitter",
        str(profile.range_footprint_jitter),
    ]
    if profile.range_max_coverage_overshoot is not None:
        args += ["--range_max_coverage_overshoot", str(profile.range_max_coverage_overshoot)]
    if profile.workload_profile_id is not None:
        args += ["--workload_profile_id", profile.workload_profile_id]
    if int(profile.range_train_workload_replicates) > 1:
        args += ["--range_train_workload_replicates", str(profile.range_train_workload_replicates)]
    args += [
        "--range_time_domain_mode",
        profile.range_time_domain_mode,
        "--range_anchor_mode",
        profile.range_anchor_mode,
        *(
            ["--range_train_anchor_modes", ",".join(profile.range_train_anchor_modes)]
            if profile.range_train_anchor_modes
            else []
        ),
        "--range_diagnostics_mode",
        profile.range_diagnostics_mode,
        "--final_metrics_mode",
        profile.final_metrics_mode,
        "--float32_matmul_precision",
        profile.float32_matmul_precision,
        "--allow_tf32" if profile.allow_tf32 else "--no-allow_tf32",
        "--amp_mode",
        profile.amp_mode,
        "--query_chunk_size",
        str(profile.query_chunk_size),
        "--train_batch_size",
        str(profile.train_batch_size),
        "--inference_batch_size",
        str(profile.inference_batch_size),
        "--model_type",
        profile.model_type,
        "--max_queries",
        str(profile.max_queries),
        "--compression_ratio",
        str(profile.compression_ratio),
        "--epochs",
        str(profile.epochs),
        "--early_stopping_patience",
        str(profile.early_stopping_patience),
        "--checkpoint_smoothing_window",
        str(profile.checkpoint_smoothing_window),
        "--checkpoint_full_score_every",
        str(profile.checkpoint_full_score_every),
        "--checkpoint_candidate_pool_size",
        str(profile.checkpoint_candidate_pool_size),
        "--loss_objective",
        profile.loss_objective,
        "--budget_loss_ratios",
        ",".join(f"{ratio:.2f}" for ratio in profile.budget_loss_ratios),
        "--range_audit_compression_ratios",
        ",".join(f"{ratio:.2f}" for ratio in RANGE_COMPRESSION_SWEEP_RATIOS),
        "--budget_loss_temperature",
        f"{profile.budget_loss_temperature:.2f}",
        "--temporal_distribution_loss_weight",
        f"{profile.temporal_distribution_loss_weight:.3f}",
        "--mlqds_temporal_fraction",
        f"{profile.mlqds_temporal_fraction:.2f}",
        "--mlqds_score_mode",
        profile.mlqds_score_mode,
        "--mlqds_score_temperature",
        f"{profile.mlqds_score_temperature:.2f}",
        "--mlqds_rank_confidence_weight",
        f"{profile.mlqds_rank_confidence_weight:.2f}",
        "--mlqds_range_geometry_blend",
        f"{profile.mlqds_range_geometry_blend:.2f}",
        "--mlqds_diversity_bonus",
        f"{profile.mlqds_diversity_bonus:.2f}",
        "--mlqds_hybrid_mode",
        profile.mlqds_hybrid_mode,
        "--selector_type",
        profile.selector_type,
        "--mlqds_stratified_center_weight",
        f"{profile.mlqds_stratified_center_weight:.2f}",
        "--temporal_residual_label_mode",
        profile.temporal_residual_label_mode,
        "--range_label_mode",
        profile.range_label_mode,
        "--range_training_target_mode",
        profile.range_training_target_mode,
        "--range_temporal_target_blend",
        f"{profile.range_temporal_target_blend:.3f}",
        "--range_target_budget_weight_power",
        f"{profile.range_target_budget_weight_power:.2f}",
        "--range_marginal_target_radius_scale",
        f"{profile.range_marginal_target_radius_scale:.2f}",
        "--range_query_spine_fraction",
        f"{profile.range_query_spine_fraction:.2f}",
        "--range_query_spine_mass_mode",
        profile.range_query_spine_mass_mode,
        "--range_query_residual_multiplier",
        f"{profile.range_query_residual_multiplier:.2f}",
        "--range_query_residual_mass_mode",
        profile.range_query_residual_mass_mode,
        "--range_set_utility_multiplier",
        f"{profile.range_set_utility_multiplier:.2f}",
        "--range_set_utility_candidate_limit",
        str(profile.range_set_utility_candidate_limit),
        "--range_set_utility_mass_mode",
        profile.range_set_utility_mass_mode,
        "--range_boundary_prior_weight",
        f"{profile.range_boundary_prior_weight:.1f}",
        "--range_teacher_distillation_mode",
        profile.range_teacher_distillation_mode,
        "--range_teacher_epochs",
        str(profile.range_teacher_epochs),
    ]
    if include_workload:
        args += ["--workload", profile.workload]
    if include_checkpoint_selection:
        args += [
            "--checkpoint_selection_metric",
            profile.checkpoint_selection_metric,
            "--checkpoint_score_variant",
            profile.checkpoint_score_variant,
        ]
    if include_validation_score_diagnostic:
        args += ["--validation_score_every", str(profile.validation_score_every)]
    return args


def benchmark_profile_settings(name: str) -> dict[str, ProfileSetting]:
    """Return compact profile settings recorded in run_config.json."""
    profile = benchmark_profile(name)
    workload_blind = is_workload_blind_model_type(profile.model_type)
    profile_role = (
        "query_driven_workload_blind_v2"
        if profile.range_training_target_mode == "query_useful_v1_factorized"
        else
        "workload_blind_teacher_distill"
        if profile.range_teacher_distillation_mode != "none"
        else "workload_blind_marginal_coverage"
        if workload_blind and profile.range_training_target_mode == "marginal_coverage_frequency"
        else "workload_blind_query_spine"
        if workload_blind and profile.range_training_target_mode == "query_spine_frequency"
        else "workload_blind_retained_frequency"
        if workload_blind and profile.range_training_target_mode == "retained_frequency"
        else "workload_blind_expected_usefulness"
        if workload_blind
        else "workload_aware_diagnostic"
    )
    final_candidate = bool(profile.final_success_allowed)
    return {
        "profile_role": profile_role,
        "profile_diagnostic_only": not final_candidate,
        "profile_note": profile.profile_note,
        "primary_metric_family": "QueryUsefulV1" if final_candidate else "RangeUsefulLegacy",
        "final_success_allowed": bool(profile.final_success_allowed),
        "final_product_candidate": final_candidate,
        "final_product_claim": False,
        "final_product_claim_gate": (
            "Requires full held-out coverage/compression grid and learning-causality ablations."
            if final_candidate
            else (
                "Diagnostic-only profile. Use the query-driven workload-blind v2 "
                "QueryUsefulV1 profile and required evidence levels for final claims."
            )
        ),
        "workload_blind": bool(workload_blind),
        "data_mode": "three_cleaned_csv_days",
        "train_day": "first sorted cleaned CSV",
        "validation_day": "second sorted cleaned CSV",
        "eval_day": "third sorted cleaned CSV",
        "n_queries": profile.n_queries,
        "max_queries": profile.max_queries,
        "query_coverage": profile.query_coverage,
        "range_spatial_fraction": profile.range_spatial_fraction,
        "range_time_fraction": profile.range_time_fraction,
        "range_spatial_km": profile.range_spatial_km,
        "range_time_hours": profile.range_time_hours,
        "range_footprint_jitter": profile.range_footprint_jitter,
        "range_max_coverage_overshoot": profile.range_max_coverage_overshoot,
        "range_time_domain_mode": profile.range_time_domain_mode,
        "range_anchor_mode": profile.range_anchor_mode,
        "range_train_anchor_modes": list(profile.range_train_anchor_modes),
        "range_train_workload_replicates": int(profile.range_train_workload_replicates),
        "workload_profile_id": profile.workload_profile_id,
        "range_diagnostics_mode": profile.range_diagnostics_mode,
        "final_metrics_mode": profile.final_metrics_mode,
        "query_chunk_size": profile.query_chunk_size,
        "train_batch_size": profile.train_batch_size,
        "inference_batch_size": profile.inference_batch_size,
        "model_type": profile.model_type,
        "compression_ratio": profile.compression_ratio,
        "epochs": profile.epochs,
        "early_stopping_patience": profile.early_stopping_patience,
        "checkpoint_selection_metric": profile.checkpoint_selection_metric,
        "checkpoint_score_variant": profile.checkpoint_score_variant,
        "float32_matmul_precision": profile.float32_matmul_precision,
        "allow_tf32": profile.allow_tf32,
        "amp_mode": profile.amp_mode,
        "checkpoint_full_score_every": profile.checkpoint_full_score_every,
        "checkpoint_candidate_pool_size": profile.checkpoint_candidate_pool_size,
        "loss_objective": profile.loss_objective,
        "budget_loss_ratios": list(profile.budget_loss_ratios),
        "budget_loss_temperature": profile.budget_loss_temperature,
        "temporal_distribution_loss_weight": profile.temporal_distribution_loss_weight,
        "mlqds_score_mode": profile.mlqds_score_mode,
        "mlqds_score_temperature": profile.mlqds_score_temperature,
        "mlqds_rank_confidence_weight": profile.mlqds_rank_confidence_weight,
        "mlqds_range_geometry_blend": profile.mlqds_range_geometry_blend,
        "mlqds_diversity_bonus": profile.mlqds_diversity_bonus,
        "mlqds_effective_diversity_bonus": effective_mlqds_diversity_bonus(
            profile.mlqds_hybrid_mode,
            profile.mlqds_diversity_bonus,
        ),
        "mlqds_hybrid_mode": profile.mlqds_hybrid_mode,
        "selector_type": profile.selector_type,
        "mlqds_stratified_center_weight": profile.mlqds_stratified_center_weight,
        "temporal_residual_label_mode": profile.temporal_residual_label_mode,
        "validation_score_every": profile.validation_score_every,
        "range_label_mode": profile.range_label_mode,
        "range_training_target_mode": profile.range_training_target_mode,
        "range_temporal_target_blend": profile.range_temporal_target_blend,
        "range_target_budget_weight_power": profile.range_target_budget_weight_power,
        "range_marginal_target_radius_scale": profile.range_marginal_target_radius_scale,
        "range_query_spine_fraction": profile.range_query_spine_fraction,
        "range_query_residual_multiplier": profile.range_query_residual_multiplier,
        "range_query_residual_mass_mode": profile.range_query_residual_mass_mode,
        "range_set_utility_multiplier": profile.range_set_utility_multiplier,
        "range_set_utility_candidate_limit": profile.range_set_utility_candidate_limit,
        "range_set_utility_mass_mode": profile.range_set_utility_mass_mode,
        "checkpoint_smoothing_window": profile.checkpoint_smoothing_window,
        "mlqds_temporal_fraction": profile.mlqds_temporal_fraction,
        "range_boundary_prior_weight": profile.range_boundary_prior_weight,
        "range_boundary_prior_enabled": profile.range_boundary_prior_weight > 0.0,
        "range_teacher_distillation_mode": profile.range_teacher_distillation_mode,
        "range_teacher_epochs": profile.range_teacher_epochs,
        "range_coverage_sweep_targets": list(RANGE_COVERAGE_SWEEP_TARGETS),
        "range_compression_sweep_ratios": list(RANGE_COMPRESSION_SWEEP_RATIOS),
    }
