"""Shared benchmark profile definitions for AIS-QDS experiment wrappers."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_PROFILE = "range_real_usecase"
PROFILE_CHOICES = (DEFAULT_PROFILE,)
ProfileSetting = int | float | str | bool | list[float] | None


@dataclass(frozen=True)
class BenchmarkProfile:
    """Stable benchmark profile shape shared by matrix and runtime wrappers."""

    name: str
    n_queries: int
    query_coverage: float
    range_spatial_fraction: float
    range_time_fraction: float
    range_spatial_km: float | None
    range_time_hours: float | None
    range_footprint_jitter: float
    query_chunk_size: int
    compression_ratio: float
    epochs: int
    early_stopping_patience: int
    checkpoint_smoothing_window: int
    checkpoint_full_f1_every: int
    checkpoint_candidate_pool_size: int
    mlqds_temporal_fraction: float
    workload: str
    checkpoint_selection_metric: str
    loss_objective: str
    budget_loss_ratios: tuple[float, ...]
    budget_loss_temperature: float
    mlqds_score_mode: str
    mlqds_score_temperature: float
    mlqds_rank_confidence_weight: float
    mlqds_diversity_bonus: float
    residual_label_mode: str
    f1_diagnostic_every: int
    range_label_mode: str
    range_boundary_prior_weight: float


RANGE_REAL_USECASE_PROFILE = BenchmarkProfile(
    name=DEFAULT_PROFILE,
    n_queries=80,
    query_coverage=0.20,
    range_spatial_fraction=0.0165,
    range_time_fraction=0.033,
    range_spatial_km=2.2,
    range_time_hours=5.0,
    range_footprint_jitter=0.0,
    query_chunk_size=2048,
    compression_ratio=0.05,
    epochs=20,
    early_stopping_patience=5,
    checkpoint_smoothing_window=1,
    checkpoint_full_f1_every=1,
    checkpoint_candidate_pool_size=1,
    mlqds_temporal_fraction=0.50,
    workload="range",
    checkpoint_selection_metric="f1",
    loss_objective="budget_topk",
    budget_loss_ratios=(0.01, 0.02, 0.05, 0.10),
    budget_loss_temperature=0.10,
    mlqds_score_mode="rank",
    mlqds_score_temperature=1.0,
    mlqds_rank_confidence_weight=0.15,
    mlqds_diversity_bonus=0.0,
    residual_label_mode="temporal",
    f1_diagnostic_every=1,
    range_label_mode="usefulness",
    range_boundary_prior_weight=0.0,
)

_PROFILES = {RANGE_REAL_USECASE_PROFILE.name: RANGE_REAL_USECASE_PROFILE}


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
    include_f1_diagnostic: bool = False,
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
        "--query_chunk_size",
        str(profile.query_chunk_size),
        "--max_queries",
        str(profile.query_chunk_size),
        "--compression_ratio",
        str(profile.compression_ratio),
        "--epochs",
        str(profile.epochs),
        "--early_stopping_patience",
        str(profile.early_stopping_patience),
        "--checkpoint_smoothing_window",
        str(profile.checkpoint_smoothing_window),
        "--checkpoint_full_f1_every",
        str(profile.checkpoint_full_f1_every),
        "--checkpoint_candidate_pool_size",
        str(profile.checkpoint_candidate_pool_size),
        "--loss_objective",
        profile.loss_objective,
        "--budget_loss_ratios",
        ",".join(f"{ratio:.2f}" for ratio in profile.budget_loss_ratios),
        "--budget_loss_temperature",
        f"{profile.budget_loss_temperature:.2f}",
        "--mlqds_temporal_fraction",
        f"{profile.mlqds_temporal_fraction:.2f}",
        "--mlqds_score_mode",
        profile.mlqds_score_mode,
        "--mlqds_score_temperature",
        f"{profile.mlqds_score_temperature:.2f}",
        "--mlqds_rank_confidence_weight",
        f"{profile.mlqds_rank_confidence_weight:.2f}",
        "--mlqds_diversity_bonus",
        f"{profile.mlqds_diversity_bonus:.2f}",
        "--residual_label_mode",
        profile.residual_label_mode,
        "--range_label_mode",
        profile.range_label_mode,
        "--range_boundary_prior_weight",
        f"{profile.range_boundary_prior_weight:.1f}",
    ]
    if include_workload:
        args += ["--workload", profile.workload]
    if include_checkpoint_selection:
        args += ["--checkpoint_selection_metric", profile.checkpoint_selection_metric]
    if include_f1_diagnostic:
        args += ["--f1_diagnostic_every", str(profile.f1_diagnostic_every)]
    return args


def benchmark_profile_settings(name: str) -> dict[str, ProfileSetting]:
    """Return compact profile settings recorded in run_config.json."""
    profile = benchmark_profile(name)
    return {
        "data_mode": "three_cleaned_csv_days",
        "train_day": "first sorted cleaned CSV",
        "validation_day": "second sorted cleaned CSV",
        "eval_day": "third sorted cleaned CSV",
        "n_queries": profile.n_queries,
        "max_queries": profile.query_chunk_size,
        "query_coverage": profile.query_coverage,
        "range_spatial_fraction": profile.range_spatial_fraction,
        "range_time_fraction": profile.range_time_fraction,
        "range_spatial_km": profile.range_spatial_km,
        "range_time_hours": profile.range_time_hours,
        "range_footprint_jitter": profile.range_footprint_jitter,
        "query_chunk_size": profile.query_chunk_size,
        "compression_ratio": profile.compression_ratio,
        "epochs": profile.epochs,
        "early_stopping_patience": profile.early_stopping_patience,
        "checkpoint_selection_metric": profile.checkpoint_selection_metric,
        "checkpoint_f1_variant": "range_usefulness",
        "checkpoint_full_f1_every": profile.checkpoint_full_f1_every,
        "checkpoint_candidate_pool_size": profile.checkpoint_candidate_pool_size,
        "loss_objective": profile.loss_objective,
        "budget_loss_ratios": list(profile.budget_loss_ratios),
        "budget_loss_temperature": profile.budget_loss_temperature,
        "mlqds_score_mode": profile.mlqds_score_mode,
        "mlqds_score_temperature": profile.mlqds_score_temperature,
        "mlqds_rank_confidence_weight": profile.mlqds_rank_confidence_weight,
        "mlqds_diversity_bonus": profile.mlqds_diversity_bonus,
        "residual_label_mode": profile.residual_label_mode,
        "f1_diagnostic_every": profile.f1_diagnostic_every,
        "range_label_mode": profile.range_label_mode,
        "checkpoint_smoothing_window": profile.checkpoint_smoothing_window,
        "mlqds_temporal_fraction": profile.mlqds_temporal_fraction,
        "range_boundary_prior_weight": profile.range_boundary_prior_weight,
        "range_boundary_prior_enabled": profile.range_boundary_prior_weight > 0.0,
    }
