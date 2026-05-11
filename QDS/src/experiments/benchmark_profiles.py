"""Shared benchmark profile definitions for AIS-QDS experiment wrappers."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_PROFILE = "range_real_usecase"
PROFILE_CHOICES = (DEFAULT_PROFILE,)


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
    mlqds_temporal_fraction: float
    workload: str
    checkpoint_selection_metric: str
    mlqds_score_mode: str
    mlqds_score_temperature: float
    mlqds_rank_confidence_weight: float
    f1_diagnostic_every: int
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
    mlqds_temporal_fraction=0.25,
    workload="range",
    checkpoint_selection_metric="f1",
    mlqds_score_mode="rank",
    mlqds_score_temperature=1.0,
    mlqds_rank_confidence_weight=0.15,
    f1_diagnostic_every=1,
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
        "--mlqds_temporal_fraction",
        f"{profile.mlqds_temporal_fraction:.2f}",
        "--mlqds_score_mode",
        profile.mlqds_score_mode,
        "--mlqds_score_temperature",
        f"{profile.mlqds_score_temperature:.2f}",
        "--mlqds_rank_confidence_weight",
        f"{profile.mlqds_rank_confidence_weight:.2f}",
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


def benchmark_profile_settings(name: str) -> dict[str, int | float | str | None]:
    """Return compact profile settings recorded in run_config.json."""
    profile = benchmark_profile(name)
    return {
        "data_mode": "two_cleaned_csv_days",
        "train_day": "first sorted cleaned CSV",
        "eval_day": "second sorted cleaned CSV",
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
        "checkpoint_f1_variant": "answer",
        "mlqds_score_mode": profile.mlqds_score_mode,
        "mlqds_score_temperature": profile.mlqds_score_temperature,
        "mlqds_rank_confidence_weight": profile.mlqds_rank_confidence_weight,
        "f1_diagnostic_every": profile.f1_diagnostic_every,
        "checkpoint_smoothing_window": profile.checkpoint_smoothing_window,
        "mlqds_temporal_fraction": profile.mlqds_temporal_fraction,
        "range_boundary_prior_weight": profile.range_boundary_prior_weight,
        "range_boundary_prior_enabled": profile.range_boundary_prior_weight > 0.0,
    }
