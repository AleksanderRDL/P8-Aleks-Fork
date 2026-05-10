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
    query_chunk_size: int
    compression_ratio: float
    epochs: int
    early_stopping_patience: int
    checkpoint_smoothing_window: int
    mlqds_temporal_fraction: float
    workload: str
    checkpoint_selection_metric: str
    f1_diagnostic_every: int


RANGE_REAL_USECASE_PROFILE = BenchmarkProfile(
    name=DEFAULT_PROFILE,
    n_queries=512,
    query_coverage=0.30,
    range_spatial_fraction=0.0165,
    range_time_fraction=0.033,
    query_chunk_size=512,
    compression_ratio=0.05,
    epochs=20,
    early_stopping_patience=8,
    checkpoint_smoothing_window=1,
    mlqds_temporal_fraction=0.10,
    workload="range",
    checkpoint_selection_metric="f1",
    f1_diagnostic_every=1,
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
        "--query_chunk_size",
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
    ]
    if include_workload:
        args += ["--workload", profile.workload]
    if include_checkpoint_selection:
        args += ["--checkpoint_selection_metric", profile.checkpoint_selection_metric]
    if include_f1_diagnostic:
        args += ["--f1_diagnostic_every", str(profile.f1_diagnostic_every)]
    return args


def benchmark_profile_settings(name: str) -> dict[str, int | float | str]:
    """Return compact profile settings recorded in run_config.json."""
    profile = benchmark_profile(name)
    return {
        "data_mode": "two_cleaned_csv_days",
        "train_day": "first sorted cleaned CSV",
        "eval_day": "second sorted cleaned CSV",
        "n_queries": profile.n_queries,
        "query_coverage": profile.query_coverage,
        "range_spatial_fraction": profile.range_spatial_fraction,
        "range_time_fraction": profile.range_time_fraction,
        "query_chunk_size": profile.query_chunk_size,
        "compression_ratio": profile.compression_ratio,
        "epochs": profile.epochs,
        "early_stopping_patience": profile.early_stopping_patience,
        "checkpoint_selection_metric": profile.checkpoint_selection_metric,
        "checkpoint_f1_variant": "answer",
        "f1_diagnostic_every": profile.f1_diagnostic_every,
        "checkpoint_smoothing_window": profile.checkpoint_smoothing_window,
        "mlqds_temporal_fraction": profile.mlqds_temporal_fraction,
    }
