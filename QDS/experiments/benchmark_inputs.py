"""Benchmark input, profile, and environment resolution helpers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.benchmark_profiles import (
    DEFAULT_PROFILE,
    ProfileSetting,
    benchmark_profile_args,
    benchmark_profile_settings,
)
from experiments.benchmark_runtime import _environment_metadata

PURE_WORKLOADS = ("range", "knn", "similarity", "clustering")
DEFAULT_WORKLOADS = ("range",)
MIN_REALISTIC_CSV_DAYS = 3
DEFAULT_PROFILE_ARGS = benchmark_profile_args(DEFAULT_PROFILE, include_checkpoint_selection=True)


@dataclass(frozen=True)
class BenchmarkDataSources:
    """Resolved CSV inputs for a benchmark run."""

    csv_path: str | None = None
    train_csv_path: str | None = None
    validation_csv_path: str | None = None
    eval_csv_path: str | None = None
    selected_cleaned_csv_files: tuple[str, ...] = ()

    @property
    def csv_sources(self) -> tuple[str, ...]:
        """Return unique CSV sources used by the run."""
        candidates = [self.csv_path, self.train_csv_path, self.validation_csv_path, self.eval_csv_path]
        values: list[str] = []
        for candidate in candidates:
            if candidate and candidate not in values:
                values.append(candidate)
        return tuple(values)

def _parse_name_list(raw: str | None, *, allowed: tuple[str, ...] | set[str], arg_name: str) -> list[str]:
    """Parse a comma-separated list and validate all names."""
    allowed_set = set(allowed)
    values = [item.strip().lower() for item in raw.split(",")] if raw else list(allowed)
    values = [item for item in values if item]
    unknown = [item for item in values if item not in allowed_set]
    if unknown:
        choices = ", ".join(sorted(allowed_set))
        raise ValueError(f"{arg_name} contains unknown value(s) {unknown}; choices: {choices}.")
    if not values:
        raise ValueError(f"{arg_name} must contain at least one value.")
    return values

def _runner_environment_metadata() -> dict[str, Any]:
    """Return parent-process environment metadata with explicit runtime scope."""
    environment = _environment_metadata("off")
    environment["scope"] = "benchmark_runner_parent_process"
    environment["note"] = (
        "Torch precision fields in this block describe the benchmark runner parent process. "
        "Each child run applies the selected benchmark profile plus any --extra_args "
        "overrides; effective child runtime settings are recorded in "
        "rows[*].child_torch_runtime and in each child example_run.json."
    )
    return environment

def _cleaned_csv_files(path: str | Path) -> list[Path]:
    """Return sorted cleaned CSV files for a file or directory input."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"CSV path does not exist: {source}")
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise ValueError(f"CSV path is neither a file nor directory: {source}")
    files = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() == ".csv")
    if not files:
        raise ValueError(f"No cleaned CSV files found in directory: {source}")
    return files

def _assert_distinct_csv_sources(named_paths: dict[str, str | None]) -> None:
    """Reject duplicate train/validation/eval CSV paths to prevent split leakage."""
    seen: dict[Path, str] = {}
    for label, value in named_paths.items():
        if value is None:
            continue
        resolved = Path(value).resolve()
        if resolved in seen:
            raise ValueError(
                f"{label} CSV path must be distinct from {seen[resolved]} CSV path: {value}"
            )
        seen[resolved] = label

def _resolve_data_sources(args: argparse.Namespace) -> BenchmarkDataSources:
    """Resolve benchmark CSV inputs, using three cleaned days for directory inputs."""
    has_explicit_sources = bool(args.train_csv_path or args.validation_csv_path or args.eval_csv_path)
    if has_explicit_sources:
        if not args.train_csv_path or not args.eval_csv_path:
            raise ValueError("--train_csv_path and --eval_csv_path must be supplied together.")
        if args.csv_path:
            raise ValueError("--csv_path cannot be combined with explicit train/validation/eval CSV paths.")
        train_path = str(Path(args.train_csv_path))
        validation_path = str(Path(args.validation_csv_path)) if args.validation_csv_path else None
        eval_path = str(Path(args.eval_csv_path))
        for source in (train_path, validation_path, eval_path):
            if source is None:
                continue
            if not Path(source).is_file():
                raise FileNotFoundError(f"CSV path does not exist or is not a file: {source}")
        _assert_distinct_csv_sources(
            {
                "train": train_path,
                "validation": validation_path,
                "eval": eval_path,
            }
        )
        selected = (train_path, validation_path, eval_path) if validation_path else (train_path, eval_path)
        return BenchmarkDataSources(
            train_csv_path=train_path,
            validation_csv_path=validation_path,
            eval_csv_path=eval_path,
            selected_cleaned_csv_files=tuple(source for source in selected if source is not None),
        )

    if not args.csv_path:
        return BenchmarkDataSources()

    source_path = Path(args.csv_path)
    files = _cleaned_csv_files(args.csv_path)
    if source_path.is_dir():
        if len(files) < MIN_REALISTIC_CSV_DAYS:
            raise ValueError(f"Expected at least {MIN_REALISTIC_CSV_DAYS} cleaned CSV files in {args.csv_path}.")
        selected = tuple(str(path) for path in files[:MIN_REALISTIC_CSV_DAYS])
        _assert_distinct_csv_sources({"train": selected[0], "validation": selected[1], "eval": selected[2]})
        return BenchmarkDataSources(
            train_csv_path=selected[0],
            validation_csv_path=selected[1],
            eval_csv_path=selected[2],
            selected_cleaned_csv_files=selected,
        )
    if len(files) == 1:
        return BenchmarkDataSources(csv_path=str(files[0]), selected_cleaned_csv_files=(str(files[0]),))
    raise ValueError(f"Expected a cleaned CSV file or directory: {args.csv_path}")

def _profile_args(
    profile: str,
    args: argparse.Namespace,
    data_sources: BenchmarkDataSources | None = None,
    *,
    include_refresh_cache: bool = True,
) -> list[str]:
    """Return effective child CLI arguments for a benchmark profile."""
    if profile != DEFAULT_PROFILE:
        raise ValueError(f"Unknown benchmark profile: {profile}")

    data_sources = data_sources or _resolve_data_sources(args)
    if data_sources.train_csv_path and data_sources.eval_csv_path:
        profile_args = [
            "--train_csv_path",
            data_sources.train_csv_path,
        ]
        if data_sources.validation_csv_path:
            profile_args += ["--validation_csv_path", data_sources.validation_csv_path]
        profile_args += [
            "--eval_csv_path",
            data_sources.eval_csv_path,
            *DEFAULT_PROFILE_ARGS,
        ]
    elif data_sources.csv_path:
        profile_args = ["--csv_path", data_sources.csv_path, *DEFAULT_PROFILE_ARGS]
    else:
        raise ValueError(
            f"{DEFAULT_PROFILE} requires --csv_path or --train_csv_path/--eval_csv_path."
        )

    if data_sources.csv_sources:
        profile_args += ["--min_points_per_segment", str(args.min_points_per_segment)]
        profile_args += ["--max_time_gap_seconds", str(args.max_time_gap_seconds)]
        if args.max_points_per_segment is not None:
            profile_args += ["--max_points_per_segment", str(args.max_points_per_segment)]
        if args.max_segments is not None:
            profile_args += ["--max_segments", str(args.max_segments)]
        if args.max_trajectories is not None:
            profile_args += ["--max_trajectories", str(args.max_trajectories)]
        if args.cache_dir is not None:
            profile_args += ["--cache_dir", str(args.cache_dir)]
        if args.refresh_cache and include_refresh_cache:
            profile_args.append("--refresh_cache")
    return profile_args

def _profile_settings(profile: str) -> dict[str, ProfileSetting]:
    """Return compact profile settings recorded in run_config.json."""
    return benchmark_profile_settings(profile)
