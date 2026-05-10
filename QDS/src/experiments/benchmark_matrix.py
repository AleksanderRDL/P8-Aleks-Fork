"""Run range-focused benchmark matrices for AIS-QDS configuration tuning."""

from __future__ import annotations

import argparse
import csv
import json
import signal
import shlex
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data.trajectory_cache import load_or_build_ais_cache
from src.experiments.benchmark_runtime import (
    EPOCH_RE,
    INFERENCE_STEP_RE,
    PHASE_DONE_RE,
    _environment_metadata,
    _git_metadata,
    _qds_root,
    _split_extra_args,
    _write_text,
)

PURE_WORKLOADS = ("range", "knn", "similarity", "clustering")
DEFAULT_WORKLOADS = ("range",)
MIN_REALISTIC_CSV_DAYS = 2
DEFAULT_CHILD_STDOUT_TAIL_CHARS = 1_000_000
DEFAULT_PROFILE = "range_real_usecase"
PROFILE_CHOICES = (DEFAULT_PROFILE,)
REAL_USECASE_RANGE_SPATIAL_FRACTION = 0.018
REAL_USECASE_RANGE_TIME_FRACTION = 0.036
REAL_USECASE_PROFILE_ARGS = [
    "--n_queries",
    "400",
    "--query_coverage",
    "0.30",
    "--range_spatial_fraction",
    f"{REAL_USECASE_RANGE_SPATIAL_FRACTION:.3f}",
    "--range_time_fraction",
    f"{REAL_USECASE_RANGE_TIME_FRACTION:.3f}",
    "--compression_ratio",
    "0.05",
    "--epochs",
    "20",
    "--early_stopping_patience",
    "8",
    "--checkpoint_smoothing_window",
    "1",
    "--mlqds_temporal_fraction",
    "0.10",
]


@dataclass(frozen=True)
class MatrixVariant:
    """Runtime/config variant for one matrix run."""

    name: str
    float32_matmul_precision: str = "highest"
    allow_tf32: bool = False
    amp_mode: str = "off"
    train_batch_size: int = 16
    inference_batch_size: int = 16
    checkpoint_f1_variant: str = "answer"


MATRIX_VARIANTS: dict[str, MatrixVariant] = {
    "fp32": MatrixVariant(name="fp32"),
    "tf32": MatrixVariant(name="tf32", float32_matmul_precision="high", allow_tf32=True),
    "tf32_bf16": MatrixVariant(
        name="tf32_bf16",
        float32_matmul_precision="high",
        allow_tf32=True,
        amp_mode="bf16",
    ),
    "tf32_bs32_inf32": MatrixVariant(
        name="tf32_bs32_inf32",
        float32_matmul_precision="high",
        allow_tf32=True,
        train_batch_size=32,
        inference_batch_size=32,
    ),
    "tf32_bf16_bs32_inf32": MatrixVariant(
        name="tf32_bf16_bs32_inf32",
        float32_matmul_precision="high",
        allow_tf32=True,
        amp_mode="bf16",
        train_batch_size=32,
        inference_batch_size=32,
    ),
    "tf32_bf16_bs32_inf32_combined": MatrixVariant(
        name="tf32_bf16_bs32_inf32_combined",
        float32_matmul_precision="high",
        allow_tf32=True,
        amp_mode="bf16",
        train_batch_size=32,
        inference_batch_size=32,
        checkpoint_f1_variant="combined",
    ),
}
DEFAULT_VARIANTS = ("tf32_bf16_bs32_inf32",)


@dataclass(frozen=True)
class MatrixDataSources:
    """Resolved CSV inputs for a benchmark matrix run."""

    csv_path: str | None = None
    train_csv_path: str | None = None
    eval_csv_path: str | None = None
    selected_cleaned_csv_files: tuple[str, ...] = ()

    @property
    def csv_sources(self) -> tuple[str, ...]:
        """Return unique CSV sources used by the run."""
        candidates = [self.csv_path, self.train_csv_path, self.eval_csv_path]
        values: list[str] = []
        for candidate in candidates:
            if candidate and candidate not in values:
                values.append(candidate)
        return tuple(values)


@dataclass
class MatrixChildResult:
    """Completed child process result with retained stdout tail, timings, and elapsed time."""

    returncode: int
    stdout: str
    stdout_truncated: bool
    timings: dict[str, Any]
    elapsed_seconds: float


def _append_stdout_tail(tail_chunks: deque[str], tail_chars: int, line: str, max_chars: int) -> tuple[int, bool]:
    """Append a line to the retained stdout tail and trim old chunks past max_chars."""
    if max_chars <= 0:
        return 0, True

    truncated = False
    if len(line) > max_chars:
        tail_chunks.clear()
        tail_chunks.append(line[-max_chars:])
        return max_chars, True

    tail_chunks.append(line)
    tail_chars += len(line)
    while tail_chars > max_chars and tail_chunks:
        overflow = tail_chars - max_chars
        first = tail_chunks[0]
        truncated = True
        if len(first) <= overflow:
            tail_chars -= len(first)
            tail_chunks.popleft()
        else:
            tail_chunks[0] = first[overflow:]
            tail_chars -= overflow
            break
    return tail_chars, truncated


def _append_timing_line(timings: dict[str, list[dict[str, Any]]], line: str) -> None:
    """Parse one child stdout line into the matrix timing accumulator."""
    phase_match = PHASE_DONE_RE.search(line)
    if phase_match:
        timings["phase_timings"].append(
            {
                "name": phase_match.group("name").strip(),
                "seconds": float(phase_match.group("seconds")),
            }
        )

    epoch_match = EPOCH_RE.search(line)
    if epoch_match:
        timings["epoch_timings"].append(
            {
                "epoch": int(epoch_match.group("epoch")),
                "total_epochs": int(epoch_match.group("total")),
                "seconds": float(epoch_match.group("seconds")),
                "line": line.strip(),
            }
        )

    inference_match = INFERENCE_STEP_RE.search(line)
    if inference_match:
        timings["inference_step_timings"].append(
            {
                "name": inference_match.group("name").strip(),
                "seconds": float(inference_match.group("seconds")),
                "line": line.strip(),
            }
        )


def _run_capture_streaming(
    command: list[str],
    cwd: Path,
    stdout_path: Path,
    *,
    max_stdout_chars: int = DEFAULT_CHILD_STDOUT_TAIL_CHARS,
) -> MatrixChildResult:
    """Run a child command while streaming stdout to console and a log file."""
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    tail_chunks: deque[str] = deque()
    tail_chars = 0
    stdout_truncated = False
    timings: dict[str, list[dict[str, Any]]] = {
        "phase_timings": [],
        "epoch_timings": [],
        "inference_step_timings": [],
    }
    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    try:
        with open(stdout_path, "w", encoding="utf-8") as log:
            assert proc.stdout is not None
            for line in proc.stdout:
                tail_chars, line_truncated = _append_stdout_tail(tail_chunks, tail_chars, line, max_stdout_chars)
                stdout_truncated = stdout_truncated or line_truncated
                _append_timing_line(timings, line)
                log.write(line)
                log.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
        returncode = int(proc.wait())
    except BaseException:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        raise
    elapsed = time.perf_counter() - started
    return MatrixChildResult(
        returncode=returncode,
        stdout="".join(tail_chunks),
        stdout_truncated=stdout_truncated,
        timings=timings,
        elapsed_seconds=float(elapsed),
    )


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


def _selected_variants(raw: str | None) -> list[MatrixVariant]:
    """Return benchmark variants selected by CLI text."""
    names = _parse_name_list(
        raw or ",".join(DEFAULT_VARIANTS),
        allowed=set(MATRIX_VARIANTS),
        arg_name="--variants",
    )
    return [MATRIX_VARIANTS[name] for name in names]


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


def _resolve_data_sources(args: argparse.Namespace) -> MatrixDataSources:
    """Resolve matrix CSV inputs, using two cleaned days for directory inputs."""
    has_train_eval = bool(args.train_csv_path or args.eval_csv_path)
    if has_train_eval:
        if not args.train_csv_path or not args.eval_csv_path:
            raise ValueError("--train_csv_path and --eval_csv_path must be supplied together.")
        if args.csv_path:
            raise ValueError("--csv_path cannot be combined with --train_csv_path/--eval_csv_path.")
        train_path = str(Path(args.train_csv_path))
        eval_path = str(Path(args.eval_csv_path))
        for source in (train_path, eval_path):
            if not Path(source).is_file():
                raise FileNotFoundError(f"CSV path does not exist or is not a file: {source}")
        return MatrixDataSources(
            train_csv_path=train_path,
            eval_csv_path=eval_path,
            selected_cleaned_csv_files=(train_path, eval_path),
        )

    if not args.csv_path:
        return MatrixDataSources()

    source_path = Path(args.csv_path)
    files = _cleaned_csv_files(args.csv_path)
    if source_path.is_dir():
        if len(files) < MIN_REALISTIC_CSV_DAYS:
            raise ValueError(f"Expected at least {MIN_REALISTIC_CSV_DAYS} cleaned CSV files in {args.csv_path}.")
        selected = tuple(str(path) for path in files[:MIN_REALISTIC_CSV_DAYS])
        return MatrixDataSources(
            train_csv_path=selected[0],
            eval_csv_path=selected[1],
            selected_cleaned_csv_files=selected,
        )
    if len(files) == 1:
        return MatrixDataSources(csv_path=str(files[0]), selected_cleaned_csv_files=(str(files[0]),))
    raise ValueError(f"Expected a cleaned CSV file or directory: {args.csv_path}")


def _profile_args(
    profile: str,
    args: argparse.Namespace,
    data_sources: MatrixDataSources | None = None,
    *,
    include_refresh_cache: bool = True,
) -> list[str]:
    """Return effective child CLI arguments for a matrix profile."""
    if profile != DEFAULT_PROFILE:
        raise ValueError(f"Unknown matrix profile: {profile}")

    data_sources = data_sources or _resolve_data_sources(args)
    if data_sources.train_csv_path and data_sources.eval_csv_path:
        profile_args = [
            "--train_csv_path",
            data_sources.train_csv_path,
            "--eval_csv_path",
            data_sources.eval_csv_path,
            *REAL_USECASE_PROFILE_ARGS,
        ]
    elif data_sources.csv_path:
        profile_args = ["--csv_path", data_sources.csv_path, *REAL_USECASE_PROFILE_ARGS]
    else:
        raise ValueError(f"{DEFAULT_PROFILE} requires --csv_path or --train_csv_path/--eval_csv_path.")

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


def _profile_settings(profile: str) -> dict[str, int | float | str]:
    """Return compact profile settings recorded in run_config.json."""
    if profile == DEFAULT_PROFILE:
        return {
            "data_mode": "two_cleaned_csv_days",
            "train_day": "first sorted cleaned CSV",
            "eval_day": "second sorted cleaned CSV",
            "n_queries": 400,
            "query_coverage": 0.30,
            "range_spatial_fraction": REAL_USECASE_RANGE_SPATIAL_FRACTION,
            "range_time_fraction": REAL_USECASE_RANGE_TIME_FRACTION,
            "compression_ratio": 0.05,
            "epochs": 20,
            "early_stopping_patience": 8,
            "checkpoint_selection_metric": "f1",
            "checkpoint_f1_variant": "answer",
            "f1_diagnostic_every": 1,
            "checkpoint_smoothing_window": 1,
            "mlqds_temporal_fraction": 0.10,
        }
    raise ValueError(f"Unknown matrix profile: {profile}")


def _variant_run_dir(results_dir: Path, workload: str, variant: MatrixVariant, workload_count: int) -> Path:
    """Return the child experiment output directory for a matrix row."""
    if workload_count == 1:
        return results_dir / "variants" / variant.name
    return results_dir / "variants" / workload / variant.name


def _variant_args(variant: MatrixVariant) -> list[str]:
    """Return CLI args for a variant."""
    return [
        "--float32_matmul_precision",
        variant.float32_matmul_precision,
        "--allow_tf32" if variant.allow_tf32 else "--no-allow_tf32",
        "--amp_mode",
        variant.amp_mode,
        "--train_batch_size",
        str(variant.train_batch_size),
        "--inference_batch_size",
        str(variant.inference_batch_size),
        "--checkpoint_selection_metric",
        "f1",
        "--checkpoint_f1_variant",
        variant.checkpoint_f1_variant,
    ]


def _phase_seconds(timings: dict[str, Any], name: str) -> float | None:
    """Extract one phase duration from parsed timings."""
    for row in timings.get("phase_timings", []):
        if row.get("name") == name:
            return float(row["seconds"])
    return None


def _phase_seconds_with_prefix(timings: dict[str, Any], prefix: str) -> float | None:
    """Extract the first phase duration whose name starts with a prefix."""
    for row in timings.get("phase_timings", []):
        if str(row.get("name", "")).startswith(prefix):
            return float(row["seconds"])
    return None


def _mean_epoch_seconds(timings: dict[str, Any]) -> float | None:
    """Return mean epoch duration from parsed stdout timings."""
    values = [float(row["seconds"]) for row in timings.get("epoch_timings", [])]
    return float(sum(values) / len(values)) if values else None


def _has_collapse_warning(run_json: dict[str, Any] | None) -> bool | None:
    """Return whether training history contains a collapse warning."""
    if not run_json:
        return None
    return any("collapse_warning" in row for row in run_json.get("training_history", []))


def _warm_csv_caches(args: argparse.Namespace, data_sources: MatrixDataSources) -> list[dict[str, Any]]:
    """Prebuild segmented AIS caches for all CSV sources used by the matrix."""
    if not args.cache_dir or not data_sources.csv_sources or args.no_cache_warmup:
        return []

    rows: list[dict[str, Any]] = []
    for source in data_sources.csv_sources:
        started = time.perf_counter()
        result = load_or_build_ais_cache(
            source,
            cache_dir=str(args.cache_dir),
            refresh_cache=bool(args.refresh_cache),
            min_points_per_segment=int(args.min_points_per_segment),
            max_points_per_segment=args.max_points_per_segment,
            max_time_gap_seconds=float(args.max_time_gap_seconds),
            max_segments=args.max_segments,
        )
        elapsed = time.perf_counter() - started
        audit = result.audit.to_dict()
        row = {
            "source_path": source,
            "cache_hit": bool(result.cache_hit),
            "elapsed_seconds": float(elapsed),
            "cache_dir": result.cache_dir,
            "manifest_path": result.manifest_path,
            "parquet_path": result.parquet_path,
            "output_segment_count": audit.get("output_segment_count"),
            "output_point_count": audit.get("output_point_count"),
            "segment_limit_reached": audit.get("segment_limit_reached"),
        }
        rows.append(row)
        state = "hit" if result.cache_hit else "built"
        print(
            "[matrix] cache warmup "
            f"{state}: {source} ({row['output_segment_count']} segments, {row['elapsed_seconds']:.2f}s)",
            flush=True,
        )
    return rows


def _row_from_run(
    *,
    workload: str,
    variant: MatrixVariant,
    command: list[str],
    returncode: int,
    elapsed_seconds: float,
    run_dir: Path,
    stdout_path: Path,
    run_json_path: Path,
    timings: dict[str, Any],
    run_json: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build one compact comparison row."""
    mlqds = (run_json or {}).get("matched", {}).get("MLQDS", {})
    uniform = (run_json or {}).get("matched", {}).get("uniform", {})
    dp = (run_json or {}).get("matched", {}).get("DouglasPeucker", {})
    cuda_memory = (run_json or {}).get("cuda_memory", {}).get("training", {})
    return {
        "workload": workload,
        "variant": variant.name,
        "returncode": int(returncode),
        "elapsed_seconds": float(elapsed_seconds),
        "train_seconds": _phase_seconds_with_prefix(timings, "train-model"),
        "evaluate_matched_seconds": _phase_seconds(timings, "evaluate-matched"),
        "epoch_mean_seconds": _mean_epoch_seconds(timings),
        "peak_allocated_mb": cuda_memory.get("max_allocated_mb"),
        "best_epoch": (run_json or {}).get("best_epoch"),
        "best_loss": (run_json or {}).get("best_loss"),
        "best_f1": (run_json or {}).get("best_f1"),
        "mlqds_f1": mlqds.get("aggregate_f1"),
        "mlqds_type_f1": (mlqds.get("per_type_f1") or {}).get(workload),
        "uniform_f1": uniform.get("aggregate_f1"),
        "douglas_peucker_f1": dp.get("aggregate_f1"),
        "mlqds_latency_ms": mlqds.get("latency_ms"),
        "avg_length_preserved": mlqds.get("avg_length_preserved"),
        "combined_query_shape_score": mlqds.get("combined_query_shape_score"),
        "collapse_warning": _has_collapse_warning(run_json),
        "checkpoint_f1_variant": variant.checkpoint_f1_variant,
        "float32_matmul_precision": variant.float32_matmul_precision,
        "allow_tf32": variant.allow_tf32,
        "amp_mode": variant.amp_mode,
        "train_batch_size": variant.train_batch_size,
        "inference_batch_size": variant.inference_batch_size,
        "run_dir": str(run_dir),
        "example_run_path": str(run_json_path) if run_json_path.exists() else None,
        "stdout_path": str(stdout_path),
        "command": command,
    }


def _format_value(value: Any) -> str:
    """Format values for a compact markdown table."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _format_markdown_table(rows: list[dict[str, Any]]) -> str:
    """Return a compact markdown comparison table."""
    columns = [
        "workload",
        "variant",
        "returncode",
        "elapsed_seconds",
        "epoch_mean_seconds",
        "peak_allocated_mb",
        "best_f1",
        "mlqds_f1",
        "uniform_f1",
        "douglas_peucker_f1",
        "mlqds_latency_ms",
        "collapse_warning",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_value(row.get(column)) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write compact rows as CSV."""
    if not rows:
        _write_text(path, "")
        return
    fieldnames = [key for key in rows[0] if key != "command"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON with a stable pretty format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _utc_now() -> str:
    """Return an ISO UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _family_root(results_dir: Path) -> Path:
    """Return the benchmark-family root that owns runs_index files."""
    if results_dir.parent.name == "runs":
        return results_dir.parent.parent
    return results_dir.parent


def _write_status(
    results_dir: Path,
    *,
    run_id: str,
    status: str,
    started_at_utc: str,
    finished_at_utc: str | None = None,
    exit_status: int | None = None,
    failures: int | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    """Write the current status marker for one run."""
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "status": status,
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "exit_status": exit_status,
        "failures": failures,
        "message": message,
        "results_dir": str(results_dir),
    }
    _write_json(results_dir / "run_status.json", payload)
    return payload


def _run_config(
    *,
    args: argparse.Namespace,
    run_id: str,
    workloads: list[str],
    variants: list[MatrixVariant],
    data_sources: MatrixDataSources,
    results_dir: Path,
) -> dict[str, Any]:
    """Build a compact config file for a matrix run."""
    return {
        "schema_version": 1,
        "run_id": run_id,
        "results_dir": str(results_dir),
        "profile": args.profile,
        "profile_settings": _profile_settings(args.profile),
        "seed": int(args.seed),
        "workloads": workloads,
        "variants": [
            {
                "name": variant.name,
                "float32_matmul_precision": variant.float32_matmul_precision,
                "allow_tf32": variant.allow_tf32,
                "amp_mode": variant.amp_mode,
                "train_batch_size": variant.train_batch_size,
                "inference_batch_size": variant.inference_batch_size,
                "checkpoint_f1_variant": variant.checkpoint_f1_variant,
            }
            for variant in variants
        ],
        "data_sources": {
            "csv_path": data_sources.csv_path,
            "train_csv_path": data_sources.train_csv_path,
            "eval_csv_path": data_sources.eval_csv_path,
            "selected_cleaned_csv_files": list(data_sources.selected_cleaned_csv_files),
        },
        "loader": {
            "cache_dir": args.cache_dir,
            "refresh_cache": bool(args.refresh_cache),
            "cache_warmup": not bool(args.no_cache_warmup),
            "min_points_per_segment": int(args.min_points_per_segment),
            "max_points_per_segment": args.max_points_per_segment,
            "max_time_gap_seconds": float(args.max_time_gap_seconds),
            "max_segments": args.max_segments,
            "max_trajectories": args.max_trajectories,
        },
        "checkpoint_selection_metric": "f1",
        "f1_diagnostic_every": int(args.f1_diagnostic_every),
        "extra_args": _split_extra_args(args.extra_args),
        "continue_on_failure": bool(args.continue_on_failure),
    }


RUN_INDEX_FIELDS = [
    "run_id",
    "status",
    "started_at_utc",
    "finished_at_utc",
    "exit_status",
    "failures",
    "profile",
    "seed",
    "workloads",
    "variants",
    "train_csv_path",
    "eval_csv_path",
    "csv_path",
    "max_points_per_segment",
    "max_segments",
    "max_trajectories",
    "results_dir",
    "best_mlqds_f1",
    "best_mlqds_variant",
    "git_commit",
    "git_dirty",
]


def _best_mlqds(rows: list[dict[str, Any]]) -> tuple[float | None, str | None]:
    """Return the best MLQDS aggregate F1 and variant name from completed rows."""
    best_value: float | None = None
    best_variant: str | None = None
    for row in rows:
        value = row.get("mlqds_f1")
        if value is None:
            continue
        numeric = float(value)
        if best_value is None or numeric > best_value:
            best_value = numeric
            best_variant = str(row.get("variant"))
    return best_value, best_variant


def _index_entry(
    *,
    run_id: str,
    status_payload: dict[str, Any],
    args: argparse.Namespace,
    workloads: list[str],
    variants: list[MatrixVariant],
    data_sources: MatrixDataSources,
    results_dir: Path,
    rows: list[dict[str, Any]],
    git: dict[str, Any],
) -> dict[str, Any]:
    """Build one family-level index row."""
    best_f1, best_variant = _best_mlqds(rows)
    return {
        "run_id": run_id,
        "status": status_payload.get("status"),
        "started_at_utc": status_payload.get("started_at_utc"),
        "finished_at_utc": status_payload.get("finished_at_utc"),
        "exit_status": status_payload.get("exit_status"),
        "failures": status_payload.get("failures"),
        "profile": args.profile,
        "seed": int(args.seed),
        "workloads": ",".join(workloads),
        "variants": ",".join(variant.name for variant in variants),
        "train_csv_path": data_sources.train_csv_path,
        "eval_csv_path": data_sources.eval_csv_path,
        "csv_path": data_sources.csv_path,
        "max_points_per_segment": args.max_points_per_segment,
        "max_segments": args.max_segments,
        "max_trajectories": args.max_trajectories,
        "results_dir": str(results_dir),
        "best_mlqds_f1": best_f1,
        "best_mlqds_variant": best_variant,
        "git_commit": git.get("commit"),
        "git_dirty": git.get("dirty"),
    }


def _write_family_indexes(family_root: Path, entry: dict[str, Any]) -> None:
    """Update current run index CSV and append an event JSONL row."""
    family_root.mkdir(parents=True, exist_ok=True)
    (family_root / "latest_run.txt").write_text(str(entry.get("results_dir", "")) + "\n", encoding="utf-8")
    csv_path = family_root / "runs_index.csv"
    rows: list[dict[str, Any]] = []
    if csv_path.exists():
        with open(csv_path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    replaced = False
    for idx, row in enumerate(rows):
        if row.get("run_id") == entry.get("run_id"):
            rows[idx] = {field: entry.get(field) for field in RUN_INDEX_FIELDS}
            replaced = True
            break
    if not replaced:
        rows.append({field: entry.get(field) for field in RUN_INDEX_FIELDS})
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    event = dict(entry)
    event["event_recorded_at_utc"] = _utc_now()
    with open(family_root / "runs_index_events.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


def _artifact_index(results_dir: Path, artifact: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a readable index of benchmark artifacts and child run outputs."""
    return {
        "schema_version": 1,
        "run_id": artifact.get("run_id"),
        "artifact_root": str(results_dir),
        "top_level_files": {
            "readme": str(results_dir / "README.md"),
            "run_config": str(results_dir / "run_config.json"),
            "run_status": str(results_dir / "run_status.json"),
            "benchmark_matrix_json": str(results_dir / "benchmark_matrix.json"),
            "benchmark_matrix_csv": str(results_dir / "benchmark_matrix.csv"),
            "benchmark_matrix_markdown": str(results_dir / "benchmark_matrix.md"),
            "artifact_index_json": str(results_dir / "artifact_index.json"),
            "family_runs_index_csv": str(_family_root(results_dir) / "runs_index.csv"),
            "family_runs_index_events_jsonl": str(_family_root(results_dir) / "runs_index_events.jsonl"),
        },
        "logs": {
            "console_log": str(results_dir / "logs" / "console.log"),
            "system_monitor_log": str(results_dir / "logs" / "system_monitor.log"),
            "tmux_status": str(results_dir / "logs" / "tmux_status.txt"),
        },
        "variant_runs": [
            {
                "workload": row.get("workload"),
                "variant": row.get("variant"),
                "returncode": row.get("returncode"),
                "run_dir": row.get("run_dir"),
                "example_run_json": row.get("example_run_path"),
                "stdout_log": row.get("stdout_path"),
                "matched_table": str(Path(str(row.get("run_dir"))) / "matched_table.txt") if row.get("run_dir") else None,
                "simplified_eval_dir": str(Path(str(row.get("run_dir"))) / "simplified_eval")
                if row.get("run_dir")
                else None,
                "range_diagnostics": str(Path(str(row.get("run_dir"))) / "range_workload_diagnostics.json")
                if row.get("run_dir")
                else None,
            }
            for row in rows
        ],
    }


def _format_artifact_readme(artifact: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    """Return a short artifact guide for one benchmark matrix run."""
    run_id = artifact.get("run_id") or "(not set)"
    lines = [
        "# QDS Benchmark Matrix Run",
        "",
        f"- Run ID: `{run_id}`",
        f"- Profile: `{artifact.get('profile')}`",
        f"- Seed: `{artifact.get('seed')}`",
        f"- Workloads: `{', '.join(artifact.get('workloads', []))}`",
        f"- Variants: `{', '.join(artifact.get('variants', []))}`",
        "",
        "## Top-Level Files",
        "",
        "- `run_config.json` - compact benchmark configuration",
        "- `run_status.json` - current/final run status marker",
        "- `benchmark_matrix.md` - compact comparison table",
        "- `benchmark_matrix.csv` - comparison table as CSV",
        "- `benchmark_matrix.json` - complete machine-readable matrix artifact",
        "- `artifact_index.json` - paths to logs and child run artifacts",
        "- `logs/console.log` - tmux/launcher console capture when launched through tmux",
        "- `logs/system_monitor.log` - RAM/GPU/system samples when launched through tmux",
        "- `logs/tmux_status.txt` - launcher start/end status when launched through tmux",
        "- family `runs_index.csv` - current status summary for sibling runs",
        "- family `runs_index_events.jsonl` - append-only status history",
        "",
        "## Variant Runs",
        "",
        "| workload | variant | returncode | run_dir |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('workload')} | {row.get('variant')} | {row.get('returncode')} | `{row.get('run_dir')}` |"
        )
    lines.append("")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    """Build benchmark matrix CLI."""
    parser = argparse.ArgumentParser(
        description="Run a range-focused AIS-QDS benchmark matrix and write compact comparison tables.",
    )
    parser.add_argument("--profile", choices=PROFILE_CHOICES, default=DEFAULT_PROFILE)
    parser.add_argument("--workloads", type=str, default=",".join(DEFAULT_WORKLOADS))
    parser.add_argument("--variants", type=str, default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_dir", type=str, default="artifacts/benchmarks/range_workload_matrix")
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Optional human-readable run identifier recorded in benchmark_matrix.json and README.md.",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help=(
            "Cleaned AIS CSV or directory. A directory selects the first two sorted "
            "CSV files as train/eval days for the range real-usecase benchmark."
        ),
    )
    parser.add_argument("--train_csv_path", "--train_csv", dest="train_csv_path", type=str, default=None)
    parser.add_argument("--eval_csv_path", "--eval_csv", dest="eval_csv_path", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--refresh_cache", action="store_true")
    parser.add_argument(
        "--no_cache_warmup",
        action="store_true",
        help="Skip prebuilding segmented AIS caches before measured child runs.",
    )
    parser.add_argument("--min_points_per_segment", type=int, default=4)
    parser.add_argument("--max_points_per_segment", type=int, default=None)
    parser.add_argument("--max_time_gap_seconds", type=float, default=3600.0)
    parser.add_argument("--max_segments", type=int, default=None)
    parser.add_argument("--max_trajectories", type=int, default=None)
    parser.add_argument(
        "--f1_diagnostic_every",
        type=int,
        default=1,
        help="Held-out query-F1 diagnostic cadence passed to each child run.",
    )
    parser.add_argument(
        "--extra_args",
        type=str,
        default=None,
        help="Quoted extra args appended to every run_ais_experiment child command.",
    )
    parser.add_argument(
        "--continue_on_failure",
        action="store_true",
        help="Continue remaining matrix runs after a child failure.",
    )
    return parser


def main() -> None:
    """Run the benchmark matrix."""
    args = _build_parser().parse_args()
    workloads = _parse_name_list(args.workloads, allowed=PURE_WORKLOADS, arg_name="--workloads")
    variants = _selected_variants(args.variants)
    data_sources = _resolve_data_sources(args)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "logs").mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or results_dir.name
    family_root = _family_root(results_dir)
    started_at_utc = _utc_now()
    git = _git_metadata()
    environment = _environment_metadata("off")
    rows: list[dict[str, Any]] = []
    failures = 0
    run_config = _run_config(
        args=args,
        run_id=run_id,
        workloads=workloads,
        variants=variants,
        data_sources=data_sources,
        results_dir=results_dir,
    )
    _write_json(results_dir / "run_config.json", run_config)
    status_payload = _write_status(
        results_dir,
        run_id=run_id,
        status="running",
        started_at_utc=started_at_utc,
        message="benchmark matrix started",
    )
    _write_family_indexes(
        family_root,
        _index_entry(
            run_id=run_id,
            status_payload=status_payload,
            args=args,
            workloads=workloads,
            variants=variants,
            data_sources=data_sources,
            results_dir=results_dir,
            rows=[],
            git=git,
        ),
    )

    def _mark_interrupted(signum: int, _frame: Any) -> None:
        signal_name = signal.Signals(signum).name
        interrupted = _write_status(
            results_dir,
            run_id=run_id,
            status="interrupted",
            started_at_utc=started_at_utc,
            finished_at_utc=_utc_now(),
            exit_status=128 + int(signum),
            failures=failures,
            message=f"benchmark matrix interrupted by {signal_name}",
        )
        _write_family_indexes(
            family_root,
            _index_entry(
                run_id=run_id,
                status_payload=interrupted,
                args=args,
                workloads=workloads,
                variants=variants,
                data_sources=data_sources,
                results_dir=results_dir,
                rows=rows,
                git=git,
            ),
        )
        raise KeyboardInterrupt(signal_name)

    signal.signal(signal.SIGINT, _mark_interrupted)
    signal.signal(signal.SIGTERM, _mark_interrupted)

    try:
        cache_warmup = _warm_csv_caches(args, data_sources)
        measured_include_refresh = bool(args.refresh_cache and not cache_warmup)

        for workload in workloads:
            for variant in variants:
                run_dir = _variant_run_dir(results_dir, workload, variant, len(workloads))
                command = [
                    sys.executable,
                    "-m",
                    "src.experiments.run_ais_experiment",
                    *_profile_args(
                        args.profile,
                        args,
                        data_sources,
                        include_refresh_cache=measured_include_refresh,
                    ),
                    "--workload",
                    workload,
                    "--seed",
                    str(args.seed),
                    "--results_dir",
                    str(run_dir),
                    "--f1_diagnostic_every",
                    str(args.f1_diagnostic_every),
                    *_variant_args(variant),
                    *_split_extra_args(args.extra_args),
                ]
                print(f"[matrix] {workload}/{variant.name}: {' '.join(shlex.quote(part) for part in command)}", flush=True)
                stdout_path = run_dir / "stdout.log"
                proc = _run_capture_streaming(command, cwd=_qds_root(), stdout_path=stdout_path)
                run_json_path = run_dir / "example_run.json"
                run_json = json.loads(run_json_path.read_text(encoding="utf-8")) if run_json_path.exists() else None
                timings = proc.timings
                row = _row_from_run(
                    workload=workload,
                    variant=variant,
                    command=command,
                    returncode=proc.returncode,
                    elapsed_seconds=float(getattr(proc, "elapsed_seconds", 0.0)),
                    run_dir=run_dir,
                    stdout_path=stdout_path,
                    run_json_path=run_json_path,
                    timings=timings,
                    run_json=run_json,
                )
                rows.append(row)
                failures += int(proc.returncode != 0)
                if proc.returncode != 0:
                    print(
                        f"[matrix] {workload}/{variant.name} failed with returncode={proc.returncode}; "
                        f"see {stdout_path}",
                        flush=True,
                    )
                if proc.returncode != 0 and not args.continue_on_failure:
                    break
            if failures and not args.continue_on_failure:
                break
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        failed_status = _write_status(
            results_dir,
            run_id=run_id,
            status="failed",
            started_at_utc=started_at_utc,
            finished_at_utc=_utc_now(),
            exit_status=1,
            failures=failures,
            message=f"{type(exc).__name__}: {exc}",
        )
        _write_family_indexes(
            family_root,
            _index_entry(
                run_id=run_id,
                status_payload=failed_status,
                args=args,
                workloads=workloads,
                variants=variants,
                data_sources=data_sources,
                results_dir=results_dir,
                rows=rows,
                git=git,
            ),
        )
        raise

    artifact = {
        "schema_version": 3,
        "timestamp_utc": _utc_now(),
        "command": [sys.executable, "-m", "src.experiments.benchmark_matrix", *sys.argv[1:]],
        "run_id": run_id,
        "artifact_root": str(results_dir),
        "family_root": str(family_root),
        "profile": args.profile,
        "seed": int(args.seed),
        "workloads": workloads,
        "variants": [variant.name for variant in variants],
        "run_config": run_config,
        "data_sources": {
            "csv_path": data_sources.csv_path,
            "train_csv_path": data_sources.train_csv_path,
            "eval_csv_path": data_sources.eval_csv_path,
            "selected_cleaned_csv_files": list(data_sources.selected_cleaned_csv_files),
        },
        "cache_warmup": cache_warmup,
        "environment": environment,
        "git": git,
        "rows": rows,
    }
    finished_at_utc = _utc_now()
    status = "failed" if failures else "completed"
    status_payload = _write_status(
        results_dir,
        run_id=run_id,
        status=status,
        started_at_utc=started_at_utc,
        finished_at_utc=finished_at_utc,
        exit_status=1 if failures else 0,
        failures=failures,
        message=f"{failures} matrix run(s) failed" if failures else "benchmark matrix completed",
    )
    artifact["run_status"] = status_payload
    _write_json(results_dir / "benchmark_matrix.json", artifact)
    _write_csv(results_dir / "benchmark_matrix.csv", rows)
    _write_text(results_dir / "benchmark_matrix.md", _format_markdown_table(rows))
    index = _artifact_index(results_dir, artifact, rows)
    _write_json(results_dir / "artifact_index.json", index)
    _write_text(results_dir / "README.md", _format_artifact_readme(artifact, rows))
    _write_family_indexes(
        family_root,
        _index_entry(
            run_id=run_id,
            status_payload=status_payload,
            args=args,
            workloads=workloads,
            variants=variants,
            data_sources=data_sources,
            results_dir=results_dir,
            rows=rows,
            git=git,
        ),
    )
    print(f"[matrix] wrote {results_dir / 'benchmark_matrix.md'}", flush=True)
    if failures:
        raise SystemExit(f"{failures} matrix run(s) failed. See {results_dir / 'benchmark_matrix.json'}.")


if __name__ == "__main__":
    main()
