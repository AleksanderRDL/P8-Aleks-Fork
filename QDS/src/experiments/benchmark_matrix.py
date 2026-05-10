"""Run pure-workload benchmark matrices for AIS-QDS configuration tuning."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.experiments.benchmark_runtime import (
    _git_metadata,
    _parse_timings,
    _qds_root,
    _run_capture,
    _split_extra_args,
    _write_text,
)

PURE_WORKLOADS = ("range", "knn", "similarity", "clustering")


@dataclass(frozen=True)
class MatrixVariant:
    """Runtime/config variant for one pure-workload matrix run."""

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
DEFAULT_VARIANTS = (
    "fp32",
    "tf32",
    "tf32_bf16",
    "tf32_bs32_inf32",
    "tf32_bf16_bs32_inf32",
    "tf32_bf16_bs32_inf32_combined",
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


def _profile_args(profile: str, args: argparse.Namespace) -> list[str]:
    """Return data-size arguments for a matrix profile."""
    if profile == "small":
        size_args = ["--n_ships", "6", "--n_points", "48", "--n_queries", "12", "--epochs", "1"]
    elif profile == "medium":
        size_args = ["--n_ships", "16", "--n_points", "128", "--n_queries", "64", "--epochs", "8"]
    elif profile == "serious":
        size_args = ["--n_ships", "32", "--n_points", "256", "--n_queries", "192", "--epochs", "20"]
    else:
        raise ValueError(f"Unknown matrix profile: {profile}")

    if args.csv_path:
        profile_args = ["--csv_path", str(args.csv_path), *size_args[4:]]
        if args.max_points_per_segment is not None:
            profile_args += ["--max_points_per_segment", str(args.max_points_per_segment)]
        if args.max_segments is not None:
            profile_args += ["--max_segments", str(args.max_segments)]
        if args.cache_dir is not None:
            profile_args += ["--cache_dir", str(args.cache_dir)]
        if args.refresh_cache:
            profile_args.append("--refresh_cache")
        return profile_args

    return size_args


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


def _row_from_run(
    *,
    workload: str,
    variant: MatrixVariant,
    command: list[str],
    returncode: int,
    elapsed_seconds: float,
    stdout_path: Path,
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


def _build_parser() -> argparse.ArgumentParser:
    """Build benchmark matrix CLI."""
    parser = argparse.ArgumentParser(
        description="Run a pure-workload AIS-QDS benchmark matrix and write compact comparison tables.",
    )
    parser.add_argument("--profile", choices=["small", "medium", "serious"], default="medium")
    parser.add_argument("--workloads", type=str, default=",".join(PURE_WORKLOADS))
    parser.add_argument("--variants", type=str, default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_dir", type=str, default="artifacts/benchmarks/pure_workload_matrix")
    parser.add_argument("--csv_path", type=str, default=None, help="Optional cleaned AIS CSV or directory for realistic runs.")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--refresh_cache", action="store_true")
    parser.add_argument("--max_points_per_segment", type=int, default=None)
    parser.add_argument("--max_segments", type=int, default=None)
    parser.add_argument(
        "--f1_diagnostic_every",
        type=int,
        default=2,
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
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    failures = 0
    for workload in workloads:
        for variant in variants:
            run_dir = results_dir / workload / variant.name
            command = [
                sys.executable,
                "-m",
                "src.experiments.run_ais_experiment",
                *_profile_args(args.profile, args),
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
            proc = _run_capture(command, cwd=_qds_root())
            stdout_path = run_dir / "stdout.log"
            _write_text(stdout_path, proc.stdout)
            run_json_path = run_dir / "example_run.json"
            run_json = json.loads(run_json_path.read_text(encoding="utf-8")) if run_json_path.exists() else None
            timings = _parse_timings(proc.stdout)
            row = _row_from_run(
                workload=workload,
                variant=variant,
                command=command,
                returncode=proc.returncode,
                elapsed_seconds=float(getattr(proc, "elapsed_seconds", 0.0)),
                stdout_path=stdout_path,
                timings=timings,
                run_json=run_json,
            )
            rows.append(row)
            failures += int(proc.returncode != 0)
            if proc.returncode != 0 and not args.continue_on_failure:
                break
        if failures and not args.continue_on_failure:
            break

    artifact = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, "-m", "src.experiments.benchmark_matrix", *sys.argv[1:]],
        "profile": args.profile,
        "seed": int(args.seed),
        "workloads": workloads,
        "variants": [variant.name for variant in variants],
        "git": _git_metadata(),
        "rows": rows,
    }
    with open(results_dir / "benchmark_matrix.json", "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    _write_csv(results_dir / "benchmark_matrix.csv", rows)
    _write_text(results_dir / "benchmark_matrix.md", _format_markdown_table(rows))
    print(f"[matrix] wrote {results_dir / 'benchmark_matrix.md'}", flush=True)
    if failures:
        raise SystemExit(f"{failures} matrix run(s) failed. See {results_dir / 'benchmark_matrix.json'}.")


if __name__ == "__main__":
    main()
