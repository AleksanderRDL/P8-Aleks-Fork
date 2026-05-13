"""Run range-focused AIS-QDS benchmark profiles and write comparison reports."""

from __future__ import annotations

import argparse
import json
import signal
import shlex
import sys
import time
from pathlib import Path
from typing import Any

from data.trajectory_cache import load_or_build_ais_cache
from experiments.benchmark_artifacts import (
    artifact_index,
    family_root,
    format_artifact_readme,
    index_entry,
    utc_now,
    write_csv,
    write_family_indexes,
    write_json,
    write_status,
)
from experiments.benchmark_profiles import (
    DEFAULT_PROFILE,
    PROFILE_CHOICES,
)
from experiments.benchmark_inputs import (
    DEFAULT_WORKLOADS,
    PURE_WORKLOADS,
    BenchmarkDataSources,
    _parse_name_list,
    _profile_args,
    _profile_settings,
    _resolve_data_sources,
    _runner_environment_metadata,
)
from experiments.benchmark_process import _run_capture_streaming
from experiments.benchmark_report import _child_run_dir, _format_report_table, _row_from_run
from experiments.benchmark_runtime import (
    _git_metadata,
    _qds_root,
    _split_extra_args,
    _write_text,
)


def _warm_csv_caches(args: argparse.Namespace, data_sources: BenchmarkDataSources) -> list[dict[str, Any]]:
    """Prebuild segmented AIS caches for all CSV sources used by the benchmark."""
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
            "[benchmark] cache warmup "
            f"{state}: {source} ({row['output_segment_count']} segments, {row['elapsed_seconds']:.2f}s)",
            flush=True,
        )
    return rows


def _run_config(
    *,
    args: argparse.Namespace,
    run_id: str,
    workloads: list[str],
    run_label: str,
    data_sources: BenchmarkDataSources,
    results_dir: Path,
) -> dict[str, Any]:
    """Build a compact config file for a benchmark run."""
    return {
        "schema_version": 1,
        "run_id": run_id,
        "results_dir": str(results_dir),
        "profile": args.profile,
        "profile_settings": _profile_settings(args.profile),
        "seed": int(args.seed),
        "workloads": workloads,
        "run_label": run_label,
        "data_sources": {
            "csv_path": data_sources.csv_path,
            "train_csv_path": data_sources.train_csv_path,
            "validation_csv_path": data_sources.validation_csv_path,
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
        "checkpoint_selection_metric": _profile_settings(args.profile).get("checkpoint_selection_metric"),
        "validation_score_every": int(args.validation_score_every),
        "extra_args": _split_extra_args(args.extra_args),
        "continue_on_failure": bool(args.continue_on_failure),
    }


def _build_parser() -> argparse.ArgumentParser:
    """Build benchmark CLI."""
    parser = argparse.ArgumentParser(
        description="Run a range-focused AIS-QDS benchmark and write compact comparison tables.",
    )
    parser.add_argument("--profile", choices=PROFILE_CHOICES, default=DEFAULT_PROFILE)
    parser.add_argument("--workloads", type=str, default=",".join(DEFAULT_WORKLOADS))
    parser.add_argument(
        "--run_label",
        type=str,
        default=None,
        help="Optional label for the child run row/directory. Defaults to the selected profile name.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--results_dir",
        type=str,
        default="artifacts/benchmarks/range_workload_aware_diagnostic/runs/manual_benchmark",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Optional human-readable run identifier recorded in benchmark_report.json and README.md.",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help=(
            "Cleaned AIS CSV or directory. A directory selects the first three sorted "
            "CSV files as train/validation/eval days for the range workload-aware diagnostic benchmark."
        ),
    )
    parser.add_argument("--train_csv_path", "--train_csv", dest="train_csv_path", type=str, default=None)
    parser.add_argument(
        "--validation_csv_path",
        "--validation_csv",
        "--val_csv_path",
        "--val_csv",
        dest="validation_csv_path",
        type=str,
        default=None,
    )
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
        "--validation_score_every",
        type=int,
        default=1,
        help="Held-out validation-score cadence passed to each child run.",
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
        help="Continue remaining benchmark runs after a child failure.",
    )
    return parser


def main() -> None:
    """Run the benchmark run."""
    args = _build_parser().parse_args()
    workloads = _parse_name_list(args.workloads, allowed=PURE_WORKLOADS, arg_name="--workloads")
    run_label = args.run_label or args.profile
    data_sources = _resolve_data_sources(args)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "logs").mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or results_dir.name
    run_family_root = family_root(results_dir)
    started_at_utc = utc_now()
    git = _git_metadata()
    environment = _runner_environment_metadata()
    rows: list[dict[str, Any]] = []
    failures = 0
    run_config = _run_config(
        args=args,
        run_id=run_id,
        workloads=workloads,
        run_label=run_label,
        data_sources=data_sources,
        results_dir=results_dir,
    )
    write_json(results_dir / "run_config.json", run_config)
    status_payload = write_status(
        results_dir,
        run_id=run_id,
        status="running",
        started_at_utc=started_at_utc,
        message="benchmark run started",
    )
    write_family_indexes(
        run_family_root,
        index_entry(
            run_id=run_id,
            status_payload=status_payload,
            args=args,
            workloads=workloads,
            run_label=run_label,
            data_sources=data_sources,
            results_dir=results_dir,
            rows=[],
            git=git,
        ),
    )

    def _mark_interrupted(signum: int, _frame: Any) -> None:
        signal_name = signal.Signals(signum).name
        interrupted = write_status(
            results_dir,
            run_id=run_id,
            status="interrupted",
            started_at_utc=started_at_utc,
            finished_at_utc=utc_now(),
            exit_status=128 + int(signum),
            failures=failures,
            message=f"benchmark run interrupted by {signal_name}",
        )
        write_family_indexes(
            run_family_root,
            index_entry(
                run_id=run_id,
                status_payload=interrupted,
                args=args,
                workloads=workloads,
                run_label=run_label,
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
            run_dir = _child_run_dir(results_dir, workload, run_label, len(workloads))
            command = [
                sys.executable,
                "-m",
                "experiments.run_ais_experiment",
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
                "--validation_score_every",
                str(args.validation_score_every),
                *_split_extra_args(args.extra_args),
            ]
            print(f"[benchmark] {workload}/{run_label}: {' '.join(shlex.quote(part) for part in command)}", flush=True)
            stdout_path = run_dir / "stdout.log"
            proc = _run_capture_streaming(command, cwd=_qds_root(), stdout_path=stdout_path)
            run_json_path = run_dir / "example_run.json"
            run_json = json.loads(run_json_path.read_text(encoding="utf-8")) if run_json_path.exists() else None
            timings = proc.timings
            row = _row_from_run(
                workload=workload,
                run_label=run_label,
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
                    f"[benchmark] {workload}/{run_label} failed with returncode={proc.returncode}; "
                    f"see {stdout_path}",
                    flush=True,
                )
            if failures and not args.continue_on_failure:
                break
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        failed_status = write_status(
            results_dir,
            run_id=run_id,
            status="failed",
            started_at_utc=started_at_utc,
            finished_at_utc=utc_now(),
            exit_status=1,
            failures=failures,
            message=f"{type(exc).__name__}: {exc}",
        )
        write_family_indexes(
            run_family_root,
            index_entry(
                run_id=run_id,
                status_payload=failed_status,
                args=args,
                workloads=workloads,
                run_label=run_label,
                data_sources=data_sources,
                results_dir=results_dir,
                rows=rows,
                git=git,
            ),
        )
        raise

    artifact = {
        "schema_version": 4,
        "timestamp_utc": utc_now(),
        "command": [sys.executable, "-m", "experiments.benchmark_runner", *sys.argv[1:]],
        "run_id": run_id,
        "artifact_root": str(results_dir),
        "family_root": str(family_root),
        "profile": args.profile,
        "seed": int(args.seed),
        "workloads": workloads,
        "run_label": run_label,
        "run_config": run_config,
        "data_sources": {
            "csv_path": data_sources.csv_path,
            "train_csv_path": data_sources.train_csv_path,
            "validation_csv_path": data_sources.validation_csv_path,
            "eval_csv_path": data_sources.eval_csv_path,
            "selected_cleaned_csv_files": list(data_sources.selected_cleaned_csv_files),
        },
        "cache_warmup": cache_warmup,
        "environment": environment,
        "git": git,
        "rows": rows,
    }
    finished_at_utc = utc_now()
    status = "failed" if failures else "completed"
    status_payload = write_status(
        results_dir,
        run_id=run_id,
        status=status,
        started_at_utc=started_at_utc,
        finished_at_utc=finished_at_utc,
        exit_status=1 if failures else 0,
        failures=failures,
        message=f"{failures} benchmark run(s) failed" if failures else "benchmark run completed",
    )
    artifact["run_status"] = status_payload
    write_json(results_dir / "benchmark_report.json", artifact)
    write_csv(results_dir / "benchmark_report.csv", rows)
    _write_text(results_dir / "benchmark_report.md", _format_report_table(rows))
    index = artifact_index(results_dir, artifact, rows)
    write_json(results_dir / "artifact_index.json", index)
    _write_text(results_dir / "README.md", format_artifact_readme(artifact, rows))
    write_family_indexes(
        run_family_root,
        index_entry(
            run_id=run_id,
            status_payload=status_payload,
            args=args,
            workloads=workloads,
            run_label=run_label,
            data_sources=data_sources,
            results_dir=results_dir,
            rows=rows,
            git=git,
        ),
    )
    print(f"[benchmark] wrote {results_dir / 'benchmark_report.md'}", flush=True)
    if failures:
        raise SystemExit(f"{failures} benchmark run(s) failed. See {results_dir / 'benchmark_report.json'}.")


if __name__ == "__main__":
    main()
