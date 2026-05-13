"""Benchmark artifact, status, and family-index writers."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
    "run_label",
    "train_csv_path",
    "validation_csv_path",
    "eval_csv_path",
    "csv_path",
    "max_points_per_segment",
    "max_segments",
    "max_trajectories",
    "results_dir",
    "best_mlqds_f1",
    "best_mlqds_run_label",
    "git_commit",
    "git_dirty",
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write compact rows as CSV."""
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return
    fieldnames = [key for key in rows[0] if key != "command"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON with a stable pretty format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def utc_now() -> str:
    """Return an ISO UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def family_root(results_dir: Path) -> Path:
    """Return the benchmark-family root that owns runs_index files."""
    if results_dir.parent.name == "runs":
        return results_dir.parent.parent
    return results_dir.parent


def write_status(
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
        "schema_version": 2,
        "run_id": run_id,
        "status": status,
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "exit_status": exit_status,
        "failures": failures,
        "message": message,
        "results_dir": str(results_dir),
    }
    write_json(results_dir / "run_status.json", payload)
    return payload


def _best_mlqds(rows: list[dict[str, Any]]) -> tuple[float | None, str | None]:
    """Return the best MLQDS aggregate F1 and run label from completed rows."""
    best_value: float | None = None
    best_run_label: str | None = None
    for row in rows:
        value = row.get("mlqds_f1")
        if value is None:
            continue
        numeric = float(value)
        if best_value is None or numeric > best_value:
            best_value = numeric
            best_run_label = str(row.get("run_label"))
    return best_value, best_run_label


def index_entry(
    *,
    run_id: str,
    status_payload: dict[str, Any],
    args: Any,
    workloads: list[str],
    run_label: str,
    data_sources: Any,
    results_dir: Path,
    rows: list[dict[str, Any]],
    git: dict[str, Any],
) -> dict[str, Any]:
    """Build one family-level index row."""
    best_f1, best_run_label = _best_mlqds(rows)
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
        "run_label": run_label,
        "train_csv_path": data_sources.train_csv_path,
        "validation_csv_path": data_sources.validation_csv_path,
        "eval_csv_path": data_sources.eval_csv_path,
        "csv_path": data_sources.csv_path,
        "max_points_per_segment": args.max_points_per_segment,
        "max_segments": args.max_segments,
        "max_trajectories": args.max_trajectories,
        "results_dir": str(results_dir),
        "best_mlqds_f1": best_f1,
        "best_mlqds_run_label": best_run_label,
        "git_commit": git.get("commit"),
        "git_dirty": git.get("dirty"),
    }


def write_family_indexes(family_root: Path, entry: dict[str, Any]) -> None:
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
    event["event_recorded_at_utc"] = utc_now()
    with open(family_root / "runs_index_events.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


def artifact_index(results_dir: Path, artifact: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a readable index of benchmark artifacts and child run outputs."""
    return {
        "schema_version": 1,
        "run_id": artifact.get("run_id"),
        "artifact_root": str(results_dir),
        "top_level_files": {
            "readme": str(results_dir / "README.md"),
            "run_config": str(results_dir / "run_config.json"),
            "run_status": str(results_dir / "run_status.json"),
            "benchmark_report_json": str(results_dir / "benchmark_report.json"),
            "benchmark_report_csv": str(results_dir / "benchmark_report.csv"),
            "benchmark_report_markdown": str(results_dir / "benchmark_report.md"),
            "artifact_index_json": str(results_dir / "artifact_index.json"),
            "family_runs_index_csv": str(family_root(results_dir) / "runs_index.csv"),
            "family_runs_index_events_jsonl": str(family_root(results_dir) / "runs_index_events.jsonl"),
        },
        "logs": {
            "console_log": str(results_dir / "logs" / "console.log"),
            "system_monitor_log": str(results_dir / "logs" / "system_monitor.log"),
            "tmux_status": str(results_dir / "logs" / "tmux_status.txt"),
        },
        "child_runs": [
            {
                "workload": row.get("workload"),
                "run_label": row.get("run_label"),
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
                "workload_distribution_comparison": str(
                    Path(str(row.get("run_dir"))) / "range_workload_distribution_comparison.json"
                )
                if row.get("run_dir")
                else None,
                "learned_fill_diagnostics": str(Path(str(row.get("run_dir"))) / "learned_fill_diagnostics.json")
                if row.get("run_dir")
                else None,
            }
            for row in rows
        ],
    }


def format_artifact_readme(artifact: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    """Return a short artifact guide for one benchmark run."""
    run_id = artifact.get("run_id") or "(not set)"
    lines = [
        "# QDS Benchmark Run",
        "",
        f"- Run ID: `{run_id}`",
        f"- Profile: `{artifact.get('profile')}`",
        f"- Seed: `{artifact.get('seed')}`",
        f"- Workloads: `{', '.join(artifact.get('workloads', []))}`",
        f"- Run label: `{artifact.get('run_label')}`",
        "",
        "## Top-Level Files",
        "",
        "- `run_config.json` - compact benchmark configuration",
        "- `run_status.json` - current/final run status marker",
        "- `benchmark_report.md` - compact comparison table",
        "- `benchmark_report.csv` - comparison table as CSV",
        "- `benchmark_report.json` - complete machine-readable benchmark artifact",
        "- `artifact_index.json` - paths to logs and child run artifacts",
        "- `logs/console.log` - tmux/launcher console capture when launched through tmux",
        "- `logs/system_monitor.log` - RAM/GPU/system samples when launched through tmux",
        "- `logs/tmux_status.txt` - launcher start/end status when launched through tmux",
        "- family `runs_index.csv` - current status summary for sibling runs",
        "- family `runs_index_events.jsonl` - append-only status history",
        "",
        "## Environment Scope",
        "",
        (
            "`benchmark_report.json.environment` describes the parent benchmark runner process. "
            "Effective child torch precision/AMP settings are recorded in "
            "`benchmark_report.json.rows[*].child_torch_runtime` and in each child `example_run.json`."
        ),
        "",
        "## Child Runs",
        "",
        "| workload | run_label | returncode | run_dir |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('workload')} | {row.get('run_label')} | {row.get('returncode')} | `{row.get('run_dir')}` |"
        )
    lines.append("")
    return "\n".join(lines)
