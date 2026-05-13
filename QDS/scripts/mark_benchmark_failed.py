#!/usr/bin/env python3
"""Mark a benchmark run failed when the launcher observes an abnormal exit."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


IndexRow = dict[str, str]


DEFAULT_INDEX_FIELDS: list[str] = [
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
    "best_mlqds_primary_metric",
    "best_mlqds_primary_score",
    "best_mlqds_aggregate_f1",
    "best_mlqds_range_point_f1",
    "best_mlqds_range_usefulness",
    "best_mlqds_f1",
    "best_mlqds_run_label",
    "git_commit",
    "git_dirty",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _family_root(results_dir: Path) -> Path:
    if results_dir.parent.name == "runs":
        return results_dir.parent.parent
    return results_dir.parent


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _index_value(value: Any) -> str:
    return "" if value is None else str(value)


def _load_index_rows(csv_path: Path) -> tuple[list[str], list[IndexRow]]:
    if not csv_path.exists():
        return list(DEFAULT_INDEX_FIELDS), []

    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or DEFAULT_INDEX_FIELDS)
        rows: list[IndexRow] = []
        for raw_row in reader:
            rows.append({key: _index_value(value) for key, value in raw_row.items() if key is not None})
    return fieldnames, rows


def _index_row_for_run(rows: list[IndexRow], run_id: str, results_dir: Path) -> IndexRow:
    for idx, row in enumerate(rows):
        if row.get("run_id") == run_id:
            replacement = dict(row)
            rows[idx] = replacement
            return replacement

    replacement = {"run_id": run_id, "results_dir": str(results_dir)}
    rows.append(replacement)
    return replacement


def _mark_status(status_file: Path, exit_status: int, message: str) -> dict[str, Any]:
    payload = _load_json(status_file)
    if payload.get("status") in {"completed", "failed", "interrupted"} and payload.get("finished_at_utc"):
        return payload

    results_dir = status_file.parent
    run_id = str(payload.get("run_id") or results_dir.name)
    payload.update(
        {
            "schema_version": int(payload.get("schema_version") or 1),
            "run_id": run_id,
            "status": "failed",
            "finished_at_utc": _utc_now(),
            "exit_status": int(exit_status),
            "failures": int(payload.get("failures") or 1),
            "message": message,
            "results_dir": str(payload.get("results_dir") or results_dir),
        }
    )
    _write_json(status_file, payload)
    return payload


def _update_family_index(status_file: Path, status_payload: dict[str, Any]) -> None:
    results_dir = status_file.parent
    family_root = _family_root(results_dir)
    csv_path = family_root / "runs_index.csv"
    fieldnames, rows = _load_index_rows(csv_path)

    for field in DEFAULT_INDEX_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)

    run_id = str(status_payload.get("run_id") or results_dir.name)
    replacement = _index_row_for_run(rows, run_id, results_dir)

    replacement.update(
        {
            "status": _index_value(status_payload.get("status")),
            "started_at_utc": _index_value(status_payload.get("started_at_utc")),
            "finished_at_utc": _index_value(status_payload.get("finished_at_utc")),
            "exit_status": _index_value(status_payload.get("exit_status")),
            "failures": _index_value(status_payload.get("failures")),
            "results_dir": _index_value(status_payload.get("results_dir") or str(results_dir)),
        }
    )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    event = {field: replacement.get(field, "") for field in fieldnames}
    event["event_recorded_at_utc"] = _utc_now()
    with open(family_root / "runs_index_events.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--exit-status", type=int, required=True)
    parser.add_argument("--message", required=True)
    args = parser.parse_args()

    status_payload = _mark_status(args.status_file, args.exit_status, args.message)
    if status_payload.get("status") == "failed":
        _update_family_index(args.status_file, status_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
