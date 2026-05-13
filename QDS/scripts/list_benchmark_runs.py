#!/usr/bin/env python3
"""Print a compact table from a QDS benchmark-family runs_index.csv file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_FAMILY = Path("artifacts/benchmarks/range_testing_baseline")


def _short(value: str | None, width: int) -> str:
    text = "" if value is None else str(value)
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "~"


def _format_f1(value: str | None) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.4f}"
    except ValueError:
        return str(value)


def _format_timestamp(value: str | None) -> str:
    if not value:
        return ""
    return str(value)[:19]


def _load_rows(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _sort_key(row: dict[str, str]) -> tuple[str, str]:
    return (row.get("started_at_utc") or "", row.get("run_id") or "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        type=Path,
        default=DEFAULT_FAMILY,
        help=f"Benchmark family directory. Default: {DEFAULT_FAMILY}",
    )
    parser.add_argument("--limit", type=int, default=20, help="Rows to show. Default: 20.")
    parser.add_argument("--all", action="store_true", help="Show all rows.")
    args = parser.parse_args()

    index_path = args.family / "runs_index.csv"
    if not index_path.exists():
        print(f"No runs index found: {index_path}")
        print("Start a benchmark first, or pass --family to another benchmark family.")
        return 0

    rows = sorted(_load_rows(index_path), key=_sort_key, reverse=True)
    if not args.all:
        rows = rows[: max(args.limit, 0)]

    headers = ["status", "started", "run_id", "profile", "best_mlqds_f1", "best_label", "results_dir"]
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                _short(row.get("status"), 11),
                _format_timestamp(row.get("started_at_utc")),
                _short(row.get("run_id"), 40),
                _short(row.get("profile"), 8),
                _format_f1(row.get("best_mlqds_f1")),
                _short(row.get("best_mlqds_run_label"), 16),
                _short(row.get("results_dir"), 72),
            ]
        )

    widths = [len(header) for header in headers]
    for table_row in table_rows:
        for idx, cell in enumerate(table_row):
            widths[idx] = max(widths[idx], len(cell))

    def render(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(cells)).rstrip()

    print(f"Benchmark family: {args.family}")
    print(render(headers))
    print(render(["-" * width for width in widths]))
    for table_row in table_rows:
        print(render(table_row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
