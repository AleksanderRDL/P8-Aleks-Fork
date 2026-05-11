#!/usr/bin/env python3
"""Validate queued benchmark child args before launching tmux."""

from __future__ import annotations

import argparse
import contextlib
import io
import shlex
import sys
from pathlib import Path

QDS_ROOT = Path(__file__).resolve().parents[1]
if str(QDS_ROOT) not in sys.path:
    sys.path.insert(0, str(QDS_ROOT))

from src.experiments.experiment_cli import build_parser as build_child_parser


def _iter_plan_rows(path: Path) -> list[tuple[int, str, int, str]]:
    rows: list[tuple[int, str, int, str]] = []
    for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = raw_line.split("\t", 2)
        if len(parts) < 2:
            raise ValueError(f"{path}:{lineno}: expected run_id<TAB>seed<TAB>child_extra_args")
        run_id = parts[0].strip()
        raw_seed = parts[1].strip()
        extra_args = parts[2].strip() if len(parts) > 2 else ""
        if not run_id:
            raise ValueError(f"{path}:{lineno}: run_id must not be empty")
        try:
            seed = int(raw_seed)
        except ValueError as exc:
            raise ValueError(f"{path}:{lineno}: seed must be an integer, got {raw_seed!r}") from exc
        rows.append((lineno, run_id, seed, extra_args))
    return rows


def validate_plan(path: Path) -> list[str]:
    """Return user-facing validation errors for a benchmark queue plan."""
    errors: list[str] = []
    try:
        rows = _iter_plan_rows(path)
    except ValueError as exc:
        return [str(exc)]

    if not rows:
        return [f"{path}: no runnable queue rows found"]

    child_parser = build_child_parser()
    for lineno, run_id, _seed, extra_args in rows:
        try:
            tokens = shlex.split(extra_args)
        except ValueError as exc:
            errors.append(f"{path}:{lineno} {run_id}: cannot parse child_extra_args: {exc}")
            continue
        if not tokens:
            continue
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                child_parser.parse_args(tokens)
        except SystemExit as exc:
            errors.append(
                f"{path}:{lineno} {run_id}: child_extra_args are invalid for "
                f"run_ais_experiment: {extra_args!r} (argparse exit {exc.code})"
            )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate benchmark queue plan child extra args.")
    parser.add_argument("plan_file", type=Path)
    args = parser.parse_args(argv)

    errors = validate_plan(args.plan_file)
    if errors:
        print("[queue-plan] validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 2

    runnable = len(_iter_plan_rows(args.plan_file))
    print(f"[queue-plan] validation passed: {args.plan_file} ({runnable} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
