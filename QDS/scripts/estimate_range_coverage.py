#!/usr/bin/env python3
"""Estimate sampled range-query coverage for benchmark query-count calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import cast

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ais_loader import load_ais_csv
from src.data.trajectory_cache import load_or_build_ais_cache
from src.queries.coverage_estimator import RangeCoverageEstimate, best_query_count, estimate_range_coverage


def _parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _resolve_csv_paths(path: str, max_files: int) -> list[Path]:
    source = Path(path)
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(path)
    files = sorted(p for p in source.glob("*.csv") if p.is_file())
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No CSV files found in {source}")
    return files


def _load_trajectories(
    csv_path: Path,
    *,
    cache_dir: str | None,
    refresh_cache: bool,
    min_points_per_segment: int,
    max_points_per_segment: int | None,
    max_time_gap_seconds: float | None,
    max_segments: int | None,
) -> list[torch.Tensor]:
    if cache_dir:
        return load_or_build_ais_cache(
            str(csv_path),
            cache_dir=cache_dir,
            refresh_cache=refresh_cache,
            min_points_per_segment=min_points_per_segment,
            max_points_per_segment=max_points_per_segment,
            max_time_gap_seconds=max_time_gap_seconds,
            max_segments=max_segments,
        ).trajectories
    loaded = load_ais_csv(
        str(csv_path),
        min_points_per_segment=min_points_per_segment,
        max_points_per_segment=max_points_per_segment,
        max_time_gap_seconds=max_time_gap_seconds,
        max_segments=max_segments,
    )
    return cast(list[torch.Tensor], loaded)


def _print_table(rows: list[RangeCoverageEstimate], target: float | None) -> None:
    print("source\tqueries\tseed\tstride\tsampled_traj\tsampled_points\tcoverage\tcovered_points")
    for row in rows:
        print(
            f"{Path(row.source or '').name}\t"
            f"{row.query_count}\t"
            f"{row.seed}\t"
            f"{row.sample_stride}\t"
            f"{row.sampled_trajectories}\t"
            f"{row.sampled_points}\t"
            f"{100.0 * row.coverage_fraction:.2f}%\t"
            f"{row.covered_points}/{row.total_points}"
        )
    if target is not None:
        by_source: dict[str, list[RangeCoverageEstimate]] = {}
        for row in rows:
            by_source.setdefault(row.source or "", []).append(row)
        print("\nclosest_to_target")
        for source, source_rows in by_source.items():
            best = best_query_count(source_rows, target)
            print(
                f"{Path(source).name}\tqueries={best.query_count}\tseed={best.seed}\t"
                f"coverage={100.0 * best.coverage_fraction:.2f}%\t"
                f"target={100.0 * target:.2f}%"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv_path", required=True, help="Cleaned AIS CSV file or directory of cleaned CSVs.")
    parser.add_argument("--max_files", type=int, default=2, help="When csv_path is a directory, estimate this many sorted CSVs.")
    parser.add_argument("--cache_dir", default=None, help="Optional AIS cache directory.")
    parser.add_argument("--refresh_cache", action="store_true", help="Rebuild AIS cache before estimating.")
    parser.add_argument("--query_counts", default="80,384,512,640,1024,2048", help="Comma-separated query counts to test.")
    parser.add_argument("--seeds", default="42", help="Comma-separated query generation seeds.")
    parser.add_argument("--sample_stride", type=int, default=20, help="Use every Nth trajectory for fast estimation.")
    parser.add_argument("--target_coverage", type=float, default=0.20, help="Target coverage fraction used by generation.")
    parser.add_argument("--range_spatial_fraction", type=float, default=0.08)
    parser.add_argument("--range_time_fraction", type=float, default=0.15)
    parser.add_argument("--range_spatial_km", type=float, default=2.2)
    parser.add_argument("--range_time_hours", type=float, default=5.0)
    parser.add_argument("--range_footprint_jitter", type=float, default=0.0)
    parser.add_argument("--min_points_per_segment", type=int, default=4)
    parser.add_argument("--max_points_per_segment", type=int, default=None)
    parser.add_argument("--max_time_gap_seconds", type=float, default=3600.0)
    parser.add_argument("--max_segments", type=int, default=None)
    parser.add_argument("--output_json", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    query_counts = _parse_int_list(args.query_counts)
    seeds = _parse_int_list(args.seeds)
    all_rows: list[RangeCoverageEstimate] = []
    for csv_path in _resolve_csv_paths(args.csv_path, int(args.max_files)):
        trajectories = _load_trajectories(
            csv_path,
            cache_dir=args.cache_dir,
            refresh_cache=bool(args.refresh_cache),
            min_points_per_segment=int(args.min_points_per_segment),
            max_points_per_segment=args.max_points_per_segment,
            max_time_gap_seconds=args.max_time_gap_seconds,
            max_segments=args.max_segments,
        )
        all_rows.extend(
            estimate_range_coverage(
                trajectories=trajectories,
                query_counts=query_counts,
                seeds=seeds,
                source=str(csv_path),
                sample_stride=int(args.sample_stride),
                target_coverage=args.target_coverage,
                range_spatial_fraction=float(args.range_spatial_fraction),
                range_time_fraction=float(args.range_time_fraction),
                range_spatial_km=args.range_spatial_km,
                range_time_hours=args.range_time_hours,
                range_footprint_jitter=float(args.range_footprint_jitter),
            )
        )

    _print_table(all_rows, args.target_coverage)
    if args.output_json:
        target = float(args.target_coverage) if args.target_coverage is not None else None
        best_rows: list[dict[str, int | float | str | None]] = []
        if target is not None:
            sources = sorted({row.source or "" for row in all_rows})
            for source in sources:
                source_rows = [row for row in all_rows if (row.source or "") == source]
                best_rows.append(best_query_count(source_rows, target).to_dict())
        payload = {
            "rows": [row.to_dict() for row in all_rows],
            "best_by_source": best_rows,
        }
        Path(args.output_json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
