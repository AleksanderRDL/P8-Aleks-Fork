"""Combine multiple preprocessed AIS CSVs into one training file.

The loader segments trajectories by MMSI plus temporal continuity, so combined
files preserve MMSIs and let the loader decide where segments start and end.

Usage:
    python -m src.data.combine_days \\
        --input /ceph/.../preprocessed_2026-02-05.csv \\
                /ceph/.../preprocessed_2026-02-06.csv \\
                /ceph/.../preprocessed_2026-02-07.csv \\
        --output /ceph/.../preprocessed_train_combined.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _mmsi_column(df: pd.DataFrame, path: Path) -> str:
    if "MMSI" in df.columns:
        return "MMSI"
    if "mmsi" in df.columns:
        return "mmsi"
    raise ValueError(f"no MMSI column in {path}; got {list(df.columns)}")


def combine(input_paths: list[Path], output_path: Path) -> None:
    if not input_paths:
        raise ValueError("at least one input CSV required")

    frames = []
    mmsi_col: str | None = None
    for p in input_paths:
        if not p.exists():
            raise FileNotFoundError(f"input CSV missing: {p}")
        df = pd.read_csv(p)
        current_mmsi_col = _mmsi_column(df, p)
        if mmsi_col is None:
            mmsi_col = current_mmsi_col
        elif current_mmsi_col != mmsi_col:
            raise ValueError(
                f"inconsistent MMSI column casing: first file used {mmsi_col!r}, "
                f"but {p} uses {current_mmsi_col!r}"
            )
        n_traj = df[mmsi_col].nunique()
        print(f"  loaded {p.name}: {len(df):>10,} rows, {n_traj:>6,} unique MMSIs", flush=True)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    if mmsi_col is None:
        raise RuntimeError("internal error: no MMSI column resolved")
    print(
        f"\nwrote {output_path}: {len(combined):,} rows, "
        f"{combined[mmsi_col].nunique():,} unique MMSIs",
        flush=True,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", nargs="+", required=True, help="paths to preprocessed CSVs (one per day)")
    p.add_argument("--output", required=True, help="combined CSV output path")
    args = p.parse_args()

    combine([Path(s) for s in args.input], Path(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
