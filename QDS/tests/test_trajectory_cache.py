"""Tests for Parquet-backed AIS trajectory caches."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from src.data.trajectory_cache import CACHE_COLUMNS, load_or_build_ais_cache


def _write_csv(path: Path) -> None:
    rows = [
        (100, 0.0, 55.0, 12.0, 7.0, 10.0),
        (100, 60.0, 55.1, 12.1, 7.1, 11.0),
        (100, 120.0, 55.2, 12.2, 7.2, 12.0),
        (100, 5000.0, 56.0, 13.0, 8.0, 20.0),
        (100, 5060.0, 56.1, 13.1, 8.1, 21.0),
        (100, 5120.0, 56.2, 13.2, 8.2, 22.0),
    ]
    lines = ["MMSI,# Timestamp,Latitude,Longitude,SOG,COG"]
    lines.extend(",".join(str(value) for value in row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_ais_cache_roundtrip_and_cache_hit(tmp_path: Path) -> None:
    csv_path = tmp_path / "ais.csv"
    cache_dir = tmp_path / "cache"
    _write_csv(csv_path)

    first = load_or_build_ais_cache(
        str(csv_path),
        str(cache_dir),
        min_points_per_segment=3,
        max_time_gap_seconds=3600.0,
    )

    assert first.cache_hit is False
    assert Path(first.manifest_path).exists()
    assert Path(first.parquet_path).exists()
    assert [int(t.shape[0]) for t in first.trajectories] == [3, 3]
    assert first.mmsis == [100, 100]
    assert first.cache_metadata() == {
        "cache_hit": False,
        "cache_dir": first.cache_dir,
        "manifest_path": first.manifest_path,
        "parquet_path": first.parquet_path,
    }

    manifest = json.loads(Path(first.manifest_path).read_text(encoding="utf-8"))
    assert manifest["audit"]["output_segment_count"] == 2
    frame = pd.read_parquet(first.parquet_path, engine="pyarrow")
    assert set(CACHE_COLUMNS).issubset(frame.columns)

    second = load_or_build_ais_cache(
        str(csv_path),
        str(cache_dir),
        min_points_per_segment=3,
        max_time_gap_seconds=3600.0,
    )

    assert second.cache_hit is True
    assert second.mmsis == first.mmsis
    assert second.audit.to_dict() == first.audit.to_dict()
    assert len(second.trajectories) == len(first.trajectories)
    for cached, original in zip(second.trajectories, first.trajectories):
        assert torch.equal(cached, original)


def test_ais_cache_refresh_rebuilds_matching_entry(tmp_path: Path) -> None:
    csv_path = tmp_path / "ais.csv"
    cache_dir = tmp_path / "cache"
    _write_csv(csv_path)

    first = load_or_build_ais_cache(
        str(csv_path),
        str(cache_dir),
        min_points_per_segment=3,
    )
    refreshed = load_or_build_ais_cache(
        str(csv_path),
        str(cache_dir),
        refresh_cache=True,
        min_points_per_segment=3,
    )

    assert first.cache_hit is False
    assert refreshed.cache_hit is False
    assert refreshed.manifest_path == first.manifest_path
    assert refreshed.parquet_path == first.parquet_path
    for cached, original in zip(refreshed.trajectories, first.trajectories):
        assert torch.equal(cached, original)
