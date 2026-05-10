"""Tests for AIS CSV segmentation and load-audit behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.ais_loader import load_ais_csv


def _write_csv(path: Path, rows: list[tuple[int, float, float, float, float, float]]) -> None:
    lines = ["MMSI,# Timestamp,Latitude,Longitude,SOG,COG"]
    lines.extend(",".join(str(value) for value in row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_ais_csv_splits_mmsi_tracks_on_time_gaps(tmp_path: Path) -> None:
    csv_path = tmp_path / "gap.csv"
    _write_csv(
        csv_path,
        [
            (100, 0.0, 55.0, 12.0, 7.0, 10.0),
            (100, 60.0, 55.1, 12.1, 7.1, 11.0),
            (100, 120.0, 55.2, 12.2, 7.2, 12.0),
            (100, 5000.0, 56.0, 13.0, 8.0, 20.0),
            (100, 5060.0, 56.1, 13.1, 8.1, 21.0),
            (100, 5120.0, 56.2, 13.2, 8.2, 22.0),
            (200, 0.0, 57.0, 14.0, 9.0, 30.0),
            (200, 60.0, 57.1, 14.1, 9.1, 31.0),
        ],
    )

    trajectories, mmsis, audit = load_ais_csv(
        str(csv_path),
        min_points_per_segment=3,
        max_time_gap_seconds=3600.0,
        return_mmsis=True,
        return_audit=True,
    )

    assert len(trajectories) == 2
    assert mmsis == [100, 100]
    assert [int(traj.shape[0]) for traj in trajectories] == [3, 3]
    assert all(float(traj[0, 5].item()) == 1.0 and float(traj[-1, 6].item()) == 1.0 for traj in trajectories)
    assert audit.time_gap_over_threshold_count == 1
    assert audit.dropped_short_segments == 1
    assert audit.output_segment_count == 2
    assert audit.output_point_count == 6
    assert audit.segment_length_stats["p50"] == pytest.approx(3.0)


def test_load_ais_csv_can_disable_time_gap_segmentation(tmp_path: Path) -> None:
    csv_path = tmp_path / "no_split.csv"
    _write_csv(
        csv_path,
        [
            (100, 0.0, 55.0, 12.0, 7.0, 10.0),
            (100, 60.0, 55.1, 12.1, 7.1, 11.0),
            (100, 120.0, 55.2, 12.2, 7.2, 12.0),
            (100, 5000.0, 56.0, 13.0, 8.0, 20.0),
            (100, 5060.0, 56.1, 13.1, 8.1, 21.0),
            (100, 5120.0, 56.2, 13.2, 8.2, 22.0),
        ],
    )

    trajectories, audit = load_ais_csv(
        str(csv_path),
        min_points_per_segment=3,
        max_time_gap_seconds=None,
        return_mmsis=False,
        return_audit=True,
    )

    assert len(trajectories) == 1
    assert int(trajectories[0].shape[0]) == 6
    assert audit.time_gap_over_threshold_count == 0


def test_load_ais_csv_audit_counts_invalid_rows_duplicates_and_downsampling(tmp_path: Path) -> None:
    csv_path = tmp_path / "audit.csv"
    _write_csv(
        csv_path,
        [
            (100, 0.0, 55.0, 12.0, 7.0, 10.0),
            (100, 0.0, 55.1, 12.1, 7.1, 11.0),
            (100, 60.0, 55.2, 12.2, 7.2, 511.0),
            (100, 120.0, 91.0, 12.3, 7.3, 13.0),
            (100, 180.0, 55.4, 12.4, 7.4, 14.0),
            (100, 240.0, 55.5, 12.5, 7.5, 15.0),
            (100, 300.0, 55.6, 12.6, 7.6, 16.0),
        ],
    )

    trajectories, audit = load_ais_csv(
        str(csv_path),
        min_points_per_segment=2,
        max_points_per_segment=3,
        return_mmsis=False,
        return_audit=True,
    )

    assert len(trajectories) == 1
    assert int(trajectories[0].shape[0]) == 3
    assert audit.rows_loaded == 7
    assert audit.rows_after_cleaning == 5
    assert audit.rows_dropped_invalid == 2
    assert audit.invalid_heading_rows == 1
    assert audit.invalid_lat_rows == 1
    assert audit.duplicate_timestamp_rows == 2
    assert audit.downsampled_segments == 1
