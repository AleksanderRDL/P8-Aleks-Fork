"""Tests for multi-file AIS CSV combination."""

from __future__ import annotations

import pandas as pd

from src.data.combine_days import MMSI_OFFSET_PER_FILE, combine


def _write_day(path, timestamp: str) -> None:
    pd.DataFrame(
        {
            "MMSI": [123456789, 123456789],
            "# Timestamp": [timestamp, timestamp],
            "Latitude": [55.0, 55.1],
            "Longitude": [12.0, 12.1],
            "SOG": [10.0, 10.5],
            "COG": [90.0, 91.0],
        }
    ).to_csv(path, index=False)


def test_combine_days_preserves_mmsi_by_default(tmp_path) -> None:
    day1 = tmp_path / "day1.csv"
    day2 = tmp_path / "day2.csv"
    out = tmp_path / "combined.csv"
    _write_day(day1, "2026-01-01T00:00:00")
    _write_day(day2, "2026-01-02T00:00:00")

    combine([day1, day2], out)

    combined = pd.read_csv(out)
    assert combined["MMSI"].nunique() == 1
    assert set(combined["MMSI"]) == {123456789}


def test_combine_days_can_offset_mmsi_per_file(tmp_path) -> None:
    day1 = tmp_path / "day1.csv"
    day2 = tmp_path / "day2.csv"
    out = tmp_path / "combined.csv"
    _write_day(day1, "2026-01-01T00:00:00")
    _write_day(day2, "2026-01-02T00:00:00")

    combine([day1, day2], out, offset_mmsi_per_file=True)

    combined = pd.read_csv(out)
    assert combined["MMSI"].nunique() == 2
    assert set(combined["MMSI"]) == {123456789, 123456789 + MMSI_OFFSET_PER_FILE}
