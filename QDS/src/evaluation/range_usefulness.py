"""Shared RangeUseful audit constants."""

from __future__ import annotations

RANGE_USEFULNESS_SCHEMA_VERSION = 2

RANGE_USEFULNESS_WEIGHTS: dict[str, float] = {
    "range_point_f1": 0.30,
    "range_ship_f1": 0.20,
    "range_entry_exit_f1": 0.15,
    "range_temporal_coverage": 0.15,
    "range_gap_coverage": 0.10,
    "range_shape_score": 0.10,
}

