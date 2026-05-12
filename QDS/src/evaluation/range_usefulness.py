"""Shared RangeUseful audit constants."""

from __future__ import annotations

RANGE_USEFULNESS_SCHEMA_VERSION = 7

RANGE_USEFULNESS_WEIGHTS: dict[str, float] = {
    "range_point_f1": 0.22,
    "range_ship_f1": 0.13,
    "range_ship_coverage": 0.13,
    "range_entry_exit_f1": 0.10,
    "range_crossing_f1": 0.10,
    "range_temporal_coverage": 0.10,
    "range_gap_coverage": 0.09,
    "range_turn_coverage": 0.07,
    "range_shape_score": 0.06,
}
