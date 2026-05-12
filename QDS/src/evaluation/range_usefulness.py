"""Shared RangeUseful audit constants."""

from __future__ import annotations

RANGE_USEFULNESS_SCHEMA_VERSION = 4

RANGE_USEFULNESS_WEIGHTS: dict[str, float] = {
    "range_point_f1": 0.23,
    "range_ship_f1": 0.14,
    "range_ship_coverage": 0.14,
    "range_entry_exit_f1": 0.14,
    "range_temporal_coverage": 0.11,
    "range_gap_coverage": 0.10,
    "range_turn_coverage": 0.07,
    "range_shape_score": 0.07,
}
