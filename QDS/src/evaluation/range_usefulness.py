"""Shared RangeUseful audit constants."""

from __future__ import annotations

RANGE_USEFULNESS_SCHEMA_VERSION = 3

RANGE_USEFULNESS_WEIGHTS: dict[str, float] = {
    "range_point_f1": 0.25,
    "range_ship_f1": 0.15,
    "range_ship_coverage": 0.15,
    "range_entry_exit_f1": 0.15,
    "range_temporal_coverage": 0.12,
    "range_gap_coverage": 0.10,
    "range_shape_score": 0.08,
}
