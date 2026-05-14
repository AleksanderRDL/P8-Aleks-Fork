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

RANGE_USEFULNESS_WEIGHT_GROUPS: dict[str, tuple[str, ...]] = {
    "point_statistical_coverage": ("range_point_f1",),
    "ship_representation": ("range_ship_f1", "range_ship_coverage"),
    "boundary_context": ("range_entry_exit_f1", "range_crossing_f1"),
    "temporal_continuity": ("range_temporal_coverage", "range_gap_coverage"),
    "route_fidelity": ("range_turn_coverage", "range_shape_score"),
}


def range_usefulness_weight_summary() -> dict[str, object]:
    """Return component and grouped RangeUseful weights for run metadata."""
    group_totals = {
        group_name: float(sum(RANGE_USEFULNESS_WEIGHTS[component] for component in components))
        for group_name, components in RANGE_USEFULNESS_WEIGHT_GROUPS.items()
    }
    return {
        "schema_version": int(RANGE_USEFULNESS_SCHEMA_VERSION),
        "component_weights": dict(RANGE_USEFULNESS_WEIGHTS),
        "weight_groups": {
            group_name: list(components)
            for group_name, components in RANGE_USEFULNESS_WEIGHT_GROUPS.items()
        },
        "group_weights": group_totals,
        "total_weight": float(sum(RANGE_USEFULNESS_WEIGHTS.values())),
    }
