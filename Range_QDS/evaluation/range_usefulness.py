"""Shared RangeUsefulLegacy audit constants.

This aggregate is retained for diagnostics and artifact comparability. It is
not the final acceptance metric for the query-driven rework.
"""

from __future__ import annotations

from collections.abc import Mapping

RANGE_USEFULNESS_SCHEMA_VERSION = 7
RANGE_USEFULNESS_GAP_ABLATION_VERSION = 1
RANGE_USEFULNESS_FINAL_SUCCESS_ALLOWED = False
RANGE_USEFULNESS_LEGACY_REASON = (
    "Old RangeUseful/scalar-target diagnostic path. "
    "Not valid for query-driven rework acceptance."
)

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

RANGE_USEFULNESS_GAP_ABLATION_FIELDS: dict[str, str] = {
    "time": "range_usefulness_gap_time_score",
    "distance": "range_usefulness_gap_distance_score",
    "min": "range_usefulness_gap_min_score",
}


def _component_value(components: Mapping[str, float], key: str, default: float = 0.0) -> float:
    """Return a numeric component value from an audit component mapping."""
    value = components.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def range_usefulness_score_from_components(
    components: Mapping[str, float],
    *,
    gap_component: str = "range_gap_coverage",
) -> float:
    """Return the RangeUseful weighted aggregate for a component mapping.

    ``gap_component`` may replace the default count-normalized gap term for
    diagnostic ablations. The primary metric still uses ``range_gap_coverage``.
    """
    score = 0.0
    for component_name, weight in RANGE_USEFULNESS_WEIGHTS.items():
        source_name = gap_component if component_name == "range_gap_coverage" else component_name
        score += float(weight) * _component_value(components, source_name)
    return float(score)


def range_usefulness_gap_ablation_scores(components: Mapping[str, float]) -> dict[str, float | int]:
    """Return diagnostic RangeUseful scores with alternate gap definitions."""
    count_gap = _component_value(components, "range_gap_coverage")
    time_gap = _component_value(components, "range_gap_time_coverage", count_gap)
    distance_gap = _component_value(components, "range_gap_distance_coverage", count_gap)
    min_gap = min(time_gap, distance_gap)
    augmented = dict(components)
    augmented["range_gap_min_coverage"] = float(min_gap)
    return {
        "range_gap_min_coverage": float(min_gap),
        "range_usefulness_gap_time_score": range_usefulness_score_from_components(
            augmented,
            gap_component="range_gap_time_coverage",
        ),
        "range_usefulness_gap_distance_score": range_usefulness_score_from_components(
            augmented,
            gap_component="range_gap_distance_coverage",
        ),
        "range_usefulness_gap_min_score": range_usefulness_score_from_components(
            augmented,
            gap_component="range_gap_min_coverage",
        ),
        "range_usefulness_gap_ablation_version": int(RANGE_USEFULNESS_GAP_ABLATION_VERSION),
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
