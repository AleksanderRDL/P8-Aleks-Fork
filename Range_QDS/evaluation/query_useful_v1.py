"""QueryUsefulV1 scoring.

This is the primary metric for query-driven, workload-blind Range-QDS
candidates.  It reuses retained-independent range audit support but changes the
aggregate: query-local point mass and behavior explanation dominate; global
geometry is only a light guardrail.
"""

QUERY_USEFUL_V1_SCHEMA_VERSION = 1

QUERY_USEFUL_V1_WEIGHTS: dict[str, float] = {
    "query_point_mass": 0.40,
    "query_local_behavior": 0.30,
    "ship_presence_and_coverage": 0.15,
    "boundary_and_event_evidence": 0.10,
    "global_sanity": 0.05,
}

QUERY_USEFUL_V1_COMPONENT_WEIGHTS: dict[str, float] = {
    "ship_balanced_query_point_recall": 0.18,
    "query_balanced_point_recall": 0.10,
    "query_point_mass_ratio": 0.07,
    "query_point_distribution_stability": 0.05,
    "query_local_interpolation_score": 0.10,
    "query_local_turn_change_coverage": 0.07,
    "query_local_speed_heading_coverage": 0.06,
    "query_local_shape_score": 0.05,
    "query_local_gap_continuity": 0.02,
    "ship_f1": 0.08,
    "ship_coverage": 0.05,
    "multi_point_ship_evidence": 0.02,
    "entry_exit_f1": 0.04,
    "crossing_f1": 0.03,
    "query_boundary_evidence": 0.03,
    "endpoint_or_skeleton_sanity": 0.02,
    "global_shape_guardrail_score": 0.02,
    "length_preservation_guardrail": 0.01,
}


def _component_value(components: dict[str, float], key: str, default: float = 0.0) -> float:
    """Return a finite component value clamped to [0, 1]."""
    try:
        value = float(components.get(key, default))
    except (TypeError, ValueError):
        value = float(default)
    if value != value:
        return float(default)
    return float(max(0.0, min(1.0, value)))


def _guardrail_from_sed(avg_sed_km: float | None) -> float:
    """Convert average SED into a soft global-shape guardrail score."""
    if avg_sed_km is None:
        return 1.0
    try:
        sed = max(0.0, float(avg_sed_km))
    except (TypeError, ValueError):
        return 1.0
    return float(1.0 / (1.0 + sed))


def query_useful_v1_components_from_range_audit(
    range_audit: dict[str, float],
    *,
    length_preservation: float | None = None,
    avg_sed_km: float | None = None,
    endpoint_sanity: float = 1.0,
) -> dict[str, float]:
    """Return QueryUsefulV1 component values from a RangeUseful audit payload."""
    range_point = _component_value(range_audit, "range_point_f1")
    ship_coverage = _component_value(range_audit, "range_ship_coverage")
    ship_f1 = _component_value(range_audit, "range_ship_f1")
    temporal = _component_value(range_audit, "range_temporal_coverage")
    gap_time = _component_value(range_audit, "range_gap_time_coverage", _component_value(range_audit, "range_gap_coverage"))
    gap_distance = _component_value(
        range_audit,
        "range_gap_distance_coverage",
        _component_value(range_audit, "range_gap_coverage"),
    )
    gap_min = min(gap_time, gap_distance)
    turn = _component_value(range_audit, "range_turn_coverage")
    shape = _component_value(range_audit, "range_shape_score")
    entry_exit = _component_value(range_audit, "range_entry_exit_f1")
    crossing = _component_value(range_audit, "range_crossing_f1")
    length_score = 1.0 if length_preservation is None else max(0.0, min(1.0, float(length_preservation)))
    global_shape = _guardrail_from_sed(avg_sed_km)

    return {
        "ship_balanced_query_point_recall": float(0.5 * range_point + 0.5 * ship_coverage),
        "query_balanced_point_recall": range_point,
        "query_point_mass_ratio": range_point,
        "query_point_distribution_stability": float(min(range_point, gap_min)),
        "query_local_interpolation_score": shape,
        "query_local_turn_change_coverage": turn,
        "query_local_speed_heading_coverage": float(0.5 * turn + 0.5 * temporal),
        "query_local_shape_score": shape,
        "query_local_gap_continuity": gap_min,
        "ship_f1": ship_f1,
        "ship_coverage": ship_coverage,
        "multi_point_ship_evidence": ship_coverage,
        "entry_exit_f1": entry_exit,
        "crossing_f1": crossing,
        "query_boundary_evidence": float(0.5 * entry_exit + 0.5 * crossing),
        "endpoint_or_skeleton_sanity": max(0.0, min(1.0, float(endpoint_sanity))),
        "global_shape_guardrail_score": global_shape,
        "length_preservation_guardrail": length_score,
    }


def query_useful_v1_score_from_components(components: dict[str, float]) -> float:
    """Return the weighted QueryUsefulV1 aggregate."""
    total = 0.0
    for component, weight in QUERY_USEFUL_V1_COMPONENT_WEIGHTS.items():
        total += float(weight) * _component_value(components, component)
    return float(total)


def query_useful_v1_from_range_audit(
    range_audit: dict[str, float],
    *,
    length_preservation: float | None = None,
    avg_sed_km: float | None = None,
    endpoint_sanity: float = 1.0,
) -> dict[str, object]:
    """Return QueryUsefulV1 score, components, and schema metadata."""
    components = query_useful_v1_components_from_range_audit(
        range_audit,
        length_preservation=length_preservation,
        avg_sed_km=avg_sed_km,
        endpoint_sanity=endpoint_sanity,
    )
    return {
        "query_useful_v1_schema_version": int(QUERY_USEFUL_V1_SCHEMA_VERSION),
        "query_useful_v1_score": query_useful_v1_score_from_components(components),
        "query_useful_v1_components": components,
        "query_useful_v1_component_weights": dict(QUERY_USEFUL_V1_COMPONENT_WEIGHTS),
        "query_useful_v1_group_weights": dict(QUERY_USEFUL_V1_WEIGHTS),
    }
