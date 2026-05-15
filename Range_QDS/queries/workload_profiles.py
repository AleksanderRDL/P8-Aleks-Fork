"""Versioned range workload profiles.

The query-driven rework treats the future range-query distribution as a product
object, not as loose benchmark knobs.  ``range_workload_v1`` is the first
concrete in-distribution profile for final candidates.  Runs without an explicit
profile remain legacy diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RangeWorkloadProfile:
    profile_id: str
    version: int
    anchor_family_weights: dict[str, float]
    footprint_family_weights: dict[str, float]
    footprint_families: dict[str, dict[str, Any]]
    target_coverage: float | None
    max_coverage_overshoot: float | None
    time_domain_mode: str = "anchor_day"
    query_count_mode: str = "calibrated_to_coverage"
    min_points_per_query: int = 3
    min_trajectories_per_query: int = 1
    max_near_duplicate_hitset_jaccard: float = 0.65
    max_empty_query_fraction: float = 0.0
    max_near_duplicate_fraction: float = 0.05
    max_broad_query_fraction: float = 0.05
    max_attempt_multiplier: int = 20
    final_success_allowed: bool = False


LEGACY_GENERATOR_PROFILE = RangeWorkloadProfile(
    profile_id="legacy_generator",
    version=0,
    anchor_family_weights={},
    footprint_family_weights={},
    footprint_families={},
    target_coverage=None,
    max_coverage_overshoot=None,
    time_domain_mode="dataset",
    query_count_mode="legacy_fixed_or_target_coverage",
    min_points_per_query=0,
    min_trajectories_per_query=0,
    max_near_duplicate_hitset_jaccard=0.85,
    max_empty_query_fraction=1.0,
    max_near_duplicate_fraction=1.0,
    max_broad_query_fraction=1.0,
    max_attempt_multiplier=50,
    final_success_allowed=False,
)

RANGE_WORKLOAD_V1_PROFILE = RangeWorkloadProfile(
    profile_id="range_workload_v1",
    version=1,
    anchor_family_weights={
        "density_route": 0.40,
        "boundary_entry_exit": 0.20,
        "crossing_turn_change": 0.15,
        "port_or_approach_zone": 0.15,
        "sparse_background_control": 0.10,
    },
    footprint_family_weights={
        "small_local": 0.25,
        "medium_operational": 0.45,
        "large_context": 0.20,
        "route_corridor_like": 0.10,
    },
    footprint_families={
        "small_local": {
            "spatial_radius_km": 1.0,
            "time_half_window_hours": 1.0,
            "elongation_allowed": False,
        },
        "medium_operational": {
            "spatial_radius_km": 2.2,
            "time_half_window_hours": 5.0,
            "elongation_allowed": False,
        },
        "large_context": {
            "spatial_radius_km": 4.5,
            "time_half_window_hours": 10.0,
            "elongation_allowed": False,
        },
        "route_corridor_like": {
            "spatial_radius_km": 3.0,
            "time_half_window_hours": 6.0,
            "elongation_allowed": True,
        },
    },
    target_coverage=None,
    max_coverage_overshoot=None,
    time_domain_mode="anchor_day",
    query_count_mode="calibrated_to_coverage",
    min_points_per_query=3,
    min_trajectories_per_query=1,
    max_near_duplicate_hitset_jaccard=0.65,
    max_empty_query_fraction=0.0,
    max_near_duplicate_fraction=0.05,
    max_broad_query_fraction=0.05,
    max_attempt_multiplier=20,
    final_success_allowed=True,
)

PROFILE_BY_ID: dict[str, RangeWorkloadProfile] = {
    LEGACY_GENERATOR_PROFILE.profile_id: LEGACY_GENERATOR_PROFILE,
    RANGE_WORKLOAD_V1_PROFILE.profile_id: RANGE_WORKLOAD_V1_PROFILE,
}


def normalize_workload_profile_id(profile_id: str | None) -> str:
    """Return a canonical workload profile id."""
    normalized = str(profile_id or LEGACY_GENERATOR_PROFILE.profile_id).strip().lower()
    if not normalized:
        return LEGACY_GENERATOR_PROFILE.profile_id
    return normalized


def range_workload_profile(profile_id: str | None) -> RangeWorkloadProfile:
    """Return a known range workload profile."""
    normalized = normalize_workload_profile_id(profile_id)
    try:
        return PROFILE_BY_ID[normalized]
    except KeyError as exc:
        choices = ", ".join(sorted(PROFILE_BY_ID))
        raise ValueError(f"Unknown workload_profile_id={profile_id!r}; choices: {choices}.") from exc


def max_point_hit_fraction_for_coverage(target_coverage: float | None) -> float:
    """Return the v1 broad-query point-hit ceiling for a coverage target."""
    if target_coverage is None:
        return 0.05
    target = float(target_coverage)
    if target > 1.0 and target <= 100.0:
        target = target / 100.0
    if target <= 0.05:
        return 0.020
    if target <= 0.10:
        return 0.025
    if target <= 0.15:
        return 0.030
    return 0.050


def workload_profile_metadata(profile: RangeWorkloadProfile) -> dict[str, Any]:
    """Return JSON-safe profile metadata for artifacts."""
    return {
        "profile_id": profile.profile_id,
        "version": int(profile.version),
        "anchor_family_weights": dict(profile.anchor_family_weights),
        "footprint_family_weights": dict(profile.footprint_family_weights),
        "footprint_families": {
            name: dict(values)
            for name, values in profile.footprint_families.items()
        },
        "target_coverage": profile.target_coverage,
        "max_coverage_overshoot": profile.max_coverage_overshoot,
        "time_domain_mode": profile.time_domain_mode,
        "query_count_mode": profile.query_count_mode,
        "query_acceptance": {
            "min_points_per_query": int(profile.min_points_per_query),
            "min_trajectories_per_query": int(profile.min_trajectories_per_query),
            "max_near_duplicate_hitset_jaccard": float(profile.max_near_duplicate_hitset_jaccard),
            "max_empty_query_fraction": float(profile.max_empty_query_fraction),
            "max_near_duplicate_fraction": float(profile.max_near_duplicate_fraction),
            "max_broad_query_fraction": float(profile.max_broad_query_fraction),
            "max_attempt_multiplier": int(profile.max_attempt_multiplier),
        },
        "final_success_allowed": bool(profile.final_success_allowed),
    }
