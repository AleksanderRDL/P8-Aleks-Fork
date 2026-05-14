"""Versioned range workload profile scaffolding.

The query-driven rework requires a stable workload profile such as range_workload_v1.
Low-level generator knobs are not enough for final acceptance.

Future rule: if ``workload_profile_id`` is absent, treat the run as
``legacy_generator`` and not final-success eligible.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RangeWorkloadProfile:
    profile_id: str
    version: int
    anchor_family_weights: dict[str, float]
    footprint_family_weights: dict[str, float]
    target_coverage: float | None
    max_coverage_overshoot: float | None
    final_success_allowed: bool = False


LEGACY_GENERATOR_PROFILE = RangeWorkloadProfile(
    profile_id="legacy_generator",
    version=0,
    anchor_family_weights={},
    footprint_family_weights={},
    target_coverage=None,
    max_coverage_overshoot=None,
    final_success_allowed=False,
)
