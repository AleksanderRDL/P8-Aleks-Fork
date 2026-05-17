from __future__ import annotations

from collections import Counter

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from queries.query_generator import _profile_query_plan
from queries.workload_profiles import range_workload_profile

pytestmark = pytest.mark.property


FAST_PROPERTY_SETTINGS = settings(max_examples=50, deadline=None)


@given(
    requested=st.integers(min_value=1, max_value=512),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@FAST_PROPERTY_SETTINGS
def test_profile_query_plan_counts_sum_to_requested(requested: int, seed: int) -> None:
    profile = range_workload_profile("range_workload_v1")

    plan = _profile_query_plan(profile, requested_queries=requested, workload_seed=seed)

    assert plan["enabled"] is True
    assert len(plan["anchor_family_sequence"]) == requested
    assert len(plan["footprint_family_sequence"]) == requested
    assert sum(plan["anchor_family_planned_counts"].values()) == requested
    assert sum(plan["footprint_family_planned_counts"].values()) == requested
    assert set(plan["anchor_family_sequence"]).issubset(profile.anchor_family_weights)
    assert set(plan["footprint_family_sequence"]).issubset(profile.footprint_family_weights)


@given(
    requested=st.integers(min_value=32, max_value=512),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@FAST_PROPERTY_SETTINGS
def test_profile_query_plan_prefixes_keep_active_families_visible(
    requested: int,
    seed: int,
) -> None:
    profile = range_workload_profile("range_workload_v1")
    plan = _profile_query_plan(profile, requested_queries=requested, workload_seed=seed)

    anchor_prefix = Counter(plan["anchor_family_sequence"][:32])
    footprint_prefix = Counter(plan["footprint_family_sequence"][:32])

    for family, weight in profile.anchor_family_weights.items():
        if weight >= 0.10:
            assert anchor_prefix[family] > 0
    for family, weight in profile.footprint_family_weights.items():
        if weight >= 0.10:
            assert footprint_prefix[family] > 0
