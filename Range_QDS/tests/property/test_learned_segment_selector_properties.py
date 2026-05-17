from __future__ import annotations

# pyright: reportPrivateImportUsage=false
import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from simplification.learned_segment_budget import simplify_with_learned_segment_budget_v1_with_trace

pytestmark = pytest.mark.property


FAST_PROPERTY_SETTINGS = settings(max_examples=50, deadline=None)


def _expected_budget(boundaries: list[tuple[int, int]], compression_ratio: float) -> int:
    ratio = min(1.0, max(0.0, float(compression_ratio)))
    total = 0
    for start, end in boundaries:
        count = int(end - start)
        if count > 0:
            total += min(count, max(2, math.ceil(ratio * count)))
    return total


@given(
    trajectory_count=st.integers(min_value=1, max_value=12),
    points_per_trajectory=st.integers(min_value=4, max_value=96),
    compression_ratio=st.floats(
        min_value=0.01,
        max_value=0.50,
        allow_nan=False,
        allow_infinity=False,
    ),
    segment_size=st.integers(min_value=2, max_value=32),
)
@FAST_PROPERTY_SETTINGS
def test_learned_segment_selector_respects_budget_and_trace_accounting(
    trajectory_count: int,
    points_per_trajectory: int,
    compression_ratio: float,
    segment_size: int,
) -> None:
    total = trajectory_count * points_per_trajectory
    scores = torch.linspace(0.0, 1.0, steps=total, dtype=torch.float32)
    boundaries = [
        (idx * points_per_trajectory, (idx + 1) * points_per_trajectory)
        for idx in range(trajectory_count)
    ]

    mask, trace = simplify_with_learned_segment_budget_v1_with_trace(
        scores,
        boundaries,
        compression_ratio,
        segment_size=segment_size,
    )

    expected_budget = _expected_budget(boundaries, compression_ratio)
    retained_count = int(mask.sum().item())
    source_counts = (
        int(trace["skeleton_retained_count"])
        + int(trace["learned_controlled_retained_slots"])
        + int(trace["fallback_retained_count"])
        + int(trace["length_repair_retained_count"])
    )

    assert mask.dtype == torch.bool
    assert mask.shape == scores.shape
    assert retained_count <= expected_budget
    assert int(trace["total_budget_count"]) == expected_budget
    assert int(trace["retained_count"]) == retained_count
    assert source_counts == retained_count
