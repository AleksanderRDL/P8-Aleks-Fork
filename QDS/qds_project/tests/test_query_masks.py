"""Tests for low-level spatiotemporal masking helpers."""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.queries.query_masks import (
    spatial_inclusion_mask,
    spatiotemporal_inclusion_mask,
    sum_values_by_query,
    sum_speed_by_query,
)


def _make_points() -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 10.0, 10.0, 1.0, 0.0],
            [1.0, 11.0, 11.0, -2.0, 0.0],
            [2.0, 12.0, 12.0, 3.0, 0.0],
            [3.0, 13.0, 13.0, -4.0, 0.0],
        ],
        dtype=torch.float32,
    )


class TestInclusionMasks:
    def test_spatial_inclusion_mask_values(self):
        points = _make_points()
        queries = torch.tensor(
            [
                [9.0, 12.0, 9.0, 12.0, 0.0, 3.0],
                [13.0, 13.0, 13.0, 13.0, 0.0, 3.0],
            ],
            dtype=torch.float32,
        )

        mask = spatial_inclusion_mask(points, queries)
        expected = torch.tensor(
            [
                [True, False],
                [True, False],
                [True, False],
                [False, True],
            ]
        )
        assert torch.equal(mask, expected)

    def test_spatiotemporal_mask_matches_precomputed_spatial_mask(self):
        points = _make_points()
        queries = torch.tensor(
            [
                [9.0, 12.0, 9.0, 12.0, 1.0, 2.0],
                [13.0, 13.0, 13.0, 13.0, 3.0, 3.0],
            ],
            dtype=torch.float32,
        )

        precomputed_spatial = spatial_inclusion_mask(points, queries)
        direct = spatiotemporal_inclusion_mask(points, queries)
        cached = spatiotemporal_inclusion_mask(
            points,
            queries,
            spatial_mask=precomputed_spatial,
        )
        assert torch.equal(cached, direct)


class TestQueryAggregations:
    def test_sum_values_by_query_requires_1d_values(self):
        inclusion = torch.tensor([[True], [False]])
        values_2d = torch.ones((2, 1))
        with pytest.raises(ValueError, match="values must be 1-D"):
            sum_values_by_query(values_2d, inclusion)

    def test_sum_speed_by_query_absolute(self):
        points = _make_points()
        inclusion = torch.tensor(
            [
                [True, False],
                [True, True],
                [False, True],
                [False, True],
            ]
        )

        signed = sum_speed_by_query(points, inclusion, absolute=False)
        absolute = sum_speed_by_query(points, inclusion, absolute=True)

        assert torch.allclose(signed, torch.tensor([-1.0, -3.0]))
        assert torch.allclose(absolute, torch.tensor([3.0, 9.0]))
