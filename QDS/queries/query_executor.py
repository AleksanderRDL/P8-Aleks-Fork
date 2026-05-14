"""Typed query execution against flattened or trajectory AIS data. See queries/README.md for details."""

from __future__ import annotations

import torch

from data.trajectory_index import (
    default_boundaries,
    trajectory_ids_intersecting_indices,
)
from queries.range_geometry import points_in_range_box


def execute_range_query(
    points: torch.Tensor,
    params: dict[str, float],
    boundaries: list[tuple[int, int]] | None = None,
) -> set[int]:
    """Execute a range query returning matching trajectory IDs. See queries/README.md for details."""
    mask = points_in_range_box(points, params)
    if not mask.any():
        return set()
    return trajectory_ids_intersecting_indices(torch.where(mask)[0], default_boundaries(points, boundaries))


def execute_typed_query(
    points: torch.Tensor,
    query: dict,
    boundaries: list[tuple[int, int]] | None = None,
) -> set[int]:
    """Execute one typed query and return type-specific result object. See queries/README.md for details."""
    query_type = query["type"]
    params = query["params"]
    if query_type == "range":
        return execute_range_query(points, params, boundaries)
    raise ValueError(f"Only range queries are supported; got query type: {query_type}")
