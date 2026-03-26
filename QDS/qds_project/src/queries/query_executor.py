"""Spatiotemporal range query execution against AIS point clouds. See src/queries/README.md."""

from __future__ import annotations

import torch
from torch import Tensor

from src.queries.query_masks import (
    spatiotemporal_inclusion_mask,
    sum_speed_by_query,
)


def run_query(points: Tensor, query: Tensor) -> Tensor:
    """Run a single spatiotemporal range query; returns sum of speed for matching points."""
    if query.dim() == 1:
        query = query.unsqueeze(0)
    mask = spatiotemporal_inclusion_mask(points, query).squeeze(1)
    return points[mask, 3].sum()


def run_queries(points: Tensor, queries: Tensor) -> Tensor:
    """Run all M queries in a vectorised, chunked manner against the point cloud."""
    n_points = points.shape[0]
    n_queries = queries.shape[0]

    if n_points == 0 or n_queries == 0:
        return torch.zeros(n_queries, dtype=points.dtype, device=points.device)

    # Chunk over points to avoid large [N, M] allocations while still vectorising
    # across queries in each chunk.
    chunk_size = 200_000
    results = torch.zeros(n_queries, dtype=points.dtype, device=points.device)

    for start in range(0, n_points, chunk_size):
        end = min(n_points, start + chunk_size)
        chunk = points[start:end]

        mask = spatiotemporal_inclusion_mask(chunk, queries)
        results += sum_speed_by_query(chunk, mask)

    return results
