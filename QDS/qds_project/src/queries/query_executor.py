"""Spatiotemporal range query execution against AIS point clouds. See src/queries/README.md."""

from __future__ import annotations

import torch
from torch import Tensor


def run_query(points: Tensor, query: Tensor) -> Tensor:
    """Run a single spatiotemporal range query; returns sum of speed for matching points."""
    lat_min, lat_max, lon_min, lon_max, time_start, time_end = (
        query[0], query[1], query[2], query[3], query[4], query[5],
    )

    mask = (
        (points[:, 1] >= lat_min)  & (points[:, 1] <= lat_max) &
        (points[:, 2] >= lon_min)  & (points[:, 2] <= lon_max) &
        (points[:, 0] >= time_start) & (points[:, 0] <= time_end)
    )

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

        mask = (
            (chunk[:, None, 1] >= queries[None, :, 0]) &
            (chunk[:, None, 1] <= queries[None, :, 1]) &
            (chunk[:, None, 2] >= queries[None, :, 2]) &
            (chunk[:, None, 2] <= queries[None, :, 3]) &
            (chunk[:, None, 0] >= queries[None, :, 4]) &
            (chunk[:, None, 0] <= queries[None, :, 5])
        )

        results += (mask.float() * chunk[:, 3].unsqueeze(1)).sum(dim=0)

    return results
