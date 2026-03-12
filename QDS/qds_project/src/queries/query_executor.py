"""
query_executor.py

Execute spatiotemporal range queries against a point cloud of AIS data.

A query selects all points whose (lat, lon, time) fall inside the query
rectangle and returns the SUM of the speed column for those points.

Points tensor columns: [time=0, lat=1, lon=2, speed=3, heading=4]
Query tensor columns:  [lat_min=0, lat_max=1, lon_min=2, lon_max=3, time_start=4, time_end=5]
"""

from __future__ import annotations

import torch
from torch import Tensor


def run_query(points: Tensor, query: Tensor) -> Tensor:
    """Run a single spatiotemporal range query against a point cloud.

    Selects all points whose lat, lon, and time fall within the query
    bounds and returns the SUM of the speed values for those points.

    Args:
        points: Tensor of shape [N, 5] with columns
                [time, lat, lon, speed, heading].
        query:  Tensor of shape [6] with columns
                [lat_min, lat_max, lon_min, lon_max, time_start, time_end].

    Returns:
        Scalar tensor containing the sum of speed values for matching points.
    """
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
    """Run multiple spatiotemporal range queries against a point cloud.

    Args:
        points:  Tensor of shape [N, 5] with columns
                 [time, lat, lon, speed, heading].
        queries: Tensor of shape [M, 6] with columns
                 [lat_min, lat_max, lon_min, lon_max, time_start, time_end].

    Returns:
        Tensor of shape [M] with the sum-of-speed result for each query.
    """
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
