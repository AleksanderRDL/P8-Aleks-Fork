"""
metrics.py

Evaluation metrics for AIS trajectory simplification.

Metrics
-------
- query_error          : Mean relative query error between original and simplified.
- compression_ratio    : Fraction of points retained after simplification.
- query_latency        : Average query execution time in seconds.
"""

from __future__ import annotations

import time

import torch
from torch import Tensor

from src.queries.query_executor import run_queries


def query_error(
    original_points: Tensor,
    simplified_points: Tensor,
    queries: Tensor,
) -> float:
    """Compute the mean relative query error between original and simplified data.

    For each query, the relative error is:
        |result(original) - result(simplified)| / max(|result(original)|, ε)

    Args:
        original_points:   Tensor of shape [N, 5] — full point cloud.
        simplified_points: Tensor of shape [K, 5] — compressed point cloud.
        queries:           Tensor of shape [M, 6] — query workload.

    Returns:
        Mean relative query error as a float (lower is better).
    """
    original_results   = run_queries(original_points,   queries)  # [M]
    simplified_results = run_queries(simplified_points, queries)  # [M]

    denom = original_results.abs().clamp(min=1e-8)
    relative_errors = (original_results - simplified_results).abs() / denom
    return relative_errors.mean().item()


def compression_ratio(
    original_points: Tensor,
    simplified_points: Tensor,
) -> float:
    """Compute the fraction of points retained after simplification.

    Args:
        original_points:   Tensor of shape [N, 5].
        simplified_points: Tensor of shape [K, 5].

    Returns:
        Compression ratio K / N in (0, 1].  A value of 1.0 means no
        compression; smaller values indicate more aggressive compression.
    """
    n_original   = original_points.shape[0]
    n_simplified = simplified_points.shape[0]
    if n_original == 0:
        return 1.0
    return n_simplified / n_original


def query_latency(points: Tensor, queries: Tensor) -> float:
    """Measure the average query execution time.

    Runs all queries against the point cloud and returns the mean time
    per query in seconds.

    Args:
        points:  Tensor of shape [N, 5].
        queries: Tensor of shape [M, 6].

    Returns:
        Mean query execution time in seconds (lower is better).
    """
    start = time.perf_counter()
    run_queries(points, queries)
    elapsed = time.perf_counter() - start
    n_queries = queries.shape[0]
    return elapsed / max(1, n_queries)
