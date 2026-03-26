"""Evaluation metrics for AIS trajectory simplification. See src/evaluation/README.md."""

from __future__ import annotations

import time

from torch import Tensor

from src.queries.query_executor import run_queries


def query_error(
    original_points: Tensor,
    simplified_points: Tensor,
    queries: Tensor,
) -> float:
    """Compute the mean relative query error between original and simplified data."""
    if queries.shape[0] == 0:
        return 0.0

    original_results   = run_queries(original_points,   queries)  # [M]
    simplified_results = run_queries(simplified_points, queries)  # [M]

    denom = original_results.abs().clamp(min=1e-8)
    relative_errors = (original_results - simplified_results).abs() / denom
    return relative_errors.mean().item()


def compression_ratio(
    original_points: Tensor,
    simplified_points: Tensor,
) -> float:
    """Compute the fraction of points retained: K / N."""
    n_original   = original_points.shape[0]
    n_simplified = simplified_points.shape[0]
    if n_original == 0:
        return 1.0
    return n_simplified / n_original


def query_latency(points: Tensor, queries: Tensor) -> float:
    """Measure the average query execution time in seconds per query."""
    start = time.perf_counter()
    run_queries(points, queries)
    elapsed = time.perf_counter() - start
    n_queries = queries.shape[0]
    return elapsed / max(1, n_queries)
