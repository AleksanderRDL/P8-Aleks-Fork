"""Typed query error metrics and aggregate scoring. See src/evaluation/README.md for details."""

from __future__ import annotations

from dataclasses import dataclass


def jaccard_distance(a: set[int], b: set[int]) -> float:
    """Compute Jaccard distance between two sets. See src/evaluation/README.md for details."""
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return 1.0 - inter / max(1, union)


def rbo_score(rank_a: list[int], rank_b: list[int], p: float = 0.9) -> float:
    """Compute rank-biased overlap for two ranked lists. See src/evaluation/README.md for details."""
    depth = max(len(rank_a), len(rank_b))
    if depth == 0:
        return 1.0
    seen_a: set[int] = set()
    seen_b: set[int] = set()
    total = 0.0
    for d in range(1, depth + 1):
        if d <= len(rank_a):
            seen_a.add(rank_a[d - 1])
        if d <= len(rank_b):
            seen_b.add(rank_b[d - 1])
        overlap = len(seen_a & seen_b)
        total += (1.0 - p) * (p ** (d - 1)) * (overlap / d)
    return total


def range_error(full: float, simplified: float) -> float:
    """Compute normalized range-query error. See src/evaluation/README.md for details."""
    return abs(simplified - full) / (abs(full) + 1e-6)


def knn_error(full: set[int], simplified: set[int]) -> float:
    """Compute kNN error as one minus Jaccard. See src/evaluation/README.md for details."""
    return jaccard_distance(full, simplified)


def similarity_error(full: list[int], simplified: list[int]) -> float:
    """Compute similarity-ranking error as one minus RBO. See src/evaluation/README.md for details."""
    return 1.0 - rbo_score(full, simplified)


def clustering_error(full: int, simplified: int) -> float:
    """Compute normalized clustering count error. See src/evaluation/README.md for details."""
    return abs(simplified - full) / max(1, full)


@dataclass
class MethodEvaluation:
    """Container for method-level aggregate and per-type errors. See src/evaluation/README.md for details."""

    aggregate_error: float
    per_type_error: dict[str, float]
    compression_ratio: float
    latency_ms: float
