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


def knn_error(full: list[float], simplified: list[float]) -> float:
    """Compute kNN error as relative distance excess.

    Both inputs are sorted-ascending lists of anchor-to-neighbour distances
    (length ``<= k``). Error is the average, clipped to ``[0, 1]``, of
    ``(d_simplified - d_full) / d_full`` over matched ranks. Missing slots
    in ``simplified`` (fewer survivors in the time window than ``k``) count
    as error ``1.0``. Zero-distance exact hits in ``full`` contribute 0 if
    ``simplified`` also hits zero, else 1.

    Intuitively: 0.0 means the simplified dataset preserves the exact
    neighbourhood; 1.0 means the k nearest surviving neighbours are at
    least twice as far as in the full dataset (or missing entirely).
    """
    n = len(full)
    if n == 0:
        return 0.0
    errors: list[float] = []
    for i in range(n):
        if i >= len(simplified):
            errors.append(1.0)
            continue
        d_full = float(full[i])
        d_simp = float(simplified[i])
        if d_full <= 1e-9:
            errors.append(0.0 if d_simp <= 1e-9 else 1.0)
        else:
            rel = (d_simp - d_full) / d_full
            errors.append(max(0.0, min(1.0, rel)))
    return sum(errors) / len(errors)


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
