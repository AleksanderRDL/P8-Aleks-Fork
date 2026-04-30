"""Typed query F1 metrics and aggregate scoring. See src/evaluation/README.md for details."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Hashable, Iterable, Mapping, Sequence

import torch


def f1_score(r_o: set[Hashable], r_s: set[Hashable]) -> float:
    """Compute F1 agreement between original and simplified query answer sets."""
    if not r_o and not r_s:
        return 1.0
    if not r_o or not r_s:
        return 0.0

    intersection = len(r_s & r_o)
    if intersection == 0:
        return 0.0

    precision = intersection / len(r_s)
    recall = intersection / len(r_o)
    denom = precision + recall
    if denom <= 0.0:
        return 0.0
    return float(2.0 * precision * recall / denom)


def _label_items(labels: Mapping[int, int] | Sequence[int] | torch.Tensor) -> Iterable[tuple[int, int]]:
    """Yield stable trajectory ID and cluster label pairs."""
    if isinstance(labels, Mapping):
        for traj_id, label in labels.items():
            yield int(traj_id), int(label)
        return
    values = labels.tolist() if isinstance(labels, torch.Tensor) else labels
    for traj_id, label in enumerate(values):
        yield int(traj_id), int(label)


def _co_membership_pairs(labels: Mapping[int, int] | Sequence[int] | torch.Tensor) -> set[tuple[int, int]]:
    """Build unordered same-cluster trajectory-pair set, excluding noise label -1."""
    clusters: dict[int, list[int]] = {}
    for traj_id, label in _label_items(labels):
        if label == -1:
            continue
        clusters.setdefault(label, []).append(traj_id)

    pairs: set[tuple[int, int]] = set()
    for members in clusters.values():
        for i, j in combinations(sorted(members), 2):
            pairs.add((i, j))
    return pairs


def clustering_f1(
    labels_o: Mapping[int, int] | Sequence[int] | torch.Tensor,
    labels_s: Mapping[int, int] | Sequence[int] | torch.Tensor,
) -> float:
    """Compute F1 over clustering co-membership pair sets."""
    return f1_score(_co_membership_pairs(labels_o), _co_membership_pairs(labels_s))


@dataclass
class MethodEvaluation:
    """Container for method-level aggregate and per-type F1 scores. See src/evaluation/README.md for details."""

    aggregate_f1: float
    per_type_f1: dict[str, float]
    compression_ratio: float
    latency_ms: float
    retained_mask: torch.Tensor | None = None
