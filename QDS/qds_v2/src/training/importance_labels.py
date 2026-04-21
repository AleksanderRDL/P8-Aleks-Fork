"""Typed leave-one-out style label construction with labelled masks. See src/training/README.md for details."""

from __future__ import annotations

from typing import Any

import torch

from src.queries.query_executor import (
    execute_knn_query,
    execute_range_query,
)
from src.queries.query_types import QUERY_NAME_TO_ID, NUM_QUERY_TYPES


def _box_mask(points: torch.Tensor, params: dict[str, float]) -> torch.Tensor:
    """Build spatiotemporal box mask. See src/training/README.md for details."""
    return (
        (points[:, 1] >= params["lat_min"])
        & (points[:, 1] <= params["lat_max"])
        & (points[:, 2] >= params["lon_min"])
        & (points[:, 2] <= params["lon_max"])
        & (points[:, 0] >= params["t_start"])
        & (points[:, 0] <= params["t_end"])
    )


def _interpolate_unlabelled_within_trajectory(
    labels: torch.Tensor,
    labelled_mask: torch.Tensor,
    boundaries: list[tuple[int, int]],
) -> torch.Tensor:
    """Fill unlabelled entries using nearest measured label in each trajectory. See src/training/README.md for details."""
    out = labels.clone()
    for t in range(labels.shape[1]):
        for start, end in boundaries:
            idx = torch.arange(start, end)
            measured = idx[labelled_mask[start:end, t]]
            if measured.numel() == 0:
                continue
            missing = idx[~labelled_mask[start:end, t]]
            if missing.numel() == 0:
                continue
            for mi in missing.tolist():
                nearest = measured[torch.argmin(torch.abs(measured - mi))]
                out[mi, t] = out[nearest, t]
    return out


def compute_typed_importance_labels(
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, Any]],
    seed: int,
    similarity_sample_rate: float = 0.70,
    clustering_sample_rate: float = 0.70,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-point per-type labels with a measured-points mask. See src/training/README.md for details."""
    n = points.shape[0]
    labels = torch.zeros((n, NUM_QUERY_TYPES), dtype=torch.float32)
    labelled_mask = torch.zeros((n, NUM_QUERY_TYPES), dtype=torch.bool)

    g = torch.Generator().manual_seed(int(seed))

    for q in typed_queries:
        qtype = q["type"]
        t_idx = QUERY_NAME_TO_ID[qtype]
        params = q["params"]

        if qtype == "range":
            base = execute_range_query(points, params)
            denom = max(abs(base), 1e-6)
            mask = _box_mask(points, params)
            contrib = points[:, 3] / denom
            labels[mask, t_idx] += torch.abs(contrib[mask])
            labelled_mask[mask, t_idx] = True

        elif qtype == "knn":
            ids = execute_knn_query(points, params)
            if ids:
                idx = torch.tensor(sorted(list(ids)), dtype=torch.long)
                labels[idx, t_idx] += 1.0 / max(1, len(ids))
                labelled_mask[idx, t_idx] = True

        elif qtype == "similarity":
            q_mask = (
                (points[:, 0] >= params["t_start"]) & (points[:, 0] <= params["t_end"]) &
                (torch.sqrt((points[:, 1] - params["lat_query_centroid"]) ** 2 + (points[:, 2] - params["lon_query_centroid"]) ** 2) <= params["radius"])
            )
            candidate = torch.where(q_mask)[0]
            if candidate.numel() == 0:
                continue
            take = max(1, int(similarity_sample_rate * candidate.numel()))
            perm = torch.randperm(candidate.numel(), generator=g)[:take]
            measured = candidate[perm]
            ref = torch.tensor(q.get("reference", []), dtype=torch.float32)
            if ref.numel() == 0:
                continue
            center = ref[:, 1:3].mean(dim=0)
            d = torch.norm(points[measured][:, 1:3] - center, dim=1)
            score = 1.0 / (1.0 + d)
            labels[measured, t_idx] += score
            labelled_mask[measured, t_idx] = True

        elif qtype == "clustering":
            mask = _box_mask(points, params)
            candidate = torch.where(mask)[0]
            if candidate.numel() == 0:
                continue
            take = max(1, int(clustering_sample_rate * candidate.numel()))
            perm = torch.randperm(candidate.numel(), generator=g)[:take]
            measured = candidate[perm]

            # Score = speed normalized by global maximum so the label is a
            # direct monotone function of the speed feature.  High-speed ships
            # are behavioural outliers that define distinct DBSCAN clusters;
            # using global normalization means the model can learn this mapping
            # from the speed feature alone without needing box-relative context.
            global_max_speed = points[:, 3].max() + 1e-6
            score = points[measured, 3] / global_max_speed
            labels[measured, t_idx] += score
            labelled_mask[measured, t_idx] = True

    labels = _interpolate_unlabelled_within_trajectory(labels, labelled_mask, boundaries)

    # Anchor all types lightly on speed so every head has a smooth baseline
    # signal even for queries that only hit a small fraction of points.
    # We keep a small 10% blend — enough for gradient stability but not so
    # much that it drowns out the type-specific signal.
    speed_col = points[:, 3]
    speed_norm = (speed_col - speed_col.min()) / (speed_col.max() - speed_col.min() + 1e-6)
    for t in range(NUM_QUERY_TYPES):
        labels[:, t] = 0.90 * labels[:, t] + 0.10 * speed_norm
        # Extend labelled mask to cover all points that have speed (i.e. all of them)
        # so every point participates in the MSE loss with at least the speed component.
        labelled_mask[:, t] = True

    for t in range(NUM_QUERY_TYPES):
        col = labels[:, t]
        if col.max() > col.min():
            labels[:, t] = (col - col.min()) / (col.max() - col.min())

    return labels, labelled_mask
