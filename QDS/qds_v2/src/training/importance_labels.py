"""Typed F1-contribution label construction. See src/training/README.md for details."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch

from src.queries.query_executor import execute_typed_query
from src.queries.query_types import QUERY_NAME_TO_ID, NUM_QUERY_TYPES

KNN_REPRESENTATIVES_PER_TRAJECTORY = 64
SIMILARITY_REPRESENTATIVES_PER_TRAJECTORY = 64


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


def _trajectory_id_per_point(n_points: int, boundaries: list[tuple[int, int]], device: torch.device) -> torch.Tensor:
    """Build a point-index to trajectory-ID lookup."""
    trajectory_ids = torch.full((n_points,), -1, dtype=torch.long, device=device)
    for trajectory_id, (start, end) in enumerate(boundaries):
        if end > start:
            trajectory_ids[start:end] = int(trajectory_id)
    return trajectory_ids


def _trajectories_from_boundaries(points: torch.Tensor, boundaries: list[tuple[int, int]]) -> list[torch.Tensor]:
    """Slice flattened points into trajectories for query execution."""
    return [points[start:end] for start, end in boundaries]


def _ids_mask(point_trajectory_ids: torch.Tensor, trajectory_ids: set[int]) -> torch.Tensor:
    """Return points whose trajectory ID belongs to the supplied set."""
    mask = torch.zeros_like(point_trajectory_ids, dtype=torch.bool)
    for trajectory_id in trajectory_ids:
        mask |= point_trajectory_ids == int(trajectory_id)
    return mask


def _set_query_singleton_gain(original_ids: set[int]) -> float:
    """F1 gained when one true-positive trajectory ID is recovered from an empty answer."""
    if not original_ids:
        return 0.0
    return float(2.0 / (len(original_ids) + 1.0))


def _haversine_km(lat1: torch.Tensor, lon1: torch.Tensor, lat2: float, lon2: float) -> torch.Tensor:
    """Compute haversine distance in km to one anchor point."""
    radius_km = 6371.0
    lat1_rad = torch.deg2rad(lat1)
    lon1_rad = torch.deg2rad(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    delta_lat = lat1_rad - lat2_rad
    delta_lon = lon1_rad - lon2_rad
    haversine = (
        torch.sin(delta_lat / 2.0) ** 2
        + torch.cos(lat1_rad) * math.cos(lat2_rad) * torch.sin(delta_lon / 2.0) ** 2
    )
    central_angle = 2.0 * torch.atan2(torch.sqrt(haversine), torch.sqrt(torch.clamp(1.0 - haversine, min=1e-9)))
    return radius_km * central_angle


def _add_distributed_hit_label(labels: torch.Tensor, support: torch.Tensor, type_idx: int, gain: float) -> None:
    """Distribute one trajectory-hit gain over its interchangeable support points."""
    support_count = int(support.sum().item())
    if support_count <= 0:
        return
    labels[support, type_idx] += float(gain) / float(support_count)


def _knn_representative_support(
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    trajectory_ids: set[int],
    params: dict[str, float],
    representatives_per_trajectory: int = KNN_REPRESENTATIVES_PER_TRAJECTORY,
) -> torch.Tensor:
    """Return nearest in-window points of kNN-selected trajectories as positive support.

    Labels up to ``representatives_per_trajectory`` points per answer trajectory.
    This keeps the signal much denser than the old top-3 limit while still
    focusing supervision on the points most likely to preserve the kNN answer.
    """
    support = torch.zeros((points.shape[0],), dtype=torch.bool, device=points.device)
    time_start = float(params["t_center"] - params["t_half_window"])
    time_end = float(params["t_center"] + params["t_half_window"])
    limit = int(representatives_per_trajectory)

    for trajectory_id in sorted(trajectory_ids):
        if trajectory_id < 0 or trajectory_id >= len(boundaries):
            continue
        start, end = boundaries[trajectory_id]
        if end <= start:
            continue
        trajectory_points = points[start:end]
        in_window = (trajectory_points[:, 0] >= time_start) & (trajectory_points[:, 0] <= time_end)
        candidate_offsets = torch.where(in_window)[0]
        if candidate_offsets.numel() == 0:
            continue
        if limit > 0 and candidate_offsets.numel() > limit:
            candidates = trajectory_points[candidate_offsets]
            distance = _haversine_km(candidates[:, 1], candidates[:, 2], float(params["lat"]), float(params["lon"]))
            distance = distance + 0.001 * torch.abs(candidates[:, 0] - float(params["t_center"]))
            nearest = torch.topk(-distance, k=limit).indices
            candidate_offsets = candidate_offsets[nearest]
        support[start + candidate_offsets] = True

    return support


def _similarity_support_mask(
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    trajectory_ids: set[int],
    query: dict[str, Any],
    representatives_per_trajectory: int = SIMILARITY_REPRESENTATIVES_PER_TRAJECTORY,
) -> torch.Tensor:
    """Return reference-nearest points for trajectories selected by similarity execution."""
    params = query["params"]
    support = torch.zeros((points.shape[0],), dtype=torch.bool, device=points.device)
    time_start = float(params["t_start"])
    time_end = float(params["t_end"])
    reference = torch.tensor(query.get("reference", []), dtype=points.dtype, device=points.device)
    limit = int(representatives_per_trajectory)
    for trajectory_id in sorted(trajectory_ids):
        if trajectory_id < 0 or trajectory_id >= len(boundaries):
            continue
        start, end = boundaries[trajectory_id]
        if end <= start:
            continue
        trajectory_points = points[start:end]
        in_window = (trajectory_points[:, 0] >= time_start) & (trajectory_points[:, 0] <= time_end)
        candidate_offsets = torch.where(in_window)[0]
        if candidate_offsets.numel() == 0:
            continue
        if reference.numel() > 0 and limit > 0 and candidate_offsets.numel() > limit:
            candidates = trajectory_points[candidate_offsets]
            spatial = torch.cdist(candidates[:, 1:3], reference[:, 1:3]).min(dim=1).values
            temporal = torch.cdist(candidates[:, 0:1], reference[:, 0:1]).min(dim=1).values
            radius = max(float(params.get("radius", 1.0)), 1e-6)
            time_span = max(time_end - time_start, 1e-6)
            distance = spatial / radius + 0.25 * temporal / time_span
            candidate_offsets = candidate_offsets[torch.topk(-distance, k=limit).indices]
        support[start + candidate_offsets] = True
    return support


def _cluster_members(labels: Mapping[int, int] | Sequence[int]) -> dict[int, list[int]]:
    """Group non-noise trajectory labels by cluster ID."""
    items = labels.items() if isinstance(labels, Mapping) else enumerate(labels)
    clusters: dict[int, list[int]] = {}
    for trajectory_id, label in items:
        label_value = int(label)
        if label_value == -1:
            continue
        clusters.setdefault(label_value, []).append(int(trajectory_id))
    return clusters


def compute_typed_importance_labels(
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, Any]],
    seed: int,
    similarity_sample_rate: float = 0.70,
    clustering_sample_rate: float = 0.70,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-point per-type labels as expected query-F1 contribution."""
    n = points.shape[0]
    labels = torch.zeros((n, NUM_QUERY_TYPES), dtype=torch.float32, device=points.device)
    labelled_mask = torch.zeros((n, NUM_QUERY_TYPES), dtype=torch.bool, device=points.device)
    query_counts = torch.zeros((NUM_QUERY_TYPES,), dtype=torch.float32, device=points.device)
    point_trajectory_ids = _trajectory_id_per_point(n, boundaries, points.device)
    trajectories = _trajectories_from_boundaries(points, boundaries)

    for q in typed_queries:
        qtype = q["type"]
        t_idx = QUERY_NAME_TO_ID[qtype]
        params = q["params"]
        query_counts[t_idx] += 1.0

        if qtype == "range":
            box_support = _box_mask(points, params)
            hit_count = int(box_support.sum().item())
            if hit_count > 0:
                labels[box_support, t_idx] += float(2.0 / (hit_count + 1.0))

        elif qtype == "knn":
            original_ids = set(execute_typed_query(points, trajectories, q, boundaries))
            gain = _set_query_singleton_gain(original_ids)
            if gain <= 0.0:
                continue
            support = _knn_representative_support(points, boundaries, original_ids, params)
            for trajectory_id in original_ids:
                trajectory_support = support & (point_trajectory_ids == int(trajectory_id))
                _add_distributed_hit_label(labels, trajectory_support, t_idx, gain)

        elif qtype == "similarity":
            original_ids = set(execute_typed_query(points, trajectories, q, boundaries))
            gain = _set_query_singleton_gain(original_ids)
            if gain > 0.0:
                similarity_support = _similarity_support_mask(points, boundaries, original_ids, q)
                for trajectory_id in original_ids:
                    support = similarity_support & (point_trajectory_ids == int(trajectory_id))
                    _add_distributed_hit_label(labels, support, t_idx, gain)

        elif qtype == "clustering":
            original_labels = execute_typed_query(points, trajectories, q, boundaries)
            clusters = _cluster_members(original_labels)
            pair_count = sum(len(members) * (len(members) - 1) // 2 for members in clusters.values())
            if pair_count <= 0:
                continue
            box_support = _box_mask(points, params)
            for members in clusters.values():
                degree = len(members) - 1
                if degree <= 0:
                    continue
                gain = float(2.0 * degree / (pair_count + degree))
                for trajectory_id in members:
                    support = box_support & (point_trajectory_ids == int(trajectory_id))
                    _add_distributed_hit_label(labels, support, t_idx, gain)

    for type_idx in range(NUM_QUERY_TYPES):
        if float(query_counts[type_idx].item()) > 0.0:
            labels[:, type_idx] = torch.clamp(labels[:, type_idx] / query_counts[type_idx], 0.0, 1.0)
            labelled_mask[:, type_idx] = True

    return labels, labelled_mask
