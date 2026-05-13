"""Typed query execution against flattened or trajectory AIS data. See src/queries/README.md for details."""

from __future__ import annotations

import math

import torch


def _boundaries_from_trajectories(trajectories: list[torch.Tensor]) -> list[tuple[int, int]]:
    """Build flattened point boundaries from trajectory order."""
    boundaries: list[tuple[int, int]] = []
    cursor = 0
    for trajectory in trajectories:
        point_count = int(trajectory.shape[0])
        boundaries.append((cursor, cursor + point_count))
        cursor += point_count
    return boundaries


def _default_boundaries(points: torch.Tensor, boundaries: list[tuple[int, int]] | None) -> list[tuple[int, int]]:
    """Use explicit boundaries when supplied, otherwise treat all points as one trajectory."""
    return boundaries if boundaries is not None else [(0, int(points.shape[0]))]


def _indices_to_trajectory_ids(indices: torch.Tensor, boundaries: list[tuple[int, int]]) -> set[int]:
    """Map flattened point indices to stable trajectory IDs derived from boundaries."""
    if indices.numel() == 0:
        return set()
    trajectory_ids: set[int] = set()
    for trajectory_id, (start, end) in enumerate(boundaries):
        if end <= start:
            continue
        if bool(((indices >= start) & (indices < end)).any().item()):
            trajectory_ids.add(trajectory_id)
    return trajectory_ids


def _haversine_km(lat1: torch.Tensor, lon1: torch.Tensor, lat2: float, lon2: float) -> torch.Tensor:
    """Compute haversine distances in km to one anchor point. See src/queries/README.md for details."""
    earth_radius_km = 6371.0
    lat1_rad = torch.deg2rad(lat1)
    lon1_rad = torch.deg2rad(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    delta_lat = lat1_rad - lat2_rad
    delta_lon = lon1_rad - lon2_rad
    haversine_term = (
        torch.sin(delta_lat / 2.0) ** 2
        + torch.cos(lat1_rad) * math.cos(lat2_rad) * torch.sin(delta_lon / 2.0) ** 2
    )
    angular_distance = 2.0 * torch.atan2(
        torch.sqrt(haversine_term),
        torch.sqrt(torch.clamp(1.0 - haversine_term, min=1e-9)),
    )
    return earth_radius_km * angular_distance


def _box_mask(points: torch.Tensor, params: dict[str, float]) -> torch.Tensor:
    """Return the point mask inside a spatiotemporal query box."""
    mask = (
        (points[:, 1] >= params["lat_min"])
        & (points[:, 1] <= params["lat_max"])
        & (points[:, 2] >= params["lon_min"])
        & (points[:, 2] <= params["lon_max"])
        & (points[:, 0] >= params["t_start"])
        & (points[:, 0] <= params["t_end"])
    )
    return mask


def execute_range_query(
    points: torch.Tensor,
    params: dict[str, float],
    boundaries: list[tuple[int, int]] | None = None,
) -> set[int]:
    """Execute a range query returning matching trajectory IDs. See src/queries/README.md for details."""
    mask = _box_mask(points, params)
    if not mask.any():
        return set()
    return _indices_to_trajectory_ids(torch.where(mask)[0], _default_boundaries(points, boundaries))


def execute_knn_query(
    points: torch.Tensor,
    params: dict[str, float],
    boundaries: list[tuple[int, int]] | None = None,
) -> set[int]:
    """Execute a kNN query returning the nearest distinct trajectory IDs."""
    requested_neighbor_count = max(1, int(params["k"]))
    t0 = float(params["t_center"] - params["t_half_window"])
    t1 = float(params["t_center"] + params["t_half_window"])
    mask = (points[:, 0] >= t0) & (points[:, 0] <= t1)
    candidate_indices = torch.where(mask)[0]
    if candidate_indices.numel() == 0:
        return set()

    candidate_points = points[candidate_indices]
    spatial_distance = _haversine_km(
        candidate_points[:, 1],
        candidate_points[:, 2],
        float(params["lat"]),
        float(params["lon"]),
    )
    temporal_distance = torch.abs(candidate_points[:, 0] - float(params["t_center"]))
    distance = spatial_distance + 0.001 * temporal_distance

    query_boundaries = _default_boundaries(points, boundaries)
    boundary_ends = torch.tensor(
        [end for _, end in query_boundaries],
        dtype=torch.long,
        device=candidate_indices.device,
    )
    trajectory_ids = torch.bucketize(candidate_indices, boundary_ends, right=True)
    min_distance = torch.full((len(query_boundaries),), float("inf"), dtype=distance.dtype, device=distance.device)
    min_distance.scatter_reduce_(0, trajectory_ids, distance, reduce="amin", include_self=True)
    valid_trajectories = torch.isfinite(min_distance)
    if not bool(valid_trajectories.any().item()):
        return set()

    valid_ids = torch.where(valid_trajectories)[0]
    effective_k = min(requested_neighbor_count, int(valid_ids.numel()))
    chosen = torch.topk(-min_distance[valid_ids], effective_k).indices
    return {int(trajectory_id) for trajectory_id in valid_ids[chosen].tolist()}


def _dtw_like_distance(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    """Compute a lightweight DTW-like distance for short snippets. See src/queries/README.md for details."""
    if candidate.numel() == 0 or reference.numel() == 0:
        return float("inf")
    candidate_length = candidate.shape[0]
    reference_length = reference.shape[0]
    resampled_length = max(candidate_length, reference_length)
    candidate_positions = torch.linspace(0, candidate_length - 1, resampled_length).long()
    reference_positions = torch.linspace(0, reference_length - 1, resampled_length).long()
    candidate_points = candidate[candidate_positions]
    reference_points = reference[reference_positions]
    distances = torch.norm(candidate_points[:, 1:3] - reference_points[:, 1:3], dim=1)
    return float(distances.mean().item())


def execute_similarity_query(
    trajectories: list[torch.Tensor],
    params: dict[str, float],
    reference: list[list[float]],
) -> list[int]:
    """Execute a similarity query returning ranked trajectory indices. See src/queries/README.md for details."""
    reference_tensor = torch.tensor(reference, dtype=torch.float32)
    ranked: list[tuple[int, float]] = []

    for trajectory_idx, trajectory in enumerate(trajectories):
        time_mask = (trajectory[:, 0] >= params["t_start"]) & (trajectory[:, 0] <= params["t_end"])
        if not time_mask.any():
            continue
        segment = trajectory[time_mask][:, :3]
        centroid = segment[:, 1:3].mean(dim=0)
        query_centroid = centroid.new_tensor([params["lat_query_centroid"], params["lon_query_centroid"]])
        if torch.norm(centroid - query_centroid) > params["radius"]:
            continue
        distance = _dtw_like_distance(segment, reference_tensor)
        ranked.append((trajectory_idx, distance))

    ranked.sort(key=lambda item: item[1])
    top_k = int(params.get("top_k", 5))
    return [trajectory_idx for trajectory_idx, _ in ranked[:top_k]]


def _dbscan_labels(points_xy: torch.Tensor, eps: float, min_samples: int) -> list[int]:
    """Compute simple DBSCAN labels for representative points."""
    if points_xy.shape[0] == 0:
        return []
    point_count = points_xy.shape[0]
    visited = torch.zeros(point_count, dtype=torch.bool)
    labels = torch.full((point_count,), -1, dtype=torch.long)
    cluster_id = 0

    distance_matrix = torch.cdist(points_xy, points_xy)

    for point_idx in range(point_count):
        if visited[point_idx]:
            continue
        visited[point_idx] = True
        neighbors = torch.where(distance_matrix[point_idx] <= eps)[0]
        if neighbors.numel() < min_samples:
            labels[point_idx] = -1
            continue
        labels[point_idx] = cluster_id
        seed_points = neighbors.tolist()
        seed_cursor = 0
        while seed_cursor < len(seed_points):
            neighbor_idx = seed_points[seed_cursor]
            seed_cursor += 1
            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True
                neighbor_neighbors = torch.where(distance_matrix[neighbor_idx] <= eps)[0]
                if neighbor_neighbors.numel() >= min_samples:
                    seed_points.extend(neighbor_neighbors.tolist())
            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id
        cluster_id += 1

    return [int(label) for label in labels.tolist()]


def execute_clustering_query(
    points: torch.Tensor,
    params: dict[str, float],
    boundaries: list[tuple[int, int]] | None = None,
) -> list[int]:
    """Execute clustering over trajectory centroids and return per-trajectory labels."""
    query_boundaries = _default_boundaries(points, boundaries)
    cluster_labels = [-1 for _ in query_boundaries]
    representatives: list[torch.Tensor] = []
    represented_trajectory_ids: list[int] = []

    for trajectory_id, (start, end) in enumerate(query_boundaries):
        if end <= start:
            continue
        trajectory_points = points[start:end]
        query_mask = _box_mask(trajectory_points, params)
        if not bool(query_mask.any().item()):
            continue
        representatives.append(trajectory_points[query_mask, 1:3].mean(dim=0))
        represented_trajectory_ids.append(trajectory_id)

    if not representatives:
        return cluster_labels

    representative_xy = torch.stack(representatives)
    representative_labels = _dbscan_labels(
        representative_xy,
        eps=float(params["eps"]),
        min_samples=int(params["min_samples"]),
    )
    for trajectory_id, label in zip(represented_trajectory_ids, representative_labels):
        cluster_labels[trajectory_id] = int(label)
    return cluster_labels


def execute_typed_query(
    points: torch.Tensor,
    trajectories: list[torch.Tensor],
    query: dict,
    boundaries: list[tuple[int, int]] | None = None,
) -> set[int] | list[int]:
    """Execute one typed query and return type-specific result object. See src/queries/README.md for details."""
    query_type = query["type"]
    params = query["params"]
    query_boundaries = boundaries if boundaries is not None else _boundaries_from_trajectories(trajectories)
    if query_type == "range":
        return execute_range_query(points, params, query_boundaries)
    if query_type == "knn":
        return execute_knn_query(points, params, query_boundaries)
    if query_type == "similarity":
        return execute_similarity_query(trajectories, params, query.get("reference", []))
    if query_type == "clustering":
        return execute_clustering_query(points, params, query_boundaries)
    raise ValueError(f"Unsupported query type: {query_type}")
