"""Typed query execution against flattened or trajectory AIS data. See src/queries/README.md for details."""

from __future__ import annotations

import math

import torch


def _haversine_km(lat1: torch.Tensor, lon1: torch.Tensor, lat2: float, lon2: float) -> torch.Tensor:
    """Compute haversine distances in km to one anchor point. See src/queries/README.md for details."""
    r = 6371.0
    lat1r = torch.deg2rad(lat1)
    lon1r = torch.deg2rad(lon1)
    lat2r = math.radians(lat2)
    lon2r = math.radians(lon2)
    dlat = lat1r - lat2r
    dlon = lon1r - lon2r
    a = torch.sin(dlat / 2.0) ** 2 + torch.cos(lat1r) * math.cos(lat2r) * torch.sin(dlon / 2.0) ** 2
    c = 2.0 * torch.atan2(torch.sqrt(a), torch.sqrt(torch.clamp(1.0 - a, min=1e-9)))
    return r * c


def execute_range_query(points: torch.Tensor, params: dict[str, float]) -> float:
    """Execute a range query returning speed sum. See src/queries/README.md for details."""
    mask = (
        (points[:, 1] >= params["lat_min"])
        & (points[:, 1] <= params["lat_max"])
        & (points[:, 2] >= params["lon_min"])
        & (points[:, 2] <= params["lon_max"])
        & (points[:, 0] >= params["t_start"])
        & (points[:, 0] <= params["t_end"])
    )
    if not mask.any():
        return 0.0
    return float(points[mask, 3].sum().item())


def execute_knn_query(points: torch.Tensor, params: dict[str, float]) -> set[int]:
    """Execute a kNN query returning point-index set. See src/queries/README.md for details."""
    k = max(1, int(params["k"]))
    t0 = float(params["t_center"] - params["t_half_window"])
    t1 = float(params["t_center"] + params["t_half_window"])
    mask = (points[:, 0] >= t0) & (points[:, 0] <= t1)
    idx = torch.where(mask)[0]
    if idx.numel() == 0:
        return set()

    cand = points[idx]
    d_space = _haversine_km(cand[:, 1], cand[:, 2], float(params["lat"]), float(params["lon"]))
    d_time = torch.abs(cand[:, 0] - float(params["t_center"]))
    dist = d_space + 0.001 * d_time

    k_eff = min(k, dist.numel())
    chosen = torch.topk(-dist, k_eff).indices
    return set(idx[chosen].tolist())


def _dtw_like_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute a lightweight DTW-like distance for short snippets. See src/queries/README.md for details."""
    if a.numel() == 0 or b.numel() == 0:
        return float("inf")
    la = a.shape[0]
    lb = b.shape[0]
    n = max(la, lb)
    ia = torch.linspace(0, la - 1, n).long()
    ib = torch.linspace(0, lb - 1, n).long()
    pa = a[ia]
    pb = b[ib]
    d = torch.norm(pa[:, :2] - pb[:, :2], dim=1)
    return float(d.mean().item())


def execute_similarity_query(
    trajectories: list[torch.Tensor],
    params: dict[str, float],
    reference: list[list[float]],
) -> list[int]:
    """Execute a similarity query returning ranked trajectory indices. See src/queries/README.md for details."""
    ref = torch.tensor(reference, dtype=torch.float32)
    ranked: list[tuple[int, float]] = []

    for i, traj in enumerate(trajectories):
        mask_t = (traj[:, 0] >= params["t_start"]) & (traj[:, 0] <= params["t_end"])
        if not mask_t.any():
            continue
        segment = traj[mask_t][:, :3]
        centroid = segment[:, 1:3].mean(dim=0)
        if torch.norm(centroid - torch.tensor([params["lat_query_centroid"], params["lon_query_centroid"]])) > params["radius"]:
            continue
        d = _dtw_like_distance(segment, ref)
        ranked.append((i, d))

    ranked.sort(key=lambda x: x[1])
    top_k = int(params.get("top_k", 5))
    return [idx for idx, _ in ranked[:top_k]]


def _dbscan_cluster_count(points_xy: torch.Tensor, eps: float, min_samples: int) -> int:
    """Compute cluster count using a simple DBSCAN implementation. See src/queries/README.md for details."""
    if points_xy.shape[0] == 0:
        return 0
    n = points_xy.shape[0]
    visited = torch.zeros(n, dtype=torch.bool)
    labels = torch.full((n,), -1, dtype=torch.long)
    cluster_id = 0

    dmat = torch.cdist(points_xy, points_xy)

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neigh = torch.where(dmat[i] <= eps)[0]
        if neigh.numel() < min_samples:
            labels[i] = -1
            continue
        labels[i] = cluster_id
        seeds = neigh.tolist()
        ptr = 0
        while ptr < len(seeds):
            j = seeds[ptr]
            ptr += 1
            if not visited[j]:
                visited[j] = True
                neigh_j = torch.where(dmat[j] <= eps)[0]
                if neigh_j.numel() >= min_samples:
                    seeds.extend(neigh_j.tolist())
            if labels[j] == -1:
                labels[j] = cluster_id
        cluster_id += 1

    return int(cluster_id)


def execute_clustering_query(points: torch.Tensor, params: dict[str, float]) -> int:
    """Execute a clustering query returning DBSCAN cluster count. See src/queries/README.md for details."""
    mask = (
        (points[:, 1] >= params["lat_min"])
        & (points[:, 1] <= params["lat_max"])
        & (points[:, 2] >= params["lon_min"])
        & (points[:, 2] <= params["lon_max"])
        & (points[:, 0] >= params["t_start"])
        & (points[:, 0] <= params["t_end"])
    )
    pts = points[mask][:, 1:3]
    return _dbscan_cluster_count(pts, eps=float(params["eps"]), min_samples=int(params["min_samples"]))


def execute_typed_query(
    points: torch.Tensor,
    trajectories: list[torch.Tensor],
    query: dict,
):
    """Execute one typed query and return type-specific result object. See src/queries/README.md for details."""
    qtype = query["type"]
    params = query["params"]
    if qtype == "range":
        return execute_range_query(points, params)
    if qtype == "knn":
        return execute_knn_query(points, params)
    if qtype == "similarity":
        return execute_similarity_query(trajectories, params, query.get("reference", []))
    if qtype == "clustering":
        return execute_clustering_query(points, params)
    raise ValueError(f"Unsupported query type: {qtype}")
