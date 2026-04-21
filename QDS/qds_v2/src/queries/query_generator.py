"""Query workload generation for the four AIS-QDS v2 query types. See src/queries/README.md for details."""

from __future__ import annotations

from typing import Any

import torch

from src.experiments.experiment_config import TypedQueryWorkload
from src.queries.query_types import normalize_workload_mix, pad_query_features


def _dataset_bounds(points: torch.Tensor) -> dict[str, float]:
    """Compute global point-cloud bounds for query generation. See src/queries/README.md for details."""
    return {
        "t_min": float(points[:, 0].min().item()),
        "t_max": float(points[:, 0].max().item()),
        "lat_min": float(points[:, 1].min().item()),
        "lat_max": float(points[:, 1].max().item()),
        "lon_min": float(points[:, 2].min().item()),
        "lon_max": float(points[:, 2].max().item()),
    }


def _pick_point(points: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Sample one point row from the cloud. See src/queries/README.md for details."""
    idx = int(torch.randint(0, points.shape[0], (1,), generator=generator).item())
    return points[idx]


def _make_range_query(points: torch.Tensor, b: dict[str, float], generator: torch.Generator) -> dict[str, Any]:
    """Generate one range query. See src/queries/README.md for details."""
    p = _pick_point(points, generator)
    lat_w = 0.08 * (b["lat_max"] - b["lat_min"]) * (0.5 + torch.rand(1, generator=generator).item())
    lon_w = 0.08 * (b["lon_max"] - b["lon_min"]) * (0.5 + torch.rand(1, generator=generator).item())
    t_w = 0.15 * (b["t_max"] - b["t_min"]) * (0.5 + torch.rand(1, generator=generator).item())
    return {
        "type": "range",
        "params": {
            "lat_min": float(max(b["lat_min"], p[1].item() - lat_w)),
            "lat_max": float(min(b["lat_max"], p[1].item() + lat_w)),
            "lon_min": float(max(b["lon_min"], p[2].item() - lon_w)),
            "lon_max": float(min(b["lon_max"], p[2].item() + lon_w)),
            "t_start": float(max(b["t_min"], p[0].item() - t_w)),
            "t_end": float(min(b["t_max"], p[0].item() + t_w)),
        },
    }


def _make_knn_query(points: torch.Tensor, b: dict[str, float], generator: torch.Generator) -> dict[str, Any]:
    """Generate one kNN query. See src/queries/README.md for details."""
    p = _pick_point(points, generator)
    return {
        "type": "knn",
        "params": {
            "lat": float(p[1].item()),
            "lon": float(p[2].item()),
            "t_center": float(p[0].item()),
            "t_half_window": float(0.08 * (b["t_max"] - b["t_min"])),
            "k": int(torch.randint(3, 8, (1,), generator=generator).item()),
        },
    }


def _make_similarity_query(
    points: torch.Tensor,
    trajectories: list[torch.Tensor],
    b: dict[str, float],
    generator: torch.Generator,
) -> dict[str, Any]:
    """Generate one similarity query with a reference snippet. See src/queries/README.md for details."""
    p = _pick_point(points, generator)
    t_half = 0.10 * (b["t_max"] - b["t_min"])
    radius = 0.10 * max(b["lat_max"] - b["lat_min"], b["lon_max"] - b["lon_min"])

    traj_idx = int(torch.randint(0, len(trajectories), (1,), generator=generator).item())
    traj = trajectories[traj_idx]
    center = int(torch.randint(2, max(3, traj.shape[0] - 2), (1,), generator=generator).item())
    ref = traj[max(0, center - 2) : min(traj.shape[0], center + 3), :3]

    return {
        "type": "similarity",
        "params": {
            "lat_query_centroid": float(p[1].item()),
            "lon_query_centroid": float(p[2].item()),
            "t_start": float(max(b["t_min"], p[0].item() - t_half)),
            "t_end": float(min(b["t_max"], p[0].item() + t_half)),
            "radius": float(radius),
            "top_k": 5,
        },
        "reference": ref.tolist(),
    }


def _make_clustering_query(points: torch.Tensor, b: dict[str, float], generator: torch.Generator) -> dict[str, Any]:
    """Generate one clustering query. See src/queries/README.md for details."""
    rq = _make_range_query(points, b, generator)
    params = dict(rq["params"])
    params.update(
        {
            "eps": float(0.02 * max(b["lat_max"] - b["lat_min"], b["lon_max"] - b["lon_min"])),
            "min_samples": int(torch.randint(3, 7, (1,), generator=generator).item()),
        }
    )
    return {"type": "clustering", "params": params}


def generate_typed_query_workload(
    trajectories: list[torch.Tensor],
    n_queries: int,
    workload_mix: dict[str, float],
    seed: int,
) -> TypedQueryWorkload:
    """Generate a mixed typed-query workload and padded feature tensor. See src/queries/README.md for details."""
    points = torch.cat(trajectories, dim=0)
    b = _dataset_bounds(points)

    mix = normalize_workload_mix(workload_mix)
    g = torch.Generator().manual_seed(int(seed))

    names = list(mix.keys())
    weights = torch.tensor([mix[n] for n in names], dtype=torch.float32)
    counts = torch.floor(weights * n_queries).to(torch.long)
    while int(counts.sum().item()) < n_queries:
        idx = int(torch.argmax(weights - counts.float() / max(1, n_queries)).item())
        counts[idx] += 1

    typed: list[dict[str, Any]] = []
    for name, count in zip(names, counts.tolist()):
        for _ in range(int(count)):
            if name == "range":
                typed.append(_make_range_query(points, b, g))
            elif name == "knn":
                typed.append(_make_knn_query(points, b, g))
            elif name == "similarity":
                typed.append(_make_similarity_query(points, trajectories, b, g))
            elif name == "clustering":
                typed.append(_make_clustering_query(points, b, g))
            else:
                raise ValueError(f"Unsupported query type: {name}")

    perm = torch.randperm(len(typed), generator=g).tolist()
    typed = [typed[i] for i in perm]

    features, type_ids = pad_query_features(typed)
    return TypedQueryWorkload(query_features=features, typed_queries=typed, type_ids=type_ids)
