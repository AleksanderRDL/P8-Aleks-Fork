"""Model input feature builders for QDS training and inference."""

from __future__ import annotations

import math
from typing import Any

import torch

from src.queries.workload import TypedQueryWorkload


RANGE_AWARE_EXTRA_DIM = 8
RANGE_AWARE_POINT_DIM = 8 + RANGE_AWARE_EXTRA_DIM


def _range_relation_features(points: torch.Tensor, typed_queries: list[dict[str, Any]]) -> torch.Tensor:
    """Return per-point relation features for pure range workloads."""
    range_queries = [query for query in typed_queries if str(query.get("type", "")).lower() == "range"]
    device = points.device
    dtype = torch.float32
    n_points = int(points.shape[0])
    features = torch.zeros((n_points, RANGE_AWARE_EXTRA_DIM), dtype=dtype, device=device)
    if n_points == 0 or not range_queries:
        return features

    query_values = torch.tensor(
        [
            [
                float(query["params"]["t_start"]),
                float(query["params"]["t_end"]),
                float(query["params"]["lat_min"]),
                float(query["params"]["lat_max"]),
                float(query["params"]["lon_min"]),
                float(query["params"]["lon_max"]),
            ]
            for query in range_queries
        ],
        dtype=dtype,
        device=device,
    )
    t0_all, t1_all, lat0_all, lat1_all, lon0_all, lon1_all = query_values.T
    t_span_all = torch.clamp(t1_all - t0_all, min=1e-6)
    lat_span_all = torch.clamp(lat1_all - lat0_all, min=1e-6)
    lon_span_all = torch.clamp(lon1_all - lon0_all, min=1e-6)
    inv_sqrt_volume_all = torch.rsqrt(torch.clamp(t_span_all * lat_span_all * lon_span_all, min=1e-12))

    sqrt3 = math.sqrt(3.0)
    query_count = len(range_queries)
    point_chunk_size = 262_144
    query_chunk_size = 32

    for point_start in range(0, n_points, point_chunk_size):
        point_end = min(n_points, point_start + point_chunk_size)
        time = points[point_start:point_end, 0].to(dtype=dtype).unsqueeze(1)
        lat = points[point_start:point_end, 1].to(dtype=dtype).unsqueeze(1)
        lon = points[point_start:point_end, 2].to(dtype=dtype).unsqueeze(1)
        local = torch.zeros((point_end - point_start, RANGE_AWARE_EXTRA_DIM), dtype=dtype, device=device)

        for query_start in range(0, query_count, query_chunk_size):
            query_end = min(query_count, query_start + query_chunk_size)
            t0 = t0_all[query_start:query_end].unsqueeze(0)
            lat0 = lat0_all[query_start:query_end].unsqueeze(0)
            lon0 = lon0_all[query_start:query_end].unsqueeze(0)
            rel_t = (time - t0) / t_span_all[query_start:query_end].unsqueeze(0)
            rel_lat = (lat - lat0) / lat_span_all[query_start:query_end].unsqueeze(0)
            rel_lon = (lon - lon0) / lon_span_all[query_start:query_end].unsqueeze(0)
            inside = (
                (rel_t >= 0.0)
                & (rel_t <= 1.0)
                & (rel_lat >= 0.0)
                & (rel_lat <= 1.0)
                & (rel_lon >= 0.0)
                & (rel_lon <= 1.0)
            )
            inside_f = inside.to(dtype=dtype)
            inv_sqrt_volume = inv_sqrt_volume_all[query_start:query_end].unsqueeze(0)

            local[:, 0] += inside_f.sum(dim=1)
            local[:, 1] += (inside_f * inv_sqrt_volume).sum(dim=1)
            local[:, 2] = torch.maximum(local[:, 2], (inside_f * inv_sqrt_volume).max(dim=1).values)

            below_t = torch.clamp(-rel_t, min=0.0)
            above_t = torch.clamp(rel_t - 1.0, min=0.0)
            below_lat = torch.clamp(-rel_lat, min=0.0)
            above_lat = torch.clamp(rel_lat - 1.0, min=0.0)
            below_lon = torch.clamp(-rel_lon, min=0.0)
            above_lon = torch.clamp(rel_lon - 1.0, min=0.0)
            outside_dist = torch.sqrt(
                torch.maximum(below_t, above_t).square()
                + torch.maximum(below_lat, above_lat).square()
                + torch.maximum(below_lon, above_lon).square()
            )
            local[:, 3] = torch.maximum(local[:, 3], torch.exp(-4.0 * outside_dist).max(dim=1).values)

            center_dist = torch.sqrt(
                ((rel_t - 0.5) * 2.0).square()
                + ((rel_lat - 0.5) * 2.0).square()
                + ((rel_lon - 0.5) * 2.0).square()
            )
            center_score = torch.clamp(1.0 - center_dist / sqrt3, min=0.0, max=1.0) * inside_f
            local[:, 4] = torch.maximum(local[:, 4], center_score.max(dim=1).values)

            t_face = torch.minimum(rel_t, 1.0 - rel_t)
            lat_face = torch.minimum(rel_lat, 1.0 - rel_lat)
            lon_face = torch.minimum(rel_lon, 1.0 - rel_lon)
            temporal_boundary = torch.clamp(1.0 - 2.0 * t_face, min=0.0, max=1.0) * inside_f
            spatial_face = torch.minimum(lat_face, lon_face)
            spatial_boundary = torch.clamp(1.0 - 2.0 * spatial_face, min=0.0, max=1.0) * inside_f
            any_face = torch.minimum(t_face, spatial_face)
            boundary = torch.clamp(1.0 - 2.0 * any_face, min=0.0, max=1.0) * inside_f
            local[:, 5] = torch.maximum(local[:, 5], boundary.max(dim=1).values)
            local[:, 6] = torch.maximum(local[:, 6], temporal_boundary.max(dim=1).values)
            local[:, 7] = torch.maximum(local[:, 7], spatial_boundary.max(dim=1).values)

        local[:, 0] = local[:, 0] / float(query_count)
        local[:, 1] = local[:, 1] / float(query_count)
        features[point_start:point_end] = local
    return features


def build_model_point_features(
    points: torch.Tensor,
    workload: TypedQueryWorkload,
    model_type: str,
) -> torch.Tensor:
    """Build the point feature matrix expected by a configured model type."""
    normalized_type = str(model_type).lower()
    if normalized_type == "baseline":
        return points[:, :7].float()
    if normalized_type == "turn_aware":
        return points[:, :8].float()
    if normalized_type == "range_aware":
        range_count = sum(1 for query in workload.typed_queries if str(query.get("type", "")).lower() == "range")
        if range_count != len(workload.typed_queries):
            raise ValueError("model_type='range_aware' requires a pure range workload.")
        base = points[:, :8].float()
        relation = _range_relation_features(points, workload.typed_queries)
        return torch.cat([base, relation], dim=1)
    raise ValueError("model_type must be 'baseline', 'turn_aware', or 'range_aware'.")


def build_model_point_features_for_dim(
    points: torch.Tensor,
    workload: TypedQueryWorkload,
    point_dim: int,
) -> torch.Tensor:
    """Infer model input features from a saved model point dimension."""
    if int(point_dim) == 7:
        return points[:, :7].float()
    if int(point_dim) == 8:
        return points[:, :8].float()
    if int(point_dim) == RANGE_AWARE_POINT_DIM:
        return build_model_point_features(points, workload, "range_aware")
    raise ValueError(f"Unsupported saved model point_dim={point_dim}.")
