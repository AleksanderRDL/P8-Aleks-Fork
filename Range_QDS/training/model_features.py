"""Model input feature builders for QDS training and inference."""

from __future__ import annotations

import math
from typing import Any

import torch

from queries.workload import TypedQueryWorkload


RANGE_AWARE_EXTRA_DIM = 8
RANGE_AWARE_POINT_DIM = 8 + RANGE_AWARE_EXTRA_DIM
OLD_WORKLOAD_BLIND_EXTRA_DIM = 9
OLD_WORKLOAD_BLIND_POINT_DIM = 8 + OLD_WORKLOAD_BLIND_EXTRA_DIM
WORKLOAD_BLIND_EXTRA_DIM = OLD_WORKLOAD_BLIND_EXTRA_DIM
WORKLOAD_BLIND_POINT_DIM = OLD_WORKLOAD_BLIND_POINT_DIM
CONTEXT_WORKLOAD_BLIND_EXTRA_DIM = 16
CONTEXT_WORKLOAD_BLIND_POINT_DIM = 8 + CONTEXT_WORKLOAD_BLIND_EXTRA_DIM
RANGE_PRIOR_CLOCK_DENSITY_POINT_DIM = CONTEXT_WORKLOAD_BLIND_POINT_DIM + 4
HISTORICAL_PRIOR_FEATURE_INDICES = (0, 1, 2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 18, 20, 21, 22, 23)
OLD_HISTORICAL_PRIOR_POINT_DIM = len(HISTORICAL_PRIOR_FEATURE_INDICES)
HISTORICAL_PRIOR_MMSI_DIM = 4
HISTORICAL_PRIOR_CLOCK_DIM = 2
HISTORICAL_PRIOR_DENSITY_DIM = 2
HISTORICAL_PRIOR_DENSITY_POINT_DIM = OLD_HISTORICAL_PRIOR_POINT_DIM + HISTORICAL_PRIOR_DENSITY_DIM
HISTORICAL_PRIOR_POINT_DIM = (
    OLD_HISTORICAL_PRIOR_POINT_DIM
    + HISTORICAL_PRIOR_CLOCK_DIM
    + HISTORICAL_PRIOR_DENSITY_DIM
)
HISTORICAL_PRIOR_MMSI_POINT_DIM = (
    OLD_HISTORICAL_PRIOR_POINT_DIM
    + HISTORICAL_PRIOR_MMSI_DIM
    + HISTORICAL_PRIOR_CLOCK_DIM
    + HISTORICAL_PRIOR_DENSITY_DIM
)
QUERY_AWARE_MODEL_TYPES = ("baseline", "turn_aware", "range_aware")
WORKLOAD_BLIND_MODEL_TYPE_CHOICES = (
    "workload_blind_range",
    "range_prior",
    "range_prior_clock_density",
    "segment_context_range",
    "historical_prior",
    "historical_prior_mmsi",
    "historical_prior_student",
)
SUPPORTED_MODEL_TYPES = QUERY_AWARE_MODEL_TYPES + WORKLOAD_BLIND_MODEL_TYPE_CHOICES
WORKLOAD_BLIND_MODEL_TYPES = frozenset(WORKLOAD_BLIND_MODEL_TYPE_CHOICES)
HISTORICAL_PRIOR_MODEL_TYPES = frozenset(
    ("historical_prior", "historical_prior_mmsi", "historical_prior_student")
)
NONPARAMETRIC_HISTORICAL_PRIOR_MODEL_TYPES = frozenset(("historical_prior", "historical_prior_mmsi"))
MODEL_TYPE_METADATA: dict[str, dict[str, object]] = {
    "baseline": {
        "model_family": "legacy_query_aware_diagnostic",
        "trainable_final_candidate": False,
        "final_success_allowed": False,
    },
    "turn_aware": {
        "model_family": "legacy_query_aware_diagnostic",
        "trainable_final_candidate": False,
        "final_success_allowed": False,
    },
    "range_aware": {
        "model_family": "legacy_query_aware_diagnostic",
        "trainable_final_candidate": False,
        "final_success_allowed": False,
    },
    "workload_blind_range": {
        "model_family": "legacy_workload_blind_scalar_scorer",
        "trainable_final_candidate": False,
        "final_success_allowed": False,
    },
    "range_prior": {
        "model_family": "legacy_workload_blind_scalar_scorer",
        "trainable_final_candidate": False,
        "final_success_allowed": False,
    },
    "range_prior_clock_density": {
        "model_family": "legacy_workload_blind_scalar_scorer",
        "trainable_final_candidate": False,
        "final_success_allowed": False,
    },
    "segment_context_range": {
        "model_family": "legacy_workload_blind_scalar_scorer",
        "trainable_final_candidate": False,
        "final_success_allowed": False,
    },
    "historical_prior": {
        "model_family": "historical_prior_knn",
        "trainable_final_candidate": False,
        "final_success_allowed": False,
    },
    "historical_prior_mmsi": {
        "model_family": "historical_prior_knn",
        "trainable_final_candidate": False,
        "final_success_allowed": False,
    },
    "historical_prior_student": {
        "model_family": "historical_prior_student",
        "trainable_final_candidate": True,
        "requires_ablation_against_standalone_knn": True,
        "final_success_allowed": False,
    },
}


def is_workload_blind_model_type(model_type: str) -> bool:
    """Return whether the configured model must not consume query features at inference."""
    return str(model_type).lower() in WORKLOAD_BLIND_MODEL_TYPES


def model_type_metadata(model_type: str) -> dict[str, object]:
    """Return final-claim guardrail metadata for a configured model type."""
    return dict(MODEL_TYPE_METADATA.get(str(model_type).lower(), {}))


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


def _build_workload_blind_context_point_features(points: torch.Tensor) -> torch.Tensor:
    """Build the full query-free context feature set for checkpoint compatibility."""
    base = points[:, :8].float().clone()
    n_points = int(points.shape[0])
    if n_points == 0:
        return torch.empty((0, CONTEXT_WORKLOAD_BLIND_POINT_DIM), dtype=torch.float32, device=points.device)

    device = points.device
    dtype = torch.float32
    extras = torch.zeros((n_points, CONTEXT_WORKLOAD_BLIND_EXTRA_DIM), dtype=dtype, device=device)
    is_start = points[:, 5].float() > 0.5 if points.shape[1] > 5 else torch.zeros(n_points, dtype=torch.bool, device=device)
    is_end = points[:, 6].float() > 0.5 if points.shape[1] > 6 else torch.zeros(n_points, dtype=torch.bool, device=device)
    indices = torch.arange(n_points, device=device)
    start_indices = torch.where(is_start)[0]
    if start_indices.numel() == 0 or int(start_indices[0].item()) != 0:
        start_indices = torch.cat([torch.zeros((1,), dtype=torch.long, device=device), start_indices])
    end_indices = torch.where(is_end)[0]
    if end_indices.numel() == 0 or int(end_indices[-1].item()) != n_points - 1:
        end_indices = torch.cat([end_indices, torch.tensor([n_points - 1], dtype=torch.long, device=device)])

    for start_tensor, end_tensor in zip(start_indices.tolist(), end_indices.tolist()):
        start = int(start_tensor)
        end_inclusive = int(end_tensor)
        if end_inclusive < start:
            continue
        end = min(n_points, end_inclusive + 1)
        length = end - start
        if length <= 0:
            continue
        local_times = points[start:end, 0].float()
        time_span = (local_times[-1] - local_times[0]).clamp(min=1e-6)
        base[start:end, 0] = (local_times - local_times[0]) / time_span
        local = indices[start:end] - start
        denom = float(max(1, length - 1))
        extras[start:end, 0] = local.float() / denom
        extras[start:end, 1] = 1.0 - extras[start:end, 0]
        extras[start:end, 2] = math.log1p(float(length))

    prev_valid = torch.ones((n_points,), dtype=torch.bool, device=device)
    next_valid = torch.ones((n_points,), dtype=torch.bool, device=device)
    prev_valid[0] = False
    next_valid[-1] = False
    prev_valid &= ~is_start
    next_valid &= ~is_end

    prev_idx = torch.clamp(indices - 1, min=0)
    next_idx = torch.clamp(indices + 1, max=n_points - 1)
    prev_dt = torch.clamp(points[:, 0].float() - points[prev_idx, 0].float(), min=0.0)
    next_dt = torch.clamp(points[next_idx, 0].float() - points[:, 0].float(), min=0.0)
    prev_dt = torch.where(prev_valid, prev_dt, torch.zeros_like(prev_dt))
    next_dt = torch.where(next_valid, next_dt, torch.zeros_like(next_dt))
    extras[:, 3] = torch.log1p(prev_dt)
    extras[:, 4] = torch.log1p(next_dt)

    prev_delta = points[:, 1:3].float() - points[prev_idx, 1:3].float()
    next_delta = points[next_idx, 1:3].float() - points[:, 1:3].float()
    prev_dist = torch.linalg.vector_norm(prev_delta, dim=1)
    next_dist = torch.linalg.vector_norm(next_delta, dim=1)
    extras[:, 5] = torch.where(prev_valid, prev_dist, torch.zeros_like(prev_dist))
    extras[:, 6] = torch.where(next_valid, next_dist, torch.zeros_like(next_dist))

    for start_tensor, end_tensor in zip(start_indices.tolist(), end_indices.tolist()):
        start = int(start_tensor)
        end_inclusive = int(end_tensor)
        if end_inclusive < start:
            continue
        end = min(n_points, end_inclusive + 1)
        length = end - start
        if length <= 0:
            continue
        local_step = extras[start:end, 5].clone()
        local_step[0] = 0.0
        cumulative_distance = torch.cumsum(local_step, dim=0)
        total_distance = cumulative_distance[-1].clamp(min=1e-6)
        extras[start:end, 9] = cumulative_distance / total_distance
        extras[start:end, 10] = 1.0 - extras[start:end, 9]

    if points.shape[1] > 4:
        heading_delta = torch.abs(points[:, 4].float() - points[prev_idx, 4].float())
        heading_delta = torch.minimum(heading_delta, 360.0 - heading_delta) / 180.0
        next_heading_delta = torch.abs(points[next_idx, 4].float() - points[:, 4].float())
        next_heading_delta = torch.minimum(next_heading_delta, 360.0 - next_heading_delta) / 180.0
        speed_delta = torch.abs(points[:, 3].float() - points[prev_idx, 3].float())
        next_speed_delta = torch.abs(points[next_idx, 3].float() - points[:, 3].float())
        extras[:, 7] = torch.where(prev_valid, heading_delta, torch.zeros_like(heading_delta))
        extras[:, 8] = torch.where(prev_valid, torch.log1p(speed_delta), torch.zeros_like(speed_delta))
        extras[:, 13] = torch.where(next_valid, next_heading_delta, torch.zeros_like(next_heading_delta))
        extras[:, 14] = torch.where(next_valid, torch.log1p(next_speed_delta), torch.zeros_like(next_speed_delta))

    both_valid = prev_valid & next_valid
    chord_dist = torch.linalg.vector_norm(points[next_idx, 1:3].float() - points[prev_idx, 1:3].float(), dim=1)
    curvature = torch.clamp(prev_dist + next_dist - chord_dist, min=0.0)
    extras[:, 11] = torch.where(both_valid, torch.log1p(prev_dt + next_dt), torch.zeros_like(prev_dt))
    extras[:, 12] = torch.where(both_valid, curvature, torch.zeros_like(curvature))
    extras[:, 15] = torch.log1p(torch.maximum(extras[:, 5], extras[:, 6]))

    return torch.cat([base, extras], dim=1)


def build_workload_blind_point_features(points: torch.Tensor) -> torch.Tensor:
    """Build the default query-free trajectory-structure features for blind scoring."""
    return _build_workload_blind_context_point_features(points)[:, :WORKLOAD_BLIND_POINT_DIM]


def build_workload_blind_point_features_for_dim(points: torch.Tensor, point_dim: int) -> torch.Tensor:
    """Build workload-blind features compatible with current and old checkpoints."""
    point_dim_int = int(point_dim)
    features = _build_workload_blind_context_point_features(points)
    if point_dim_int == WORKLOAD_BLIND_POINT_DIM:
        return features[:, :WORKLOAD_BLIND_POINT_DIM]
    if point_dim_int == CONTEXT_WORKLOAD_BLIND_POINT_DIM:
        return features
    raise ValueError(
        f"Unsupported workload-blind point_dim={point_dim_int}; expected "
        f"{WORKLOAD_BLIND_POINT_DIM} or {CONTEXT_WORKLOAD_BLIND_POINT_DIM}."
    )


def _spatial_density_features(points: torch.Tensor, bins: int = 64) -> torch.Tensor:
    """Return query-free spatial density/sparsity features from the current point cloud."""
    n_points = int(points.shape[0])
    if n_points == 0:
        return torch.empty((0, HISTORICAL_PRIOR_DENSITY_DIM), dtype=torch.float32, device=points.device)

    bin_count = max(1, int(bins))
    lat = points[:, 1].float()
    lon = points[:, 2].float()
    lat_min = lat.min()
    lon_min = lon.min()
    lat_span = torch.clamp(lat.max() - lat_min, min=1e-6)
    lon_span = torch.clamp(lon.max() - lon_min, min=1e-6)
    lat_bins = torch.clamp(((lat - lat_min) / lat_span * (bin_count - 1)).long(), 0, bin_count - 1)
    lon_bins = torch.clamp(((lon - lon_min) / lon_span * (bin_count - 1)).long(), 0, bin_count - 1)
    bin_ids = lat_bins * bin_count + lon_bins
    cell_counts = torch.bincount(bin_ids.cpu(), minlength=bin_count * bin_count).to(
        device=points.device,
        dtype=torch.float32,
    )
    counts = cell_counts[bin_ids]
    density = torch.log1p(counts) / math.log1p(float(max(1, n_points)))
    sparsity = torch.rsqrt(torch.clamp(counts, min=1.0))
    sparsity = sparsity / sparsity.max().clamp(min=1e-6)
    return torch.stack([density, sparsity], dim=1)


def _clock_time_features(points: torch.Tensor) -> torch.Tensor:
    """Return circular query-free clock-time features for historical-prior KNN."""
    n_points = int(points.shape[0])
    if n_points == 0:
        return torch.empty((0, HISTORICAL_PRIOR_CLOCK_DIM), dtype=torch.float32, device=points.device)
    phase = torch.remainder(points[:, 0].float(), 86_400.0) / 86_400.0
    angle = phase * (2.0 * math.pi)
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=1)


def point_mmsis_from_trajectory_mmsis(
    *,
    point_count: int,
    boundaries: list[tuple[int, int]],
    trajectory_mmsis: list[int],
    device: torch.device,
) -> torch.Tensor:
    """Expand per-trajectory MMSI ids to one id per flattened point."""
    if len(trajectory_mmsis) != len(boundaries):
        raise ValueError(
            "trajectory_mmsis must match boundaries length for MMSI-aware features: "
            f"got {len(trajectory_mmsis)} ids for {len(boundaries)} boundaries."
        )
    point_mmsis = torch.zeros((int(point_count),), dtype=torch.long, device=device)
    for mmsi, (start, end) in zip(trajectory_mmsis, boundaries, strict=True):
        if int(start) < 0 or int(end) < int(start) or int(end) > int(point_count):
            raise ValueError(f"Invalid boundary ({start}, {end}) for point_count={point_count}.")
        point_mmsis[int(start) : int(end)] = int(mmsi)
    return point_mmsis


def _mmsi_hash_features(point_mmsis: torch.Tensor) -> torch.Tensor:
    """Return deterministic query-free identity hashes for vessel-specific priors."""
    if point_mmsis.ndim != 1:
        raise ValueError("point_mmsis must be a vector.")
    if int(point_mmsis.numel()) == 0:
        return torch.empty((0, HISTORICAL_PRIOR_MMSI_DIM), dtype=torch.float32, device=point_mmsis.device)
    mmsi = point_mmsis.to(dtype=torch.float64)
    valid = mmsi > 0
    freqs = torch.tensor([12.9898, 78.233], dtype=torch.float64, device=point_mmsis.device)
    offsets = torch.tensor([37.719, 19.19], dtype=torch.float64, device=point_mmsis.device)
    hashed = torch.frac(torch.sin(mmsi.unsqueeze(1) * freqs + offsets) * 43758.5453123)
    angles = hashed * (2.0 * math.pi)
    features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1).to(dtype=torch.float32)
    return torch.where(valid.unsqueeze(1), features, torch.zeros_like(features))


def _legacy_historical_prior_point_features(points: torch.Tensor) -> torch.Tensor:
    """Build the query-free route-context features used by historical-prior KNN."""
    features = _build_workload_blind_context_point_features(points)
    return features[:, list(HISTORICAL_PRIOR_FEATURE_INDICES)]


def build_historical_prior_point_features(points: torch.Tensor) -> torch.Tensor:
    """Build query-free route, clock-time, and density features for historical-prior KNN."""
    route_context = _legacy_historical_prior_point_features(points)
    clock = _clock_time_features(points)
    density = _spatial_density_features(points)
    return torch.cat([route_context, clock, density], dim=1)


def build_historical_prior_mmsi_point_features(
    points: torch.Tensor,
    boundaries: list[tuple[int, int]] | None,
    trajectory_mmsis: list[int] | None,
) -> torch.Tensor:
    """Build historical-prior features with deterministic vessel-identity hashes."""
    if boundaries is None or trajectory_mmsis is None:
        raise ValueError("model_type='historical_prior_mmsi' requires trajectory_mmsis and boundaries.")
    route_context = _legacy_historical_prior_point_features(points)
    point_mmsis = point_mmsis_from_trajectory_mmsis(
        point_count=int(points.shape[0]),
        boundaries=boundaries,
        trajectory_mmsis=trajectory_mmsis,
        device=points.device,
    )
    mmsi_hash = _mmsi_hash_features(point_mmsis)
    clock = _clock_time_features(points)
    density = _spatial_density_features(points)
    return torch.cat([route_context, mmsi_hash, clock, density], dim=1)


def build_range_prior_clock_density_point_features(points: torch.Tensor) -> torch.Tensor:
    """Build full blind route-context features plus clock-time and density priors."""
    context = build_workload_blind_point_features_for_dim(points, CONTEXT_WORKLOAD_BLIND_POINT_DIM)
    clock = _clock_time_features(points)
    density = _spatial_density_features(points)
    return torch.cat([context, clock, density], dim=1)


def build_model_point_features(
    points: torch.Tensor,
    workload: TypedQueryWorkload,
    model_type: str,
    boundaries: list[tuple[int, int]] | None = None,
    trajectory_mmsis: list[int] | None = None,
) -> torch.Tensor:
    """Build the point feature matrix expected by a configured model type."""
    normalized_type = str(model_type).lower()
    if normalized_type == "baseline":
        return points[:, :7].float()
    if normalized_type == "turn_aware":
        return points[:, :8].float()
    if normalized_type == "workload_blind_range":
        return build_workload_blind_point_features(points)
    if normalized_type == "range_prior":
        return build_workload_blind_point_features_for_dim(points, CONTEXT_WORKLOAD_BLIND_POINT_DIM)
    if normalized_type in {"range_prior_clock_density", "segment_context_range"}:
        return build_range_prior_clock_density_point_features(points)
    if normalized_type in {"historical_prior", "historical_prior_student"}:
        return build_historical_prior_point_features(points)
    if normalized_type == "historical_prior_mmsi":
        return build_historical_prior_mmsi_point_features(points, boundaries, trajectory_mmsis)
    if normalized_type == "range_aware":
        range_count = sum(1 for query in workload.typed_queries if str(query.get("type", "")).lower() == "range")
        if range_count != len(workload.typed_queries):
            raise ValueError("model_type='range_aware' requires a pure range workload.")
        base = points[:, :8].float()
        relation = _range_relation_features(points, workload.typed_queries)
        return torch.cat([base, relation], dim=1)
    raise ValueError(
        "model_type must be one of: "
        + ", ".join(repr(model_type) for model_type in SUPPORTED_MODEL_TYPES)
        + "."
    )


def build_model_point_features_for_dim(
    points: torch.Tensor,
    workload: TypedQueryWorkload,
    point_dim: int,
    boundaries: list[tuple[int, int]] | None = None,
    trajectory_mmsis: list[int] | None = None,
) -> torch.Tensor:
    """Infer model input features from a saved model point dimension."""
    point_dim_int = int(point_dim)
    if point_dim_int == 7:
        return points[:, :7].float()
    if point_dim_int == 8:
        return points[:, :8].float()
    if point_dim_int == RANGE_AWARE_POINT_DIM:
        return build_model_point_features(points, workload, "range_aware")
    try:
        return _build_query_free_point_features_for_dim(
            points,
            point_dim_int,
            boundaries=boundaries,
            trajectory_mmsis=trajectory_mmsis,
        )
    except ValueError as exc:
        raise ValueError(f"Unsupported saved model point_dim={point_dim}.") from exc


def _build_query_free_point_features_for_dim(
    points: torch.Tensor,
    point_dim: int,
    boundaries: list[tuple[int, int]] | None = None,
    trajectory_mmsis: list[int] | None = None,
) -> torch.Tensor:
    """Infer query-free point features for workload-blind saved checkpoints."""
    point_dim_int = int(point_dim)
    if point_dim_int == WORKLOAD_BLIND_POINT_DIM:
        return build_workload_blind_point_features(points)
    if point_dim_int == CONTEXT_WORKLOAD_BLIND_POINT_DIM:
        return build_workload_blind_point_features_for_dim(points, CONTEXT_WORKLOAD_BLIND_POINT_DIM)
    if point_dim_int == RANGE_PRIOR_CLOCK_DENSITY_POINT_DIM:
        return build_range_prior_clock_density_point_features(points)
    if point_dim_int == OLD_HISTORICAL_PRIOR_POINT_DIM:
        return _legacy_historical_prior_point_features(points)
    if point_dim_int == HISTORICAL_PRIOR_DENSITY_POINT_DIM:
        legacy = _legacy_historical_prior_point_features(points)
        density = _spatial_density_features(points)
        return torch.cat([legacy, density], dim=1)
    if point_dim_int == HISTORICAL_PRIOR_POINT_DIM:
        return build_historical_prior_point_features(points)
    if point_dim_int == HISTORICAL_PRIOR_MMSI_POINT_DIM:
        return build_historical_prior_mmsi_point_features(points, boundaries, trajectory_mmsis)
    raise ValueError(f"Unsupported workload-blind saved model point_dim={point_dim}.")


def build_query_free_point_features_for_dim(
    points: torch.Tensor,
    point_dim: int,
    boundaries: list[tuple[int, int]] | None = None,
    trajectory_mmsis: list[int] | None = None,
) -> torch.Tensor:
    """Infer query-free point features for workload-blind saved checkpoints."""
    return _build_query_free_point_features_for_dim(
        points,
        point_dim,
        boundaries=boundaries,
        trajectory_mmsis=trajectory_mmsis,
    )
