"""Query workload generation for the AIS-QDS query types. See src/queries/README.md for details."""

from __future__ import annotations

import math
from typing import Any

import torch

from src.data.trajectory_index import boundaries_from_trajectories
from src.queries.range_geometry import haversine_km_to_point, points_in_range_box
from src.queries.query_types import normalize_pure_workload_map, pad_query_features
from src.queries.workload import TypedQueryWorkload
from src.queries.workload_diagnostics import range_query_diagnostic

DENSITY_ANCHOR_PROBABILITY = 0.70
DENSITY_GRID_BINS = 64
DEFAULT_RANGE_SPATIAL_FRACTION = 0.08
DEFAULT_RANGE_TIME_FRACTION = 0.15
DEFAULT_RANGE_FOOTPRINT_JITTER = 0.5
DEFAULT_SIMILARITY_RADIUS_FRACTION = 0.04
DEFAULT_SIMILARITY_TIME_FRACTION = 0.04
DEFAULT_KNN_K = 12


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


def _density_anchor_weights(points: torch.Tensor, bins: int = DENSITY_GRID_BINS) -> torch.Tensor:
    """Return per-point spatial density weights from a lat/lon grid density map."""
    if points.shape[0] == 0:
        return torch.empty((0,), dtype=torch.float32, device=points.device)

    bin_count = max(1, int(bins))
    lat = points[:, 1]
    lon = points[:, 2]
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
    weights = cell_counts[bin_ids]
    total = weights.sum()
    if float(total.item()) <= 0.0:
        return torch.ones((points.shape[0],), dtype=torch.float32, device=points.device) / max(1, points.shape[0])
    return weights / total


def _weighted_sample_one(
    weights: torch.Tensor,
    generator: torch.Generator,
) -> int:
    """Sample one index from a non-negative weight vector.

    torch.multinomial caps at 2^24 categories, which the AIS combined-day CSVs
    exceed (23M+ points). Falls back to inverse-CDF sampling via cumsum +
    searchsorted, which has no size limit.
    """
    weight_count = int(weights.numel())
    if weight_count == 0:
        return 0
    total = float(weights.sum().item())
    if total <= 0.0:
        return int(torch.randint(0, weight_count, (1,), generator=generator).item())
    if weight_count <= (1 << 24):
        return int(torch.multinomial(weights, 1, generator=generator).item())
    cdf = torch.cumsum(weights, dim=0)
    sample_threshold = float(torch.rand(1, generator=generator).item()) * total
    return int(torch.searchsorted(cdf, torch.tensor(sample_threshold, dtype=cdf.dtype)).item())


def _sample_anchor_point(
    points: torch.Tensor,
    generator: torch.Generator,
    candidate_mask: torch.Tensor | None = None,
    density_weights: torch.Tensor | None = None,
    density_probability: float = DENSITY_ANCHOR_PROBABILITY,
) -> torch.Tensor:
    """Sample one point row from the cloud. See src/queries/README.md for details."""
    if candidate_mask is not None and bool(candidate_mask.any().item()):
        candidate_indices = torch.where(candidate_mask)[0]
    else:
        candidate_indices = None

    use_density = (
        density_weights is not None
        and density_weights.numel() == points.shape[0]
        and float(torch.rand(1, generator=generator).item()) < float(density_probability)
    )
    if use_density:
        assert density_weights is not None
        if candidate_indices is not None:
            candidate_weights = density_weights[candidate_indices].float()
            if float(candidate_weights.sum().item()) > 0.0:
                sampled_candidate_offset = _weighted_sample_one(candidate_weights, generator)
                return points[int(candidate_indices[sampled_candidate_offset].item())]
        else:
            weights = density_weights.float()
            if float(weights.sum().item()) > 0.0:
                sampled_point_idx = _weighted_sample_one(weights, generator)
                return points[sampled_point_idx]

    if candidate_indices is not None:
        candidate_offset = int(torch.randint(0, candidate_indices.shape[0], (1,), generator=generator).item())
        point_idx = int(candidate_indices[candidate_offset].item())
    else:
        point_idx = int(torch.randint(0, points.shape[0], (1,), generator=generator).item())
    return points[point_idx]


def _jitter_scale(generator: torch.Generator, jitter: float) -> float:
    """Return a random multiplicative scale in [1-jitter, 1+jitter]."""
    amount = float(jitter)
    if amount < 0.0:
        raise ValueError("range_footprint_jitter must be non-negative.")
    if amount <= 0.0:
        return 1.0
    scale = 1.0 + amount * (2.0 * float(torch.rand(1, generator=generator).item()) - 1.0)
    return max(1e-6, scale)


def _make_range_query(
    points: torch.Tensor,
    bounds: dict[str, float],
    generator: torch.Generator,
    anchor_mask: torch.Tensor | None = None,
    density_weights: torch.Tensor | None = None,
    range_spatial_fraction: float = DEFAULT_RANGE_SPATIAL_FRACTION,
    range_time_fraction: float = DEFAULT_RANGE_TIME_FRACTION,
    range_spatial_km: float | None = None,
    range_time_hours: float | None = None,
    range_footprint_jitter: float = DEFAULT_RANGE_FOOTPRINT_JITTER,
) -> dict[str, Any]:
    """Generate one range query. See src/queries/README.md for details."""
    spatial_fraction = float(range_spatial_fraction)
    time_fraction = float(range_time_fraction)
    spatial_km = None if range_spatial_km is None else float(range_spatial_km)
    time_hours = None if range_time_hours is None else float(range_time_hours)
    if (spatial_km is None and spatial_fraction <= 0.0) or (time_hours is None and time_fraction <= 0.0):
        raise ValueError("range_spatial_fraction and range_time_fraction must be positive.")
    if spatial_km is not None and spatial_km <= 0.0:
        raise ValueError("range_spatial_km must be positive when provided.")
    if time_hours is not None and time_hours <= 0.0:
        raise ValueError("range_time_hours must be positive when provided.")
    anchor_point = _sample_anchor_point(points, generator, candidate_mask=anchor_mask, density_weights=density_weights)
    lat_jitter = _jitter_scale(generator, range_footprint_jitter)
    lon_jitter = _jitter_scale(generator, range_footprint_jitter)
    time_jitter = _jitter_scale(generator, range_footprint_jitter)
    if spatial_km is None:
        lat_w = spatial_fraction * (bounds["lat_max"] - bounds["lat_min"]) * lat_jitter
        lon_w = spatial_fraction * (bounds["lon_max"] - bounds["lon_min"]) * lon_jitter
    else:
        lat_w = (spatial_km / 111.32) * lat_jitter
        cos_lat = max(0.10, abs(math.cos(math.radians(float(anchor_point[1].item())))))
        lon_w = (spatial_km / (111.32 * cos_lat)) * lon_jitter
    if time_hours is None:
        t_w = time_fraction * (bounds["t_max"] - bounds["t_min"]) * time_jitter
    else:
        t_w = time_hours * 3600.0 * time_jitter
    return {
        "type": "range",
        "params": {
            "lat_min": float(max(bounds["lat_min"], anchor_point[1].item() - lat_w)),
            "lat_max": float(min(bounds["lat_max"], anchor_point[1].item() + lat_w)),
            "lon_min": float(max(bounds["lon_min"], anchor_point[2].item() - lon_w)),
            "lon_max": float(min(bounds["lon_max"], anchor_point[2].item() + lon_w)),
            "t_start": float(max(bounds["t_min"], anchor_point[0].item() - t_w)),
            "t_end": float(min(bounds["t_max"], anchor_point[0].item() + t_w)),
        },
    }


def _make_knn_query(
    points: torch.Tensor,
    bounds: dict[str, float],
    generator: torch.Generator,
    anchor_mask: torch.Tensor | None = None,
    density_weights: torch.Tensor | None = None,
    knn_k: int | None = DEFAULT_KNN_K,
) -> dict[str, Any]:
    """Generate one kNN query. See src/queries/README.md for details."""
    anchor_point = _sample_anchor_point(points, generator, candidate_mask=anchor_mask, density_weights=density_weights)
    neighbor_count = (
        int(knn_k)
        if knn_k is not None and int(knn_k) > 0
        else int(torch.randint(3, 8, (1,), generator=generator).item())
    )
    return {
        "type": "knn",
        "params": {
            "lat": float(anchor_point[1].item()),
            "lon": float(anchor_point[2].item()),
            "t_center": float(anchor_point[0].item()),
            "t_half_window": float(0.25 * (bounds["t_max"] - bounds["t_min"])),  # 25% of day ≈ 6 h
            "k": max(1, neighbor_count),
        },
    }


def _make_similarity_query(
    points: torch.Tensor,
    trajectories: list[torch.Tensor],
    bounds: dict[str, float],
    generator: torch.Generator,
    anchor_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Generate one similarity query with a reference snippet. See src/queries/README.md for details."""
    anchor_point = _sample_anchor_point(points, generator, candidate_mask=anchor_mask)
    t_half = DEFAULT_SIMILARITY_TIME_FRACTION * (bounds["t_max"] - bounds["t_min"])
    radius = DEFAULT_SIMILARITY_RADIUS_FRACTION * max(
        bounds["lat_max"] - bounds["lat_min"],
        bounds["lon_max"] - bounds["lon_min"],
    )

    trajectory_idx = int(torch.randint(0, len(trajectories), (1,), generator=generator).item())
    trajectory = trajectories[trajectory_idx]
    center = int(torch.randint(2, max(3, trajectory.shape[0] - 2), (1,), generator=generator).item())
    reference = trajectory[max(0, center - 2) : min(trajectory.shape[0], center + 3), :3]

    return {
        "type": "similarity",
        "params": {
            "lat_query_centroid": float(anchor_point[1].item()),
            "lon_query_centroid": float(anchor_point[2].item()),
            "t_start": float(max(bounds["t_min"], anchor_point[0].item() - t_half)),
            "t_end": float(min(bounds["t_max"], anchor_point[0].item() + t_half)),
            "radius": float(radius),
            "top_k": 5,
        },
        "reference": reference.tolist(),
    }


def _make_clustering_query(
    points: torch.Tensor,
    bounds: dict[str, float],
    generator: torch.Generator,
    anchor_mask: torch.Tensor | None = None,
    density_weights: torch.Tensor | None = None,
    range_spatial_fraction: float = DEFAULT_RANGE_SPATIAL_FRACTION,
    range_time_fraction: float = DEFAULT_RANGE_TIME_FRACTION,
    range_spatial_km: float | None = None,
    range_time_hours: float | None = None,
    range_footprint_jitter: float = DEFAULT_RANGE_FOOTPRINT_JITTER,
) -> dict[str, Any]:
    """Generate one clustering query. See src/queries/README.md for details."""
    range_query = _make_range_query(
        points,
        bounds,
        generator,
        anchor_mask=anchor_mask,
        density_weights=density_weights,
        range_spatial_fraction=range_spatial_fraction,
        range_time_fraction=range_time_fraction,
        range_spatial_km=range_spatial_km,
        range_time_hours=range_time_hours,
        range_footprint_jitter=range_footprint_jitter,
    )
    range_params = dict(range_query["params"])
    range_params.update(
        {
            "eps": float(0.02 * max(bounds["lat_max"] - bounds["lat_min"], bounds["lon_max"] - bounds["lon_min"])),
            "min_samples": int(torch.randint(3, 7, (1,), generator=generator).item()),
        }
    )
    return {"type": "clustering", "params": range_params}


def point_coverage_mask_for_query(points: torch.Tensor, query: dict[str, Any]) -> torch.Tensor:
    """Return the point-level dataset coverage induced by one query.

    Coverage follows the same regions that produce query-specific training
    signal: exact boxes for range/clustering, a dense neighbourhood around the
    kNN anchor within the time window, and the spatiotemporal radius for
    similarity queries.
    """
    mask = torch.zeros((points.shape[0],), dtype=torch.bool, device=points.device)
    if points.numel() == 0:
        return mask

    query_type = str(query["type"]).lower()
    params = query["params"]
    if query_type in {"range", "clustering"}:
        return points_in_range_box(points, params)

    if query_type == "knn":
        t0 = float(params["t_center"] - params["t_half_window"])
        t1 = float(params["t_center"] + params["t_half_window"])
        in_window = (points[:, 0] >= t0) & (points[:, 0] <= t1)
        candidate_indices = torch.where(in_window)[0]
        if candidate_indices.numel() == 0:
            return mask
        candidate_points = points[candidate_indices]
        spatial_distance = haversine_km_to_point(
            candidate_points[:, 1],
            candidate_points[:, 2],
            float(params["lat"]),
            float(params["lon"]),
        )
        temporal_distance = torch.abs(candidate_points[:, 0] - float(params["t_center"]))
        distance = spatial_distance + 0.001 * temporal_distance
        effective_k = min(max(1, int(params["k"])), distance.numel())
        kth_distance = torch.topk(-distance, effective_k).values[-1].neg()
        covered_indices = candidate_indices[distance <= (3.0 * (float(kth_distance.item()) + 1e-6))]
        mask[covered_indices] = True
        return mask

    if query_type == "similarity":
        radius = float(params["radius"])
        return (
            (points[:, 0] >= params["t_start"])
            & (points[:, 0] <= params["t_end"])
            & (
                torch.sqrt(
                    (points[:, 1] - params["lat_query_centroid"]) ** 2
                    + (points[:, 2] - params["lon_query_centroid"]) ** 2
                )
                <= radius
            )
        )

    raise ValueError(f"Unsupported query type for coverage: {query_type}")


def query_coverage_mask(points: torch.Tensor, typed_queries: list[dict[str, Any]]) -> torch.Tensor:
    """Return the union of covered points for a workload."""
    covered = torch.zeros((points.shape[0],), dtype=torch.bool, device=points.device)
    for query in typed_queries:
        covered |= point_coverage_mask_for_query(points, query)
    return covered


def query_coverage_fraction(points: torch.Tensor, typed_queries: list[dict[str, Any]]) -> float:
    """Return the fraction of points covered by a workload."""
    if points.shape[0] == 0:
        return 0.0
    return float(query_coverage_mask(points, typed_queries).float().mean().item())


def _normalize_target_coverage(target_coverage: float | None) -> float | None:
    """Normalize coverage targets supplied as fractions or percentages."""
    if target_coverage is None:
        return None
    target = float(target_coverage)
    if target > 1.0:
        if target <= 100.0:
            target = target / 100.0
        else:
            raise ValueError("target_coverage must be a fraction in (0, 1] or a percent in (0, 100].")
    if target <= 0.0 or target > 1.0:
        raise ValueError("target_coverage must be a fraction in (0, 1] or a percent in (0, 100].")
    return target


def _make_query(
    query_type: str,
    points: torch.Tensor,
    trajectories: list[torch.Tensor],
    bounds: dict[str, float],
    generator: torch.Generator,
    anchor_mask: torch.Tensor | None = None,
    density_weights: torch.Tensor | None = None,
    range_spatial_fraction: float = DEFAULT_RANGE_SPATIAL_FRACTION,
    range_time_fraction: float = DEFAULT_RANGE_TIME_FRACTION,
    range_spatial_km: float | None = None,
    range_time_hours: float | None = None,
    range_footprint_jitter: float = DEFAULT_RANGE_FOOTPRINT_JITTER,
    knn_k: int | None = DEFAULT_KNN_K,
) -> dict[str, Any]:
    """Generate one query of a named type."""
    if query_type == "range":
        return _make_range_query(
            points,
            bounds,
            generator,
            anchor_mask=anchor_mask,
            density_weights=density_weights,
            range_spatial_fraction=range_spatial_fraction,
            range_time_fraction=range_time_fraction,
            range_spatial_km=range_spatial_km,
            range_time_hours=range_time_hours,
            range_footprint_jitter=range_footprint_jitter,
        )
    if query_type == "knn":
        return _make_knn_query(
            points,
            bounds,
            generator,
            anchor_mask=anchor_mask,
            density_weights=density_weights,
            knn_k=knn_k,
        )
    if query_type == "similarity":
        return _make_similarity_query(points, trajectories, bounds, generator, anchor_mask=anchor_mask)
    if query_type == "clustering":
        return _make_clustering_query(
            points,
            bounds,
            generator,
            anchor_mask=anchor_mask,
            density_weights=density_weights,
            range_spatial_fraction=range_spatial_fraction,
            range_time_fraction=range_time_fraction,
            range_spatial_km=range_spatial_km,
            range_time_hours=range_time_hours,
            range_footprint_jitter=range_footprint_jitter,
        )
    raise ValueError(f"Unsupported query type: {query_type}")


def _finalize_workload(
    points: torch.Tensor,
    typed_queries: list[dict[str, Any]],
    generator: torch.Generator,
    generation_diagnostics: dict[str, Any] | None = None,
) -> TypedQueryWorkload:
    """Shuffle, featurize, and attach point-coverage metadata."""
    if typed_queries:
        shuffle_order = torch.randperm(len(typed_queries), generator=generator).tolist()
        typed_queries = [typed_queries[i] for i in shuffle_order]

    features, type_ids = pad_query_features(typed_queries)
    covered = query_coverage_mask(points, typed_queries)
    covered_points = int(covered.sum().item())
    total_points = int(points.shape[0])
    coverage_fraction = float(covered_points / total_points) if total_points > 0 else 0.0
    diagnostics = dict(generation_diagnostics or {})
    query_generation = dict(diagnostics.get("query_generation") or {})
    type_counts: dict[str, int] = {}
    for query in typed_queries:
        query_type = str(query.get("type", "unknown"))
        type_counts[query_type] = int(type_counts.get(query_type, 0)) + 1
    query_generation.update(
        {
            "final_query_count": int(len(typed_queries)),
            "type_counts": type_counts,
            "covered_points": covered_points,
            "total_points": total_points,
            "final_coverage": coverage_fraction,
        }
    )
    diagnostics["query_generation"] = query_generation
    return TypedQueryWorkload(
        query_features=features,
        typed_queries=typed_queries,
        type_ids=type_ids,
        coverage_fraction=coverage_fraction,
        covered_points=covered_points,
        total_points=total_points,
        generation_diagnostics=diagnostics,
    )


def _range_acceptance_enabled(
    range_min_point_hits: int | None,
    range_max_point_hit_fraction: float | None,
    range_min_trajectory_hits: int | None,
    range_max_trajectory_hit_fraction: float | None,
    range_max_box_volume_fraction: float | None,
    range_duplicate_iou_threshold: float | None,
) -> bool:
    """Return whether any range acceptance filter is active."""
    return any(
        value is not None
        for value in (
            range_min_point_hits,
            range_max_point_hit_fraction,
            range_min_trajectory_hits,
            range_max_trajectory_hit_fraction,
            range_max_box_volume_fraction,
            range_duplicate_iou_threshold,
        )
    )


def _range_acceptance_state(enabled: bool, max_attempts: int | None, requested_queries: int) -> dict[str, Any]:
    """Create JSON-safe acceptance counters for workload generation."""
    return {
        "enabled": bool(enabled),
        "attempts": 0,
        "accepted": 0,
        "rejected": 0,
        "rejection_reasons": {},
        "exhausted": False,
        "max_attempts": int(max_attempts) if max_attempts is not None else None,
        "minimum_queries": int(requested_queries),
        "requested_queries": int(requested_queries),
    }


def _record_rejection(state: dict[str, Any], reason: str) -> None:
    """Update range acceptance rejection counters."""
    state["rejected"] = int(state.get("rejected", 0)) + 1
    reasons = state.setdefault("rejection_reasons", {})
    reasons[reason] = int(reasons.get(reason, 0)) + 1


def _accept_range_query(
    query: dict[str, Any],
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    accepted_range_queries: list[dict[str, Any]],
    bounds: dict[str, float],
    *,
    range_min_point_hits: int | None,
    range_max_point_hit_fraction: float | None,
    range_min_trajectory_hits: int | None,
    range_max_trajectory_hit_fraction: float | None,
    range_max_box_volume_fraction: float | None,
    range_duplicate_iou_threshold: float | None,
) -> tuple[bool, str]:
    """Validate a generated range query against optional acceptance filters."""
    diagnostic = range_query_diagnostic(
        points,
        boundaries,
        query,
        query_index=len(accepted_range_queries),
        previous_range_queries=accepted_range_queries,
        bounds=bounds,
        max_point_hit_fraction=range_max_point_hit_fraction,
        max_trajectory_hit_fraction=range_max_trajectory_hit_fraction,
        max_box_volume_fraction=range_max_box_volume_fraction,
        duplicate_iou_threshold=range_duplicate_iou_threshold,
    )
    if range_min_point_hits is not None and diagnostic["point_hits"] < int(range_min_point_hits):
        return False, "too_few_point_hits"
    if range_min_trajectory_hits is not None and diagnostic["trajectory_hits"] < int(range_min_trajectory_hits):
        return False, "too_few_trajectory_hits"
    if diagnostic["is_too_broad"]:
        return False, "too_broad"
    if range_duplicate_iou_threshold is not None and diagnostic["near_duplicate_of"] is not None:
        return False, "near_duplicate"
    return True, "accepted"


def generate_typed_query_workload(
    trajectories: list[torch.Tensor],
    n_queries: int,
    workload_map: dict[str, float],
    seed: int,
    target_coverage: float | None = None,
    max_queries: int | None = None,
    range_spatial_fraction: float = DEFAULT_RANGE_SPATIAL_FRACTION,
    range_time_fraction: float = DEFAULT_RANGE_TIME_FRACTION,
    range_spatial_km: float | None = None,
    range_time_hours: float | None = None,
    range_footprint_jitter: float = DEFAULT_RANGE_FOOTPRINT_JITTER,
    knn_k: int | None = DEFAULT_KNN_K,
    front_load_knn: int = 0,
    range_min_point_hits: int | None = None,
    range_max_point_hit_fraction: float | None = None,
    range_min_trajectory_hits: int | None = None,
    range_max_trajectory_hit_fraction: float | None = None,
    range_max_box_volume_fraction: float | None = None,
    range_duplicate_iou_threshold: float | None = None,
    range_acceptance_max_attempts: int | None = None,
) -> TypedQueryWorkload:
    """Generate a typed-query workload and padded feature tensor. See src/queries/README.md for details.

    front_load_knn: generate this many kNN queries first, before proportional
    scheduling starts.  Use this when kNN weight is high but coverage is low,
    so kNN queries are guaranteed their full quota even if generation stops early.
    """
    points = torch.cat(trajectories, dim=0)
    bounds = _dataset_bounds(points)
    boundaries = boundaries_from_trajectories(trajectories)

    normalized_workload = normalize_pure_workload_map(workload_map)
    generator = torch.Generator().manual_seed(int(seed))
    density_weights = _density_anchor_weights(points)

    query_types = list(normalized_workload.keys())
    query_type_weights = torch.tensor(
        [normalized_workload[query_type] for query_type in query_types],
        dtype=torch.float32,
    )
    coverage_target = _normalize_target_coverage(target_coverage)
    acceptance_enabled = _range_acceptance_enabled(
        range_min_point_hits,
        range_max_point_hit_fraction,
        range_min_trajectory_hits,
        range_max_trajectory_hit_fraction,
        range_max_box_volume_fraction,
        range_duplicate_iou_threshold,
    )
    requested_for_attempts = max(1, int(n_queries))
    default_max_attempts = 50 * requested_for_attempts if acceptance_enabled else None
    max_range_attempts = (
        int(range_acceptance_max_attempts)
        if range_acceptance_max_attempts is not None
        else default_max_attempts
    )
    if max_range_attempts is not None and max_range_attempts <= 0:
        raise ValueError("range_acceptance_max_attempts must be positive when provided.")
    range_acceptance = _range_acceptance_state(acceptance_enabled, max_range_attempts, requested_for_attempts)
    accepted_range_queries: list[dict[str, Any]] = []

    def build_query(query_type: str, anchor_mask: torch.Tensor | None = None) -> dict[str, Any] | None:
        """Build one query, applying optional range acceptance filters."""
        if query_type == "range" and acceptance_enabled:
            if max_range_attempts is not None and int(range_acceptance["attempts"]) >= max_range_attempts:
                range_acceptance["exhausted"] = True
                return None
            range_acceptance["attempts"] = int(range_acceptance["attempts"]) + 1
        query = _make_query(
            query_type,
            points,
            trajectories,
            bounds,
            generator,
            anchor_mask=anchor_mask,
            density_weights=density_weights,
            range_spatial_fraction=range_spatial_fraction,
            range_time_fraction=range_time_fraction,
            range_spatial_km=range_spatial_km,
            range_time_hours=range_time_hours,
            range_footprint_jitter=range_footprint_jitter,
            knn_k=knn_k,
        )
        if query_type != "range" or not acceptance_enabled:
            return query
        accepted, reason = _accept_range_query(
            query,
            points,
            boundaries,
            accepted_range_queries,
            bounds,
            range_min_point_hits=range_min_point_hits,
            range_max_point_hit_fraction=range_max_point_hit_fraction,
            range_min_trajectory_hits=range_min_trajectory_hits,
            range_max_trajectory_hit_fraction=range_max_trajectory_hit_fraction,
            range_max_box_volume_fraction=range_max_box_volume_fraction,
            range_duplicate_iou_threshold=range_duplicate_iou_threshold,
        )
        if not accepted:
            _record_rejection(range_acceptance, reason)
            return None
        range_acceptance["accepted"] = int(range_acceptance["accepted"]) + 1
        accepted_range_queries.append({"params": query["params"], "query_index": len(accepted_range_queries)})
        return query

    if coverage_target is not None:
        requested_queries = max(1, int(n_queries))
        if max_queries is not None and int(max_queries) <= 0:
            raise ValueError("max_queries must be positive when target_coverage is set.")
        generated_queries: list[dict[str, Any]] = []
        query_type_counts = torch.zeros((len(query_types),), dtype=torch.long)
        covered = torch.zeros((points.shape[0],), dtype=torch.bool, device=points.device)

        query_limit = max(requested_queries, int(max_queries) if max_queries is not None else requested_queries)
        stop_reason = "max_queries_reached"

        # Generate kNN queries first so they always get their initial quota even
        # when other types advance coverage faster.
        if front_load_knn > 0 and "knn" in query_types:
            knn_idx = query_types.index("knn")
            for _ in range(min(front_load_knn, query_limit)):
                query = build_query("knn", anchor_mask=None)
                if query is None:
                    break
                generated_queries.append(query)
                query_type_counts[knn_idx] += 1
                covered |= point_coverage_mask_for_query(points, query)

        while len(generated_queries) < query_limit:
            current_coverage = float(covered.float().mean().item()) if points.shape[0] > 0 else 0.0
            if len(generated_queries) >= requested_queries and current_coverage >= coverage_target:
                stop_reason = "target_coverage_reached"
                break
            desired_counts = query_type_weights * float(len(generated_queries) + 1)
            query_type_idx = int(torch.argmax(desired_counts - query_type_counts.float()).item())
            query_type = query_types[query_type_idx]
            anchor_mask = (~covered) if current_coverage < coverage_target else None
            query = build_query(query_type, anchor_mask=anchor_mask)
            if query is None:
                if query_type == "range" and range_acceptance.get("exhausted"):
                    stop_reason = "range_acceptance_exhausted"
                    break
                continue
            generated_queries.append(query)
            query_type_counts[query_type_idx] += 1
            covered |= point_coverage_mask_for_query(points, query)

        final_coverage = float(covered.float().mean().item()) if points.shape[0] > 0 else 0.0
        if (
            stop_reason == "max_queries_reached"
            and len(generated_queries) >= requested_queries
            and final_coverage >= coverage_target
        ):
            stop_reason = "target_coverage_reached"
        query_generation = {
            "mode": "target_coverage",
            "minimum_queries": int(requested_queries),
            "requested_queries": int(requested_queries),
            "max_queries": int(query_limit),
            "target_coverage": float(coverage_target),
            "stop_reason": stop_reason,
        }
        return _finalize_workload(
            points,
            generated_queries,
            generator,
            generation_diagnostics={
                "range_acceptance": range_acceptance,
                "query_generation": query_generation,
            },
        )

    query_type_counts = torch.floor(query_type_weights * n_queries).to(torch.long)
    while int(query_type_counts.sum().item()) < n_queries:
        underfilled_type_idx = int(
            torch.argmax(query_type_weights - query_type_counts.float() / max(1, n_queries)).item()
        )
        query_type_counts[underfilled_type_idx] += 1

    generated_queries: list[dict[str, Any]] = []
    stop_reason = "fixed_count_completed"
    # Front-load kNN queries before proportional types
    if front_load_knn > 0 and "knn" in query_types:
        knn_idx = query_types.index("knn")
        frontloaded_knn_count = min(front_load_knn, int(query_type_counts[knn_idx].item()))
        for _ in range(frontloaded_knn_count):
            query = build_query("knn")
            if query is not None:
                generated_queries.append(query)
        query_type_counts[knn_idx] = max(0, query_type_counts[knn_idx] - frontloaded_knn_count)
    for query_type, count in zip(query_types, query_type_counts.tolist()):
        for _ in range(int(count)):
            query = build_query(query_type)
            if query is None:
                if query_type == "range" and range_acceptance.get("exhausted"):
                    stop_reason = "range_acceptance_exhausted"
                    break
                continue
            generated_queries.append(query)

    return _finalize_workload(
        points,
        generated_queries,
        generator,
        generation_diagnostics={
            "range_acceptance": range_acceptance,
            "query_generation": {
                "mode": "fixed_count",
                "minimum_queries": int(n_queries),
                "requested_queries": int(n_queries),
                "max_queries": int(n_queries),
                "target_coverage": None,
                "stop_reason": stop_reason,
            },
        },
    )
