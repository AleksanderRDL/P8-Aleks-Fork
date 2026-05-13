"""Query workload generation for the AIS-QDS query types. See queries/README.md for details."""

from __future__ import annotations

import math
from typing import Any

import torch

from data.trajectory_index import boundaries_from_trajectories
from queries.range_geometry import points_in_range_box
from queries.query_types import normalize_pure_workload_map, pad_query_features
from queries.workload import TypedQueryWorkload
from queries.workload_diagnostics import range_query_diagnostic

DENSITY_ANCHOR_PROBABILITY = 0.70
DENSITY_GRID_BINS = 64
DEFAULT_RANGE_SPATIAL_FRACTION = 0.08
DEFAULT_RANGE_TIME_FRACTION = 0.15
DEFAULT_RANGE_FOOTPRINT_JITTER = 0.5


def _dataset_bounds(points: torch.Tensor) -> dict[str, float]:
    """Compute global point-cloud bounds for query generation. See queries/README.md for details."""
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
    """Sample one point row from the cloud. See queries/README.md for details."""
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
    """Generate one range query. See queries/README.md for details."""
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


def point_coverage_mask_for_query(points: torch.Tensor, query: dict[str, Any]) -> torch.Tensor:
    """Return the point-level dataset coverage induced by one query.

    Coverage follows the exact range box that produces query-specific training
    signal.
    """
    mask = torch.zeros((points.shape[0],), dtype=torch.bool, device=points.device)
    if points.numel() == 0:
        return mask

    query_type = str(query["type"]).lower()
    params = query["params"]
    if query_type == "range":
        return points_in_range_box(points, params)

    raise ValueError(f"Only range queries are supported for coverage; got query type: {query_type}")


def query_coverage_mask(points: torch.Tensor, typed_queries: list[dict[str, Any]]) -> torch.Tensor:
    """Return the union of covered points for a workload."""
    covered = torch.zeros((points.shape[0],), dtype=torch.bool, device=points.device)
    for query in typed_queries:
        covered |= point_coverage_mask_for_query(points, query)
    return covered


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
    range_min_point_hits: int | None = None,
    range_max_point_hit_fraction: float | None = None,
    range_min_trajectory_hits: int | None = None,
    range_max_trajectory_hit_fraction: float | None = None,
    range_max_box_volume_fraction: float | None = None,
    range_duplicate_iou_threshold: float | None = None,
    range_acceptance_max_attempts: int | None = None,
) -> TypedQueryWorkload:
    """Generate a range-query workload and padded feature tensor. See queries/README.md for details."""
    points = torch.cat(trajectories, dim=0)
    bounds = _dataset_bounds(points)
    boundaries = boundaries_from_trajectories(trajectories)

    normalize_pure_workload_map(workload_map)
    generator = torch.Generator().manual_seed(int(seed))
    density_weights = _density_anchor_weights(points)

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

    def build_query(anchor_mask: torch.Tensor | None = None) -> dict[str, Any] | None:
        """Build one query, applying optional range acceptance filters."""
        if acceptance_enabled:
            if max_range_attempts is not None and int(range_acceptance["attempts"]) >= max_range_attempts:
                range_acceptance["exhausted"] = True
                return None
            range_acceptance["attempts"] = int(range_acceptance["attempts"]) + 1
        query = _make_range_query(
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
        if not acceptance_enabled:
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
        covered = torch.zeros((points.shape[0],), dtype=torch.bool, device=points.device)

        query_limit = max(requested_queries, int(max_queries) if max_queries is not None else requested_queries)
        stop_reason = "max_queries_reached"

        while len(generated_queries) < query_limit:
            current_coverage = float(covered.float().mean().item()) if points.shape[0] > 0 else 0.0
            if len(generated_queries) >= requested_queries and current_coverage >= coverage_target:
                stop_reason = "target_coverage_reached"
                break
            anchor_mask = (~covered) if current_coverage < coverage_target else None
            query = build_query(anchor_mask=anchor_mask)
            if query is None:
                if range_acceptance.get("exhausted"):
                    stop_reason = "range_acceptance_exhausted"
                    break
                continue
            generated_queries.append(query)
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

    generated_queries: list[dict[str, Any]] = []
    stop_reason = "fixed_count_completed"
    for _ in range(int(n_queries)):
        query = build_query()
        if query is None:
            if range_acceptance.get("exhausted"):
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
