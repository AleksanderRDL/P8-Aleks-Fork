"""Query workload generation for the AIS-QDS query types. See queries/README.md for details."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import torch

from data.trajectory_index import boundaries_from_trajectories
from queries.range_geometry import points_in_range_box
from queries.query_types import normalize_pure_workload_map, pad_query_features
from queries.workload import TypedQueryWorkload
from queries.workload_diagnostics import compute_range_workload_diagnostics, range_query_diagnostic
from queries.workload_profiles import (
    LEGACY_GENERATOR_PROFILE,
    RangeWorkloadProfile,
    max_point_hit_fraction_for_coverage,
    range_workload_profile,
    workload_profile_metadata,
)

DENSITY_ANCHOR_PROBABILITY = 0.70
DENSITY_GRID_BINS = 64
DEFAULT_RANGE_SPATIAL_FRACTION = 0.08
DEFAULT_RANGE_TIME_FRACTION = 0.15
DEFAULT_RANGE_FOOTPRINT_JITTER = 0.5
DEFAULT_RANGE_TIME_DOMAIN_MODE = "dataset"
DEFAULT_RANGE_ANCHOR_MODE = "mixed_density"
RANGE_TIME_DOMAIN_MODES = ("dataset", "anchor_day")
RANGE_ANCHOR_MODES = ("mixed_density", "dense", "uniform", "sparse")
RANGE_COVERAGE_CALIBRATION_MODES = ("profile_sampled_query_count", "uncovered_anchor_chasing")
RANGE_WORKLOAD_V1_ANCHOR_FAMILIES = (
    "density_route",
    "boundary_entry_exit",
    "crossing_turn_change",
    "port_or_approach_zone",
    "sparse_background_control",
)
SECONDS_PER_DAY = 24.0 * 3600.0
EPOCH_LIKE_SECONDS = 366.0 * SECONDS_PER_DAY


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


def _sparse_anchor_weights(points: torch.Tensor, bins: int = DENSITY_GRID_BINS) -> torch.Tensor:
    """Return per-point weights that sample occupied spatial cells more evenly."""
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
    weights = 1.0 / torch.clamp(cell_counts[bin_ids], min=1.0)
    total = weights.sum()
    if float(total.item()) <= 0.0:
        return torch.ones((points.shape[0],), dtype=torch.float32, device=points.device) / max(1, points.shape[0])
    return weights / total


def _normalize_weights(weights: torch.Tensor) -> torch.Tensor:
    """Return a probability vector, falling back to uniform when needed."""
    if int(weights.numel()) == 0:
        return weights.to(dtype=torch.float32)
    clean = torch.nan_to_num(weights.float().clamp(min=0.0), nan=0.0, posinf=0.0, neginf=0.0)
    total = float(clean.sum().item())
    if total <= 1e-12:
        return torch.ones_like(clean, dtype=torch.float32) / float(clean.numel())
    return clean / total


def _endpoint_anchor_weights(points: torch.Tensor) -> torch.Tensor:
    """Return endpoint-biased anchor weights from query-free trajectory flags."""
    if points.shape[0] == 0:
        return torch.empty((0,), dtype=torch.float32, device=points.device)
    weights = torch.ones((points.shape[0],), dtype=torch.float32, device=points.device)
    if points.shape[1] > 6:
        endpoint = (points[:, 5].float() > 0.5) | (points[:, 6].float() > 0.5)
        weights = weights + 8.0 * endpoint.float()
    if points.shape[1] > 7:
        weights = weights + 3.0 * points[:, 7].float().clamp(min=0.0)
    return _normalize_weights(weights)


def _turn_change_anchor_weights(points: torch.Tensor) -> torch.Tensor:
    """Return turn/change-biased anchor weights from query-free point features."""
    if points.shape[0] == 0:
        return torch.empty((0,), dtype=torch.float32, device=points.device)
    weights = torch.ones((points.shape[0],), dtype=torch.float32, device=points.device)
    if points.shape[1] > 7:
        weights = weights + 8.0 * points[:, 7].float().clamp(min=0.0)
    if points.shape[1] > 4:
        prev_idx = torch.clamp(torch.arange(points.shape[0], device=points.device) - 1, min=0)
        heading_delta = torch.abs(points[:, 4].float() - points[prev_idx, 4].float())
        heading_delta = torch.minimum(heading_delta, 360.0 - heading_delta).clamp(min=0.0) / 180.0
        weights = weights + 4.0 * heading_delta
    if points.shape[1] > 3:
        prev_idx = torch.clamp(torch.arange(points.shape[0], device=points.device) - 1, min=0)
        speed_delta = torch.abs(points[:, 3].float() - points[prev_idx, 3].float())
        if float(speed_delta.max().item()) > 1e-6:
            speed_delta = speed_delta / speed_delta.max().clamp(min=1e-6)
        weights = weights + 2.0 * speed_delta
    return _normalize_weights(weights)


def _port_or_approach_anchor_weights(points: torch.Tensor) -> torch.Tensor:
    """Return hotspot/approach weights distinct from generic route density."""
    if points.shape[0] == 0:
        return torch.empty((0,), dtype=torch.float32, device=points.device)
    density = _density_anchor_weights(points)
    endpoint = _endpoint_anchor_weights(points)
    weights = 0.35 * density + 0.35 * endpoint
    if points.shape[1] > 3:
        speed = points[:, 3].float().clamp(min=0.0)
        if int(speed.numel()) > 0:
            low_speed_cutoff = torch.quantile(speed, 0.30)
            slow = (speed <= low_speed_cutoff).float()
            weights = weights + 0.30 * _normalize_weights(slow)
    return _normalize_weights(weights)


def _normalize_range_anchor_mode(mode: str) -> str:
    """Normalize range-query anchor sampling mode names."""
    normalized = str(mode).strip().lower()
    if normalized not in RANGE_ANCHOR_MODES:
        raise ValueError(f"range_anchor_mode must be one of {RANGE_ANCHOR_MODES}; got {mode!r}.")
    return normalized


def _normalize_coverage_calibration_mode(mode: str | None, fallback: str) -> str:
    """Return a known target-coverage calibration mode."""
    normalized = str(mode or fallback).strip().lower()
    if normalized not in RANGE_COVERAGE_CALIBRATION_MODES:
        raise ValueError(
            "coverage_calibration_mode must be one of "
            f"{RANGE_COVERAGE_CALIBRATION_MODES}; got {mode!r}."
        )
    return normalized


def _anchor_weights_for_mode(points: torch.Tensor, mode: str) -> tuple[torch.Tensor | None, float]:
    """Return optional point weights and use probability for the configured anchor mode."""
    normalized = _normalize_range_anchor_mode(mode)
    if normalized == "uniform":
        return None, 0.0
    if normalized == "sparse":
        return _sparse_anchor_weights(points), 1.0
    weights = _density_anchor_weights(points)
    if normalized == "dense":
        return weights, 1.0
    return weights, DENSITY_ANCHOR_PROBABILITY


def _anchor_weights_for_family(
    points: torch.Tensor,
    family: str,
) -> tuple[torch.Tensor | None, float]:
    """Return anchor weights for a range_workload_v1 anchor family."""
    normalized = str(family).strip().lower()
    if normalized == "sparse_background_control":
        return _sparse_anchor_weights(points), 1.0
    if normalized == "boundary_entry_exit":
        return _endpoint_anchor_weights(points), 1.0
    if normalized == "crossing_turn_change":
        return _turn_change_anchor_weights(points), 1.0
    if normalized == "port_or_approach_zone":
        return _port_or_approach_anchor_weights(points), 1.0
    if normalized == "density_route":
        return _density_anchor_weights(points), 1.0
    raise ValueError(f"Unknown range workload anchor family: {family!r}.")


def _weighted_choice(mapping: dict[str, float], generator: torch.Generator, fallback: str) -> str:
    """Sample one key from a non-negative weight mapping."""
    if not mapping:
        return fallback
    keys = [str(key) for key in mapping]
    weights = torch.tensor([max(0.0, float(mapping[key])) for key in keys], dtype=torch.float32)
    if float(weights.sum().item()) <= 0.0:
        return keys[int(torch.randint(0, len(keys), (1,), generator=generator).item())]
    idx = _weighted_sample_one(weights, generator)
    return keys[int(idx)]


def _deterministic_unit_from_payload(*parts: object) -> float:
    """Return a deterministic unit-uniform-like value from arbitrary key material."""
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    raw = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return (raw / float(1 << 64))


def _weighted_choice_with_deterministic_key(
    mapping: dict[str, float],
    generator: torch.Generator,
    fallback: str,
    deterministic_value: float | None = None,
) -> str:
    """Sample one key from a non-negative weight mapping using deterministic ordering."""
    if not mapping:
        return fallback
    keys = [str(key) for key in mapping]
    weights = torch.tensor([max(0.0, float(mapping[key])) for key in keys], dtype=torch.float32)
    total = float(weights.sum().item())
    if total <= 0.0:
        idx = _weighted_sample_one(torch.ones((max(1, len(keys)),), dtype=torch.float32), generator)
        return keys[min(int(idx), len(keys) - 1)]
    if deterministic_value is None:
        idx = _weighted_sample_one(weights, generator)
        return keys[int(idx)]
    u = float(deterministic_value) % 1.0
    cdf = torch.cumsum(weights, dim=0)
    target = u * total
    idx = int(torch.searchsorted(cdf, torch.tensor(target, dtype=cdf.dtype)).item())
    return keys[min(max(idx, 0), len(keys) - 1)]


def _profile_query_settings(
    profile: RangeWorkloadProfile,
    generator: torch.Generator,
    query_index: int | None = None,
    workload_seed: int | None = None,
) -> dict[str, Any]:
    """Sample query-level profile settings for one range_workload_v1 query."""
    if profile.profile_id == LEGACY_GENERATOR_PROFILE.profile_id:
        return {}
    query_key = str(int(workload_seed) if workload_seed is not None else "unseeded")
    if query_index is None or query_index < 0:
        anchor_value: float | None = None
        footprint_value: float | None = None
    else:
        anchor_value = _deterministic_unit_from_payload(profile.profile_id, "anchor", query_key, query_index)
        footprint_value = _deterministic_unit_from_payload(
            profile.profile_id,
            "footprint",
            query_key,
            query_index,
        )
    anchor_family = _weighted_choice_with_deterministic_key(
        profile.anchor_family_weights,
        generator,
        fallback="density_route",
        deterministic_value=anchor_value,
    )
    footprint_family = _weighted_choice_with_deterministic_key(
        profile.footprint_family_weights,
        generator,
        fallback="medium_operational",
        deterministic_value=footprint_value,
    )
    footprint = dict(profile.footprint_families.get(footprint_family) or {})
    return {
        "anchor_family": anchor_family,
        "footprint_family": footprint_family,
        "range_spatial_km": float(footprint.get("spatial_radius_km", 2.2)),
        "range_time_hours": float(footprint.get("time_half_window_hours", 5.0)),
        "elongation_allowed": bool(footprint.get("elongation_allowed", False)),
    }


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
    anchor_weights: torch.Tensor | None = None,
    anchor_weight_probability: float = 1.0,
) -> torch.Tensor:
    """Sample one point row from the cloud. See queries/README.md for details."""
    if candidate_mask is not None and bool(candidate_mask.any().item()):
        candidate_indices = torch.where(candidate_mask)[0]
    else:
        candidate_indices = None

    use_weighted = (
        anchor_weights is not None
        and anchor_weights.numel() == points.shape[0]
        and float(torch.rand(1, generator=generator).item()) < float(anchor_weight_probability)
    )
    if use_weighted:
        if anchor_weights is None:
            raise RuntimeError("Weighted anchor sampling requested without anchor weights.")
        if candidate_indices is not None:
            candidate_weights = anchor_weights[candidate_indices].float()
            if float(candidate_weights.sum().item()) > 0.0:
                sampled_candidate_offset = _weighted_sample_one(candidate_weights, generator)
                return points[int(candidate_indices[sampled_candidate_offset].item())]
        else:
            weights = anchor_weights.float()
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


def _normalize_range_time_domain_mode(mode: str) -> str:
    """Normalize range-query time-domain mode names."""
    normalized = str(mode).strip().lower()
    if normalized not in RANGE_TIME_DOMAIN_MODES:
        raise ValueError(
            f"range_time_domain_mode must be one of {RANGE_TIME_DOMAIN_MODES}; got {mode!r}."
        )
    return normalized


def _anchor_day_time_bounds(anchor_time: float, bounds: dict[str, float]) -> tuple[float, float]:
    """Return the 24-hour time domain containing the anchor point.

    AIS tensors usually carry seconds relative to the loaded CSV minimum. If a
    caller passes epoch-like seconds, align to calendar UTC day boundaries;
    otherwise align to 24-hour source-file chunks from the dataset lower bound.
    """
    dataset_min = float(bounds["t_min"])
    dataset_max = float(bounds["t_max"])
    if dataset_max <= dataset_min:
        return dataset_min, dataset_max

    if dataset_min >= EPOCH_LIKE_SECONDS:
        day_start = math.floor(float(anchor_time) / SECONDS_PER_DAY) * SECONDS_PER_DAY
    else:
        day_offset = max(0.0, float(anchor_time) - dataset_min)
        day_start = dataset_min + math.floor(day_offset / SECONDS_PER_DAY) * SECONDS_PER_DAY
    day_end = day_start + SECONDS_PER_DAY
    return max(dataset_min, day_start), min(dataset_max, day_end)


def _query_time_bounds_for_mode(
    anchor_time: float,
    bounds: dict[str, float],
    range_time_domain_mode: str,
) -> tuple[float, float]:
    """Return the allowed temporal clamp bounds for one range query."""
    mode = _normalize_range_time_domain_mode(range_time_domain_mode)
    if mode == "dataset":
        return float(bounds["t_min"]), float(bounds["t_max"])
    return _anchor_day_time_bounds(float(anchor_time), bounds)


def _make_range_query(
    points: torch.Tensor,
    bounds: dict[str, float],
    generator: torch.Generator,
    anchor_mask: torch.Tensor | None = None,
    anchor_weights: torch.Tensor | None = None,
    anchor_weight_probability: float = 1.0,
    range_spatial_fraction: float = DEFAULT_RANGE_SPATIAL_FRACTION,
    range_time_fraction: float = DEFAULT_RANGE_TIME_FRACTION,
    range_spatial_km: float | None = None,
    range_time_hours: float | None = None,
    range_footprint_jitter: float = DEFAULT_RANGE_FOOTPRINT_JITTER,
    range_time_domain_mode: str = DEFAULT_RANGE_TIME_DOMAIN_MODE,
    elongation_allowed: bool = False,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate one range query. See queries/README.md for details."""
    time_domain_mode = _normalize_range_time_domain_mode(range_time_domain_mode)
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
    anchor_point = _sample_anchor_point(
        points,
        generator,
        candidate_mask=anchor_mask,
        anchor_weights=anchor_weights,
        anchor_weight_probability=anchor_weight_probability,
    )
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
    corridor_axis = "none"
    elongation_factor = 1.0
    cross_axis_factor = 1.0
    if bool(elongation_allowed):
        heading = float(anchor_point[4].item()) if int(anchor_point.numel()) > 4 else 90.0
        heading_rad = math.radians(heading % 180.0)
        north_south_alignment = abs(math.cos(heading_rad))
        east_west_alignment = abs(math.sin(heading_rad))
        elongation_factor = 2.50
        cross_axis_factor = 0.45
        if east_west_alignment >= north_south_alignment:
            lon_w *= elongation_factor
            lat_w *= cross_axis_factor
            corridor_axis = "east_west"
        else:
            lat_w *= elongation_factor
            lon_w *= cross_axis_factor
            corridor_axis = "north_south"
    if time_hours is None:
        t_w = time_fraction * (bounds["t_max"] - bounds["t_min"]) * time_jitter
    else:
        t_w = time_hours * 3600.0 * time_jitter
    anchor_time = float(anchor_point[0].item())
    time_min, time_max = _query_time_bounds_for_mode(anchor_time, bounds, time_domain_mode)
    query = {
        "type": "range",
        "params": {
            "lat_min": float(max(bounds["lat_min"], anchor_point[1].item() - lat_w)),
            "lat_max": float(min(bounds["lat_max"], anchor_point[1].item() + lat_w)),
            "lon_min": float(max(bounds["lon_min"], anchor_point[2].item() - lon_w)),
            "lon_max": float(min(bounds["lon_max"], anchor_point[2].item() + lon_w)),
            "t_start": float(max(time_min, anchor_time - t_w)),
            "t_end": float(min(time_max, anchor_time + t_w)),
        },
    }
    query_metadata = dict(metadata or {})
    if bool(elongation_allowed):
        query_metadata.update(
            {
                "corridor_axis": corridor_axis,
                "corridor_elongation_factor": float(elongation_factor),
                "corridor_cross_axis_factor": float(cross_axis_factor),
            }
        )
    if query_metadata:
        query["_metadata"] = query_metadata
    return query


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


def _normalize_coverage_overshoot(range_max_coverage_overshoot: float | None) -> float | None:
    """Normalize coverage overshoot tolerances supplied as fractions or percentages."""
    if range_max_coverage_overshoot is None:
        return None
    tolerance = float(range_max_coverage_overshoot)
    if tolerance > 1.0:
        if tolerance <= 100.0:
            tolerance = tolerance / 100.0
        else:
            raise ValueError("range_max_coverage_overshoot must be a non-negative fraction or percent.")
    if tolerance < 0.0:
        raise ValueError("range_max_coverage_overshoot must be non-negative when provided.")
    return tolerance


def _finalize_workload(
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
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
    if typed_queries and all(str(query.get("type", "")).lower() == "range" for query in typed_queries):
        profile_metadata = diagnostics.get("workload_profile")
        diagnostics["workload_signature"] = _range_workload_signature(
            points=points,
            boundaries=boundaries,
            typed_queries=typed_queries,
            coverage_fraction=coverage_fraction,
            profile_metadata=profile_metadata if isinstance(profile_metadata, dict) else None,
        )
    return TypedQueryWorkload(
        query_features=features,
        typed_queries=typed_queries,
        type_ids=type_ids,
        coverage_fraction=coverage_fraction,
        covered_points=covered_points,
        total_points=total_points,
        generation_diagnostics=diagnostics,
    )


def _counts_from_metadata(typed_queries: list[dict[str, Any]], key: str) -> dict[str, int]:
    """Count query metadata values."""
    counts: dict[str, int] = {}
    for query in typed_queries:
        metadata = query.get("_metadata") or {}
        value = str(metadata.get(key, "legacy_or_unspecified"))
        counts[value] = int(counts.get(value, 0)) + 1
    return counts


def _range_workload_signature(
    *,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, Any]],
    coverage_fraction: float,
    profile_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return the v1 workload signature artifact described by the rework guide."""
    diagnostics = compute_range_workload_diagnostics(
        points=points,
        boundaries=boundaries,
        typed_queries=typed_queries,
        duplicate_iou_threshold=0.65,
        coverage_fraction=coverage_fraction,
    )
    summary = diagnostics["summary"]
    query_rows = diagnostics["queries"]
    point_hit_counts = [int(row["point_hits"]) for row in query_rows]
    trajectory_hit_counts = [int(row["trajectory_hits"]) for row in query_rows]
    spatial_radii = []
    time_spans = []
    for query in typed_queries:
        metadata = query.get("_metadata") or {}
        params = query["params"]
        spatial_radii.append(float(metadata.get("spatial_radius_km", metadata.get("range_spatial_km", 0.0)) or 0.0))
        time_spans.append(float(params["t_end"] - params["t_start"]) / 3600.0)

    def quantiles(values: list[float]) -> dict[str, float]:
        if not values:
            return {"p10": 0.0, "p50": 0.0, "p90": 0.0}
        tensor = torch.tensor(values, dtype=torch.float32)
        return {
            "p10": float(torch.quantile(tensor, 0.10).item()),
            "p50": float(torch.quantile(tensor, 0.50).item()),
            "p90": float(torch.quantile(tensor, 0.90).item()),
        }

    profile_id = "legacy_generator"
    if profile_metadata is not None:
        profile_id = str(profile_metadata.get("profile_id", profile_id))
    return {
        "profile_id": profile_id,
        "coverage_actual": float(coverage_fraction),
        "query_count": int(len(typed_queries)),
        "anchor_family_counts": _counts_from_metadata(typed_queries, "anchor_family"),
        "footprint_family_counts": _counts_from_metadata(typed_queries, "footprint_family"),
        "point_hits_per_query": {
            "p10": float(summary["point_hit_count_p10"]),
            "p50": float(summary["point_hit_count_p50"]),
            "p90": float(summary["point_hit_count_p90"]),
        },
        "point_hit_counts_per_query": point_hit_counts,
        "ship_hits_per_query": {
            "p10": float(summary["trajectory_hit_count_p10"]),
            "p50": float(summary["trajectory_hit_count_p50"]),
            "p90": float(summary["trajectory_hit_count_p90"]),
        },
        "ship_hit_counts_per_query": trajectory_hit_counts,
        "trajectory_hits_per_query": {
            "p10": float(summary["trajectory_hit_count_p10"]),
            "p50": float(summary["trajectory_hit_count_p50"]),
            "p90": float(summary["trajectory_hit_count_p90"]),
        },
        "trajectory_hit_counts_per_query": trajectory_hit_counts,
        "time_span_hours_per_query": quantiles(time_spans),
        "spatial_radius_km_per_query": quantiles(spatial_radii),
        "near_duplicate_rate": float(summary["near_duplicate_query_rate"]),
        "broad_query_rate": float(summary["too_broad_query_rate"]),
        "empty_query_rate": float(summary["empty_query_rate"]),
        "train_eval_signature_distance": None,
    }


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
    range_time_domain_mode: str = DEFAULT_RANGE_TIME_DOMAIN_MODE,
    range_anchor_mode: str = DEFAULT_RANGE_ANCHOR_MODE,
    range_min_point_hits: int | None = None,
    range_max_point_hit_fraction: float | None = None,
    range_min_trajectory_hits: int | None = None,
    range_max_trajectory_hit_fraction: float | None = None,
    range_max_box_volume_fraction: float | None = None,
    range_duplicate_iou_threshold: float | None = None,
    range_acceptance_max_attempts: int | None = None,
    range_max_coverage_overshoot: float | None = None,
    workload_profile_id: str | None = None,
    coverage_calibration_mode: str | None = None,
) -> TypedQueryWorkload:
    """Generate a range-query workload and padded feature tensor. See queries/README.md for details."""
    profile = range_workload_profile(workload_profile_id)
    profile_enabled = profile.profile_id != LEGACY_GENERATOR_PROFILE.profile_id
    time_domain_mode = _normalize_range_time_domain_mode(
        profile.time_domain_mode if profile_enabled else range_time_domain_mode
    )
    coverage_mode = _normalize_coverage_calibration_mode(
        coverage_calibration_mode,
        profile.coverage_calibration_mode if profile_enabled else "uncovered_anchor_chasing",
    )
    anchor_mode = _normalize_range_anchor_mode(range_anchor_mode)
    points = torch.cat(trajectories, dim=0)
    bounds = _dataset_bounds(points)
    boundaries = boundaries_from_trajectories(trajectories)

    normalize_pure_workload_map(workload_map)
    generator = torch.Generator().manual_seed(int(seed))
    anchor_weights, anchor_weight_probability = _anchor_weights_for_mode(points, anchor_mode)

    coverage_target = _normalize_target_coverage(target_coverage)
    coverage_overshoot = _normalize_coverage_overshoot(range_max_coverage_overshoot)
    if profile_enabled:
        if range_min_point_hits is None:
            range_min_point_hits = profile.min_points_per_query
        if range_min_trajectory_hits is None:
            range_min_trajectory_hits = profile.min_trajectories_per_query
        if range_max_point_hit_fraction is None:
            range_max_point_hit_fraction = max_point_hit_fraction_for_coverage(coverage_target)
        if range_duplicate_iou_threshold is None:
            range_duplicate_iou_threshold = profile.max_near_duplicate_hitset_jaccard
        if range_acceptance_max_attempts is None:
            range_acceptance_max_attempts = max(1, int(profile.max_attempt_multiplier) * max(1, int(n_queries)))
    coverage_guard_enabled = coverage_target is not None and coverage_overshoot is not None
    max_allowed_coverage = (
        min(1.0, float(coverage_target) + float(coverage_overshoot))
        if coverage_guard_enabled and coverage_target is not None and coverage_overshoot is not None
        else None
    )
    query_acceptance_enabled = _range_acceptance_enabled(
        range_min_point_hits,
        range_max_point_hit_fraction,
        range_min_trajectory_hits,
        range_max_trajectory_hit_fraction,
        range_max_box_volume_fraction,
        range_duplicate_iou_threshold,
    )
    acceptance_enabled = query_acceptance_enabled or coverage_guard_enabled
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

    def commit_query(query: dict[str, Any]) -> None:
        """Record a query as accepted after all filters have passed."""
        if not acceptance_enabled:
            return
        range_acceptance["accepted"] = int(range_acceptance["accepted"]) + 1
        accepted_range_queries.append({"params": query["params"], "query_index": len(accepted_range_queries)})

    def build_query(
        anchor_mask: torch.Tensor | None = None,
        query_index: int | None = None,
    ) -> dict[str, Any] | None:
        """Build one query, applying optional range acceptance filters."""
        if acceptance_enabled:
            if max_range_attempts is not None and int(range_acceptance["attempts"]) >= max_range_attempts:
                range_acceptance["exhausted"] = True
                return None
            range_acceptance["attempts"] = int(range_acceptance["attempts"]) + 1
        profile_query = (
            _profile_query_settings(
                profile,
                generator,
                query_index=query_index,
                workload_seed=int(seed),
            )
            if profile_enabled
            else {}
        )
        query_anchor_weights = anchor_weights
        query_anchor_probability = anchor_weight_probability
        query_spatial_km = range_spatial_km
        query_time_hours = range_time_hours
        query_metadata: dict[str, Any] = {}
        if profile_query:
            query_anchor_weights, query_anchor_probability = _anchor_weights_for_family(
                points,
                str(profile_query["anchor_family"]),
            )
            query_spatial_km = float(profile_query["range_spatial_km"])
            query_time_hours = float(profile_query["range_time_hours"])
            query_metadata = {
                "workload_profile_id": profile.profile_id,
                "anchor_family": str(profile_query["anchor_family"]),
                "footprint_family": str(profile_query["footprint_family"]),
                "spatial_radius_km": float(query_spatial_km),
                "time_half_window_hours": float(query_time_hours),
                "elongation_allowed": bool(profile_query.get("elongation_allowed", False)),
            }
        query = _make_range_query(
            points,
            bounds,
            generator,
            anchor_mask=anchor_mask,
            anchor_weights=query_anchor_weights,
            anchor_weight_probability=query_anchor_probability,
            range_spatial_fraction=range_spatial_fraction,
            range_time_fraction=range_time_fraction,
            range_spatial_km=query_spatial_km,
            range_time_hours=query_time_hours,
            range_footprint_jitter=range_footprint_jitter,
            range_time_domain_mode=time_domain_mode,
            elongation_allowed=bool(profile_query.get("elongation_allowed", False)),
            metadata=query_metadata,
        )
        if not query_acceptance_enabled:
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
        return query

    if coverage_target is not None:
        requested_queries = max(1, int(n_queries))
        if max_queries is not None and int(max_queries) <= 0:
            raise ValueError("max_queries must be positive when target_coverage is set.")
        generated_queries: list[dict[str, Any]] = []
        covered = torch.zeros((points.shape[0],), dtype=torch.bool, device=points.device)
        target_reached_query_count: int | None = None
        coverage_at_target_reached: float | None = None

        query_limit = max(requested_queries, int(max_queries) if max_queries is not None else requested_queries)
        stop_reason = "max_queries_reached"
        calibrated_query_count_mode = profile.query_count_mode == "calibrated_to_coverage"

        while len(generated_queries) < query_limit:
            current_coverage = float(covered.float().mean().item()) if points.shape[0] > 0 else 0.0
            if (
                not calibrated_query_count_mode
                and len(generated_queries) >= requested_queries
                and current_coverage >= coverage_target
            ):
                stop_reason = "target_coverage_reached"
                break
            anchor_mask = (
                (~covered)
                if coverage_mode == "uncovered_anchor_chasing" and current_coverage < coverage_target
                else None
            )
            query = build_query(anchor_mask=anchor_mask, query_index=len(generated_queries))
            if query is None:
                if range_acceptance.get("exhausted"):
                    stop_reason = "range_acceptance_exhausted"
                    break
                continue
            query_mask = point_coverage_mask_for_query(points, query)
            if coverage_guard_enabled and max_allowed_coverage is not None:
                candidate_coverage = (
                    float((covered | query_mask).float().mean().item()) if points.shape[0] > 0 else 0.0
                )
                if candidate_coverage > max_allowed_coverage:
                    _record_rejection(range_acceptance, "coverage_overshoot")
                    if (
                        max_range_attempts is not None
                        and int(range_acceptance.get("attempts", 0)) >= max_range_attempts
                    ):
                        range_acceptance["exhausted"] = True
                        stop_reason = "range_coverage_guard_exhausted"
                        break
                    continue
            commit_query(query)
            generated_queries.append(query)
            covered |= query_mask
            new_coverage = float(covered.float().mean().item()) if points.shape[0] > 0 else 0.0
            if target_reached_query_count is None and new_coverage >= coverage_target:
                target_reached_query_count = int(len(generated_queries))
                coverage_at_target_reached = float(new_coverage)
                if calibrated_query_count_mode:
                    stop_reason = "target_coverage_reached"
                    break

            final_coverage = float(covered.float().mean().item()) if points.shape[0] > 0 else 0.0
            if (
                stop_reason == "max_queries_reached"
                and len(generated_queries) >= requested_queries
                and final_coverage >= coverage_target
            ):
                stop_reason = "target_coverage_reached"
                break

        final_coverage = float(covered.float().mean().item()) if points.shape[0] > 0 else 0.0
        if (
            stop_reason == "max_queries_reached"
            and len(generated_queries) >= requested_queries
            and final_coverage >= coverage_target
        ):
            stop_reason = "target_coverage_reached"
        query_generation = {
            "mode": "target_coverage",
            "workload_profile_id": profile.profile_id,
            "workload_profile_version": int(profile.version),
            "query_count_mode": profile.query_count_mode,
            "coverage_calibration_mode": coverage_mode,
            "minimum_queries": int(requested_queries),
            "requested_queries": int(requested_queries),
            "max_queries": int(query_limit),
            "target_coverage": float(coverage_target),
            "range_time_domain_mode": time_domain_mode,
            "range_anchor_mode": anchor_mode,
            "range_spatial_fraction": float(range_spatial_fraction),
            "range_time_fraction": float(range_time_fraction),
            "range_spatial_km": None if range_spatial_km is None else float(range_spatial_km),
            "range_time_hours": None if range_time_hours is None else float(range_time_hours),
            "range_footprint_jitter": float(range_footprint_jitter),
            "range_max_coverage_overshoot": coverage_overshoot,
            "coverage_guard_enabled": bool(coverage_guard_enabled),
            "max_allowed_coverage": max_allowed_coverage,
            "stop_reason": stop_reason,
            "target_reached_query_count": target_reached_query_count,
            "coverage_at_target_reached": coverage_at_target_reached,
            "extra_queries_after_target_reached": (
                int(len(generated_queries) - target_reached_query_count)
                if target_reached_query_count is not None
                else None
            ),
        }
        return _finalize_workload(
            points,
            boundaries,
            generated_queries,
            generator,
            generation_diagnostics={
                "workload_profile": workload_profile_metadata(profile),
                "range_acceptance": range_acceptance,
                "query_generation": query_generation,
            },
        )

    generated_queries: list[dict[str, Any]] = []
    requested_queries = max(0, int(n_queries))
    stop_reason = "fixed_count_completed"
    while len(generated_queries) < requested_queries:
        query = build_query(query_index=len(generated_queries))
        if query is None:
            if range_acceptance.get("exhausted"):
                stop_reason = "range_acceptance_exhausted"
                break
            continue
        commit_query(query)
        generated_queries.append(query)

    return _finalize_workload(
        points,
        boundaries,
        generated_queries,
        generator,
        generation_diagnostics={
            "range_acceptance": range_acceptance,
            "workload_profile": workload_profile_metadata(profile),
            "query_generation": {
                "mode": "fixed_count",
                "workload_profile_id": profile.profile_id,
                "workload_profile_version": int(profile.version),
                "query_count_mode": profile.query_count_mode,
                "coverage_calibration_mode": coverage_mode,
                "minimum_queries": requested_queries,
                "requested_queries": requested_queries,
                "max_queries": requested_queries,
                "target_coverage": None,
                "range_time_domain_mode": time_domain_mode,
                "range_anchor_mode": anchor_mode,
                "range_spatial_fraction": float(range_spatial_fraction),
                "range_time_fraction": float(range_time_fraction),
                "range_spatial_km": None if range_spatial_km is None else float(range_spatial_km),
                "range_time_hours": None if range_time_hours is None else float(range_time_hours),
                "range_footprint_jitter": float(range_footprint_jitter),
                "range_max_coverage_overshoot": coverage_overshoot,
                "coverage_guard_enabled": False,
                "max_allowed_coverage": None,
                "stop_reason": stop_reason,
            },
        },
    )
