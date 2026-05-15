"""Train-derived query prior fields for QueryUsefulV1 models."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.functional as F

from queries.range_geometry import points_in_range_box, segment_box_bracket_indices

QUERY_PRIOR_FIELD_SCHEMA_VERSION = 3
QUERY_PRIOR_FIELD_NAMES = (
    "spatial_query_hit_probability",
    "spatiotemporal_query_hit_probability",
    "boundary_entry_exit_likelihood",
    "crossing_likelihood",
    "behavior_utility_prior",
    "route_density_prior",
)


def zero_query_prior_field_like(prior_field: dict[str, Any]) -> dict[str, Any]:
    """Return a zero-valued prior field that preserves train extent and provenance."""
    zeroed = copy.deepcopy(prior_field)
    for name in QUERY_PRIOR_FIELD_NAMES:
        value = zeroed.get(name)
        if isinstance(value, torch.Tensor):
            zeroed[name] = torch.zeros_like(value)
    diagnostics = dict(zeroed.get("diagnostics") or {})
    diagnostics["ablation"] = "zero_query_prior_features"
    diagnostics["zeroed_prior_features_preserve_train_extent"] = True
    zeroed["diagnostics"] = diagnostics
    zeroed["ablation"] = "zero_query_prior_features"
    zeroed["built_from_split"] = prior_field.get("built_from_split", "train_only")
    zeroed["contains_eval_queries"] = False
    zeroed["contains_validation_queries"] = False
    return zeroed


def _bounds(points: torch.Tensor, typed_queries: list[dict[str, Any]] | None = None) -> dict[str, float]:
    """Return train/query extent used for stable cross-day prior sampling."""
    if int(points.numel()) == 0:
        bounds = {"t_min": 0.0, "t_max": 1.0, "lat_min": 0.0, "lat_max": 1.0, "lon_min": 0.0, "lon_max": 1.0}
    else:
        bounds = {
            "t_min": float(points[:, 0].min().item()),
            "t_max": float(points[:, 0].max().item()),
            "lat_min": float(points[:, 1].min().item()),
            "lat_max": float(points[:, 1].max().item()),
            "lon_min": float(points[:, 2].min().item()),
            "lon_max": float(points[:, 2].max().item()),
        }
    for query in typed_queries or []:
        if str(query.get("type", "")).lower() != "range":
            continue
        params = query.get("params") or {}
        try:
            bounds["t_min"] = min(bounds["t_min"], float(params["t_start"]))
            bounds["t_max"] = max(bounds["t_max"], float(params["t_end"]))
            bounds["lat_min"] = min(bounds["lat_min"], float(params["lat_min"]))
            bounds["lat_max"] = max(bounds["lat_max"], float(params["lat_max"]))
            bounds["lon_min"] = min(bounds["lon_min"], float(params["lon_min"]))
            bounds["lon_max"] = max(bounds["lon_max"], float(params["lon_max"]))
        except (KeyError, TypeError, ValueError):
            continue
    return bounds


def _spatial_bins(points: torch.Tensor, extent: dict[str, float], grid_bins: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return lat/lon/cell ids under the fixed training extent."""
    bins = max(1, int(grid_bins))
    lat_span = max(1e-9, float(extent["lat_max"]) - float(extent["lat_min"]))
    lon_span = max(1e-9, float(extent["lon_max"]) - float(extent["lon_min"]))
    lat_bin = torch.clamp(
        ((points[:, 1].float() - float(extent["lat_min"])) / lat_span * bins).floor().long(),
        0,
        bins - 1,
    )
    lon_bin = torch.clamp(
        ((points[:, 2].float() - float(extent["lon_min"])) / lon_span * bins).floor().long(),
        0,
        bins - 1,
    )
    return lat_bin, lon_bin, lat_bin * bins + lon_bin


def _time_bins(points: torch.Tensor, time_bins: int) -> torch.Tensor:
    """Return clock-time bins independent of eval query content."""
    bins = max(1, int(time_bins))
    phase = torch.remainder(points[:, 0].float(), 86_400.0) / 86_400.0
    return torch.clamp((phase * bins).floor().long(), 0, bins - 1)


def _aggregate_point_values_to_grid(
    values: torch.Tensor,
    cell_ids: torch.Tensor,
    cell_count: int,
) -> torch.Tensor:
    """Average per-point values by spatial grid cell."""
    sums = torch.bincount(cell_ids.detach().cpu(), weights=values.detach().cpu().float(), minlength=cell_count)
    counts = torch.bincount(cell_ids.detach().cpu(), minlength=cell_count).float()
    return (sums / counts.clamp(min=1.0)).to(dtype=torch.float32)


def _smooth_spatial_grid(values: torch.Tensor, grid_bins: int, passes: int) -> torch.Tensor:
    """Smooth a flattened spatial grid with a small fixed Gaussian-like kernel."""
    bins = max(1, int(grid_bins))
    smooth_passes = max(0, int(passes))
    if bins <= 1 or smooth_passes <= 0 or int(values.numel()) != bins * bins:
        return values.to(dtype=torch.float32)
    grid = values.detach().cpu().float().reshape(1, 1, bins, bins)
    kernel = torch.tensor(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
        dtype=torch.float32,
    ).reshape(1, 1, 3, 3)
    kernel = kernel / kernel.sum()
    for _ in range(smooth_passes):
        grid = F.conv2d(F.pad(grid, (1, 1, 1, 1), mode="replicate"), kernel)
    return grid.reshape(bins * bins).clamp(0.0, 1.0)


def _aggregate_point_values_to_spacetime_grid(
    values: torch.Tensor,
    cell_ids: torch.Tensor,
    time_ids: torch.Tensor,
    cell_count: int,
    time_bins: int,
) -> torch.Tensor:
    """Average per-point values by spatial cell and clock-time bin."""
    ids = (time_ids.detach().cpu() * int(cell_count) + cell_ids.detach().cpu()).long()
    total = int(cell_count) * max(1, int(time_bins))
    sums = torch.bincount(ids, weights=values.detach().cpu().float(), minlength=total)
    counts = torch.bincount(ids, minlength=total).float()
    return (sums / counts.clamp(min=1.0)).to(dtype=torch.float32).reshape(max(1, int(time_bins)), int(cell_count))


def _interval_overlap_mask(edges: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    """Return bins whose half-open intervals overlap [lower, upper]."""
    lo = float(min(lower, upper))
    hi = float(max(lower, upper))
    return (edges[:-1] <= hi) & (edges[1:] >= lo)


def _clock_overlap_mask(time_bins: int, t_start: float, t_end: float) -> torch.Tensor:
    """Return clock bins overlapping an absolute query time interval."""
    bins = max(1, int(time_bins))
    edges = torch.linspace(0.0, 86_400.0, steps=bins + 1, dtype=torch.float32)
    duration = max(0.0, float(t_end) - float(t_start))
    if duration >= 86_400.0:
        return torch.ones((bins,), dtype=torch.bool)
    start = float(t_start) % 86_400.0
    end = float(t_end) % 86_400.0
    if start <= end:
        return _interval_overlap_mask(edges, start, end)
    return _interval_overlap_mask(edges, start, 86_400.0) | _interval_overlap_mask(edges, 0.0, end)


def _query_box_prior_grids(
    *,
    range_queries: list[dict[str, Any]],
    extent: dict[str, float],
    grid_bins: int,
    time_bins: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rasterize train query boxes into query-probability prior fields."""
    bins = max(1, int(grid_bins))
    clock_bins = max(1, int(time_bins))
    spatial = torch.zeros((bins, bins), dtype=torch.float32)
    spacetime = torch.zeros((clock_bins, bins, bins), dtype=torch.float32)
    if not range_queries:
        return spatial.reshape(bins * bins), spacetime.reshape(clock_bins, bins * bins)
    lat_edges = torch.linspace(float(extent["lat_min"]), float(extent["lat_max"]), steps=bins + 1, dtype=torch.float32)
    lon_edges = torch.linspace(float(extent["lon_min"]), float(extent["lon_max"]), steps=bins + 1, dtype=torch.float32)
    for query in range_queries:
        params = query.get("params") or {}
        try:
            lat_mask = _interval_overlap_mask(lat_edges, float(params["lat_min"]), float(params["lat_max"]))
            lon_mask = _interval_overlap_mask(lon_edges, float(params["lon_min"]), float(params["lon_max"]))
            time_mask = _clock_overlap_mask(clock_bins, float(params["t_start"]), float(params["t_end"]))
        except (KeyError, TypeError, ValueError):
            continue
        spatial_mask = lat_mask[:, None] & lon_mask[None, :]
        spatial += spatial_mask.float()
        spacetime[time_mask] += spatial_mask.float()
    query_count = float(max(1, len(range_queries)))
    return (
        (spatial / query_count).reshape(bins * bins).clamp(0.0, 1.0),
        (spacetime / query_count).reshape(clock_bins, bins * bins).clamp(0.0, 1.0),
    )


def _canonical_out_of_extent_sampling(mode: str | None) -> str:
    """Normalize query-prior out-of-extent sampling mode."""
    normalized = str(mode or "zero").strip().lower().replace("-", "_")
    if normalized in {"clamp", "clamped"}:
        normalized = "nearest"
    if normalized not in {"zero", "nearest"}:
        raise ValueError(f"Unknown out_of_extent_sampling mode: {mode!r}")
    return normalized


def _smooth_spacetime_grid(values: torch.Tensor, grid_bins: int, time_bins: int, passes: int) -> torch.Tensor:
    """Smooth every time slice of a spatial-temporal grid."""
    bins = max(1, int(grid_bins))
    clock_bins = max(1, int(time_bins))
    smooth_passes = max(0, int(passes))
    if (
        bins <= 1
        or smooth_passes <= 0
        or values.shape != (clock_bins, bins * bins)
    ):
        return values.to(dtype=torch.float32)
    smoothed = [
        _smooth_spatial_grid(values[time_idx], bins, smooth_passes)
        for time_idx in range(clock_bins)
    ]
    return torch.stack(smoothed, dim=0).clamp(0.0, 1.0)


def _boundary_indices(range_mask: torch.Tensor, boundaries: list[tuple[int, int]]) -> torch.Tensor:
    """Return entry/exit indices for a range mask."""
    parts: list[torch.Tensor] = []
    mask_cpu = range_mask.detach().cpu()
    for start, end in boundaries:
        local = mask_cpu[start:end]
        if not bool(local.any().item()):
            continue
        enters = torch.zeros_like(local)
        exits = torch.zeros_like(local)
        enters[1:] = local[1:] & ~local[:-1]
        enters[0] = local[0]
        exits[:-1] = local[:-1] & ~local[1:]
        exits[-1] = local[-1]
        offsets = torch.where(enters | exits)[0]
        if int(offsets.numel()) > 0:
            parts.append(offsets.to(dtype=torch.long) + int(start))
    if not parts:
        return torch.empty((0,), dtype=torch.long, device=range_mask.device)
    return torch.cat(parts).to(device=range_mask.device, dtype=torch.long).unique(sorted=True)


def build_train_query_prior_fields(
    *,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, Any]],
    labels: torch.Tensor | None = None,
    behavior_values: torch.Tensor | None = None,
    workload_profile_id: str = "range_workload_v1",
    train_workload_seed: int | None = None,
    grid_bins: int = 64,
    time_bins: int = 24,
    smoothing_passes: int = 2,
    out_of_extent_sampling: str = "zero",
) -> dict[str, Any]:
    """Build query-prior fields from training points and training queries only."""
    out_of_extent_sampling = _canonical_out_of_extent_sampling(out_of_extent_sampling)
    range_queries = [query for query in typed_queries if str(query.get("type", "")).lower() == "range"]
    extent = _bounds(points, range_queries)
    bins = max(1, int(grid_bins))
    clock_bins = max(1, int(time_bins))
    cell_count = bins * bins
    _lat_bin, _lon_bin, cell_ids = _spatial_bins(points, extent, bins)
    point_count = int(points.shape[0])
    query_hits = torch.zeros((point_count,), dtype=torch.float32, device=points.device)
    boundary_hits = torch.zeros_like(query_hits)
    crossing_hits = torch.zeros_like(query_hits)
    points_cpu = points.detach().cpu()

    for query in range_queries:
        mask = points_in_range_box(points, query["params"])
        if bool(mask.any().item()):
            query_hits[mask] += 1.0
        boundary_idx = _boundary_indices(mask, boundaries)
        if int(boundary_idx.numel()) > 0:
            boundary_hits[boundary_idx] += 1.0
        crossing_idx = segment_box_bracket_indices(points_cpu, boundaries, query["params"]).to(device=points.device)
        if int(crossing_idx.numel()) > 0:
            crossing_hits[crossing_idx] += 1.0

    query_count = float(max(1, len(range_queries)))
    query_probability = (query_hits / query_count).clamp(0.0, 1.0)
    boundary_probability = (boundary_hits / query_count).clamp(0.0, 1.0)
    crossing_probability = (crossing_hits / query_count).clamp(0.0, 1.0)
    if behavior_values is not None and int(behavior_values.numel()) == point_count:
        behavior_values = behavior_values.reshape(point_count).float().clamp(0.0, 1.0).to(device=points.device)
    elif labels is not None and labels.ndim == 2 and labels.shape[0] == point_count:
        behavior_values = (
            labels[:, 0].float().clamp(0.0, 1.0)
            if labels.shape[1] == 1
            else labels.max(dim=1).values.float().clamp(0.0, 1.0)
        )
    else:
        behavior_values = query_probability

    route_counts = torch.bincount(cell_ids.detach().cpu(), minlength=cell_count).float()
    route_density = torch.log1p(route_counts) / torch.log1p(route_counts.max().clamp(min=1.0))
    smooth_passes = max(0, int(smoothing_passes))
    raw_spatial_query_grid, raw_spatiotemporal_query_grid = _query_box_prior_grids(
        range_queries=range_queries,
        extent=extent,
        grid_bins=bins,
        time_bins=clock_bins,
    )
    spatial_query_grid = _smooth_spatial_grid(
        raw_spatial_query_grid,
        bins,
        smooth_passes,
    )
    spatiotemporal_query_grid = _smooth_spacetime_grid(
        raw_spatiotemporal_query_grid,
        bins,
        clock_bins,
        smooth_passes,
    )
    raw_point_hit_grid = _aggregate_point_values_to_grid(query_probability, cell_ids, cell_count)
    boundary_grid = _smooth_spatial_grid(
        _aggregate_point_values_to_grid(boundary_probability, cell_ids, cell_count),
        bins,
        smooth_passes,
    )
    crossing_grid = _smooth_spatial_grid(
        _aggregate_point_values_to_grid(crossing_probability, cell_ids, cell_count),
        bins,
        smooth_passes,
    )
    behavior_grid = _smooth_spatial_grid(
        _aggregate_point_values_to_grid(behavior_values, cell_ids, cell_count),
        bins,
        smooth_passes,
    )
    route_density_grid = _smooth_spatial_grid(route_density.to(dtype=torch.float32), bins, smooth_passes)

    return {
        "schema_version": int(QUERY_PRIOR_FIELD_SCHEMA_VERSION),
        "field_names": list(QUERY_PRIOR_FIELD_NAMES),
        "profile_id": str(workload_profile_id),
        "built_from_split": "train_only",
        "train_workload_seed": train_workload_seed,
        "contains_eval_queries": False,
        "contains_validation_queries": False,
        "grid_projection": "lat_lon_training_extent",
        "spatial_query_field_source": "train_query_box_density",
        "out_of_extent_sampling": out_of_extent_sampling,
        "grid_bins": int(bins),
        "time_bins": int(clock_bins),
        "smoothing": {
            "kernel": "3x3_binomial",
            "passes": int(smooth_passes),
        },
        "extent": extent,
        "spatial_query_hit_probability": spatial_query_grid,
        "spatiotemporal_query_hit_probability": spatiotemporal_query_grid,
        "boundary_entry_exit_likelihood": boundary_grid,
        "crossing_likelihood": crossing_grid,
        "behavior_utility_prior": behavior_grid,
        "route_density_prior": route_density_grid,
        "diagnostics": {
            "train_query_count": int(len(range_queries)),
            "train_point_count": point_count,
            "smoothing_passes": int(smooth_passes),
            "spatial_query_field_source": "train_query_box_density",
            "raw_nonzero_spatial_query_cells": int((raw_spatial_query_grid > 0.0).sum().item()),
            "nonzero_spatial_query_cells": int((spatial_query_grid > 0.0).sum().item()),
            "raw_nonzero_point_hit_cells": int((raw_point_hit_grid > 0.0).sum().item()),
            "raw_nonzero_spatiotemporal_query_cells": int((raw_spatiotemporal_query_grid > 0.0).sum().item()),
            "nonzero_spatiotemporal_query_cells": int((spatiotemporal_query_grid > 0.0).sum().item()),
            "contains_eval_queries": False,
        },
    }


def sample_query_prior_fields(points: torch.Tensor, prior_field: dict[str, Any] | None) -> torch.Tensor:
    """Sample train-derived prior fields at point coordinates and clock bins."""
    if prior_field is None:
        return torch.zeros((int(points.shape[0]), len(QUERY_PRIOR_FIELD_NAMES)), dtype=torch.float32, device=points.device)
    bins = int(prior_field.get("grid_bins", 64))
    time_bins = int(prior_field.get("time_bins", 24))
    extent = dict(prior_field.get("extent") or _bounds(points))
    out_of_extent_sampling = _canonical_out_of_extent_sampling(prior_field.get("out_of_extent_sampling", "zero"))
    sampling_points = points.detach()
    if out_of_extent_sampling == "nearest":
        sampling_points = points.detach().clone()
        sampling_points[:, 1] = torch.clamp(
            sampling_points[:, 1],
            min=float(extent["lat_min"]),
            max=float(extent["lat_max"]),
        )
        sampling_points[:, 2] = torch.clamp(
            sampling_points[:, 2],
            min=float(extent["lon_min"]),
            max=float(extent["lon_max"]),
        )
    _lat_bin, _lon_bin, cell_ids = _spatial_bins(sampling_points.to(dtype=torch.float32), extent, bins)
    time_ids = _time_bins(points, time_bins)
    cell_ids_cpu = cell_ids.detach().cpu().long()
    time_ids_cpu = time_ids.detach().cpu().long()
    lat = points[:, 1].detach().cpu().float()
    lon = points[:, 2].detach().cpu().float()
    outside_extent = (
        (lat < float(extent["lat_min"]))
        | (lat > float(extent["lat_max"]))
        | (lon < float(extent["lon_min"]))
        | (lon > float(extent["lon_max"]))
    )

    def spatial(name: str) -> torch.Tensor:
        values = prior_field.get(name)
        if not isinstance(values, torch.Tensor):
            return torch.zeros((int(points.shape[0]),), dtype=torch.float32)
        return values.detach().cpu().float()[cell_ids_cpu]

    spatiotemporal = prior_field.get("spatiotemporal_query_hit_probability")
    if isinstance(spatiotemporal, torch.Tensor):
        st_values = spatiotemporal.detach().cpu().float()[time_ids_cpu, cell_ids_cpu]
    else:
        st_values = torch.zeros((int(points.shape[0]),), dtype=torch.float32)

    sampled = torch.stack(
        [
            spatial("spatial_query_hit_probability"),
            st_values,
            spatial("boundary_entry_exit_likelihood"),
            spatial("crossing_likelihood"),
            spatial("behavior_utility_prior"),
            spatial("route_density_prior"),
        ],
        dim=1,
    )
    if out_of_extent_sampling == "zero" and bool(outside_extent.any().item()):
        sampled[outside_extent] = 0.0
    return sampled.to(device=points.device, dtype=torch.float32).clamp(0.0, 1.0)


def query_prior_field_metadata(prior_field: dict[str, Any] | None) -> dict[str, Any]:
    """Return JSON-safe prior-field provenance."""
    if prior_field is None:
        return {"available": False}
    return {
        "available": True,
        "schema_version": int(prior_field.get("schema_version", 0)),
        "field_names": list(prior_field.get("field_names", [])),
        "profile_id": prior_field.get("profile_id"),
        "built_from_split": prior_field.get("built_from_split"),
        "train_workload_seed": prior_field.get("train_workload_seed"),
        "grid_projection": prior_field.get("grid_projection"),
        "spatial_query_field_source": prior_field.get("spatial_query_field_source"),
        "out_of_extent_sampling": prior_field.get("out_of_extent_sampling"),
        "grid_bins": prior_field.get("grid_bins"),
        "time_bins": prior_field.get("time_bins"),
        "smoothing": prior_field.get("smoothing"),
        "contains_eval_queries": bool(prior_field.get("contains_eval_queries", True)),
        "contains_validation_queries": bool(prior_field.get("contains_validation_queries", True)),
        "ablation": prior_field.get("ablation"),
        "diagnostics": dict(prior_field.get("diagnostics") or {}),
    }
