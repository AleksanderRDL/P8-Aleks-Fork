"""Method evaluation and fixed-width results table helpers. See src/evaluation/README.md for details."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, cast

import torch

from src.evaluation.baselines import Method
from src.evaluation.metrics import (
    MethodEvaluation,
    _polyline_length_km,
    _trajectory_sed_ped_km,
    clustering_f1,
    compute_geometric_distortion,
    compute_length_preservation,
    f1_score,
)
from src.evaluation.query_cache import (
    EvaluationQueryCache,
    RangeQueryAuditSupport,
    RangeSegmentAuditGeometry,
    RangeTrajectoryAuditSupport,
    split_by_boundaries,
)
from src.evaluation.range_usefulness import (
    RANGE_USEFULNESS_SCHEMA_VERSION,
    RANGE_USEFULNESS_WEIGHTS,
    range_usefulness_weight_summary,
)
from src.evaluation.tables import (
    print_geometric_distortion_table,
    print_method_comparison_table,
    print_range_usefulness_table,
    print_shift_table,
)
from src.queries.query_executor import execute_typed_query
from src.queries.range_geometry import segment_box_bracket_indices, segment_pairs_box_crossings
from src.queries.query_types import normalize_pure_workload_map

POINT_AWARE_KNN_REPRESENTATIVES_PER_TRAJECTORY = 64
POINT_AWARE_SIMILARITY_REPRESENTATIVES_PER_TRAJECTORY = 64


def _range_box_mask(points: torch.Tensor, params: dict[str, float]) -> torch.Tensor:
    """Return point-level hits inside a range query box."""
    return (
        (points[:, 1] >= params["lat_min"])
        & (points[:, 1] <= params["lat_max"])
        & (points[:, 2] >= params["lon_min"])
        & (points[:, 2] <= params["lon_max"])
        & (points[:, 0] >= params["t_start"])
        & (points[:, 0] <= params["t_end"])
    )


def _range_point_f1(retained_mask: torch.Tensor, range_mask: torch.Tensor) -> float:
    """Compute range F1 over retained point instances inside the query box."""
    return _point_subset_f1(retained_mask.to(device=range_mask.device, dtype=torch.bool), range_mask)


def score_range_boundary_preservation(
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    retained_mask: torch.Tensor,
    typed_queries: list[dict],
    query_cache: EvaluationQueryCache | None = None,
) -> float:
    """Score retained range entry/exit points separately from range point F1."""
    audit = score_range_usefulness(
        points=points,
        boundaries=boundaries,
        retained_mask=retained_mask,
        typed_queries=typed_queries,
        query_cache=query_cache,
    )
    return float(audit.get("range_entry_exit_f1", 0.0))


def _trajectory_id_per_point(n_points: int, boundaries: list[tuple[int, int]], device: torch.device) -> torch.Tensor:
    trajectory_ids = torch.full((n_points,), -1, dtype=torch.long, device=device)
    for trajectory_id, (start, end) in enumerate(boundaries):
        if end > start:
            trajectory_ids[start:end] = int(trajectory_id)
    return trajectory_ids


def _ids_mask(point_trajectory_ids: torch.Tensor, trajectory_ids: set[int]) -> torch.Tensor:
    mask = torch.zeros_like(point_trajectory_ids, dtype=torch.bool)
    for trajectory_id in trajectory_ids:
        mask |= point_trajectory_ids == int(trajectory_id)
    return mask


def _haversine_km(lat1: torch.Tensor, lon1: torch.Tensor, lat2: float, lon2: float) -> torch.Tensor:
    radius_km = 6371.0
    lat1_rad = torch.deg2rad(lat1)
    lon1_rad = torch.deg2rad(lon1)
    lat2_rad = torch.deg2rad(torch.tensor(lat2, dtype=lat1.dtype, device=lat1.device))
    lon2_rad = torch.deg2rad(torch.tensor(lon2, dtype=lon1.dtype, device=lon1.device))
    delta_lat = lat1_rad - lat2_rad
    delta_lon = lon1_rad - lon2_rad
    haversine = (
        torch.sin(delta_lat / 2.0) ** 2
        + torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(delta_lon / 2.0) ** 2
    )
    central_angle = 2.0 * torch.atan2(torch.sqrt(haversine), torch.sqrt(torch.clamp(1.0 - haversine, min=1e-9)))
    return radius_km * central_angle


def _point_subset_f1(retained_mask: torch.Tensor, support_mask: torch.Tensor) -> float:
    full_hits = int(support_mask.sum().item())
    if full_hits <= 0:
        return 1.0
    retained_hits = int((retained_mask & support_mask).sum().item())
    if retained_hits <= 0:
        return 0.0
    recall = float(retained_hits / full_hits)
    return float((2.0 * recall) / (1.0 + recall))


def _point_index_subset_f1(retained_mask: torch.Tensor, support_indices_cpu: torch.Tensor) -> float:
    """Compute support-point F1 from compact support indices."""
    full_hits = int(support_indices_cpu.numel())
    if full_hits <= 0:
        return 1.0
    support_indices = support_indices_cpu.to(device=retained_mask.device, dtype=torch.long)
    retained_hits = int(retained_mask[support_indices].sum().item())
    if retained_hits <= 0:
        return 0.0
    recall = float(retained_hits / full_hits)
    return float((2.0 * recall) / (1.0 + recall))


def _mean(values: list[float], default: float = 0.0) -> float:
    """Return a float mean with an explicit empty-list default."""
    return float(sum(values) / len(values)) if values else float(default)


def _trajectory_ids_for_mask(mask: torch.Tensor, point_trajectory_ids: torch.Tensor) -> list[int]:
    """Return sorted trajectory IDs with at least one true point in a mask."""
    if not bool(mask.any().item()):
        return []
    ids = torch.unique(point_trajectory_ids[mask])
    return sorted(int(value) for value in ids.detach().cpu().tolist() if int(value) >= 0)


def _range_boundary_indices_for_trajectories(
    range_mask: torch.Tensor,
    boundaries: list[tuple[int, int]],
    trajectory_ids: list[int],
) -> torch.Tensor:
    """Return compact in-box entry/exit point indices for hit trajectories."""
    boundary_indices: list[torch.Tensor] = []
    for trajectory_id in trajectory_ids:
        if trajectory_id < 0 or trajectory_id >= len(boundaries):
            continue
        start, end = boundaries[trajectory_id]
        if end <= start:
            continue
        traj_in = range_mask[start:end]
        if not bool(traj_in.any().item()):
            continue
        enters = torch.zeros_like(traj_in)
        enters[1:] = traj_in[1:] & ~traj_in[:-1]
        enters[0] = traj_in[0]
        exits = torch.zeros_like(traj_in)
        exits[:-1] = traj_in[:-1] & ~traj_in[1:]
        exits[-1] = traj_in[-1]
        local_indices = torch.where(enters | exits)[0]
        if local_indices.numel() > 0:
            boundary_indices.append((local_indices + int(start)).detach().cpu())
    if not boundary_indices:
        return torch.empty((0,), dtype=torch.long)
    return torch.cat(boundary_indices).to(dtype=torch.long)


def _range_crossing_bracket_indices_for_trajectories(
    points_cpu: torch.Tensor,
    params: dict[str, float],
    boundaries: list[tuple[int, int]],
    segment_geometry: RangeSegmentAuditGeometry | None = None,
) -> torch.Tensor:
    """Return point pairs bracketing range-box boundary/pass-through crossings."""
    if segment_geometry is not None:
        candidates = (
            (segment_geometry.time_max_cpu >= float(params["t_start"]))
            & (segment_geometry.time_min_cpu <= float(params["t_end"]))
            & (segment_geometry.lat_max_cpu >= float(params["lat_min"]))
            & (segment_geometry.lat_min_cpu <= float(params["lat_max"]))
            & (segment_geometry.lon_max_cpu >= float(params["lon_min"]))
            & (segment_geometry.lon_min_cpu <= float(params["lon_max"]))
        )
        candidate_starts = segment_geometry.start_indices_cpu[candidates]
        if candidate_starts.numel() == 0:
            return torch.empty((0,), dtype=torch.long)
        crossing = segment_pairs_box_crossings(
            points_cpu[candidate_starts],
            points_cpu[candidate_starts + 1],
            params,
        )
        crossing_starts = candidate_starts[crossing]
        if crossing_starts.numel() == 0:
            return torch.empty((0,), dtype=torch.long)
        bracket_indices = torch.empty((crossing_starts.numel() * 2,), dtype=torch.long)
        bracket_indices[0::2] = crossing_starts
        bracket_indices[1::2] = crossing_starts + 1
        return torch.unique(bracket_indices, sorted=True).to(dtype=torch.long)
    return segment_box_bracket_indices(points_cpu, boundaries, params).detach().cpu()


def _range_turn_weights_for_points(points_cpu: torch.Tensor) -> torch.Tensor:
    """Return retained-independent route-change weights for one in-query trajectory slice."""
    count = int(points_cpu.shape[0])
    weights = torch.zeros((count,), dtype=torch.float32)
    if count >= 3:
        coords = points_cpu[:, 1:3].float()
        before = torch.linalg.vector_norm(coords[1:-1] - coords[:-2], dim=1)
        after = torch.linalg.vector_norm(coords[2:] - coords[1:-1], dim=1)
        shortcut = torch.linalg.vector_norm(coords[2:] - coords[:-2], dim=1)
        curvature = torch.clamp(before + after - shortcut, min=0.0)
        weights[1:-1] = curvature
    if points_cpu.shape[1] >= 8:
        weights = torch.maximum(weights, points_cpu[:, 7].float().clamp(min=0.0))
    return weights


def _build_range_query_audit_support(
    points_cpu: torch.Tensor,
    boundaries: list[tuple[int, int]],
    range_mask: torch.Tensor,
    point_trajectory_ids: torch.Tensor,
    params: dict[str, float],
    segment_geometry: RangeSegmentAuditGeometry | None = None,
) -> RangeQueryAuditSupport:
    """Build retained-independent support for one range query."""
    range_mask = range_mask.bool()
    full_ids = tuple(_trajectory_ids_for_mask(range_mask, point_trajectory_ids))
    boundary_indices_cpu = _range_boundary_indices_for_trajectories(range_mask, boundaries, list(full_ids))
    crossing_bracket_indices_cpu = _range_crossing_bracket_indices_for_trajectories(
        points_cpu,
        params,
        boundaries,
        segment_geometry,
    )
    range_mask_cpu = range_mask.detach().cpu()
    trajectory_support: list[RangeTrajectoryAuditSupport] = []
    for trajectory_id in full_ids:
        if trajectory_id < 0 or trajectory_id >= len(boundaries):
            continue
        start, end = boundaries[trajectory_id]
        if end <= start:
            continue
        in_offsets = torch.where(range_mask_cpu[start:end])[0].cpu()
        if in_offsets.numel() == 0:
            continue

        times = points_cpu[start:end, 0]
        full_span = float((times[in_offsets[-1]] - times[in_offsets[0]]).item())
        full_points = points_cpu[start + in_offsets]
        full_length = _polyline_length_km(full_points[:, 1], full_points[:, 2])
        turn_weights = _range_turn_weights_for_points(full_points)
        trajectory_support.append(
            RangeTrajectoryAuditSupport(
                trajectory_id=int(trajectory_id),
                start=int(start),
                in_offsets_cpu=in_offsets,
                turn_weights_cpu=turn_weights.cpu(),
                full_time_span=float(full_span),
                full_length_km=float(full_length),
            )
        )

    return RangeQueryAuditSupport(
        range_mask=range_mask,
        boundary_indices_cpu=boundary_indices_cpu,
        crossing_bracket_indices_cpu=crossing_bracket_indices_cpu,
        full_trajectory_ids=full_ids,
        trajectories=tuple(trajectory_support),
    )


def _range_query_audit_support(
    *,
    points: torch.Tensor,
    points_cpu: torch.Tensor,
    boundaries: list[tuple[int, int]],
    query_index: int,
    query: dict,
    point_trajectory_ids: torch.Tensor,
    query_cache: EvaluationQueryCache | None,
) -> RangeQueryAuditSupport:
    """Return retained-independent audit support, using caller cache when available."""
    def build_range_mask() -> torch.Tensor:
        return _range_box_mask(points, query["params"])

    def build_support() -> RangeQueryAuditSupport:
        if query_cache is not None:
            range_mask = query_cache.get_support_mask(query_index, build_range_mask)
            segment_geometry = query_cache.get_range_segment_geometry(points_cpu, boundaries)
        else:
            range_mask = build_range_mask()
            segment_geometry = None
        return _build_range_query_audit_support(
            points_cpu=points_cpu,
            boundaries=boundaries,
            range_mask=range_mask,
            point_trajectory_ids=point_trajectory_ids,
            params=query["params"],
            segment_geometry=segment_geometry,
        )

    if query_cache is not None:
        return query_cache.get_range_audit_support(query_index, build_support)
    return build_support()


def _range_gap_coverage_for_offsets(in_offsets: torch.Tensor, retained_offsets: torch.Tensor) -> float:
    """Score whether retained in-query points avoid one large local gap."""
    full_count = int(in_offsets.numel())
    retained_count = int(retained_offsets.numel())
    if full_count <= 0:
        return 1.0
    if full_count == 1:
        return 1.0 if retained_count > 0 else 0.0
    if retained_count <= 0:
        return 0.0

    retained_positions = torch.searchsorted(in_offsets, retained_offsets)
    leading_missing = retained_positions[0]
    trailing_missing = (full_count - 1) - retained_positions[-1]
    max_missing = torch.maximum(leading_missing, trailing_missing)
    if retained_positions.numel() >= 2:
        interior_missing = retained_positions[1:] - retained_positions[:-1] - 1
        if interior_missing.numel() > 0:
            max_missing = torch.maximum(max_missing, interior_missing.max())

    denom = float(max(1, full_count - 2))
    return float(max(0.0, min(1.0, 1.0 - float(max_missing.item()) / denom)))


def _range_ship_coverage_for_offsets(in_offsets: torch.Tensor, retained_offsets: torch.Tensor) -> float:
    """Return per-ship point-subset F1 for one in-query trajectory slice."""
    full_count = int(in_offsets.numel())
    if full_count <= 0:
        return 1.0
    retained_count = int(retained_offsets.numel())
    if retained_count <= 0:
        return 0.0
    recall = float(retained_count / full_count)
    return float((2.0 * recall) / (1.0 + recall))


def _range_turn_coverage_for_mask(turn_weights: torch.Tensor, retained_local: torch.Tensor) -> float:
    """Return weighted point-subset F1 over route-change support."""
    turn_weights = turn_weights.to(dtype=torch.float32).clamp(min=0.0)
    full_mass = float(turn_weights.sum().item())
    if full_mass <= 1e-12:
        return 1.0
    retained_mass = float(turn_weights[retained_local].sum().item())
    if retained_mass <= 0.0:
        return 0.0
    recall = retained_mass / full_mass
    return float((2.0 * recall) / (1.0 + recall))


def _range_trajectory_detail_scores_for_query(
    points_cpu: torch.Tensor,
    retained_cpu: torch.Tensor,
    trajectory_support: tuple[RangeTrajectoryAuditSupport, ...],
) -> tuple[float, float, float, float, float]:
    """Return query-level per-ship coverage, temporal, gap, turn, and route-fidelity scores."""
    ship_coverage_scores: list[float] = []
    temporal_scores: list[float] = []
    gap_scores: list[float] = []
    turn_scores: list[float] = []
    shape_scores: list[float] = []
    times = points_cpu[:, 0]
    for support in trajectory_support:
        in_offsets = support.in_offsets_cpu
        start = int(support.start)
        retained_local = retained_cpu[start + in_offsets]
        retained_offsets = in_offsets[retained_local]
        if retained_offsets.numel() == 0:
            ship_coverage_scores.append(0.0)
            temporal_scores.append(0.0)
            gap_scores.append(0.0)
            turn_scores.append(0.0)
            shape_scores.append(0.0)
            continue

        ship_coverage_scores.append(_range_ship_coverage_for_offsets(in_offsets, retained_offsets))

        if support.full_time_span <= 1e-9:
            temporal_score = 1.0
        elif retained_offsets.numel() < 2:
            temporal_score = 0.0
        else:
            retained_span = float((times[start + retained_offsets[-1]] - times[start + retained_offsets[0]]).item())
            temporal_score = float(max(0.0, min(1.0, retained_span / support.full_time_span)))
        temporal_scores.append(temporal_score)
        gap_scores.append(_range_gap_coverage_for_offsets(in_offsets, retained_offsets))
        turn_scores.append(_range_turn_coverage_for_mask(support.turn_weights_cpu, retained_local))

        if support.full_length_km <= 1e-9:
            shape_scores.append(1.0)
        elif retained_offsets.numel() < 2:
            shape_scores.append(0.0)
        else:
            local_points = points_cpu[start + in_offsets]
            sed_sum, _sed_max, ped_sum, _ped_max, removed_count = _trajectory_sed_ped_km(
                local_points[:, 0],
                local_points[:, 1],
                local_points[:, 2],
                retained_local,
            )
            if removed_count <= 0:
                fidelity = 1.0
            else:
                avg_error_km = (sed_sum + ped_sum) / float(2 * removed_count)
                avg_segment_km = support.full_length_km / float(max(1, int(in_offsets.numel()) - 1))
                fidelity = 1.0 / (1.0 + avg_error_km / max(avg_segment_km, 1e-6))
            shape_scores.append(float(max(0.0, min(1.0, temporal_score * fidelity))))

    return (
        _mean(ship_coverage_scores, default=1.0),
        _mean(temporal_scores, default=1.0),
        _mean(gap_scores, default=1.0),
        _mean(turn_scores, default=1.0),
        _mean(shape_scores, default=1.0),
    )


def score_range_usefulness(
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    retained_mask: torch.Tensor,
    typed_queries: list[dict],
    query_cache: EvaluationQueryCache | None = None,
) -> dict[str, Any]:
    """Audit range simplification with point, ship, crossing, temporal, gap, turn, and shape components.

    `range_point_f1` is the current retained in-box point metric. The combined
    `range_usefulness_score` is an audit score for comparing candidates; it is
    intentionally reported separately instead of replacing final F1 semantics.
    """
    if query_cache is not None:
        query_cache.validate(points, boundaries, typed_queries)

    retained_bool = retained_mask.to(device=points.device, dtype=torch.bool)
    point_trajectory_ids = _trajectory_id_per_point(points.shape[0], boundaries, points.device)
    points_cpu = points.detach().cpu()
    retained_cpu = retained_bool.detach().cpu()

    point_scores: list[float] = []
    ship_scores: list[float] = []
    ship_coverage_scores: list[float] = []
    entry_exit_scores: list[float] = []
    crossing_scores: list[float] = []
    temporal_scores: list[float] = []
    gap_scores: list[float] = []
    turn_scores: list[float] = []
    shape_scores: list[float] = []

    for query_index, query in enumerate(typed_queries):
        if str(query.get("type", "")).lower() != "range":
            continue
        support = _range_query_audit_support(
            points=points,
            points_cpu=points_cpu,
            boundaries=boundaries,
            query_index=query_index,
            query=query,
            point_trajectory_ids=point_trajectory_ids,
            query_cache=query_cache,
        )
        range_mask = support.range_mask.to(device=points.device, dtype=torch.bool)
        retained_in_range = retained_bool & range_mask
        point_scores.append(_range_point_f1(retained_bool, range_mask))

        retained_ids = _trajectory_ids_for_mask(retained_in_range, point_trajectory_ids)
        ship_scores.append(f1_score(set(support.full_trajectory_ids), set(retained_ids)))

        entry_exit_scores.append(_point_index_subset_f1(retained_bool, support.boundary_indices_cpu))
        crossing_scores.append(_point_index_subset_f1(retained_bool, support.crossing_bracket_indices_cpu))

        ship_coverage, temporal_score, gap_score, turn_score, shape_score = _range_trajectory_detail_scores_for_query(
            points_cpu=points_cpu,
            retained_cpu=retained_cpu,
            trajectory_support=support.trajectories,
        )
        ship_coverage_scores.append(ship_coverage)
        temporal_scores.append(temporal_score)
        gap_scores.append(gap_score)
        turn_scores.append(turn_score)
        shape_scores.append(shape_score)

    query_count = len(point_scores)
    range_point_f1 = _mean(point_scores)
    range_ship_f1 = _mean(ship_scores)
    range_ship_coverage = _mean(ship_coverage_scores)
    range_entry_exit_f1 = _mean(entry_exit_scores)
    range_crossing_f1 = _mean(crossing_scores)
    range_temporal_coverage = _mean(temporal_scores)
    range_gap_coverage = _mean(gap_scores)
    range_turn_coverage = _mean(turn_scores)
    range_shape_score = _mean(shape_scores)
    range_usefulness_score = sum(
        float(RANGE_USEFULNESS_WEIGHTS[name]) * value
        for name, value in (
            ("range_point_f1", range_point_f1),
            ("range_ship_f1", range_ship_f1),
            ("range_ship_coverage", range_ship_coverage),
            ("range_entry_exit_f1", range_entry_exit_f1),
            ("range_crossing_f1", range_crossing_f1),
            ("range_temporal_coverage", range_temporal_coverage),
            ("range_gap_coverage", range_gap_coverage),
            ("range_turn_coverage", range_turn_coverage),
            ("range_shape_score", range_shape_score),
        )
    )
    return {
        "range_usefulness_schema_version": int(RANGE_USEFULNESS_SCHEMA_VERSION),
        "range_query_count": int(query_count),
        "range_point_f1": float(range_point_f1),
        "pure_range_f1": float(range_point_f1),
        "range_ship_f1": float(range_ship_f1),
        "range_ship_coverage": float(range_ship_coverage),
        "range_entry_exit_f1": float(range_entry_exit_f1),
        "range_crossing_f1": float(range_crossing_f1),
        "range_temporal_coverage": float(range_temporal_coverage),
        "range_gap_coverage": float(range_gap_coverage),
        "range_turn_coverage": float(range_turn_coverage),
        "range_shape_score": float(range_shape_score),
        "range_usefulness_score": float(range_usefulness_score),
        "range_usefulness_weights": dict(RANGE_USEFULNESS_WEIGHTS),
        "range_usefulness_weight_summary": range_usefulness_weight_summary(),
    }


def _knn_representative_mask(
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    trajectory_ids: set[int],
    params: dict[str, float],
    representatives_per_trajectory: int = POINT_AWARE_KNN_REPRESENTATIVES_PER_TRAJECTORY,
) -> torch.Tensor:
    support = torch.zeros((points.shape[0],), dtype=torch.bool, device=points.device)
    time_start = float(params["t_center"] - params["t_half_window"])
    time_end = float(params["t_center"] + params["t_half_window"])
    limit = int(representatives_per_trajectory)
    for trajectory_id in sorted(trajectory_ids):
        if trajectory_id < 0 or trajectory_id >= len(boundaries):
            continue
        start, end = boundaries[trajectory_id]
        trajectory_points = points[start:end]
        in_window = (trajectory_points[:, 0] >= time_start) & (trajectory_points[:, 0] <= time_end)
        candidate_offsets = torch.where(in_window)[0]
        if candidate_offsets.numel() == 0:
            continue
        if limit > 0 and candidate_offsets.numel() > limit:
            candidates = trajectory_points[candidate_offsets]
            distance = _haversine_km(candidates[:, 1], candidates[:, 2], float(params["lat"]), float(params["lon"]))
            distance = distance + 0.001 * torch.abs(candidates[:, 0] - float(params["t_center"]))
            candidate_offsets = candidate_offsets[torch.topk(-distance, k=limit).indices]
        support[start + candidate_offsets] = True
    return support


def _similarity_support_mask(
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    trajectory_ids: set[int],
    query: dict,
    representatives_per_trajectory: int = POINT_AWARE_SIMILARITY_REPRESENTATIVES_PER_TRAJECTORY,
) -> torch.Tensor:
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


def _clustering_support_mask(
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    labels: list[int],
    params: dict[str, float],
) -> torch.Tensor:
    clustered_ids = {trajectory_id for trajectory_id, label in enumerate(labels) if int(label) != -1}
    if not clustered_ids:
        return torch.zeros((points.shape[0],), dtype=torch.bool, device=points.device)
    point_trajectory_ids = _trajectory_id_per_point(points.shape[0], boundaries, points.device)
    return _range_box_mask(points, params) & _ids_mask(point_trajectory_ids, clustered_ids)


def score_retained_mask(
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    retained_mask: torch.Tensor,
    typed_queries: list[dict],
    workload_map: dict[str, float],
    query_cache: EvaluationQueryCache | None = None,
) -> tuple[float, dict[str, float], float, dict[str, float]]:
    """Score a precomputed retained mask with the final query-F1 semantics.

    Returns (aggregate_answer_f1, per_type_answer_f1, aggregate_combined,
    per_type_combined). The "answer" variant uses pure set/cluster F1 between
    queries on the full vs simplified data (the natural, defensible metric).
    The "combined" variant is the legacy answer_f1 * point_subset_f1 product
    kept for diagnostic comparison; it double-penalizes a method that returns
    the right answer set via different points than ground truth's "support".
    """
    if query_cache is not None:
        query_cache.validate(points, boundaries, typed_queries)

    full_traj: list[torch.Tensor] | None = None
    simplified: torch.Tensor | None = None
    simp_boundaries: list[tuple[int, int]] | None = None
    simp_traj: list[torch.Tensor] | None = None

    def full_views() -> list[torch.Tensor]:
        nonlocal full_traj
        if query_cache is not None:
            return query_cache.get_full_traj(points, boundaries)
        if full_traj is None:
            full_traj = split_by_boundaries(points, boundaries)
        return full_traj

    def simplified_views() -> tuple[torch.Tensor, list[torch.Tensor], list[tuple[int, int]]]:
        nonlocal simplified, simp_boundaries, simp_traj
        if simplified is None or simp_boundaries is None or simp_traj is None:
            simplified = points[retained_mask]
            simp_boundaries = []
            cursor = 0
            for start, end in boundaries:
                n = int(retained_mask[start:end].sum().item())
                simp_boundaries.append((cursor, cursor + n))
                cursor += n
            simp_traj = split_by_boundaries(simplified, simp_boundaries)
        return simplified, simp_traj, simp_boundaries

    def full_result(query_index: int, query: dict) -> Any:
        def build() -> Any:
            return execute_typed_query(points, full_views(), query, boundaries)

        if query_cache is not None:
            return query_cache.get_full_result(query_index, build)
        return build()

    def support_mask(query_index: int, builder: Callable[[], torch.Tensor]) -> torch.Tensor:
        if query_cache is not None:
            return query_cache.get_support_mask(query_index, builder)
        return builder()

    answer_scores: dict[str, list[float]] = {"range": [], "knn": [], "similarity": [], "clustering": []}
    combined_scores: dict[str, list[float]] = {"range": [], "knn": [], "similarity": [], "clustering": []}
    for query_index, query in enumerate(typed_queries):
        qtype = query["type"]
        if qtype == "range":
            range_mask = support_mask(query_index, lambda query=query: _range_box_mask(points, query["params"]))
            point_f1 = _range_point_f1(retained_mask, range_mask)
            answer_scores[qtype].append(point_f1)
            combined_scores[qtype].append(point_f1)
            continue

        full_res = full_result(query_index, query)
        simplified_now, simp_traj_now, simp_boundaries_now = simplified_views()
        simp_res = execute_typed_query(simplified_now, simp_traj_now, query, simp_boundaries_now)
        if qtype == "knn":
            answer_f1 = f1_score(set(full_res), set(simp_res))
            support = support_mask(
                query_index,
                lambda full_res=full_res, query=query: _knn_representative_mask(
                    points,
                    boundaries,
                    set(full_res),
                    query["params"],
                ),
            )
            answer_scores[qtype].append(answer_f1)
            combined_scores[qtype].append(answer_f1 * _point_subset_f1(retained_mask, support))
        elif qtype == "similarity":
            answer_f1 = f1_score(set(full_res), set(simp_res))
            support = support_mask(
                query_index,
                lambda full_res=full_res, query=query: _similarity_support_mask(points, boundaries, set(full_res), query),
            )
            answer_scores[qtype].append(answer_f1)
            combined_scores[qtype].append(answer_f1 * _point_subset_f1(retained_mask, support))
        elif qtype == "clustering":
            full_labels = cast(list[int], full_res)
            simp_labels = cast(list[int], simp_res)
            answer_f1 = clustering_f1(full_labels, simp_labels)
            support = support_mask(
                query_index,
                lambda full_res=full_labels, query=query: _clustering_support_mask(
                    points,
                    boundaries,
                    list(full_res),
                    query["params"],
                ),
            )
            answer_scores[qtype].append(answer_f1)
            combined_scores[qtype].append(answer_f1 * _point_subset_f1(retained_mask, support))

    per_type_answer = {name: (sum(v) / len(v) if v else 0.0) for name, v in answer_scores.items()}
    per_type_combined = {name: (sum(v) / len(v) if v else 0.0) for name, v in combined_scores.items()}
    workload_weights = normalize_pure_workload_map(workload_map)
    normalized_map = {name: workload_weights.get(name, 0.0) for name in per_type_answer}
    aggregate_answer = sum(normalized_map[name] * per_type_answer[name] for name in per_type_answer)
    aggregate_combined = sum(normalized_map[name] * per_type_combined[name] for name in per_type_combined)
    return float(aggregate_answer), per_type_answer, float(aggregate_combined), per_type_combined


def _retained_point_gap_stats(
    retained_mask: torch.Tensor,
    boundaries: list[tuple[int, int]],
) -> tuple[float, float, float]:
    """Return average and max original-index gaps between retained points."""
    total_gap = 0.0
    total_norm_gap = 0.0
    max_gap = 0.0
    gap_count = 0
    for start, end in boundaries:
        n = int(end - start)
        if n <= 1:
            continue
        offsets = torch.where(retained_mask[start:end])[0].float()
        if offsets.numel() < 2:
            continue
        gaps = offsets[1:] - offsets[:-1]
        denom = float(max(1, n - 1))
        total_gap += float(gaps.sum().item())
        total_norm_gap += float((gaps / denom).sum().item())
        max_gap = max(max_gap, float(gaps.max().item()))
        gap_count += int(gaps.numel())

    if gap_count <= 0:
        return 0.0, 0.0, 0.0
    return float(total_gap / gap_count), float(total_norm_gap / gap_count), float(max_gap)


def evaluate_method(
    method: Method,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict],
    workload_map: dict[str, float],
    compression_ratio: float,
    return_mask: bool = False,
    query_cache: EvaluationQueryCache | None = None,
) -> MethodEvaluation:
    """Evaluate one simplification method on typed queries at matched ratio. See src/evaluation/README.md for details."""
    t0 = time.time()
    retained_mask = method.simplify(points, boundaries, compression_ratio)
    latency_ms = (time.time() - t0) * 1000.0

    range_only = bool(typed_queries) and all(str(query.get("type", "")).lower() == "range" for query in typed_queries)
    range_audit: dict[str, Any] | None = None
    if range_only:
        range_audit = score_range_usefulness(
            points=points,
            boundaries=boundaries,
            retained_mask=retained_mask,
            typed_queries=typed_queries,
            query_cache=query_cache,
        )
        range_point = float(range_audit.get("range_point_f1", 0.0))
        aggregate = range_point
        aggregate_combined = range_point
        per_type = {"range": range_point, "knn": 0.0, "similarity": 0.0, "clustering": 0.0}
        per_type_combined = dict(per_type)
    else:
        aggregate, per_type, aggregate_combined, per_type_combined = score_retained_mask(
            points=points,
            boundaries=boundaries,
            retained_mask=retained_mask,
            typed_queries=typed_queries,
            workload_map=workload_map,
            query_cache=query_cache,
        )
    comp = float(retained_mask.float().mean().item())
    avg_gap, avg_norm_gap, max_gap = _retained_point_gap_stats(retained_mask, boundaries)
    geometric = compute_geometric_distortion(points, boundaries, retained_mask)
    avg_length_preserved = compute_length_preservation(points, boundaries, retained_mask)
    combined = float(aggregate) * max(0.0, min(1.0, avg_length_preserved))
    if range_audit is None:
        range_audit = score_range_usefulness(
            points=points,
            boundaries=boundaries,
            retained_mask=retained_mask,
            typed_queries=typed_queries,
            query_cache=query_cache,
        )
    boundary_f1 = float(range_audit.get("range_entry_exit_f1", 0.0))

    return MethodEvaluation(
        aggregate_f1=float(aggregate),
        per_type_f1=per_type,
        aggregate_combined_f1=float(aggregate_combined),
        per_type_combined_f1=per_type_combined,
        compression_ratio=comp,
        latency_ms=latency_ms,
        avg_retained_point_gap=avg_gap,
        avg_retained_point_gap_norm=avg_norm_gap,
        max_retained_point_gap=max_gap,
        geometric_distortion=geometric,
        avg_length_preserved=avg_length_preserved,
        combined_query_shape_score=combined,
        range_point_f1=float(range_audit.get("range_point_f1", per_type.get("range", 0.0))),
        range_ship_f1=float(range_audit.get("range_ship_f1", 0.0)),
        range_ship_coverage=float(range_audit.get("range_ship_coverage", 0.0)),
        range_entry_exit_f1=boundary_f1,
        range_crossing_f1=float(range_audit.get("range_crossing_f1", 0.0)),
        range_temporal_coverage=float(range_audit.get("range_temporal_coverage", 0.0)),
        range_gap_coverage=float(range_audit.get("range_gap_coverage", 0.0)),
        range_turn_coverage=float(range_audit.get("range_turn_coverage", 0.0)),
        range_shape_score=float(range_audit.get("range_shape_score", 0.0)),
        range_usefulness_score=float(range_audit.get("range_usefulness_score", 0.0)),
        range_usefulness_schema_version=int(range_audit.get("range_usefulness_schema_version", 0) or 0),
        range_audit=range_audit,
        retained_mask=retained_mask if return_mask else None,
    )
