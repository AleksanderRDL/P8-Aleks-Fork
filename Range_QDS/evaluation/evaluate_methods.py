"""Method evaluation and fixed-width results table helpers. See evaluation/README.md for details."""

from __future__ import annotations

import time
from typing import Any, cast

import torch

from data.trajectory_index import (
    trajectory_ids_for_points,
    trajectory_ids_from_mask,
)
from evaluation.baselines import Method
from evaluation.metrics import (
    KM_PER_DEG_LAT,
    MethodEvaluation,
    _cumulative_polyline_length_km,
    _polyline_length_km,
    _trajectory_sed_ped_km,
    compute_geometric_distortion,
    compute_length_preservation,
    f1_score,
)
from evaluation.query_cache import (
    EvaluationQueryCache,
    RangeQueryAuditSupport,
    RangeSegmentAuditGeometry,
    RangeTrajectoryAuditSupport,
)
from evaluation.range_usefulness import (
    RANGE_USEFULNESS_SCHEMA_VERSION,
    RANGE_USEFULNESS_WEIGHTS,
    range_usefulness_gap_ablation_scores,
    range_usefulness_score_from_components,
    range_usefulness_weight_summary,
)
from evaluation.query_useful_v1 import query_useful_v1_from_range_audit
from queries.range_geometry import (
    points_in_range_box,
    segment_box_bracket_indices,
    segment_pairs_box_crossings,
)
from queries.query_types import normalize_pure_workload_map


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
    full_ids = tuple(trajectory_ids_from_mask(range_mask, point_trajectory_ids))
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
        distance_offsets = _cumulative_polyline_length_km(full_points[:, 1], full_points[:, 2])
        turn_weights = _range_turn_weights_for_points(full_points)
        trajectory_support.append(
            RangeTrajectoryAuditSupport(
                trajectory_id=int(trajectory_id),
                start=int(start),
                end=int(end),
                in_offsets_cpu=in_offsets,
                turn_weights_cpu=turn_weights.cpu(),
                distance_offsets_km_cpu=distance_offsets.cpu(),
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
        return points_in_range_box(points, query["params"])

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


def _range_gap_span_coverage_for_positions(
    values: torch.Tensor,
    retained_positions: torch.Tensor,
    full_span: float,
) -> float:
    """Score the largest missing run by elapsed time or along-track distance."""
    full_count = int(values.numel())
    retained_count = int(retained_positions.numel())
    if full_count <= 0:
        return 1.0
    if full_count == 1:
        return 1.0 if retained_count > 0 else 0.0
    if retained_count <= 0:
        return 0.0
    if full_span <= 1e-9:
        return 1.0

    retained_positions = retained_positions.to(dtype=torch.long)
    max_gap = values.new_tensor(0.0, dtype=torch.float32)
    first_pos = int(retained_positions[0].item())
    last_pos = int(retained_positions[-1].item())
    if first_pos > 0:
        max_gap = torch.maximum(max_gap, values[first_pos].float() - values[0].float())
    if last_pos < full_count - 1:
        max_gap = torch.maximum(max_gap, values[-1].float() - values[last_pos].float())
    if retained_positions.numel() >= 2:
        left = retained_positions[:-1]
        right = retained_positions[1:]
        has_missing = (right - left) > 1
        if bool(has_missing.any().item()):
            interior_spans = values[right[has_missing]].float() - values[left[has_missing]].float()
            max_gap = torch.maximum(max_gap, interior_spans.max())

    return float(max(0.0, min(1.0, 1.0 - float(max_gap.item()) / float(full_span))))


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


def _query_local_interpolation_fidelity(
    *,
    points_cpu: torch.Tensor,
    retained_cpu: torch.Tensor,
    support: RangeTrajectoryAuditSupport,
) -> float:
    """Score reconstruction of in-query points from retained full-trajectory anchors."""
    start = int(support.start)
    end = int(support.end)
    if end <= start:
        return 1.0
    in_offsets = support.in_offsets_cpu.long()
    if int(in_offsets.numel()) <= 0:
        return 1.0
    local_points = points_cpu[start:end]
    full_retained = retained_cpu[start:end].bool()
    retained_idx = torch.where(full_retained)[0]
    if int(retained_idx.numel()) < 2:
        return 0.0
    query_retained = full_retained[in_offsets]
    removed_offsets = in_offsets[~query_retained]
    if int(removed_offsets.numel()) == 0:
        return 1.0
    pos = torch.searchsorted(retained_idx, removed_offsets)
    valid = (pos > 0) & (pos < int(retained_idx.numel()))
    if not bool(valid.any().item()):
        return 0.0
    removed_offsets = removed_offsets[valid]
    pos = pos[valid]
    left_idx = retained_idx[pos - 1]
    right_idx = retained_idx[pos]
    times = local_points[:, 0].float()
    lats = local_points[:, 1].float()
    lons = local_points[:, 2].float()
    t_l = times[left_idx]
    t_r = times[right_idx]
    t_p = times[removed_offsets]
    alpha = ((t_p - t_l) / (t_r - t_l).clamp(min=1e-9)).clamp(0.0, 1.0)
    interp_lat = lats[left_idx] + alpha * (lats[right_idx] - lats[left_idx])
    interp_lon = lons[left_idx] + alpha * (lons[right_idx] - lons[left_idx])
    cos_lat = torch.cos(torch.deg2rad(lats[removed_offsets]))
    dx_km = (lons[removed_offsets] - interp_lon) * cos_lat * KM_PER_DEG_LAT
    dy_km = (lats[removed_offsets] - interp_lat) * KM_PER_DEG_LAT
    sed_km = torch.sqrt(dx_km * dx_km + dy_km * dy_km)

    cos_lat_left = torch.cos(torch.deg2rad(lats[left_idx]))
    bx_km = (lons[right_idx] - lons[left_idx]) * cos_lat_left * KM_PER_DEG_LAT
    by_km = (lats[right_idx] - lats[left_idx]) * KM_PER_DEG_LAT
    px_km = (lons[removed_offsets] - lons[left_idx]) * cos_lat_left * KM_PER_DEG_LAT
    py_km = (lats[removed_offsets] - lats[left_idx]) * KM_PER_DEG_LAT
    chord_len = torch.sqrt(bx_km * bx_km + by_km * by_km).clamp(min=1e-9)
    ped_km = torch.abs(bx_km * py_km - by_km * px_km) / chord_len
    avg_error_km = float(((sed_km + ped_km) * 0.5).mean().item())
    avg_segment_km = support.full_length_km / float(max(1, int(in_offsets.numel()) - 1))
    return float(max(0.0, min(1.0, 1.0 / (1.0 + avg_error_km / max(avg_segment_km, 1e-6)))))


def _range_trajectory_detail_scores_for_query(
    points_cpu: torch.Tensor,
    retained_cpu: torch.Tensor,
    trajectory_support: tuple[RangeTrajectoryAuditSupport, ...],
) -> tuple[float, float, float, float, float, float, float, float]:
    """Return query-level per-ship coverage, temporal, gap, turn, and route-fidelity scores."""
    ship_coverage_scores: list[float] = []
    temporal_scores: list[float] = []
    gap_scores: list[float] = []
    gap_time_scores: list[float] = []
    gap_distance_scores: list[float] = []
    turn_scores: list[float] = []
    shape_scores: list[float] = []
    interpolation_scores: list[float] = []
    interpolation_scores: list[float] = []
    interpolation_scores: list[float] = []
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
            gap_time_scores.append(0.0)
            gap_distance_scores.append(0.0)
            turn_scores.append(0.0)
            shape_scores.append(0.0)
            interpolation_scores.append(
                _query_local_interpolation_fidelity(
                    points_cpu=points_cpu,
                    retained_cpu=retained_cpu,
                    support=support,
                )
            )
            continue

        ship_coverage_scores.append(_range_ship_coverage_for_offsets(in_offsets, retained_offsets))
        retained_positions = torch.searchsorted(in_offsets, retained_offsets)

        if support.full_time_span <= 1e-9:
            temporal_score = 1.0
        elif retained_offsets.numel() < 2:
            temporal_score = 0.0
        else:
            retained_span = float((times[start + retained_offsets[-1]] - times[start + retained_offsets[0]]).item())
            temporal_score = float(max(0.0, min(1.0, retained_span / support.full_time_span)))
        temporal_scores.append(temporal_score)
        gap_scores.append(_range_gap_coverage_for_offsets(in_offsets, retained_offsets))
        gap_time_scores.append(
            _range_gap_span_coverage_for_positions(
                values=times[start + in_offsets].cpu(),
                retained_positions=retained_positions,
                full_span=support.full_time_span,
            )
        )
        gap_distance_scores.append(
            _range_gap_span_coverage_for_positions(
                values=support.distance_offsets_km_cpu,
                retained_positions=retained_positions,
                full_span=support.full_length_km,
            )
        )
        turn_scores.append(_range_turn_coverage_for_mask(support.turn_weights_cpu, retained_local))
        interpolation_scores.append(
            _query_local_interpolation_fidelity(
                points_cpu=points_cpu,
                retained_cpu=retained_cpu,
                support=support,
            )
        )

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
        _mean(gap_time_scores, default=1.0),
        _mean(gap_distance_scores, default=1.0),
        _mean(turn_scores, default=1.0),
        _mean(shape_scores, default=1.0),
        _mean(interpolation_scores, default=1.0),
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
    point_trajectory_ids = trajectory_ids_for_points(points.shape[0], boundaries, points.device)
    points_cpu = points.detach().cpu()
    retained_cpu = retained_bool.detach().cpu()

    point_scores: list[float] = []
    ship_scores: list[float] = []
    ship_coverage_scores: list[float] = []
    entry_exit_scores: list[float] = []
    crossing_scores: list[float] = []
    temporal_scores: list[float] = []
    gap_scores: list[float] = []
    gap_time_scores: list[float] = []
    gap_distance_scores: list[float] = []
    turn_scores: list[float] = []
    shape_scores: list[float] = []
    interpolation_scores: list[float] = []

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

        retained_ids = trajectory_ids_from_mask(retained_in_range, point_trajectory_ids)
        ship_scores.append(f1_score(set(support.full_trajectory_ids), set(retained_ids)))

        entry_exit_scores.append(_point_index_subset_f1(retained_bool, support.boundary_indices_cpu))
        crossing_scores.append(_point_index_subset_f1(retained_bool, support.crossing_bracket_indices_cpu))

        (
            ship_coverage,
            temporal_score,
            gap_score,
            gap_time_score,
            gap_distance_score,
            turn_score,
            shape_score,
            interpolation_score,
        ) = _range_trajectory_detail_scores_for_query(
            points_cpu=points_cpu,
            retained_cpu=retained_cpu,
            trajectory_support=support.trajectories,
        )
        ship_coverage_scores.append(ship_coverage)
        temporal_scores.append(temporal_score)
        gap_scores.append(gap_score)
        gap_time_scores.append(gap_time_score)
        gap_distance_scores.append(gap_distance_score)
        turn_scores.append(turn_score)
        shape_scores.append(shape_score)
        interpolation_scores.append(interpolation_score)

    query_count = len(point_scores)
    range_point_f1 = _mean(point_scores)
    range_ship_f1 = _mean(ship_scores)
    range_ship_coverage = _mean(ship_coverage_scores)
    range_entry_exit_f1 = _mean(entry_exit_scores)
    range_crossing_f1 = _mean(crossing_scores)
    range_temporal_coverage = _mean(temporal_scores)
    range_gap_coverage = _mean(gap_scores)
    range_gap_time_coverage = _mean(gap_time_scores)
    range_gap_distance_coverage = _mean(gap_distance_scores)
    range_turn_coverage = _mean(turn_scores)
    range_shape_score = _mean(shape_scores)
    range_query_local_interpolation_fidelity = _mean(interpolation_scores)
    components = {
        "range_point_f1": range_point_f1,
        "range_ship_f1": range_ship_f1,
        "range_ship_coverage": range_ship_coverage,
        "range_entry_exit_f1": range_entry_exit_f1,
        "range_crossing_f1": range_crossing_f1,
        "range_temporal_coverage": range_temporal_coverage,
        "range_gap_coverage": range_gap_coverage,
        "range_gap_time_coverage": range_gap_time_coverage,
        "range_gap_distance_coverage": range_gap_distance_coverage,
        "range_turn_coverage": range_turn_coverage,
        "range_shape_score": range_shape_score,
        "range_query_local_interpolation_fidelity": range_query_local_interpolation_fidelity,
    }
    range_usefulness_score = range_usefulness_score_from_components(components)
    gap_ablation_scores = range_usefulness_gap_ablation_scores(components)
    return {
        "range_usefulness_schema_version": int(RANGE_USEFULNESS_SCHEMA_VERSION),
        "range_query_count": int(query_count),
        "range_point_f1": float(range_point_f1),
        "range_ship_f1": float(range_ship_f1),
        "range_ship_coverage": float(range_ship_coverage),
        "range_entry_exit_f1": float(range_entry_exit_f1),
        "range_crossing_f1": float(range_crossing_f1),
        "range_temporal_coverage": float(range_temporal_coverage),
        "range_gap_coverage": float(range_gap_coverage),
        "range_gap_time_coverage": float(range_gap_time_coverage),
        "range_gap_distance_coverage": float(range_gap_distance_coverage),
        "range_gap_min_coverage": float(gap_ablation_scores["range_gap_min_coverage"]),
        "range_turn_coverage": float(range_turn_coverage),
        "range_shape_score": float(range_shape_score),
        "range_query_local_interpolation_fidelity": float(range_query_local_interpolation_fidelity),
        "range_usefulness_score": float(range_usefulness_score),
        "range_usefulness_gap_time_score": float(gap_ablation_scores["range_usefulness_gap_time_score"]),
        "range_usefulness_gap_distance_score": float(
            gap_ablation_scores["range_usefulness_gap_distance_score"]
        ),
        "range_usefulness_gap_min_score": float(gap_ablation_scores["range_usefulness_gap_min_score"]),
        "range_usefulness_gap_ablation_version": int(
            gap_ablation_scores["range_usefulness_gap_ablation_version"]
        ),
        "range_usefulness_weights": dict(RANGE_USEFULNESS_WEIGHTS),
        "range_usefulness_weight_summary": range_usefulness_weight_summary(),
    }


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
    per_type_combined). Range workloads use point-subset F1 over each query box;
    answer and combined scores are identical under the range-only contract.
    """
    if query_cache is not None:
        query_cache.validate(points, boundaries, typed_queries)

    workload_weights = normalize_pure_workload_map(workload_map)
    scores: list[float] = []
    for query_index, query in enumerate(typed_queries):
        qtype = str(query.get("type", "")).lower()
        if qtype != "range":
            raise ValueError(f"Only range queries are supported for evaluation; got query type: {qtype}")
        if query_cache is not None:
            range_mask = query_cache.get_support_mask(
                query_index,
                lambda query=query: points_in_range_box(points, query["params"]),
            )
        else:
            range_mask = points_in_range_box(points, query["params"])
        scores.append(_range_point_f1(retained_mask, range_mask))

    range_score = _mean(scores)
    aggregate = float(workload_weights["range"] * range_score)
    per_type = {"range": range_score}
    return aggregate, per_type, aggregate, dict(per_type)


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


def _endpoint_sanity(retained_mask: torch.Tensor, boundaries: list[tuple[int, int]]) -> float:
    """Return the fraction of eligible trajectories retaining both endpoints."""
    eligible = 0
    passing = 0
    for start, end in boundaries:
        n = int(end - start)
        if n <= 1:
            continue
        retained_count = int(retained_mask[start:end].sum().item())
        if retained_count < 2:
            continue
        eligible += 1
        if bool(retained_mask[start].item()) and bool(retained_mask[end - 1].item()):
            passing += 1
    if eligible <= 0:
        return 1.0
    return float(passing / eligible)


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
    """Evaluate one simplification method on typed queries at matched ratio. See evaluation/README.md for details."""
    t0 = time.time()
    retained_mask = method.simplify(points, boundaries, compression_ratio)
    measured_latency_ms = (time.time() - t0) * 1000.0
    latency_ms = float(getattr(method, "latency_ms", measured_latency_ms) or measured_latency_ms)

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
        per_type = {"range": range_point}
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
    endpoint_sanity = _endpoint_sanity(retained_mask, boundaries)
    range_audit["endpoint_sanity"] = endpoint_sanity
    boundary_f1 = float(range_audit.get("range_entry_exit_f1", 0.0))
    query_useful_v1 = query_useful_v1_from_range_audit(
        range_audit,
        length_preservation=avg_length_preserved,
        avg_sed_km=float(geometric.get("avg_sed_km", 0.0)),
        endpoint_sanity=endpoint_sanity,
    )
    range_audit.update(query_useful_v1)
    query_useful_components_raw = query_useful_v1.get("query_useful_v1_components", {})
    query_useful_components = (
        {str(key): float(value) for key, value in query_useful_components_raw.items()}
        if isinstance(query_useful_components_raw, dict)
        else {}
    )
    query_useful_score = float(cast(Any, query_useful_v1.get("query_useful_v1_score", 0.0)) or 0.0)
    query_useful_schema = int(cast(Any, query_useful_v1.get("query_useful_v1_schema_version", 0)) or 0)

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
        range_gap_time_coverage=float(range_audit.get("range_gap_time_coverage", 0.0)),
        range_gap_distance_coverage=float(range_audit.get("range_gap_distance_coverage", 0.0)),
        range_gap_min_coverage=float(range_audit.get("range_gap_min_coverage", 0.0)),
        range_turn_coverage=float(range_audit.get("range_turn_coverage", 0.0)),
        range_shape_score=float(range_audit.get("range_shape_score", 0.0)),
        range_query_local_interpolation_fidelity=float(
            range_audit.get("range_query_local_interpolation_fidelity", 0.0)
        ),
        range_usefulness_score=float(range_audit.get("range_usefulness_score", 0.0)),
        range_usefulness_gap_time_score=float(range_audit.get("range_usefulness_gap_time_score", 0.0)),
        range_usefulness_gap_distance_score=float(
            range_audit.get("range_usefulness_gap_distance_score", 0.0)
        ),
        range_usefulness_gap_min_score=float(range_audit.get("range_usefulness_gap_min_score", 0.0)),
        range_usefulness_schema_version=int(range_audit.get("range_usefulness_schema_version", 0) or 0),
        range_usefulness_gap_ablation_version=int(
            range_audit.get("range_usefulness_gap_ablation_version", 0) or 0
        ),
        query_useful_v1_score=query_useful_score,
        query_useful_v1_schema_version=query_useful_schema,
        query_useful_v1_components=query_useful_components,
        range_audit=range_audit,
        retained_mask=retained_mask if return_mask else None,
    )
