"""Method evaluation and fixed-width results table helpers. See src/evaluation/README.md for details."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
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
from src.evaluation.range_usefulness import RANGE_USEFULNESS_SCHEMA_VERSION, RANGE_USEFULNESS_WEIGHTS
from src.queries.query_executor import execute_typed_query
from src.queries.query_types import normalize_pure_workload_map

POINT_AWARE_KNN_REPRESENTATIVES_PER_TRAJECTORY = 64
POINT_AWARE_SIMILARITY_REPRESENTATIVES_PER_TRAJECTORY = 64


@dataclass(frozen=True)
class RangeTrajectoryAuditSupport:
    """Retained-independent per-trajectory support for one range query."""

    trajectory_id: int
    start: int
    in_offsets_cpu: torch.Tensor
    turn_weights_cpu: torch.Tensor
    full_time_span: float
    full_length_km: float


@dataclass(frozen=True)
class RangeQueryAuditSupport:
    """Retained-independent support reused across range audit methods and ratios."""

    range_mask: torch.Tensor
    boundary_indices_cpu: torch.Tensor
    crossing_bracket_indices_cpu: torch.Tensor
    full_trajectory_ids: tuple[int, ...]
    trajectories: tuple[RangeTrajectoryAuditSupport, ...]


def _points_cache_token(points: torch.Tensor) -> tuple[int, int, tuple[int, ...], str, str]:
    """Return an identity token for caller-owned evaluation caches."""
    data_ptr = int(points.data_ptr()) if points.numel() > 0 else 0
    return (id(points), data_ptr, tuple(int(dim) for dim in points.shape), str(points.device), str(points.dtype))


def _queries_cache_token(typed_queries: list[dict]) -> tuple[int, int, tuple[int, ...]]:
    """Return an identity token for a typed-query workload list."""
    return (id(typed_queries), len(typed_queries), tuple(id(query) for query in typed_queries))


@dataclass
class EvaluationQueryCache:
    """Caller-owned cache for full-data query results during repeated method evaluation."""

    points_token: tuple[int, int, tuple[int, ...], str, str]
    boundaries_key: tuple[tuple[int, int], ...]
    queries_token: tuple[int, int, tuple[int, ...]]
    full_traj: list[torch.Tensor] | None = None
    full_results: dict[int, Any] = field(default_factory=dict)
    support_masks: dict[int, torch.Tensor] = field(default_factory=dict)
    range_audit_supports: dict[int, RangeQueryAuditSupport] = field(default_factory=dict)

    @classmethod
    def for_workload(
        cls,
        points: torch.Tensor,
        boundaries: list[tuple[int, int]],
        typed_queries: list[dict],
    ) -> EvaluationQueryCache:
        """Build a cache scoped to exactly one points/boundaries/workload object."""
        return cls(
            points_token=_points_cache_token(points),
            boundaries_key=tuple((int(start), int(end)) for start, end in boundaries),
            queries_token=_queries_cache_token(typed_queries),
        )

    def validate(
        self,
        points: torch.Tensor,
        boundaries: list[tuple[int, int]],
        typed_queries: list[dict],
    ) -> None:
        """Fail fast if this cache is reused for a different evaluation scope."""
        if (
            self.points_token != _points_cache_token(points)
            or self.boundaries_key != tuple((int(start), int(end)) for start, end in boundaries)
            or self.queries_token != _queries_cache_token(typed_queries)
        ):
            raise ValueError("EvaluationQueryCache was built for different points, boundaries, or typed queries.")

    def get_full_traj(self, points: torch.Tensor, boundaries: list[tuple[int, int]]) -> list[torch.Tensor]:
        """Return split full-data trajectories, building them once per cache."""
        if self.full_traj is None:
            self.full_traj = _split_by_boundaries(points, boundaries)
        return self.full_traj

    def get_full_result(self, query_index: int, builder: Callable[[], Any]) -> Any:
        """Return a cached full-data query answer."""
        if query_index not in self.full_results:
            self.full_results[query_index] = builder()
        return self.full_results[query_index]

    def get_support_mask(self, query_index: int, builder: Callable[[], torch.Tensor]) -> torch.Tensor:
        """Return a cached full-data support mask."""
        if query_index not in self.support_masks:
            self.support_masks[query_index] = builder()
        return self.support_masks[query_index]

    def get_range_audit_support(
        self,
        query_index: int,
        builder: Callable[[], RangeQueryAuditSupport],
    ) -> RangeQueryAuditSupport:
        """Return cached retained-independent range-audit support."""
        if query_index not in self.range_audit_supports:
            self.range_audit_supports[query_index] = builder()
        return self.range_audit_supports[query_index]


def _split_by_boundaries(points: torch.Tensor, boundaries: list[tuple[int, int]]) -> list[torch.Tensor]:
    """Split flattened points into trajectory list by boundaries. See src/evaluation/README.md for details."""
    return [points[s:e] for s, e in boundaries]


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


def _segment_box_intersections(points_cpu: torch.Tensor, params: dict[str, float]) -> torch.Tensor:
    """Return true for trajectory segments that intersect the query space-time box."""
    if points_cpu.shape[0] < 2:
        return torch.empty((0,), dtype=torch.bool)
    start_xyz = points_cpu[:-1, [0, 1, 2]].float()
    end_xyz = points_cpu[1:, [0, 1, 2]].float()
    delta = end_xyz - start_xyz
    lows = torch.tensor(
        [float(params["t_start"]), float(params["lat_min"]), float(params["lon_min"])],
        dtype=torch.float32,
    )
    highs = torch.tensor(
        [float(params["t_end"]), float(params["lat_max"]), float(params["lon_max"])],
        dtype=torch.float32,
    )

    u_min = torch.zeros((start_xyz.shape[0],), dtype=torch.float32)
    u_max = torch.ones((start_xyz.shape[0],), dtype=torch.float32)
    valid = torch.ones((start_xyz.shape[0],), dtype=torch.bool)
    eps = 1e-12
    for dim in range(3):
        dim_delta = delta[:, dim]
        dim_start = start_xyz[:, dim]
        parallel = torch.abs(dim_delta) <= eps
        valid &= (~parallel) | ((dim_start >= lows[dim]) & (dim_start <= highs[dim]))

        non_parallel = ~parallel
        if bool(non_parallel.any().item()):
            u1 = (lows[dim] - dim_start[non_parallel]) / dim_delta[non_parallel]
            u2 = (highs[dim] - dim_start[non_parallel]) / dim_delta[non_parallel]
            u_low = torch.minimum(u1, u2)
            u_high = torch.maximum(u1, u2)
            u_min[non_parallel] = torch.maximum(u_min[non_parallel], u_low)
            u_max[non_parallel] = torch.minimum(u_max[non_parallel], u_high)
    return valid & (u_max >= u_min) & (u_max >= 0.0) & (u_min <= 1.0)


def _range_crossing_bracket_indices_for_trajectories(
    points_cpu: torch.Tensor,
    params: dict[str, float],
    boundaries: list[tuple[int, int]],
) -> torch.Tensor:
    """Return point pairs bracketing segment-box intersections."""
    bracket_indices: list[torch.Tensor] = []
    for start, end in boundaries:
        if end - start < 2:
            continue
        intersecting = _segment_box_intersections(points_cpu[start:end], params)
        segment_offsets = torch.where(intersecting)[0]
        if segment_offsets.numel() == 0:
            continue
        pairs = torch.stack((segment_offsets, segment_offsets + 1), dim=1).reshape(-1)
        bracket_indices.append((torch.unique(pairs) + int(start)).detach().cpu())
    if not bracket_indices:
        return torch.empty((0,), dtype=torch.long)
    return torch.unique(torch.cat(bracket_indices).to(dtype=torch.long))


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
) -> RangeQueryAuditSupport:
    """Build retained-independent support for one range query."""
    range_mask = range_mask.bool()
    full_ids = tuple(_trajectory_ids_for_mask(range_mask, point_trajectory_ids))
    boundary_indices_cpu = _range_boundary_indices_for_trajectories(range_mask, boundaries, list(full_ids))
    crossing_bracket_indices_cpu = _range_crossing_bracket_indices_for_trajectories(
        points_cpu,
        params,
        boundaries,
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
        else:
            range_mask = build_range_mask()
        return _build_range_query_audit_support(
            points_cpu=points_cpu,
            boundaries=boundaries,
            range_mask=range_mask,
            point_trajectory_ids=point_trajectory_ids,
            params=query["params"],
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
        "range_boundary_f1": float(range_entry_exit_f1),
        "range_crossing_f1": float(range_crossing_f1),
        "range_temporal_coverage": float(range_temporal_coverage),
        "range_gap_coverage": float(range_gap_coverage),
        "range_turn_coverage": float(range_turn_coverage),
        "range_shape_score": float(range_shape_score),
        "range_usefulness_score": float(range_usefulness_score),
        "range_usefulness_weights": dict(RANGE_USEFULNESS_WEIGHTS),
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
            full_traj = _split_by_boundaries(points, boundaries)
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
            simp_traj = _split_by_boundaries(simplified, simp_boundaries)
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
            ans = f1_score(set(full_res), set(simp_res))
            support = support_mask(
                query_index,
                lambda full_res=full_res, query=query: _knn_representative_mask(
                    points,
                    boundaries,
                    set(full_res),
                    query["params"],
                ),
            )
            answer_scores[qtype].append(ans)
            combined_scores[qtype].append(ans * _point_subset_f1(retained_mask, support))
        elif qtype == "similarity":
            ans = f1_score(set(full_res), set(simp_res))
            support = support_mask(
                query_index,
                lambda full_res=full_res, query=query: _similarity_support_mask(points, boundaries, set(full_res), query),
            )
            answer_scores[qtype].append(ans)
            combined_scores[qtype].append(ans * _point_subset_f1(retained_mask, support))
        elif qtype == "clustering":
            full_labels = cast(list[int], full_res)
            simp_labels = cast(list[int], simp_res)
            ans = clustering_f1(full_labels, simp_labels)
            support = support_mask(
                query_index,
                lambda full_res=full_labels, query=query: _clustering_support_mask(
                    points,
                    boundaries,
                    list(full_res),
                    query["params"],
                ),
            )
            answer_scores[qtype].append(ans)
            combined_scores[qtype].append(ans * _point_subset_f1(retained_mask, support))

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
        range_boundary_f1=boundary_f1,
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


def _range_focused_results(results: dict[str, MethodEvaluation]) -> bool:
    """Return true when a result table represents a pure range workload."""
    saw_range = False
    for metrics in results.values():
        if int(metrics.range_audit.get("range_query_count", 0) or 0) > 0:
            saw_range = True
        elif set(metrics.per_type_f1) <= {"range"} and "range" in metrics.per_type_f1:
            saw_range = True
        for qtype, value in metrics.per_type_f1.items():
            if qtype != "range" and abs(float(value)) > 1e-12:
                return False
    return saw_range


def _range_point_metric(metrics: MethodEvaluation) -> float:
    """Return the explicit range point metric with compatibility fallback."""
    if int(metrics.range_audit.get("range_query_count", 0) or 0) > 0:
        return float(metrics.range_point_f1)
    return float(metrics.per_type_f1.get("range", metrics.aggregate_f1))


def _range_usefulness_metric(metrics: MethodEvaluation) -> float:
    """Return the explicit range usefulness metric with compatibility fallback."""
    if int(metrics.range_audit.get("range_query_count", 0) or 0) > 0:
        return float(metrics.range_usefulness_score)
    if metrics.range_usefulness_score > 0.0:
        return float(metrics.range_usefulness_score)
    return float(metrics.aggregate_combined_f1 or _range_point_metric(metrics))


def print_method_comparison_table(results: dict[str, MethodEvaluation]) -> str:
    """Render fixed-width method comparison table with workload-specific F1 labels."""
    range_focused = _range_focused_results(results)
    col1, col2, col3, col4, col5, col6, col7 = 24, 14, 13, 12, 12, 14, 13
    primary_label = "RangePointF1" if range_focused else "AnswerF1"
    secondary_label = "RangeUseful" if range_focused else "CombinedF1"
    boundary_label = "EntryExitF1" if range_focused else "BoundaryF1"
    lines = []
    header = (
        f"{'Method':<{col1}}"
        f"{primary_label:>{col2}}"
        f"{secondary_label:>{col3}}"
        f"{'Compression':>{col4}}"
        f"{'AvgPtGap':>{col5}}"
        f"{'Latency(ms)':>{col6}}"
        f"{boundary_label:>{col7}}"
        f"{'Type':>{col7}}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    type_rows: tuple[str, ...] = () if range_focused else ("range", "knn", "similarity", "clustering")
    for name, metrics in results.items():
        primary = _range_point_metric(metrics) if range_focused else float(metrics.aggregate_f1)
        secondary = _range_usefulness_metric(metrics) if range_focused else float(metrics.aggregate_combined_f1)
        entry_exit = float(metrics.range_entry_exit_f1 or metrics.range_boundary_f1)
        lines.append(
            f"{name:<{col1}}"
            f"{primary:>{col2}.6f}"
            f"{secondary:>{col3}.6f}"
            f"{metrics.compression_ratio:>{col4}.4f}"
            f"{metrics.avg_retained_point_gap:>{col5}.2f}"
            f"{metrics.latency_ms:>{col6}.2f}"
            f"{entry_exit:>{col7}.6f}"
            f"{'all':>{col7}}"
        )
        for t_name in type_rows:
            lines.append(
                f"{'  - ' + t_name:<{col1}}"
                f"{metrics.per_type_f1.get(t_name, 0.0):>{col2}.6f}"
                f"{metrics.per_type_combined_f1.get(t_name, 0.0):>{col3}.6f}"
                f"{'':>{col4}}"
                f"{'':>{col5}}"
                f"{'':>{col6}}"
                f"{'':>{col7}}"
                f"{t_name:>{col7}}"
            )

    def _rel_pct(diff: float, baseline: float) -> str:
        """Format a percentage of baseline, with safe div-by-zero handling."""
        if abs(baseline) < 1e-9:
            return "  n/a"
        return f"{100.0 * diff / baseline:+.1f}%"

    mlqds = results.get("MLQDS")
    diff_references = [
        ("uniform", results.get("uniform") or results.get("newUniformTemporal")),
        ("DouglasPeucker", results.get("DouglasPeucker")),
    ]
    if mlqds is not None and any(ref is not None for _, ref in diff_references):
        lines.append("-" * len(header))
        metric_pair = "RangePointF1 / RangeUseful" if range_focused else "AnswerF1 / CombinedF1"
        lines.append(f"{f'Diff vs MLQDS ({metric_pair}; abs and % vs baseline)':<{col1}}")
        for ref_name, ref in diff_references:
            if ref is None:
                continue
            if range_focused:
                agg_ans = _range_point_metric(mlqds) - _range_point_metric(ref)
                agg_comb = _range_usefulness_metric(mlqds) - _range_usefulness_metric(ref)
                agg_ans_pct = _rel_pct(agg_ans, _range_point_metric(ref))
                agg_comb_pct = _rel_pct(agg_comb, _range_usefulness_metric(ref))
            else:
                agg_ans = mlqds.aggregate_f1 - ref.aggregate_f1
                agg_comb = mlqds.aggregate_combined_f1 - ref.aggregate_combined_f1
                agg_ans_pct = _rel_pct(agg_ans, ref.aggregate_f1)
                agg_comb_pct = _rel_pct(agg_comb, ref.aggregate_combined_f1)
            label = f"  vs {ref_name}"
            lines.append(
                f"{label:<{col1}}"
                f"{agg_ans:>+{col2}.6f}"
                f"{agg_comb:>+{col3}.6f}"
                f"{'':>{col4}}"
                f"{'':>{col5}}"
                f"{'':>{col6}}"
                f"{'':>{col7}}"
                f"{'all':>{col7}}"
            )
            lines.append(
                f"{'      (% vs baseline)':<{col1}}"
                f"{agg_ans_pct:>{col2}}"
                f"{agg_comb_pct:>{col3}}"
                f"{'':>{col4}}"
                f"{'':>{col5}}"
                f"{'':>{col6}}"
                f"{'':>{col7}}"
                f"{'all':>{col7}}"
            )
            for t_name in type_rows:
                ref_ans = ref.per_type_f1.get(t_name, 0.0)
                ref_comb = ref.per_type_combined_f1.get(t_name, 0.0)
                t_ans = mlqds.per_type_f1.get(t_name, 0.0) - ref_ans
                t_comb = mlqds.per_type_combined_f1.get(t_name, 0.0) - ref_comb
                t_ans_pct = _rel_pct(t_ans, ref_ans)
                t_comb_pct = _rel_pct(t_comb, ref_comb)
                lines.append(
                    f"{'    - ' + t_name:<{col1}}"
                    f"{t_ans:>+{col2}.6f}"
                    f"{t_comb:>+{col3}.6f}"
                    f"{'':>{col4}}"
                    f"{'':>{col5}}"
                    f"{'':>{col6}}"
                    f"{'':>{col7}}"
                    f"{t_name:>{col7}}"
                )
                lines.append(
                    f"{'      (% vs baseline)':<{col1}}"
                    f"{t_ans_pct:>{col2}}"
                    f"{t_comb_pct:>{col3}}"
                    f"{'':>{col4}}"
                    f"{'':>{col5}}"
                    f"{'':>{col6}}"
                    f"{'':>{col7}}"
                    f"{t_name:>{col7}}"
                )
    return "\n".join(lines)


def print_range_usefulness_table(results: dict[str, MethodEvaluation]) -> str:
    """Render detailed range usefulness audit components."""
    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11 = 24, 14, 10, 10, 13, 11, 13, 10, 10, 12, 13
    header = (
        f"{'Method':<{col1}}"
        f"{'RangePointF1':>{col2}}"
        f"{'ShipF1':>{col3}}"
        f"{'ShipCov':>{col4}}"
        f"{'EntryExitF1':>{col5}}"
        f"{'CrossingF1':>{col6}}"
        f"{'TemporalCov':>{col7}}"
        f"{'GapCov':>{col8}}"
        f"{'TurnCov':>{col9}}"
        f"{'ShapeScore':>{col10}}"
        f"{'RangeUseful':>{col11}}"
    )
    lines = [header, "-" * len(header)]
    for name, metrics in results.items():
        lines.append(
            f"{name:<{col1}}"
            f"{_range_point_metric(metrics):>{col2}.6f}"
            f"{metrics.range_ship_f1:>{col3}.6f}"
            f"{metrics.range_ship_coverage:>{col4}.6f}"
            f"{float(metrics.range_entry_exit_f1 or metrics.range_boundary_f1):>{col5}.6f}"
            f"{metrics.range_crossing_f1:>{col6}.6f}"
            f"{metrics.range_temporal_coverage:>{col7}.6f}"
            f"{metrics.range_gap_coverage:>{col8}.6f}"
            f"{metrics.range_turn_coverage:>{col9}.6f}"
            f"{metrics.range_shape_score:>{col10}.6f}"
            f"{_range_usefulness_metric(metrics):>{col11}.6f}"
        )
    return "\n".join(lines)


def print_geometric_distortion_table(results: dict[str, MethodEvaluation]) -> str:
    """Render geometric-distortion + shape-aware utility comparison.

    SED (Meratnia & de By 2004) and PED (Imai & Iri 1988; what Douglas-Peucker
    minimises) are reported in km — lower is better. LengthPres is the fraction of
    total path length preserved (sum_simp_km / sum_orig_km) in [0, 1] — higher is better.
    F1xLen combines aggregate query F1 with shape preservation: equals F1 when shape
    is perfect (length_preserved=1.0) and 0 when simplified trajectory collapses
    (length_preserved=0.0) — higher is better. Use this column as the single
    shape-aware utility number when comparing methods.
    """
    col1, col2, col3, col4, col5, col6, col7 = 24, 11, 11, 11, 11, 13, 13
    header = (
        f"{'Method':<{col1}}"
        f"{'AvgSED_km':>{col2}}"
        f"{'MaxSED_km':>{col3}}"
        f"{'AvgPED_km':>{col4}}"
        f"{'MaxPED_km':>{col5}}"
        f"{'LengthPres':>{col6}}"
        f"{'F1xLen':>{col7}}"
    )
    lines = [header, "-" * len(header)]
    for name, metrics in results.items():
        g = metrics.geometric_distortion or {}
        lines.append(
            f"{name:<{col1}}"
            f"{g.get('avg_sed_km', 0.0):>{col2}.4f}"
            f"{g.get('max_sed_km', 0.0):>{col3}.2f}"
            f"{g.get('avg_ped_km', 0.0):>{col4}.4f}"
            f"{g.get('max_ped_km', 0.0):>{col5}.2f}"
            f"{metrics.avg_length_preserved:>{col6}.4f}"
            f"{metrics.combined_query_shape_score:>{col7}.6f}"
        )
    return "\n".join(lines)


def print_shift_table(shift_grid: dict[str, dict[str, float]]) -> str:
    """Render train-workload to eval-workload aggregate F1 matrix table."""
    eval_cols = sorted({k for row in shift_grid.values() for k in row.keys()})
    col_w = 22
    header_label = "Train\\Eval"
    line = f"{header_label:<{col_w}}" + "".join(f"{c:>{col_w}}" for c in eval_cols)
    out = [line, "-" * len(line)]
    for train_name in sorted(shift_grid.keys()):
        row = f"{train_name:<{col_w}}"
        for eval_name in eval_cols:
            val = shift_grid[train_name].get(eval_name, float("nan"))
            row += f"{val:>{col_w}.4f}"
        out.append(row)
    return "\n".join(out)
