"""Method evaluation and fixed-width results table helpers. See src/evaluation/README.md for details."""

from __future__ import annotations

import time

import torch

from src.evaluation.baselines import Method
from src.evaluation.metrics import (
    MethodEvaluation,
    clustering_f1,
    compute_average_length_loss,
    compute_geometric_distortion,
    f1_score,
)
from src.queries.query_executor import execute_typed_query

POINT_AWARE_KNN_REPRESENTATIVES_PER_TRAJECTORY = 64
POINT_AWARE_SIMILARITY_REPRESENTATIVES_PER_TRAJECTORY = 64


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


def _range_point_f1(points: torch.Tensor, simplified: torch.Tensor, params: dict[str, float]) -> float:
    """Compute range F1 over point hits instead of trajectory-presence hits."""
    full_mask = _range_box_mask(points, params)
    simplified_mask = _range_box_mask(simplified, params)

    full_points = {tuple(row) for row in points[full_mask].tolist()}
    simplified_points = {tuple(row) for row in simplified[simplified_mask].tolist()}

    full_hits = len(full_points)
    simplified_hits = len(simplified_points)
    if full_hits == 0 and simplified_hits == 0:
        return 1.0
    if full_hits == 0 or simplified_hits == 0:
        return 0.0

    true_positives = len(full_points.intersection(simplified_points))
    precision = float(true_positives / simplified_hits)
    recall = float(true_positives / full_hits)
    if precision + recall == 0.0:
        return 0.0
    return float((2.0 * precision * recall) / (precision + recall))


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
    workload_mix: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """Score a precomputed retained mask with the final query-F1 semantics."""
    simplified = points[retained_mask]
    full_traj = _split_by_boundaries(points, boundaries)
    simp_boundaries: list[tuple[int, int]] = []
    cursor = 0
    for start, end in boundaries:
        n = int(retained_mask[start:end].sum().item())
        simp_boundaries.append((cursor, cursor + n))
        cursor += n
    simp_traj = _split_by_boundaries(simplified, simp_boundaries)

    scores: dict[str, list[float]] = {"range": [], "knn": [], "similarity": [], "clustering": []}
    for query in typed_queries:
        qtype = query["type"]
        if qtype == "range":
            scores[qtype].append(_range_point_f1(points, simplified, query["params"]))
            continue

        full_res = execute_typed_query(points, full_traj, query, boundaries)
        simp_res = execute_typed_query(simplified, simp_traj, query, simp_boundaries)
        if qtype == "knn":
            answer_f1 = f1_score(set(full_res), set(simp_res))
            support = _knn_representative_mask(points, boundaries, set(full_res), query["params"])
            scores[qtype].append(answer_f1 * _point_subset_f1(retained_mask, support))
        elif qtype == "similarity":
            answer_f1 = f1_score(set(full_res), set(simp_res))
            support = _similarity_support_mask(points, boundaries, set(full_res), query)
            scores[qtype].append(answer_f1 * _point_subset_f1(retained_mask, support))
        elif qtype == "clustering":
            cluster_f1 = clustering_f1(full_res, simp_res)
            support = _clustering_support_mask(points, boundaries, list(full_res), query["params"])
            scores[qtype].append(cluster_f1 * _point_subset_f1(retained_mask, support))

    per_type = {name: (sum(values) / len(values) if values else 0.0) for name, values in scores.items()}
    weight_sum = sum(workload_mix.values()) if workload_mix else 0.0
    if weight_sum <= 0.0:
        normalized_mix = {name: 1.0 / 4.0 for name in per_type}
    else:
        normalized_mix = {name: workload_mix.get(name, 0.0) / weight_sum for name in per_type}
    aggregate = sum(normalized_mix[name] * per_type[name] for name in per_type)
    return float(aggregate), per_type


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
    workload_mix: dict[str, float],
    compression_ratio: float,
    return_mask: bool = False,
) -> MethodEvaluation:
    """Evaluate one simplification method on typed queries at matched ratio. See src/evaluation/README.md for details."""
    t0 = time.time()
    retained_mask = method.simplify(points, boundaries, compression_ratio)
    latency_ms = (time.time() - t0) * 1000.0

    aggregate, per_type = score_retained_mask(
        points=points,
        boundaries=boundaries,
        retained_mask=retained_mask,
        typed_queries=typed_queries,
        workload_mix=workload_mix,
    )
    comp = float(retained_mask.float().mean().item())
    avg_gap, avg_norm_gap, max_gap = _retained_point_gap_stats(retained_mask, boundaries)
    geometric = compute_geometric_distortion(points, boundaries, retained_mask)
    avg_length_loss = compute_average_length_loss(points, boundaries, retained_mask)
    combined = float(aggregate) * (1.0 - max(0.0, min(1.0, avg_length_loss)))

    return MethodEvaluation(
        aggregate_f1=float(aggregate),
        per_type_f1=per_type,
        compression_ratio=comp,
        latency_ms=latency_ms,
        avg_retained_point_gap=avg_gap,
        avg_retained_point_gap_norm=avg_norm_gap,
        max_retained_point_gap=max_gap,
        geometric_distortion=geometric,
        avg_length_loss=avg_length_loss,
        combined_query_shape_score=combined,
        retained_mask=retained_mask if return_mask else None,
    )


def print_method_comparison_table(results: dict[str, MethodEvaluation]) -> str:
    """Render fixed-width method comparison table with per-type rows. See src/evaluation/README.md for details."""
    col1, col2, col3, col4, col5, col6 = 24, 14, 14, 12, 14, 12
    lines = []
    header = (
        f"{'Method':<{col1}}"
        f"{'AggregateF1':>{col2}}"
        f"{'Compression':>{col3}}"
        f"{'AvgPtGap':>{col4}}"
        f"{'Latency(ms)':>{col5}}"
        f"{'Type':>{col6}}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for name, metrics in results.items():
        lines.append(
            f"{name:<{col1}}"
            f"{metrics.aggregate_f1:>{col2}.6f}"
            f"{metrics.compression_ratio:>{col3}.4f}"
            f"{metrics.avg_retained_point_gap:>{col4}.2f}"
            f"{metrics.latency_ms:>{col5}.2f}"
            f"{'all':>{col6}}"
        )
        for t_name in ("range", "knn", "similarity", "clustering"):
            lines.append(
                f"{'  - ' + t_name:<{col1}}"
                f"{metrics.per_type_f1.get(t_name, 0.0):>{col2}.6f}"
                f"{'':>{col3}}"
                f"{'':>{col4}}"
                f"{'':>{col5}}"
                f"{t_name:>{col6}}"
            )
    return "\n".join(lines)


def print_geometric_distortion_table(results: dict[str, MethodEvaluation]) -> str:
    """Render geometric-distortion + shape-aware utility comparison.

    SED (Meratnia & de By 2004) and PED (Imai & Iri 1988; what Douglas-Peucker
    minimises) are reported in km — lower is better. Length-loss is the average
    fractional path-length lost vs the original trajectory in [0, 1] — lower is better.
    F1x(1-L) combines aggregate query F1 with shape preservation: equals F1 when shape
    is perfect (length_loss=0) and equals 0 when the simplified trajectory collapses
    (length_loss=1) — higher is better. Use this column as the single shape-aware
    utility number when comparing methods.
    """
    col1, col2, col3, col4, col5, col6, col7 = 24, 11, 11, 11, 11, 13, 13
    header = (
        f"{'Method':<{col1}}"
        f"{'AvgSED_km':>{col2}}"
        f"{'MaxSED_km':>{col3}}"
        f"{'AvgPED_km':>{col4}}"
        f"{'MaxPED_km':>{col5}}"
        f"{'AvgLengthL':>{col6}}"
        f"{'F1x(1-L)':>{col7}}"
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
            f"{metrics.avg_length_loss:>{col6}.4f}"
            f"{metrics.combined_query_shape_score:>{col7}.6f}"
        )
    return "\n".join(lines)


def print_shift_table(shift_grid: dict[str, dict[str, float]]) -> str:
    """Render train-mix to eval-mix aggregate F1 matrix table. See src/evaluation/README.md for details."""
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
