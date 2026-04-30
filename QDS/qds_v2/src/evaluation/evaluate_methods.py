"""Method evaluation and fixed-width results table helpers. See src/evaluation/README.md for details."""

from __future__ import annotations

import time

import torch

from src.evaluation.baselines import Method
from src.evaluation.metrics import (
    MethodEvaluation,
    clustering_f1,
    f1_score,
)
from src.queries.query_executor import execute_typed_query


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
    full_hits = int(_range_box_mask(points, params).sum().item())
    simplified_hits = int(_range_box_mask(simplified, params).sum().item())
    if full_hits == 0 and simplified_hits == 0:
        return 1.0
    if full_hits == 0 or simplified_hits == 0:
        return 0.0
    return float((2.0 * simplified_hits) / (full_hits + simplified_hits))


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

    simplified = points[retained_mask]
    full_traj = _split_by_boundaries(points, boundaries)
    simp_boundaries: list[tuple[int, int]] = []
    cursor = 0
    for s, e in boundaries:
        n = int(retained_mask[s:e].sum().item())
        simp_boundaries.append((cursor, cursor + n))
        cursor += n
    simp_traj = _split_by_boundaries(simplified, simp_boundaries)

    scores: dict[str, list[float]] = {"range": [], "knn": [], "similarity": [], "clustering": []}
    for q in typed_queries:
        qtype = q["type"]
        if qtype == "range":
            scores[qtype].append(_range_point_f1(points, simplified, q["params"]))
            continue

        full_res = execute_typed_query(points, full_traj, q, boundaries)
        simp_res = execute_typed_query(simplified, simp_traj, q, simp_boundaries)
        if qtype in {"knn", "similarity"}:
            scores[qtype].append(f1_score(set(full_res), set(simp_res)))
        elif qtype == "clustering":
            scores[qtype].append(clustering_f1(full_res, simp_res))

    per_type = {k: (sum(v) / len(v) if v else 0.0) for k, v in scores.items()}
    wsum = sum(workload_mix.values()) if workload_mix else 0.0
    if wsum <= 0.0:
        wmix = {k: 1.0 / 4.0 for k in per_type}
    else:
        wmix = {k: workload_mix.get(k, 0.0) / wsum for k in per_type}
    aggregate = sum(wmix[k] * per_type[k] for k in per_type)
    comp = float(retained_mask.float().mean().item())

    return MethodEvaluation(
        aggregate_f1=float(aggregate),
        per_type_f1=per_type,
        compression_ratio=comp,
        latency_ms=latency_ms,
        retained_mask=retained_mask if return_mask else None,
    )


def print_method_comparison_table(results: dict[str, MethodEvaluation]) -> str:
    """Render fixed-width method comparison table with per-type rows. See src/evaluation/README.md for details."""
    col1, col2, col3, col4, col5 = 24, 14, 14, 14, 12
    lines = []
    header = (
        f"{'Method':<{col1}}"
        f"{'AggregateF1':>{col2}}"
        f"{'Compression':>{col3}}"
        f"{'Latency(ms)':>{col4}}"
        f"{'Type':>{col5}}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for name, metrics in results.items():
        lines.append(
            f"{name:<{col1}}"
            f"{metrics.aggregate_f1:>{col2}.6f}"
            f"{metrics.compression_ratio:>{col3}.4f}"
            f"{metrics.latency_ms:>{col4}.2f}"
            f"{'all':>{col5}}"
        )
        for t_name in ("range", "knn", "similarity", "clustering"):
            lines.append(
                f"{'  - ' + t_name:<{col1}}"
                f"{metrics.per_type_f1.get(t_name, 0.0):>{col2}.6f}"
                f"{'':>{col3}}"
                f"{'':>{col4}}"
                f"{t_name:>{col5}}"
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
