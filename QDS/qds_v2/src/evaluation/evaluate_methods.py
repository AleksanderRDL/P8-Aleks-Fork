"""Method evaluation and fixed-width results table helpers. See src/evaluation/README.md for details."""

from __future__ import annotations

import time

import torch

from src.evaluation.baselines import Method
from src.evaluation.metrics import (
    MethodEvaluation,
    clustering_error,
    knn_error,
    range_error,
    similarity_error,
)
from src.queries.query_executor import execute_typed_query


def _split_by_boundaries(points: torch.Tensor, boundaries: list[tuple[int, int]]) -> list[torch.Tensor]:
    """Split flattened points into trajectory list by boundaries. See src/evaluation/README.md for details."""
    return [points[s:e] for s, e in boundaries]


def evaluate_method(
    method: Method,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict],
    workload_mix: dict[str, float],
    compression_ratio: float,
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

    errors: dict[str, list[float]] = {"range": [], "knn": [], "similarity": [], "clustering": []}
    for q in typed_queries:
        qtype = q["type"]
        full_res = execute_typed_query(points, full_traj, q)
        simp_res = execute_typed_query(simplified, simp_traj, q)
        if qtype == "range":
            errors[qtype].append(range_error(float(full_res), float(simp_res)))
        elif qtype == "knn":
            errors[qtype].append(knn_error(full_res, simp_res))
        elif qtype == "similarity":
            errors[qtype].append(similarity_error(full_res, simp_res))
        elif qtype == "clustering":
            errors[qtype].append(clustering_error(int(full_res), int(simp_res)))

    per_type = {k: (sum(v) / len(v) if v else 0.0) for k, v in errors.items()}
    wsum = sum(workload_mix.values()) if workload_mix else 0.0
    if wsum <= 0.0:
        wmix = {k: 1.0 / 4.0 for k in per_type}
    else:
        wmix = {k: workload_mix.get(k, 0.0) / wsum for k in per_type}
    aggregate = sum(wmix[k] * per_type[k] for k in per_type)
    comp = float(retained_mask.float().mean().item())

    return MethodEvaluation(
        aggregate_error=float(aggregate),
        per_type_error=per_type,
        compression_ratio=comp,
        latency_ms=latency_ms,
    )


def print_method_comparison_table(results: dict[str, MethodEvaluation]) -> str:
    """Render fixed-width method comparison table with per-type rows. See src/evaluation/README.md for details."""
    col1, col2, col3, col4, col5 = 24, 14, 14, 14, 12
    lines = []
    header = (
        f"{'Method':<{col1}}"
        f"{'AggregateErr':>{col2}}"
        f"{'Compression':>{col3}}"
        f"{'Latency(ms)':>{col4}}"
        f"{'Type':>{col5}}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for name, metrics in results.items():
        lines.append(
            f"{name:<{col1}}"
            f"{metrics.aggregate_error:>{col2}.4f}"
            f"{metrics.compression_ratio:>{col3}.4f}"
            f"{metrics.latency_ms:>{col4}.2f}"
            f"{'all':>{col5}}"
        )
        for t_name in ("range", "knn", "similarity", "clustering"):
            lines.append(
                f"{'  - ' + t_name:<{col1}}"
                f"{metrics.per_type_error.get(t_name, 0.0):>{col2}.4f}"
                f"{'':>{col3}}"
                f"{'':>{col4}}"
                f"{t_name:>{col5}}"
            )
    return "\n".join(lines)


def print_shift_table(shift_grid: dict[str, dict[str, float]]) -> str:
    """Render train-mix to eval-mix aggregate error matrix table. See src/evaluation/README.md for details."""
    eval_cols = sorted({k for row in shift_grid.values() for k in row.keys()})
    col_w = 22
    line = f"{'Train\\Eval':<{col_w}}" + "".join(f"{c:>{col_w}}" for c in eval_cols)
    out = [line, "-" * len(line)]
    for train_name in sorted(shift_grid.keys()):
        row = f"{train_name:<{col_w}}"
        for eval_name in eval_cols:
            val = shift_grid[train_name].get(eval_name, float("nan"))
            row += f"{val:>{col_w}.4f}"
        out.append(row)
    return "\n".join(out)
