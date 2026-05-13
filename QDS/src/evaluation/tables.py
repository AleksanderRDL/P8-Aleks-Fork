"""Fixed-width evaluation table renderers."""

from __future__ import annotations

from src.evaluation.metrics import MethodEvaluation


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
    """Return the explicit range point metric, falling back for synthetic/test rows."""
    if int(metrics.range_audit.get("range_query_count", 0) or 0) > 0:
        return float(metrics.range_point_f1)
    return float(metrics.per_type_f1.get("range", metrics.aggregate_f1))


def _range_usefulness_metric(metrics: MethodEvaluation) -> float:
    """Return the explicit range usefulness metric, falling back for synthetic/test rows."""
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
        entry_exit = float(metrics.range_entry_exit_f1)
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
            f"{metrics.range_entry_exit_f1:>{col5}.6f}"
            f"{metrics.range_crossing_f1:>{col6}.6f}"
            f"{metrics.range_temporal_coverage:>{col7}.6f}"
            f"{metrics.range_gap_coverage:>{col8}.6f}"
            f"{metrics.range_turn_coverage:>{col9}.6f}"
            f"{metrics.range_shape_score:>{col10}.6f}"
            f"{_range_usefulness_metric(metrics):>{col11}.6f}"
        )
    return "\n".join(lines)


def print_geometric_distortion_table(results: dict[str, MethodEvaluation]) -> str:
    """Render geometric-distortion + shape-aware utility comparison."""
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
