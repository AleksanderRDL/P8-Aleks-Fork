"""Benchmark row shaping and report table helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from training.model_features import is_workload_blind_model_type, model_type_metadata

LOW_COMPRESSION_THRESHOLD = 0.05 + 1e-9
MIN_MATCHED_LEARNED_SLOT_FRACTION_FOR_BLIND_CLAIM = 0.25
QUERY_DRIVEN_FINAL_COVERAGE_TARGETS = (0.05, 0.10, 0.15, 0.30)
QUERY_DRIVEN_FINAL_COMPRESSION_RATIOS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30)
QUERY_DRIVEN_MIN_UNIFORM_WINS = 19
QUERY_DRIVEN_MIN_DP_WINS = 24
QUERY_DRIVEN_MIN_LOW_BUDGET_UNIFORM_WINS = 7
QUERY_DRIVEN_MIN_MATCHED_5_PERCENT_UNIFORM_WINS = 3
RANGE_COMPONENT_KEYS = (
    "range_point_f1",
    "range_ship_f1",
    "range_ship_coverage",
    "range_entry_exit_f1",
    "range_crossing_f1",
    "range_temporal_coverage",
    "range_gap_coverage",
    "range_turn_coverage",
    "range_shape_score",
    "range_query_local_interpolation_fidelity",
)
RANGE_USEFULNESS_GAP_VARIANT_KEYS = (
    ("gap_time", "range_usefulness_gap_time_score"),
    ("gap_distance", "range_usefulness_gap_distance_score"),
    ("gap_min", "range_usefulness_gap_min_score"),
)


def _child_run_dir(results_dir: Path, workload: str, run_label: str, workload_count: int) -> Path:
    """Return the child experiment output directory for a benchmark row."""
    if workload_count == 1:
        return results_dir / run_label
    return results_dir / workload / run_label

def _as_float(value: Any) -> float | None:
    """Coerce a metric-like value to float, preserving missing/non-numeric values."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _metric_delta(left: dict[str, Any], right: dict[str, Any], key: str) -> float | None:
    """Return left - right for one numeric metric."""
    left_value = _as_float(left.get(key))
    right_value = _as_float(right.get(key))
    if left_value is None or right_value is None:
        return None
    return left_value - right_value

def _metric_beats(left: dict[str, Any], right: dict[str, Any], key: str) -> bool | None:
    """Return whether left strictly beats right for a higher-is-better metric."""
    delta = _metric_delta(left, right, key)
    return None if delta is None else bool(delta > 0.0)

def _geometry_fields(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    """Flatten geometric-distortion metrics for one method."""
    geometry = metrics.get("geometric_distortion") or {}
    return {
        f"{prefix}_avg_sed_km": geometry.get("avg_sed_km"),
        f"{prefix}_max_sed_km": geometry.get("max_sed_km"),
        f"{prefix}_avg_ped_km": geometry.get("avg_ped_km"),
        f"{prefix}_max_ped_km": geometry.get("max_ped_km"),
        f"{prefix}_removed_points": geometry.get("removed_points"),
        f"{prefix}_avg_length_preserved": metrics.get("avg_length_preserved"),
        f"{prefix}_latency_ms": metrics.get("latency_ms"),
    }

def _csv_path_list(raw: Any) -> tuple[str, ...]:
    """Parse benchmark CSV path-list fields for report metadata."""
    if raw is None:
        return ()
    return tuple(part.strip() for part in str(raw).split(",") if part.strip())

def _data_source_row_fields(data_sources: dict[str, Any] | None) -> dict[str, Any]:
    """Flatten train/validation/eval CSV source metadata into one report row."""
    data_sources = data_sources or {}
    csv_path = data_sources.get("csv_path")
    train_csv_path = data_sources.get("train_csv_path")
    validation_csv_path = data_sources.get("validation_csv_path")
    eval_csv_path = data_sources.get("eval_csv_path")
    selected_files = tuple(str(path) for path in data_sources.get("selected_cleaned_csv_files") or ())
    return {
        "csv_path": csv_path,
        "train_csv_path": train_csv_path,
        "validation_csv_path": validation_csv_path,
        "eval_csv_path": eval_csv_path,
        "csv_file_count": len(_csv_path_list(csv_path)),
        "train_csv_file_count": len(_csv_path_list(train_csv_path)),
        "validation_csv_file_count": len(_csv_path_list(validation_csv_path)),
        "eval_csv_file_count": len(_csv_path_list(eval_csv_path)),
        "selected_cleaned_csv_file_count": len(selected_files),
        "selected_cleaned_csv_files": ";".join(selected_files),
    }

def _worst_uniform_component_delta(component_deltas: dict[str, float | None]) -> dict[str, Any]:
    """Return the most negative MLQDS-vs-uniform range component delta."""
    numeric = [(key, value) for key, value in component_deltas.items() if value is not None]
    if not numeric:
        return {"worst_uniform_component_delta_metric": None, "worst_uniform_component_delta": None}
    key, value = min(numeric, key=lambda item: float(item[1]))
    if float(value) >= 0.0:
        return {"worst_uniform_component_delta_metric": "none", "worst_uniform_component_delta": 0.0}
    return {"worst_uniform_component_delta_metric": key, "worst_uniform_component_delta": float(value)}

def _dominant_runtime_phase(timings: dict[str, Any], elapsed_seconds: float) -> dict[str, Any]:
    """Return the largest parsed child phase and its fraction of wall time."""
    phase_rows = [
        row for row in timings.get("phase_timings", [])
        if row.get("name") is not None and _as_float(row.get("seconds")) is not None
    ]
    if not phase_rows:
        return {
            "runtime_bottleneck_phase": None,
            "runtime_bottleneck_seconds": None,
            "runtime_bottleneck_fraction": None,
        }
    best = max(phase_rows, key=lambda row: float(row["seconds"]))
    seconds = float(best["seconds"])
    elapsed = _as_float(elapsed_seconds)
    return {
        "runtime_bottleneck_phase": str(best["name"]),
        "runtime_bottleneck_seconds": seconds,
        "runtime_bottleneck_fraction": seconds / elapsed if elapsed and elapsed > 0.0 else None,
    }

def _mean_history_value(run_json: dict[str, Any] | None, key: str) -> float | None:
    """Return mean numeric training-history value for one key."""
    history = (run_json or {}).get("training_history", [])
    values = [_as_float(row.get(key)) for row in history if isinstance(row, dict)]
    numeric = [value for value in values if value is not None]
    return float(sum(numeric) / len(numeric)) if numeric else None

def _last_history_value(run_json: dict[str, Any] | None, key: str) -> float | None:
    """Return the last numeric training-history value for one key."""
    history = (run_json or {}).get("training_history", [])
    for row in reversed(history):
        if not isinstance(row, dict):
            continue
        value = _as_float(row.get(key))
        if value is not None:
            return value
    return None

def _single_cell_range_status(
    *,
    returncode: int,
    model_type: Any,
    protocol_enabled: Any,
    primary_frozen: Any,
    audit_frozen: Any,
    audit_ratio_count: int,
    beats_uniform: bool | None,
    beats_dp: bool | None,
    selector_claim_status: str,
) -> str:
    """Classify one benchmark row against the single-cell blind RangeUseful gate."""
    if int(returncode) != 0:
        return "child_failed"
    if beats_uniform is None or beats_dp is None:
        return "missing_range_usefulness"
    if model_type == "range_aware":
        return "diagnostic_upper_bound"
    workload_blind = is_workload_blind_model_type(model_type)
    if not workload_blind:
        return "non_blind_model"
    protocol_ok = bool(protocol_enabled) and bool(primary_frozen)
    if audit_ratio_count > 0:
        protocol_ok = protocol_ok and bool(audit_frozen)
    if not protocol_ok:
        return "protocol_fail"
    if beats_uniform and beats_dp:
        if selector_claim_status in {"missing_selector_evidence", "selector_scaffold_dominated"}:
            return selector_claim_status
        return "beats_uniform_and_douglas_peucker"
    if beats_dp:
        return "fails_uniform"
    if beats_uniform:
        return "fails_douglas_peucker"
    return "fails_uniform_and_douglas_peucker"


def _selector_claim_evidence(selector_budget_row: dict[str, Any], model_type: Any) -> dict[str, Any]:
    """Classify whether the matched selector budget leaves room for learned behavior.

    This is a reporting guard, not a model constraint. A workload-blind run that
    beats baselines with a tiny learned slot fraction is still useful as a
    diagnostic, but it is not evidence that the learned model caused the win.
    """
    if not is_workload_blind_model_type(model_type):
        return {
            "selector_claim_status": "not_workload_blind",
            "selector_claim_has_material_learned_budget": None,
            "selector_claim_min_learned_slot_fraction": None,
        }
    learned_fraction = _as_float(selector_budget_row.get("learned_slot_fraction_of_budget"))
    if learned_fraction is None:
        return {
            "selector_claim_status": "missing_selector_evidence",
            "selector_claim_has_material_learned_budget": None,
            "selector_claim_min_learned_slot_fraction": MIN_MATCHED_LEARNED_SLOT_FRACTION_FOR_BLIND_CLAIM,
        }
    has_material_budget = learned_fraction >= MIN_MATCHED_LEARNED_SLOT_FRACTION_FOR_BLIND_CLAIM
    return {
        "selector_claim_status": (
            "model_has_material_budget" if has_material_budget else "selector_scaffold_dominated"
        ),
        "selector_claim_has_material_learned_budget": bool(has_material_budget),
        "selector_claim_min_learned_slot_fraction": MIN_MATCHED_LEARNED_SLOT_FRACTION_FOR_BLIND_CLAIM,
    }


def _effective_diversity_bonus(model_config: dict[str, Any]) -> float | None:
    """Return the diversity bonus actually consumed by the configured selector."""
    configured = model_config.get("mlqds_diversity_bonus")
    if configured is None:
        return None
    if str(model_config.get("mlqds_hybrid_mode", "fill")).lower() in {"stratified", "global_budget"}:
        return 0.0
    return float(configured)


def _audit_ratio_prefix(ratio: float) -> str:
    """Return a stable, CSV-safe field prefix for one audit compression ratio."""
    return f"audit_ratio_{float(ratio):.4f}".replace(".", "p")


def _ratio_close(left: float | None, right: float, tol: float = 1e-9) -> bool:
    """Return whether two optional ratios should be treated as the same grid value."""
    return left is not None and abs(float(left) - float(right)) <= tol


def _normalized_grid_float(value: Any) -> float | None:
    """Coerce a grid fraction or percent to a normalized fraction."""
    number = _as_float(value)
    if number is None:
        return None
    if number > 1.0 and number <= 100.0:
        number /= 100.0
    return float(number)


def _audit_summary(run_json: dict[str, Any] | None) -> dict[str, Any]:
    """Summarize multi-compression RangeUseful audit wins and low-ratio failures."""
    audit = (run_json or {}).get("range_compression_audit") or {}
    if not isinstance(audit, dict):
        audit = {}
    ratios: list[float] = []
    uniform_deltas: list[float] = []
    dp_deltas: list[float] = []
    random_fill_deltas: list[float] = []
    query_uniform_deltas: list[float] = []
    query_dp_deltas: list[float] = []
    low_uniform_deltas: list[float] = []
    low_dp_deltas: list[float] = []
    low_random_fill_deltas: list[float] = []
    low_query_uniform_deltas: list[float] = []
    low_query_dp_deltas: list[float] = []
    variant_uniform_deltas: dict[str, list[float]] = {
        suffix: [] for suffix, _metric_key in RANGE_USEFULNESS_GAP_VARIANT_KEYS
    }
    variant_low_uniform_deltas: dict[str, list[float]] = {
        suffix: [] for suffix, _metric_key in RANGE_USEFULNESS_GAP_VARIANT_KEYS
    }
    missing_baseline_count = 0
    missing_temporal_random_fill_count = 0
    missing_query_useful_v1_count = 0
    per_ratio_fields: dict[str, Any] = {}

    ratio_rows: list[tuple[float, dict[str, Any]]] = []
    for raw_ratio, methods in audit.items():
        if not isinstance(methods, dict):
            continue
        try:
            ratio = float(raw_ratio)
        except (TypeError, ValueError):
            missing_baseline_count += 1
            continue
        ratio_rows.append((ratio, methods))

    for ratio, methods in sorted(ratio_rows, key=lambda item: item[0]):
        mlqds = methods.get("MLQDS") or {}
        uniform = methods.get("uniform") or {}
        dp = methods.get("DouglasPeucker") or {}
        random_fill = methods.get("TemporalRandomFill") or {}
        mlqds_score = _as_float(mlqds.get("range_usefulness_score"))
        uniform_score = _as_float(uniform.get("range_usefulness_score"))
        dp_score = _as_float(dp.get("range_usefulness_score"))
        random_fill_score = _as_float(random_fill.get("range_usefulness_score"))
        if mlqds_score is None or uniform_score is None or dp_score is None:
            missing_baseline_count += 1
            continue
        prefix = _audit_ratio_prefix(ratio)
        ratios.append(ratio)
        uniform_delta = mlqds_score - uniform_score
        dp_delta = mlqds_score - dp_score
        uniform_deltas.append(uniform_delta)
        dp_deltas.append(dp_delta)
        random_fill_delta: float | None = None
        if random_fill_score is None:
            missing_temporal_random_fill_count += 1
        else:
            random_fill_delta = mlqds_score - random_fill_score
            random_fill_deltas.append(random_fill_delta)
        mlqds_query_score = _as_float(mlqds.get("query_useful_v1_score"))
        uniform_query_score = _as_float(uniform.get("query_useful_v1_score"))
        dp_query_score = _as_float(dp.get("query_useful_v1_score"))
        query_uniform_delta: float | None = None
        query_dp_delta: float | None = None
        query_fields: dict[str, Any] = {}
        if mlqds_query_score is None or uniform_query_score is None or dp_query_score is None:
            missing_query_useful_v1_count += 1
        else:
            query_uniform_delta = float(mlqds_query_score - uniform_query_score)
            query_dp_delta = float(mlqds_query_score - dp_query_score)
            query_uniform_deltas.append(query_uniform_delta)
            query_dp_deltas.append(query_dp_delta)
            query_fields = {
                f"{prefix}_mlqds_query_useful_v1": float(mlqds_query_score),
                f"{prefix}_uniform_query_useful_v1": float(uniform_query_score),
                f"{prefix}_douglas_peucker_query_useful_v1": float(dp_query_score),
                f"{prefix}_mlqds_vs_uniform_query_useful_v1": query_uniform_delta,
                f"{prefix}_mlqds_vs_douglas_peucker_query_useful_v1": query_dp_delta,
            }
        variant_fields: dict[str, Any] = {}
        for suffix, metric_key in RANGE_USEFULNESS_GAP_VARIANT_KEYS:
            mlqds_variant = _as_float(mlqds.get(metric_key))
            uniform_variant = _as_float(uniform.get(metric_key))
            if mlqds_variant is None or uniform_variant is None:
                continue
            variant_delta = mlqds_variant - uniform_variant
            variant_uniform_deltas[suffix].append(float(variant_delta))
            if ratio <= LOW_COMPRESSION_THRESHOLD:
                variant_low_uniform_deltas[suffix].append(float(variant_delta))
            variant_fields.update(
                {
                    f"{prefix}_mlqds_vs_uniform_range_usefulness_{suffix}": float(variant_delta),
                }
            )
        per_ratio_fields.update(
            {
                f"{prefix}_compression_ratio": float(ratio),
                f"{prefix}_mlqds_range_usefulness": float(mlqds_score),
                f"{prefix}_uniform_range_usefulness": float(uniform_score),
                f"{prefix}_douglas_peucker_range_usefulness": float(dp_score),
                f"{prefix}_temporal_random_fill_range_usefulness": random_fill_score,
                f"{prefix}_mlqds_vs_uniform_range_usefulness": float(uniform_delta),
                f"{prefix}_mlqds_vs_douglas_peucker_range_usefulness": float(dp_delta),
                f"{prefix}_mlqds_vs_temporal_random_fill_range_usefulness": random_fill_delta,
                **query_fields,
                **variant_fields,
            }
        )
        if ratio <= LOW_COMPRESSION_THRESHOLD:
            low_uniform_deltas.append(uniform_delta)
            low_dp_deltas.append(dp_delta)
            if random_fill_delta is not None:
                low_random_fill_deltas.append(random_fill_delta)
            if query_uniform_delta is not None:
                low_query_uniform_deltas.append(query_uniform_delta)
            if query_dp_delta is not None:
                low_query_dp_deltas.append(query_dp_delta)

    def _mean(values: list[float]) -> float | None:
        return float(sum(values) / len(values)) if values else None

    low_both = [
        1
        for uniform_delta, dp_delta in zip(low_uniform_deltas, low_dp_deltas)
        if uniform_delta > 0.0 and dp_delta > 0.0
    ]
    all_both = [
        1
        for uniform_delta, dp_delta in zip(uniform_deltas, dp_deltas)
        if uniform_delta > 0.0 and dp_delta > 0.0
    ]
    query_low_both = [
        1
        for uniform_delta, dp_delta in zip(low_query_uniform_deltas, low_query_dp_deltas)
        if uniform_delta > 0.0 and dp_delta > 0.0
    ]
    query_all_both = [
        1
        for uniform_delta, dp_delta in zip(query_uniform_deltas, query_dp_deltas)
        if uniform_delta > 0.0 and dp_delta > 0.0
    ]
    summary: dict[str, Any] = {
        "audit_compression_ratio_count": len(ratios),
        "audit_low_compression_ratio_count": len(low_uniform_deltas),
        "audit_missing_baseline_count": int(missing_baseline_count),
        "audit_missing_temporal_random_fill_count": int(missing_temporal_random_fill_count),
        "audit_missing_query_useful_v1_count": int(missing_query_useful_v1_count),
        "audit_beats_uniform_range_usefulness_count": sum(1 for value in uniform_deltas if value > 0.0),
        "audit_beats_douglas_peucker_range_usefulness_count": sum(1 for value in dp_deltas if value > 0.0),
        "audit_beats_temporal_random_fill_range_usefulness_count": sum(
            1 for value in random_fill_deltas if value > 0.0
        ),
        "audit_beats_both_range_usefulness_count": len(all_both),
        "audit_low_beats_uniform_range_usefulness_count": sum(1 for value in low_uniform_deltas if value > 0.0),
        "audit_low_beats_douglas_peucker_range_usefulness_count": sum(1 for value in low_dp_deltas if value > 0.0),
        "audit_low_beats_temporal_random_fill_range_usefulness_count": sum(
            1 for value in low_random_fill_deltas if value > 0.0
        ),
        "audit_low_beats_both_range_usefulness_count": len(low_both),
        "audit_beats_uniform_query_useful_v1_count": sum(1 for value in query_uniform_deltas if value > 0.0),
        "audit_beats_douglas_peucker_query_useful_v1_count": sum(
            1 for value in query_dp_deltas if value > 0.0
        ),
        "audit_beats_both_query_useful_v1_count": len(query_all_both),
        "audit_low_beats_uniform_query_useful_v1_count": sum(
            1 for value in low_query_uniform_deltas if value > 0.0
        ),
        "audit_low_beats_douglas_peucker_query_useful_v1_count": sum(
            1 for value in low_query_dp_deltas if value > 0.0
        ),
        "audit_low_beats_both_query_useful_v1_count": len(query_low_both),
        "audit_min_vs_uniform_range_usefulness": min(uniform_deltas) if uniform_deltas else None,
        "audit_mean_vs_uniform_range_usefulness": _mean(uniform_deltas),
        "audit_min_vs_uniform_query_useful_v1": min(query_uniform_deltas) if query_uniform_deltas else None,
        "audit_mean_vs_uniform_query_useful_v1": _mean(query_uniform_deltas),
        "audit_min_vs_temporal_random_fill_range_usefulness": (
            min(random_fill_deltas) if random_fill_deltas else None
        ),
        "audit_mean_vs_temporal_random_fill_range_usefulness": _mean(random_fill_deltas),
        "audit_min_low_vs_uniform_range_usefulness": min(low_uniform_deltas) if low_uniform_deltas else None,
        "audit_mean_low_vs_uniform_range_usefulness": _mean(low_uniform_deltas),
        "audit_min_low_vs_uniform_query_useful_v1": (
            min(low_query_uniform_deltas) if low_query_uniform_deltas else None
        ),
        "audit_mean_low_vs_uniform_query_useful_v1": _mean(low_query_uniform_deltas),
        "audit_min_low_vs_temporal_random_fill_range_usefulness": (
            min(low_random_fill_deltas) if low_random_fill_deltas else None
        ),
        "audit_mean_low_vs_temporal_random_fill_range_usefulness": _mean(low_random_fill_deltas),
    }
    for suffix, _metric_key in RANGE_USEFULNESS_GAP_VARIANT_KEYS:
        deltas = variant_uniform_deltas[suffix]
        low_deltas = variant_low_uniform_deltas[suffix]
        summary.update(
            {
                f"audit_beats_uniform_range_usefulness_{suffix}_count": sum(
                    1 for value in deltas if value > 0.0
                ),
                f"audit_low_beats_uniform_range_usefulness_{suffix}_count": sum(
                    1 for value in low_deltas if value > 0.0
                ),
                f"audit_min_vs_uniform_range_usefulness_{suffix}": min(deltas) if deltas else None,
                f"audit_mean_vs_uniform_range_usefulness_{suffix}": _mean(deltas),
                f"audit_min_low_vs_uniform_range_usefulness_{suffix}": (
                    min(low_deltas) if low_deltas else None
                ),
                f"audit_mean_low_vs_uniform_range_usefulness_{suffix}": _mean(low_deltas),
            }
        )
    summary.update(per_ratio_fields)
    return summary


def query_driven_final_grid_summary(
    rows: list[dict[str, Any]],
    run_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return benchmark-level QueryUsefulV1 final-grid acceptance evidence."""
    run_config = run_config or {}
    profile_settings = run_config.get("profile_settings") or {}
    required_coverages = tuple(
        float(value)
        for value in (
            profile_settings.get("range_coverage_sweep_targets") or QUERY_DRIVEN_FINAL_COVERAGE_TARGETS
        )
    )
    required_ratios = tuple(
        float(value)
        for value in (
            profile_settings.get("range_compression_sweep_ratios")
            or QUERY_DRIVEN_FINAL_COMPRESSION_RATIOS
        )
    )
    final_candidate = bool(profile_settings.get("final_product_candidate")) or any(
        row.get("mlqds_primary_metric") == "query_useful_v1" for row in rows
    )
    coverage_rows: dict[float, dict[str, Any]] = {}
    duplicate_coverages: list[float] = []
    for row in rows:
        coverage_raw = row.get("query_target_coverage")
        if coverage_raw is None:
            coverage_raw = row.get("workload_stability_configured_target_coverage")
        coverage = _normalized_grid_float(coverage_raw)
        matched_target = next(
            (target for target in required_coverages if _ratio_close(coverage, target, tol=1e-6)),
            None,
        )
        if matched_target is None:
            continue
        if matched_target in coverage_rows:
            duplicate_coverages.append(float(matched_target))
            continue
        coverage_rows[matched_target] = row

    missing_coverages = [
        float(target)
        for target in required_coverages
        if target not in coverage_rows
    ]
    cells: list[dict[str, Any]] = []
    missing_cells: list[dict[str, Any]] = []
    for coverage in required_coverages:
        row = coverage_rows.get(coverage)
        for ratio in required_ratios:
            if row is None:
                missing_cells.append(
                    {"coverage": float(coverage), "compression_ratio": float(ratio), "reason": "missing_coverage_row"}
                )
                continue
            prefix = _audit_ratio_prefix(ratio)
            mlqds = _as_float(row.get(f"{prefix}_mlqds_query_useful_v1"))
            uniform = _as_float(row.get(f"{prefix}_uniform_query_useful_v1"))
            dp = _as_float(row.get(f"{prefix}_douglas_peucker_query_useful_v1"))
            if mlqds is None or uniform is None or dp is None:
                if _ratio_close(_normalized_grid_float(row.get("compression_ratio")), ratio, tol=1e-6):
                    mlqds = _as_float(row.get("mlqds_query_useful_v1_score"))
                    uniform = _as_float(row.get("uniform_query_useful_v1_score"))
                    dp = _as_float(row.get("douglas_peucker_query_useful_v1_score"))
            if mlqds is None or uniform is None or dp is None:
                missing_cells.append(
                    {
                        "coverage": float(coverage),
                        "compression_ratio": float(ratio),
                        "reason": "missing_query_useful_v1_scores",
                    }
                )
                continue
            cells.append(
                {
                    "coverage": float(coverage),
                    "compression_ratio": float(ratio),
                    "mlqds_query_useful_v1": float(mlqds),
                    "uniform_query_useful_v1": float(uniform),
                    "douglas_peucker_query_useful_v1": float(dp),
                    "mlqds_vs_uniform_query_useful_v1": float(mlqds - uniform),
                    "mlqds_vs_douglas_peucker_query_useful_v1": float(mlqds - dp),
                    "beats_uniform": bool(mlqds > uniform),
                    "beats_douglas_peucker": bool(mlqds > dp),
                    "low_budget": bool(ratio <= LOW_COMPRESSION_THRESHOLD),
                }
            )

    uniform_wins = sum(1 for cell in cells if cell["beats_uniform"])
    dp_wins = sum(1 for cell in cells if cell["beats_douglas_peucker"])
    low_uniform_wins = sum(1 for cell in cells if cell["low_budget"] and cell["beats_uniform"])
    matched_5_percent_uniform_wins = sum(
        1
        for cell in cells
        if _ratio_close(_as_float(cell.get("compression_ratio")), 0.05, tol=1e-9)
        and cell["beats_uniform"]
    )
    required_cell_count = int(len(required_coverages) * len(required_ratios))
    grid_complete = len(cells) == required_cell_count and not missing_cells and not missing_coverages
    required_single_run_gate_names = (
        "workload_stability_gate_pass",
        "support_overlap_gate_pass",
        "predictability_gate_pass",
        "target_diffusion_gate_pass",
        "workload_signature_gate_pass",
        "learning_causality_gate_pass",
        "prior_sample_gate_pass",
        "global_sanity_gate_pass",
    )
    child_gate_failures: list[dict[str, Any]] = []
    for coverage, row in sorted(coverage_rows.items()):
        failed = [
            name
            for name in required_single_run_gate_names
            if row.get(name) is not True
        ]
        if int(row.get("returncode", 1) or 0) != 0:
            failed.append("child_returncode_nonzero")
        if failed:
            child_gate_failures.append(
                {
                    "coverage": float(coverage),
                    "run_label": row.get("run_label"),
                    "failed_gates": failed,
                }
            )

    numeric_success_pass = (
        grid_complete
        and uniform_wins >= QUERY_DRIVEN_MIN_UNIFORM_WINS
        and dp_wins >= QUERY_DRIVEN_MIN_DP_WINS
        and low_uniform_wins >= QUERY_DRIVEN_MIN_LOW_BUDGET_UNIFORM_WINS
        and matched_5_percent_uniform_wins >= QUERY_DRIVEN_MIN_MATCHED_5_PERCENT_UNIFORM_WINS
    )
    failed_checks: list[str] = []
    if not final_candidate:
        failed_checks.append("not_final_product_candidate_profile")
    if missing_coverages:
        failed_checks.append("coverage_grid_incomplete")
    if missing_cells:
        failed_checks.append("compression_grid_incomplete")
    if duplicate_coverages:
        failed_checks.append("duplicate_coverage_rows")
    if uniform_wins < QUERY_DRIVEN_MIN_UNIFORM_WINS:
        failed_checks.append("too_few_uniform_queryuseful_wins")
    if dp_wins < QUERY_DRIVEN_MIN_DP_WINS:
        failed_checks.append("too_few_douglas_peucker_queryuseful_wins")
    if low_uniform_wins < QUERY_DRIVEN_MIN_LOW_BUDGET_UNIFORM_WINS:
        failed_checks.append("too_few_low_budget_uniform_queryuseful_wins")
    if matched_5_percent_uniform_wins < QUERY_DRIVEN_MIN_MATCHED_5_PERCENT_UNIFORM_WINS:
        failed_checks.append("too_few_matched_5_percent_uniform_queryuseful_wins")
    if child_gate_failures:
        failed_checks.append("required_single_run_gates_failed")

    final_success_allowed = bool(final_candidate and numeric_success_pass and not child_gate_failures and not failed_checks)
    return {
        "schema_version": 1,
        "primary_metric": "QueryUsefulV1",
        "status": "final_grid_pass" if final_success_allowed else "final_grid_blocked",
        "final_success_allowed": final_success_allowed,
        "failed_checks": failed_checks,
        "final_product_candidate_profile": bool(final_candidate),
        "required_coverage_targets": list(required_coverages),
        "required_compression_ratios": list(required_ratios),
        "required_cell_count": required_cell_count,
        "observed_cell_count": int(len(cells)),
        "grid_complete": bool(grid_complete),
        "missing_coverage_targets": missing_coverages,
        "duplicate_coverage_targets": duplicate_coverages,
        "missing_cells": missing_cells,
        "beats_uniform_queryuseful_cells": int(uniform_wins),
        "beats_uniform_queryuseful_cells_min": QUERY_DRIVEN_MIN_UNIFORM_WINS,
        "beats_douglas_peucker_queryuseful_cells": int(dp_wins),
        "beats_douglas_peucker_queryuseful_cells_min": QUERY_DRIVEN_MIN_DP_WINS,
        "low_budget_beats_uniform_queryuseful_cells": int(low_uniform_wins),
        "low_budget_beats_uniform_queryuseful_cells_min": QUERY_DRIVEN_MIN_LOW_BUDGET_UNIFORM_WINS,
        "matched_5_percent_coverage_cells_uniform": int(matched_5_percent_uniform_wins),
        "matched_5_percent_coverage_cells_uniform_min": QUERY_DRIVEN_MIN_MATCHED_5_PERCENT_UNIFORM_WINS,
        "numeric_success_bars_pass": bool(numeric_success_pass),
        "required_single_run_gate_names": list(required_single_run_gate_names),
        "child_gate_failures": child_gate_failures,
        "cells": cells,
    }


def _phase_seconds(timings: dict[str, Any], name: str) -> float | None:
    """Extract one phase duration from parsed timings."""
    for row in timings.get("phase_timings", []):
        if row.get("name") == name:
            return float(row["seconds"])
    return None

def _phase_seconds_with_prefix(timings: dict[str, Any], prefix: str) -> float | None:
    """Extract the first phase duration whose name starts with a prefix."""
    for row in timings.get("phase_timings", []):
        if str(row.get("name", "")).startswith(prefix):
            return float(row["seconds"])
    return None

def _mean_epoch_seconds(timings: dict[str, Any]) -> float | None:
    """Return mean epoch duration from parsed stdout timings."""
    values = [float(row["seconds"]) for row in timings.get("epoch_timings", [])]
    return float(sum(values) / len(values)) if values else None

def _collapse_warning_summary(run_json: dict[str, Any] | None) -> dict[str, Any]:
    """Summarize collapse diagnostics without conflating any epoch with the selected checkpoint."""
    if not run_json:
        return {
            "collapse_warning_any": None,
            "collapse_warning_count": None,
            "best_epoch_collapse_warning": None,
            "min_pred_std": None,
            "best_epoch_pred_std": None,
        }
    history = run_json.get("training_history", [])
    collapse_count = sum(1 for row in history if bool(row.get("collapse_warning", False)))
    pred_std_values = [
        float(row["pred_std"])
        for row in history
        if row.get("pred_std") is not None
    ]
    best_epoch = run_json.get("best_epoch")
    best_row = None
    if best_epoch is not None:
        best_epoch_int = int(best_epoch)
        for idx, row in enumerate(history):
            epoch_one_based = int(row.get("epoch", idx)) + 1
            if epoch_one_based == best_epoch_int:
                best_row = row
                break
    return {
        "collapse_warning_any": bool(collapse_count > 0),
        "collapse_warning_count": int(collapse_count),
        "best_epoch_collapse_warning": (
            bool(best_row.get("collapse_warning", False)) if best_row is not None else None
        ),
        "min_pred_std": min(pred_std_values) if pred_std_values else None,
        "best_epoch_pred_std": (
            float(best_row["pred_std"])
            if best_row is not None and best_row.get("pred_std") is not None
            else None
        ),
    }

def _target_budget_row(target_diagnostics: dict[str, Any], compression_ratio: Any) -> dict[str, Any]:
    """Return the target-diagnostics budget row closest to the run compression ratio."""
    rows = target_diagnostics.get("budget_rows") or []
    if not isinstance(rows, list) or not rows:
        return {}
    try:
        target_ratio = float(compression_ratio)
    except (TypeError, ValueError):
        last_row = rows[-1]
        return last_row if isinstance(last_row, dict) else {}

    best_row: dict[str, Any] = {}
    best_distance = float("inf")
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_ratio = row.get("total_budget_ratio")
        if raw_ratio is None:
            continue
        try:
            ratio = float(raw_ratio)
        except (TypeError, ValueError):
            continue
        distance = abs(ratio - target_ratio)
        if distance < best_distance:
            best_distance = distance
            best_row = row
    return best_row

def _selector_budget_row(selector_diagnostics: dict[str, Any], compression_ratio: Any) -> dict[str, Any]:
    """Return the selector-capacity budget row closest to the run compression ratio."""
    rows = selector_diagnostics.get("budget_rows") or []
    if not isinstance(rows, list) or not rows:
        return {}
    try:
        target_ratio = float(compression_ratio)
    except (TypeError, ValueError):
        last_row = rows[-1]
        return last_row if isinstance(last_row, dict) else {}

    best_row: dict[str, Any] = {}
    best_distance = float("inf")
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_ratio = row.get("compression_ratio")
        if raw_ratio is None:
            continue
        try:
            ratio = float(raw_ratio)
        except (TypeError, ValueError):
            continue
        distance = abs(ratio - target_ratio)
        if distance < best_distance:
            best_distance = distance
            best_row = row
    return best_row

def _selector_low_budget_summary(selector_diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Summarize learned-slot capacity over low compression ratios."""
    rows = []
    for row in selector_diagnostics.get("budget_rows") or []:
        if not isinstance(row, dict):
            continue
        ratio = _as_float(row.get("compression_ratio"))
        if ratio is not None and ratio <= LOW_COMPRESSION_THRESHOLD:
            rows.append(row)
    if not rows:
        return {
            "eval_selector_low_budget_zero_learned_ratio_count": None,
            "eval_selector_low_budget_min_learned_slot_fraction": None,
        }
    learned_fractions = [
        float(row.get("learned_slot_fraction_of_budget") or 0.0)
        for row in rows
    ]
    return {
        "eval_selector_low_budget_zero_learned_ratio_count": sum(
            1 for row in rows if int(row.get("learned_slot_count") or 0) <= 0
        ),
        "eval_selector_low_budget_min_learned_slot_fraction": min(learned_fractions),
    }

def _query_generation(run_json: dict[str, Any] | None, split: str) -> dict[str, Any]:
    """Return query-generation diagnostics for one split."""
    diagnostics = (run_json or {}).get("query_generation_diagnostics") or {}
    split_payload = diagnostics.get(split) or {}
    if not isinstance(split_payload, dict):
        return {}
    query_generation = split_payload.get("query_generation") or {}
    return query_generation if isinstance(query_generation, dict) else {}

def _workload_distribution_summary(run_json: dict[str, Any] | None, split: str) -> dict[str, Any]:
    """Return workload distribution summary for one split."""
    summaries = ((run_json or {}).get("workload_distribution_comparison") or {}).get("summaries") or {}
    summary = summaries.get(split) or {}
    return summary if isinstance(summary, dict) else {}

def _query_floor_fields(prefix: str, generation: dict[str, Any]) -> dict[str, Any]:
    """Return coverage-target query-floor diagnostics for one split."""
    target_coverage = _as_float(generation.get("target_coverage"))
    final_coverage = _as_float(generation.get("final_coverage"))
    final_query_count = _as_float(generation.get("final_query_count"))
    target_reached_query_count = _as_float(generation.get("target_reached_query_count"))
    extra_after_target = _as_float(generation.get("extra_queries_after_target_reached"))
    target_reached = (
        None
        if target_coverage is None or final_coverage is None
        else bool(final_coverage + 1e-12 >= target_coverage)
    )
    target_shortfall = (
        None
        if target_coverage is None or final_coverage is None
        else max(0.0, target_coverage - final_coverage)
    )
    target_overshoot = (
        None
        if target_coverage is None or final_coverage is None
        else max(0.0, final_coverage - target_coverage)
    )
    return {
        f"{prefix}_query_generation_mode": generation.get("mode"),
        f"{prefix}_query_generation_stop_reason": generation.get("stop_reason"),
        f"{prefix}_query_coverage_calibration_mode": generation.get("coverage_calibration_mode"),
        f"{prefix}_query_target_coverage": generation.get("target_coverage"),
        f"{prefix}_query_final_coverage": generation.get("final_coverage"),
        f"{prefix}_query_target_reached": target_reached,
        f"{prefix}_query_target_shortfall": target_shortfall,
        f"{prefix}_query_target_overshoot": target_overshoot,
        f"{prefix}_query_target_missed_by_max_queries": (
            bool(generation.get("stop_reason") == "max_queries_reached" and target_shortfall and target_shortfall > 0.0)
            if target_shortfall is not None
            else None
        ),
        f"{prefix}_query_minimum_queries": generation.get("minimum_queries"),
        f"{prefix}_query_max_queries": generation.get("max_queries"),
        f"{prefix}_query_final_count": generation.get("final_query_count"),
        f"{prefix}_query_target_reached_count": generation.get("target_reached_query_count"),
        f"{prefix}_query_coverage_at_target_reached": generation.get("coverage_at_target_reached"),
        f"{prefix}_query_extra_after_target_reached": generation.get("extra_queries_after_target_reached"),
        f"{prefix}_query_extra_after_target_fraction": (
            float(extra_after_target) / float(final_query_count)
            if extra_after_target is not None and final_query_count and final_query_count > 0.0
            else None
        ),
        f"{prefix}_query_floor_dominated": (
            bool(
                target_reached_query_count is not None
                and final_query_count is not None
                and final_query_count > target_reached_query_count
            )
        ),
        f"{prefix}_query_coverage_guard_enabled": generation.get("coverage_guard_enabled"),
        f"{prefix}_query_max_allowed_coverage": generation.get("max_allowed_coverage"),
    }

def _workload_generation_fields(run_json: dict[str, Any] | None, split: str) -> dict[str, Any]:
    """Flatten query-generation and workload-distribution diagnostics for one split."""
    generation = _query_generation(run_json, split)
    summary = _workload_distribution_summary(run_json, split)
    fields = _query_floor_fields(split, generation)
    fields.update(
        {
            f"{split}_workload_range_query_count": summary.get("range_query_count"),
            f"{split}_workload_coverage_fraction": summary.get("coverage_fraction"),
            f"{split}_workload_empty_query_rate": summary.get("empty_query_rate"),
            f"{split}_workload_too_broad_query_rate": summary.get("too_broad_query_rate"),
            f"{split}_workload_near_duplicate_query_rate": summary.get("near_duplicate_query_rate"),
            f"{split}_workload_point_hit_count_p50": summary.get("point_hit_count_p50"),
            f"{split}_workload_trajectory_hit_count_p50": summary.get("trajectory_hit_count_p50"),
            f"{split}_workload_oracle_gap_over_best_baseline": summary.get("oracle_gap_over_best_baseline"),
            f"{split}_workload_best_baseline": summary.get("best_baseline"),
        }
    )
    return fields

def _row_from_run(
    *,
    workload: str,
    run_label: str,
    command: list[str],
    returncode: int,
    elapsed_seconds: float,
    run_dir: Path,
    stdout_path: Path,
    run_json_path: Path,
    timings: dict[str, Any],
    run_json: dict[str, Any] | None,
    data_sources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one compact comparison row."""
    mlqds = (run_json or {}).get("matched", {}).get("MLQDS", {})
    uniform = (run_json or {}).get("matched", {}).get("uniform", {})
    dp = (run_json or {}).get("matched", {}).get("DouglasPeucker", {})
    learned_fill = (run_json or {}).get("learned_fill_diagnostics", {})
    temporal_random_fill = learned_fill.get("TemporalRandomFill", {})
    temporal_oracle_fill = learned_fill.get("TemporalOracleFill", {})
    cuda_memory = (run_json or {}).get("cuda_memory", {}).get("training", {})
    child_torch_runtime = (run_json or {}).get("torch_runtime") or {}
    child_amp = child_torch_runtime.get("amp") or {}
    data_config = (run_json or {}).get("config", {}).get("data", {})
    model_config = (run_json or {}).get("config", {}).get("model", {})
    query_config = (run_json or {}).get("config", {}).get("query", {})
    baseline_config = (run_json or {}).get("config", {}).get("baselines", {})
    oracle_diagnostic = (run_json or {}).get("oracle_diagnostic") or {}
    workload_blind_protocol = (run_json or {}).get("workload_blind_protocol") or {}
    teacher_distillation = (run_json or {}).get("teacher_distillation") or {}
    collapse_summary = _collapse_warning_summary(run_json)
    train_label_diagnostics = (
        (run_json or {})
        .get("workload_diagnostics", {})
        .get("train", {})
        .get("range_signal", {})
        .get("labels", {})
    )
    label_mass_fraction = train_label_diagnostics.get("component_positive_label_mass_fraction", {})
    target_diagnostics = (run_json or {}).get("training_target_diagnostics") or {}
    target_transform = (run_json or {}).get("range_training_target_transform") or {}
    fit_diagnostics = (run_json or {}).get("training_fit_diagnostics") or {}
    final_claim_summary = (run_json or {}).get("final_claim_summary") or {}
    legacy_range_useful_summary = (run_json or {}).get("legacy_range_useful_summary") or {}
    predictability_audit = (run_json or {}).get("predictability_audit") or {}
    predictability_metrics = predictability_audit.get("metrics") or {}
    learning_causality = (run_json or {}).get("learning_causality_summary") or {}
    learning_delta_gate = learning_causality.get("learning_causality_delta_gate") or {}
    prior_sensitivity = learning_causality.get("prior_sensitivity_diagnostics") or {}
    shuffled_prior_sample = (
        (prior_sensitivity.get("shuffled_prior_fields") or {}).get("sampled_prior_features") or {}
    )
    no_prior_sample = (
        (prior_sensitivity.get("without_query_prior_features") or {}).get("sampled_prior_features") or {}
    )
    workload_stability_gate = (run_json or {}).get("workload_stability_gate") or {}
    support_overlap_gate = (run_json or {}).get("support_overlap_gate") or {}
    global_sanity_gate = (run_json or {}).get("global_sanity_gate") or {}
    target_diffusion_gate = (run_json or {}).get("target_diffusion_gate") or {}
    workload_signature_gate = ((run_json or {}).get("workload_distribution_comparison") or {}).get(
        "workload_signature_gate"
    ) or {}
    signature_pairs = workload_signature_gate.get("pairs") or {}
    signature_train_pair = (workload_signature_gate.get("pairs") or {}).get("train") or {}
    signature_train_metrics = signature_train_pair.get("metrics") or {}
    point_hit_signature_distance = signature_train_metrics.get(
        "point_hit_distribution_ks",
        signature_train_metrics.get("point_hit_distribution_ks_proxy"),
    )
    ship_hit_signature_distance = signature_train_metrics.get(
        "ship_hit_distribution_ks",
        signature_train_metrics.get("ship_hit_distribution_ks_proxy"),
    )
    eval_selector_diagnostics = ((run_json or {}).get("selector_budget_diagnostics") or {}).get("eval") or {}
    target_budget_row = _target_budget_row(target_diagnostics, model_config.get("compression_ratio"))
    selector_budget_row = _selector_budget_row(eval_selector_diagnostics, model_config.get("compression_ratio"))
    selector_low_budget_summary = _selector_low_budget_summary(eval_selector_diagnostics)
    selector_claim_evidence = _selector_claim_evidence(
        selector_budget_row,
        model_config.get("model_type"),
    )
    mlqds_aggregate_f1 = mlqds.get("aggregate_f1")
    mlqds_range_point_f1 = mlqds.get("range_point_f1", mlqds_aggregate_f1)
    mlqds_range_usefulness = mlqds.get("range_usefulness_score")
    mlqds_query_useful_v1 = mlqds.get("query_useful_v1_score")
    mlqds_gap_time_usefulness = mlqds.get("range_usefulness_gap_time_score")
    mlqds_gap_distance_usefulness = mlqds.get("range_usefulness_gap_distance_score")
    mlqds_gap_min_usefulness = mlqds.get("range_usefulness_gap_min_score")
    if final_claim_summary.get("primary_metric") == "QueryUsefulV1" and mlqds_query_useful_v1 is not None:
        mlqds_primary_score = mlqds_query_useful_v1
        mlqds_primary_metric = "query_useful_v1"
    else:
        mlqds_primary_score = mlqds_range_usefulness if mlqds_range_usefulness is not None else mlqds_range_point_f1
        mlqds_primary_metric = "range_usefulness" if mlqds_range_usefulness is not None else "range_point_f1"
    random_fill_range_usefulness = temporal_random_fill.get("range_usefulness_score")
    oracle_fill_range_usefulness = temporal_oracle_fill.get("range_usefulness_score")
    uniform_aggregate_f1 = uniform.get("aggregate_f1")
    uniform_range_point_f1 = uniform.get("range_point_f1", uniform_aggregate_f1)
    uniform_range_usefulness = uniform.get("range_usefulness_score")
    uniform_query_useful_v1 = uniform.get("query_useful_v1_score")
    uniform_gap_time_usefulness = uniform.get("range_usefulness_gap_time_score")
    uniform_gap_distance_usefulness = uniform.get("range_usefulness_gap_distance_score")
    uniform_gap_min_usefulness = uniform.get("range_usefulness_gap_min_score")
    dp_aggregate_f1 = dp.get("aggregate_f1")
    dp_range_point_f1 = dp.get("range_point_f1", dp_aggregate_f1)
    dp_range_usefulness = dp.get("range_usefulness_score")
    dp_query_useful_v1 = dp.get("query_useful_v1_score")
    dp_gap_time_usefulness = dp.get("range_usefulness_gap_time_score")
    dp_gap_distance_usefulness = dp.get("range_usefulness_gap_distance_score")
    dp_gap_min_usefulness = dp.get("range_usefulness_gap_min_score")
    component_deltas = {
        f"mlqds_vs_uniform_{key}": _metric_delta(mlqds, uniform, key)
        for key in RANGE_COMPONENT_KEYS
    }
    worst_component_delta = _worst_uniform_component_delta(component_deltas)
    audit = _audit_summary(run_json)
    runtime_bottleneck = _dominant_runtime_phase(timings, elapsed_seconds)
    beats_uniform_range_usefulness = _metric_beats(mlqds, uniform, "range_usefulness_score")
    beats_dp_range_usefulness = _metric_beats(mlqds, dp, "range_usefulness_score")
    beats_temporal_random_fill_range_usefulness = _metric_beats(
        mlqds,
        temporal_random_fill,
        "range_usefulness_score",
    )
    single_cell_range_status = _single_cell_range_status(
        returncode=returncode,
        model_type=model_config.get("model_type"),
        protocol_enabled=workload_blind_protocol.get("enabled"),
        primary_frozen=workload_blind_protocol.get("primary_masks_frozen_before_eval_query_scoring"),
        audit_frozen=workload_blind_protocol.get("audit_masks_frozen_before_eval_query_scoring"),
        audit_ratio_count=int(audit["audit_compression_ratio_count"]),
        beats_uniform=beats_uniform_range_usefulness,
        beats_dp=beats_dp_range_usefulness,
        selector_claim_status=str(selector_claim_evidence["selector_claim_status"]),
    )
    return {
        "workload": workload,
        "run_label": run_label,
        **_data_source_row_fields(data_sources),
        "returncode": int(returncode),
        "elapsed_seconds": float(elapsed_seconds),
        "train_seconds": _phase_seconds_with_prefix(timings, "train-model"),
        "evaluate_matched_seconds": _phase_seconds(timings, "evaluate-matched"),
        "epoch_mean_seconds": _mean_epoch_seconds(timings),
        "peak_allocated_mb": cuda_memory.get("max_allocated_mb"),
        "best_epoch": (run_json or {}).get("best_epoch"),
        "best_loss": (run_json or {}).get("best_loss"),
        "best_selection_score": (run_json or {}).get("best_selection_score"),
        "final_loss": _last_history_value(run_json, "loss"),
        "final_kendall_tau_t0": _last_history_value(run_json, "kendall_tau_t0"),
        "final_pred_std": _last_history_value(run_json, "pred_std"),
        "epoch_forward_mean_seconds": _mean_history_value(run_json, "epoch_forward_seconds"),
        "epoch_loss_mean_seconds": _mean_history_value(run_json, "epoch_loss_seconds"),
        "epoch_backward_mean_seconds": _mean_history_value(run_json, "epoch_backward_seconds"),
        "epoch_diagnostic_mean_seconds": _mean_history_value(run_json, "epoch_diagnostic_seconds"),
        "epoch_validation_score_mean_seconds": _mean_history_value(run_json, "epoch_validation_score_seconds"),
        "single_cell_range_status": single_cell_range_status,
        "final_claim_status": final_claim_summary.get("status", "not_available_until_query_useful_v1"),
        "final_success_allowed": bool(final_claim_summary.get("final_success_allowed", False)),
        "final_claim_blocking_gates": final_claim_summary.get("blocking_gates"),
        "workload_stability_gate_pass": workload_stability_gate.get("gate_pass"),
        "workload_stability_failed_checks": workload_stability_gate.get("failed_checks"),
        "workload_stability_train_replicates": workload_stability_gate.get("train_workload_replicate_count"),
        "workload_stability_configured_target_coverage": workload_stability_gate.get(
            "configured_target_coverage"
        ),
        "support_overlap_gate_pass": support_overlap_gate.get("gate_pass"),
        "support_overlap_failed_checks": support_overlap_gate.get("failed_checks"),
        "support_eval_points_outside_train_prior_extent_fraction": support_overlap_gate.get(
            "eval_points_outside_train_prior_extent_fraction"
        ),
        "support_sampled_prior_nonzero_fraction": support_overlap_gate.get("sampled_prior_nonzero_fraction"),
        "support_primary_sampled_prior_nonzero_fraction": support_overlap_gate.get(
            "primary_sampled_prior_nonzero_fraction"
        ),
        "support_route_density_overlap": support_overlap_gate.get("route_density_overlap"),
        "support_query_prior_support_overlap": support_overlap_gate.get("query_prior_support_overlap"),
        "support_train_eval_spatial_extent_intersection_fraction": support_overlap_gate.get(
            "train_eval_spatial_extent_intersection_fraction"
        ),
        "global_sanity_gate_pass": global_sanity_gate.get("gate_pass"),
        "global_sanity_failed_checks": global_sanity_gate.get("failed_checks"),
        "global_sanity_endpoint_sanity": global_sanity_gate.get("endpoint_sanity"),
        "global_sanity_avg_sed_ratio_vs_uniform": global_sanity_gate.get("avg_sed_ratio_vs_uniform"),
        "global_sanity_avg_sed_ratio_vs_uniform_max": global_sanity_gate.get(
            "avg_sed_ratio_vs_uniform_max"
        ),
        "global_sanity_avg_length_preserved": global_sanity_gate.get("avg_length_preserved"),
        "target_diffusion_gate_pass": target_diffusion_gate.get("gate_pass"),
        "target_diffusion_failed_checks": target_diffusion_gate.get("failed_checks"),
        "target_diffusion_final_label_support_fraction": target_diffusion_gate.get(
            "final_label_support_fraction"
        ),
        "predictability_gate_pass": predictability_audit.get("gate_pass"),
        "predictability_spearman": predictability_metrics.get("spearman"),
        "predictability_kendall_tau": predictability_metrics.get("kendall_tau"),
        "predictability_lift_at_1_percent": predictability_metrics.get("lift_at_1_percent"),
        "predictability_lift_at_2_percent": predictability_metrics.get("lift_at_2_percent"),
        "predictability_lift_at_5_percent": predictability_metrics.get("lift_at_5_percent"),
        "predictability_pr_auc_lift_over_base_rate": predictability_metrics.get("pr_auc_lift_over_base_rate"),
        "workload_signature_gate_pass": workload_signature_gate.get("all_pass"),
        "workload_signature_gate_available": workload_signature_gate.get("all_available"),
        "workload_signature_pair_count": len(signature_pairs) if isinstance(signature_pairs, dict) else None,
        "workload_signature_failed_pairs": (
            [
                label
                for label, pair in signature_pairs.items()
                if isinstance(pair, dict) and not bool(pair.get("gate_pass", False))
            ]
            if isinstance(signature_pairs, dict)
            else None
        ),
        "train_eval_anchor_family_l1_distance": signature_train_metrics.get("anchor_family_l1_distance"),
        "train_eval_footprint_family_l1_distance": signature_train_metrics.get("footprint_family_l1_distance"),
        "train_eval_point_hit_distribution_ks": point_hit_signature_distance,
        "train_eval_ship_hit_distribution_ks": ship_hit_signature_distance,
        "train_eval_point_hit_distribution_used_quantile_proxy": signature_train_metrics.get(
            "point_hit_distribution_used_quantile_proxy"
        ),
        "train_eval_ship_hit_distribution_used_quantile_proxy": signature_train_metrics.get(
            "ship_hit_distribution_used_quantile_proxy"
        ),
        "train_eval_point_hit_distribution_ks_proxy": point_hit_signature_distance,
        "train_eval_ship_hit_distribution_ks_proxy": ship_hit_signature_distance,
        "learning_causality_ablation_status": learning_causality.get("learning_causality_ablation_status"),
        "learning_causality_gate_pass": learning_causality.get("learning_causality_gate_pass"),
        "learning_causality_failed_checks": learning_causality.get("learning_causality_failed_checks"),
        "causality_ablation_missing": learning_causality.get("causality_ablation_missing"),
        "learned_controlled_retained_slot_fraction": learning_causality.get(
            "learned_controlled_retained_slot_fraction"
        ),
        "planned_learned_controlled_retained_slot_fraction": learning_causality.get(
            "planned_learned_controlled_retained_slot_fraction"
        ),
        "actual_learned_controlled_retained_slot_fraction": learning_causality.get(
            "actual_learned_controlled_retained_slot_fraction"
        ),
        "trajectories_with_at_least_one_learned_decision": learning_causality.get(
            "trajectories_with_at_least_one_learned_decision"
        ),
        "trajectories_with_zero_learned_decisions": learning_causality.get(
            "trajectories_with_zero_learned_decisions"
        ),
        "segment_budget_entropy": learning_causality.get("segment_budget_entropy"),
        "segment_budget_entropy_normalized": learning_causality.get("segment_budget_entropy_normalized"),
        "selector_trace_retained_mask_matches_primary": learning_causality.get(
            "selector_trace_retained_mask_matches_primary"
        ),
        "shuffled_score_ablation_delta": learning_causality.get("shuffled_score_ablation_delta"),
        "untrained_score_ablation_delta": learning_causality.get("untrained_score_ablation_delta"),
        "shuffled_prior_field_ablation_delta": learning_causality.get("shuffled_prior_field_ablation_delta"),
        "prior_field_only_score_ablation_delta": learning_causality.get("prior_field_only_score_ablation_delta"),
        "no_query_prior_field_ablation_delta": learning_causality.get("no_query_prior_field_ablation_delta"),
        "no_behavior_head_ablation_delta": learning_causality.get("no_behavior_head_ablation_delta"),
        "no_segment_budget_head_ablation_delta": learning_causality.get("no_segment_budget_head_ablation_delta"),
        "learning_causality_min_material_delta": learning_delta_gate.get("min_material_query_useful_delta"),
        "learning_causality_shuffled_fraction_of_uniform_gap_min": learning_delta_gate.get(
            "shuffled_score_delta_fraction_of_uniform_gap_min"
        ),
        "learning_causality_mlqds_uniform_gap": learning_delta_gate.get("mlqds_uniform_query_useful_gap"),
        "learning_causality_delta_thresholds": learning_delta_gate.get("thresholds"),
        "segment_budget_head_ablation_mode": learning_causality.get("segment_budget_head_ablation_mode"),
        "prior_sample_gate_pass": learning_causality.get("prior_sample_gate_pass"),
        "prior_sample_gate_failures": learning_causality.get("prior_sample_gate_failures"),
        "shuffled_prior_sampled_inputs_changed": shuffled_prior_sample.get("sampled_inputs_changed"),
        "shuffled_prior_sampled_primary_nonzero_fraction": shuffled_prior_sample.get("primary_nonzero_fraction"),
        "shuffled_prior_sampled_ablation_nonzero_fraction": shuffled_prior_sample.get("ablation_nonzero_fraction"),
        "shuffled_prior_sampled_mean_abs_feature_delta": shuffled_prior_sample.get("mean_abs_feature_delta"),
        "shuffled_prior_sampled_max_abs_feature_delta": shuffled_prior_sample.get("max_abs_feature_delta"),
        "shuffled_prior_sampled_outside_extent_fraction": shuffled_prior_sample.get(
            "points_outside_prior_extent_fraction"
        ),
        "no_prior_sampled_primary_nonzero_fraction": no_prior_sample.get("primary_nonzero_fraction"),
        "no_prior_sampled_mean_abs_feature_delta": no_prior_sample.get("mean_abs_feature_delta"),
        "no_prior_sampled_outside_extent_fraction": no_prior_sample.get("points_outside_prior_extent_fraction"),
        "legacy_range_useful_diagnostic_only": bool(legacy_range_useful_summary.get("diagnostic_only", True)),
        **selector_claim_evidence,
        "workload_blind_candidate": is_workload_blind_model_type(model_config.get("model_type")),
        "workload_blind_protocol_enabled": workload_blind_protocol.get("enabled"),
        "primary_masks_frozen_before_eval_query_scoring": workload_blind_protocol.get(
            "primary_masks_frozen_before_eval_query_scoring"
        ),
        "audit_masks_frozen_before_eval_query_scoring": workload_blind_protocol.get(
            "audit_masks_frozen_before_eval_query_scoring"
        ),
        "eval_geometry_blend_allowed": workload_blind_protocol.get("eval_geometry_blend_allowed"),
        "beats_uniform_range_usefulness": beats_uniform_range_usefulness,
        "beats_douglas_peucker_range_usefulness": beats_dp_range_usefulness,
        "beats_temporal_random_fill_range_usefulness": beats_temporal_random_fill_range_usefulness,
        **audit,
        **runtime_bottleneck,
        "mlqds_primary_metric": mlqds_primary_metric,
        "mlqds_primary_score": mlqds_primary_score,
        "mlqds_aggregate_f1": mlqds_aggregate_f1,
        "mlqds_range_point_f1": mlqds_range_point_f1,
        "mlqds_range_usefulness": mlqds_range_usefulness,
        "mlqds_range_usefulness_score": mlqds_range_usefulness,
        "mlqds_query_useful_v1_score": mlqds_query_useful_v1,
        "mlqds_range_usefulness_gap_time_score": mlqds_gap_time_usefulness,
        "mlqds_range_usefulness_gap_distance_score": mlqds_gap_distance_usefulness,
        "mlqds_range_usefulness_gap_min_score": mlqds_gap_min_usefulness,
        "mlqds_type_f1": (mlqds.get("per_type_f1") or {}).get(workload),
        "mlqds_range_ship_f1": mlqds.get("range_ship_f1"),
        "mlqds_range_ship_coverage": mlqds.get("range_ship_coverage"),
        "mlqds_range_entry_exit_f1": mlqds.get("range_entry_exit_f1"),
        "mlqds_range_crossing_f1": mlqds.get("range_crossing_f1"),
        "mlqds_range_temporal_coverage": mlqds.get("range_temporal_coverage"),
        "mlqds_range_gap_coverage": mlqds.get("range_gap_coverage"),
        "mlqds_range_gap_time_coverage": mlqds.get("range_gap_time_coverage"),
        "mlqds_range_gap_distance_coverage": mlqds.get("range_gap_distance_coverage"),
        "mlqds_range_gap_min_coverage": mlqds.get("range_gap_min_coverage"),
        "mlqds_range_turn_coverage": mlqds.get("range_turn_coverage"),
        "mlqds_range_shape_score": mlqds.get("range_shape_score"),
        **_geometry_fields("mlqds", mlqds),
        "range_usefulness_schema_version": mlqds.get("range_usefulness_schema_version"),
        "range_usefulness_gap_ablation_version": mlqds.get("range_usefulness_gap_ablation_version"),
        "final_metrics_mode": (run_json or {}).get("final_metrics_mode", baseline_config.get("final_metrics_mode")),
        "uniform_aggregate_f1": uniform_aggregate_f1,
        "uniform_range_point_f1": uniform_range_point_f1,
        "uniform_range_usefulness": uniform_range_usefulness,
        "uniform_range_usefulness_score": uniform_range_usefulness,
        "uniform_query_useful_v1_score": uniform_query_useful_v1,
        "uniform_range_usefulness_gap_time_score": uniform_gap_time_usefulness,
        "uniform_range_usefulness_gap_distance_score": uniform_gap_distance_usefulness,
        "uniform_range_usefulness_gap_min_score": uniform_gap_min_usefulness,
        "uniform_range_ship_f1": uniform.get("range_ship_f1"),
        "uniform_range_ship_coverage": uniform.get("range_ship_coverage"),
        "uniform_range_entry_exit_f1": uniform.get("range_entry_exit_f1"),
        "uniform_range_crossing_f1": uniform.get("range_crossing_f1"),
        "uniform_range_temporal_coverage": uniform.get("range_temporal_coverage"),
        "uniform_range_gap_coverage": uniform.get("range_gap_coverage"),
        "uniform_range_turn_coverage": uniform.get("range_turn_coverage"),
        "uniform_range_shape_score": uniform.get("range_shape_score"),
        **_geometry_fields("uniform", uniform),
        "douglas_peucker_aggregate_f1": dp_aggregate_f1,
        "douglas_peucker_range_point_f1": dp_range_point_f1,
        "douglas_peucker_range_usefulness": dp_range_usefulness,
        "douglas_peucker_range_usefulness_score": dp_range_usefulness,
        "douglas_peucker_query_useful_v1_score": dp_query_useful_v1,
        "douglas_peucker_range_usefulness_gap_time_score": dp_gap_time_usefulness,
        "douglas_peucker_range_usefulness_gap_distance_score": dp_gap_distance_usefulness,
        "douglas_peucker_range_usefulness_gap_min_score": dp_gap_min_usefulness,
        "douglas_peucker_range_ship_f1": dp.get("range_ship_f1"),
        "douglas_peucker_range_ship_coverage": dp.get("range_ship_coverage"),
        "douglas_peucker_range_entry_exit_f1": dp.get("range_entry_exit_f1"),
        "douglas_peucker_range_crossing_f1": dp.get("range_crossing_f1"),
        "douglas_peucker_range_temporal_coverage": dp.get("range_temporal_coverage"),
        "douglas_peucker_range_gap_coverage": dp.get("range_gap_coverage"),
        "douglas_peucker_range_turn_coverage": dp.get("range_turn_coverage"),
        "douglas_peucker_range_shape_score": dp.get("range_shape_score"),
        **_geometry_fields("douglas_peucker", dp),
        "mlqds_vs_uniform_range_point_f1": (
            float(mlqds_range_point_f1) - float(uniform_range_point_f1)
            if mlqds_range_point_f1 is not None and uniform_range_point_f1 is not None
            else None
        ),
        "mlqds_vs_douglas_peucker_range_point_f1": (
            float(mlqds_range_point_f1) - float(dp_range_point_f1)
            if mlqds_range_point_f1 is not None and dp_range_point_f1 is not None
            else None
        ),
        "mlqds_vs_uniform_range_usefulness": (
            float(mlqds_range_usefulness) - float(uniform_range_usefulness)
            if mlqds_range_usefulness is not None and uniform_range_usefulness is not None
            else None
        ),
        "mlqds_vs_uniform_query_useful_v1": (
            float(mlqds_query_useful_v1) - float(uniform_query_useful_v1)
            if mlqds_query_useful_v1 is not None and uniform_query_useful_v1 is not None
            else None
        ),
        "mlqds_vs_douglas_peucker_range_usefulness": (
            float(mlqds_range_usefulness) - float(dp_range_usefulness)
            if mlqds_range_usefulness is not None and dp_range_usefulness is not None
            else None
        ),
        "mlqds_vs_douglas_peucker_query_useful_v1": (
            float(mlqds_query_useful_v1) - float(dp_query_useful_v1)
            if mlqds_query_useful_v1 is not None and dp_query_useful_v1 is not None
            else None
        ),
        "mlqds_vs_uniform_range_usefulness_gap_time": (
            float(mlqds_gap_time_usefulness) - float(uniform_gap_time_usefulness)
            if mlqds_gap_time_usefulness is not None and uniform_gap_time_usefulness is not None
            else None
        ),
        "mlqds_vs_uniform_range_usefulness_gap_distance": (
            float(mlqds_gap_distance_usefulness) - float(uniform_gap_distance_usefulness)
            if mlqds_gap_distance_usefulness is not None and uniform_gap_distance_usefulness is not None
            else None
        ),
        "mlqds_vs_uniform_range_usefulness_gap_min": (
            float(mlqds_gap_min_usefulness) - float(uniform_gap_min_usefulness)
            if mlqds_gap_min_usefulness is not None and uniform_gap_min_usefulness is not None
            else None
        ),
        "mlqds_vs_douglas_peucker_range_usefulness_gap_time": (
            float(mlqds_gap_time_usefulness) - float(dp_gap_time_usefulness)
            if mlqds_gap_time_usefulness is not None and dp_gap_time_usefulness is not None
            else None
        ),
        "mlqds_vs_douglas_peucker_range_usefulness_gap_distance": (
            float(mlqds_gap_distance_usefulness) - float(dp_gap_distance_usefulness)
            if mlqds_gap_distance_usefulness is not None and dp_gap_distance_usefulness is not None
            else None
        ),
        "mlqds_vs_douglas_peucker_range_usefulness_gap_min": (
            float(mlqds_gap_min_usefulness) - float(dp_gap_min_usefulness)
            if mlqds_gap_min_usefulness is not None and dp_gap_min_usefulness is not None
            else None
        ),
        **component_deltas,
        **worst_component_delta,
        "mlqds_vs_uniform_avg_sed_km": _metric_delta(
            {"value": (mlqds.get("geometric_distortion") or {}).get("avg_sed_km")},
            {"value": (uniform.get("geometric_distortion") or {}).get("avg_sed_km")},
            "value",
        ),
        "mlqds_vs_uniform_avg_ped_km": _metric_delta(
            {"value": (mlqds.get("geometric_distortion") or {}).get("avg_ped_km")},
            {"value": (uniform.get("geometric_distortion") or {}).get("avg_ped_km")},
            "value",
        ),
        "mlqds_vs_uniform_avg_length_preserved": _metric_delta(
            mlqds,
            uniform,
            "avg_length_preserved",
        ),
        "mlqds_latency_ms": mlqds.get("latency_ms"),
        "avg_length_preserved": mlqds.get("avg_length_preserved"),
        "combined_query_shape_score": mlqds.get("combined_query_shape_score"),
        "temporal_random_fill_range_point_f1": temporal_random_fill.get("range_point_f1"),
        "temporal_random_fill_range_usefulness_score": random_fill_range_usefulness,
        "temporal_oracle_fill_range_point_f1": temporal_oracle_fill.get("range_point_f1"),
        "temporal_oracle_fill_range_usefulness_score": oracle_fill_range_usefulness,
        "mlqds_vs_temporal_random_fill_range_usefulness": (
            float(mlqds_range_usefulness) - float(random_fill_range_usefulness)
            if mlqds_range_usefulness is not None and random_fill_range_usefulness is not None
            else None
        ),
        "temporal_oracle_fill_gap_range_usefulness": (
            float(oracle_fill_range_usefulness) - float(mlqds_range_usefulness)
            if mlqds_range_usefulness is not None and oracle_fill_range_usefulness is not None
            else None
        ),
        "collapse_warning": collapse_summary["collapse_warning_any"],
        "collapse_warning_any": collapse_summary["collapse_warning_any"],
        "collapse_warning_count": collapse_summary["collapse_warning_count"],
        "best_epoch_collapse_warning": collapse_summary["best_epoch_collapse_warning"],
        "min_pred_std": collapse_summary["min_pred_std"],
        "best_epoch_pred_std": collapse_summary["best_epoch_pred_std"],
        "model_type": model_config.get("model_type"),
        **{
            f"model_metadata_{key}": value
            for key, value in model_type_metadata(str(model_config.get("model_type", ""))).items()
        },
        "historical_prior_k": model_config.get("historical_prior_k"),
        "historical_prior_clock_weight": model_config.get("historical_prior_clock_weight"),
        "historical_prior_mmsi_weight": model_config.get("historical_prior_mmsi_weight"),
        "historical_prior_density_weight": model_config.get("historical_prior_density_weight"),
        "historical_prior_min_target": model_config.get("historical_prior_min_target"),
        "historical_prior_support_ratio": model_config.get("historical_prior_support_ratio"),
        "historical_prior_source_aggregation": model_config.get("historical_prior_source_aggregation"),
        "historical_prior_source_count": target_diagnostics.get("historical_prior_source_count"),
        "historical_prior_stored_support_count": target_diagnostics.get("historical_prior_stored_support_count"),
        "checkpoint_score_variant": model_config.get("checkpoint_score_variant"),
        "compression_ratio": model_config.get("compression_ratio"),
        "n_queries": query_config.get("n_queries"),
        "max_queries": query_config.get("max_queries"),
        "query_target_coverage": query_config.get("target_coverage"),
        "range_spatial_km": query_config.get("range_spatial_km"),
        "range_time_hours": query_config.get("range_time_hours"),
        "loss_objective": model_config.get("loss_objective"),
        "budget_loss_ratios": model_config.get("budget_loss_ratios"),
        "budget_loss_temperature": model_config.get("budget_loss_temperature"),
        "temporal_distribution_loss_weight": model_config.get("temporal_distribution_loss_weight"),
        "range_train_workload_replicates": query_config.get("range_train_workload_replicates"),
        "validation_split_mode": data_config.get("validation_split_mode"),
        "val_fraction": data_config.get("val_fraction"),
        "eval_selector_matched_learned_slot_fraction": selector_budget_row.get(
            "learned_slot_fraction_of_budget"
        ),
        "eval_selector_matched_zero_learned_trajectory_fraction": selector_budget_row.get(
            "zero_learned_slot_trajectory_fraction"
        ),
        "eval_selector_matched_endpoint_only_trajectory_fraction": selector_budget_row.get(
            "endpoint_only_trajectory_fraction"
        ),
        **selector_low_budget_summary,
        "range_time_domain_mode": query_config.get("range_time_domain_mode"),
        "range_anchor_mode": query_config.get("range_anchor_mode"),
        "range_train_anchor_modes": query_config.get("range_train_anchor_modes"),
        "range_train_footprints": query_config.get("range_train_footprints"),
        "range_max_coverage_overshoot": query_config.get("range_max_coverage_overshoot"),
        "coverage_calibration_mode": query_config.get("coverage_calibration_mode"),
        **_workload_generation_fields(run_json, "train"),
        **_workload_generation_fields(run_json, "eval"),
        **_workload_generation_fields(run_json, "selection"),
        "checkpoint_full_score_every": model_config.get("checkpoint_full_score_every"),
        "checkpoint_candidate_pool_size": model_config.get("checkpoint_candidate_pool_size"),
        "mlqds_temporal_fraction": model_config.get("mlqds_temporal_fraction"),
        "mlqds_diversity_bonus": model_config.get("mlqds_diversity_bonus"),
        "mlqds_effective_diversity_bonus": _effective_diversity_bonus(model_config),
        "mlqds_hybrid_mode": model_config.get("mlqds_hybrid_mode"),
        "mlqds_stratified_center_weight": model_config.get("mlqds_stratified_center_weight"),
        "mlqds_min_learned_swaps": model_config.get("mlqds_min_learned_swaps"),
        "mlqds_score_mode": model_config.get("mlqds_score_mode"),
        "mlqds_score_temperature": model_config.get("mlqds_score_temperature"),
        "mlqds_rank_confidence_weight": model_config.get("mlqds_rank_confidence_weight"),
        "mlqds_range_geometry_blend": model_config.get("mlqds_range_geometry_blend"),
        "temporal_residual_label_mode": model_config.get("temporal_residual_label_mode"),
        "range_label_mode": model_config.get("range_label_mode"),
        "range_training_target_mode": model_config.get("range_training_target_mode"),
        "range_target_balance_mode": model_config.get("range_target_balance_mode"),
        "range_replicate_target_aggregation": model_config.get("range_replicate_target_aggregation"),
        "range_component_target_blend": model_config.get("range_component_target_blend"),
        "range_temporal_target_blend": model_config.get("range_temporal_target_blend"),
        "range_structural_target_blend": model_config.get("range_structural_target_blend"),
        "range_structural_target_source_mode": model_config.get("range_structural_target_source_mode"),
        "range_target_budget_weight_power": model_config.get("range_target_budget_weight_power"),
        "range_marginal_target_radius_scale": model_config.get("range_marginal_target_radius_scale"),
        "range_query_spine_fraction": model_config.get("range_query_spine_fraction"),
        "range_query_spine_mass_mode": model_config.get("range_query_spine_mass_mode"),
        "range_query_residual_multiplier": model_config.get("range_query_residual_multiplier"),
        "range_query_residual_mass_mode": model_config.get("range_query_residual_mass_mode"),
        "range_set_utility_multiplier": model_config.get("range_set_utility_multiplier"),
        "range_set_utility_candidate_limit": model_config.get("range_set_utility_candidate_limit"),
        "range_set_utility_mass_mode": model_config.get("range_set_utility_mass_mode"),
        "local_swap_utility_scored_candidate_count": target_transform.get(
            "local_swap_utility_scored_candidate_count"
        ),
        "local_swap_utility_positive_gain_candidate_count": target_transform.get(
            "local_swap_utility_positive_gain_candidate_count"
        ),
        "local_swap_utility_selected_count": target_transform.get("local_swap_utility_selected_count"),
        "local_swap_utility_selected_gain_mass": target_transform.get(
            "local_swap_utility_selected_gain_mass"
        ),
        "local_swap_utility_source_positive_mass": target_transform.get(
            "local_swap_utility_source_positive_mass"
        ),
        "local_swap_gain_cost_scored_candidate_count": target_transform.get(
            "local_swap_gain_cost_scored_candidate_count"
        ),
        "local_swap_gain_cost_positive_net_gain_count": target_transform.get(
            "local_swap_gain_cost_positive_net_gain_count"
        ),
        "local_swap_gain_cost_selected_count": target_transform.get("local_swap_gain_cost_selected_count"),
        "local_swap_gain_cost_selected_candidate_value_mass": target_transform.get(
            "local_swap_gain_cost_selected_candidate_value_mass"
        ),
        "local_swap_gain_cost_selected_removal_cost_mass": target_transform.get(
            "local_swap_gain_cost_selected_removal_cost_mass"
        ),
        "local_swap_gain_cost_source_positive_mass": target_transform.get(
            "local_swap_gain_cost_source_positive_mass"
        ),
        "range_boundary_prior_weight": model_config.get("range_boundary_prior_weight"),
        "range_boundary_prior_enabled": bool(float(model_config.get("range_boundary_prior_weight") or 0.0) > 0.0),
        "range_teacher_distillation_mode": model_config.get("range_teacher_distillation_mode"),
        "range_teacher_epochs": model_config.get("range_teacher_epochs"),
        "teacher_distillation_enabled": teacher_distillation.get("enabled"),
        "teacher_distillation_mode": teacher_distillation.get("mode"),
        "teacher_model_type": teacher_distillation.get("teacher_model_type"),
        "teacher_replicate_count": teacher_distillation.get("replicate_count"),
        "teacher_positive_label_count": teacher_distillation.get("positive_label_count"),
        "teacher_positive_label_fraction": teacher_distillation.get("positive_label_fraction"),
        "teacher_positive_label_mass": teacher_distillation.get("positive_label_mass"),
        "train_positive_label_mass": train_label_diagnostics.get("positive_label_mass"),
        "train_label_mass_basis": train_label_diagnostics.get("component_label_mass_basis"),
        "train_label_mass_range_point_f1": label_mass_fraction.get("range_point_f1"),
        "train_label_mass_range_ship_f1": label_mass_fraction.get("range_ship_f1"),
        "train_label_mass_range_ship_coverage": label_mass_fraction.get("range_ship_coverage"),
        "train_label_mass_range_entry_exit_f1": label_mass_fraction.get("range_entry_exit_f1"),
        "train_label_mass_range_crossing_f1": label_mass_fraction.get("range_crossing_f1"),
        "train_label_mass_range_temporal_coverage": label_mass_fraction.get("range_temporal_coverage"),
        "train_label_mass_range_gap_coverage": label_mass_fraction.get("range_gap_coverage"),
        "train_label_mass_range_turn_coverage": label_mass_fraction.get("range_turn_coverage"),
        "train_label_mass_range_shape_score": label_mass_fraction.get("range_shape_score"),
        "train_target_positive_label_mass": target_diagnostics.get("positive_label_mass"),
        "range_target_transform_mode": target_transform.get("mode"),
        "range_target_transform_target_family": target_transform.get("target_family"),
        "range_target_transform_final_success_allowed": target_transform.get("final_success_allowed"),
        "range_target_transform_positive_label_count": target_transform.get("positive_label_count"),
        "range_target_transform_positive_label_fraction": target_transform.get("positive_label_fraction"),
        "range_target_transform_positive_label_mass": target_transform.get("positive_label_mass"),
        "range_target_transform_base_positive_label_mass": target_transform.get(
            "base_retained_frequency_positive_label_mass"
        ),
        "range_structural_score_positive_mass": target_transform.get("structural_score_positive_mass"),
        "range_structural_score_p95": target_transform.get("structural_score_p95"),
        "historical_prior_teacher_score_p95": target_transform.get("historical_prior_teacher_score_p95"),
        "historical_prior_teacher_score_mass": target_transform.get("historical_prior_teacher_score_mass"),
        "historical_prior_teacher_positive_score_fraction": target_transform.get(
            "historical_prior_teacher_positive_score_fraction"
        ),
        "historical_prior_teacher_support_count": target_transform.get("historical_prior_stored_support_count"),
        "train_fit_score_target_kendall_tau": fit_diagnostics.get("score_target_kendall_tau"),
        "train_fit_model_fits_stored_train_support": fit_diagnostics.get("model_fits_stored_train_support"),
        "train_fit_matched_mlqds_target_recall": fit_diagnostics.get("matched_mlqds_target_recall"),
        "train_fit_matched_uniform_target_recall": fit_diagnostics.get("matched_uniform_target_recall"),
        "train_fit_matched_mlqds_vs_uniform_target_recall": fit_diagnostics.get(
            "matched_mlqds_vs_uniform_target_recall"
        ),
        "train_fit_low_budget_mean_mlqds_vs_uniform_target_recall": fit_diagnostics.get(
            "low_budget_mean_mlqds_vs_uniform_target_recall"
        ),
        "train_target_budget_ratio": target_budget_row.get("total_budget_ratio"),
        "train_target_effective_fill_budget_ratio": target_budget_row.get("effective_fill_budget_ratio"),
        "train_target_temporal_base_label_mass_fraction": target_budget_row.get(
            "temporal_base_label_mass_fraction"
        ),
        "train_target_residual_label_mass_fraction": target_budget_row.get("residual_label_mass_fraction"),
        "train_target_residual_positive_label_fraction": target_budget_row.get("residual_positive_label_fraction"),
        "oracle_kind": oracle_diagnostic.get("kind"),
        "oracle_exact_optimum": oracle_diagnostic.get("exact_optimum"),
        "float32_matmul_precision": model_config.get("float32_matmul_precision"),
        "allow_tf32": model_config.get("allow_tf32"),
        "amp_mode": model_config.get("amp_mode"),
        "extra_args": "",
        "child_float32_matmul_precision": child_torch_runtime.get("float32_matmul_precision"),
        "child_tf32_matmul_allowed": child_torch_runtime.get("tf32_matmul_allowed"),
        "child_tf32_cudnn_allowed": child_torch_runtime.get("tf32_cudnn_allowed"),
        "child_amp_enabled": child_amp.get("enabled"),
        "child_amp_dtype": child_amp.get("dtype"),
        "child_torch_runtime": child_torch_runtime or None,
        "train_batch_size": model_config.get("train_batch_size"),
        "inference_batch_size": model_config.get("inference_batch_size"),
        "run_dir": str(run_dir),
        "example_run_path": str(run_json_path) if run_json_path.exists() else None,
        "stdout_path": str(stdout_path),
        "command": command,
    }

def _format_value(value: Any) -> str:
    """Format values for a compact markdown table."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)

def _format_report_table(rows: list[dict[str, Any]]) -> str:
    """Return a compact markdown comparison table."""
    columns = [
        "workload",
        "run_label",
        "returncode",
        "elapsed_seconds",
        "epoch_mean_seconds",
        "peak_allocated_mb",
        "best_selection_score",
        "single_cell_range_status",
        "final_claim_status",
        "final_success_allowed",
        "predictability_gate_pass",
        "target_diffusion_gate_pass",
        "workload_signature_gate_pass",
        "workload_stability_gate_pass",
        "support_overlap_gate_pass",
        "learning_causality_gate_pass",
        "global_sanity_gate_pass",
        "legacy_range_useful_diagnostic_only",
        "selector_claim_status",
        "selector_claim_has_material_learned_budget",
        "selector_claim_min_learned_slot_fraction",
        "workload_blind_protocol_enabled",
        "primary_masks_frozen_before_eval_query_scoring",
        "audit_masks_frozen_before_eval_query_scoring",
        "beats_uniform_range_usefulness",
        "beats_douglas_peucker_range_usefulness",
        "beats_temporal_random_fill_range_usefulness",
        "audit_compression_ratio_count",
        "audit_missing_temporal_random_fill_count",
        "audit_beats_uniform_range_usefulness_count",
        "audit_beats_both_range_usefulness_count",
        "audit_beats_temporal_random_fill_range_usefulness_count",
        "audit_beats_uniform_query_useful_v1_count",
        "audit_beats_douglas_peucker_query_useful_v1_count",
        "audit_low_beats_uniform_query_useful_v1_count",
        "audit_low_compression_ratio_count",
        "audit_low_beats_uniform_range_usefulness_count",
        "audit_low_beats_both_range_usefulness_count",
        "audit_low_beats_temporal_random_fill_range_usefulness_count",
        "audit_beats_uniform_range_usefulness_gap_time_count",
        "audit_low_beats_uniform_range_usefulness_gap_time_count",
        "audit_beats_uniform_range_usefulness_gap_distance_count",
        "audit_low_beats_uniform_range_usefulness_gap_distance_count",
        "audit_beats_uniform_range_usefulness_gap_min_count",
        "audit_low_beats_uniform_range_usefulness_gap_min_count",
        "audit_min_low_vs_uniform_range_usefulness",
        "audit_mean_low_vs_uniform_range_usefulness",
        "audit_mean_low_vs_uniform_range_usefulness_gap_time",
        "audit_mean_low_vs_uniform_range_usefulness_gap_distance",
        "audit_mean_low_vs_uniform_range_usefulness_gap_min",
        "audit_min_vs_temporal_random_fill_range_usefulness",
        "audit_mean_vs_temporal_random_fill_range_usefulness",
        "audit_min_low_vs_temporal_random_fill_range_usefulness",
        "audit_mean_low_vs_temporal_random_fill_range_usefulness",
        "worst_uniform_component_delta_metric",
        "worst_uniform_component_delta",
        "runtime_bottleneck_phase",
        "runtime_bottleneck_seconds",
        "runtime_bottleneck_fraction",
        "epoch_loss_mean_seconds",
        "epoch_validation_score_mean_seconds",
        "compression_ratio",
        "n_queries",
        "max_queries",
        "query_target_coverage",
        "range_spatial_km",
        "range_time_hours",
        "loss_objective",
        "temporal_distribution_loss_weight",
        "range_train_workload_replicates",
        "validation_split_mode",
        "val_fraction",
        "eval_selector_matched_learned_slot_fraction",
        "eval_selector_matched_zero_learned_trajectory_fraction",
        "eval_selector_low_budget_zero_learned_ratio_count",
        "eval_selector_low_budget_min_learned_slot_fraction",
        "range_time_domain_mode",
        "range_anchor_mode",
        "range_train_anchor_modes",
        "range_train_footprints",
        "range_max_coverage_overshoot",
        "train_query_final_count",
        "train_query_final_coverage",
        "train_query_target_reached",
        "train_query_target_shortfall",
        "train_query_extra_after_target_reached",
        "eval_query_final_count",
        "eval_query_final_coverage",
        "eval_query_target_reached",
        "eval_query_target_shortfall",
        "eval_query_target_missed_by_max_queries",
        "eval_query_extra_after_target_reached",
        "eval_query_extra_after_target_fraction",
        "eval_query_floor_dominated",
        "eval_query_generation_stop_reason",
        "eval_workload_near_duplicate_query_rate",
        "selection_query_final_count",
        "selection_query_final_coverage",
        "selection_query_target_reached",
        "selection_query_target_shortfall",
        "selection_query_extra_after_target_reached",
        "model_type",
        "model_metadata_model_family",
        "model_metadata_trainable_final_candidate",
        "model_metadata_requires_ablation_against_standalone_knn",
        "model_metadata_final_success_allowed",
        "historical_prior_k",
        "historical_prior_clock_weight",
        "historical_prior_mmsi_weight",
        "historical_prior_density_weight",
        "historical_prior_min_target",
        "historical_prior_support_ratio",
        "historical_prior_source_aggregation",
        "historical_prior_source_count",
        "historical_prior_stored_support_count",
        "temporal_residual_label_mode",
        "range_training_target_mode",
        "range_target_balance_mode",
        "range_replicate_target_aggregation",
        "range_component_target_blend",
        "range_temporal_target_blend",
        "range_structural_target_blend",
        "range_structural_target_source_mode",
        "range_target_budget_weight_power",
        "range_marginal_target_radius_scale",
        "range_query_spine_fraction",
        "range_query_spine_mass_mode",
        "range_query_residual_multiplier",
        "range_query_residual_mass_mode",
        "range_set_utility_multiplier",
        "range_set_utility_candidate_limit",
        "range_set_utility_mass_mode",
        "local_swap_utility_scored_candidate_count",
        "local_swap_utility_positive_gain_candidate_count",
        "local_swap_utility_selected_count",
        "local_swap_utility_source_positive_mass",
        "local_swap_gain_cost_scored_candidate_count",
        "local_swap_gain_cost_positive_net_gain_count",
        "local_swap_gain_cost_selected_count",
        "local_swap_gain_cost_source_positive_mass",
        "range_teacher_distillation_mode",
        "mlqds_diversity_bonus",
        "mlqds_effective_diversity_bonus",
        "mlqds_hybrid_mode",
        "mlqds_stratified_center_weight",
        "mlqds_min_learned_swaps",
        "mlqds_range_geometry_blend",
        "train_label_mass_range_point_f1",
        "train_label_mass_range_ship_coverage",
        "train_label_mass_range_crossing_f1",
        "train_label_mass_range_temporal_coverage",
        "train_label_mass_range_gap_coverage",
        "train_label_mass_range_turn_coverage",
        "train_label_mass_range_shape_score",
        "range_target_transform_mode",
        "range_target_transform_target_family",
        "range_target_transform_final_success_allowed",
        "range_target_transform_positive_label_fraction",
        "range_target_transform_positive_label_mass",
        "range_target_transform_base_positive_label_mass",
        "range_structural_score_positive_mass",
        "range_structural_score_p95",
        "historical_prior_teacher_score_p95",
        "historical_prior_teacher_positive_score_fraction",
        "train_target_residual_label_mass_fraction",
        "train_fit_score_target_kendall_tau",
        "train_fit_model_fits_stored_train_support",
        "train_fit_matched_mlqds_vs_uniform_target_recall",
        "train_fit_low_budget_mean_mlqds_vs_uniform_target_recall",
        "train_target_effective_fill_budget_ratio",
        "checkpoint_full_score_every",
        "checkpoint_candidate_pool_size",
        "final_metrics_mode",
        "mlqds_primary_metric",
        "mlqds_primary_score",
        "mlqds_aggregate_f1",
        "mlqds_range_point_f1",
        "mlqds_range_usefulness",
        "mlqds_range_usefulness_gap_time_score",
        "mlqds_range_usefulness_gap_distance_score",
        "mlqds_range_usefulness_gap_min_score",
        "range_usefulness_schema_version",
        "range_usefulness_gap_ablation_version",
        "uniform_range_point_f1",
        "uniform_range_usefulness",
        "uniform_range_usefulness_gap_time_score",
        "uniform_range_usefulness_gap_distance_score",
        "uniform_range_usefulness_gap_min_score",
        "douglas_peucker_range_point_f1",
        "douglas_peucker_range_usefulness",
        "douglas_peucker_range_usefulness_gap_time_score",
        "douglas_peucker_range_usefulness_gap_distance_score",
        "douglas_peucker_range_usefulness_gap_min_score",
        "mlqds_vs_uniform_range_point_f1",
        "mlqds_vs_uniform_range_usefulness",
        "mlqds_vs_douglas_peucker_range_point_f1",
        "mlqds_vs_douglas_peucker_range_usefulness",
        "mlqds_vs_uniform_range_usefulness_gap_time",
        "mlqds_vs_uniform_range_usefulness_gap_distance",
        "mlqds_vs_uniform_range_usefulness_gap_min",
        "temporal_random_fill_range_usefulness_score",
        "mlqds_vs_temporal_random_fill_range_usefulness",
        "mlqds_avg_sed_km",
        "uniform_avg_sed_km",
        "douglas_peucker_avg_sed_km",
        "mlqds_vs_uniform_avg_sed_km",
        "mlqds_avg_length_preserved",
        "uniform_avg_length_preserved",
        "douglas_peucker_avg_length_preserved",
        "mlqds_vs_uniform_avg_length_preserved",
        "mlqds_range_ship_coverage",
        "mlqds_range_entry_exit_f1",
        "mlqds_range_crossing_f1",
        "mlqds_range_gap_coverage",
        "mlqds_range_gap_time_coverage",
        "mlqds_range_gap_distance_coverage",
        "mlqds_range_gap_min_coverage",
        "mlqds_range_turn_coverage",
        "mlqds_latency_ms",
        "best_epoch_collapse_warning",
        "collapse_warning_count",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_value(row.get(column)) for column in columns) + " |")
    return "\n".join(lines) + "\n"
