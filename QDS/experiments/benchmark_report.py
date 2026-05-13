"""Benchmark row shaping and report table helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _child_run_dir(results_dir: Path, workload: str, run_label: str, workload_count: int) -> Path:
    """Return the child experiment output directory for a benchmark row."""
    if workload_count == 1:
        return results_dir / run_label
    return results_dir / workload / run_label

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

def _has_collapse_warning(run_json: dict[str, Any] | None) -> bool | None:
    """Return whether training history contains a collapse warning."""
    if not run_json:
        return None
    return bool(_collapse_warning_summary(run_json)["collapse_warning_any"])

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
    model_config = (run_json or {}).get("config", {}).get("model", {})
    baseline_config = (run_json or {}).get("config", {}).get("baselines", {})
    oracle_diagnostic = (run_json or {}).get("oracle_diagnostic") or {}
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
    target_budget_row = _target_budget_row(target_diagnostics, model_config.get("compression_ratio"))
    mlqds_aggregate_f1 = mlqds.get("aggregate_f1")
    mlqds_range_point_f1 = mlqds.get("range_point_f1", mlqds_aggregate_f1)
    mlqds_range_usefulness = mlqds.get("range_usefulness_score")
    mlqds_primary_score = mlqds_range_usefulness if mlqds_range_usefulness is not None else mlqds_range_point_f1
    mlqds_primary_metric = "range_usefulness" if mlqds_range_usefulness is not None else "range_point_f1"
    random_fill_range_usefulness = temporal_random_fill.get("range_usefulness_score")
    oracle_fill_range_usefulness = temporal_oracle_fill.get("range_usefulness_score")
    uniform_aggregate_f1 = uniform.get("aggregate_f1")
    uniform_range_point_f1 = uniform.get("range_point_f1", uniform_aggregate_f1)
    uniform_range_usefulness = uniform.get("range_usefulness_score")
    dp_aggregate_f1 = dp.get("aggregate_f1")
    dp_range_point_f1 = dp.get("range_point_f1", dp_aggregate_f1)
    dp_range_usefulness = dp.get("range_usefulness_score")
    return {
        "workload": workload,
        "run_label": run_label,
        "returncode": int(returncode),
        "elapsed_seconds": float(elapsed_seconds),
        "train_seconds": _phase_seconds_with_prefix(timings, "train-model"),
        "evaluate_matched_seconds": _phase_seconds(timings, "evaluate-matched"),
        "epoch_mean_seconds": _mean_epoch_seconds(timings),
        "peak_allocated_mb": cuda_memory.get("max_allocated_mb"),
        "best_epoch": (run_json or {}).get("best_epoch"),
        "best_loss": (run_json or {}).get("best_loss"),
        "best_selection_score": (run_json or {}).get("best_selection_score"),
        "mlqds_primary_metric": mlqds_primary_metric,
        "mlqds_primary_score": mlqds_primary_score,
        "mlqds_aggregate_f1": mlqds_aggregate_f1,
        "mlqds_range_point_f1": mlqds_range_point_f1,
        "mlqds_range_usefulness": mlqds_range_usefulness,
        "mlqds_range_usefulness_score": mlqds_range_usefulness,
        "mlqds_type_f1": (mlqds.get("per_type_f1") or {}).get(workload),
        "mlqds_range_ship_f1": mlqds.get("range_ship_f1"),
        "mlqds_range_ship_coverage": mlqds.get("range_ship_coverage"),
        "mlqds_range_entry_exit_f1": mlqds.get("range_entry_exit_f1"),
        "mlqds_range_crossing_f1": mlqds.get("range_crossing_f1"),
        "mlqds_range_temporal_coverage": mlqds.get("range_temporal_coverage"),
        "mlqds_range_gap_coverage": mlqds.get("range_gap_coverage"),
        "mlqds_range_turn_coverage": mlqds.get("range_turn_coverage"),
        "mlqds_range_shape_score": mlqds.get("range_shape_score"),
        "range_usefulness_schema_version": mlqds.get("range_usefulness_schema_version"),
        "final_metrics_mode": (run_json or {}).get("final_metrics_mode", baseline_config.get("final_metrics_mode")),
        "uniform_aggregate_f1": uniform_aggregate_f1,
        "uniform_range_point_f1": uniform_range_point_f1,
        "uniform_range_usefulness": uniform_range_usefulness,
        "uniform_range_usefulness_score": uniform_range_usefulness,
        "douglas_peucker_aggregate_f1": dp_aggregate_f1,
        "douglas_peucker_range_point_f1": dp_range_point_f1,
        "douglas_peucker_range_usefulness": dp_range_usefulness,
        "douglas_peucker_range_usefulness_score": dp_range_usefulness,
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
        "mlqds_vs_douglas_peucker_range_usefulness": (
            float(mlqds_range_usefulness) - float(dp_range_usefulness)
            if mlqds_range_usefulness is not None and dp_range_usefulness is not None
            else None
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
        "checkpoint_score_variant": model_config.get("checkpoint_score_variant"),
        "compression_ratio": model_config.get("compression_ratio"),
        "loss_objective": model_config.get("loss_objective"),
        "budget_loss_ratios": model_config.get("budget_loss_ratios"),
        "budget_loss_temperature": model_config.get("budget_loss_temperature"),
        "checkpoint_full_score_every": model_config.get("checkpoint_full_score_every"),
        "checkpoint_candidate_pool_size": model_config.get("checkpoint_candidate_pool_size"),
        "mlqds_temporal_fraction": model_config.get("mlqds_temporal_fraction"),
        "mlqds_diversity_bonus": model_config.get("mlqds_diversity_bonus"),
        "mlqds_hybrid_mode": model_config.get("mlqds_hybrid_mode"),
        "mlqds_score_mode": model_config.get("mlqds_score_mode"),
        "mlqds_score_temperature": model_config.get("mlqds_score_temperature"),
        "mlqds_rank_confidence_weight": model_config.get("mlqds_rank_confidence_weight"),
        "mlqds_range_geometry_blend": model_config.get("mlqds_range_geometry_blend"),
        "temporal_residual_label_mode": model_config.get("temporal_residual_label_mode"),
        "range_label_mode": model_config.get("range_label_mode"),
        "range_boundary_prior_weight": model_config.get("range_boundary_prior_weight"),
        "range_boundary_prior_enabled": bool(float(model_config.get("range_boundary_prior_weight") or 0.0) > 0.0),
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
        "compression_ratio",
        "loss_objective",
        "model_type",
        "temporal_residual_label_mode",
        "mlqds_diversity_bonus",
        "mlqds_hybrid_mode",
        "mlqds_range_geometry_blend",
        "train_label_mass_range_point_f1",
        "train_label_mass_range_ship_coverage",
        "train_label_mass_range_crossing_f1",
        "train_label_mass_range_temporal_coverage",
        "train_label_mass_range_gap_coverage",
        "train_label_mass_range_turn_coverage",
        "train_label_mass_range_shape_score",
        "train_target_residual_label_mass_fraction",
        "train_target_effective_fill_budget_ratio",
        "checkpoint_full_score_every",
        "checkpoint_candidate_pool_size",
        "final_metrics_mode",
        "mlqds_primary_metric",
        "mlqds_primary_score",
        "mlqds_aggregate_f1",
        "mlqds_range_point_f1",
        "mlqds_range_usefulness",
        "range_usefulness_schema_version",
        "uniform_range_point_f1",
        "uniform_range_usefulness",
        "douglas_peucker_range_point_f1",
        "douglas_peucker_range_usefulness",
        "mlqds_vs_uniform_range_point_f1",
        "mlqds_vs_uniform_range_usefulness",
        "mlqds_vs_douglas_peucker_range_point_f1",
        "mlqds_vs_douglas_peucker_range_usefulness",
        "temporal_random_fill_range_usefulness_score",
        "mlqds_vs_temporal_random_fill_range_usefulness",
        "mlqds_range_ship_coverage",
        "mlqds_range_entry_exit_f1",
        "mlqds_range_crossing_f1",
        "mlqds_range_gap_coverage",
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
