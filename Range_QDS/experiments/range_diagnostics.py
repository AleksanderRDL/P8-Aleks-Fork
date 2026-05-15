"""Range diagnostics and range-specific experiment payload helpers."""

from __future__ import annotations

from typing import Any, Callable

import torch

from evaluation.baselines import DouglasPeuckerMethod, Method, OracleMethod, UniformTemporalMethod
from evaluation.evaluate_methods import score_retained_mask
from evaluation.metrics import MethodEvaluation
from evaluation.query_cache import EvaluationQueryCache
from experiments.experiment_config import ExperimentConfig
from experiments.range_cache import (
    RangeRuntimeCache,
    ensure_range_runtime_labels,
    load_range_diagnostics_cache,
    range_diagnostic_duplicate_threshold,
    range_diagnostics_cache_key,
    range_diagnostics_cache_payload,
    range_only_queries,
    runtime_evaluation_query_cache,
    write_range_diagnostics_cache,
)
from queries.workload import TypedQueryWorkload
from queries.workload_diagnostics import (
    compute_range_label_diagnostics,
    compute_range_workload_diagnostics,
    range_box_mask,
)


def _range_signal_diagnostics(
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    range_queries: list[dict[str, Any]],
    workload_map: dict[str, float],
    compression_ratio: float,
    seed: int,
    range_label_mode: str = "usefulness",
    range_boundary_prior_weight: float = 0.0,
    runtime_cache: RangeRuntimeCache | None = None,
    cache_typed_queries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute label, Oracle, and baseline diagnostics for range workloads."""
    if not range_queries:
        return {
            "range_query_count": 0,
            "range_label_mode": str(range_label_mode),
            "labels": compute_range_label_diagnostics(
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0, 4), dtype=torch.bool),
            ),
            "methods": {},
            "best_baseline": None,
            "best_baseline_range_f1": 0.0,
            "oracle_range_f1": 0.0,
            "oracle_gap_over_best_baseline": 0.0,
        }

    component_labels: dict[str, torch.Tensor] | None = None
    if runtime_cache is not None and runtime_cache.labels is not None and runtime_cache.labelled_mask is not None:
        labels = runtime_cache.labels
        labelled_mask = runtime_cache.labelled_mask
        component_labels = runtime_cache.component_labels
    else:
        labels, labelled_mask = ensure_range_runtime_labels(
            points=points,
            boundaries=boundaries,
            range_queries=range_queries,
            seed=seed,
            range_label_mode=range_label_mode,
            range_boundary_prior_weight=range_boundary_prior_weight,
            runtime_cache=runtime_cache,
        )
        component_labels = runtime_cache.component_labels if runtime_cache is not None else None
    label_diagnostics = compute_range_label_diagnostics(labels, labelled_mask, component_labels)
    oracle_labels = labels
    methods: list[Method] = [
        UniformTemporalMethod(),
        DouglasPeuckerMethod(),
        OracleMethod(labels=oracle_labels, workload_type="range"),
    ]
    method_scores: dict[str, dict[str, float]] = {}
    scored_queries = cache_typed_queries if cache_typed_queries is not None else range_queries
    query_cache = runtime_evaluation_query_cache(
        runtime_cache,
        points,
        boundaries,
        scored_queries,
    )
    for method in methods:
        retained_mask = method.simplify(points, boundaries, compression_ratio)
        aggregate, per_type, _, _ = score_retained_mask(
            points=points,
            boundaries=boundaries,
            retained_mask=retained_mask,
            typed_queries=scored_queries,
            workload_map={"range": 1.0},
            query_cache=query_cache,
        )
        method_scores[method.name] = {
            "aggregate_f1": float(aggregate),
            "range_f1": float(per_type.get("range", 0.0)),
        }

    baseline_names = ["uniform", "DouglasPeucker"]
    best_baseline = max(baseline_names, key=lambda name: method_scores.get(name, {}).get("range_f1", 0.0))
    best_baseline_range_f1 = float(method_scores[best_baseline]["range_f1"])
    oracle_range_f1 = float(method_scores.get("Oracle", {}).get("range_f1", 0.0))
    normalized_map = sum(float(v) for v in workload_map.values())
    range_weight = float(workload_map.get("range", 0.0)) / normalized_map if normalized_map > 0.0 else 0.0
    return {
        "range_query_count": int(len(range_queries)),
        "range_workload_weight": float(range_weight),
        "range_boundary_prior_weight": float(range_boundary_prior_weight),
        "range_boundary_prior_enabled": bool(float(range_boundary_prior_weight) > 0.0),
        "range_label_mode": str(range_label_mode),
        "oracle_label_mode": str(range_label_mode),
        "oracle_kind": "additive_label_greedy",
        "oracle_exact_optimum": False,
        "labels": label_diagnostics,
        "methods": method_scores,
        "best_baseline": best_baseline,
        "best_baseline_range_f1": best_baseline_range_f1,
        "oracle_range_f1": oracle_range_f1,
        "oracle_gap_over_best_baseline": float(oracle_range_f1 - best_baseline_range_f1),
    }

def _range_workload_diagnostics(
    label: str,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    workload: TypedQueryWorkload,
    workload_map: dict[str, float],
    config: ExperimentConfig,
    seed: int,
    runtime_cache: RangeRuntimeCache | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build summary and JSONL rows for one workload."""
    range_queries = range_only_queries(workload.typed_queries)
    is_pure_range_workload = len(range_queries) == len(workload.typed_queries)
    scored_queries = workload.typed_queries if is_pure_range_workload else range_queries
    query_cache: EvaluationQueryCache | None = None
    mask_provider: Callable[[int, dict[str, Any]], torch.Tensor] | None = None
    if is_pure_range_workload:
        query_cache = runtime_evaluation_query_cache(
            runtime_cache,
            points,
            boundaries,
            scored_queries,
        )
        if query_cache is None:
            raise RuntimeError("Pure range diagnostics expected a prepared query cache.")

        def _mask_provider(query_index: int, query: dict[str, Any]) -> torch.Tensor:
            return query_cache.get_support_mask(
                query_index,
                lambda query=query: range_box_mask(points, query["params"]),
            )

        mask_provider = _mask_provider

    cache_payload = range_diagnostics_cache_payload(
        points=points,
        boundaries=boundaries,
        workload=workload,
        workload_map=workload_map,
        config=config,
        seed=seed,
    )
    cache_key = range_diagnostics_cache_key(cache_payload)
    cached = load_range_diagnostics_cache(
        config=config,
        label=label,
        key=cache_key,
        points=points,
        boundaries=boundaries,
        scored_queries=scored_queries,
        runtime_cache=runtime_cache,
    )
    if cached is not None:
        return cached

    workload_diagnostics = compute_range_workload_diagnostics(
        points=points,
        boundaries=boundaries,
        typed_queries=workload.typed_queries,
        max_point_hit_fraction=config.query.range_max_point_hit_fraction,
        max_trajectory_hit_fraction=config.query.range_max_trajectory_hit_fraction,
        max_box_volume_fraction=config.query.range_max_box_volume_fraction,
        duplicate_iou_threshold=range_diagnostic_duplicate_threshold(config),
        coverage_fraction=workload.coverage_fraction,
        mask_provider=mask_provider,
    )
    signal = _range_signal_diagnostics(
        points=points,
        boundaries=boundaries,
        range_queries=range_queries,
        workload_map=workload_map,
        compression_ratio=config.model.compression_ratio,
        seed=seed,
        range_label_mode=str(getattr(config.model, "range_label_mode", "usefulness")),
        range_boundary_prior_weight=float(getattr(config.model, "range_boundary_prior_weight", 0.0)),
        runtime_cache=runtime_cache,
        cache_typed_queries=workload.typed_queries if is_pure_range_workload else None,
    )
    summary = {
        "range": workload_diagnostics["summary"],
        "range_signal": signal,
        "generation": workload.generation_diagnostics or {},
    }
    rows = [{"workload": label, **row} for row in workload_diagnostics["queries"]]
    write_range_diagnostics_cache(
        config=config,
        label=label,
        key=cache_key,
        summary=summary,
        rows=rows,
        runtime_cache=runtime_cache,
    )
    return summary, rows

def _print_range_diagnostics_summary(range_diagnostics_summary: dict[str, Any]) -> None:
    """Print compact range diagnostics for each split."""
    for label, summary in range_diagnostics_summary.items():
        range_summary = summary["range"]
        signal = summary["range_signal"]
        diagnostics_cache = summary.get("range_diagnostics_cache") or {}
        cache_text = "disabled"
        if isinstance(diagnostics_cache, dict) and diagnostics_cache:
            cache_text = "hit" if diagnostics_cache.get("hit") else "miss"
        print(
            f"  {label}: range_queries={range_summary['range_query_count']}  "
            f"empty={range_summary['empty_query_rate']:.2%}  "
            f"broad={range_summary['too_broad_query_rate']:.2%}  "
            f"duplicates={range_summary['near_duplicate_query_rate']:.2%}  "
            f"oracle_gap={signal['oracle_gap_over_best_baseline']:+.6f}  "
            f"diag_cache={cache_text}",
            flush=True,
        )

def _print_range_distribution_comparison(workload_distribution_comparison: dict[str, Any]) -> None:
    """Print compact train/selection-vs-eval workload distribution deltas."""
    for label, delta in workload_distribution_comparison["deltas_vs_eval"].items():
        coverage_delta = delta.get("coverage_fraction_minus_eval")
        point_p50_delta = delta.get("point_hit_count_p50_minus_eval")
        traj_p50_delta = delta.get("trajectory_hit_count_p50_minus_eval")
        coverage_text = f"{coverage_delta:+.4f}" if isinstance(coverage_delta, (int, float)) else "n/a"
        point_text = f"{point_p50_delta:+.2f}" if isinstance(point_p50_delta, (int, float)) else "n/a"
        traj_text = f"{traj_p50_delta:+.2f}" if isinstance(traj_p50_delta, (int, float)) else "n/a"
        print(
            f"  {label}_vs_eval: "
            f"coverage_delta={coverage_text}  "
            f"point_hit_p50_delta={point_text}  "
            f"trajectory_hit_p50_delta={traj_text}",
            flush=True,
        )

def _compact_range_workload_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Extract comparable workload-shape fields from verbose diagnostics."""
    range_summary = summary.get("range", {}) if isinstance(summary, dict) else {}
    range_signal = summary.get("range_signal", {}) if isinstance(summary, dict) else {}
    generation = summary.get("generation", {}) if isinstance(summary, dict) else {}
    workload_signature = generation.get("workload_signature", {}) if isinstance(generation, dict) else {}
    fields = (
        "range_query_count",
        "coverage_fraction",
        "empty_query_rate",
        "too_broad_query_rate",
        "near_duplicate_query_rate",
        "point_hit_count_p50",
        "point_hit_count_p90",
        "trajectory_hit_count_p50",
        "trajectory_hit_count_p90",
        "point_hit_fraction_p50",
        "point_hit_fraction_p90",
        "trajectory_hit_fraction_p50",
        "trajectory_hit_fraction_p90",
        "box_volume_fraction_p50",
        "box_volume_fraction_p90",
    )
    compact = {field: range_summary.get(field) for field in fields}
    compact["oracle_gap_over_best_baseline"] = range_signal.get("oracle_gap_over_best_baseline")
    compact["best_baseline"] = range_signal.get("best_baseline")
    compact["workload_signature"] = workload_signature if isinstance(workload_signature, dict) else {}
    return compact


def _normalized_counts(counts: object, keys: set[str]) -> dict[str, float]:
    """Return a normalized count map on the requested key universe."""
    if not isinstance(counts, dict):
        return {key: 0.0 for key in keys}
    total = sum(float(value) for value in counts.values() if isinstance(value, (int, float)))
    if total <= 0.0:
        return {key: 0.0 for key in keys}
    return {key: float(counts.get(key, 0.0)) / total for key in keys}


def _l1_count_distance(left: object, right: object) -> float | None:
    """Return L1 distance between two normalized count dictionaries."""
    if not isinstance(left, dict) or not isinstance(right, dict):
        return None
    keys = set(str(key) for key in left) | set(str(key) for key in right)
    if not keys:
        return None
    left_norm = _normalized_counts(left, keys)
    right_norm = _normalized_counts(right, keys)
    return float(sum(abs(left_norm[key] - right_norm[key]) for key in keys))


def _quantile_linf_distance(left: object, right: object) -> float | None:
    """Return a bounded quantile-distance proxy when raw distributions are unavailable."""
    if not isinstance(left, dict) or not isinstance(right, dict):
        return None
    keys = ("p10", "p50", "p90")
    distances: list[float] = []
    for key in keys:
        left_value = left.get(key)
        right_value = right.get(key)
        if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
            scale = max(abs(float(left_value)), abs(float(right_value)), 1.0)
            distances.append(abs(float(left_value) - float(right_value)) / scale)
    if not distances:
        return None
    return float(max(distances))


def _ks_distance(left: object, right: object) -> float | None:
    """Return two-sample Kolmogorov-Smirnov distance for persisted scalar lists."""
    if not isinstance(left, list) or not isinstance(right, list) or not left or not right:
        return None
    left_values = sorted(float(value) for value in left if isinstance(value, (int, float)))
    right_values = sorted(float(value) for value in right if isinstance(value, (int, float)))
    if not left_values or not right_values:
        return None
    values = sorted(set(left_values) | set(right_values))
    left_index = 0
    right_index = 0
    max_distance = 0.0
    left_count = float(len(left_values))
    right_count = float(len(right_values))
    for value in values:
        while left_index < len(left_values) and left_values[left_index] <= value:
            left_index += 1
        while right_index < len(right_values) and right_values[right_index] <= value:
            right_index += 1
        max_distance = max(max_distance, abs(left_index / left_count - right_index / right_count))
    return float(max_distance)


def _workload_signature_gate_for_pair(train_like: dict[str, Any], eval_like: dict[str, Any]) -> dict[str, Any]:
    """Compare profile signatures against guide defaults."""
    train_sig = train_like.get("workload_signature", {})
    eval_sig = eval_like.get("workload_signature", {})
    if not isinstance(train_sig, dict) or not isinstance(eval_sig, dict) or not train_sig or not eval_sig:
        return {
            "gate_available": False,
            "gate_pass": False,
            "reason": "missing_workload_signature",
        }
    thresholds = {
        "anchor_family_l1_distance_max": 0.12,
        "footprint_family_l1_distance_max": 0.12,
        "point_hit_distribution_ks_max": 0.20,
        "ship_hit_distribution_ks_max": 0.20,
        "near_duplicate_rate_max": 0.05,
        "broad_query_rate_max": 0.05,
    }
    point_hit_ks = _ks_distance(
        train_sig.get("point_hit_counts_per_query"),
        eval_sig.get("point_hit_counts_per_query"),
    )
    ship_hit_ks = _ks_distance(
        train_sig.get("ship_hit_counts_per_query"),
        eval_sig.get("ship_hit_counts_per_query"),
    )
    point_hit_proxy = _quantile_linf_distance(
        train_sig.get("point_hits_per_query"),
        eval_sig.get("point_hits_per_query"),
    )
    ship_hit_proxy = _quantile_linf_distance(
        train_sig.get("ship_hits_per_query"),
        eval_sig.get("ship_hits_per_query"),
    )
    metrics = {
        "anchor_family_l1_distance": _l1_count_distance(
            train_sig.get("anchor_family_counts"),
            eval_sig.get("anchor_family_counts"),
        ),
        "footprint_family_l1_distance": _l1_count_distance(
            train_sig.get("footprint_family_counts"),
            eval_sig.get("footprint_family_counts"),
        ),
        "point_hit_distribution_ks": point_hit_ks if point_hit_ks is not None else point_hit_proxy,
        "ship_hit_distribution_ks": ship_hit_ks if ship_hit_ks is not None else ship_hit_proxy,
        "point_hit_distribution_used_quantile_proxy": point_hit_ks is None,
        "ship_hit_distribution_used_quantile_proxy": ship_hit_ks is None,
        "near_duplicate_rate_max_observed": max(
            float(train_sig.get("near_duplicate_rate", 1.0)),
            float(eval_sig.get("near_duplicate_rate", 1.0)),
        ),
        "broad_query_rate_max_observed": max(
            float(train_sig.get("broad_query_rate", 1.0)),
            float(eval_sig.get("broad_query_rate", 1.0)),
        ),
    }

    checks = {
        "anchor_family_l1_distance": (
            metrics["anchor_family_l1_distance"],
            thresholds["anchor_family_l1_distance_max"],
        ),
        "footprint_family_l1_distance": (
            metrics["footprint_family_l1_distance"],
            thresholds["footprint_family_l1_distance_max"],
        ),
        "point_hit_distribution_ks": (
            metrics["point_hit_distribution_ks"],
            thresholds["point_hit_distribution_ks_max"],
        ),
        "ship_hit_distribution_ks": (
            metrics["ship_hit_distribution_ks"],
            thresholds["ship_hit_distribution_ks_max"],
        ),
        "near_duplicate_rate_max_observed": (
            metrics["near_duplicate_rate_max_observed"],
            thresholds["near_duplicate_rate_max"],
        ),
        "broad_query_rate_max_observed": (
            metrics["broad_query_rate_max_observed"],
            thresholds["broad_query_rate_max"],
        ),
    }
    failed = [
        name
        for name, (value, threshold) in checks.items()
        if not isinstance(value, (int, float)) or float(value) > float(threshold)
    ]
    return {
        "gate_available": True,
        "gate_pass": not failed,
        "failed_checks": failed,
        "thresholds": thresholds,
        "metrics": metrics,
        "profile_id_train": train_sig.get("profile_id"),
        "profile_id_eval": eval_sig.get("profile_id"),
        "distribution_metric_note": (
            "Point/ship hit distribution checks use persisted per-query hit-count KS distance when available; "
            "older signatures fall back to p10/p50/p90 quantile-distance proxies."
        ),
    }


def _range_workload_distribution_comparison(summaries: dict[str, Any]) -> dict[str, Any]:
    """Compare train/selection workload shape against final eval workload shape."""
    compact = {label: _compact_range_workload_summary(summary) for label, summary in summaries.items()}
    eval_summary = compact.get("eval", {})
    numeric_fields = (
        "range_query_count",
        "coverage_fraction",
        "empty_query_rate",
        "too_broad_query_rate",
        "near_duplicate_query_rate",
        "point_hit_count_p50",
        "trajectory_hit_count_p50",
        "point_hit_fraction_p50",
        "trajectory_hit_fraction_p50",
        "box_volume_fraction_p50",
        "oracle_gap_over_best_baseline",
    )
    deltas: dict[str, dict[str, float | None]] = {}
    for label, row in compact.items():
        if label == "eval":
            continue
        label_delta: dict[str, float | None] = {}
        for field in numeric_fields:
            left = row.get(field)
            right = eval_summary.get(field)
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                label_delta[f"{field}_minus_eval"] = float(left) - float(right)
            else:
                label_delta[f"{field}_minus_eval"] = None
        deltas[label] = label_delta
    signature_gates = {
        label: _workload_signature_gate_for_pair(row, eval_summary)
        for label, row in compact.items()
        if label != "eval"
    }
    return {
        "summaries": compact,
        "deltas_vs_eval": deltas,
        "workload_signature_gate": {
            "schema_version": 1,
            "all_available": all(bool(row.get("gate_available")) for row in signature_gates.values()),
            "all_pass": bool(signature_gates) and all(bool(row.get("gate_pass")) for row in signature_gates.values()),
            "pairs": signature_gates,
        },
    }

def _range_audit_ratios(config: ExperimentConfig) -> list[float]:
    """Return configured multi-budget range-audit ratios, deduped and sorted."""
    raw = getattr(config.model, "range_audit_compression_ratios", []) or []
    return sorted({float(value) for value in raw if 0.0 < float(value) <= 1.0})

def _evaluation_metrics_payload(metrics: MethodEvaluation) -> dict[str, Any]:
    """Serialize method metrics with explicit range fields."""
    return {
        "aggregate_f1": metrics.aggregate_f1,
        "per_type_f1": metrics.per_type_f1,
        "compression_ratio": metrics.compression_ratio,
        "latency_ms": metrics.latency_ms,
        "avg_retained_point_gap": metrics.avg_retained_point_gap,
        "avg_retained_point_gap_norm": metrics.avg_retained_point_gap_norm,
        "max_retained_point_gap": metrics.max_retained_point_gap,
        "geometric_distortion": metrics.geometric_distortion,
        "avg_length_preserved": metrics.avg_length_preserved,
        "combined_query_shape_score": metrics.combined_query_shape_score,
        "range_point_f1": metrics.range_point_f1,
        "range_ship_f1": metrics.range_ship_f1,
        "range_ship_coverage": metrics.range_ship_coverage,
        "range_entry_exit_f1": metrics.range_entry_exit_f1,
        "range_crossing_f1": metrics.range_crossing_f1,
        "range_temporal_coverage": metrics.range_temporal_coverage,
        "range_gap_coverage": metrics.range_gap_coverage,
        "range_gap_time_coverage": metrics.range_gap_time_coverage,
        "range_gap_distance_coverage": metrics.range_gap_distance_coverage,
        "range_gap_min_coverage": metrics.range_gap_min_coverage,
        "range_turn_coverage": metrics.range_turn_coverage,
        "range_shape_score": metrics.range_shape_score,
        "range_usefulness_score": metrics.range_usefulness_score,
        "range_usefulness_gap_time_score": metrics.range_usefulness_gap_time_score,
        "range_usefulness_gap_distance_score": metrics.range_usefulness_gap_distance_score,
        "range_usefulness_gap_min_score": metrics.range_usefulness_gap_min_score,
        "range_usefulness_schema_version": metrics.range_usefulness_schema_version,
        "range_usefulness_gap_ablation_version": metrics.range_usefulness_gap_ablation_version,
        "query_useful_v1_score": metrics.query_useful_v1_score,
        "query_useful_v1_schema_version": metrics.query_useful_v1_schema_version,
        "query_useful_v1_components": metrics.query_useful_v1_components,
        "range_audit": metrics.range_audit,
    }

def _target_budget_row(target_diagnostics: dict[str, Any], compression_ratio: float) -> dict[str, Any]:
    """Return target-diagnostics row closest to the run compression ratio."""
    rows = target_diagnostics.get("budget_rows") or []
    if not isinstance(rows, list) or not rows:
        return {}
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
        distance = abs(ratio - float(compression_ratio))
        if distance < best_distance:
            best_distance = distance
            best_row = row
    return best_row

def _optional_delta(left: float | None, right: float | None) -> float | None:
    """Return ``left - right`` when both values are present."""
    if left is None or right is None:
        return None
    return float(left) - float(right)

def _metric_value(metrics: MethodEvaluation | None, field_name: str) -> float | None:
    """Read one float field from optional method metrics."""
    if metrics is None:
        return None
    value = getattr(metrics, field_name)
    return float(value) if value is not None else None

def _range_learned_fill_summary(
    *,
    learned_fill_diagnostics: dict[str, MethodEvaluation],
    training_target_diagnostics: dict[str, Any],
    range_diagnostics_summary: dict[str, Any],
    compression_ratio: float,
) -> dict[str, Any]:
    """Build a compact run-level summary for learned-fill diagnostics."""
    mlqds = learned_fill_diagnostics.get("MLQDS")
    random_fill = learned_fill_diagnostics.get("TemporalRandomFill")
    oracle_fill = learned_fill_diagnostics.get("TemporalOracleFill")
    mlqds_usefulness = _metric_value(mlqds, "range_usefulness_score")
    random_usefulness = _metric_value(random_fill, "range_usefulness_score")
    oracle_usefulness = _metric_value(oracle_fill, "range_usefulness_score")
    mlqds_point = _metric_value(mlqds, "range_point_f1")
    random_point = _metric_value(random_fill, "range_point_f1")
    oracle_point = _metric_value(oracle_fill, "range_point_f1")

    train_signal = (
        range_diagnostics_summary.get("train", {})
        .get("range_signal", {})
        if isinstance(range_diagnostics_summary, dict)
        else {}
    )
    train_labels = train_signal.get("labels", {}) if isinstance(train_signal, dict) else {}
    target_row = _target_budget_row(training_target_diagnostics, float(compression_ratio))
    usefulness_delta = _optional_delta(mlqds_usefulness, random_usefulness)
    point_delta = _optional_delta(mlqds_point, random_point)

    return {
        "summary_version": 1,
        "compression_ratio": float(compression_ratio),
        "methods": sorted(learned_fill_diagnostics.keys()),
        "mlqds_range_usefulness_score": mlqds_usefulness,
        "temporal_random_fill_range_usefulness_score": random_usefulness,
        "temporal_oracle_fill_range_usefulness_score": oracle_usefulness,
        "mlqds_vs_temporal_random_fill_range_usefulness": usefulness_delta,
        "temporal_oracle_fill_gap_range_usefulness": _optional_delta(oracle_usefulness, mlqds_usefulness),
        "mlqds_range_point_f1": mlqds_point,
        "temporal_random_fill_range_point_f1": random_point,
        "temporal_oracle_fill_range_point_f1": oracle_point,
        "mlqds_vs_temporal_random_fill_range_point_f1": point_delta,
        "temporal_oracle_fill_gap_range_point_f1": _optional_delta(oracle_point, mlqds_point),
        "learned_fill_beats_temporal_random_usefulness": (
            bool(usefulness_delta > 0.0) if usefulness_delta is not None else None
        ),
        "learned_fill_beats_temporal_random_point_f1": bool(point_delta > 0.0) if point_delta is not None else None,
        "oracle_notes": {
            "temporal_oracle_fill_kind": "temporal_base_plus_additive_label_fill",
            "exact_optimum": False,
            "purpose": (
                "diagnostic upper reference for learned residual fill, "
                "not exact retained-set RangeUseful optimum"
            ),
        },
        "train_positive_label_mass": (
            train_labels.get("positive_label_mass") if isinstance(train_labels, dict) else None
        ),
        "train_label_component_mass_basis": (
            train_labels.get("component_label_mass_basis") if isinstance(train_labels, dict) else None
        ),
        "train_label_component_mass_fraction": (
            train_labels.get("component_positive_label_mass_fraction", {}) if isinstance(train_labels, dict) else {}
        ),
        "target_budget_row": target_row,
        "target_positive_label_mass": training_target_diagnostics.get("positive_label_mass"),
        "target_budget_ratio": target_row.get("total_budget_ratio"),
        "target_effective_fill_budget_ratio": target_row.get("effective_fill_budget_ratio"),
        "target_temporal_base_label_mass_fraction": target_row.get("temporal_base_label_mass_fraction"),
        "target_residual_label_mass_fraction": target_row.get("residual_label_mass_fraction"),
        "target_residual_positive_label_fraction": target_row.get("residual_positive_label_fraction"),
    }
