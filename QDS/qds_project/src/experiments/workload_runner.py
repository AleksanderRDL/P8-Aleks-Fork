"""Single-workload execution for the AIS QDS experiment pipeline."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch
from torch import Tensor

from src.data.ais_loader import generate_synthetic_ais_data, load_ais_csv
from src.data.trajectory_dataset import TrajectoryDataset
from src.evaluation.metrics import compression_ratio as compute_compression_ratio
from src.experiments.experiment_config import (
    ExperimentConfig,
    MethodMetrics,
    ModelConfig,
    ModelSimplificationResult,
)
from src.experiments.experiment_pipeline_helpers import (
    _build_methods_for_evaluation,
    _clean_csv_output_path,
    _count_removed_overlap_stats,
    _evaluate_methods,
    _generate_queries_for_workload,
    _print_method_comparison_table,
    _resolve_effective_max_train_points,
    _resolve_model_variants,
    _save_retained_points_csv,
    _save_visualizations,
    _simplify_with_model,
    _workload_title,
)
from src.queries.query_executor import run_queries
from src.training.importance_labels import compute_importance
from src.training.train_model import train_model


@dataclass(frozen=True)
class WorkloadData:
    """Loaded/generated dataset assets used by a single workload run."""

    trajectories: list[Tensor]
    points: Tensor
    trajectory_boundaries: list[tuple[int, int]]
    loaded_from_csv: bool


@dataclass(frozen=True)
class SimplificationRun:
    """Simplification outputs for all trained model variants."""

    results_by_type: dict[str, ModelSimplificationResult]
    primary_result: ModelSimplificationResult
    ml_ratio: float


def _print_workload_header(workload: str, model_cfg: ModelConfig) -> None:
    print("=" * 65)
    print("AIS Query-Driven Simplification (QDS) Experiment")
    print(f"Workload: {_workload_title(workload)}")
    print(f"Turn score method: {model_cfg.turn_score_method}")
    print("=" * 65)


def _load_workload_data(config: ExperimentConfig, device: torch.device) -> WorkloadData:
    """Load AIS trajectories from CSV or generate synthetic trajectories."""
    data_cfg = config.data
    model_cfg = config.model
    loaded_from_csv = bool(data_cfg.csv_path and os.path.exists(data_cfg.csv_path))

    start = time.time()
    if loaded_from_csv:
        assert data_cfg.csv_path is not None
        print(f"\n[1/8] Loading AIS data from {data_cfg.csv_path} …")
        trajectories = load_ais_csv(data_cfg.csv_path, turn_score_method=model_cfg.turn_score_method)
    else:
        print(
            "\n[1/8] Generating synthetic AIS data "
            f"({data_cfg.n_ships} ships × {data_cfg.n_points_per_ship} pts) …"
        )
        trajectories = generate_synthetic_ais_data(
            n_ships=data_cfg.n_ships,
            n_points_per_ship=data_cfg.n_points_per_ship,
            turn_score_method=model_cfg.turn_score_method,
        )

    dataset = TrajectoryDataset(trajectories)
    points = dataset.get_all_points().to(device)  # [N, 7] or [N, 8]
    trajectory_boundaries = dataset.get_trajectory_boundaries()

    print(f"       Total ships: {len(trajectories)}, Total points: {points.shape[0]}")
    print(f"       Data loading time: {time.time() - start:.2f}s")

    return WorkloadData(
        trajectories=trajectories,
        points=points,
        trajectory_boundaries=trajectory_boundaries,
        loaded_from_csv=loaded_from_csv,
    )


def _generate_workload_queries(config: ExperimentConfig, trajectories: list[Tensor], device: torch.device) -> Tensor:
    """Build the query tensor for the configured workload."""
    query_cfg = config.query
    workload = query_cfg.workload

    start = time.time()
    print(
        f"\n[2/8] Generating {query_cfg.n_queries} queries "
        f"({_workload_title(workload)}) …"
    )
    queries = _generate_queries_for_workload(trajectories, query_cfg).to(device)
    print(f"       Query tensor shape: {queries.shape}")

    lat_span = (queries[:, 1] - queries[:, 0]).mean().item()
    lon_span = (queries[:, 3] - queries[:, 2]).mean().item()
    time_span = (queries[:, 5] - queries[:, 4]).mean().item()
    print(
        "       Avg query spans "
        f"lat={lat_span:.4f}, lon={lon_span:.4f}, time={time_span:.2f}"
    )
    print(f"       Query generation time: {time.time() - start:.2f}s")
    return queries


def _compute_importance_labels(points: Tensor, queries: Tensor, model_cfg: ModelConfig) -> Tensor:
    """Compute full-dataset ground-truth point importance labels."""
    start = time.time()
    print("\n[3/8] Computing ground-truth importance labels …")
    importance = compute_importance(points, queries, chunk_size=model_cfg.importance_chunk_size)
    print(f"       Importance range: [{importance.min():.4f}, {importance.max():.4f}]")
    print(f"       Importance computation time: {time.time() - start:.2f}s")
    return importance


def _train_models_for_workload(
    trajectories: list[Tensor],
    queries: Tensor,
    points: Tensor,
    importance: Tensor,
    model_cfg: ModelConfig,
) -> dict[str, object]:
    """Train one or more model variants for this workload."""
    effective_max_train_points, used_auto_cap = _resolve_effective_max_train_points(
        n_points=points.shape[0],
        model_cfg=model_cfg,
    )
    if used_auto_cap and effective_max_train_points is not None:
        print(
            "       Large dataset detected: using sampled training subset "
            f"{effective_max_train_points}/{points.shape[0]} points."
        )

    start = time.time()
    model_variants = _resolve_model_variants(model_cfg.model_type)
    models: dict[str, object] = {}
    for model_variant in model_variants:
        model_name = "TrajectoryQDSModel" if model_variant == "baseline" else "TurnAwareQDSModel"
        print(
            f"\n[4/8] Training {model_name} "
            f"({model_cfg.epochs} epochs, model_type={model_variant}) …"
        )
        models[model_variant] = train_model(
            trajectories,
            queries,
            epochs=model_cfg.epochs,
            importance=importance,
            max_points=effective_max_train_points,
            importance_chunk_size=model_cfg.importance_chunk_size,
            point_batch_size=model_cfg.point_batch_size,
            model_type=model_variant,
        )

    print(f"       Model training time: {time.time() - start:.2f}s")
    return models


def _run_model_simplification(
    points: Tensor,
    queries: Tensor,
    importance: Tensor,
    models: dict[str, object],
    model_cfg: ModelConfig,
    trajectory_boundaries: list[tuple[int, int]],
) -> SimplificationRun:
    """Simplify trajectories using all trained model variants."""
    start = time.time()
    print("\n[5/8] Simplifying trajectories with trained model(s) …")

    results_by_type: dict[str, ModelSimplificationResult] = {}
    for model_variant, trained_model in models.items():
        run_result = _simplify_with_model(
            points=points,
            queries=queries,
            importance=importance,
            trained_model=trained_model,
            model_variant=model_variant,
            model_cfg=model_cfg,
            trajectory_boundaries=trajectory_boundaries,
        )
        ratio_for_model = compute_compression_ratio(points, run_result.simplified_points)
        print(
            f"       {run_result.label} retained "
            f"{run_result.simplified_points.shape[0]}/{points.shape[0]} points "
            f"(ratio={ratio_for_model:.3f})"
        )
        results_by_type[model_variant] = run_result

    model_variants = _resolve_model_variants(model_cfg.model_type)
    primary_result = results_by_type[model_variants[0]]
    ml_ratio = compute_compression_ratio(points, primary_result.simplified_points)
    print(f"       Simplification time: {time.time() - start:.2f}s")

    return SimplificationRun(
        results_by_type=results_by_type,
        primary_result=primary_result,
        ml_ratio=ml_ratio,
    )


def _print_trajectory_retention_stats(
    points: Tensor,
    trajectories: list[Tensor],
    trajectory_boundaries: list[tuple[int, int]],
    retained_mask: Tensor,
    simplified_points: Tensor,
) -> None:
    """Print trajectory-level point-retention diagnostics."""
    n_trajectories = len(trajectories)
    n_traj_retained = sum(
        1 for start, end in trajectory_boundaries if retained_mask[start:end].any().item()
    )
    avg_pts_before = points.shape[0] / max(1, n_trajectories)
    avg_pts_after = simplified_points.shape[0] / max(1, n_trajectories)
    print(f"       Trajectories retained: {n_traj_retained}/{n_trajectories}")
    print(f"       Avg points per trajectory before: {avg_pts_before:.1f}")
    print(f"       Avg points per trajectory after:  {avg_pts_after:.1f}")


def _maybe_save_retained_csv(
    data_cfg,
    loaded_from_csv: bool,
    simplified_points: Tensor,
) -> None:
    """Persist retained points to CSV when configured for CSV-backed runs."""
    if not loaded_from_csv or data_cfg.csv_path is None:
        return

    if data_cfg.save_csv:
        clean_csv_path = _clean_csv_output_path(data_cfg.csv_path)
        _save_retained_points_csv(simplified_points, clean_csv_path)
        print(f"       Saved retained points CSV: {clean_csv_path}")
        return

    print("       Skipped saving cleaned file (set --save_csv to enable).")


def _print_removed_overlap_diagnostics(
    points: Tensor,
    retained_mask: Tensor,
    queries: Tensor,
    chunk_size: int,
) -> None:
    """Print removed-point overlap diagnostics against query regions."""
    removed_mask = ~retained_mask
    removed_in_spatial, removed_in_spatiotemporal = _count_removed_overlap_stats(
        points,
        removed_mask,
        queries,
        chunk_size=chunk_size,
    )
    removed_total = int(removed_mask.sum().item())

    print(
        "       Removed points in spatial rectangles: "
        f"{removed_in_spatial}/{removed_total}"
    )
    print(
        "       Removed points in full spatiotemporal queries: "
        f"{removed_in_spatiotemporal}/{removed_total}"
    )

    if removed_in_spatial > 0 and removed_in_spatiotemporal == 0:
        print(
            "       Note: query rectangles in the plot are spatial only; "
            "removed points can be outside query time windows."
        )


def _print_query_change_diagnostics(
    points: Tensor,
    simplified_points: Tensor,
    queries: Tensor,
) -> None:
    """Print query drift diagnostics between original and simplified points."""
    full_query_results = run_queries(points, queries)
    ml_query_results = run_queries(simplified_points, queries)
    abs_diff = (full_query_results - ml_query_results).abs()
    rel_diff = abs_diff / full_query_results.abs().clamp(min=1e-8)

    changed_queries_any = int((abs_diff > 1e-12).sum().item())
    changed_queries_meaningful = int((rel_diff > 1e-4).sum().item())

    print(
        "       Queries changed by ML QDS "
        f"(any delta > 1e-12): {changed_queries_any}/{queries.shape[0]}"
    )
    print(
        "       Queries with meaningful relative change "
        f"(> 1e-4): {changed_queries_meaningful}/{queries.shape[0]}"
    )


def _print_score_distribution_diagnostics(
    points: Tensor,
    simplified_points: Tensor,
    scores: Tensor,
) -> None:
    """Print score quantiles and tiny-retained-set warnings."""
    q = torch.tensor([0.50, 0.90, 0.95, 0.99], dtype=scores.dtype, device=scores.device)
    q_vals = torch.quantile(scores, q).tolist()
    print(
        "       Score quantiles "
        f"p50={q_vals[0]:.4f}, p90={q_vals[1]:.4f}, p95={q_vals[2]:.4f}, p99={q_vals[3]:.4f}"
    )

    if simplified_points.shape[0] <= max(3, int(0.01 * points.shape[0])):
        print(
            "       Warning: very few points retained. "
            "Try a lower threshold (e.g. near p90/p95 score quantiles)."
        )


def _print_primary_simplification_diagnostics(
    config: ExperimentConfig,
    workload_data: WorkloadData,
    queries: Tensor,
    simplification: SimplificationRun,
) -> None:
    """Print diagnostics for the primary simplification result."""
    data_cfg = config.data
    model_cfg = config.model
    points = workload_data.points

    primary_result = simplification.primary_result
    ml_simplified = primary_result.simplified_points
    retained_mask = primary_result.retained_mask

    _print_trajectory_retention_stats(
        points,
        workload_data.trajectories,
        workload_data.trajectory_boundaries,
        retained_mask,
        ml_simplified,
    )
    _maybe_save_retained_csv(data_cfg, workload_data.loaded_from_csv, ml_simplified)
    _print_removed_overlap_diagnostics(
        points,
        retained_mask,
        queries,
        chunk_size=model_cfg.importance_chunk_size,
    )
    _print_query_change_diagnostics(points, ml_simplified, queries)
    _print_score_distribution_diagnostics(points, ml_simplified, primary_result.scores)


def run_single_workload(config: ExperimentConfig) -> dict[str, MethodMetrics]:
    """Run the full AIS QDS pipeline for a single query workload."""
    query_cfg = config.query
    model_cfg = config.model
    baseline_cfg = config.baselines
    viz_cfg = config.visualization
    workload = query_cfg.workload

    _print_workload_header(workload, model_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"       Device: {device}")

    workload_data = _load_workload_data(config, device)
    queries = _generate_workload_queries(config, workload_data.trajectories, device)
    importance = _compute_importance_labels(workload_data.points, queries, model_cfg)

    models = _train_models_for_workload(
        trajectories=workload_data.trajectories,
        queries=queries,
        points=workload_data.points,
        importance=importance,
        model_cfg=model_cfg,
    )
    simplification = _run_model_simplification(
        points=workload_data.points,
        queries=queries,
        importance=importance,
        models=models,
        model_cfg=model_cfg,
        trajectory_boundaries=workload_data.trajectory_boundaries,
    )

    _print_primary_simplification_diagnostics(
        config=config,
        workload_data=workload_data,
        queries=queries,
        simplification=simplification,
    )

    print(f"\n[6/8] Running baselines at ratio ≈ {simplification.ml_ratio:.3f} …")
    methods = _build_methods_for_evaluation(
        points=workload_data.points,
        ml_ratio=simplification.ml_ratio,
        ml_results_by_type=simplification.results_by_type,
        baseline_cfg=baseline_cfg,
    )

    start = time.time()
    print("\n[7/8] Evaluating all methods …")
    results = _evaluate_methods(workload_data.points, queries, methods)
    print(f"       Evaluation time: {time.time() - start:.2f}s")

    _print_method_comparison_table(results)

    if viz_cfg.skip_visualizations:
        print("\n[8/8] Skipping visualizations (--skip_visualizations enabled).")
        print("\nDone.")
        return results

    _save_visualizations(
        workload=workload,
        trajectories=workload_data.trajectories,
        points=workload_data.points,
        importance=importance,
        retained_mask=simplification.primary_result.retained_mask,
        ml_scores=simplification.primary_result.scores,
        queries=queries,
        primary_label=simplification.primary_result.label,
        ml_results_by_type=simplification.results_by_type,
        viz_cfg=viz_cfg,
    )

    return results
