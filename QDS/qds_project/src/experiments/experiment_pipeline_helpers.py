"""Helper utilities for the AIS experiment pipeline."""

from __future__ import annotations

import csv
import math
import os
import tempfile

import torch

from src.evaluation.baselines import (
    douglas_peucker,
    random_sampling,
    uniform_temporal_sampling,
)
from src.evaluation.metrics import (
    compression_ratio as compute_compression_ratio,
    query_error,
    query_latency,
)
from src.experiments.experiment_config import (
    BaselineConfig,
    MethodMetrics,
    ModelConfig,
    ModelSimplificationResult,
    QueryConfig,
    VisualizationConfig,
)
from src.queries.query_generator import (
    generate_density_biased_queries,
    generate_mixed_queries,
    generate_uniform_queries,
)
from src.queries.query_masks import spatial_inclusion_mask, spatiotemporal_inclusion_mask
from src.simplification.simplify_trajectories import simplify_trajectories


def _format_query_error_value(err: float) -> str:
    """Format query error for readability across very small and typical values."""
    return f"{err:.4f}" if err >= 1e-4 else f"{err:.3e}"


def _count_removed_overlap_stats(
    points: torch.Tensor,
    removed_mask: torch.Tensor,
    queries: torch.Tensor,
    chunk_size: int,
) -> tuple[int, int]:
    """Count removed points overlapping spatial-only and full spatiotemporal queries."""
    n_points = points.shape[0]
    removed_in_spatial = 0
    removed_in_spatiotemporal = 0

    for start in range(0, n_points, chunk_size):
        end = min(n_points, start + chunk_size)
        chunk = points[start:end]
        removed_chunk = removed_mask[start:end]
        if not removed_chunk.any():
            continue

        spatial_matches = spatial_inclusion_mask(chunk, queries)
        spatial_any = spatial_matches.any(dim=1)
        spatiotemporal_any = spatiotemporal_inclusion_mask(
            chunk,
            queries,
            spatial_mask=spatial_matches,
        ).any(dim=1)

        removed_in_spatial += int((removed_chunk & spatial_any).sum().item())
        removed_in_spatiotemporal += int((removed_chunk & spatiotemporal_any).sum().item())

    return removed_in_spatial, removed_in_spatiotemporal


def _downsample_trajectories_for_visualization(
    trajectories: list[torch.Tensor],
    max_ships: int,
    max_points_per_ship: int,
) -> list[torch.Tensor]:
    """Create a visualization-friendly subset of trajectories."""
    if len(trajectories) <= max_ships:
        selected = trajectories
    else:
        idx = torch.linspace(0, len(trajectories) - 1, steps=max_ships).long().tolist()
        selected = [trajectories[i] for i in idx]

    reduced: list[torch.Tensor] = []
    for traj in selected:
        if traj.shape[0] > max_points_per_ship:
            step = max(1, math.ceil(traj.shape[0] / max_points_per_ship))
            reduced.append(traj[::step])
        else:
            reduced.append(traj)

    return reduced


def _downsample_points_for_visualization(
    points: torch.Tensor,
    importance: torch.Tensor,
    retained_mask: torch.Tensor,
    scores: torch.Tensor,
    max_points: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Downsample point-level arrays consistently for plotting."""
    n_points = points.shape[0]
    if n_points <= max_points:
        return points, importance, retained_mask, scores

    idx = torch.randperm(n_points, device=points.device)[:max_points]
    idx = torch.sort(idx).values

    return points[idx], importance[idx], retained_mask[idx], scores[idx]


def _clean_csv_output_path(source_csv_path: str) -> str:
    """Build output path for retained-point CSV next to the source file."""
    source_dir = os.path.dirname(source_csv_path) or "."
    source_name = os.path.basename(source_csv_path)
    return os.path.join(source_dir, f"MLClean-{source_name}")


def _save_retained_points_csv(
    retained_points: torch.Tensor,
    output_path: str,
    write_chunk_size: int = 100_000,
) -> None:
    """Save retained points to CSV with schema: timestamp,lat,lon,speed,heading."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    points_cpu = retained_points.detach().cpu()
    points_export = points_cpu[:, :5]
    with open(output_path, "w", newline="", encoding="utf-8") as out_file:
        writer = csv.writer(out_file)
        writer.writerow(["timestamp", "lat", "lon", "speed", "heading"])

        for start in range(0, points_export.shape[0], write_chunk_size):
            end = min(points_export.shape[0], start + write_chunk_size)
            writer.writerows(points_export[start:end].tolist())


def _generate_queries_for_workload(
    trajectories: list[torch.Tensor],
    query_cfg: QueryConfig,
) -> torch.Tensor:
    """Return a query tensor for the requested workload type."""
    workload = query_cfg.workload

    if workload == "uniform":
        return generate_uniform_queries(
            trajectories,
            n_queries=query_cfg.n_queries,
            spatial_fraction=query_cfg.spatial_fraction,
            temporal_fraction=query_cfg.temporal_fraction,
            spatial_bound_lower_quantile=query_cfg.spatial_lower_quantile,
            spatial_bound_upper_quantile=query_cfg.spatial_upper_quantile,
        )
    if workload == "density":
        return generate_density_biased_queries(
            trajectories,
            n_queries=query_cfg.n_queries,
            spatial_fraction=query_cfg.spatial_fraction,
            temporal_fraction=query_cfg.temporal_fraction,
        )
    if workload == "mixed":
        return generate_mixed_queries(
            trajectories,
            total_queries=query_cfg.n_queries,
            density_ratio=query_cfg.density_ratio,
            spatial_fraction=query_cfg.spatial_fraction,
            temporal_fraction=query_cfg.temporal_fraction,
            spatial_bound_lower_quantile=query_cfg.spatial_lower_quantile,
            spatial_bound_upper_quantile=query_cfg.spatial_upper_quantile,
        )
    raise ValueError(
        f"Unknown workload '{workload}'. Choose from: uniform, density, mixed."
    )


def _workload_title(workload: str) -> str:
    """Return a human-readable title string for the given workload type."""
    return {
        "uniform": "Uniform Query Workload",
        "density": "Density-Biased Query Workload",
        "mixed": "Mixed Query Workload",
    }.get(workload, workload.capitalize() + " Query Workload")


def _print_workload_comparison_table(
    all_results: dict[str, dict[str, MethodMetrics]],
) -> None:
    """Print a combined comparison table across multiple workloads."""
    col_w = 20
    col_n = 12
    col_e = 14
    col_r = 16
    col_l = 14
    total_w = col_w + col_n + col_e + col_r + col_l + 5

    print()
    print("=" * total_w)
    print("## Workload Comparison")
    print("=" * total_w)
    header = (
        f"{'Workload':<{col_w}} "
        f"{'Method':<{col_n}} "
        f"{'Query Error':>{col_e}} "
        f"{'Compression Ratio':>{col_r}} "
        f"{'Latency (ms)':>{col_l}}"
    )
    print(header)
    print("-" * total_w)

    for wl_name, methods in all_results.items():
        wl_label = _workload_title(wl_name)
        for method_name, metrics in methods.items():
            err_str = _format_query_error_value(metrics.query_error)
            print(
                f"{wl_label:<{col_w}} "
                f"{method_name:<{col_n}} "
                f"{err_str:>{col_e}} "
                f"{metrics.compression_ratio:>{col_r}.4f} "
                f"{metrics.latency_ms:>{col_l}.4f}"
            )
            wl_label = ""

    print("=" * total_w)


def _resolve_effective_max_train_points(
    n_points: int,
    model_cfg: ModelConfig,
) -> tuple[int | None, bool]:
    """Resolve the effective max training points and whether auto-capping was applied."""
    if model_cfg.max_train_points is not None:
        return model_cfg.max_train_points, False

    if model_cfg.model_max_points is not None and n_points > int(model_cfg.model_max_points):
        return int(model_cfg.model_max_points), True

    return None, False


def _resolve_model_variants(model_type: str) -> list[str]:
    """Return the model variants to execute based on CLI/model config."""
    return ["baseline", "turn_aware"] if model_type == "all" else [model_type]


def _build_model_simplification_result(
    simplified_points: torch.Tensor,
    retained_mask: torch.Tensor,
    scores: torch.Tensor,
    label: str,
) -> ModelSimplificationResult:
    """Create a strongly-typed simplification result object."""
    return ModelSimplificationResult(
        simplified_points=simplified_points,
        retained_mask=retained_mask,
        scores=scores,
        label=label,
    )


def _run_simplify_trajectories(
    points: torch.Tensor,
    queries: torch.Tensor,
    importance: torch.Tensor,
    trained_model: object,
    model_cfg: ModelConfig,
    trajectory_boundaries: list[tuple[int, int]],
    *,
    threshold: float,
    compression_ratio: float | None,
    turn_bias_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute simplify_trajectories with shared experiment defaults."""
    return simplify_trajectories(
        points,
        trained_model,
        queries,
        threshold=threshold,
        query_scores=importance,
        model_max_points=model_cfg.model_max_points,
        importance_chunk_size=model_cfg.importance_chunk_size,
        trajectory_boundaries=trajectory_boundaries,
        compression_ratio=compression_ratio,
        min_points_per_trajectory=model_cfg.min_points_per_trajectory,
        turn_bias_weight=turn_bias_weight,
    )


def _simplify_with_model(
    points: torch.Tensor,
    queries: torch.Tensor,
    importance: torch.Tensor,
    trained_model: object,
    model_variant: str,
    model_cfg: ModelConfig,
    trajectory_boundaries: list[tuple[int, int]],
) -> ModelSimplificationResult:
    """Run trajectory simplification for a single trained model variant."""
    eff_turn_bias = model_cfg.turn_bias_weight if model_variant == "turn_aware" else 0.0
    label = "ML QDS" if model_variant == "baseline" else "ML QDS (turn-aware)"

    if model_cfg.compression_ratio is not None:
        print(
            "       Mode: per-trajectory compression "
            f"(ratio={model_cfg.compression_ratio}, "
            f"min_pts={model_cfg.min_points_per_trajectory}, model={model_variant})"
        )
        ml_simplified, retained_mask, ml_scores = _run_simplify_trajectories(
            points,
            queries,
            importance,
            trained_model,
            model_cfg,
            trajectory_boundaries,
            threshold=model_cfg.threshold,
            compression_ratio=model_cfg.compression_ratio,
            turn_bias_weight=eff_turn_bias,
        )
        return _build_model_simplification_result(
            ml_simplified,
            retained_mask,
            ml_scores,
            label,
        )

    if model_cfg.target_ratio is not None:
        if not (0.0 < model_cfg.target_ratio <= 1.0):
            raise ValueError("target_ratio must be in (0, 1].")

        _, _, ml_scores = _run_simplify_trajectories(
            points,
            queries,
            importance,
            trained_model,
            model_cfg,
            trajectory_boundaries,
            threshold=0.0,
            compression_ratio=None,
            turn_bias_weight=eff_turn_bias,
        )

        n_points_total = points.shape[0]
        target_count = max(
            1,
            min(n_points_total, int(round(model_cfg.target_ratio * n_points_total))),
        )
        topk_vals, _ = torch.topk(ml_scores, k=target_count)
        effective_threshold = float(topk_vals.min().item())

        # Re-run in global-threshold mode so endpoint/min-point trajectory
        # constraints are applied consistently.
        ml_simplified, retained_mask, _ = _run_simplify_trajectories(
            points,
            queries,
            importance,
            trained_model,
            model_cfg,
            trajectory_boundaries,
            threshold=effective_threshold,
            compression_ratio=None,
            turn_bias_weight=eff_turn_bias,
        )
        retained_count = int(retained_mask.sum().item())

        print(
            f"       target_ratio={model_cfg.target_ratio:.4f} "
            f"-> auto-threshold={effective_threshold:.4f} "
            f"(retained={retained_count}/{n_points_total})"
        )
        return _build_model_simplification_result(
            ml_simplified,
            retained_mask,
            ml_scores,
            label,
        )

    ml_simplified, retained_mask, ml_scores = _run_simplify_trajectories(
        points,
        queries,
        importance,
        trained_model,
        model_cfg,
        trajectory_boundaries,
        threshold=model_cfg.threshold,
        compression_ratio=None,
        turn_bias_weight=eff_turn_bias,
    )
    return _build_model_simplification_result(
        ml_simplified,
        retained_mask,
        ml_scores,
        label,
    )


def _build_methods_for_evaluation(
    points: torch.Tensor,
    ml_ratio: float,
    ml_results_by_type: dict[str, ModelSimplificationResult],
    baseline_cfg: BaselineConfig,
) -> dict[str, torch.Tensor]:
    """Build the method-to-points map for evaluation, including optional baselines."""
    methods: dict[str, torch.Tensor] = {
        result.label: result.simplified_points for result in ml_results_by_type.values()
    }

    if baseline_cfg.skip_baselines:
        print("       Skipping baselines (--skip_baselines enabled).")
        return methods

    methods["Random"] = random_sampling(points, ratio=ml_ratio)
    methods["Temporal"] = uniform_temporal_sampling(points, ratio=ml_ratio)

    if points.shape[0] <= baseline_cfg.dp_max_points:
        methods["Douglas-Peucker"] = douglas_peucker(points, epsilon=0.01)
    else:
        print(
            "       Skipping Douglas-Peucker baseline for large dataset "
            f"({points.shape[0]} points > dp_max_points={baseline_cfg.dp_max_points})."
        )

    return methods


def _evaluate_methods(
    points: torch.Tensor,
    queries: torch.Tensor,
    methods: dict[str, torch.Tensor],
) -> dict[str, MethodMetrics]:
    """Evaluate all simplified-method outputs against query metrics."""
    results: dict[str, MethodMetrics] = {}
    for name, simplified in methods.items():
        results[name] = MethodMetrics(
            query_error=query_error(points, simplified, queries),
            compression_ratio=compute_compression_ratio(points, simplified),
            latency_ms=query_latency(simplified, queries) * 1000,
        )
    return results


def _print_method_comparison_table(results: dict[str, MethodMetrics]) -> None:
    """Print per-method evaluation metrics in tabular form."""
    print("\n" + "=" * 67)
    print(f"{'Method':<20} {'Query Error':>12} {'Comp. Ratio':>12} {'Latency (ms)':>14}")
    print("-" * 67)
    for name, metrics in results.items():
        err_str = _format_query_error_value(metrics.query_error)
        print(
            f"{name:<20} {err_str:>12} "
            f"{metrics.compression_ratio:>12.4f} {metrics.latency_ms:>14.4f}"
        )
    print("=" * 67)


def _save_visualizations(
    workload: str,
    trajectories: list[torch.Tensor],
    points: torch.Tensor,
    importance: torch.Tensor,
    retained_mask: torch.Tensor,
    ml_scores: torch.Tensor,
    queries: torch.Tensor,
    primary_label: str,
    ml_results_by_type: dict[str, ModelSimplificationResult],
    viz_cfg: VisualizationConfig,
) -> None:
    """Generate and save experiment visualization artifacts."""
    from src.visualization.trajectory_visualizer import (
        plot_queries_on_trajectories,
        plot_trajectories,
    )
    from src.visualization.importance_visualizer import (
        plot_importance,
        plot_simplification_results,
        plot_simplification_time_slices,
        plot_trajectories_with_importance_and_queries,
    )

    temp_dir = tempfile.gettempdir()
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    trajectories_path = os.path.join(temp_dir, "ais_trajectories.png")
    queries_path = os.path.join(temp_dir, f"ais_queries_{workload}.png")
    importance_path = os.path.join(temp_dir, "ais_importance.png")
    combined_path = os.path.join(temp_dir, "ais_combined.png")
    time_slices_path = "results/simplification_time_slices.png"

    print(f"\n[8/8] Saving visualizations to {temp_dir} and results/ …")

    viz_trajectories = _downsample_trajectories_for_visualization(
        trajectories,
        max_ships=viz_cfg.max_visualization_ships,
        max_points_per_ship=viz_cfg.max_points_per_ship_plot,
    )
    viz_points, viz_importance, viz_retained_mask, viz_scores = _downsample_points_for_visualization(
        points,
        importance,
        retained_mask,
        ml_scores,
        max_points=viz_cfg.max_visualization_points,
    )

    if viz_points.shape[0] < points.shape[0] or len(viz_trajectories) < len(trajectories):
        print(
            "       Visualization downsampling: "
            f"points {viz_points.shape[0]}/{points.shape[0]}, "
            f"ships {len(viz_trajectories)}/{len(trajectories)}."
        )

    viz_trajectories_cpu = [traj.detach().cpu() for traj in viz_trajectories]
    viz_points_cpu = viz_points.detach().cpu()
    viz_importance_cpu = viz_importance.detach().cpu()
    viz_retained_mask_cpu = viz_retained_mask.detach().cpu()
    viz_scores_cpu = viz_scores.detach().cpu()
    queries_cpu = queries.detach().cpu()

    plot_trajectories(
        viz_trajectories_cpu,
        title="AIS Trajectories",
        save_path=trajectories_path,
    )
    plot_queries_on_trajectories(
        viz_trajectories_cpu,
        queries_cpu,
        title=f"AIS Trajectories with {_workload_title(workload)}",
        save_path=queries_path,
    )
    plot_importance(
        viz_points_cpu,
        viz_importance_cpu,
        title="Ground-Truth Point Importance",
        save_path=importance_path,
    )
    plot_trajectories_with_importance_and_queries(
        viz_trajectories_cpu,
        viz_points_cpu,
        viz_importance_cpu,
        queries_cpu,
        title=f"AIS QDS: Importance + Queries ({_workload_title(workload)})",
        save_path=combined_path,
    )

    plot_simplification_results(
        viz_trajectories_cpu,
        viz_points_cpu,
        viz_retained_mask_cpu,
        viz_scores_cpu,
        queries_cpu,
        title=f"AIS Trajectory Simplification Results ({primary_label})",
        save_path="results/simplification_visualization.png",
    )
    plot_simplification_time_slices(
        viz_points_cpu,
        viz_retained_mask_cpu,
        viz_scores_cpu,
        queries_cpu,
        title="AIS Simplification (Time-Sliced Query Context)",
        n_slices=4,
        save_path=time_slices_path,
    )

    if "turn_aware" in ml_results_by_type:
        from src.visualization.importance_visualizer import plot_turn_scores

        turn_path = os.path.join(temp_dir, "ais_turn_scores.png")
        plot_turn_scores(
            viz_points_cpu,
            viz_retained_mask_cpu,
            title="AIS Turn-Score Visualization (Turn-Aware Model)",
            save_path=turn_path,
        )
        print(f"       Saved: {turn_path}")

    print(f"       Saved: {trajectories_path}")
    print(f"       Saved: {queries_path}")
    print(f"       Saved: {importance_path}")
    print(f"       Saved: {combined_path}")
    print("       Saved: results/simplification_visualization.png")
    print(f"       Saved: {time_slices_path}")
    print("\nDone.")
