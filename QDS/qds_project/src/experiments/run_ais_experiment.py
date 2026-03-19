"""End-to-end AIS QDS experiment pipeline. See src/experiments/README.md for full details."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import torch

from src.data.ais_loader import generate_synthetic_ais_data, load_ais_csv
from src.data.trajectory_dataset import TrajectoryDataset
from src.queries.query_generator import (
    generate_uniform_queries,
    generate_density_biased_queries,
    generate_mixed_queries,
)
from src.queries.query_executor import run_queries
from src.training.importance_labels import compute_importance
from src.training.train_model import train_model
from src.simplification.simplify_trajectories import simplify_trajectories
from src.models.turn_aware_qds_model import TurnAwareQDSModel
from src.evaluation.metrics import query_error, compression_ratio as compute_compression_ratio, query_latency
from src.evaluation.baselines import (
    random_sampling,
    uniform_temporal_sampling,
    douglas_peucker,
)
from src.visualization.trajectory_visualizer import (
    plot_trajectories,
    plot_queries_on_trajectories,
)
from src.visualization.importance_visualizer import (
    plot_importance,
    plot_trajectories_with_importance_and_queries,
    plot_simplification_results,
    plot_simplification_time_slices,
)


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

        spatial_matches = (
            (chunk[:, None, 1] >= queries[None, :, 0]) &
            (chunk[:, None, 1] <= queries[None, :, 1]) &
            (chunk[:, None, 2] >= queries[None, :, 2]) &
            (chunk[:, None, 2] <= queries[None, :, 3])
        )
        spatial_any = spatial_matches.any(dim=1)
        spatiotemporal_any = (
            spatial_matches &
            (chunk[:, None, 0] >= queries[None, :, 4]) &
            (chunk[:, None, 0] <= queries[None, :, 5])
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
    # Only export the core 5 columns: timestamp, lat, lon, speed, heading.
    # The is_start/is_end endpoint flags (columns 5–6) are internal model
    # features and are not written to the output CSV.
    points_export = points_cpu[:, :5]
    with open(output_path, "w", newline="", encoding="utf-8") as out_file:
        writer = csv.writer(out_file)
        writer.writerow(["timestamp", "lat", "lon", "speed", "heading"])

        for start in range(0, points_export.shape[0], write_chunk_size):
            end = min(points_export.shape[0], start + write_chunk_size)
            writer.writerows(points_export[start:end].tolist())


def _generate_queries_for_workload(
    trajectories: list[torch.Tensor],
    n_queries: int,
    workload: str,
    density_ratio: float = 0.7,
    spatial_fraction: float = 0.03,
    temporal_fraction: float = 0.10,
    spatial_bound_lower_quantile: float = 0.01,
    spatial_bound_upper_quantile: float = 0.99,
) -> torch.Tensor:
    """Return a query tensor for the requested *workload* type."""
    if workload == "uniform":
        return generate_uniform_queries(
            trajectories,
            n_queries=n_queries,
            spatial_fraction=spatial_fraction,
            temporal_fraction=temporal_fraction,
            spatial_bound_lower_quantile=spatial_bound_lower_quantile,
            spatial_bound_upper_quantile=spatial_bound_upper_quantile,
        )
    if workload == "density":
        return generate_density_biased_queries(
            trajectories,
            n_queries=n_queries,
            spatial_fraction=spatial_fraction,
            temporal_fraction=temporal_fraction,
        )
    if workload == "mixed":
        return generate_mixed_queries(
            trajectories,
            total_queries=n_queries,
            density_ratio=density_ratio,
            spatial_fraction=spatial_fraction,
            temporal_fraction=temporal_fraction,
            spatial_bound_lower_quantile=spatial_bound_lower_quantile,
            spatial_bound_upper_quantile=spatial_bound_upper_quantile,
        )
    raise ValueError(
        f"Unknown workload '{workload}'. Choose from: uniform, density, mixed."
    )


def _workload_title(workload: str) -> str:
    """Return a human-readable title string for the given workload type."""
    return {
        "uniform": "Uniform Query Workload",
        "density": "Density-Biased Query Workload",
        "mixed":   "Mixed Query Workload",
    }.get(workload, workload.capitalize() + " Query Workload")


def run_ais_experiment(
    n_ships: int = 10,
    n_points: int = 100,
    n_queries: int = 100,
    epochs: int = 50,
    threshold: float = 0.5,
    target_ratio: float | None = None,
    compression_ratio: float | None = 0.2,
    min_points_per_trajectory: int = 5,
    max_train_points: int | None = None,
    model_max_points: int | None = 300_000,
    point_batch_size: int = 50_000,
    importance_chunk_size: int = 200_000,
    dp_max_points: int = 200_000,
    skip_baselines: bool = False,
    skip_visualizations: bool = False,
    max_visualization_points: int = 200_000,
    max_visualization_ships: int = 200,
    max_points_per_ship_plot: int = 2_000,
    csv_path: str | None = None,
    save_csv: bool = False,
    workload: str = "density",
    density_ratio: float = 0.7,
    query_spatial_fraction: float = 0.03,
    query_temporal_fraction: float = 0.10,
    query_spatial_lower_quantile: float = 0.01,
    query_spatial_upper_quantile: float = 0.99,
    model_type: str = "baseline",
    turn_bias_weight: float = 0.1,
    turn_score_method: str = "heading",
) -> None:
    """Run the full AIS QDS experiment and print a results table."""
    if workload == "all":
        # Run each workload independently and collect results for a combined table
        all_results: dict[str, dict[str, tuple[float, float, float]]] = {}
        for wl in ("uniform", "density", "mixed"):
            print(f"\n{'='*65}")
            print(f"Running workload: {_workload_title(wl)}")
            print(f"{'='*65}")
            wl_results = _run_single_workload(
                workload=wl,
                density_ratio=density_ratio,
                n_ships=n_ships,
                n_points=n_points,
                n_queries=n_queries,
                epochs=epochs,
                threshold=threshold,
                target_ratio=target_ratio,
                compression_ratio=compression_ratio,
                min_points_per_trajectory=min_points_per_trajectory,
                max_train_points=max_train_points,
                model_max_points=model_max_points,
                point_batch_size=point_batch_size,
                importance_chunk_size=importance_chunk_size,
                dp_max_points=dp_max_points,
                skip_baselines=skip_baselines,
                skip_visualizations=skip_visualizations,
                max_visualization_points=max_visualization_points,
                max_visualization_ships=max_visualization_ships,
                max_points_per_ship_plot=max_points_per_ship_plot,
                csv_path=csv_path,
                save_csv=save_csv,
                query_spatial_fraction=query_spatial_fraction,
                query_temporal_fraction=query_temporal_fraction,
                query_spatial_lower_quantile=query_spatial_lower_quantile,
                query_spatial_upper_quantile=query_spatial_upper_quantile,
                model_type=model_type,
                turn_bias_weight=turn_bias_weight,
                turn_score_method=turn_score_method,
            )
            all_results[wl] = wl_results

        # Print combined comparison table
        _print_workload_comparison_table(all_results)
        return

    _run_single_workload(
        workload=workload,
        density_ratio=density_ratio,
        n_ships=n_ships,
        n_points=n_points,
        n_queries=n_queries,
        epochs=epochs,
        threshold=threshold,
        target_ratio=target_ratio,
        compression_ratio=compression_ratio,
        min_points_per_trajectory=min_points_per_trajectory,
        max_train_points=max_train_points,
        model_max_points=model_max_points,
        point_batch_size=point_batch_size,
        importance_chunk_size=importance_chunk_size,
        dp_max_points=dp_max_points,
        skip_baselines=skip_baselines,
        skip_visualizations=skip_visualizations,
        max_visualization_points=max_visualization_points,
        max_visualization_ships=max_visualization_ships,
        max_points_per_ship_plot=max_points_per_ship_plot,
        csv_path=csv_path,
        save_csv=save_csv,
        query_spatial_fraction=query_spatial_fraction,
        query_temporal_fraction=query_temporal_fraction,
        query_spatial_lower_quantile=query_spatial_lower_quantile,
        query_spatial_upper_quantile=query_spatial_upper_quantile,
        model_type=model_type,
        turn_bias_weight=turn_bias_weight,
        turn_score_method=turn_score_method,
    )


def _print_workload_comparison_table(
    all_results: dict[str, dict[str, tuple[float, float, float]]],
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
        for method_name, (err, ratio, latency) in methods.items():
            err_str = _format_query_error_value(err)
            print(
                f"{wl_label:<{col_w}} "
                f"{method_name:<{col_n}} "
                f"{err_str:>{col_e}} "
                f"{ratio:>{col_r}.4f} "
                f"{latency:>{col_l}.4f}"
            )
            wl_label = ""  # only print workload label on the first method row

    print("=" * total_w)


def _run_single_workload(
    workload: str,
    density_ratio: float,
    n_ships: int,
    n_points: int,
    n_queries: int,
    epochs: int,
    threshold: float,
    target_ratio: float | None,
    compression_ratio: float | None,
    min_points_per_trajectory: int,
    max_train_points: int | None,
    model_max_points: int | None,
    point_batch_size: int,
    importance_chunk_size: int,
    dp_max_points: int,
    skip_baselines: bool,
    skip_visualizations: bool,
    max_visualization_points: int,
    max_visualization_ships: int,
    max_points_per_ship_plot: int,
    csv_path: str | None,
    save_csv: bool,
    query_spatial_fraction: float,
    query_temporal_fraction: float,
    query_spatial_lower_quantile: float,
    query_spatial_upper_quantile: float,
    model_type: str = "baseline",
    turn_bias_weight: float = 0.1,
    turn_score_method: str = "heading",
) -> dict[str, tuple[float, float, float]]:
    """Run the full AIS QDS pipeline for a single query workload."""
    print("=" * 65)
    print(f"AIS Query-Driven Simplification (QDS) Experiment")
    print(f"Workload: {_workload_title(workload)}")
    print(f"Turn score method: {turn_score_method}")
    print("=" * 65)

    # Select compute device (GPU if available, otherwise CPU).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"       Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load or generate AIS data
    # ------------------------------------------------------------------
    loaded_from_csv = bool(csv_path and os.path.exists(csv_path))

    _t0 = time.time()
    if loaded_from_csv:
        assert csv_path is not None
        print(f"\n[1/8] Loading AIS data from {csv_path} …")
        trajectories = load_ais_csv(csv_path, turn_score_method=turn_score_method)
    else:
        print(f"\n[1/8] Generating synthetic AIS data ({n_ships} ships × {n_points} pts) …")
        trajectories = generate_synthetic_ais_data(
            n_ships=n_ships,
            n_points_per_ship=n_points,
            turn_score_method=turn_score_method,
        )

    dataset = TrajectoryDataset(trajectories)
    points  = dataset.get_all_points().to(device)  # [N, 7]
    traj_boundaries = dataset.get_trajectory_boundaries()
    print(f"       Total ships: {len(trajectories)}, Total points: {points.shape[0]}")
    print(f"       Data loading time: {time.time() - _t0:.2f}s")

    # ------------------------------------------------------------------
    # 2. Generate query workload
    # ------------------------------------------------------------------
    _t0 = time.time()
    print(f"\n[2/8] Generating {n_queries} queries ({_workload_title(workload)}) …")
    queries = _generate_queries_for_workload(
        trajectories,
        n_queries,
        workload,
        density_ratio=density_ratio,
        spatial_fraction=query_spatial_fraction,
        temporal_fraction=query_temporal_fraction,
        spatial_bound_lower_quantile=query_spatial_lower_quantile,
        spatial_bound_upper_quantile=query_spatial_upper_quantile,
    ).to(device)
    print(f"       Query tensor shape: {queries.shape}")
    lat_span = (queries[:, 1] - queries[:, 0]).mean().item()
    lon_span = (queries[:, 3] - queries[:, 2]).mean().item()
    time_span = (queries[:, 5] - queries[:, 4]).mean().item()
    print(
        "       Avg query spans "
        f"lat={lat_span:.4f}, lon={lon_span:.4f}, time={time_span:.2f}"
    )
    print(f"       Query generation time: {time.time() - _t0:.2f}s")

    # ------------------------------------------------------------------
    # 3. Compute importance labels
    # ------------------------------------------------------------------
    _t0 = time.time()
    print("\n[3/8] Computing ground-truth importance labels …")
    importance = compute_importance(points, queries, chunk_size=importance_chunk_size)
    print(f"       Importance range: [{importance.min():.4f}, {importance.max():.4f}]")
    print(f"       Importance computation time: {time.time() - _t0:.2f}s")

    effective_max_train_points = max_train_points
    if (
        effective_max_train_points is None
        and model_max_points is not None
        and points.shape[0] > int(model_max_points)
    ):
        effective_max_train_points = int(model_max_points)
        print(
            "       Large dataset detected: using sampled training subset "
            f"{effective_max_train_points}/{points.shape[0]} points."
        )

    # ------------------------------------------------------------------
    # 4. Train model(s)
    # ------------------------------------------------------------------
    _t0 = time.time()

    # Support model_type="all" to train and compare both models.
    _model_types_to_run = ["baseline", "turn_aware"] if model_type == "all" else [model_type]

    models: dict[str, object] = {}
    for mt in _model_types_to_run:
        label = "TrajectoryQDSModel" if mt == "baseline" else "TurnAwareQDSModel"
        print(f"\n[4/8] Training {label} ({epochs} epochs, model_type={mt}) …")
        models[mt] = train_model(
            trajectories,
            queries,
            epochs=epochs,
            importance=importance,
            max_points=effective_max_train_points,
            importance_chunk_size=importance_chunk_size,
            point_batch_size=point_batch_size,
            model_type=mt,
        )
    print(f"       Model training time: {time.time() - _t0:.2f}s")

    # ------------------------------------------------------------------
    # 5. Simplify with model(s)
    # ------------------------------------------------------------------
    _t0 = time.time()
    print("\n[5/8] Simplifying trajectories with trained model(s) …")

    # We may run simplification for one or both model types.
    # Results are stored per-model-type so the comparison table can show them.
    ml_results_by_type: dict[str, tuple] = {}

    for mt, trained_model in models.items():
        eff_turn_bias = turn_bias_weight if mt == "turn_aware" else 0.0
        label = "ML QDS" if mt == "baseline" else "ML QDS (turn-aware)"

        if compression_ratio is not None:
            print(
                f"       Mode: per-trajectory compression "
                f"(ratio={compression_ratio}, min_pts={min_points_per_trajectory}, "
                f"model={mt})"
            )
            ml_simp, ret_mask, ml_sc = simplify_trajectories(
                points,
                trained_model,
                queries,
                query_scores=importance,
                model_max_points=model_max_points,
                importance_chunk_size=importance_chunk_size,
                trajectory_boundaries=traj_boundaries,
                compression_ratio=compression_ratio,
                min_points_per_trajectory=min_points_per_trajectory,
                turn_bias_weight=eff_turn_bias,
            )
            effective_threshold = None
        else:
            effective_threshold = threshold

            if target_ratio is not None:
                if not (0.0 < target_ratio <= 1.0):
                    raise ValueError("target_ratio must be in (0, 1].")

                _, _, ml_sc = simplify_trajectories(
                    points,
                    trained_model,
                    queries,
                    threshold=0.0,
                    query_scores=importance,
                    model_max_points=model_max_points,
                    importance_chunk_size=importance_chunk_size,
                    trajectory_boundaries=traj_boundaries,
                    compression_ratio=None,
                    turn_bias_weight=eff_turn_bias,
                )

                n_points_total = points.shape[0]
                target_count = max(1, min(n_points_total, int(round(target_ratio * n_points_total))))
                topk_vals, topk_idx = torch.topk(ml_sc, k=target_count)
                effective_threshold = float(topk_vals.min().item())

                ret_mask = torch.zeros(n_points_total, dtype=torch.bool, device=points.device)
                ret_mask[topk_idx] = True
                ml_simp = points[ret_mask]

                print(
                    f"       target_ratio={target_ratio:.4f} "
                    f"-> auto-threshold={effective_threshold:.4f}"
                )
            else:
                ml_simp, ret_mask, ml_sc = simplify_trajectories(
                    points,
                    trained_model,
                    queries,
                    threshold=threshold,
                    query_scores=importance,
                    model_max_points=model_max_points,
                    importance_chunk_size=importance_chunk_size,
                    trajectory_boundaries=traj_boundaries,
                    compression_ratio=None,
                    min_points_per_trajectory=min_points_per_trajectory,
                    turn_bias_weight=eff_turn_bias,
                )

        ml_ratio_mt = compute_compression_ratio(points, ml_simp)
        print(
            f"       {label} retained {ml_simp.shape[0]}/{points.shape[0]} points "
            f"(ratio={ml_ratio_mt:.3f})"
        )
        ml_results_by_type[mt] = (ml_simp, ret_mask, ml_sc, label)

    # Use the first (or only) model result for single-model statistics/visualisation.
    primary_mt = _model_types_to_run[0]
    ml_simplified, retained_mask, ml_scores, primary_label = ml_results_by_type[primary_mt]
    ml_ratio = compute_compression_ratio(points, ml_simplified)

    print(f"       Simplification time: {time.time() - _t0:.2f}s")

    # Trajectory retention statistics (Feature 6)
    n_trajectories = len(trajectories)
    n_traj_retained = sum(
        1 for start, end in traj_boundaries
        if retained_mask[start:end].any().item()
    )
    avg_pts_before = points.shape[0] / max(1, n_trajectories)
    avg_pts_after  = ml_simplified.shape[0] / max(1, n_trajectories)
    print(f"       Trajectories retained: {n_traj_retained}/{n_trajectories}")
    print(f"       Avg points per trajectory before: {avg_pts_before:.1f}")
    print(f"       Avg points per trajectory after:  {avg_pts_after:.1f}")

    if loaded_from_csv and csv_path is not None:
        if save_csv:
            clean_csv_path = _clean_csv_output_path(csv_path)
            _save_retained_points_csv(ml_simplified, clean_csv_path)
            print(f"       Saved retained points CSV: {clean_csv_path}")
        else:
            print("       Skipped saving cleaned file (set --save_csv to enable).")

    removed_mask = ~retained_mask
    removed_in_spatial, removed_in_spatiotemporal = _count_removed_overlap_stats(
        points,
        removed_mask,
        queries,
        chunk_size=importance_chunk_size,
    )

    print(
        "       Removed points in spatial rectangles: "
        f"{removed_in_spatial}/{int(removed_mask.sum().item())}"
    )
    print(
        "       Removed points in full spatiotemporal queries: "
        f"{removed_in_spatiotemporal}/{int(removed_mask.sum().item())}"
    )

    if removed_in_spatial > 0 and removed_in_spatiotemporal == 0:
        print(
            "       Note: query rectangles in the plot are spatial only; "
            "removed points can be outside query time windows."
        )

    full_query_results = run_queries(points, queries)
    ml_query_results = run_queries(ml_simplified, queries)
    abs_diff = (full_query_results - ml_query_results).abs()
    rel_diff = abs_diff / full_query_results.abs().clamp(min=1e-8)

    # Raw numerical differences can be non-zero due to floating-point reduction
    # order even when practical query outcomes are unchanged.
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

    q = torch.tensor([0.50, 0.90, 0.95, 0.99], dtype=ml_scores.dtype, device=ml_scores.device)
    q_vals = torch.quantile(ml_scores, q).tolist()
    print(
        "       Score quantiles "
        f"p50={q_vals[0]:.4f}, p90={q_vals[1]:.4f}, p95={q_vals[2]:.4f}, p99={q_vals[3]:.4f}"
    )

    if ml_simplified.shape[0] <= max(3, int(0.01 * points.shape[0])):
        print(
            "       Warning: very few points retained. "
            "Try a lower threshold (e.g. near p90/p95 score quantiles)."
        )

    # ------------------------------------------------------------------
    # 6. Baselines at the same compression ratio
    # ------------------------------------------------------------------
    print(f"\n[6/8] Running baselines at ratio ≈ {ml_ratio:.3f} …")

    # Start with all trained-model results
    methods: dict[str, torch.Tensor] = {
        label: ml_simp for _, (ml_simp, _, _, label) in ml_results_by_type.items()
    }

    if skip_baselines:
        print("       Skipping baselines (--skip_baselines enabled).")
    else:
        rand_simplified = random_sampling(points, ratio=ml_ratio)
        temp_simplified = uniform_temporal_sampling(points, ratio=ml_ratio)
        methods["Random"] = rand_simplified
        methods["Temporal"] = temp_simplified

        if points.shape[0] <= dp_max_points:
            methods["Douglas-Peucker"] = douglas_peucker(points, epsilon=0.01)
        else:
            print(
                "       Skipping Douglas-Peucker baseline for large dataset "
                f"({points.shape[0]} points > dp_max_points={dp_max_points})."
            )

    # ------------------------------------------------------------------
    # 7. Evaluate all methods
    # ------------------------------------------------------------------
    _t0 = time.time()
    print("\n[7/8] Evaluating all methods …")
    results: dict[str, tuple[float, float, float]] = {}
    for name, simplified in methods.items():
        err     = query_error(points, simplified, queries)
        ratio   = compute_compression_ratio(points, simplified)
        latency = query_latency(simplified, queries) * 1000  # ms
        results[name] = (err, ratio, latency)
    print(f"       Evaluation time: {time.time() - _t0:.2f}s")

    # ------------------------------------------------------------------
    # 8. Print comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 67)
    print(f"{'Method':<20} {'Query Error':>12} {'Comp. Ratio':>12} {'Latency (ms)':>14}")
    print("-" * 67)
    for name, (err, ratio, latency) in results.items():
        err_str = _format_query_error_value(err)
        print(f"{name:<20} {err_str:>12} {ratio:>12.4f} {latency:>14.4f}")
    print("=" * 67)

    # ------------------------------------------------------------------
    # 9. Visualizations
    # ------------------------------------------------------------------
    if skip_visualizations:
        print("\n[8/8] Skipping visualizations (--skip_visualizations enabled).")
        print("\nDone.")
        return results

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
        max_ships=max_visualization_ships,
        max_points_per_ship=max_points_per_ship_plot,
    )
    viz_points, viz_importance, viz_retained_mask, viz_scores = _downsample_points_for_visualization(
        points,
        importance,
        retained_mask,
        ml_scores,
        max_points=max_visualization_points,
    )

    if viz_points.shape[0] < points.shape[0] or len(viz_trajectories) < len(trajectories):
        print(
            "       Visualization downsampling: "
            f"points {viz_points.shape[0]}/{points.shape[0]}, "
            f"ships {len(viz_trajectories)}/{len(trajectories)}."
        )

    # Matplotlib plotting functions expect CPU-backed tensors.
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
        viz_trajectories_cpu, queries_cpu,
        title=f"AIS Trajectories with {_workload_title(workload)}",
        save_path=queries_path,
    )
    plot_importance(
        viz_points_cpu, viz_importance_cpu,
        title="Ground-Truth Point Importance",
        save_path=importance_path,
    )
    plot_trajectories_with_importance_and_queries(
        viz_trajectories_cpu, viz_points_cpu, viz_importance_cpu, queries_cpu,
        title=f"AIS QDS: Importance + Queries ({_workload_title(workload)})",
        save_path=combined_path,
    )

    simplif_title = (
        f"AIS Trajectory Simplification Results ({primary_label})"
    )
    plot_simplification_results(
        viz_trajectories_cpu,
        viz_points_cpu,
        viz_retained_mask_cpu,
        viz_scores_cpu,
        queries_cpu,
        title=simplif_title,
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

    # If turn-aware model was run, generate an additional turn-score visualization.
    if "turn_aware" in ml_results_by_type and not skip_visualizations:
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

    return results


def main() -> None:
    """Command-line entry point for the AIS QDS experiment."""
    parser = argparse.ArgumentParser(
        description="Run the AIS Query-Driven Simplification experiment"
    )
    parser.add_argument("--n_ships",   type=int,   default=10,   help="Number of ships")
    parser.add_argument("--n_points",  type=int,   default=100,  help="Points per ship")
    parser.add_argument("--n_queries", type=int,   default=100,  help="Number of queries")
    parser.add_argument("--epochs",    type=int,   default=50,   help="Training epochs")
    parser.add_argument("--threshold", type=float, default=0.5,  help="Compression threshold (global mode only)")
    parser.add_argument(
        "--target_ratio",
        type=float,
        default=None,
        help="Target retained ratio in (0, 1]; overrides --threshold (global mode only)",
    )
    parser.add_argument(
        "--compression_ratio",
        type=float,
        default=0.2,
        help=(
            "Per-trajectory compression fraction in (0, 1] (default: 0.2). "
            "Each trajectory keeps max(min_points_per_trajectory, "
            "int(compression_ratio * traj_len)) points. "
            "Pass 0 to disable per-trajectory mode and use global --threshold instead."
        ),
    )
    parser.add_argument(
        "--min_points_per_trajectory",
        type=int,
        default=5,
        help="Minimum number of points to retain per trajectory (default: 5).",
    )
    parser.add_argument(
        "--max_train_points",
        type=int,
        default=None,
        help="Optional cap on number of points used for model training",
    )
    parser.add_argument(
        "--model_max_points",
        type=int,
        default=300000,
        help="Optional cap for full-set model inference during simplification",
    )
    parser.add_argument(
        "--point_batch_size",
        type=int,
        default=50000,
        help="Mini-batch size over points during training",
    )
    parser.add_argument(
        "--importance_chunk_size",
        type=int,
        default=200000,
        help="Chunk size for large-scale importance computation",
    )
    parser.add_argument(
        "--dp_max_points",
        type=int,
        default=200000,
        help="Maximum points for Douglas-Peucker baseline",
    )
    parser.add_argument(
        "--skip_baselines",
        action="store_true",
        help="Skip baseline generation/evaluation",
    )
    parser.add_argument(
        "--skip_visualizations",
        action="store_true",
        help="Skip all visualization generation",
    )
    parser.add_argument(
        "--max_visualization_points",
        type=int,
        default=200000,
        help="Maximum points used in visualization scatter plots",
    )
    parser.add_argument(
        "--max_visualization_ships",
        type=int,
        default=200,
        help="Maximum trajectories used in visualization line plots",
    )
    parser.add_argument(
        "--max_points_per_ship_plot",
        type=int,
        default=2000,
        help="Maximum points per trajectory line in visualization",
    )
    parser.add_argument("--csv_path",  type=str,   default=None, help="Path to AIS CSV file")
    parser.add_argument(
        "--save_csv",
        action="store_true",
        help="Save cleaned retained points CSV when loading data from --csv_path",
    )
    parser.add_argument(
        "--workload",
        type=str,
        default="density",
        choices=["uniform", "density", "mixed", "all"],
        help=(
            "Query workload type: 'uniform' (bounding-box sampling), "
            "'density' (AIS-point-anchored sampling), "
            "'mixed' (blend of both), "
            "or 'all' (run all three and print a comparison table)."
        ),
    )
    parser.add_argument(
        "--density_ratio",
        type=float,
        default=0.7,
        help=(
            "Fraction of density-biased queries in a 'mixed' workload "
            "(ignored for other workload types). Must be in [0, 1]."
        ),
    )
    parser.add_argument(
        "--query_spatial_fraction",
        type=float,
        default=0.03,
        help=(
            "Maximum spatial query width as a fraction of effective lat/lon "
            "range. Lower values produce tighter query boxes."
        ),
    )
    parser.add_argument(
        "--query_temporal_fraction",
        type=float,
        default=0.10,
        help=(
            "Maximum temporal query width as a fraction of time range. "
            "Lower values produce shorter query windows."
        ),
    )
    parser.add_argument(
        "--query_spatial_lower_quantile",
        type=float,
        default=0.01,
        help=(
            "Lower quantile for robust spatial bounds used by uniform query "
            "placement (and uniform part of mixed workload)."
        ),
    )
    parser.add_argument(
        "--query_spatial_upper_quantile",
        type=float,
        default=0.99,
        help=(
            "Upper quantile for robust spatial bounds used by uniform query "
            "placement (and uniform part of mixed workload)."
        ),
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="baseline",
        choices=["baseline", "turn_aware", "all"],
        help=(
            "Model variant to use: 'baseline' (TrajectoryQDSModel, 7 features), "
            "'turn_aware' (TurnAwareQDSModel, 8 features with turn bias), "
            "or 'all' (train and compare both models). Default: 'baseline'."
        ),
    )
    parser.add_argument(
        "--turn_bias_weight",
        type=float,
        default=0.1,
        help=(
            "Additive weight for turn-score bias applied during simplification "
            "when model_type is 'turn_aware'. Default: 0.1."
        ),
    )
    parser.add_argument(
        "--turn_score_method",
        type=str,
        default="heading",
        choices=["heading", "geometry"],
        help=(
            "Method used to compute turn_score: 'heading' (default, wrapped "
            "COG/heading deltas) or 'geometry' (turn angle from lat/lon vectors)."
        ),
    )
    args = parser.parse_args()

    # --compression_ratio 0 disables per-trajectory mode.
    parsed_compression_ratio: float | None = (
        args.compression_ratio if args.compression_ratio > 0.0 else None
    )

    run_ais_experiment(
        n_ships=args.n_ships,
        n_points=args.n_points,
        n_queries=args.n_queries,
        epochs=args.epochs,
        threshold=args.threshold,
        target_ratio=args.target_ratio,
        compression_ratio=parsed_compression_ratio,
        min_points_per_trajectory=args.min_points_per_trajectory,
        max_train_points=args.max_train_points,
        model_max_points=args.model_max_points,
        point_batch_size=args.point_batch_size,
        importance_chunk_size=args.importance_chunk_size,
        dp_max_points=args.dp_max_points,
        skip_baselines=args.skip_baselines,
        skip_visualizations=args.skip_visualizations,
        max_visualization_points=args.max_visualization_points,
        max_visualization_ships=args.max_visualization_ships,
        max_points_per_ship_plot=args.max_points_per_ship_plot,
        csv_path=args.csv_path,
        save_csv=args.save_csv,
        workload=args.workload,
        density_ratio=args.density_ratio,
        query_spatial_fraction=args.query_spatial_fraction,
        query_temporal_fraction=args.query_temporal_fraction,
        query_spatial_lower_quantile=args.query_spatial_lower_quantile,
        query_spatial_upper_quantile=args.query_spatial_upper_quantile,
        model_type=args.model_type,
        turn_bias_weight=args.turn_bias_weight,
        turn_score_method=args.turn_score_method,
    )


if __name__ == "__main__":
    main()
