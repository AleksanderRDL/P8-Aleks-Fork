"""Importance score visualization utilities. See src/visualization/README.md."""

from __future__ import annotations

import math
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless rendering — must come before pyplot import

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import torch
from torch import Tensor

from src.queries.query_masks import spatial_inclusion_mask, spatiotemporal_inclusion_mask


def _np(t: Tensor):
    """Convert tensor to a NumPy array safely from any device."""
    return t.detach().cpu().numpy()


def _save_and_close(fig, save_path: Optional[str]) -> None:
    """Save figure when requested, then close it."""
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _compute_removed_masks(
    all_points: Tensor,
    retained_mask: Tensor,
    queries: Tensor,
) -> tuple[Tensor, Tensor]:
    """Split removed points into query-relevant and query-irrelevant subsets."""
    removed_mask = ~retained_mask
    spatial_matches = spatial_inclusion_mask(all_points, queries)
    spatiotemporal_matches = spatiotemporal_inclusion_mask(
        all_points,
        queries,
        spatial_mask=spatial_matches,
    )
    in_spatiotemporal_query = spatiotemporal_matches.any(dim=1)

    removed_relevant_mask = removed_mask & in_spatiotemporal_query
    removed_irrelevant_mask = removed_mask & ~in_spatiotemporal_query
    return removed_irrelevant_mask, removed_relevant_mask


def _plot_removed_points(
    ax,
    all_points: Tensor,
    removed_irrelevant_mask: Tensor,
    removed_relevant_mask: Tensor,
    *,
    irrelevant_size: int,
    irrelevant_alpha: float,
    relevant_size: int,
    relevant_alpha: float,
    with_labels: bool,
) -> None:
    """Scatter removed points with separate styling for relevance classes."""
    if removed_irrelevant_mask.any():
        ax.scatter(
            _np(all_points[removed_irrelevant_mask, 2]),
            _np(all_points[removed_irrelevant_mask, 1]),
            s=irrelevant_size,
            color="red",
            alpha=irrelevant_alpha,
            label="Removed (outside query-time)" if with_labels else None,
            zorder=3,
        )

    if removed_relevant_mask.any():
        ax.scatter(
            _np(all_points[removed_relevant_mask, 2]),
            _np(all_points[removed_relevant_mask, 1]),
            s=relevant_size,
            color="orange",
            alpha=relevant_alpha,
            label="Removed (query-relevant)" if with_labels else None,
            zorder=5,
        )


def _draw_query_rectangles(
    ax,
    queries: Tensor,
    *,
    edgecolor: str,
    facecolor: str,
    alpha: float,
    linewidth: float,
    zorder: int,
) -> bool:
    """Draw query rectangles and return whether any rectangle was added."""
    has_queries = False
    for q in queries:
        rect = mpatches.Rectangle(
            (float(q[2]), float(q[0])),
            float(q[3] - q[2]),
            float(q[1] - q[0]),
            linewidth=linewidth,
            edgecolor=edgecolor,
            facecolor=facecolor,
            alpha=alpha,
            zorder=zorder,
        )
        ax.add_patch(rect)
        has_queries = True
    return has_queries


def plot_importance(
    points: Tensor,
    importance_scores: Tensor,
    title: str = "Point Importance",
    save_path: Optional[str] = None,
) -> None:
    """Scatter plot of trajectory points coloured by importance score."""
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(
        _np(points[:, 2]),
        _np(points[:, 1]),
        c=_np(importance_scores),
        cmap="plasma",
        s=8,
        alpha=0.8,
        vmin=0.0,
        vmax=1.0,
    )
    plt.colorbar(sc, ax=ax, label="Importance Score")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(title)
    plt.tight_layout()
    _save_and_close(fig, save_path)


def plot_simplification_results(
    trajectories: list[Tensor],
    all_points: Tensor,
    retained_mask: Tensor,
    importance_scores: Tensor,
    queries: Tensor,
    title: str = "AIS Trajectory Simplification Results",
    save_path: Optional[str] = None,
) -> None:
    """Visualise retained/removed points alongside query regions."""
    fig, ax = plt.subplots(figsize=(12, 8))

    for traj in trajectories:
        ax.plot(_np(traj[:, 2]), _np(traj[:, 1]), color="lightgray", linewidth=0.5)

    retained_points = all_points[retained_mask]
    removed_irrelevant_mask, removed_relevant_mask = _compute_removed_masks(
        all_points,
        retained_mask,
        queries,
    )
    _plot_removed_points(
        ax,
        all_points,
        removed_irrelevant_mask,
        removed_relevant_mask,
        irrelevant_size=5,
        irrelevant_alpha=0.25,
        relevant_size=12,
        relevant_alpha=0.8,
        with_labels=True,
    )

    sc = ax.scatter(
        _np(retained_points[:, 2]),
        _np(retained_points[:, 1]),
        c=_np(importance_scores[retained_mask]),
        cmap="viridis",
        s=8,
        label="Retained Points",
        zorder=4,
    )
    plt.colorbar(sc, ax=ax, label="Importance Score")

    if all_points.shape[1] >= 7:
        retained_start_mask = retained_mask & (all_points[:, 5] > 0.5)
        retained_end_mask = retained_mask & (all_points[:, 6] > 0.5)

        if retained_start_mask.any():
            ax.scatter(
                _np(all_points[retained_start_mask, 2]),
                _np(all_points[retained_start_mask, 1]),
                s=50,
                color="green",
                marker="o",
                zorder=7,
                label="Trajectory start",
            )
        if retained_end_mask.any():
            ax.scatter(
                _np(all_points[retained_end_mask, 2]),
                _np(all_points[retained_end_mask, 1]),
                s=50,
                color="blue",
                marker="s",
                zorder=7,
                label="Trajectory end",
            )

    has_queries = _draw_query_rectangles(
        ax,
        queries,
        edgecolor="blue",
        facecolor="blue",
        alpha=0.15,
        linewidth=1.0,
        zorder=2,
    )

    handles, labels = ax.get_legend_handles_labels()
    if has_queries:
        handles.append(
            mpatches.Patch(
                facecolor="blue",
                edgecolor="blue",
                alpha=0.15,
                label="Queries",
            )
        )
        labels.append("Queries")
    ax.legend(handles=handles, labels=labels, fontsize=8, loc="best")

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    _save_and_close(fig, save_path)


def plot_trajectories_with_importance_and_queries(
    trajectories: list[Tensor],
    points: Tensor,
    importance_scores: Tensor,
    queries: Tensor,
    title: str = "Trajectories: Importance + Queries",
    save_path: Optional[str] = None,
) -> None:
    """Combined plot: trajectory lines coloured by importance with query overlays."""
    fig, ax = plt.subplots(figsize=(12, 8))
    cmap_traj = plt.get_cmap("tab20")

    for idx, traj in enumerate(trajectories):
        ax.plot(
            _np(traj[:, 2]),
            _np(traj[:, 1]),
            "-",
            color=cmap_traj(idx % 20),
            linewidth=0.8,
            alpha=0.4,
        )

    sc = ax.scatter(
        _np(points[:, 2]),
        _np(points[:, 1]),
        c=_np(importance_scores),
        cmap="plasma",
        s=10,
        alpha=0.9,
        vmin=0.0,
        vmax=1.0,
        zorder=3,
    )
    plt.colorbar(sc, ax=ax, label="Importance Score")

    _draw_query_rectangles(
        ax,
        queries,
        edgecolor="cyan",
        facecolor="cyan",
        alpha=0.07,
        linewidth=0.8,
        zorder=2,
    )

    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(title)
    plt.tight_layout()
    _save_and_close(fig, save_path)


def plot_simplification_time_slices(
    all_points: Tensor,
    retained_mask: Tensor,
    importance_scores: Tensor,
    queries: Tensor,
    title: str = "AIS Simplification (Time-Sliced)",
    n_slices: int = 4,
    save_path: Optional[str] = None,
) -> None:
    """Visualise simplification results across n_slices temporal windows."""
    n_slices = max(1, int(n_slices))
    times = all_points[:, 0]
    t_min = float(times.min().item())
    t_max = float(times.max().item())

    n_cols = 2 if n_slices > 1 else 1
    n_rows = math.ceil(n_slices / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7 * n_cols, 4.5 * n_rows),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    removed_irrelevant_mask, removed_relevant_mask = _compute_removed_masks(
        all_points,
        retained_mask,
        queries,
    )
    edges = torch.linspace(t_min, t_max, steps=n_slices + 1)
    retained_artist = None

    for idx in range(n_slices):
        ax = axes_flat[idx]
        t0 = float(edges[idx].item())
        t1 = float(edges[idx + 1].item())
        in_slice = (times >= t0) & ((times <= t1) if idx == n_slices - 1 else (times < t1))

        _plot_removed_points(
            ax,
            all_points,
            removed_irrelevant_mask & in_slice,
            removed_relevant_mask & in_slice,
            irrelevant_size=6,
            irrelevant_alpha=0.25,
            relevant_size=14,
            relevant_alpha=0.85,
            with_labels=False,
        )

        retained_slice = retained_mask & in_slice
        if retained_slice.any():
            retained_artist = ax.scatter(
                _np(all_points[retained_slice, 2]),
                _np(all_points[retained_slice, 1]),
                c=_np(importance_scores[retained_slice]),
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                s=10,
                zorder=4,
            )

        active_queries = (queries[:, 5] >= t0) & (queries[:, 4] <= t1)
        _draw_query_rectangles(
            ax,
            queries[active_queries],
            edgecolor="blue",
            facecolor="blue",
            alpha=0.10,
            linewidth=0.8,
            zorder=2,
        )

        ax.set_title(f"t=[{t0:.0f}, {t1:.0f}]  active queries={int(active_queries.sum().item())}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    for idx in range(n_slices, len(axes_flat)):
        axes_flat[idx].axis("off")

    if retained_artist is not None:
        cax = fig.add_axes([0.915, 0.22, 0.015, 0.62])
        cbar = fig.colorbar(retained_artist, cax=cax)
        cbar.set_label("Importance Score")

    legend_handles = [
        mpatches.Patch(
            facecolor="blue",
            edgecolor="blue",
            alpha=0.10,
            label="Queries active in slice",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="red",
            alpha=0.25,
            markersize=6,
            label="Removed (outside query-time)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="orange",
            alpha=0.85,
            markersize=7,
            label="Removed (query-relevant)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="green",
            alpha=0.9,
            markersize=6,
            label="Retained",
        ),
    ]
    legend = fig.legend(
        handles=legend_handles,
        fontsize=8,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=True,
    )
    legend.get_frame().set_alpha(0.95)

    fig.suptitle(title)
    fig.subplots_adjust(
        left=0.06,
        right=0.89,
        bottom=0.18,
        top=0.90,
        wspace=0.22,
        hspace=0.30,
    )
    _save_and_close(fig, save_path)


def plot_turn_scores(
    points: Tensor,
    retained_mask: Tensor,
    title: str = "AIS Turn-Score Visualization",
    save_path: Optional[str] = None,
) -> None:
    """Scatter plot highlighting trajectory points by turn intensity (column 7)."""
    if points.shape[1] < 8:
        print(
            "plot_turn_scores: turn_score column (col 7) not present "
            f"(points has {points.shape[1]} features). Skipping."
        )
        return

    fig, ax = plt.subplots(figsize=(11, 7))
    sc = ax.scatter(
        _np(points[:, 2]),
        _np(points[:, 1]),
        c=_np(points[:, 7]),
        cmap="hot_r",
        s=8,
        alpha=0.6,
        vmin=0.0,
        vmax=1.0,
        zorder=3,
        label="All points (turn intensity)",
    )
    plt.colorbar(sc, ax=ax, label="Turn Score (angle / π)")

    high_turn_threshold = 0.2
    high_turn_retained = retained_mask & (points[:, 7] > high_turn_threshold)
    if high_turn_retained.any():
        ax.scatter(
            _np(points[high_turn_retained, 2]),
            _np(points[high_turn_retained, 1]),
            s=60,
            marker="*",
            color="deepskyblue",
            edgecolors="black",
            linewidths=0.5,
            zorder=6,
            label=f"Retained high-turn (score > {high_turn_threshold:.1f})",
        )

    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    _save_and_close(fig, save_path)
