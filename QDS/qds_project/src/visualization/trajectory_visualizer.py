"""Trajectory visualization utilities for AIS data. See src/visualization/README.md."""

from __future__ import annotations

from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # headless rendering — must come before pyplot import

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import torch
from torch import Tensor


def plot_trajectories(
    trajectories: List[Tensor],
    title: str = "AIS Trajectories",
    save_path: Optional[str] = None,
) -> None:
    """Plot vessel trajectories as coloured lines in lat/lon space."""
    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = plt.get_cmap("tab20")

    for i, traj in enumerate(trajectories):
        lats = traj[:, 1].numpy()
        lons = traj[:, 2].numpy()
        color = cmap(i % 20)
        ax.plot(lons, lats, "-", color=color, linewidth=1.0, alpha=0.8, label=f"Ship {i}")
        ax.plot(lons[0],  lats[0],  "o", color="green", markersize=6, zorder=5)
        ax.plot(lons[-1], lats[-1], "s", color="blue",  markersize=6, zorder=5)

    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(title)
    if len(trajectories) <= 10:
        ax.legend(fontsize=7, loc="best")

    # Endpoint legend entries
    start_handle = plt.Line2D(
        [0], [0], marker="o", color="w", markerfacecolor="green", markersize=7,
    )
    end_handle = plt.Line2D(
        [0], [0], marker="s", color="w", markerfacecolor="blue", markersize=7,
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles + [start_handle, end_handle],
        labels=labels + ["Start point", "End point"],
        fontsize=7, loc="best",
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_queries_on_trajectories(
    trajectories: List[Tensor],
    queries: Tensor,
    title: str = "Trajectories with Queries",
    save_path: Optional[str] = None,
) -> None:
    """Overlay semi-transparent query rectangles on trajectory paths."""
    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = plt.get_cmap("tab20")

    # Draw trajectories
    for i, traj in enumerate(trajectories):
        lats = traj[:, 1].numpy()
        lons = traj[:, 2].numpy()
        ax.plot(lons, lats, "-", color=cmap(i % 20), linewidth=1.0, alpha=0.7)

    # Draw query rectangles (lon on x-axis, lat on y-axis)
    for q in queries:
        lat_min, lat_max = float(q[0]), float(q[1])
        lon_min, lon_max = float(q[2]), float(q[3])
        width  = lon_max - lon_min
        height = lat_max - lat_min
        rect = mpatches.Rectangle(
            (lon_min, lat_min), width, height,
            linewidth=0.8,
            edgecolor="red",
            facecolor="red",
            alpha=0.05,
        )
        ax.add_patch(rect)

    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
