"""
trajectory_visualizer.py

Visualization utilities for AIS trajectory data.

Uses matplotlib with the Agg backend so plots can be generated in
headless environments (no display required).
"""

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
    """Plot ship trajectories as lines in lat/lon space.

    Each trajectory is drawn as a separate coloured line.  The first
    point of each trajectory is marked with a circle and the last point
    with an 'x'.

    Args:
        trajectories: List of trajectory tensors, each [T, 5] with columns
                      [time, lat, lon, speed, heading].
        title:        Figure title.
        save_path:    If provided, save the figure to this file path.
                      Otherwise, the figure is created but not displayed.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = plt.get_cmap("tab20")

    for i, traj in enumerate(trajectories):
        lats = traj[:, 1].numpy()
        lons = traj[:, 2].numpy()
        color = cmap(i % 20)
        ax.plot(lons, lats, "-", color=color, linewidth=1.0, alpha=0.8, label=f"Ship {i}")
        ax.plot(lons[0],  lats[0],  "o", color=color, markersize=5)
        ax.plot(lons[-1], lats[-1], "x", color=color, markersize=5)

    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(title)
    if len(trajectories) <= 10:
        ax.legend(fontsize=7, loc="best")
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
    """Plot trajectories overlaid with semi-transparent query rectangles.

    Args:
        trajectories: List of trajectory tensors, each [T, 5].
        queries:      Tensor of shape [M, 6] with columns
                      [lat_min, lat_max, lon_min, lon_max, time_start, time_end].
        title:        Figure title.
        save_path:    If provided, save the figure to this file path.
    """
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
