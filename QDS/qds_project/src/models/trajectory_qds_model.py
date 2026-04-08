"""Baseline QDS model for AIS trajectories. See src/models/README.md for architecture details."""

import torch
from torch import Tensor

from src.models.attention_qds_model_base import AttentionQDSModelBase


def normalize_points_and_queries(points: Tensor, queries: Tensor) -> tuple[Tensor, Tensor]:
    """Min-max normalise points and queries using point-cloud feature ranges."""
    norm_points = points.clone()
    norm_queries = queries.clone()

    eps = torch.tensor(1e-8, dtype=points.dtype, device=points.device)

    # Normalise only the first 5 features (time, lat, lon, speed, heading).
    # The is_start and is_end binary flags (columns 5 and 6) are already in
    # [0, 1] and are passed through unchanged.
    n_spatial_features = min(5, points.shape[1])
    p_min = points[:, :n_spatial_features].min(dim=0).values
    p_max = points[:, :n_spatial_features].max(dim=0).values
    p_range = torch.maximum(p_max - p_min, eps)

    norm_points[:, :n_spatial_features] = (
        (norm_points[:, :n_spatial_features] - p_min) / p_range
    ).clamp(0.0, 1.0)

    # Query lat bounds use point lat scale
    norm_queries[:, 0] = (norm_queries[:, 0] - p_min[1]) / p_range[1]  # lat_min
    norm_queries[:, 1] = (norm_queries[:, 1] - p_min[1]) / p_range[1]  # lat_max

    # Query lon bounds use point lon scale
    norm_queries[:, 2] = (norm_queries[:, 2] - p_min[2]) / p_range[2]  # lon_min
    norm_queries[:, 3] = (norm_queries[:, 3] - p_min[2]) / p_range[2]  # lon_max

    # Query time bounds use point time scale
    norm_queries[:, 4] = (norm_queries[:, 4] - p_min[0]) / p_range[0]  # time_start
    norm_queries[:, 5] = (norm_queries[:, 5] - p_min[0]) / p_range[0]  # time_end

    norm_queries = norm_queries.clamp(0.0, 1.0)

    return norm_points, norm_queries


class TrajectoryQDSModel(AttentionQDSModelBase):
    """Query-Driven Simplification model for AIS trajectory data."""

    POINT_FEATURES: int = 7

    def __init__(self, embed_dim: int = 64, num_heads: int = 4) -> None:
        super().__init__(
            point_features=self.POINT_FEATURES,
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
