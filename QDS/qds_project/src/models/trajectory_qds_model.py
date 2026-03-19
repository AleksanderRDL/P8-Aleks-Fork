"""Baseline QDS model for AIS trajectories. See src/models/README.md for architecture details."""

import torch
import torch.nn as nn
from torch import Tensor


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


class TrajectoryQDSModel(nn.Module):
    """Query-Driven Simplification model for AIS trajectory data."""

    def __init__(self, embed_dim: int = 64, num_heads: int = 4) -> None:
        super().__init__()

        # --- Point encoder: [time, lat, lon, speed, heading, is_start, is_end] → embed_dim ---
        self.point_encoder = nn.Sequential(
            nn.Linear(7, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # --- Query encoder: [lat_min, lat_max, lon_min, lon_max, t_start, t_end] → embed_dim ---
        self.query_encoder = nn.Sequential(
            nn.Linear(6, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # --- Cross-attention (queries attend to points) ---
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # --- Importance predictor ---
        self.importance_predictor = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, points: Tensor, queries: Tensor) -> Tensor:
        """Predict per-point importance scores."""
        # Ensure queries has a batch-compatible shape; handle single-query case
        if queries.dim() == 1:
            queries = queries.unsqueeze(0)

        point_emb = self.point_encoder(points).unsqueeze(0)   # [1, N, D]
        query_emb = self.query_encoder(queries).unsqueeze(0)  # [1, M, D]

        # Queries (Q) attend to points (K, V); gives per-query context over points.
        attn_out, attn_weights = self.cross_attention(
            query=query_emb,
            key=point_emb,
            value=point_emb,
        )

        # Build per-point context: [N, M] @ [M, D] = [N, D]
        attn_w = attn_weights.squeeze(0).transpose(0, 1)  # [N, M]
        attn_o = attn_out.squeeze(0)                       # [M, D]
        per_point_context = torch.mm(attn_w, attn_o)       # [N, D]

        point_features = point_emb.squeeze(0) + per_point_context  # [N, D]
        scores = self.importance_predictor(point_features).squeeze(-1)
        return scores
