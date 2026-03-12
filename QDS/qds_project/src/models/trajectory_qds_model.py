"""
trajectory_qds_model.py

Defines the TrajectoryQDSModel — a neural network that predicts per-point
importance scores for AIS trajectory data given a spatiotemporal query workload.

Architecture
------------
1. Point Encoder   : MLP  [5 → 64 → 64]
2. Query Encoder   : MLP  [6 → 64 → 64]
3. Cross-Attention : MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
                     Queries attend to points so each point accumulates
                     query-driven context.
4. Importance Predictor : MLP [64 → 32 → 1] + Sigmoid

Point columns:  [time, lat, lon, speed, heading]
Query columns:  [lat_min, lat_max, lon_min, lon_max, time_start, time_end]
"""

import torch
import torch.nn as nn
from torch import Tensor


def normalize_points_and_queries(points: Tensor, queries: Tensor) -> tuple[Tensor, Tensor]:
    """Min-max normalise points and queries using point-cloud feature ranges.

    The same scaling is applied to matching semantic fields so train-time and
    inference-time model inputs are on consistent scales:
      - time      ↔ query time_start/time_end
      - latitude  ↔ query lat_min/lat_max
      - longitude ↔ query lon_min/lon_max

    Args:
        points:  Tensor [N, 5] with columns [time, lat, lon, speed, heading].
        queries: Tensor [M, 6] with columns
                 [lat_min, lat_max, lon_min, lon_max, time_start, time_end].

    Returns:
        Tuple (norm_points, norm_queries).
    """
    norm_points = points.clone()
    norm_queries = queries.clone()

    eps = torch.tensor(1e-8, dtype=points.dtype, device=points.device)

    p_min = points.min(dim=0).values
    p_max = points.max(dim=0).values
    p_range = torch.maximum(p_max - p_min, eps)

    norm_points = (norm_points - p_min) / p_range

    # Query lat bounds use point lat scale
    norm_queries[:, 0] = (norm_queries[:, 0] - p_min[1]) / p_range[1]  # lat_min
    norm_queries[:, 1] = (norm_queries[:, 1] - p_min[1]) / p_range[1]  # lat_max

    # Query lon bounds use point lon scale
    norm_queries[:, 2] = (norm_queries[:, 2] - p_min[2]) / p_range[2]  # lon_min
    norm_queries[:, 3] = (norm_queries[:, 3] - p_min[2]) / p_range[2]  # lon_max

    # Query time bounds use point time scale
    norm_queries[:, 4] = (norm_queries[:, 4] - p_min[0]) / p_range[0]  # time_start
    norm_queries[:, 5] = (norm_queries[:, 5] - p_min[0]) / p_range[0]  # time_end

    norm_points = norm_points.clamp(0.0, 1.0)
    norm_queries = norm_queries.clamp(0.0, 1.0)

    return norm_points, norm_queries


class TrajectoryQDSModel(nn.Module):
    """Query-Driven Simplification model for AIS trajectory data.

    Given a set of trajectory points and a spatiotemporal query workload,
    predicts an importance score in [0, 1] for every point.  High-scoring
    points are more influential for answering the queries and should be
    retained during compression.

    Args:
        embed_dim: Embedding dimensionality used throughout the model.
        num_heads: Number of attention heads in the cross-attention layer.
    """

    def __init__(self, embed_dim: int = 64, num_heads: int = 4) -> None:
        super().__init__()

        # --- Point encoder: [time, lat, lon, speed, heading] → embed_dim ---
        self.point_encoder = nn.Sequential(
            nn.Linear(5, embed_dim),
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
        """Predict per-point importance scores.

        Args:
            points:  Tensor of shape [N, 5] with columns
                     [time, lat, lon, speed, heading].
            queries: Tensor of shape [M, 6] with columns
                     [lat_min, lat_max, lon_min, lon_max, time_start, time_end].

        Returns:
            Tensor of shape [N] with importance scores in [0, 1].
        """
        # Ensure queries has a batch-compatible shape; handle single-query case
        if queries.dim() == 1:
            queries = queries.unsqueeze(0)  # [1, 6]

        # Encode inputs and add batch dimension for attention (batch size = 1)
        # point_emb : [1, N, D]
        # query_emb : [1, M, D]
        point_emb = self.point_encoder(points).unsqueeze(0)
        query_emb = self.query_encoder(queries).unsqueeze(0)

        # Cross-attention: queries (Q) attend to points (K, V).
        # attn_out     : [1, M, D]  — attended query embeddings
        # attn_weights : [1, M, N]  — how much each query attends to each point
        attn_out, attn_weights = self.cross_attention(
            query=query_emb,
            key=point_emb,
            value=point_emb,
        )

        # Build per-point context as a weighted sum of attended query embeddings.
        # attn_weights : [1, M, N] → squeeze → [M, N] → transpose → [N, M]
        # attn_out     : [1, M, D] → squeeze → [M, D]
        # per_point_context = [N, M] @ [M, D] = [N, D]
        attn_w = attn_weights.squeeze(0).transpose(0, 1)  # [N, M]
        attn_o = attn_out.squeeze(0)                       # [M, D]
        per_point_context = torch.mm(attn_w, attn_o)       # [N, D]

        # Combine original point embeddings with query-conditioned context
        point_features = point_emb.squeeze(0) + per_point_context  # [N, D]

        # Predict importance scores: [N]
        scores = self.importance_predictor(point_features).squeeze(-1)
        return scores
