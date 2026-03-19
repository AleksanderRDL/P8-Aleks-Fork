"""
turn_aware_qds_model.py

Defines the TurnAwareQDSModel — a neural network variant that predicts
per-point importance scores for AIS trajectory data while taking turn
intensity into account.

Architecture
------------
Identical to TrajectoryQDSModel but accepts an 8-feature point vector:

1. Point Encoder   : MLP  [8 → 64 → 64]
2. Query Encoder   : MLP  [6 → 64 → 64]
3. Cross-Attention : MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
                     Queries attend to points so each point accumulates
                     query-driven context.
4. Importance Predictor : MLP [64 → 32 → 1] + Sigmoid

Point columns:  [time, lat, lon, speed, heading, is_start, is_end, turn_score]
Query columns:  [lat_min, lat_max, lon_min, lon_max, time_start, time_end]

The extra ``turn_score`` feature allows the model to learn that points at
trajectory bends may carry additional structural importance independent of
the spatiotemporal query workload.
"""

import torch
import torch.nn as nn
from torch import Tensor


class TurnAwareQDSModel(nn.Module):
    """Turn-aware Query-Driven Simplification model for AIS trajectory data.

    Identical architecture to :class:`TrajectoryQDSModel` but accepts an
    extended 8-feature point vector that includes a ``turn_score`` column.
    This allows the model to learn relationships between direction changes
    and query-relevant importance, resulting in simplified trajectories that
    better preserve trajectory shapes (bends and turns).

    Args:
        embed_dim: Embedding dimensionality used throughout the model.
        num_heads: Number of attention heads in the cross-attention layer.
    """

    #: Number of point features expected by this model.
    POINT_FEATURES: int = 8

    def __init__(self, embed_dim: int = 64, num_heads: int = 4) -> None:
        super().__init__()

        # --- Point encoder: [time, lat, lon, speed, heading, is_start, is_end, turn_score] → embed_dim ---
        self.point_encoder = nn.Sequential(
            nn.Linear(self.POINT_FEATURES, embed_dim),
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
            points:  Tensor of shape [N, 8] with columns
                     [time, lat, lon, speed, heading, is_start, is_end,
                     turn_score].
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
