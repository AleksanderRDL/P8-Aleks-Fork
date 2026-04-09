"""Boundary-aware turn model for AIS trajectories. See src/models/README.md for architecture details."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def compute_boundary_proximity(
    points: Tensor,
    queries: Tensor,
    sigma: float = 1.0,
    point_chunk_size: int | None = 50_000,
    query_chunk_size: int | None = 128,
) -> Tensor:
    """Compute a chunked boundary-proximity feature for each point."""
    if queries.dim() == 1:
        queries = queries.unsqueeze(0)

    lat = points[:, 1]  # [N]
    lon = points[:, 2]  # [N]

    lat_min = queries[:, 0]   # [M]
    lat_max = queries[:, 1]   # [M]
    lon_min = queries[:, 2]   # [M]
    lon_max = queries[:, 3]   # [M]

    n_points = points.shape[0]
    n_queries = queries.shape[0]

    # Preserve previous behavior for empty query tensors (reduction over empty dim).
    if n_queries == 0:
        return torch.empty(n_points, 0, dtype=points.dtype, device=points.device).max(dim=1).values

    p_chunk = n_points if point_chunk_size is None else max(1, int(point_chunk_size))
    q_chunk = n_queries if query_chunk_size is None else max(1, int(query_chunk_size))
    sigma_safe = max(float(sigma), 1e-8)

    proximity_max = torch.empty(n_points, dtype=points.dtype, device=points.device)

    for p_start in range(0, n_points, p_chunk):
        p_end = min(n_points, p_start + p_chunk)
        lat_chunk = lat[p_start:p_end]
        lon_chunk = lon[p_start:p_end]

        chunk_max = torch.full(
            (p_end - p_start,),
            float("-inf"),
            dtype=points.dtype,
            device=points.device,
        )

        for q_start in range(0, n_queries, q_chunk):
            q_end = min(n_queries, q_start + q_chunk)

            lat_min_chunk = lat_min[q_start:q_end]
            lat_max_chunk = lat_max[q_start:q_end]
            lon_min_chunk = lon_min[q_start:q_end]
            lon_max_chunk = lon_max[q_start:q_end]

            # Compute min distance to any boundary edge for each point-query pair
            # without materializing an additional stacked [P, Q, 4] tensor.
            boundary_distance = (lat_chunk[:, None] - lat_min_chunk[None, :]).abs()
            boundary_distance = torch.minimum(
                boundary_distance,
                (lat_chunk[:, None] - lat_max_chunk[None, :]).abs(),
            )
            boundary_distance = torch.minimum(
                boundary_distance,
                (lon_chunk[:, None] - lon_min_chunk[None, :]).abs(),
            )
            boundary_distance = torch.minimum(
                boundary_distance,
                (lon_chunk[:, None] - lon_max_chunk[None, :]).abs(),
            )

            proximity = torch.exp(-boundary_distance / sigma_safe)
            chunk_max = torch.maximum(chunk_max, proximity.max(dim=1).values)

        proximity_max[p_start:p_end] = chunk_max

    return proximity_max  # [N]


def extract_boundary_features(
    points: Tensor,
    queries: Tensor,
    sigma: float = 1.0,
    point_chunk_size: int | None = 50_000,
    query_chunk_size: int | None = 128,
) -> Tensor:
    """Return an ``[N, 1]`` boundary-proximity feature tensor."""
    return compute_boundary_proximity(
        points,
        queries,
        sigma=sigma,
        point_chunk_size=point_chunk_size,
        query_chunk_size=query_chunk_size,
    ).unsqueeze(-1)


class BoundaryAwareTurnModel(nn.Module):
    """Boundary-aware turn model for AIS trajectory QDS.

    Extends :class:`~src.models.turn_aware_qds_model.TurnAwareQDSModel` by
    appending a ``boundary_proximity`` feature (column 8) to the 8-feature
    turn-aware point vector.  The resulting 9-feature input lets the model
    learn that points near query-boundary edges are likely to affect query
    results when removed.

    Point feature schema (9 features):
        ``[time, lat, lon, speed, heading, is_start, is_end, turn_score,
        boundary_proximity]``

    Query feature schema (6 features, unchanged):
        ``[lat_min, lat_max, lon_min, lon_max, time_start, time_end]``

    Hyperparameters:
        sigma  — boundary-proximity decay bandwidth (higher = smoother decay)
        alpha  — additive weight for boundary proximity during score fusion
        beta   — additive weight for turn score during score fusion
    """

    #: Number of point features expected by this model.
    POINT_FEATURES: int = 9

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        sigma: float = 1.0,
        alpha: float = 0.1,
        beta: float = 0.1,
    ) -> None:
        """Initialise the boundary-aware turn model."""
        super().__init__()
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta

        # --- Point encoder: 9 features → embed_dim ---
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
        """Predict per-point importance scores from points and queries."""
        if queries.dim() == 1:
            queries = queries.unsqueeze(0)

        point_emb = self.point_encoder(points).unsqueeze(0)   # [1, N, D]
        query_emb = self.query_encoder(queries).unsqueeze(0)  # [1, M, D]

        # Queries (Q) attend to points (K, V)
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
        base_scores = self.importance_predictor(point_features).squeeze(-1)  # [N]

        # Additive bias from boundary_proximity (col 8) and turn_score (col 7)
        boundary_proximity = points[:, 8]
        turn_score = points[:, 7]
        scores = base_scores + self.alpha * boundary_proximity + self.beta * turn_score

        return scores.clamp(0.0, 1.0)
