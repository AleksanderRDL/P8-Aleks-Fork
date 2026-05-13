"""Per-trajectory transformer model for typed importance prediction. See models/README.md for details."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from models.attention_utils import chunked_cross_attention_context


class TrajectoryQDSModel(nn.Module):
    """Transformer + query cross-attention predictor. See models/README.md for details."""

    def __init__(
        self,
        point_dim: int,
        query_dim: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        type_embed_dim: int = 16,
        query_chunk_size: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.point_dim = point_dim
        self.query_dim = query_dim
        self.embed_dim = embed_dim
        self.query_chunk_size = query_chunk_size
        self.register_buffer("_positional_encoding_cache", torch.empty(0), persistent=False)

        self.point_encoder = nn.Sequential(
            nn.Linear(point_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.local_transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # The project trains one pure workload per model. Query type still
        # conditions the query embedding, but prediction uses one shared score
        # head instead of per-type output heads.
        self.type_embedding = nn.Embedding(4, type_embed_dim)
        self.query_encoder = nn.Sequential(
            nn.Linear(query_dim + type_embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.score_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

    def _build_positional_encoding(
        self,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build sinusoidal positional encoding. See models/README.md for details."""
        pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, self.embed_dim, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / self.embed_dim)
        )
        pe = torch.zeros((length, self.embed_dim), device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.to(dtype=dtype)

    def _positional_encoding(
        self,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return cached sinusoidal positional encoding for the requested shape."""
        cache = self._positional_encoding_cache
        if (
            cache.ndim != 2
            or cache.shape[0] < length
            or cache.shape[1] != self.embed_dim
            or cache.device != device
            or cache.dtype != dtype
        ):
            cache = self._build_positional_encoding(length, device, dtype)
            self._positional_encoding_cache = cache
        return cache[:length]

    def forward(
        self,
        points: torch.Tensor,
        queries: torch.Tensor,
        query_type_ids: torch.Tensor | None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict one per-point score stream for a pure workload."""
        if query_type_ids is None:
            raise RuntimeError("query_type_ids must be provided for pure-workload prediction.")

        h = self.point_encoder(points)
        h = h + self._positional_encoding(h.shape[1], h.device, h.dtype).unsqueeze(0)
        h = self.local_transformer(h, src_key_padding_mask=padding_mask)

        q_type_emb = self.type_embedding(query_type_ids)
        q_in = torch.cat([queries, q_type_emb], dim=1)
        q_emb = self.query_encoder(q_in).unsqueeze(0).expand(h.shape[0], -1, -1)

        context = chunked_cross_attention_context(
            self.cross_attention,
            point_features=h,
            query_features=q_emb,
            query_chunk_size=self.query_chunk_size,
            point_padding_mask=padding_mask,
        )

        return self.score_head(h + context).squeeze(-1)


def normalize_points_and_queries(
    points: torch.Tensor,
    queries: torch.Tensor,
    point_min: torch.Tensor,
    point_max: torch.Tensor,
    query_min: torch.Tensor,
    query_max: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Min-max normalize points and queries with persisted stats. See models/README.md for details."""
    eps = 1e-6
    normalized_points = (points - point_min) / torch.clamp(point_max - point_min, min=eps)
    normalized_queries = (queries - query_min) / torch.clamp(query_max - query_min, min=eps)
    return normalized_points, normalized_queries
