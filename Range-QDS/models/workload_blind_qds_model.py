"""Workload-blind trajectory scorer for range-prior compression."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.positional_encoding import CachedSinusoidalPositionalEncodingMixin


class WorkloadBlindRangeQDSModel(CachedSinusoidalPositionalEncodingMixin, nn.Module):
    """Point-only transformer scorer.

    Training may use query-derived labels, but forward inference deliberately
    ignores query tensors so one retained mask can be built before future
    range queries are known.
    """

    workload_blind = True

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
        del type_embed_dim
        self.point_dim = int(point_dim)
        self.query_dim = int(query_dim)
        self.embed_dim = int(embed_dim)
        self.query_chunk_size = int(query_chunk_size)
        self.num_layers = max(0, int(num_layers))
        self.register_buffer("_positional_encoding_cache", torch.empty(0), persistent=False)

        self.point_encoder = nn.Sequential(
            nn.Linear(self.point_dim, self.embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        if self.num_layers > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=num_heads,
                dim_feedforward=self.embed_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.local_transformer: nn.TransformerEncoder | None = nn.TransformerEncoder(
                enc_layer,
                num_layers=self.num_layers,
            )
        else:
            self.local_transformer = None

        self.score_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim // 2, 1),
        )

    def forward(
        self,
        points: torch.Tensor,
        queries: torch.Tensor | None = None,
        query_type_ids: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict one query-independent per-point score stream."""
        del queries, query_type_ids
        h = self.point_encoder(points)
        if self.local_transformer is not None:
            h = h + self._positional_encoding(h.shape[1], h.device, h.dtype).unsqueeze(0)
            h = self.local_transformer(h, src_key_padding_mask=padding_mask)
        return self.score_head(h).squeeze(-1)


class SegmentContextRangeQDSModel(CachedSinusoidalPositionalEncodingMixin, nn.Module):
    """Query-free point scorer with trajectory segment context.

    This is a lightweight MLSimp-inspired architecture: point states are
    enriched with fixed trajectory-order segment summaries, plus explicit
    local uniqueness and trajectory-global similarity signals. It remains
    workload-blind because forward inference ignores query tensors.
    """

    workload_blind = True

    def __init__(
        self,
        point_dim: int,
        query_dim: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        type_embed_dim: int = 16,
        query_chunk_size: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        del type_embed_dim
        self.point_dim = int(point_dim)
        self.query_dim = int(query_dim)
        self.embed_dim = int(embed_dim)
        self.query_chunk_size = int(query_chunk_size)
        self.num_layers = max(0, int(num_layers))
        self.segment_count = 8
        self.register_buffer("_positional_encoding_cache", torch.empty(0), persistent=False)

        self.point_encoder = nn.Sequential(
            nn.Linear(self.point_dim, self.embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        if self.num_layers > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=num_heads,
                dim_feedforward=self.embed_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.local_transformer: nn.TransformerEncoder | None = nn.TransformerEncoder(
                enc_layer,
                num_layers=self.num_layers,
            )
        else:
            self.local_transformer = None

        self.segment_attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.score_head = nn.Sequential(
            nn.Linear(self.embed_dim * 3 + 2, self.embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Linear(self.embed_dim // 2, 1),
        )

    def _valid_mask(self, h: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        """Return [batch, length] valid-position mask."""
        if padding_mask is None:
            return torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
        return ~padding_mask.to(device=h.device, dtype=torch.bool)

    def _segment_tokens(self, h: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Pool point states into fixed trajectory-order segment summaries."""
        batch, length, _embed = h.shape
        segment_count = min(self.segment_count, max(1, int(length)))
        positions = torch.arange(length, device=h.device)
        segment_ids = torch.clamp((positions * segment_count) // max(1, length), max=segment_count - 1)
        valid_f = valid_mask.to(dtype=h.dtype)
        global_mean = (h * valid_f.unsqueeze(-1)).sum(dim=1) / valid_f.sum(dim=1).clamp(min=1.0).unsqueeze(-1)

        segments: list[torch.Tensor] = []
        for segment_id in range(segment_count):
            local = (segment_ids == segment_id).unsqueeze(0) & valid_mask
            local_f = local.to(dtype=h.dtype)
            count = local_f.sum(dim=1)
            pooled = (h * local_f.unsqueeze(-1)).sum(dim=1) / count.clamp(min=1.0).unsqueeze(-1)
            pooled = torch.where((count > 0).unsqueeze(-1), pooled, global_mean)
            segments.append(pooled)
        return torch.stack(segments, dim=1)

    def _structural_scalars(
        self,
        h: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return local uniqueness and globality scalars per point."""
        valid_f = valid_mask.to(dtype=h.dtype)
        global_mean = (h * valid_f.unsqueeze(-1)).sum(dim=1) / valid_f.sum(dim=1).clamp(min=1.0).unsqueeze(-1)

        prev_h = torch.cat([h[:, :1], h[:, :-1]], dim=1)
        next_h = torch.cat([h[:, 1:], h[:, -1:]], dim=1)
        prev_valid = torch.cat([torch.zeros_like(valid_mask[:, :1]), valid_mask[:, :-1]], dim=1)
        next_valid = torch.cat([valid_mask[:, 1:], torch.zeros_like(valid_mask[:, -1:])], dim=1)
        neighbor_weight = prev_valid.to(dtype=h.dtype) + next_valid.to(dtype=h.dtype)
        neighbor_sum = (
            prev_h * prev_valid.to(dtype=h.dtype).unsqueeze(-1)
            + next_h * next_valid.to(dtype=h.dtype).unsqueeze(-1)
        )
        neighbor_mean = neighbor_sum / neighbor_weight.clamp(min=1.0).unsqueeze(-1)
        neighbor_mean = torch.where((neighbor_weight > 0).unsqueeze(-1), neighbor_mean, h)

        uniqueness = torch.linalg.vector_norm(h - neighbor_mean, dim=-1, keepdim=True)
        globality = F.cosine_similarity(h, global_mean.unsqueeze(1), dim=-1).unsqueeze(-1)
        return uniqueness, globality

    def forward(
        self,
        points: torch.Tensor,
        queries: torch.Tensor | None = None,
        query_type_ids: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict query-independent per-point scores using segment context."""
        del queries, query_type_ids
        h = self.point_encoder(points)
        valid_mask = self._valid_mask(h, padding_mask)
        if self.local_transformer is not None:
            h = h + self._positional_encoding(h.shape[1], h.device, h.dtype).unsqueeze(0)
            h = self.local_transformer(h, src_key_padding_mask=padding_mask)
        segment_tokens = self._segment_tokens(h, valid_mask)
        segment_context, _weights = self.segment_attention(
            query=h,
            key=segment_tokens,
            value=segment_tokens,
            need_weights=False,
        )
        uniqueness, globality = self._structural_scalars(h, valid_mask)
        score_input = torch.cat(
            [
                h,
                segment_context,
                h - segment_context,
                uniqueness,
                globality,
            ],
            dim=-1,
        )
        return self.score_head(score_input).squeeze(-1)
