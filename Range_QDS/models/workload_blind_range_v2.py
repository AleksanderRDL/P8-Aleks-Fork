"""Factorized workload-blind range model for QueryUsefulV1."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.positional_encoding import CachedSinusoidalPositionalEncodingMixin
from training.query_prior_fields import QUERY_PRIOR_FIELD_NAMES
from training.query_useful_targets import QUERY_USEFUL_V1_HEAD_NAMES

WORKLOAD_BLIND_RANGE_V2_SCHEMA_VERSION = 6
WORKLOAD_BLIND_RANGE_V2_PRIOR_FEATURE_DIM = len(QUERY_PRIOR_FIELD_NAMES)
_LEGACY_CALIBRATION_HEAD_INPUT_DIM = 5
_PRIOR_FEATURE_SCALE_INIT = 0.25


class WorkloadBlindRangeV2Model(CachedSinusoidalPositionalEncodingMixin, nn.Module):
    """Small trainable factorized scorer for query-driven blind simplification."""

    workload_blind = True
    factorized_query_useful_v1 = True

    def __init__(
        self,
        point_dim: int,
        query_dim: int,
        embed_dim: int = 128,
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
        self.prior_feature_dim = min(WORKLOAD_BLIND_RANGE_V2_PRIOR_FEATURE_DIM, self.point_dim)
        self.head_names = tuple(QUERY_USEFUL_V1_HEAD_NAMES)
        self.register_buffer("_positional_encoding_cache", torch.empty(0), persistent=False)

        self.point_encoder = nn.Sequential(
            nn.Linear(self.point_dim, self.embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
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
            self.local_context_encoder: nn.TransformerEncoder | None = nn.TransformerEncoder(
                enc_layer,
                num_layers=self.num_layers,
            )
        else:
            self.local_context_encoder = None
        self.segment_context = nn.Conv1d(
            in_channels=self.embed_dim,
            out_channels=self.embed_dim,
            kernel_size=5,
            padding=2,
            groups=1,
        )
        self.prior_encoder = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(self.embed_dim, self.embed_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.embed_dim // 2, 1),
                )
                for name in self.head_names
            }
        )
        self.calibration_head = nn.Sequential(
            nn.Linear(_LEGACY_CALIBRATION_HEAD_INPUT_DIM, self.embed_dim // 4),
            nn.GELU(),
            nn.Linear(self.embed_dim // 4, 1),
        )
        # Retained for legacy checkpoint compatibility. The final score now
        # uses only the factorized interpretable composition so head ablations
        # cannot route around the disabled head through calibration.
        for parameter in self.calibration_head.parameters():
            parameter.requires_grad_(False)
        self.prior_feature_encoder = nn.Sequential(
            nn.Linear(self.prior_feature_dim, self.embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        prior_output = self.prior_feature_encoder[-1]
        if isinstance(prior_output, nn.Linear):
            nn.init.xavier_uniform_(prior_output.weight)
            nn.init.zeros_(prior_output.bias)
        self.prior_feature_scale = nn.Parameter(torch.tensor(_PRIOR_FEATURE_SCALE_INIT, dtype=torch.float32))

    def reset_parameters(self) -> None:
        """Reset standalone trainable parameters not owned by child modules."""
        with torch.no_grad():
            self.prior_feature_scale.fill_(_PRIOR_FEATURE_SCALE_INIT)

    def final_logit_from_head_logits(
        self,
        head_logits: torch.Tensor,
        *,
        disabled_head_names: tuple[str, ...] = (),
    ) -> torch.Tensor:
        """Compose final logits from factorized heads, with optional neutralized heads."""
        if head_logits.shape[-1] != len(self.head_names):
            raise ValueError(
                "head_logits last dimension must match factorized head count: "
                f"got {int(head_logits.shape[-1])}, expected {len(self.head_names)}."
            )
        disabled = {str(name) for name in disabled_head_names}
        q_hit = torch.sigmoid(head_logits[..., 0])
        behavior = torch.sigmoid(head_logits[..., 1])
        boundary = torch.sigmoid(head_logits[..., 2])
        replacement = torch.sigmoid(head_logits[..., 3])
        if "query_hit_probability" in disabled:
            q_hit = torch.full_like(q_hit, 0.5)
        if "conditional_behavior_utility" in disabled:
            behavior = torch.zeros_like(behavior)
        if "boundary_event_utility" in disabled:
            boundary = torch.zeros_like(boundary)
        if "replacement_representative_value" in disabled or "marginal_replacement_gain" in disabled:
            replacement = torch.full_like(replacement, 0.5)
        replacement_multiplier = 0.75 + 0.25 * replacement
        interpretable_score = q_hit * (0.5 + behavior) * replacement_multiplier + 0.25 * boundary
        return torch.logit(interpretable_score.clamp(1e-5, 1.0 - 1e-5))

    def _encoded(
        self,
        points: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Encode point, local, and segment context."""
        h = self.point_encoder(points)
        prior_features = self._prior_features(points)
        h = h + self.prior_feature_scale * self.prior_feature_encoder(prior_features)
        if self.local_context_encoder is not None:
            h = h + self._positional_encoding(h.shape[1], h.device, h.dtype).unsqueeze(0)
            h = self.local_context_encoder(h, src_key_padding_mask=padding_mask)
        segment = self.segment_context(h.transpose(1, 2)).transpose(1, 2)
        return self.prior_encoder(h + segment)

    def _prior_features(self, points: torch.Tensor) -> torch.Tensor:
        """Return normalized query-prior feature channels from the v2 point tensor."""
        return points[..., -self.prior_feature_dim :].float()

    def forward_with_heads(
        self,
        points: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return final score logits and factorized head logits."""
        h = self._encoded(points, padding_mask)
        head_logits = torch.cat([self.heads[name](h) for name in self.head_names], dim=-1)
        final_logit = self.final_logit_from_head_logits(head_logits)
        return final_logit, head_logits

    def forward(
        self,
        points: torch.Tensor,
        queries: torch.Tensor | None = None,
        query_type_ids: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict query-independent per-point final score logits."""
        del queries, query_type_ids
        final_logit, _head_logits = self.forward_with_heads(points, padding_mask=padding_mask)
        return final_logit
