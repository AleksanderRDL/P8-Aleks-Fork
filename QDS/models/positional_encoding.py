"""Shared cached sinusoidal positional encoding helpers."""

from __future__ import annotations

import math

import torch


class CachedSinusoidalPositionalEncodingMixin:
    """Mixin for modules with an ``embed_dim`` and positional-encoding cache."""

    embed_dim: int
    _positional_encoding_cache: torch.Tensor

    def _build_positional_encoding(
        self,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build sinusoidal positional encoding."""
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
