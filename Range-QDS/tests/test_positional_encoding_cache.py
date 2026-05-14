"""Tests cached positional encodings in the trajectory model."""

from __future__ import annotations

import torch

from models.trajectory_qds_model import TrajectoryQDSModel


def test_positional_encoding_cache_reuses_matching_buffer() -> None:
    model = TrajectoryQDSModel(point_dim=7, query_dim=12, embed_dim=16, num_heads=2, num_layers=1)
    device = torch.device("cpu")

    first = model._positional_encoding(32, device, torch.float32)
    first_ptr = model._positional_encoding_cache.data_ptr()
    shorter = model._positional_encoding(16, device, torch.float32)

    assert first.shape == (32, 16)
    assert shorter.shape == (16, 16)
    assert model._positional_encoding_cache.data_ptr() == first_ptr
    assert torch.allclose(shorter, first[:16])


def test_positional_encoding_cache_rebuilds_for_dtype() -> None:
    model = TrajectoryQDSModel(point_dim=7, query_dim=12, embed_dim=16, num_heads=2, num_layers=1)
    device = torch.device("cpu")

    model._positional_encoding(16, device, torch.float32)
    encoded = model._positional_encoding(16, device, torch.float64)

    assert encoded.dtype == torch.float64
    assert model._positional_encoding_cache.dtype == torch.float64
