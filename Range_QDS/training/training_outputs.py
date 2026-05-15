"""Shared training output payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from training.scaler import FeatureScaler


@dataclass
class TrainingOutputs:
    """Training artifact container. See training/README.md for details."""

    model: torch.nn.Module
    scaler: FeatureScaler
    labels: torch.Tensor
    labelled_mask: torch.Tensor
    history: list[dict[str, float]]
    epochs_trained: int = 0
    best_epoch: int = 0
    best_loss: float = float("inf")
    best_selection_score: float = 0.0
    target_diagnostics: dict[str, Any] = field(default_factory=dict)
    fit_diagnostics: dict[str, Any] = field(default_factory=dict)
    feature_context: dict[str, Any] = field(default_factory=dict)
