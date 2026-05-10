"""Torch runtime precision controls for experiment entrypoints."""

from __future__ import annotations

from typing import Any

import torch

FLOAT32_MATMUL_PRECISION_CHOICES = ("highest", "high", "medium")


def _normalize_float32_matmul_precision(value: str) -> str:
    """Validate a torch float32 matmul precision setting."""
    precision = str(value).strip().lower()
    if precision not in FLOAT32_MATMUL_PRECISION_CHOICES:
        choices = ", ".join(FLOAT32_MATMUL_PRECISION_CHOICES)
        raise ValueError(f"float32_matmul_precision must be one of: {choices}.")
    return precision


def torch_runtime_snapshot() -> dict[str, Any]:
    """Return the currently active torch precision settings."""
    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "tf32_matmul_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn_allowed": bool(torch.backends.cudnn.allow_tf32),
    }


def apply_torch_runtime_settings(
    *,
    float32_matmul_precision: str = "highest",
    allow_tf32: bool = False,
) -> dict[str, Any]:
    """Apply process-local torch precision settings and return the effective values."""
    precision = _normalize_float32_matmul_precision(float32_matmul_precision)
    torch.set_float32_matmul_precision(precision)
    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    snapshot = torch_runtime_snapshot()
    snapshot["requested_float32_matmul_precision"] = precision
    snapshot["requested_tf32_matmul_allowed"] = bool(allow_tf32)
    return snapshot
