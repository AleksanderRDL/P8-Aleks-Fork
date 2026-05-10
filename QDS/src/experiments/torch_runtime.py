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


def reset_cuda_peak_memory_stats() -> dict[str, Any]:
    """Reset CUDA peak memory stats for the active device when CUDA is available."""
    if not torch.cuda.is_available():
        return {"available": False}
    device = torch.cuda.current_device()
    torch.cuda.reset_peak_memory_stats(device)
    return {"available": True, "device_index": int(device)}


def cuda_memory_snapshot() -> dict[str, Any]:
    """Return current and peak CUDA memory stats in MiB for the active device."""
    if not torch.cuda.is_available():
        return {"available": False}
    device = torch.cuda.current_device()
    torch.cuda.synchronize(device)
    mib = 1024.0 * 1024.0
    return {
        "available": True,
        "device_index": int(device),
        "allocated_mb": float(torch.cuda.memory_allocated(device) / mib),
        "reserved_mb": float(torch.cuda.memory_reserved(device) / mib),
        "max_allocated_mb": float(torch.cuda.max_memory_allocated(device) / mib),
        "max_reserved_mb": float(torch.cuda.max_memory_reserved(device) / mib),
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
