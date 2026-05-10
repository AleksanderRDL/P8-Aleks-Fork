"""Tests for torch runtime precision controls."""

from __future__ import annotations

import torch

from src.experiments.experiment_config import ExperimentConfig, build_experiment_config
from src.experiments.torch_runtime import apply_torch_runtime_settings


def test_apply_torch_runtime_settings_sets_precision_and_tf32() -> None:
    old_precision = torch.get_float32_matmul_precision()
    old_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    try:
        snapshot = apply_torch_runtime_settings(float32_matmul_precision="high", allow_tf32=True)

        assert snapshot["float32_matmul_precision"] == "high"
        assert snapshot["tf32_matmul_allowed"] is True
        assert torch.get_float32_matmul_precision() == "high"
        assert bool(torch.backends.cuda.matmul.allow_tf32) is True
    finally:
        torch.set_float32_matmul_precision(old_precision)
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


def test_experiment_config_roundtrips_precision_controls() -> None:
    cfg = build_experiment_config(float32_matmul_precision="high", allow_tf32=True)
    restored = ExperimentConfig.from_dict(cfg.to_dict())

    assert restored.model.float32_matmul_precision == "high"
    assert restored.model.allow_tf32 is True


def test_experiment_config_loads_legacy_precision_defaults() -> None:
    payload = build_experiment_config().to_dict()
    payload["model"].pop("float32_matmul_precision")
    payload["model"].pop("allow_tf32")

    restored = ExperimentConfig.from_dict(payload)

    assert restored.model.float32_matmul_precision == "highest"
    assert restored.model.allow_tf32 is False
