"""Tests for torch runtime precision controls."""

from __future__ import annotations

import torch

from src.experiments.benchmark_runtime import (
    _batch_size_sweep_summary,
    _parse_train_batch_sizes,
    _runtime_child_args,
)
from src.experiments.experiment_config import ExperimentConfig, build_experiment_config
from src.experiments.torch_runtime import (
    amp_runtime_snapshot,
    apply_torch_runtime_settings,
    normalize_amp_mode,
    torch_autocast_context,
)


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
    cfg = build_experiment_config(
        float32_matmul_precision="high",
        allow_tf32=True,
        train_batch_size=64,
        amp_mode="bf16",
    )
    restored = ExperimentConfig.from_dict(cfg.to_dict())

    assert restored.model.float32_matmul_precision == "high"
    assert restored.model.allow_tf32 is True
    assert restored.model.train_batch_size == 64
    assert restored.model.amp_mode == "bf16"


def test_experiment_config_loads_legacy_precision_defaults() -> None:
    payload = build_experiment_config().to_dict()
    payload["model"].pop("float32_matmul_precision")
    payload["model"].pop("allow_tf32")
    payload["model"].pop("amp_mode")

    restored = ExperimentConfig.from_dict(payload)

    assert restored.model.float32_matmul_precision == "highest"
    assert restored.model.allow_tf32 is False
    assert restored.model.amp_mode == "off"


def test_amp_helpers_default_to_cuda_only_autocast() -> None:
    assert normalize_amp_mode(None) == "off"
    assert normalize_amp_mode(" BF16 ") == "bf16"

    cpu_snapshot = amp_runtime_snapshot("bf16", device="cpu")

    assert cpu_snapshot == {
        "mode": "bf16",
        "enabled": False,
        "device_type": "cpu",
        "dtype": "bfloat16",
    }
    with torch_autocast_context("cpu", "bf16"):
        value = torch.ones((2,), dtype=torch.float32) + 1.0
    assert value.dtype == torch.float32


def test_runtime_child_args_forward_amp_mode() -> None:
    assert _runtime_child_args("high", True, "bf16") == [
        "--float32_matmul_precision",
        "high",
        "--allow_tf32",
        "--amp_mode",
        "bf16",
    ]


def test_parse_train_batch_sizes() -> None:
    assert _parse_train_batch_sizes("16, 32,64") == [16, 32, 64]
    assert _parse_train_batch_sizes(None) is None


def test_batch_size_sweep_summary_extracts_timing_memory_and_f1() -> None:
    rows = _batch_size_sweep_summary(
        [
            {
                "name": "train_bs32",
                "train_batch_size": 32,
                "returncode": 0,
                "elapsed_seconds": 12.5,
                "timings": {"epoch_timings": [{"seconds": 2.0}, {"seconds": 3.0}]},
                "metrics": {
                    "best_f1": 0.4,
                    "batch_size": {"train_batch_size": 32},
                    "cuda_memory": {
                        "training": {
                            "max_allocated_mb": 123.0,
                            "max_reserved_mb": 256.0,
                        }
                    },
                    "methods": {"MLQDS": {"aggregate_f1": 0.5}},
                },
            }
        ]
    )

    assert rows == [
        {
            "train_batch_size": 32,
            "returncode": 0,
            "elapsed_seconds": 12.5,
            "epoch_time_mean_seconds": 2.5,
            "epoch_time_min_seconds": 2.0,
            "epoch_time_max_seconds": 3.0,
            "peak_allocated_mb": 123.0,
            "peak_reserved_mb": 256.0,
            "best_f1": 0.4,
            "mlqds_aggregate_f1": 0.5,
        }
    ]
