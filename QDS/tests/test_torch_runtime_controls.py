"""Tests for torch runtime precision controls."""

from __future__ import annotations

import torch

from src.experiments.benchmark_runtime import (
    DEFAULT_PROFILE,
    _batch_size_sweep_summary,
    _extra_args_include_training_data_source,
    _parse_train_batch_sizes,
    _profile_train_args,
    _runtime_child_args,
)
from src.experiments.experiment_cli import build_parser
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
        train_csv_path="train.csv",
        validation_csv_path="validation.csv",
        eval_csv_path="eval.csv",
        float32_matmul_precision="high",
        allow_tf32=True,
        train_batch_size=64,
        inference_batch_size=32,
        query_chunk_size=512,
        amp_mode="bf16",
        range_boundary_prior_weight=1.0,
        range_label_mode="point_f1",
        loss_objective="ranking_bce",
        budget_loss_ratios=[0.02, 0.05],
        budget_loss_temperature=0.2,
        ranking_pairs_per_type=192,
        ranking_top_quantile=0.9,
        range_spatial_km=2.2,
        range_time_hours=6.0,
        mlqds_score_mode="rank_confidence",
        mlqds_score_temperature=0.5,
        mlqds_rank_confidence_weight=0.3,
        range_audit_compression_ratios=[0.01, 0.05],
    )
    restored = ExperimentConfig.from_dict(cfg.to_dict())

    assert restored.model.float32_matmul_precision == "high"
    assert restored.data.train_csv_path == "train.csv"
    assert restored.data.validation_csv_path == "validation.csv"
    assert restored.data.eval_csv_path == "eval.csv"
    assert restored.model.allow_tf32 is True
    assert restored.model.train_batch_size == 64
    assert restored.model.inference_batch_size == 32
    assert restored.model.query_chunk_size == 512
    assert restored.model.amp_mode == "bf16"
    assert restored.model.range_boundary_prior_weight == 1.0
    assert restored.model.range_label_mode == "point_f1"
    assert restored.model.loss_objective == "ranking_bce"
    assert restored.model.budget_loss_ratios == [0.02, 0.05]
    assert restored.model.budget_loss_temperature == 0.2
    assert restored.model.ranking_pairs_per_type == 192
    assert restored.model.ranking_top_quantile == 0.9
    assert restored.query.range_spatial_km == 2.2
    assert restored.query.range_time_hours == 6.0
    assert restored.model.mlqds_score_mode == "rank_confidence"
    assert restored.model.mlqds_score_temperature == 0.5
    assert restored.model.mlqds_rank_confidence_weight == 0.3
    assert restored.model.range_audit_compression_ratios == [0.01, 0.05]
    assert restored.model.checkpoint_selection_metric == "f1"


def test_cli_exposes_training_and_scoring_tuning_controls() -> None:
    args = build_parser().parse_args(
        [
            "--ranking_pairs_per_type",
            "64",
            "--ranking_top_quantile",
            "0.70",
            "--mlqds_score_mode",
            "rank_confidence",
            "--mlqds_score_temperature",
            "0.50",
            "--mlqds_rank_confidence_weight",
            "0.30",
            "--range_audit_compression_ratios",
            "0.01,0.02,0.10",
            "--range_label_mode",
            "point_f1",
            "--loss_objective",
            "budget_topk",
            "--budget_loss_ratios",
            "0.01,0.05",
            "--budget_loss_temperature",
            "0.20",
            "--validation_csv_path",
            "validation.csv",
        ]
    )

    cfg = build_experiment_config(
        ranking_pairs_per_type=args.ranking_pairs_per_type,
        ranking_top_quantile=args.ranking_top_quantile,
        mlqds_score_mode=args.mlqds_score_mode,
        mlqds_score_temperature=args.mlqds_score_temperature,
        mlqds_rank_confidence_weight=args.mlqds_rank_confidence_weight,
        range_audit_compression_ratios=args.range_audit_compression_ratios,
        range_label_mode=args.range_label_mode,
        loss_objective=args.loss_objective,
        budget_loss_ratios=args.budget_loss_ratios,
        budget_loss_temperature=args.budget_loss_temperature,
        validation_csv_path=args.validation_csv_path,
    )

    assert args.ranking_pairs_per_type == 64
    assert args.ranking_top_quantile == 0.70
    assert args.mlqds_score_mode == "rank_confidence"
    assert args.mlqds_score_temperature == 0.50
    assert args.mlqds_rank_confidence_weight == 0.30
    assert args.range_audit_compression_ratios == [0.01, 0.02, 0.10]
    assert args.range_label_mode == "point_f1"
    assert args.loss_objective == "budget_topk"
    assert args.budget_loss_ratios == [0.01, 0.05]
    assert args.budget_loss_temperature == 0.20
    assert args.validation_csv_path == "validation.csv"
    assert cfg.model.ranking_pairs_per_type == 64
    assert cfg.model.ranking_top_quantile == 0.70
    assert cfg.model.mlqds_score_mode == "rank_confidence"
    assert cfg.model.mlqds_score_temperature == 0.50
    assert cfg.model.mlqds_rank_confidence_weight == 0.30
    assert cfg.model.range_audit_compression_ratios == [0.01, 0.02, 0.10]
    assert cfg.model.range_label_mode == "point_f1"
    assert cfg.model.loss_objective == "budget_topk"
    assert cfg.model.budget_loss_ratios == [0.01, 0.05]
    assert cfg.model.budget_loss_temperature == 0.20
    assert cfg.data.validation_csv_path == "validation.csv"


def test_experiment_config_loads_missing_runtime_and_mlqds_defaults() -> None:
    payload = build_experiment_config().to_dict()
    payload["model"].pop("float32_matmul_precision")
    payload["model"].pop("allow_tf32")
    payload["model"].pop("inference_batch_size")
    payload["model"].pop("amp_mode")
    payload["model"].pop("range_boundary_prior_weight")
    payload["model"].pop("range_label_mode")
    payload["model"].pop("loss_objective")
    payload["model"].pop("budget_loss_ratios")
    payload["model"].pop("budget_loss_temperature")
    payload["model"].pop("range_audit_compression_ratios")
    payload["model"].pop("mlqds_score_mode")
    payload["model"].pop("mlqds_score_temperature")
    payload["model"].pop("mlqds_rank_confidence_weight")

    restored = ExperimentConfig.from_dict(payload)

    assert restored.model.float32_matmul_precision == "highest"
    assert restored.model.allow_tf32 is False
    assert restored.model.inference_batch_size == 16
    assert restored.model.amp_mode == "off"
    assert restored.model.range_boundary_prior_weight == 0.0
    assert restored.model.range_label_mode == "usefulness"
    assert restored.model.loss_objective == "budget_topk"
    assert restored.model.budget_loss_ratios == [0.01, 0.02, 0.05, 0.10]
    assert restored.model.budget_loss_temperature == 0.10
    assert restored.model.range_audit_compression_ratios == []
    assert restored.model.mlqds_score_mode == "rank"
    assert restored.model.mlqds_score_temperature == 1.0
    assert restored.model.mlqds_rank_confidence_weight == 0.15
    assert restored.model.checkpoint_selection_metric == "f1"
    assert restored.model.checkpoint_f1_variant == "range_usefulness"


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


def test_runtime_profile_uses_real_usecase_shape(tmp_path) -> None:
    args = _profile_train_args(DEFAULT_PROFILE, seed=42, results_dir=tmp_path / "run", checkpoint=tmp_path / "m.pt")

    assert "--n_queries" in args
    assert args[args.index("--n_queries") + 1] == "80"
    assert args[args.index("--max_queries") + 1] == "2048"
    assert args[args.index("--compression_ratio") + 1] == "0.05"
    assert args[args.index("--query_chunk_size") + 1] == "2048"
    assert args[args.index("--query_coverage") + 1] == "0.20"
    assert args[args.index("--range_spatial_km") + 1] == "2.2"
    assert args[args.index("--range_time_hours") + 1] == "5.0"
    assert args[args.index("--range_footprint_jitter") + 1] == "0.0"
    assert args[args.index("--early_stopping_patience") + 1] == "5"
    assert args[args.index("--f1_diagnostic_every") + 1] == "1"
    assert args[args.index("--checkpoint_smoothing_window") + 1] == "1"
    assert args[args.index("--loss_objective") + 1] == "budget_topk"
    assert args[args.index("--budget_loss_ratios") + 1] == "0.01,0.02,0.05,0.10"
    assert args[args.index("--budget_loss_temperature") + 1] == "0.10"
    assert args[args.index("--mlqds_temporal_fraction") + 1] == "0.50"
    assert args[args.index("--mlqds_score_mode") + 1] == "rank"
    assert args[args.index("--mlqds_score_temperature") + 1] == "1.00"
    assert args[args.index("--mlqds_rank_confidence_weight") + 1] == "0.15"
    assert args[args.index("--mlqds_diversity_bonus") + 1] == "0.00"
    assert args[args.index("--residual_label_mode") + 1] == "temporal"
    assert args[args.index("--range_label_mode") + 1] == "usefulness"
    assert args[args.index("--range_boundary_prior_weight") + 1] == "0.0"
    assert "--n_ships" not in args
    assert "--n_points" not in args


def test_runtime_profile_requires_real_training_data_source() -> None:
    assert _extra_args_include_training_data_source("--csv_path ../AISDATA/cleaned/day.csv")
    assert _extra_args_include_training_data_source(
        "--train_csv_path=train.csv --validation_csv_path validation.csv --eval_csv_path eval.csv"
    )
    assert not _extra_args_include_training_data_source("--max_segments 10")
    assert not _extra_args_include_training_data_source("--validation_csv_path validation.csv")


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
            "best_selection_score": 0.4,
            "best_f1": 0.4,
            "mlqds_aggregate_f1": 0.5,
            "mlqds_range_usefulness_score": None,
        }
    ]
