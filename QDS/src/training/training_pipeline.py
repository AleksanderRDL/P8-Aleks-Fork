"""Training orchestration and checkpoint persistence for AIS-QDS v2. See src/training/README.md for details."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.experiments.experiment_config import ExperimentConfig
from src.experiments.torch_runtime import normalize_amp_mode, torch_autocast_context
from src.models.trajectory_qds_model import TrajectoryQDSModel
from src.models.turn_aware_qds_model import TurnAwareQDSModel
from src.queries.query_types import NUM_QUERY_TYPES
from src.training.scaler import FeatureScaler
from src.training.train_model import TrainingOutputs
from src.training.trajectory_batching import batch_windows, build_trajectory_windows


@dataclass
class ModelArtifacts:
    """Model + scaler checkpoint payload. See src/training/README.md for details."""

    model: TrajectoryQDSModel
    scaler: FeatureScaler
    config: ExperimentConfig
    epochs_trained: int = 0
    train_workload_mix: dict[str, float] | None = None
    eval_workload_mix: dict[str, float] | None = None


def save_checkpoint(path: str, artifacts: ModelArtifacts) -> None:
    """Save model weights, scaler stats, and config to a checkpoint. See src/training/README.md for details."""
    payload = {
        "model_state": artifacts.model.state_dict(),
        "point_dim": artifacts.model.point_dim,
        "query_dim": artifacts.model.query_dim,
        "embed_dim": artifacts.model.embed_dim,
        "query_chunk_size": artifacts.model.query_chunk_size,
        "model_type": artifacts.config.model.model_type,
        "scaler": artifacts.scaler.to_dict(),
        "config": artifacts.config.to_dict(),
        "epochs_trained": int(artifacts.epochs_trained),
        "train_workload_mix": artifacts.train_workload_mix,
        "eval_workload_mix": artifacts.eval_workload_mix,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str) -> ModelArtifacts:
    """Load model weights, scaler stats, and config from checkpoint. See src/training/README.md for details."""
    payload = torch.load(path, map_location="cpu")
    cfg = ExperimentConfig.from_dict(payload["config"])
    model_cls = TurnAwareQDSModel if payload["model_type"] == "turn_aware" else TrajectoryQDSModel
    model = model_cls(
        point_dim=int(payload["point_dim"]),
        query_dim=int(payload["query_dim"]),
        embed_dim=int(payload["embed_dim"]),
        query_chunk_size=int(payload["query_chunk_size"]),
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        type_embed_dim=cfg.model.type_embed_dim,
        dropout=cfg.model.dropout,
    )
    model.load_state_dict(payload["model_state"])
    model.eval()
    scaler = FeatureScaler.from_dict(payload["scaler"])
    return ModelArtifacts(
        model=model,
        scaler=scaler,
        config=cfg,
        epochs_trained=int(payload.get("epochs_trained", 0)),
        train_workload_mix=payload.get("train_workload_mix"),
        eval_workload_mix=payload.get("eval_workload_mix"),
    )


def save_training_summary(path: str, outputs: TrainingOutputs) -> None:
    """Save training diagnostics history to JSON. See src/training/README.md for details."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(outputs.history, f, indent=2)


def _model_device(model: torch.nn.Module) -> torch.device:
    """Return the current device of a model with parameters or buffers."""
    for tensor in model.parameters():
        return tensor.device
    for tensor in model.buffers():
        return tensor.device
    return torch.device("cpu")


def _resolve_predict_device(model: torch.nn.Module, device: torch.device | str | None) -> torch.device:
    """Resolve the inference device for windowed prediction."""
    if device is not None:
        return torch.device(device)
    return _model_device(model)


def windowed_predict(
    model: TrajectoryQDSModel,
    norm_points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    queries: torch.Tensor,
    query_type_ids: torch.Tensor,
    window_length: int = 512,
    window_stride: int = 256,
    batch_size: int = 16,
    device: torch.device | str | None = None,
    amp_mode: str = "off",
) -> torch.Tensor:
    """Run per-window inference and average overlapping predictions. See src/training/README.md for details.

    Using per-window inference ensures the transformer never attends across
    trajectory boundaries, matching the behaviour seen during training.
    When ``device`` is provided, the model and batched windows are evaluated on
    that device and predictions are moved back to ``norm_points.device``.
    """
    output_device = norm_points.device
    predict_device = _resolve_predict_device(model, device)
    amp_mode = normalize_amp_mode(amp_mode)
    original_device = _model_device(model)
    original_training = model.training
    if original_device != predict_device:
        model = model.to(predict_device)

    windows = build_trajectory_windows(norm_points, boundaries, window_length, window_stride)
    windows = batch_windows(windows, max(1, int(batch_size)))
    n = norm_points.shape[0]
    all_pred = torch.zeros((n, NUM_QUERY_TYPES), dtype=norm_points.dtype, device=predict_device)
    pred_count = torch.zeros((n,), dtype=norm_points.dtype, device=predict_device)
    queries_dev = queries.to(predict_device)
    query_type_ids_dev = query_type_ids.to(predict_device)

    try:
        model.eval()
        with torch.no_grad():
            for w in windows:
                points_dev = w.points.to(predict_device)
                padding_dev = w.padding_mask.to(predict_device)
                indices_dev = w.global_indices.to(predict_device)
                with torch_autocast_context(predict_device, amp_mode):
                    wp = model(
                        points=points_dev,
                        queries=queries_dev,
                        query_type_ids=query_type_ids_dev,
                        padding_mask=padding_dev,
                    )
                wp = wp.to(dtype=all_pred.dtype)
                for batch_idx in range(wp.shape[0]):
                    widx = indices_dev[batch_idx]
                    valid = widx >= 0
                    all_pred[widx[valid]] = all_pred[widx[valid]] + wp[batch_idx][valid]
                    pred_count[widx[valid]] = pred_count[widx[valid]] + 1.0

        pred_count = pred_count.clamp(min=1.0)
        return (all_pred / pred_count.unsqueeze(1)).to(output_device)
    finally:
        if original_device != predict_device:
            model.to(original_device)
        model.train(original_training)


def default_inference_device() -> torch.device:
    """Return the default device for saved-model inference."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def forward_predict(
    artifacts: ModelArtifacts,
    points: torch.Tensor,
    queries: torch.Tensor,
    query_type_ids: torch.Tensor,
    boundaries: list[tuple[int, int]] | None = None,
    window_length: int = 512,
    window_stride: int = 256,
    device: torch.device | str | None = None,
    amp_mode: str = "off",
    batch_size: int = 16,
) -> torch.Tensor:
    """Run deterministic predictions with persisted scaler and model. See src/training/README.md for details."""
    p, q = artifacts.scaler.transform(points[:, : artifacts.model.point_dim], queries)
    if boundaries is None:
        boundaries = [(0, p.shape[0])]
    return windowed_predict(
        model=artifacts.model,
        norm_points=p,
        boundaries=boundaries,
        queries=q,
        query_type_ids=query_type_ids,
        window_length=window_length,
        window_stride=window_stride,
        batch_size=batch_size,
        device=device,
        amp_mode=amp_mode,
    )
