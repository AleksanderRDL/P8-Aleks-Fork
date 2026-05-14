"""Inference helpers for persisted AIS-QDS models."""

from __future__ import annotations

import torch

from experiments.torch_runtime import normalize_amp_mode, torch_autocast_context
from models.trajectory_qds_model import TrajectoryQDSModel
from training.checkpoints import ModelArtifacts
from training.trajectory_batching import batch_windows, build_trajectory_windows


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


def _pure_query_type_id(query_type_ids: torch.Tensor) -> int:
    """Return the only query type in a pure workload, rejecting mixed IDs."""
    ids = torch.unique(query_type_ids.detach().cpu())
    if int(ids.numel()) != 1:
        raise ValueError("Pure-workload prediction requires exactly one query type id.")
    return int(ids[0].item())


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
    """Run per-window pure-workload inference and average overlapping predictions."""
    output_device = norm_points.device
    predict_device = _resolve_predict_device(model, device)
    amp_mode = normalize_amp_mode(amp_mode)
    original_device = _model_device(model)
    original_training = model.training
    if original_device != predict_device:
        model = model.to(predict_device)
    _pure_query_type_id(query_type_ids)

    windows = build_trajectory_windows(norm_points, boundaries, window_length, window_stride)
    windows = batch_windows(windows, max(1, int(batch_size)))
    point_count = norm_points.shape[0]
    point_score_sum = torch.zeros((point_count,), dtype=norm_points.dtype, device=predict_device)
    point_score_count = torch.zeros((point_count,), dtype=norm_points.dtype, device=predict_device)
    queries_dev = queries.to(predict_device)
    query_type_ids_dev = query_type_ids.to(predict_device)

    try:
        model.eval()
        with torch.no_grad():
            for window_batch in windows:
                points_dev = window_batch.points.to(predict_device)
                padding_dev = window_batch.padding_mask.to(predict_device)
                indices_dev = window_batch.global_indices.to(predict_device)
                with torch_autocast_context(predict_device, amp_mode):
                    window_scores = model(
                        points=points_dev,
                        queries=queries_dev,
                        query_type_ids=query_type_ids_dev,
                        padding_mask=padding_dev,
                    )
                window_scores = window_scores.to(dtype=point_score_sum.dtype)
                for batch_idx in range(window_scores.shape[0]):
                    point_indices = indices_dev[batch_idx]
                    valid_points = point_indices >= 0
                    point_score_sum[point_indices[valid_points]] += window_scores[batch_idx, valid_points]
                    point_score_count[point_indices[valid_points]] += 1.0

        point_score_count = point_score_count.clamp(min=1.0)
        return (point_score_sum / point_score_count).to(output_device)
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
    """Run deterministic predictions with persisted scaler and model."""
    norm_points, norm_queries = artifacts.scaler.transform(points[:, : artifacts.model.point_dim], queries)
    if boundaries is None:
        boundaries = [(0, norm_points.shape[0])]
    return windowed_predict(
        model=artifacts.model,
        norm_points=norm_points,
        boundaries=boundaries,
        queries=norm_queries,
        query_type_ids=query_type_ids,
        window_length=window_length,
        window_stride=window_stride,
        batch_size=batch_size,
        device=device,
        amp_mode=amp_mode,
    )
