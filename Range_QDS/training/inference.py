"""Inference helpers for persisted AIS-QDS models."""

from __future__ import annotations

from typing import Callable, cast

import torch

from experiments.torch_runtime import normalize_amp_mode, torch_autocast_context
from training.checkpoints import ModelArtifacts
from training.model_features import (
    build_query_free_point_features_for_dim,
)
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


def _is_workload_blind_model(model: torch.nn.Module) -> bool:
    """Return whether the model forward path ignores query tensors."""
    return bool(getattr(model, "workload_blind", False))


def _model_point_dim(model: torch.nn.Module) -> int:
    """Return a model point dimension with a clear error for invalid artifacts."""
    point_dim = getattr(model, "point_dim", None)
    if point_dim is None:
        raise AttributeError(f"{type(model).__name__} must expose point_dim for AIS-QDS inference.")
    return int(point_dim)


def windowed_predict_with_heads(
    model: torch.nn.Module,
    norm_points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    queries: torch.Tensor | None,
    query_type_ids: torch.Tensor | None,
    window_length: int = 512,
    window_stride: int = 256,
    batch_size: int = 16,
    device: torch.device | str | None = None,
    amp_mode: str = "off",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run windowed inference and return optional factorized head logits."""
    output_device = norm_points.device
    predict_device = _resolve_predict_device(model, device)
    amp_mode = normalize_amp_mode(amp_mode)
    original_device = _model_device(model)
    original_training = model.training
    if original_device != predict_device:
        model = model.to(predict_device)
    workload_blind = _is_workload_blind_model(model)
    forward_with_heads = getattr(model, "forward_with_heads", None)
    factorized_forward = (
        cast(Callable[..., tuple[torch.Tensor, torch.Tensor]], forward_with_heads)
        if workload_blind and callable(forward_with_heads)
        else None
    )
    if not workload_blind:
        if queries is None or query_type_ids is None:
            raise RuntimeError("query-aware prediction requires queries and query_type_ids.")
        _pure_query_type_id(query_type_ids)

    if workload_blind and bool(getattr(model, "window_independent", False)):
        try:
            model.eval()
            with torch.no_grad():
                points_dev = norm_points.to(predict_device).unsqueeze(0)
                with torch_autocast_context(predict_device, amp_mode):
                    if factorized_forward is not None:
                        point_scores, head_logits = factorized_forward(
                            points=points_dev,
                            padding_mask=None,
                        )
                    else:
                        point_scores = model(
                            points=points_dev,
                            queries=None,
                            query_type_ids=None,
                            padding_mask=None,
                        )
                        head_logits = None
            if point_scores.ndim != 2 or int(point_scores.shape[0]) != 1:
                raise ValueError(
                    "window_independent models must return scores with shape [1, n_points]; "
                    f"got {tuple(point_scores.shape)}."
                )
            heads = None
            if head_logits is not None:
                if head_logits.ndim != 3 or int(head_logits.shape[0]) != 1:
                    raise ValueError(
                        "factorized window_independent models must return head logits with shape "
                        f"[1, n_points, n_heads]; got {tuple(head_logits.shape)}."
                    )
                heads = head_logits.reshape(int(head_logits.shape[1]), int(head_logits.shape[2])).to(output_device)
            return point_scores.reshape(-1).to(output_device), heads
        finally:
            if original_device != predict_device:
                model.to(original_device)
            model.train(original_training)

    windows = build_trajectory_windows(norm_points, boundaries, window_length, window_stride)
    windows = batch_windows(windows, max(1, int(batch_size)))
    point_count = norm_points.shape[0]
    point_score_sum = torch.zeros((point_count,), dtype=norm_points.dtype, device=predict_device)
    point_score_count = torch.zeros((point_count,), dtype=norm_points.dtype, device=predict_device)
    head_logit_sum: torch.Tensor | None = None
    head_logit_count: torch.Tensor | None = None
    queries_dev = None if queries is None else queries.to(predict_device)
    query_type_ids_dev = None if query_type_ids is None else query_type_ids.to(predict_device)

    try:
        model.eval()
        with torch.no_grad():
            for window_batch in windows:
                points_dev = window_batch.points.to(predict_device)
                padding_dev = window_batch.padding_mask.to(predict_device)
                indices_dev = window_batch.global_indices.to(predict_device)
                with torch_autocast_context(predict_device, amp_mode):
                    if factorized_forward is not None:
                        window_scores, window_head_logits = factorized_forward(
                            points=points_dev,
                            padding_mask=padding_dev,
                        )
                    else:
                        window_scores = model(
                            points=points_dev,
                            queries=queries_dev,
                            query_type_ids=query_type_ids_dev,
                            padding_mask=padding_dev,
                        )
                        window_head_logits = None
                window_scores = window_scores.to(dtype=point_score_sum.dtype)
                if window_head_logits is not None and head_logit_sum is None:
                    head_count = int(window_head_logits.shape[-1])
                    head_logit_sum = torch.zeros(
                        (point_count, head_count),
                        dtype=window_head_logits.dtype,
                        device=predict_device,
                    )
                    head_logit_count = torch.zeros((point_count,), dtype=window_head_logits.dtype, device=predict_device)
                for batch_idx in range(window_scores.shape[0]):
                    point_indices = indices_dev[batch_idx]
                    valid_points = point_indices >= 0
                    point_score_sum[point_indices[valid_points]] += window_scores[batch_idx, valid_points]
                    point_score_count[point_indices[valid_points]] += 1.0
                    if window_head_logits is not None and head_logit_sum is not None and head_logit_count is not None:
                        head_logit_sum[point_indices[valid_points]] += window_head_logits[batch_idx, valid_points]
                        head_logit_count[point_indices[valid_points]] += 1.0

        point_score_count = point_score_count.clamp(min=1.0)
        scores = (point_score_sum / point_score_count).to(output_device)
        head_logits = None
        if head_logit_sum is not None and head_logit_count is not None:
            head_logits = (head_logit_sum / head_logit_count.clamp(min=1.0).unsqueeze(-1)).to(output_device)
        return scores, head_logits
    finally:
        if original_device != predict_device:
            model.to(original_device)
        model.train(original_training)


def windowed_predict(
    model: torch.nn.Module,
    norm_points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    queries: torch.Tensor | None,
    query_type_ids: torch.Tensor | None,
    window_length: int = 512,
    window_stride: int = 256,
    batch_size: int = 16,
    device: torch.device | str | None = None,
    amp_mode: str = "off",
) -> torch.Tensor:
    """Run per-window pure-workload inference and average overlapping predictions."""
    scores, _head_logits = windowed_predict_with_heads(
        model=model,
        norm_points=norm_points,
        boundaries=boundaries,
        queries=queries,
        query_type_ids=query_type_ids,
        window_length=window_length,
        window_stride=window_stride,
        batch_size=batch_size,
        device=device,
        amp_mode=amp_mode,
    )
    return scores


def default_inference_device() -> torch.device:
    """Return the default device for saved-model inference."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def forward_predict(
    artifacts: ModelArtifacts,
    points: torch.Tensor,
    queries: torch.Tensor | None,
    query_type_ids: torch.Tensor | None,
    boundaries: list[tuple[int, int]] | None = None,
    trajectory_mmsis: list[int] | None = None,
    window_length: int = 512,
    window_stride: int = 256,
    device: torch.device | str | None = None,
    amp_mode: str = "off",
    batch_size: int = 16,
) -> torch.Tensor:
    """Run deterministic predictions with persisted scaler and model."""
    if boundaries is None:
        boundaries = [(0, points.shape[0])]
    point_dim = _model_point_dim(artifacts.model)
    if _is_workload_blind_model(artifacts.model):
        model_points = build_query_free_point_features_for_dim(
            points,
            point_dim,
            boundaries=boundaries,
            trajectory_mmsis=trajectory_mmsis,
            query_prior_field=getattr(artifacts, "query_prior_field", None),
        )
        norm_points = artifacts.scaler.transform_points(model_points)
        norm_queries = None
        query_type_ids = None
    else:
        model_points = points[:, :point_dim]
        if queries is None or query_type_ids is None:
            raise RuntimeError("query-aware checkpoint inference requires queries and query_type_ids.")
        norm_points, norm_queries = artifacts.scaler.transform(model_points, queries)
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
