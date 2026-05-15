"""Diagnostics for factorized query-driven target labels."""

from __future__ import annotations

from typing import Any, Sequence

import torch


def _entropy(values: torch.Tensor) -> float:
    """Return normalized binary entropy for [0, 1] target values."""
    if int(values.numel()) == 0:
        return 0.0
    probs = values.float().clamp(1e-6, 1.0 - 1e-6)
    entropy = -(probs * torch.log2(probs) + (1.0 - probs) * torch.log2(1.0 - probs))
    return float(entropy.mean().item())


def _topk_mass(values: torch.Tensor, ratio: float) -> float:
    """Return fraction of total mass captured by top-k values."""
    count = int(values.numel())
    if count <= 0:
        return 0.0
    mass = float(values.sum().item())
    if mass <= 1e-12:
        return 0.0
    keep = min(count, max(1, int(torch.ceil(torch.tensor(float(ratio) * count)).item())))
    return float(torch.topk(values.float(), k=keep).values.sum().item() / mass)


def support_fraction_by_threshold(
    values: torch.Tensor,
    valid: torch.Tensor | None = None,
    thresholds: Sequence[float] = (0.0, 0.01, 0.05, 0.10),
) -> dict[str, float]:
    """Return support fractions above practical thresholds."""
    if valid is None:
        valid = torch.ones_like(values, dtype=torch.bool)
    valid = valid.to(dtype=torch.bool)
    denominator = max(1, int(valid.sum().item()))
    out: dict[str, float] = {}
    for threshold in thresholds:
        key = f"gt_{float(threshold):.2f}"
        supported = valid & (values.float() > float(threshold))
        out[key] = float(supported.sum().item() / denominator)
    return out


def _segment_position_mass(
    values: torch.Tensor,
    boundaries: list[tuple[int, int]],
) -> dict[str, float]:
    """Return label mass by coarse position within each trajectory."""
    buckets = {
        "start": 0.0,
        "middle": 0.0,
        "end": 0.0,
    }
    total = 0.0
    for start, end in boundaries:
        count = int(end - start)
        if count <= 0:
            continue
        local = values[start:end].float().clamp(min=0.0)
        total += float(local.sum().item())
        if count == 1:
            buckets["middle"] += float(local.sum().item())
            continue
        positions = torch.linspace(0.0, 1.0, steps=count, device=values.device)
        buckets["start"] += float(local[positions <= 0.20].sum().item())
        buckets["middle"] += float(local[(positions > 0.20) & (positions < 0.80)].sum().item())
        buckets["end"] += float(local[positions >= 0.80].sum().item())
    if total <= 1e-12:
        return {key: 0.0 for key in buckets}
    return {key: float(value / total) for key, value in buckets.items()}


def factorized_target_diagnostics(
    head_targets: torch.Tensor,
    head_mask: torch.Tensor,
    head_names: Sequence[str],
    boundaries: list[tuple[int, int]],
    budget_grid: Sequence[float] = (0.01, 0.02, 0.05, 0.10),
) -> dict[str, Any]:
    """Return compact target mass, entropy, and top-k diagnostics by head."""
    if head_targets.ndim != 2 or head_mask.shape != head_targets.shape:
        raise ValueError("head_targets and head_mask must have matching shape [n_points, n_heads].")
    if len(head_names) != int(head_targets.shape[1]):
        raise ValueError("head_names length must match target head count.")

    positive_fraction_by_head: dict[str, float] = {}
    positive_mass_by_head: dict[str, float] = {}
    positive_point_count_by_head: dict[str, int] = {}
    support_fraction_by_threshold_by_head: dict[str, dict[str, float]] = {}
    entropy_by_head: dict[str, float] = {}
    topk_label_mass_budget_grid: dict[str, dict[str, float]] = {}
    label_mass_by_segment_position: dict[str, dict[str, float]] = {}

    for head_idx, head_name in enumerate(head_names):
        valid = head_mask[:, head_idx].to(dtype=torch.bool)
        values = head_targets[:, head_idx].float().clamp(0.0, 1.0)
        valid_values = values[valid]
        positive = valid & (values > 0.0)
        positive_fraction_by_head[str(head_name)] = float(
            positive.sum().item() / max(1, int(valid.sum().item()))
        )
        positive_mass_by_head[str(head_name)] = float(values[positive].sum().item()) if bool(positive.any().item()) else 0.0
        positive_point_count_by_head[str(head_name)] = int(positive.sum().item())
        support_fraction_by_threshold_by_head[str(head_name)] = support_fraction_by_threshold(values, valid)
        entropy_by_head[str(head_name)] = _entropy(valid_values)
        topk_label_mass_budget_grid[str(head_name)] = {
            f"{float(ratio):.2f}": _topk_mass(valid_values, float(ratio))
            for ratio in budget_grid
        }
        label_mass_by_segment_position[str(head_name)] = _segment_position_mass(values, boundaries)

    return {
        "schema_version": 1,
        "positive_fraction_by_head": positive_fraction_by_head,
        "positive_label_mass_by_head": positive_mass_by_head,
        "positive_point_count_by_head": positive_point_count_by_head,
        "support_fraction_by_threshold_by_head": support_fraction_by_threshold_by_head,
        "entropy_by_head": entropy_by_head,
        "topk_label_mass_budget_grid": topk_label_mass_budget_grid,
        "label_mass_by_segment_position": label_mass_by_segment_position,
        "label_mass_by_query_family": {},
        "label_mass_by_ship": {},
        "train_eval_label_drift": None,
    }
