"""Offline predictability audit for train-derived query priors."""

from __future__ import annotations

from typing import Any

import torch

from training.query_prior_fields import sample_query_prior_fields
from training.query_useful_targets import build_query_useful_v1_targets

PREDICTABILITY_AUDIT_SCHEMA_VERSION = 1
PREDICTABILITY_GATE_THRESHOLDS = {
    "lift_at_1_percent": 1.10,
    "lift_at_2_percent": 1.15,
    "lift_at_5_percent": 1.20,
    "spearman_min": 0.15,
    "pr_auc_lift_over_base_rate": 1.25,
}


def _rankdata(values: torch.Tensor) -> torch.Tensor:
    """Return deterministic ordinal ranks normalized only by caller statistics."""
    flat = values.detach().cpu().float().flatten()
    if int(flat.numel()) == 0:
        return flat
    order = torch.argsort(flat, stable=True)
    ranks = torch.empty_like(flat)
    ranks[order] = torch.arange(int(flat.numel()), dtype=torch.float32)
    return ranks


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return Pearson correlation for finite 1-D tensors."""
    left = left.detach().cpu().float().flatten()
    right = right.detach().cpu().float().flatten()
    finite = torch.isfinite(left) & torch.isfinite(right)
    if int(finite.sum().item()) < 2:
        return 0.0
    left = left[finite]
    right = right[finite]
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denom = torch.linalg.vector_norm(left_centered) * torch.linalg.vector_norm(right_centered)
    if float(denom.item()) <= 1e-12:
        return 0.0
    return float((left_centered * right_centered).sum().item() / float(denom.item()))


def _spearman(score: torch.Tensor, target: torch.Tensor) -> float:
    """Return rank correlation between score and target."""
    return _pearson(_rankdata(score), _rankdata(target))


def _kendall_tau_sampled(score: torch.Tensor, target: torch.Tensor, max_pairs: int = 50_000) -> float:
    """Return deterministic sampled Kendall tau for audit-scale diagnostics."""
    score = score.detach().cpu().float().flatten()
    target = target.detach().cpu().float().flatten()
    n = int(score.numel())
    if n < 2:
        return 0.0
    pair_count = n * (n - 1) // 2
    if pair_count <= max_pairs:
        left, right = torch.triu_indices(n, n, offset=1)
    else:
        generator = torch.Generator().manual_seed(1_706_011)
        left = torch.randint(0, n, (max_pairs,), generator=generator)
        right = torch.randint(0, n, (max_pairs,), generator=generator)
        keep = left != right
        left = left[keep]
        right = right[keep]
    score_delta = score[left] - score[right]
    target_delta = target[left] - target[right]
    valid = (score_delta.abs() > 1e-8) & (target_delta.abs() > 1e-8)
    if not bool(valid.any().item()):
        return 0.0
    concordant = (score_delta[valid] * target_delta[valid]) > 0.0
    return float((2.0 * concordant.float().mean() - 1.0).item())


def _auc(score: torch.Tensor, positive: torch.Tensor) -> float | None:
    """Return ROC AUC via rank-sum, or None when undefined."""
    score = score.detach().cpu().float().flatten()
    positive = positive.detach().cpu().bool().flatten()
    pos_count = int(positive.sum().item())
    neg_count = int((~positive).sum().item())
    if pos_count <= 0 or neg_count <= 0:
        return None
    ranks = _rankdata(score) + 1.0
    rank_sum_pos = float(ranks[positive].sum().item())
    auc = (rank_sum_pos - pos_count * (pos_count + 1) / 2.0) / max(1.0, float(pos_count * neg_count))
    return float(max(0.0, min(1.0, auc)))


def _pr_auc(score: torch.Tensor, positive: torch.Tensor) -> float | None:
    """Return average precision as a PR-AUC proxy."""
    score = score.detach().cpu().float().flatten()
    positive = positive.detach().cpu().bool().flatten()
    pos_count = int(positive.sum().item())
    if pos_count <= 0:
        return None
    order = torch.argsort(score, descending=True, stable=True)
    sorted_positive = positive[order].float()
    cumulative_positive = torch.cumsum(sorted_positive, dim=0)
    ranks = torch.arange(1, int(score.numel()) + 1, dtype=torch.float32)
    precision_at_hits = cumulative_positive / ranks
    return float((precision_at_hits * sorted_positive).sum().item() / max(1, pos_count))


def _topk_indices(score: torch.Tensor, ratio: float) -> torch.Tensor:
    """Return global top-k indices for a budget ratio."""
    n = int(score.numel())
    if n <= 0:
        return torch.empty((0,), dtype=torch.long)
    keep = min(n, max(1, int(torch.ceil(torch.tensor(float(ratio) * n)).item())))
    return torch.topk(score.detach().cpu().float(), k=keep, largest=True).indices


def _ndcg_at(score: torch.Tensor, target: torch.Tensor, ratio: float) -> float:
    """Return NDCG at global budget ratio."""
    score_cpu = score.detach().cpu().float().flatten()
    target_cpu = target.detach().cpu().float().flatten().clamp(min=0.0)
    idx = _topk_indices(score_cpu, ratio)
    ideal_idx = _topk_indices(target_cpu, ratio)
    if int(idx.numel()) == 0 or int(ideal_idx.numel()) == 0:
        return 0.0
    gains = target_cpu[idx]
    ideal_gains = target_cpu[ideal_idx]
    discounts = 1.0 / torch.log2(torch.arange(2, int(idx.numel()) + 2, dtype=torch.float32))
    dcg = float((gains * discounts).sum().item())
    idcg = float((ideal_gains * discounts).sum().item())
    if idcg <= 1e-12:
        return 0.0
    return float(dcg / idcg)


def _lift_at(score: torch.Tensor, target: torch.Tensor, ratio: float) -> float:
    """Return top-budget mean-target lift over base target mean."""
    score_cpu = score.detach().cpu().float().flatten()
    target_cpu = target.detach().cpu().float().flatten().clamp(min=0.0)
    if int(score_cpu.numel()) == 0:
        return 0.0
    base = float(target_cpu.mean().item())
    if base <= 1e-12:
        return 0.0
    idx = _topk_indices(score_cpu, ratio)
    if int(idx.numel()) == 0:
        return 0.0
    return float(target_cpu[idx].mean().item() / base)


def _prior_predictability_score(points: torch.Tensor, query_prior_field: dict[str, Any]) -> torch.Tensor:
    """Build a simple train-prior score from sampled prior-field channels."""
    sampled = sample_query_prior_fields(points, query_prior_field).float()
    if sampled.shape[1] < 6:
        return torch.zeros((int(points.shape[0]),), dtype=torch.float32, device=points.device)
    score = (
        0.30 * sampled[:, 0]
        + 0.25 * sampled[:, 1]
        + 0.15 * sampled[:, 4]
        + 0.10 * sampled[:, 2]
        + 0.10 * sampled[:, 3]
        + 0.10 * sampled[:, 5]
    )
    return score.clamp(0.0, 1.0)


def query_prior_predictability_scores(points: torch.Tensor, query_prior_field: dict[str, Any]) -> torch.Tensor:
    """Return the query-prior-only score used by predictability and causality diagnostics."""
    return _prior_predictability_score(points, query_prior_field)


def query_prior_predictability_audit(
    *,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    eval_typed_queries: list[dict[str, Any]],
    query_prior_field: dict[str, Any] | None,
) -> dict[str, Any]:
    """Measure whether train-derived query-prior fields predict held-out eval usefulness."""
    if query_prior_field is None:
        return {
            "schema_version": PREDICTABILITY_AUDIT_SCHEMA_VERSION,
            "available": False,
            "gate_pass": False,
            "reason": "missing_query_prior_field",
        }
    eval_targets = build_query_useful_v1_targets(
        points=points,
        boundaries=boundaries,
        typed_queries=eval_typed_queries,
    )
    target = eval_targets.labels[:, 0].float().detach().cpu()
    score = _prior_predictability_score(points, query_prior_field).detach().cpu()
    positive = target > 0.0
    base_rate = float(positive.float().mean().item()) if int(target.numel()) > 0 else 0.0
    pr_auc = _pr_auc(score, positive)
    auc = _auc(score, positive)
    budget_ratios = (0.01, 0.02, 0.05, 0.10)
    lifts = {f"lift_at_{int(ratio * 100)}_percent": _lift_at(score, target, ratio) for ratio in budget_ratios}
    ndcg = {f"ndcg_at_{int(ratio * 100)}_percent": _ndcg_at(score, target, ratio) for ratio in budget_ratios}
    pr_auc_lift = float(pr_auc / max(base_rate, 1e-12)) if pr_auc is not None else None
    spearman = _spearman(score, target)
    metrics: dict[str, Any] = {
        "spearman": spearman,
        "kendall_tau": _kendall_tau_sampled(score, target),
        "auc": auc,
        "pr_auc": pr_auc,
        "base_positive_rate": base_rate,
        "pr_auc_lift_over_base_rate": pr_auc_lift,
        **lifts,
        **ndcg,
    }
    checks = {
        "lift_at_1_percent": lifts["lift_at_1_percent"] >= PREDICTABILITY_GATE_THRESHOLDS["lift_at_1_percent"],
        "lift_at_2_percent": lifts["lift_at_2_percent"] >= PREDICTABILITY_GATE_THRESHOLDS["lift_at_2_percent"],
        "lift_at_5_percent": lifts["lift_at_5_percent"] >= PREDICTABILITY_GATE_THRESHOLDS["lift_at_5_percent"],
        "spearman_min": spearman >= PREDICTABILITY_GATE_THRESHOLDS["spearman_min"],
        "pr_auc_lift_over_base_rate": (
            pr_auc_lift is not None
            and pr_auc_lift >= PREDICTABILITY_GATE_THRESHOLDS["pr_auc_lift_over_base_rate"]
        ),
    }
    return {
        "schema_version": PREDICTABILITY_AUDIT_SCHEMA_VERSION,
        "available": True,
        "evaluation_stage": "after_masks_frozen_diagnostic_only",
        "score_source": "train_query_prior_fields",
        "target_source": "heldout_eval_query_useful_v1_targets",
        "used_for_training": False,
        "used_for_checkpoint_selection": False,
        "used_for_retained_mask_decision": False,
        "thresholds": dict(PREDICTABILITY_GATE_THRESHOLDS),
        "metrics": metrics,
        "gate_checks": checks,
        "gate_pass": all(checks.values()),
        "eval_point_count": int(points.shape[0]),
        "eval_positive_target_count": int(positive.sum().item()),
        "eval_query_count": int(len([q for q in eval_typed_queries if str(q.get("type", "")).lower() == "range"])),
    }
