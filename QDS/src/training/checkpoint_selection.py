"""Checkpoint selection helpers for model training."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.experiments.experiment_config import ModelConfig
from src.queries.query_types import normalize_pure_workload_map


@dataclass
class CheckpointCandidate:
    """Candidate snapshot awaiting exact validation in a full-score round."""

    epoch_number: int
    epoch_index: int
    cheap_score: float
    loss: float
    state_dict: dict[str, torch.Tensor]
    stats: dict[str, float]
    avg_tau: float


def selection_score(avg_tau: float, pred_std: float, loss: float | None = None) -> float:
    """Score checkpoint quality while strongly penalizing collapsed predictions."""
    collapse_penalty = 1.0 if pred_std < 1e-3 else 0.0
    if loss is None:
        return float(avg_tau - collapse_penalty)
    return float(-float(loss) + 1e-3 * avg_tau - collapse_penalty)


def normalize_checkpoint_selection_metric(selection_metric: str) -> str:
    """Normalize checkpoint selection metric names, including legacy aliases."""
    metric = str(selection_metric).lower()
    if metric == "f1":
        return "score"
    return metric


def validation_score_selection_score(validation_score: float, pred_std: float) -> float:
    """Score checkpoints by active validation score while rejecting collapsed predictions."""
    collapse_penalty = 1.0 if pred_std < 1e-3 else 0.0
    return float(validation_score - collapse_penalty)


def f1_selection_score(query_f1: float, pred_std: float) -> float:
    """Legacy alias for validation_score_selection_score."""
    return validation_score_selection_score(query_f1, pred_std)


def _normalized_workload_map(workload_map: dict[str, float]) -> dict[str, float]:
    """Normalize a pure workload map into the fixed query-type key set."""
    names = ["range", "knn", "similarity", "clustering"]
    normalized = normalize_pure_workload_map(workload_map)
    return {name: float(normalized.get(name, 0.0)) for name in names}


def uniform_type_deficit(
    per_type_score: dict[str, float],
    uniform_per_type_score: dict[str, float],
    workload_map: dict[str, float],
) -> float:
    """Weighted amount by which a checkpoint loses to fair uniform per query type."""
    type_weights = _normalized_workload_map(workload_map)
    return float(
        sum(
            type_weights[name]
            * max(
                0.0,
                float(uniform_per_type_score.get(name, 0.0)) - float(per_type_score.get(name, 0.0)),
            )
            for name in type_weights
        )
    )


def uniform_gap_selection_score(
    validation_score: float,
    per_type_score: dict[str, float],
    uniform_score: float,
    uniform_per_type_score: dict[str, float],
    workload_map: dict[str, float],
    pred_std: float,
    aggregate_gap_weight: float = 0.5,
    type_penalty_weight: float = 1.0,
) -> float:
    """Score checkpoints by held-out validation score while penalizing losses to fair uniform."""
    collapse_penalty = 1.0 if pred_std < 1e-3 else 0.0
    aggregate_gap = float(validation_score) - float(uniform_score)
    type_deficit = uniform_type_deficit(per_type_score, uniform_per_type_score, workload_map)
    return float(
        float(validation_score)
        + float(aggregate_gap_weight) * aggregate_gap
        - float(type_penalty_weight) * type_deficit
        - collapse_penalty
    )


def record_validation_stats(
    stats: dict[str, float],
    *,
    validation_score: float,
    per_type_score: dict[str, float],
    validation_metrics: dict[str, float],
    validation_uniform_result: tuple[float, dict[str, float]] | None,
    validation_workload_map: dict[str, float] | None,
) -> None:
    """Attach exact validation metrics to an epoch stats row."""
    stats["val_selection_score"] = float(validation_score)
    for metric_name, metric_value in validation_metrics.items():
        stats[f"val_{metric_name}"] = float(metric_value)
    for type_name, value in per_type_score.items():
        stats[f"val_selection_score_{type_name}"] = float(value)
    if validation_uniform_result is not None:
        uniform_score, uniform_per_type_score = validation_uniform_result
        stats["val_uniform_score"] = float(uniform_score)
        stats["val_uniform_f1"] = stats["val_uniform_score"]
        stats["val_selection_uniform_gap"] = float(validation_score - uniform_score)
        stats["val_query_uniform_gap"] = stats["val_selection_uniform_gap"]
        stats["val_selection_type_deficit"] = uniform_type_deficit(
            per_type_score,
            uniform_per_type_score,
            validation_workload_map or {},
        )
        stats["val_query_type_deficit"] = stats["val_selection_type_deficit"]
        for type_name, value in uniform_per_type_score.items():
            stats[f"val_uniform_score_{type_name}"] = float(value)
            stats[f"val_uniform_f1_{type_name}"] = stats[f"val_uniform_score_{type_name}"]
            stats[f"val_selection_score_gap_{type_name}"] = float(per_type_score.get(type_name, 0.0) - value)


def selection_from_stats(
    *,
    stats: dict[str, float],
    avg_tau: float,
    selection_metric: str,
    validation_uniform_result: tuple[float, dict[str, float]] | None,
    validation_workload_map: dict[str, float] | None,
    model_config: ModelConfig,
) -> float | None:
    """Return the active checkpoint selection score for one stats row."""
    selection_metric = normalize_checkpoint_selection_metric(selection_metric)
    if (
        selection_metric == "uniform_gap"
        and "val_selection_score" in stats
        and validation_uniform_result is not None
    ):
        uniform_score, uniform_per_type_score = validation_uniform_result
        per_type_score = {
            name: stats.get(f"val_selection_score_{name}", 0.0)
            for name in ["range", "knn", "similarity", "clustering"]
        }
        return uniform_gap_selection_score(
            validation_score=stats["val_selection_score"],
            per_type_score=per_type_score,
            uniform_score=uniform_score,
            uniform_per_type_score=uniform_per_type_score,
            workload_map=validation_workload_map or {},
            pred_std=stats["pred_std"],
            aggregate_gap_weight=float(getattr(model_config, "checkpoint_uniform_gap_weight", 0.5)),
            type_penalty_weight=float(getattr(model_config, "checkpoint_type_penalty_weight", 1.0)),
        )
    if selection_metric == "score" and "val_selection_score" in stats:
        return validation_score_selection_score(stats["val_selection_score"], stats["pred_std"])
    if selection_metric in {"score", "uniform_gap"}:
        return None
    return selection_score(avg_tau, stats["pred_std"], stats["loss"])
