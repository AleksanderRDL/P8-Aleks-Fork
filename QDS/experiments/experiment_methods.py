"""Method construction and evaluation helpers for experiment pipelines."""

from __future__ import annotations

import torch

from evaluation.baselines import (
    DouglasPeuckerMethod,
    MLQDSMethod,
    Method,
    ScoreHybridMethod,
    UniformTemporalMethod,
)
from evaluation.evaluate_methods import evaluate_method
from evaluation.query_cache import EvaluationQueryCache
from experiments.experiment_config import ExperimentConfig, SeedBundle
from experiments.experiment_workloads import workload_name
from experiments.range_cache import RangeRuntimeCache, prepare_range_label_cache
from queries.query_types import single_workload_type
from queries.workload import TypedQueryWorkload
from simplification.mlqds_scoring import workload_type_head
from training.importance_labels import compute_typed_importance_labels
from training.training_outputs import TrainingOutputs


def build_primary_methods(
    *,
    trained: TrainingOutputs,
    eval_workload: TypedQueryWorkload,
    eval_workload_map: dict[str, float],
    config: ExperimentConfig,
) -> list[Method]:
    """Build the standard matched-evaluation methods."""
    return [
        MLQDSMethod(
            name="MLQDS",
            trained=trained,
            workload=eval_workload,
            workload_type=single_workload_type(eval_workload_map),
            score_mode=config.model.mlqds_score_mode,
            score_temperature=config.model.mlqds_score_temperature,
            rank_confidence_weight=config.model.mlqds_rank_confidence_weight,
            temporal_fraction=config.model.mlqds_temporal_fraction,
            diversity_bonus=config.model.mlqds_diversity_bonus,
            hybrid_mode=config.model.mlqds_hybrid_mode,
            range_geometry_blend=config.model.mlqds_range_geometry_blend,
            inference_batch_size=config.model.inference_batch_size,
            amp_mode=config.model.amp_mode,
        ),
        UniformTemporalMethod(),
        DouglasPeuckerMethod(),
    ]


def prepare_eval_query_cache(
    *,
    test_points: torch.Tensor,
    test_boundaries: list[tuple[int, int]],
    eval_workload: TypedQueryWorkload,
    eval_is_range_only: bool,
    runtime_cache: RangeRuntimeCache,
) -> EvaluationQueryCache:
    """Return a validated eval query cache, reusing range runtime state when available."""
    eval_query_cache = runtime_cache.query_cache if eval_is_range_only else None
    if eval_query_cache is None:
        eval_query_cache = EvaluationQueryCache.for_workload(
            test_points,
            test_boundaries,
            eval_workload.typed_queries,
        )
        if eval_is_range_only:
            runtime_cache.query_cache = eval_query_cache
    else:
        eval_query_cache.validate(test_points, test_boundaries, eval_workload.typed_queries)
    return eval_query_cache


def prepare_eval_labels(
    *,
    test_points: torch.Tensor,
    test_boundaries: list[tuple[int, int]],
    eval_workload: TypedQueryWorkload,
    eval_workload_map: dict[str, float],
    config: ExperimentConfig,
    seeds: SeedBundle,
    eval_is_range_only: bool,
    run_oracle_baseline: bool,
    runtime_cache: RangeRuntimeCache,
) -> torch.Tensor | None:
    """Prepare eval labels needed by oracle, learned-fill, or geometry diagnostics."""
    if eval_is_range_only:
        prepared_eval_labels = prepare_range_label_cache(
            cache_label="eval",
            points=test_points,
            boundaries=test_boundaries,
            workload=eval_workload,
            workload_map=eval_workload_map,
            config=config,
            seed=seeds.eval_query_seed,
            runtime_cache=runtime_cache,
            range_boundary_prior_weight=float(getattr(config.model, "range_boundary_prior_weight", 0.0)),
        )
        if prepared_eval_labels is not None:
            labels, _ = prepared_eval_labels
            return labels
        return None
    if run_oracle_baseline:
        labels, _ = compute_typed_importance_labels(
            points=test_points,
            boundaries=test_boundaries,
            typed_queries=eval_workload.typed_queries,
            range_label_mode=str(getattr(config.model, "range_label_mode", "usefulness")),
            range_boundary_prior_weight=float(getattr(config.model, "range_boundary_prior_weight", 0.0)),
        )
        return labels
    return None


def attach_range_geometry_scores(
    *,
    methods: list[Method],
    eval_labels: torch.Tensor,
    eval_workload_map: dict[str, float],
) -> None:
    """Attach precomputed range-geometry scores to MLQDS when geometry blending is enabled."""
    _, eval_type_id = workload_type_head(single_workload_type(eval_workload_map))
    mlqds_method = methods[0]
    if isinstance(mlqds_method, MLQDSMethod):
        mlqds_method.range_geometry_scores = eval_labels[:, eval_type_id].float()


def build_learned_fill_methods(
    *,
    test_points: torch.Tensor,
    eval_labels: torch.Tensor,
    eval_workload_map: dict[str, float],
    config: ExperimentConfig,
    seeds: SeedBundle,
) -> list[Method]:
    """Build temporal random/oracle fill diagnostics for pure range evaluation."""
    _, eval_type_id = workload_type_head(single_workload_type(eval_workload_map))
    random_generator = torch.Generator().manual_seed(int(seeds.eval_query_seed) + 404)
    random_scores = torch.rand((test_points.shape[0],), generator=random_generator)
    return [
        ScoreHybridMethod(
            name="TemporalRandomFill",
            scores=random_scores,
            temporal_fraction=config.model.mlqds_temporal_fraction,
            diversity_bonus=config.model.mlqds_diversity_bonus,
            hybrid_mode=config.model.mlqds_hybrid_mode,
        ),
        ScoreHybridMethod(
            name="TemporalOracleFill",
            scores=eval_labels[:, eval_type_id].float(),
            temporal_fraction=config.model.mlqds_temporal_fraction,
            diversity_bonus=config.model.mlqds_diversity_bonus,
            hybrid_mode=config.model.mlqds_hybrid_mode,
        ),
    ]


def evaluate_shift_pairs(
    *,
    matched_mlqds_score: float,
    trained: TrainingOutputs,
    train_workload: TypedQueryWorkload,
    train_workload_map: dict[str, float],
    eval_workload_map: dict[str, float],
    config: ExperimentConfig,
    test_points: torch.Tensor,
    test_boundaries: list[tuple[int, int]],
) -> dict[str, dict[str, float]]:
    """Evaluate the train-workload self score needed by the shift table."""
    train_name = workload_name(train_workload_map)
    eval_name = workload_name(eval_workload_map)
    shift_pairs = {train_name: {eval_name: float(matched_mlqds_score)}}
    if train_name == eval_name:
        shift_pairs[train_name][train_name] = float(matched_mlqds_score)
        return shift_pairs

    train_query_cache = EvaluationQueryCache.for_workload(
        test_points,
        test_boundaries,
        train_workload.typed_queries,
    )
    shift_pairs[train_name][train_name] = float(
        evaluate_method(
            method=MLQDSMethod(
                name="MLQDS",
                trained=trained,
                workload=train_workload,
                workload_type=single_workload_type(train_workload_map),
                score_mode=config.model.mlqds_score_mode,
                score_temperature=config.model.mlqds_score_temperature,
                rank_confidence_weight=config.model.mlqds_rank_confidence_weight,
                temporal_fraction=config.model.mlqds_temporal_fraction,
                diversity_bonus=config.model.mlqds_diversity_bonus,
                hybrid_mode=config.model.mlqds_hybrid_mode,
                inference_batch_size=config.model.inference_batch_size,
                amp_mode=config.model.amp_mode,
            ),
            points=test_points,
            boundaries=test_boundaries,
            typed_queries=train_workload.typed_queries,
            workload_map=train_workload_map,
            compression_ratio=config.model.compression_ratio,
            query_cache=train_query_cache,
        ).aggregate_f1
    )
    return shift_pairs
