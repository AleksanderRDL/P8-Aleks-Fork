"""Experiment orchestration helpers for training and evaluation runs. See experiments/README.md for details."""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from evaluation.baselines import (
    FrozenMaskMethod,
    Method,
    MLQDSMethod,
    OracleMethod,
)
from evaluation.evaluate_methods import evaluate_method
from evaluation.metrics import MethodEvaluation
from evaluation.query_cache import EvaluationQueryCache
from evaluation.range_usefulness import range_usefulness_weight_summary
from evaluation.tables import (
    print_geometric_distortion_table,
    print_method_comparison_table,
    print_range_usefulness_table,
    print_shift_table,
)
from experiments.experiment_config import ExperimentConfig, derive_seed_bundle
from experiments.experiment_data import build_experiment_datasets, prepare_experiment_split
from experiments.geojson_writers import report_trajectory_length_loss, write_queries_geojson, write_simplified_csv
from experiments.experiment_methods import (
    attach_range_geometry_scores,
    build_learned_fill_methods,
    build_primary_methods,
    evaluate_shift_pairs,
    prepare_eval_labels,
    prepare_eval_query_cache,
)
from experiments.experiment_outputs import ExperimentOutputs, write_experiment_results
from experiments.range_cache import (
    RangeRuntimeCache,
    prepare_range_label_cache,
    range_only_queries,
)
from experiments.range_diagnostics import (
    _evaluation_metrics_payload,
    _print_range_diagnostics_summary,
    _print_range_distribution_comparison,
    _range_audit_ratios,
    _range_learned_fill_summary,
    _range_workload_diagnostics,
    _range_workload_distribution_comparison,
)
from experiments.experiment_workloads import (
    generate_experiment_workloads,
    resolve_workload_maps,
    workload_name,
)
from queries.query_types import QUERY_TYPE_ID_RANGE, single_workload_type
from simplification.mlqds_scoring import workload_type_head
from simplification.simplify_trajectories import temporal_hybrid_selector_budget_diagnostics
from training.train_model import train_model
from training.checkpoints import ModelArtifacts, save_checkpoint
from training.model_features import is_workload_blind_model_type, model_type_metadata
from training.teacher_distillation import (
    build_range_teacher_config,
    distill_range_teacher_labels,
    range_teacher_distillation_enabled,
)
from training.training_targets import (
    aggregate_range_component_label_sets,
    aggregate_range_component_retained_frequency_training_labels,
    aggregate_range_continuity_retained_frequency_training_labels,
    aggregate_range_global_budget_retained_frequency_training_labels,
    aggregate_range_label_sets,
    aggregate_range_marginal_coverage_training_labels,
    aggregate_range_retained_frequency_training_labels,
    aggregate_range_structural_retained_frequency_training_labels,
    balance_range_training_target_by_trajectory,
    range_component_retained_frequency_training_labels,
    range_continuity_retained_frequency_training_labels,
    range_global_budget_retained_frequency_training_labels,
    range_historical_prior_retained_frequency_training_labels,
    range_local_swap_gain_cost_frequency_training_labels,
    range_local_swap_utility_frequency_training_labels,
    range_query_residual_frequency_training_labels,
    range_set_utility_frequency_training_labels,
    range_query_spine_frequency_training_labels,
    range_marginal_coverage_training_labels,
    range_retained_frequency_training_labels,
    range_structural_retained_frequency_training_labels,
)
from experiments.torch_runtime import (
    amp_runtime_snapshot,
    cuda_memory_snapshot,
    reset_cuda_peak_memory_stats,
    torch_runtime_snapshot,
)


@contextmanager
def _phase(name: str):
    """Log a named phase with wall-clock timing."""
    print(f"[{name}] starting...", flush=True)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        print(f"[{name}] done in {dt:.2f}s", flush=True)


def run_experiment_pipeline(
    config: ExperimentConfig,
    trajectories: list[torch.Tensor],
    results_dir: str,
    save_model: str | None = None,
    save_queries_dir: str | None = None,
    save_simplified_dir: str | None = None,
    trajectory_mmsis: list[int] | None = None,
    validation_trajectories: list[torch.Tensor] | None = None,
    eval_trajectories: list[torch.Tensor] | None = None,
    eval_trajectory_mmsis: list[int] | None = None,
    trajectory_source_ids: list[int] | None = None,
    data_audit: dict[str, Any] | None = None,
) -> ExperimentOutputs:
    """Run training, matched evaluation, and shifted evaluation tables. See experiments/README.md for details."""
    pipeline_t0 = time.perf_counter()
    train_workload_map, eval_workload_map = resolve_workload_maps(config.query.workload)
    if eval_trajectories is None:
        print(
            f"[pipeline] {len(trajectories)} trajectories, workload={workload_name(eval_workload_map)}",
            flush=True,
        )
    else:
        validation_part = (
            f", validation={len(validation_trajectories)} trajectories"
            if validation_trajectories is not None
            else ""
        )
        print(
            f"[pipeline] train={len(trajectories)} trajectories{validation_part}, "
            f"eval={len(eval_trajectories)} trajectories, "
            f"workload={workload_name(eval_workload_map)}",
            flush=True,
        )

    seeds = derive_seed_bundle(config.data.seed)
    selection_metric = str(getattr(config.model, "checkpoint_selection_metric", "score")).lower()
    validation_score_every = int(getattr(config.model, "validation_score_every", 0) or 0)
    needs_validation_score = selection_metric in {"score", "uniform_gap"} or validation_score_every > 0
    with _phase("split"):
        data_split = prepare_experiment_split(
            config=config,
            seeds=seeds,
            trajectories=trajectories,
            needs_validation_score=needs_validation_score,
            trajectory_mmsis=trajectory_mmsis,
            validation_trajectories=validation_trajectories,
            eval_trajectories=eval_trajectories,
            eval_trajectory_mmsis=eval_trajectory_mmsis,
            trajectory_source_ids=trajectory_source_ids,
        )
        train_traj = data_split.train_traj
        test_traj = data_split.test_traj
        selection_traj = data_split.selection_traj
        train_mmsis = data_split.train_mmsis
        test_mmsis = data_split.test_mmsis
        train_source_ids = data_split.train_source_ids

    with _phase("build-datasets"):
        datasets = build_experiment_datasets(data_split)
        train_points = datasets.train_points
        test_points = datasets.test_points
        selection_points = datasets.selection_points
        train_boundaries = datasets.train_boundaries
        test_boundaries = datasets.test_boundaries
        selection_boundaries = datasets.selection_boundaries

    with _phase("generate-workloads"):
        workloads = generate_experiment_workloads(
            config=config,
            seeds=seeds,
            train_traj=train_traj,
            test_traj=test_traj,
            selection_traj=selection_traj,
            train_points=train_points,
            test_points=test_points,
            selection_points=selection_points,
            train_boundaries=train_boundaries,
            test_boundaries=test_boundaries,
            selection_boundaries=selection_boundaries,
            train_workload_map=train_workload_map,
            eval_workload_map=eval_workload_map,
        )
        train_workload = workloads.train_workload
        train_label_workloads = workloads.train_label_workloads
        train_label_workload_seeds = workloads.train_label_workload_seeds
        eval_workload = workloads.eval_workload
        selection_workload = workloads.selection_workload

    range_diagnostics_summary: dict[str, Any] = {}
    range_diagnostics_rows: list[dict[str, Any]] = []
    range_runtime_caches = {
        "train": RangeRuntimeCache(),
        "eval": RangeRuntimeCache(),
        "selection": RangeRuntimeCache(),
    }
    workload_distribution_comparison: dict[str, Any] = {"deltas_vs_eval": {}}

    if save_queries_dir:
        with _phase("write-queries-geojson"):
            write_queries_geojson(save_queries_dir, eval_workload.typed_queries)

    reset_cuda_peak_memory_stats()
    train_labels: tuple[torch.Tensor, torch.Tensor] | None = None
    range_training_target_mode = str(getattr(config.model, "range_training_target_mode", "point_value")).lower()
    range_replicate_target_aggregation = str(
        getattr(config.model, "range_replicate_target_aggregation", "label_mean")
    ).lower()
    if range_replicate_target_aggregation not in {"label_mean", "label_max", "frequency_mean"}:
        raise ValueError(
            "range_replicate_target_aggregation must be 'label_mean', 'label_max', or 'frequency_mean'."
        )
    if len(train_label_workloads) > 1 and not is_workload_blind_model_type(config.model.model_type):
        raise RuntimeError("range_train_workload_replicates > 1 is only valid for workload-blind model types.")
    range_training_target_transform: dict[str, Any] = {
        "mode": range_training_target_mode,
        "enabled": False,
    }
    range_target_balance_diagnostics: dict[str, Any] = {
        "enabled": False,
        "mode": str(getattr(config.model, "range_target_balance_mode", "none")).lower(),
    }
    range_training_label_aggregation: dict[str, Any] = {
        "enabled": False,
        "replicate_count": int(len(train_label_workloads)),
        "seeds": [int(seed) for seed in train_label_workload_seeds],
    }
    teacher_distillation_diagnostics: dict[str, Any] = {
        "enabled": False,
        "mode": str(getattr(config.model, "range_teacher_distillation_mode", "none")),
    }
    if range_training_target_mode == "query_useful_v1_factorized":
        raise RuntimeError(
            "QueryUsefulV1 factorized targets are not implemented yet. "
            "See Range_QDS/docs/query-driven-rework-guide.md."
        )
    selection_query_cache: EvaluationQueryCache | None = None
    selection_geometry_scores: torch.Tensor | None = None
    mlqds_range_geometry_blend = max(0.0, min(1.0, float(getattr(config.model, "mlqds_range_geometry_blend", 0.0))))
    with _phase("range-training-prep"):
        train_label_sets: list[tuple[torch.Tensor, torch.Tensor]] = []
        train_component_label_sets: list[dict[str, torch.Tensor] | None] = []
        for replicate_index, label_workload in enumerate(train_label_workloads):
            label_cache_name = "train" if replicate_index == 0 else f"train_r{replicate_index}"
            runtime_cache = range_runtime_caches["train"] if replicate_index == 0 else RangeRuntimeCache()
            label_result = prepare_range_label_cache(
                cache_label=label_cache_name,
                points=train_points,
                boundaries=train_boundaries,
                workload=label_workload,
                workload_map=train_workload_map,
                config=config,
                seed=train_label_workload_seeds[replicate_index],
                runtime_cache=runtime_cache,
                range_boundary_prior_weight=float(getattr(config.model, "range_boundary_prior_weight", 0.0)),
            )
            if label_result is not None:
                train_label_sets.append(label_result)
                train_component_label_sets.append(runtime_cache.component_labels)
        if train_label_sets:
            train_labels = train_label_sets[0]
            if (
                len(train_label_sets) > 1
                and range_training_target_mode == "point_value"
                and not range_teacher_distillation_enabled(config.model)
            ):
                if range_replicate_target_aggregation == "frequency_mean":
                    raise ValueError("range_replicate_target_aggregation='frequency_mean' requires a frequency target.")
                labels, labelled_mask, aggregation_diagnostics = aggregate_range_label_sets(
                    train_label_sets,
                    aggregation="max" if range_replicate_target_aggregation == "label_max" else "mean",
                )
                train_labels = (labels, labelled_mask)
                range_training_label_aggregation.update(aggregation_diagnostics)
                range_training_label_aggregation["enabled"] = True
                range_training_label_aggregation["replicate_target_aggregation"] = (
                    range_replicate_target_aggregation
                )
        if (
            selection_workload is not None
            and selection_points is not None
            and selection_boundaries is not None
            and len(range_only_queries(selection_workload.typed_queries)) == len(selection_workload.typed_queries)
        ):
            selection_query_cache = EvaluationQueryCache.for_workload(
                selection_points,
                selection_boundaries,
                selection_workload.typed_queries,
            )
            range_runtime_caches["selection"].query_cache = selection_query_cache
            if mlqds_range_geometry_blend > 0.0:
                selection_labels = prepare_range_label_cache(
                    cache_label="selection",
                    points=selection_points,
                    boundaries=selection_boundaries,
                    workload=selection_workload,
                    workload_map=eval_workload_map,
                    config=config,
                    seed=seeds.eval_query_seed + 17,
                    runtime_cache=range_runtime_caches["selection"],
                    range_boundary_prior_weight=float(getattr(config.model, "range_boundary_prior_weight", 0.0)),
                )
                if selection_labels is not None:
                    labels, _labelled_mask = selection_labels
                    _, selection_type_id = workload_type_head(single_workload_type(eval_workload_map))
                    selection_geometry_scores = labels[:, selection_type_id].float()
    if (
        train_labels is not None
        and len(range_only_queries(train_workload.typed_queries)) == len(train_workload.typed_queries)
    ):
        print("  prepared train range labels for precomputed training target", flush=True)
    if selection_query_cache is not None:
        print("  prepared checkpoint-validation range query cache", flush=True)
    if range_teacher_distillation_enabled(config.model):
        if not is_workload_blind_model_type(config.model.model_type):
            raise RuntimeError("range teacher distillation is only valid for workload-blind model types.")
        if train_labels is None:
            raise RuntimeError("range teacher distillation requires precomputed range training labels.")
        for label_workload in train_label_workloads:
            if len(range_only_queries(label_workload.typed_queries)) != len(label_workload.typed_queries):
                raise RuntimeError("range teacher distillation requires pure range training workloads.")
        teacher_config = build_range_teacher_config(config.model)
        print(
            f"  range teacher distillation enabled: mode={config.model.range_teacher_distillation_mode} "
            f"teacher_epochs={teacher_config.epochs} "
            f"replicates={len(train_label_workloads)}",
            flush=True,
        )
        distilled_label_sets: list[tuple[torch.Tensor, torch.Tensor]] = []
        per_teacher: list[dict[str, Any]] = []
        for replicate_index, label_workload in enumerate(train_label_workloads):
            with _phase(f"train-range-teacher-r{replicate_index} ({teacher_config.epochs} epochs)"):
                teacher_trained = train_model(
                    train_trajectories=train_traj,
                    train_boundaries=train_boundaries,
                    workload=label_workload,
                    model_config=teacher_config,
                    seed=seeds.torch_seed + 31 + replicate_index,
                    train_workload_map=train_workload_map,
                    precomputed_labels=train_label_sets[replicate_index],
                    train_trajectory_source_ids=train_source_ids,
                    train_trajectory_mmsis=train_mmsis,
                )
            with _phase(f"distill-range-teacher-r{replicate_index}-labels"):
                distilled_labels, replicate_diagnostics = distill_range_teacher_labels(
                    teacher=teacher_trained,
                    teacher_model_type=teacher_config.model_type,
                    points=train_points,
                    boundaries=train_boundaries,
                    workload=label_workload,
                    model_config=config.model,
                )
            replicate_diagnostics["replicate_index"] = int(replicate_index)
            replicate_diagnostics["seed"] = int(train_label_workload_seeds[replicate_index])
            per_teacher.append(replicate_diagnostics)
            distilled_label_sets.append(distilled_labels)
        if len(distilled_label_sets) == 1:
            train_labels = distilled_label_sets[0]
            teacher_distillation_diagnostics = dict(per_teacher[0])
        else:
            teacher_aggregation_mode = "max" if range_replicate_target_aggregation == "label_max" else "mean"
            labels, labelled_mask, aggregation_diagnostics = aggregate_range_label_sets(
                distilled_label_sets,
                source="range_teacher_distillation_replicates",
                aggregation=teacher_aggregation_mode,
            )
            train_labels = (labels, labelled_mask)
            positive = labelled_mask[:, QUERY_TYPE_ID_RANGE] & (labels[:, QUERY_TYPE_ID_RANGE] > 0.0)
            teacher_distillation_diagnostics = {
                "enabled": True,
                "mode": str(getattr(config.model, "range_teacher_distillation_mode", "none")),
                "teacher_model_type": str(teacher_config.model_type),
                "teacher_epochs": int(teacher_config.epochs),
                "replicate_count": int(len(distilled_label_sets)),
                "replicate_target_aggregation": range_replicate_target_aggregation,
                "aggregation": aggregation_diagnostics,
                "per_replicate": per_teacher,
                "labelled_point_count": int(labelled_mask[:, QUERY_TYPE_ID_RANGE].sum().item()),
                "positive_label_count": int(positive.sum().item()),
                "positive_label_fraction": float(positive.sum().item() / max(1, int(labels.shape[0]))),
                "positive_label_mass": (
                    float(labels[positive, QUERY_TYPE_ID_RANGE].sum().item()) if bool(positive.any().item()) else 0.0
                ),
                "budget_loss_ratios": list(getattr(config.model, "budget_loss_ratios", [])),
                "mlqds_temporal_fraction": float(getattr(config.model, "mlqds_temporal_fraction", 0.0)),
                "mlqds_hybrid_mode": str(getattr(config.model, "mlqds_hybrid_mode", "fill")),
            }
            range_training_label_aggregation.update(aggregation_diagnostics)
            range_training_label_aggregation["enabled"] = True
            range_training_label_aggregation["target_mode"] = "teacher_distillation"
            range_training_label_aggregation["replicate_target_aggregation"] = range_replicate_target_aggregation
            print(
                f"  distilled range labels: replicate_count={len(distilled_label_sets)} "
                f"positives={teacher_distillation_diagnostics['positive_label_count']} "
                f"fraction={teacher_distillation_diagnostics['positive_label_fraction']:.4f} "
                f"mass={teacher_distillation_diagnostics['positive_label_mass']:.4f}",
                flush=True,
            )
    elif range_training_target_mode in {
        "query_spine_frequency",
        "query_residual_frequency",
        "set_utility_frequency",
        "local_swap_utility_frequency",
        "local_swap_gain_cost_frequency",
    }:
        if train_labels is None:
            raise RuntimeError(f"{range_training_target_mode} target mode requires precomputed range training labels.")
        if len(train_label_sets) > 1:
            raise RuntimeError(f"{range_training_target_mode} does not yet support multiple train workload replicates.")
        target_phase = range_training_target_mode.replace("_", "-")
        with _phase(f"range-{target_phase}-target"):
            labels, labelled_mask = train_labels
            target_fn = (
                range_local_swap_gain_cost_frequency_training_labels
                if range_training_target_mode == "local_swap_gain_cost_frequency"
                else (
                    range_local_swap_utility_frequency_training_labels
                    if range_training_target_mode == "local_swap_utility_frequency"
                    else (
                        range_set_utility_frequency_training_labels
                        if range_training_target_mode == "set_utility_frequency"
                        else (
                            range_query_residual_frequency_training_labels
                            if range_training_target_mode == "query_residual_frequency"
                            else range_query_spine_frequency_training_labels
                        )
                    )
                )
            )
            labels, labelled_mask, range_training_target_transform = target_fn(
                labels=labels,
                labelled_mask=labelled_mask,
                points=train_points,
                boundaries=train_boundaries,
                typed_queries=train_workload.typed_queries,
                model_config=config.model,
            )
            range_training_target_transform["enabled"] = True
            range_training_target_transform["replicate_count"] = len(train_label_sets)
            train_labels = (labels, labelled_mask)
            print(
                f"  {target_phase} target: "
                f"positives={range_training_target_transform['positive_label_count']} "
                f"fraction={range_training_target_transform['positive_label_fraction']:.4f} "
                f"mass={range_training_target_transform['positive_label_mass']:.4f}",
                flush=True,
            )
    elif range_training_target_mode in {
        "retained_frequency",
        "global_budget_retained_frequency",
        "marginal_coverage_frequency",
        "historical_prior_retained_frequency",
        "structural_retained_frequency",
    }:
        if train_labels is None:
            raise RuntimeError(
                f"{range_training_target_mode} target mode requires precomputed range training labels."
            )
        target_fn = (
            range_marginal_coverage_training_labels
            if range_training_target_mode == "marginal_coverage_frequency"
            else range_global_budget_retained_frequency_training_labels
            if range_training_target_mode == "global_budget_retained_frequency"
            else range_structural_retained_frequency_training_labels
            if range_training_target_mode == "structural_retained_frequency"
            else range_historical_prior_retained_frequency_training_labels
            if range_training_target_mode == "historical_prior_retained_frequency"
            else range_retained_frequency_training_labels
        )
        aggregate_target_fn = (
            aggregate_range_marginal_coverage_training_labels
            if range_training_target_mode == "marginal_coverage_frequency"
            else aggregate_range_global_budget_retained_frequency_training_labels
            if range_training_target_mode == "global_budget_retained_frequency"
            else aggregate_range_structural_retained_frequency_training_labels
            if range_training_target_mode == "structural_retained_frequency"
            else aggregate_range_retained_frequency_training_labels
        )
        target_phase = range_training_target_mode.replace("_", "-")
        with _phase(f"range-{target_phase}-target"):
            if len(train_label_sets) > 1:
                if range_replicate_target_aggregation == "frequency_mean":
                    if range_training_target_mode == "historical_prior_retained_frequency":
                        raise RuntimeError(
                            "historical_prior_retained_frequency does not support "
                            "range_replicate_target_aggregation='frequency_mean'; use label_mean or label_max."
                        )
                    aggregate_target_kwargs = {
                        "label_sets": train_label_sets,
                        "boundaries": train_boundaries,
                        "model_config": config.model,
                    }
                    if range_training_target_mode == "structural_retained_frequency":
                        aggregate_target_kwargs["points"] = train_points
                    labels, labelled_mask, range_training_target_transform = aggregate_target_fn(
                        **aggregate_target_kwargs
                    )
                    range_training_label_aggregation["enabled"] = True
                    range_training_label_aggregation["target_mode"] = range_training_target_mode
                    range_training_label_aggregation["replicate_target_aggregation"] = "frequency_mean"
                else:
                    labels, labelled_mask, aggregation_diagnostics = aggregate_range_label_sets(
                        label_sets=train_label_sets,
                        source=(
                            f"range_label_{'max' if range_replicate_target_aggregation == 'label_max' else 'mean'}"
                            f"_before_{range_training_target_mode}"
                        ),
                        aggregation="max" if range_replicate_target_aggregation == "label_max" else "mean",
                    )
                    range_training_label_aggregation.update(aggregation_diagnostics)
                    range_training_label_aggregation["enabled"] = True
                    range_training_label_aggregation["target_mode"] = range_training_target_mode
                    range_training_label_aggregation["replicate_target_aggregation"] = (
                        range_replicate_target_aggregation
                    )
                    target_kwargs = {
                        "labels": labels,
                        "labelled_mask": labelled_mask,
                        "boundaries": train_boundaries,
                        "model_config": config.model,
                    }
                    if range_training_target_mode in {
                        "historical_prior_retained_frequency",
                        "structural_retained_frequency",
                    }:
                        target_kwargs["points"] = train_points
                    labels, labelled_mask, range_training_target_transform = target_fn(**target_kwargs)
                    range_training_target_transform["label_aggregation"] = aggregation_diagnostics
                range_training_target_transform["replicate_target_aggregation"] = (
                    range_replicate_target_aggregation
                )
            else:
                labels, labelled_mask = train_labels
                target_kwargs = {
                    "labels": labels,
                    "labelled_mask": labelled_mask,
                    "boundaries": train_boundaries,
                    "model_config": config.model,
                }
                if range_training_target_mode in {
                    "historical_prior_retained_frequency",
                    "structural_retained_frequency",
                }:
                    target_kwargs["points"] = train_points
                labels, labelled_mask, range_training_target_transform = target_fn(**target_kwargs)
            range_training_target_transform["enabled"] = True
            range_training_target_transform["replicate_count"] = len(train_label_sets)
            train_labels = (labels, labelled_mask)
            print(
                f"  {target_phase} target: positives={range_training_target_transform['positive_label_count']} "
                f"fraction={range_training_target_transform['positive_label_fraction']:.4f} "
                f"mass={range_training_target_transform['positive_label_mass']:.4f}",
                flush=True,
            )
    elif range_training_target_mode != "point_value":
        if range_training_target_mode in {"component_retained_frequency", "continuity_retained_frequency"}:
            if train_labels is None:
                raise RuntimeError(
                    f"{range_training_target_mode} target mode requires precomputed range training labels."
                )
            if not train_component_label_sets or any(component_labels is None for component_labels in train_component_label_sets):
                raise RuntimeError(
                    f"{range_training_target_mode} requires range component labels; use range_label_mode=usefulness."
                )
            target_fn = (
                range_continuity_retained_frequency_training_labels
                if range_training_target_mode == "continuity_retained_frequency"
                else range_component_retained_frequency_training_labels
            )
            aggregate_target_fn = (
                aggregate_range_continuity_retained_frequency_training_labels
                if range_training_target_mode == "continuity_retained_frequency"
                else aggregate_range_component_retained_frequency_training_labels
            )
            target_phase = range_training_target_mode.replace("_", "-")
            with _phase(f"range-{target_phase}-target"):
                if len(train_label_sets) > 1:
                    if range_replicate_target_aggregation == "frequency_mean":
                        labels, labelled_mask, range_training_target_transform = (
                            aggregate_target_fn(
                                label_sets=train_label_sets,
                                component_label_sets=train_component_label_sets,
                                boundaries=train_boundaries,
                                model_config=config.model,
                            )
                        )
                        range_training_label_aggregation["replicate_target_aggregation"] = "frequency_mean"
                    else:
                        aggregation_mode = "max" if range_replicate_target_aggregation == "label_max" else "mean"
                        labels, labelled_mask, component_labels, aggregation_diagnostics = (
                            aggregate_range_component_label_sets(
                                label_sets=train_label_sets,
                                component_label_sets=train_component_label_sets,
                                aggregation=aggregation_mode,
                            )
                        )
                        range_training_label_aggregation.update(aggregation_diagnostics)
                        range_training_label_aggregation["replicate_target_aggregation"] = (
                            range_replicate_target_aggregation
                        )
                        labels, labelled_mask, range_training_target_transform = (
                            target_fn(
                                labels=labels,
                                labelled_mask=labelled_mask,
                                component_labels=component_labels,
                                boundaries=train_boundaries,
                                model_config=config.model,
                            )
                        )
                        range_training_target_transform["label_aggregation"] = aggregation_diagnostics
                    range_training_label_aggregation["enabled"] = True
                    range_training_label_aggregation["target_mode"] = range_training_target_mode
                else:
                    labels, labelled_mask = train_labels
                    component_labels = train_component_label_sets[0]
                    if component_labels is None:
                        raise RuntimeError("component_retained_frequency requires component labels.")
                    labels, labelled_mask, range_training_target_transform = (
                        target_fn(
                            labels=labels,
                            labelled_mask=labelled_mask,
                            component_labels=component_labels,
                            boundaries=train_boundaries,
                            model_config=config.model,
                        )
                    )
                range_training_target_transform["enabled"] = True
                range_training_target_transform["replicate_count"] = len(train_label_sets)
                range_training_target_transform["replicate_target_aggregation"] = range_replicate_target_aggregation
                train_labels = (labels, labelled_mask)
                print(
                    f"  {target_phase} target: "
                    f"positives={range_training_target_transform['positive_label_count']} "
                    f"fraction={range_training_target_transform['positive_label_fraction']:.4f} "
                    f"mass={range_training_target_transform['positive_label_mass']:.4f}",
                    flush=True,
                )
        else:
            raise RuntimeError(
                "range_training_target_mode must be 'point_value', 'retained_frequency', "
                "'global_budget_retained_frequency', 'historical_prior_retained_frequency', "
                "'marginal_coverage_frequency', 'query_spine_frequency', "
                "'query_residual_frequency', 'set_utility_frequency', 'local_swap_utility_frequency', "
                "'local_swap_gain_cost_frequency', 'structural_retained_frequency', "
                "'component_retained_frequency', or "
                "'continuity_retained_frequency', or 'query_useful_v1_factorized'."
            )
    range_target_balance_mode = str(getattr(config.model, "range_target_balance_mode", "none")).lower()
    if range_target_balance_mode != "none":
        if train_labels is None:
            raise RuntimeError("range_target_balance_mode requires precomputed range training labels.")
        with _phase("range-target-balance"):
            labels, labelled_mask = train_labels
            labels, labelled_mask, range_target_balance_diagnostics = balance_range_training_target_by_trajectory(
                labels=labels,
                labelled_mask=labelled_mask,
                boundaries=train_boundaries,
                mode=range_target_balance_mode,
            )
            train_labels = (labels, labelled_mask)
            print(
                f"  target balance={range_target_balance_diagnostics['mode']} "
                f"positives={range_target_balance_diagnostics['positive_label_count']} "
                f"mass={range_target_balance_diagnostics['positive_label_mass']:.4f} "
                f"trajectories={range_target_balance_diagnostics['balanced_trajectory_count']}",
                flush=True,
            )
    if range_training_target_mode != "query_useful_v1_factorized":
        range_training_target_transform.setdefault("target_family", "legacy_range_useful_scalar")
        range_training_target_transform.setdefault("final_success_allowed", False)
        range_training_target_transform.setdefault(
            "legacy_reason",
            "Old RangeUseful/scalar-target diagnostic path. "
            "Not valid for query-driven rework acceptance.",
        )
    with _phase(f"train-model ({config.model.epochs} epochs)"):
        trained = train_model(
            train_trajectories=train_traj,
            train_boundaries=train_boundaries,
            workload=train_workload,
            model_config=config.model,
            seed=seeds.torch_seed,
            train_workload_map=train_workload_map,
            validation_trajectories=selection_traj,
            validation_boundaries=selection_boundaries,
            validation_workload=selection_workload,
            validation_workload_map=eval_workload_map if selection_workload is not None else None,
            precomputed_labels=train_labels,
            validation_points=selection_points,
            precomputed_validation_query_cache=selection_query_cache,
            precomputed_validation_geometry_scores=selection_geometry_scores,
            train_trajectory_source_ids=train_source_ids,
            train_trajectory_mmsis=train_mmsis,
        )
    training_cuda_memory = cuda_memory_snapshot()
    if training_cuda_memory.get("available"):
        print(
            f"  train_cuda_peak_allocated={training_cuda_memory['max_allocated_mb']:.1f} MiB  "
            f"peak_reserved={training_cuda_memory['max_reserved_mb']:.1f} MiB",
            flush=True,
        )

    if save_model:
        with _phase("save-model"):
            artifacts = ModelArtifacts(
                model=trained.model,
                scaler=trained.scaler,
                config=config,
                epochs_trained=trained.epochs_trained,
                workload_type=single_workload_type(eval_workload_map),
            )
            save_checkpoint(save_model, artifacts)
            print(
                f"  saved checkpoint to {save_model}  "
                f"(epochs_trained={trained.epochs_trained}, "
                f"best_epoch={trained.best_epoch}, best_loss={trained.best_loss:.8f}, "
                f"workload={workload_name(eval_workload_map)})",
                flush=True,
            )
    methods = build_primary_methods(
        trained=trained,
        eval_workload=eval_workload,
        eval_workload_map=eval_workload_map,
        config=config,
        trajectory_mmsis=test_mmsis,
    )
    retention_methods = list(methods)
    workload_blind_eval = is_workload_blind_model_type(config.model.model_type)
    audit_ratios = _range_audit_ratios(config)
    selector_budget_ratios = tuple(
        sorted({float(config.model.compression_ratio), *(float(ratio) for ratio in audit_ratios)})
    )
    selector_budget_diagnostics = {
        "train": temporal_hybrid_selector_budget_diagnostics(
            train_boundaries,
            selector_budget_ratios,
            temporal_fraction=float(config.model.mlqds_temporal_fraction),
            hybrid_mode=str(config.model.mlqds_hybrid_mode),
            min_learned_swaps=int(config.model.mlqds_min_learned_swaps),
        ),
        "eval": temporal_hybrid_selector_budget_diagnostics(
            test_boundaries,
            selector_budget_ratios,
            temporal_fraction=float(config.model.mlqds_temporal_fraction),
            hybrid_mode=str(config.model.mlqds_hybrid_mode),
            min_learned_swaps=int(config.model.mlqds_min_learned_swaps),
        ),
    }
    frozen_primary_masks: dict[str, torch.Tensor] = {}
    frozen_audit_methods_by_ratio: dict[str, list[Method]] = {}
    if workload_blind_eval:
        with _phase("freeze-retained-masks"):
            for method in methods:
                with _phase(f"  freeze {method.name}"):
                    freeze_t0 = time.perf_counter()
                    frozen_primary_masks[method.name] = method.simplify(
                        test_points,
                        test_boundaries,
                        config.model.compression_ratio,
                    ).detach().cpu()
                    setattr(method, "latency_ms", float((time.perf_counter() - freeze_t0) * 1000.0))
        methods = [
            FrozenMaskMethod(
                name=method.name,
                retained_mask=frozen_primary_masks[method.name],
                latency_ms=float(getattr(method, "latency_ms", 0.0)),
            )
            for method in methods
        ]
        print(
            "  workload_blind_protocol=enabled: primary retained masks frozen before eval query scoring",
            flush=True,
        )
        if audit_ratios:
            with _phase("freeze-audit-retained-masks"):
                for ratio in audit_ratios:
                    if abs(float(ratio) - float(config.model.compression_ratio)) <= 1e-9:
                        continue
                    ratio_key = f"{float(ratio):.4f}"
                    frozen_ratio_methods: list[Method] = []
                    for method in retention_methods:
                        with _phase(f"  freeze audit {method.name} ratio={ratio:.4f}"):
                            freeze_t0 = time.perf_counter()
                            retained_mask = method.simplify(
                                test_points,
                                test_boundaries,
                                float(ratio),
                            ).detach().cpu()
                            frozen_ratio_methods.append(
                                FrozenMaskMethod(
                                    name=method.name,
                                    retained_mask=retained_mask,
                                    latency_ms=float((time.perf_counter() - freeze_t0) * 1000.0),
                                )
                            )
                    frozen_audit_methods_by_ratio[ratio_key] = frozen_ratio_methods
            print(
                "  workload_blind_protocol=enabled: audit retained masks frozen before eval query scoring",
                flush=True,
            )

    matched: dict[str, MethodEvaluation] = {}
    oracle_method: OracleMethod | None = None
    eval_labels: torch.Tensor | None = None
    save_masks = bool(save_simplified_dir)
    eval_is_range_only = len(range_only_queries(eval_workload.typed_queries)) == len(eval_workload.typed_queries)
    final_metrics_mode = str(getattr(config.baselines, "final_metrics_mode", "diagnostic")).lower()
    if final_metrics_mode not in {"diagnostic", "core"}:
        raise ValueError("final_metrics_mode must be either 'diagnostic' or 'core'.")
    run_final_diagnostics = final_metrics_mode == "diagnostic"
    run_oracle_baseline = bool(config.baselines.include_oracle and run_final_diagnostics)
    run_learned_fill_diagnostics = bool(eval_is_range_only and run_final_diagnostics)
    with _phase("eval-query-cache-prep"):
        eval_query_cache = prepare_eval_query_cache(
            test_points=test_points,
            test_boundaries=test_boundaries,
            eval_workload=eval_workload,
            eval_is_range_only=eval_is_range_only,
            runtime_cache=range_runtime_caches["eval"],
        )
    if run_oracle_baseline or run_learned_fill_diagnostics or mlqds_range_geometry_blend > 0.0:
        with _phase("eval-label-prep"):
            eval_labels = prepare_eval_labels(
                test_points=test_points,
                test_boundaries=test_boundaries,
                eval_workload=eval_workload,
                eval_workload_map=eval_workload_map,
                config=config,
                seeds=seeds,
                eval_is_range_only=eval_is_range_only,
                run_oracle_baseline=run_oracle_baseline,
                runtime_cache=range_runtime_caches["eval"],
            )
    if mlqds_range_geometry_blend > 0.0:
        if eval_labels is None:
            raise RuntimeError("MLQDS range geometry blend requested but eval labels were not prepared.")
        attach_range_geometry_scores(
            methods=methods,
            eval_labels=eval_labels,
            eval_workload_map=eval_workload_map,
        )
    with _phase("evaluate-matched"):
        for method in methods:
            with _phase(f"  eval {method.name}"):
                matched[method.name] = evaluate_method(
                    method=method,
                    points=test_points,
                    boundaries=test_boundaries,
                    typed_queries=eval_workload.typed_queries,
                    workload_map=eval_workload_map,
                    compression_ratio=config.model.compression_ratio,
                    return_mask=method.name == "MLQDS" or save_masks,
                    query_cache=eval_query_cache,
                )

        if run_oracle_baseline:
            if eval_labels is None:
                raise RuntimeError("Oracle baseline requested but eval labels were not prepared.")
            oracle_method = OracleMethod(labels=eval_labels, workload_type=single_workload_type(eval_workload_map))
            with _phase(f"  eval {oracle_method.name}"):
                matched[oracle_method.name] = evaluate_method(
                    method=oracle_method,
                    points=test_points,
                    boundaries=test_boundaries,
                    typed_queries=eval_workload.typed_queries,
                    workload_map=eval_workload_map,
                    compression_ratio=config.model.compression_ratio,
                    query_cache=eval_query_cache,
                )

    learned_fill_diagnostics: dict[str, MethodEvaluation] = {"MLQDS": matched["MLQDS"]}
    learned_fill_table = ""
    diagnostic_methods: list[Method] = []
    if run_learned_fill_diagnostics:
        if eval_labels is None:
            raise RuntimeError("Learned-fill diagnostics requested but eval labels were not prepared.")
        diagnostic_methods = build_learned_fill_methods(
            test_points=test_points,
            eval_labels=eval_labels,
            eval_workload_map=eval_workload_map,
            config=config,
            seeds=seeds,
        )
        with _phase("learned-fill-diagnostics"):
            for method in diagnostic_methods:
                with _phase(f"  fill {method.name}"):
                    learned_fill_diagnostics[method.name] = evaluate_method(
                        method=method,
                        points=test_points,
                        boundaries=test_boundaries,
                        typed_queries=eval_workload.typed_queries,
                        workload_map=eval_workload_map,
                        compression_ratio=config.model.compression_ratio,
                        query_cache=eval_query_cache,
                    )
        learned_fill_table = print_range_usefulness_table(learned_fill_diagnostics)

    matched_table = print_method_comparison_table(matched)
    geometric_table = print_geometric_distortion_table(matched)
    range_usefulness_table = print_range_usefulness_table(matched)
    range_compression_audit: dict[str, dict[str, Any]] = {}
    range_compression_audit_table = ""
    if audit_ratios:
        audit_methods = [*(retention_methods if workload_blind_eval else methods), *diagnostic_methods]
        if oracle_method is not None:
            audit_methods.append(oracle_method)
        audit_sections: list[str] = []
        with _phase("range-compression-audit"):
            for ratio in audit_ratios:
                if abs(float(ratio) - float(config.model.compression_ratio)) <= 1e-9:
                    ratio_results = {
                        **matched,
                        **{
                            name: metrics
                            for name, metrics in learned_fill_diagnostics.items()
                            if name not in matched
                        },
                    }
                else:
                    ratio_results: dict[str, MethodEvaluation] = {}
                    ratio_key = f"{float(ratio):.4f}"
                    ratio_audit_methods = audit_methods
                    if workload_blind_eval and ratio_key in frozen_audit_methods_by_ratio:
                        ratio_audit_methods = [*frozen_audit_methods_by_ratio[ratio_key], *diagnostic_methods]
                        if oracle_method is not None:
                            ratio_audit_methods.append(oracle_method)
                    for method in ratio_audit_methods:
                        with _phase(f"  audit {method.name} ratio={ratio:.4f}"):
                            ratio_results[method.name] = evaluate_method(
                                method=method,
                                points=test_points,
                                boundaries=test_boundaries,
                                typed_queries=eval_workload.typed_queries,
                                workload_map=eval_workload_map,
                                compression_ratio=float(ratio),
                                query_cache=eval_query_cache,
                            )
                ratio_key = f"{float(ratio):.4f}"
                range_compression_audit[ratio_key] = {
                    name: _evaluation_metrics_payload(metrics) for name, metrics in ratio_results.items()
                }
                audit_sections.append(f"compression_ratio={ratio_key}\n{print_range_usefulness_table(ratio_results)}")
        range_compression_audit_table = "\n\n".join(audit_sections)

    with _phase("evaluate-shift"):
        shift_pairs = evaluate_shift_pairs(
            matched_mlqds_score=float(matched["MLQDS"].aggregate_f1),
            trained=trained,
            train_workload=train_workload,
            train_workload_map=train_workload_map,
            eval_workload_map=eval_workload_map,
            config=config,
        test_points=test_points,
        test_boundaries=test_boundaries,
        test_mmsis=test_mmsis,
    )
    shift_table = print_shift_table(shift_pairs)

    with _phase("range-diagnostics"):
        train_summary, train_rows = _range_workload_diagnostics(
            "train",
            train_points,
            train_boundaries,
            train_workload,
            train_workload_map,
            config,
            seeds.train_query_seed,
            range_runtime_caches["train"],
        )
        eval_summary, eval_rows = _range_workload_diagnostics(
            "eval",
            test_points,
            test_boundaries,
            eval_workload,
            eval_workload_map,
            config,
            seeds.eval_query_seed,
            range_runtime_caches["eval"],
        )
        range_diagnostics_summary["train"] = train_summary
        range_diagnostics_summary["eval"] = eval_summary
        range_diagnostics_rows.extend(train_rows)
        range_diagnostics_rows.extend(eval_rows)
        if selection_workload is not None and selection_points is not None and selection_boundaries is not None:
            selection_summary, selection_rows = _range_workload_diagnostics(
                "selection",
                selection_points,
                selection_boundaries,
                selection_workload,
                eval_workload_map,
                config,
                seeds.eval_query_seed + 17,
                range_runtime_caches["selection"],
            )
            range_diagnostics_summary["selection"] = selection_summary
            range_diagnostics_rows.extend(selection_rows)
        _print_range_diagnostics_summary(range_diagnostics_summary)
        workload_distribution_comparison = _range_workload_distribution_comparison(range_diagnostics_summary)
        _print_range_distribution_comparison(workload_distribution_comparison)

    range_learned_fill_summary = _range_learned_fill_summary(
        learned_fill_diagnostics=learned_fill_diagnostics,
        training_target_diagnostics=trained.target_diagnostics,
        range_diagnostics_summary=range_diagnostics_summary,
        compression_ratio=float(config.model.compression_ratio),
    )
    final_claim_summary = {
        "primary_metric": None,
        "status": "not_available_until_query_useful_v1",
        "final_success_allowed": False,
        "reason": "Old RangeUseful/RangeUsefulLegacy outputs are diagnostic only.",
    }
    uniform_eval = matched.get("uniform")
    douglas_peucker_eval = matched.get("DouglasPeucker")
    legacy_range_useful_summary = {
        "metric": "RangeUsefulLegacy",
        "schema": "range_usefulness_schema_version",
        "diagnostic_only": True,
        "mlqds_score": matched["MLQDS"].range_usefulness_score,
        "uniform_score": uniform_eval.range_usefulness_score if uniform_eval is not None else None,
        "douglas_peucker_score": (
            douglas_peucker_eval.range_usefulness_score
            if douglas_peucker_eval is not None
            else None
        ),
    }
    learning_causality_summary = {
        "selector_diagnostics_present": bool(selector_budget_diagnostics),
        "training_fit_diagnostics_present": bool(trained.fit_diagnostics),
        "legacy_temporal_hybrid_selector": True,
        "final_success_allowed": False,
    }

    dump = {
        "config": config.to_dict(),
        "final_claim_summary": final_claim_summary,
        "diagnostic_summary": {
            "legacy_range_useful_available": True,
            "range_component_diagnostics_available": True,
            "workload_blind_protocol_available": True,
        },
        "legacy_range_useful_summary": legacy_range_useful_summary,
        "learning_causality_summary": learning_causality_summary,
        "workload": single_workload_type(eval_workload_map),
        "train_query_count": len(train_workload.typed_queries),
        "train_label_workload_count": len(train_label_workloads),
        "train_label_workload_query_counts": [len(workload.typed_queries) for workload in train_label_workloads],
        "eval_query_count": len(eval_workload.typed_queries),
        "selection_query_count": len(selection_workload.typed_queries) if selection_workload is not None else None,
        "train_query_coverage": train_workload.coverage_fraction,
        "train_label_workload_coverages": [workload.coverage_fraction for workload in train_label_workloads],
        "eval_query_coverage": eval_workload.coverage_fraction,
        "selection_query_coverage": selection_workload.coverage_fraction if selection_workload is not None else None,
        "query_generation_diagnostics": {
            "train": train_workload.generation_diagnostics,
            "train_label_workloads": [workload.generation_diagnostics for workload in train_label_workloads],
            "eval": eval_workload.generation_diagnostics,
            "selection": selection_workload.generation_diagnostics if selection_workload is not None else None,
        },
        "data_split_diagnostics": data_split.split_diagnostics,
        "selector_budget_diagnostics": selector_budget_diagnostics,
        "matched": {name: _evaluation_metrics_payload(m) for name, m in matched.items()},
        "learned_fill_diagnostics": {
            name: _evaluation_metrics_payload(metrics) for name, metrics in learned_fill_diagnostics.items()
        },
        "range_learned_fill_summary": range_learned_fill_summary,
        "range_compression_audit": range_compression_audit,
        "shift": shift_pairs,
        "training_history": trained.history,
        "training_target_diagnostics": trained.target_diagnostics,
        "training_fit_diagnostics": trained.fit_diagnostics,
        "range_training_target_transform": range_training_target_transform,
        "model_metadata": model_type_metadata(config.model.model_type),
        "range_target_balance": range_target_balance_diagnostics,
        "range_training_label_aggregation": range_training_label_aggregation,
        "teacher_distillation": teacher_distillation_diagnostics,
        "best_epoch": trained.best_epoch,
        "best_loss": trained.best_loss,
        "best_selection_score": trained.best_selection_score,
        "checkpoint_selection_metric": selection_metric,
        "checkpoint_selection_metric_requested": config.model.checkpoint_selection_metric,
        "checkpoint_score_variant": config.model.checkpoint_score_variant,
        "final_metrics_mode": config.baselines.final_metrics_mode,
        "workload_blind_protocol": {
            "enabled": bool(workload_blind_eval),
            "model_type": config.model.model_type,
            "primary_masks_frozen_before_eval_query_scoring": bool(workload_blind_eval),
            "audit_masks_frozen_before_eval_query_scoring": bool(
                workload_blind_eval and bool(frozen_audit_methods_by_ratio)
            ),
            "frozen_audit_ratio_count": int(len(frozen_audit_methods_by_ratio)),
            "frozen_method_names": sorted(frozen_primary_masks),
            "frozen_audit_ratios": sorted(frozen_audit_methods_by_ratio),
            "eval_geometry_blend_allowed": not bool(workload_blind_eval),
        },
        "range_usefulness_weight_summary": range_usefulness_weight_summary(),
        "checkpoint_smoothing_window": config.model.checkpoint_smoothing_window,
        "mlqds_score_mode": config.model.mlqds_score_mode,
        "mlqds_score_temperature": config.model.mlqds_score_temperature,
        "mlqds_rank_confidence_weight": config.model.mlqds_rank_confidence_weight,
        "mlqds_range_geometry_blend": config.model.mlqds_range_geometry_blend,
        "mlqds_hybrid_mode": config.model.mlqds_hybrid_mode,
        "mlqds_stratified_center_weight": config.model.mlqds_stratified_center_weight,
        "mlqds_min_learned_swaps": config.model.mlqds_min_learned_swaps,
        "oracle_diagnostic": {
            "kind": "additive_label_greedy",
            "enabled": run_oracle_baseline,
            "exact_optimum": False,
            "retained_mask_constructor": "per_trajectory_topk_with_endpoints",
            "purpose": "diagnostic label-greedy reference, not exact retained-set RangeUseful optimum",
        },
        "range_label_mode": config.model.range_label_mode,
        "range_boundary_prior_weight": config.model.range_boundary_prior_weight,
        "range_boundary_prior_enabled": config.model.range_boundary_prior_weight > 0.0,
        "data_audit": data_audit,
        "workload_diagnostics": range_diagnostics_summary,
        "workload_distribution_comparison": workload_distribution_comparison,
        "torch_runtime": {
            **torch_runtime_snapshot(),
            "amp": amp_runtime_snapshot(config.model.amp_mode),
        },
        "cuda_memory": {
            "training": training_cuda_memory,
        },
    }

    with _phase("write-results"):
        out_dir = write_experiment_results(
            results_dir=results_dir,
            matched_table=matched_table,
            shift_table=shift_table,
            geometric_table=geometric_table,
            range_usefulness_table=range_usefulness_table,
            learned_fill_table=learned_fill_table,
            learned_fill_diagnostics=learned_fill_diagnostics,
            range_learned_fill_summary=range_learned_fill_summary,
            range_compression_audit=range_compression_audit,
            range_compression_audit_table=range_compression_audit_table,
            range_diagnostics_summary=range_diagnostics_summary,
            workload_distribution_comparison=workload_distribution_comparison,
            range_diagnostics_rows=range_diagnostics_rows,
            dump=dump,
        )
        print(f"  wrote results to {out_dir}", flush=True)

    if save_simplified_dir:
        with _phase("write-simplified-csv"):
            out_dir = Path(save_simplified_dir)
            eval_mask = matched["MLQDS"].retained_mask
            if eval_mask is None:
                eval_mlqds = MLQDSMethod(
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
                    stratified_center_weight=config.model.mlqds_stratified_center_weight,
                    min_learned_swaps=config.model.mlqds_min_learned_swaps,
                    trajectory_mmsis=test_mmsis,
                    inference_batch_size=config.model.inference_batch_size,
                    amp_mode=config.model.amp_mode,
                )
                eval_mask = eval_mlqds.simplify(test_points, test_boundaries, config.model.compression_ratio)
            write_simplified_csv(
                str(out_dir / "ML_simplified_eval.csv"),
                test_points,
                test_boundaries,
                eval_mask,
                trajectory_mmsis=test_mmsis,
            )
            for ref_name, csv_name in (("uniform", "uniform_simplified_eval.csv"),
                                       ("DouglasPeucker", "DP_simplified_eval.csv")):
                ref_eval = matched.get(ref_name)
                ref_mask = ref_eval.retained_mask if ref_eval is not None else None
                if ref_mask is not None:
                    write_simplified_csv(
                        str(out_dir / csv_name),
                        test_points,
                        test_boundaries,
                        ref_mask,
                        trajectory_mmsis=test_mmsis,
                    )

        with _phase("trajectory-length-loss"):
            report_trajectory_length_loss(
                test_points,
                test_boundaries,
                eval_mask,
                top_k=25,
                trajectory_mmsis=test_mmsis,
            )

    print(f"[pipeline] total runtime {time.perf_counter() - pipeline_t0:.2f}s", flush=True)
    return ExperimentOutputs(
        matched_table=matched_table,
        shift_table=shift_table,
        metrics_dump=dump,
        geometric_table=geometric_table,
        range_usefulness_table=range_usefulness_table,
        range_compression_audit_table=range_compression_audit_table,
    )
