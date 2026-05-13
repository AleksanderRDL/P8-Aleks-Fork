"""Experiment orchestration helpers for training and evaluation runs. See experiments/README.md for details."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from data.trajectory_dataset import TrajectoryDataset
from evaluation.baselines import (
    DouglasPeuckerMethod,
    Method,
    MLQDSMethod,
    OracleMethod,
    ScoreHybridMethod,
    UniformTemporalMethod,
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
from experiments.geojson_writers import report_trajectory_length_loss, write_queries_geojson, write_simplified_csv
from experiments.range_cache import (
    RangeRuntimeCache,
    prepare_range_label_cache,
    prepare_range_training_cache,
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
from experiments.workload_cache import (
    coverage_name,
    generate_typed_query_workload_for_config,
    workload_cache_name,
)
from queries.query_types import single_workload_type
from queries.workload import TypedQueryWorkload
from simplification.mlqds_scoring import workload_type_head
from training.importance_labels import compute_typed_importance_labels
from training.train_model import train_model
from training.checkpoints import ModelArtifacts, save_checkpoint
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


@dataclass
class ExperimentOutputs:
    """Experiment run output payload. See experiments/README.md for details."""

    matched_table: str
    shift_table: str
    metrics_dump: dict
    geometric_table: str = ""
    range_usefulness_table: str = ""
    range_compression_audit_table: str = ""


def split_trajectories(
    trajectories: list[torch.Tensor],
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Deterministically split trajectories at trajectory level. See experiments/README.md for details."""
    trajectory_count = len(trajectories)
    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(trajectory_count, generator=generator).tolist()

    train_count = max(1, int(trajectory_count * train_fraction))
    val_count = max(1, int(trajectory_count * val_fraction)) if trajectory_count - train_count > 1 else 0
    test_count = max(1, trajectory_count - train_count - val_count)
    if train_count + val_count + test_count > trajectory_count:
        test_count = trajectory_count - train_count - val_count

    train = [trajectories[i] for i in permutation[:train_count]]
    val = [trajectories[i] for i in permutation[train_count : train_count + val_count]]
    test = [trajectories[i] for i in permutation[train_count + val_count : train_count + val_count + test_count]]
    if not test:
        test = val if val else train
    return train, val, test


def _workload_name(workload_map: dict[str, float]) -> str:
    """Build compact string name for a pure workload map."""
    return ",".join(
        f"{query_type}={weight:.1f}"
        for query_type, weight in sorted(workload_map.items())
    )


def _normalized_coverage_target(value: float | None) -> float | None:
    """Normalize coverage target for pipeline warnings."""
    if value is None:
        return None
    target = float(value)
    return target / 100.0 if target > 1.0 else target


def _validation_query_count(config: ExperimentConfig) -> int:
    """Use the same minimum query count for validation and final eval workloads."""
    return max(1, int(config.query.n_queries))


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
    data_audit: dict[str, Any] | None = None,
) -> ExperimentOutputs:
    """Run training, matched evaluation, and shifted evaluation tables. See experiments/README.md for details."""
    pipeline_t0 = time.perf_counter()
    train_workload_map, eval_workload_map = resolve_workload_maps(config.query.workload)
    if eval_trajectories is None:
        print(
            f"[pipeline] {len(trajectories)} trajectories, workload={_workload_name(eval_workload_map)}",
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
            f"workload={_workload_name(eval_workload_map)}",
            flush=True,
        )

    seeds = derive_seed_bundle(config.data.seed)
    selection_metric = str(getattr(config.model, "checkpoint_selection_metric", "score")).lower()
    validation_score_every = int(getattr(config.model, "validation_score_every", 0) or 0)
    needs_validation_score = selection_metric in {"score", "uniform_gap"} or validation_score_every > 0
    with _phase("split"):
        selection_traj: list[torch.Tensor] | None = None
        if eval_trajectories is None:
            # Reproduce split_trajectories' permutation here so we can align the
            # MMSI list with the test split (the helper itself doesn't carry ids).
            trajectory_count = len(trajectories)
            generator = torch.Generator().manual_seed(int(seeds.split_seed))
            permutation = torch.randperm(trajectory_count, generator=generator).tolist()
            train_count = max(1, int(trajectory_count * config.data.train_fraction))
            val_count = (
                max(1, int(trajectory_count * config.data.val_fraction))
                if trajectory_count - train_count > 1
                else 0
            )
            train_traj = [trajectories[i] for i in permutation[:train_count]]
            _val_traj = [trajectories[i] for i in permutation[train_count : train_count + val_count]]
            test_traj = [trajectories[i] for i in permutation[train_count + val_count :]]
            if not test_traj:
                test_traj = _val_traj if _val_traj else train_traj
            selection_traj = _val_traj if needs_validation_score and _val_traj else None
            if trajectory_mmsis is not None and len(trajectory_mmsis) == trajectory_count:
                train_mmsis = [trajectory_mmsis[i] for i in permutation[:train_count]]
                test_mmsis = [trajectory_mmsis[i] for i in permutation[train_count + val_count :]]
                if not test_mmsis:
                    test_mmsis = [trajectory_mmsis[i] for i in permutation[train_count : train_count + val_count]] or \
                                 [trajectory_mmsis[i] for i in permutation[:train_count]]
            else:
                train_mmsis = None
                test_mmsis = None
            print(f"  split mode=single dataset  train={len(train_traj)}  test={len(test_traj)}", flush=True)
        else:
            train_traj = trajectories
            test_traj = eval_trajectories
            train_mmsis = trajectory_mmsis
            test_mmsis = eval_trajectory_mmsis
            if validation_trajectories is not None:
                selection_traj = validation_trajectories if needs_validation_score else None
            elif needs_validation_score:
                train_count = len(train_traj)
                generator = torch.Generator().manual_seed(int(seeds.split_seed))
                permutation = torch.randperm(train_count, generator=generator).tolist()
                val_count = max(1, int(train_count * config.data.val_fraction)) if train_count > 1 else 0
                val_indices = set(permutation[:val_count])
                selection_traj = [
                    trajectory
                    for trajectory_idx, trajectory in enumerate(train_traj)
                    if trajectory_idx in val_indices
                ]
                train_traj = [
                    trajectory
                    for trajectory_idx, trajectory in enumerate(train_traj)
                    if trajectory_idx not in val_indices
                ]
                if train_mmsis is not None and len(train_mmsis) == train_count:
                    train_mmsis = [
                        mmsi
                        for trajectory_idx, mmsi in enumerate(train_mmsis)
                        if trajectory_idx not in val_indices
                    ]
            print(f"  split mode=separate CSVs  train={len(train_traj)}  eval={len(test_traj)}", flush=True)
        if selection_traj:
            print(f"  checkpoint-selection validation={len(selection_traj)} trajectories", flush=True)

    with _phase("build-datasets"):
        train_ds = TrajectoryDataset(train_traj)
        test_ds = TrajectoryDataset(test_traj)
        selection_ds = TrajectoryDataset(selection_traj) if selection_traj else None
        train_points = train_ds.get_all_points()
        test_points = test_ds.get_all_points()
        selection_points = selection_ds.get_all_points() if selection_ds is not None else None
        train_boundaries = train_ds.get_trajectory_boundaries()
        test_boundaries = test_ds.get_trajectory_boundaries()
        selection_boundaries = selection_ds.get_trajectory_boundaries() if selection_ds is not None else None

    with _phase("generate-workloads"):
        # Front-load all kNN queries before proportional scheduling for training
        # so kNN always gets its full quota even if n_queries is small.
        knn_front_load = int(train_workload_map.get("knn", 0.0) * config.query.n_queries)
        train_workload = generate_typed_query_workload_for_config(
            trajectories=train_traj,
            n_queries=config.query.n_queries,
            workload_map=train_workload_map,
            seed=seeds.train_query_seed,
            front_load_knn=knn_front_load,
            config=config,
            points=train_points,
            boundaries=train_boundaries,
            cache_label="train",
        )
        eval_workload = generate_typed_query_workload_for_config(
            trajectories=test_traj,
            n_queries=config.query.n_queries,
            workload_map=eval_workload_map,
            seed=seeds.eval_query_seed,
            config=config,
            points=test_points,
            boundaries=test_boundaries,
            cache_label="eval",
        )
        selection_workload = None
        if selection_traj:
            selection_workload = generate_typed_query_workload_for_config(
                trajectories=selection_traj,
                n_queries=_validation_query_count(config),
                workload_map=eval_workload_map,
                seed=seeds.eval_query_seed + 17,
                config=config,
                points=selection_points,
                boundaries=selection_boundaries,
                cache_label="selection",
            )
        print(
            f"  train_workload={len(train_workload.typed_queries)} queries  "
            f"coverage={coverage_name(train_workload)}  cache={workload_cache_name(train_workload)}",
            flush=True,
        )
        print(
            f"  eval_workload={len(eval_workload.typed_queries)} queries  "
            f"coverage={coverage_name(eval_workload)}  cache={workload_cache_name(eval_workload)}",
            flush=True,
        )
        if selection_workload is not None:
            print(
                f"  selection_workload={len(selection_workload.typed_queries)} queries  "
                f"coverage={coverage_name(selection_workload)}  cache={workload_cache_name(selection_workload)}",
                flush=True,
            )
        target = _normalized_coverage_target(config.query.target_coverage)
        if target is not None:
            workloads_to_check = [("train", train_workload), ("eval", eval_workload)]
            if selection_workload is not None:
                workloads_to_check.append(("selection", selection_workload))
            for label, workload in workloads_to_check:
                coverage = float(workload.coverage_fraction or 0.0)
                if coverage + 1e-9 < target:
                    print(
                        f"  WARNING: {label} workload stopped below requested coverage "
                        f"({coverage:.2%} < {target:.2%}); raise --max_queries "
                        "or query footprint to cover more points.",
                        flush=True,
                    )
                elif label == "selection" and coverage > target + 0.05:
                    print(
                        f"  WARNING: {label} workload remains above requested coverage "
                        f"({coverage:.2%} > {target:.2%}); lower --n_queries or query footprint.",
                        flush=True,
                    )

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
    selection_query_cache: EvaluationQueryCache | None = None
    selection_geometry_scores: torch.Tensor | None = None
    mlqds_range_geometry_blend = max(0.0, min(1.0, float(getattr(config.model, "mlqds_range_geometry_blend", 0.0))))
    with _phase("range-training-prep"):
        train_labels = prepare_range_training_cache(
            points=train_points,
            boundaries=train_boundaries,
            workload=train_workload,
            workload_map=train_workload_map,
            config=config,
            seed=seeds.train_query_seed,
            runtime_cache=range_runtime_caches["train"],
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
                f"workload={_workload_name(eval_workload_map)})",
                flush=True,
            )
    methods = [
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
        eval_query_cache = (
            range_runtime_caches["eval"].query_cache
            if eval_is_range_only
            else None
        )
        if eval_query_cache is None:
            eval_query_cache = EvaluationQueryCache.for_workload(
                test_points,
                test_boundaries,
                eval_workload.typed_queries,
            )
            if eval_is_range_only:
                range_runtime_caches["eval"].query_cache = eval_query_cache
        else:
            eval_query_cache.validate(test_points, test_boundaries, eval_workload.typed_queries)
    if run_oracle_baseline or run_learned_fill_diagnostics or mlqds_range_geometry_blend > 0.0:
        with _phase("eval-label-prep"):
            if eval_is_range_only:
                prepared_eval_labels = prepare_range_label_cache(
                    cache_label="eval",
                    points=test_points,
                    boundaries=test_boundaries,
                    workload=eval_workload,
                    workload_map=eval_workload_map,
                    config=config,
                    seed=seeds.eval_query_seed,
                    runtime_cache=range_runtime_caches["eval"],
                    range_boundary_prior_weight=float(getattr(config.model, "range_boundary_prior_weight", 0.0)),
                )
                if prepared_eval_labels is not None:
                    eval_labels, _ = prepared_eval_labels
            elif run_oracle_baseline:
                eval_labels, _ = compute_typed_importance_labels(
                    points=test_points,
                    boundaries=test_boundaries,
                    typed_queries=eval_workload.typed_queries,
                    seed=seeds.eval_query_seed,
                    range_label_mode=str(getattr(config.model, "range_label_mode", "usefulness")),
                    range_boundary_prior_weight=float(getattr(config.model, "range_boundary_prior_weight", 0.0)),
                )
    if mlqds_range_geometry_blend > 0.0:
        if eval_labels is None:
            raise RuntimeError("MLQDS range geometry blend requested but eval labels were not prepared.")
        _, eval_type_id = workload_type_head(single_workload_type(eval_workload_map))
        mlqds_method = methods[0]
        if isinstance(mlqds_method, MLQDSMethod):
            mlqds_method.range_geometry_scores = eval_labels[:, eval_type_id].float()
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
        assert eval_labels is not None
        _, eval_type_id = workload_type_head(single_workload_type(eval_workload_map))
        random_generator = torch.Generator().manual_seed(int(seeds.eval_query_seed) + 404)
        random_scores = torch.rand((test_points.shape[0],), generator=random_generator)
        diagnostic_methods = [
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
    audit_ratios = _range_audit_ratios(config)
    if audit_ratios:
        audit_methods = [*methods, *diagnostic_methods]
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
                    for method in audit_methods:
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
        train_name = _workload_name(train_workload_map)
        eval_name = _workload_name(eval_workload_map)
        shift_pairs = {train_name: {eval_name: float(matched["MLQDS"].aggregate_f1)}}
        if train_name == eval_name:
            shift_pairs[train_name][train_name] = float(matched["MLQDS"].aggregate_f1)
        else:
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

    dump = {
        "config": config.to_dict(),
        "workload": single_workload_type(eval_workload_map),
        "train_query_count": len(train_workload.typed_queries),
        "eval_query_count": len(eval_workload.typed_queries),
        "selection_query_count": len(selection_workload.typed_queries) if selection_workload is not None else None,
        "train_query_coverage": train_workload.coverage_fraction,
        "eval_query_coverage": eval_workload.coverage_fraction,
        "selection_query_coverage": selection_workload.coverage_fraction if selection_workload is not None else None,
        "query_generation_diagnostics": {
            "train": train_workload.generation_diagnostics,
            "eval": eval_workload.generation_diagnostics,
            "selection": selection_workload.generation_diagnostics if selection_workload is not None else None,
        },
        "matched": {name: _evaluation_metrics_payload(m) for name, m in matched.items()},
        "learned_fill_diagnostics": {
            name: _evaluation_metrics_payload(metrics) for name, metrics in learned_fill_diagnostics.items()
        },
        "range_learned_fill_summary": range_learned_fill_summary,
        "range_compression_audit": range_compression_audit,
        "shift": shift_pairs,
        "training_history": trained.history,
        "training_target_diagnostics": trained.target_diagnostics,
        "best_epoch": trained.best_epoch,
        "best_loss": trained.best_loss,
        "best_selection_score": trained.best_selection_score,
        "checkpoint_selection_metric": selection_metric,
        "checkpoint_selection_metric_requested": config.model.checkpoint_selection_metric,
        "checkpoint_score_variant": config.model.checkpoint_score_variant,
        "final_metrics_mode": config.baselines.final_metrics_mode,
        "range_usefulness_weight_summary": range_usefulness_weight_summary(),
        "checkpoint_smoothing_window": config.model.checkpoint_smoothing_window,
        "mlqds_score_mode": config.model.mlqds_score_mode,
        "mlqds_score_temperature": config.model.mlqds_score_temperature,
        "mlqds_rank_confidence_weight": config.model.mlqds_rank_confidence_weight,
        "mlqds_range_geometry_blend": config.model.mlqds_range_geometry_blend,
        "mlqds_hybrid_mode": config.model.mlqds_hybrid_mode,
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
        out_dir = Path(results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "matched_table.txt").write_text(matched_table + "\n", encoding="utf-8")
        (out_dir / "shift_table.txt").write_text(shift_table + "\n", encoding="utf-8")
        (out_dir / "geometric_distortion_table.txt").write_text(geometric_table + "\n", encoding="utf-8")
        (out_dir / "range_usefulness_table.txt").write_text(range_usefulness_table + "\n", encoding="utf-8")
        if learned_fill_table:
            (out_dir / "learned_fill_diagnostics_table.txt").write_text(
                learned_fill_table + "\n",
                encoding="utf-8",
            )
        (out_dir / "learned_fill_diagnostics.json").write_text(
            json.dumps(
                {name: _evaluation_metrics_payload(metrics) for name, metrics in learned_fill_diagnostics.items()},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (out_dir / "range_learned_fill_summary.json").write_text(
            json.dumps(range_learned_fill_summary, indent=2) + "\n",
            encoding="utf-8",
        )
        if range_compression_audit:
            (out_dir / "range_compression_audit.json").write_text(
                json.dumps(range_compression_audit, indent=2) + "\n",
                encoding="utf-8",
            )
            (out_dir / "range_compression_audit_table.txt").write_text(
                range_compression_audit_table + "\n",
                encoding="utf-8",
            )
        (out_dir / "range_workload_diagnostics.json").write_text(
            json.dumps(range_diagnostics_summary, indent=2) + "\n",
            encoding="utf-8",
        )
        (out_dir / "range_workload_distribution_comparison.json").write_text(
            json.dumps(workload_distribution_comparison, indent=2) + "\n",
            encoding="utf-8",
        )
        with open(out_dir / "range_query_diagnostics.jsonl", "w", encoding="utf-8") as f:
            for row in range_diagnostics_rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        with open(out_dir / "example_run.json", "w", encoding="utf-8") as f:
            json.dump(dump, f, indent=2)
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


def _workload_keyword_to_map(keyword: str | None) -> dict[str, float] | None:
    """Translate a --workload keyword to a concrete pure workload map, or return None.

    - "range"/"knn"/"similarity"/"clustering" -> 100% that type.
    - anything else -> None (fall back to caller default).
    """
    if not keyword:
        return None
    k = keyword.strip().lower()
    if k in {"mixed", "local_mixed", "global_mixed"}:
        raise ValueError(
            f"workload='{k}' is no longer supported for model runs; use one pure type: "
            "range, knn, similarity, or clustering."
        )
    if k in {"range", "knn", "similarity", "clustering"}:
        return {k: 1.0}
    return None


def resolve_workload_maps(workload_keyword: str | None = None) -> tuple[dict[str, float], dict[str, float]]:
    """Return identical pure train/eval workload maps for one model run."""
    keyword_map = _workload_keyword_to_map(workload_keyword)
    workload_map = keyword_map if keyword_map is not None else {"range": 1.0}
    return workload_map, workload_map
