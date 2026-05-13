"""Experiment orchestration helpers for training and evaluation runs. See experiments/README.md for details."""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from evaluation.baselines import (
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
from experiments.experiment_workloads import (
    _workload_name,
    generate_experiment_workloads,
    resolve_workload_maps,
)
from queries.query_types import single_workload_type
from simplification.mlqds_scoring import workload_type_head
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
        data_split = prepare_experiment_split(
            config=config,
            seeds=seeds,
            trajectories=trajectories,
            needs_validation_score=needs_validation_score,
            trajectory_mmsis=trajectory_mmsis,
            validation_trajectories=validation_trajectories,
            eval_trajectories=eval_trajectories,
            eval_trajectory_mmsis=eval_trajectory_mmsis,
        )
        train_traj = data_split.train_traj
        test_traj = data_split.test_traj
        selection_traj = data_split.selection_traj
        train_mmsis = data_split.train_mmsis
        test_mmsis = data_split.test_mmsis

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
    methods = build_primary_methods(
        trained=trained,
        eval_workload=eval_workload,
        eval_workload_map=eval_workload_map,
        config=config,
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
        assert eval_labels is not None
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
        shift_pairs = evaluate_shift_pairs(
            matched_mlqds_score=float(matched["MLQDS"].aggregate_f1),
            trained=trained,
            train_workload=train_workload,
            train_workload_map=train_workload_map,
            eval_workload_map=eval_workload_map,
            config=config,
            test_points=test_points,
            test_boundaries=test_boundaries,
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
