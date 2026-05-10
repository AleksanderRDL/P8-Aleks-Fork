"""Experiment orchestration helpers for training and evaluation runs. See src/experiments/README.md for details."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


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

from src.data.trajectory_dataset import TrajectoryDataset
from src.evaluation.baselines import (
    DouglasPeuckerMethod,
    MLQDSMethod,
    NewUniformTemporalMethod,
    OracleMethod,
)
from src.evaluation.evaluate_methods import (
    EvaluationQueryCache,
    evaluate_method,
    print_geometric_distortion_table,
    print_method_comparison_table,
    print_shift_table,
    score_retained_mask,
)
from src.experiments.experiment_config import ExperimentConfig, TypedQueryWorkload, derive_seed_bundle
from src.experiments.geojson_writers import report_trajectory_length_loss, write_queries_geojson, write_simplified_csv
from src.queries.query_generator import generate_typed_query_workload
from src.queries.query_types import parse_workload_mix
from src.queries.workload_diagnostics import compute_range_label_diagnostics, compute_range_workload_diagnostics
from src.training.importance_labels import compute_typed_importance_labels
from src.training.train_model import train_model
from src.training.training_pipeline import ModelArtifacts, save_checkpoint
from src.experiments.torch_runtime import cuda_memory_snapshot, reset_cuda_peak_memory_stats, torch_runtime_snapshot


@dataclass
class ExperimentOutputs:
    """Experiment run output payload. See src/experiments/README.md for details."""

    matched_table: str
    shift_table: str
    metrics_dump: dict
    geometric_table: str = ""


def split_trajectories(
    trajectories: list[torch.Tensor],
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Deterministically split trajectories at trajectory level. See src/experiments/README.md for details."""
    n = len(trajectories)
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g).tolist()

    n_train = max(1, int(n * train_fraction))
    n_val = max(1, int(n * val_fraction)) if n - n_train > 1 else 0
    n_test = max(1, n - n_train - n_val)
    if n_train + n_val + n_test > n:
        n_test = n - n_train - n_val

    train = [trajectories[i] for i in perm[:n_train]]
    val = [trajectories[i] for i in perm[n_train : n_train + n_val]]
    test = [trajectories[i] for i in perm[n_train + n_val :]]
    if not test:
        test = val if val else train
    return train, val, test


def _mix_name(mix: dict[str, float]) -> str:
    """Build compact string name for workload mix maps. See src/experiments/README.md for details."""
    return ",".join(f"{k}={v:.1f}" for k, v in sorted(mix.items()))


def _coverage_name(workload: TypedQueryWorkload) -> str:
    """Format workload point-coverage metadata for logs."""
    if workload.coverage_fraction is None:
        return "unknown"
    covered = workload.covered_points if workload.covered_points is not None else 0
    total = workload.total_points if workload.total_points is not None else 0
    return f"{100.0 * workload.coverage_fraction:.2f}% ({covered}/{total})"


def _normalized_coverage_target(value: float | None) -> float | None:
    """Normalize coverage target for pipeline warnings."""
    if value is None:
        return None
    target = float(value)
    return target / 100.0 if target > 1.0 else target


def _validation_query_count(config: ExperimentConfig) -> int:
    """Use a broader validation workload for less seed-specific F1 checkpointing."""
    doubled = max(int(config.query.n_queries), int(config.query.n_queries) * 2)
    if config.query.max_queries is None:
        return doubled
    return min(doubled, max(int(config.query.n_queries), int(config.query.max_queries)))


def _range_diagnostic_duplicate_threshold(config: ExperimentConfig) -> float | None:
    """Use explicit duplicate threshold for diagnostics, or a diagnostic-only default."""
    threshold = config.query.range_duplicate_iou_threshold
    return 0.85 if threshold is None else threshold


def _range_only_queries(typed_queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only range queries from a mixed workload."""
    return [query for query in typed_queries if str(query.get("type", "")).lower() == "range"]


def _range_signal_diagnostics(
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    range_queries: list[dict[str, Any]],
    workload_mix: dict[str, float],
    compression_ratio: float,
    seed: int,
) -> dict[str, Any]:
    """Compute label, Oracle, and baseline diagnostics for range workloads."""
    if not range_queries:
        return {
            "range_query_count": 0,
            "labels": compute_range_label_diagnostics(
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0, 4), dtype=torch.bool),
            ),
            "methods": {},
            "best_baseline": None,
            "best_baseline_range_f1": 0.0,
            "oracle_range_f1": 0.0,
            "oracle_gap_over_best_baseline": 0.0,
        }

    labels, labelled_mask = compute_typed_importance_labels(
        points=points,
        boundaries=boundaries,
        typed_queries=range_queries,
        seed=seed,
    )
    label_diagnostics = compute_range_label_diagnostics(labels, labelled_mask)
    methods = [
        NewUniformTemporalMethod(),
        DouglasPeuckerMethod(),
        OracleMethod(labels=labels, workload_mix={"range": 1.0}),
    ]
    method_scores: dict[str, dict[str, float]] = {}
    query_cache = EvaluationQueryCache.for_workload(points, boundaries, range_queries)
    for method in methods:
        retained_mask = method.simplify(points, boundaries, compression_ratio)
        aggregate, per_type, _, _ = score_retained_mask(
            points=points,
            boundaries=boundaries,
            retained_mask=retained_mask,
            typed_queries=range_queries,
            workload_mix={"range": 1.0},
            query_cache=query_cache,
        )
        method_scores[method.name] = {
            "aggregate_f1": float(aggregate),
            "range_f1": float(per_type.get("range", 0.0)),
        }

    baseline_names = ["uniform", "DouglasPeucker"]
    best_baseline = max(baseline_names, key=lambda name: method_scores.get(name, {}).get("range_f1", 0.0))
    best_baseline_range_f1 = float(method_scores[best_baseline]["range_f1"])
    oracle_range_f1 = float(method_scores.get("Oracle", {}).get("range_f1", 0.0))
    normalized_mix = sum(float(v) for v in workload_mix.values())
    range_weight = float(workload_mix.get("range", 0.0)) / normalized_mix if normalized_mix > 0.0 else 0.0
    return {
        "range_query_count": int(len(range_queries)),
        "range_workload_weight": float(range_weight),
        "labels": label_diagnostics,
        "methods": method_scores,
        "best_baseline": best_baseline,
        "best_baseline_range_f1": best_baseline_range_f1,
        "oracle_range_f1": oracle_range_f1,
        "oracle_gap_over_best_baseline": float(oracle_range_f1 - best_baseline_range_f1),
    }


def _range_workload_diagnostics(
    label: str,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    workload: TypedQueryWorkload,
    workload_mix: dict[str, float],
    config: ExperimentConfig,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build summary and JSONL rows for one workload."""
    workload_diagnostics = compute_range_workload_diagnostics(
        points=points,
        boundaries=boundaries,
        typed_queries=workload.typed_queries,
        max_point_hit_fraction=config.query.range_max_point_hit_fraction,
        max_trajectory_hit_fraction=config.query.range_max_trajectory_hit_fraction,
        max_box_volume_fraction=config.query.range_max_box_volume_fraction,
        duplicate_iou_threshold=_range_diagnostic_duplicate_threshold(config),
    )
    range_queries = _range_only_queries(workload.typed_queries)
    signal = _range_signal_diagnostics(
        points=points,
        boundaries=boundaries,
        range_queries=range_queries,
        workload_mix=workload_mix,
        compression_ratio=config.model.compression_ratio,
        seed=seed,
    )
    summary = {
        "range": workload_diagnostics["summary"],
        "range_signal": signal,
        "generation": workload.generation_diagnostics or {},
    }
    rows = [{"workload": label, **row} for row in workload_diagnostics["queries"]]
    return summary, rows


def run_experiment_pipeline(
    config: ExperimentConfig,
    trajectories: list[torch.Tensor],
    train_mix: dict[str, float],
    eval_mix: dict[str, float],
    results_dir: str,
    save_model: str | None = None,
    save_queries_dir: str | None = None,
    save_simplified_dir: str | None = None,
    trajectory_mmsis: list[int] | None = None,
    eval_trajectories: list[torch.Tensor] | None = None,
    eval_trajectory_mmsis: list[int] | None = None,
    data_audit: dict[str, Any] | None = None,
) -> ExperimentOutputs:
    """Run training, matched evaluation, and shifted evaluation tables. See src/experiments/README.md for details."""
    pipeline_t0 = time.perf_counter()
    if eval_trajectories is None:
        print(
            f"[pipeline] {len(trajectories)} trajectories, train_mix={_mix_name(train_mix)}, "
            f"eval_mix={_mix_name(eval_mix)}",
            flush=True,
        )
    else:
        print(
            f"[pipeline] train={len(trajectories)} trajectories, eval={len(eval_trajectories)} trajectories, "
            f"train_mix={_mix_name(train_mix)}, eval_mix={_mix_name(eval_mix)}",
            flush=True,
        )

    seeds = derive_seed_bundle(config.data.seed)
    selection_metric = str(getattr(config.model, "checkpoint_selection_metric", "loss")).lower()
    f1_diag_every = int(getattr(config.model, "f1_diagnostic_every", 0) or 0)
    needs_validation_f1 = selection_metric in {"f1", "uniform_gap"} or f1_diag_every > 0
    with _phase("split"):
        selection_traj: list[torch.Tensor] | None = None
        if eval_trajectories is None:
            # Reproduce split_trajectories' permutation here so we can align the
            # MMSI list with the test split (the helper itself doesn't carry ids).
            n = len(trajectories)
            g = torch.Generator().manual_seed(int(seeds.split_seed))
            perm = torch.randperm(n, generator=g).tolist()
            n_train = max(1, int(n * config.data.train_fraction))
            n_val = max(1, int(n * config.data.val_fraction)) if n - n_train > 1 else 0
            train_traj = [trajectories[i] for i in perm[:n_train]]
            _val_traj = [trajectories[i] for i in perm[n_train : n_train + n_val]]
            test_traj = [trajectories[i] for i in perm[n_train + n_val :]]
            if not test_traj:
                test_traj = _val_traj if _val_traj else train_traj
            selection_traj = _val_traj if needs_validation_f1 and _val_traj else None
            if trajectory_mmsis is not None and len(trajectory_mmsis) == n:
                train_mmsis = [trajectory_mmsis[i] for i in perm[:n_train]]
                test_mmsis = [trajectory_mmsis[i] for i in perm[n_train + n_val :]]
                if not test_mmsis:
                    test_mmsis = [trajectory_mmsis[i] for i in perm[n_train : n_train + n_val]] or \
                                 [trajectory_mmsis[i] for i in perm[:n_train]]
            else:
                train_mmsis = None
                test_mmsis = None
            print(f"  split mode=single dataset  train={len(train_traj)}  test={len(test_traj)}", flush=True)
        else:
            train_traj = trajectories
            test_traj = eval_trajectories
            train_mmsis = trajectory_mmsis
            test_mmsis = eval_trajectory_mmsis
            if needs_validation_f1:
                n = len(train_traj)
                g = torch.Generator().manual_seed(int(seeds.split_seed))
                perm = torch.randperm(n, generator=g).tolist()
                n_val = max(1, int(n * config.data.val_fraction)) if n > 1 else 0
                val_idx = set(perm[:n_val])
                selection_traj = [traj for idx, traj in enumerate(train_traj) if idx in val_idx]
                train_traj = [traj for idx, traj in enumerate(train_traj) if idx not in val_idx]
                if train_mmsis is not None and len(train_mmsis) == n:
                    train_mmsis = [mmsi for idx, mmsi in enumerate(train_mmsis) if idx not in val_idx]
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
        knn_front_load = int(train_mix.get("knn", 0.0) * config.query.n_queries)
        train_workload = generate_typed_query_workload(
            trajectories=train_traj,
            n_queries=config.query.n_queries,
            workload_mix=train_mix,
            seed=seeds.train_query_seed,
            target_coverage=config.query.target_coverage,
            max_queries=config.query.max_queries,
            range_spatial_fraction=config.query.range_spatial_fraction,
            range_time_fraction=config.query.range_time_fraction,
            knn_k=config.query.knn_k,
            front_load_knn=knn_front_load,
            range_min_point_hits=config.query.range_min_point_hits,
            range_max_point_hit_fraction=config.query.range_max_point_hit_fraction,
            range_min_trajectory_hits=config.query.range_min_trajectory_hits,
            range_max_trajectory_hit_fraction=config.query.range_max_trajectory_hit_fraction,
            range_max_box_volume_fraction=config.query.range_max_box_volume_fraction,
            range_duplicate_iou_threshold=config.query.range_duplicate_iou_threshold,
            range_acceptance_max_attempts=config.query.range_acceptance_max_attempts,
        )
        eval_workload = generate_typed_query_workload(
            trajectories=test_traj,
            n_queries=config.query.n_queries,
            workload_mix=eval_mix,
            seed=seeds.eval_query_seed,
            target_coverage=config.query.target_coverage,
            max_queries=config.query.max_queries,
            range_spatial_fraction=config.query.range_spatial_fraction,
            range_time_fraction=config.query.range_time_fraction,
            knn_k=config.query.knn_k,
            range_min_point_hits=config.query.range_min_point_hits,
            range_max_point_hit_fraction=config.query.range_max_point_hit_fraction,
            range_min_trajectory_hits=config.query.range_min_trajectory_hits,
            range_max_trajectory_hit_fraction=config.query.range_max_trajectory_hit_fraction,
            range_max_box_volume_fraction=config.query.range_max_box_volume_fraction,
            range_duplicate_iou_threshold=config.query.range_duplicate_iou_threshold,
            range_acceptance_max_attempts=config.query.range_acceptance_max_attempts,
        )
        selection_workload = None
        if selection_traj:
            selection_workload = generate_typed_query_workload(
                trajectories=selection_traj,
                n_queries=_validation_query_count(config),
                workload_mix=eval_mix,
                seed=seeds.eval_query_seed + 17,
                target_coverage=config.query.target_coverage,
                max_queries=config.query.max_queries,
                range_spatial_fraction=config.query.range_spatial_fraction,
                range_time_fraction=config.query.range_time_fraction,
                knn_k=config.query.knn_k,
                range_min_point_hits=config.query.range_min_point_hits,
                range_max_point_hit_fraction=config.query.range_max_point_hit_fraction,
                range_min_trajectory_hits=config.query.range_min_trajectory_hits,
                range_max_trajectory_hit_fraction=config.query.range_max_trajectory_hit_fraction,
                range_max_box_volume_fraction=config.query.range_max_box_volume_fraction,
                range_duplicate_iou_threshold=config.query.range_duplicate_iou_threshold,
                range_acceptance_max_attempts=config.query.range_acceptance_max_attempts,
            )
        print(
            f"  train_workload={len(train_workload.typed_queries)} queries  "
            f"coverage={_coverage_name(train_workload)}",
            flush=True,
        )
        print(
            f"  eval_workload={len(eval_workload.typed_queries)} queries  "
            f"coverage={_coverage_name(eval_workload)}",
            flush=True,
        )
        if selection_workload is not None:
            print(
                f"  selection_workload={len(selection_workload.typed_queries)} queries  "
                f"coverage={_coverage_name(selection_workload)}",
                flush=True,
            )
        target = _normalized_coverage_target(config.query.target_coverage)
        if target is not None:
            for label, workload in (("train", train_workload), ("eval", eval_workload)):
                coverage = float(workload.coverage_fraction or 0.0)
                if coverage + 1e-9 < target:
                    print(
                        f"  WARNING: {label} workload stopped below requested coverage "
                        f"({coverage:.2%} < {target:.2%}); raise --n_queries or query footprint to cover more points.",
                        flush=True,
                    )

    range_diagnostics_summary: dict[str, Any] = {}
    range_diagnostics_rows: list[dict[str, Any]] = []
    with _phase("range-diagnostics"):
        train_summary, train_rows = _range_workload_diagnostics(
            "train",
            train_points,
            train_boundaries,
            train_workload,
            train_mix,
            config,
            seeds.train_query_seed,
        )
        eval_summary, eval_rows = _range_workload_diagnostics(
            "eval",
            test_points,
            test_boundaries,
            eval_workload,
            eval_mix,
            config,
            seeds.eval_query_seed,
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
                eval_mix,
                config,
                seeds.eval_query_seed + 17,
            )
            range_diagnostics_summary["selection"] = selection_summary
            range_diagnostics_rows.extend(selection_rows)
        for label, summary in range_diagnostics_summary.items():
            range_summary = summary["range"]
            signal = summary["range_signal"]
            print(
                f"  {label}: range_queries={range_summary['range_query_count']}  "
                f"empty={range_summary['empty_query_rate']:.2%}  "
                f"broad={range_summary['too_broad_query_rate']:.2%}  "
                f"duplicates={range_summary['near_duplicate_query_rate']:.2%}  "
                f"oracle_gap={signal['oracle_gap_over_best_baseline']:+.6f}",
                flush=True,
            )

    if save_queries_dir:
        with _phase("write-queries-geojson"):
            write_queries_geojson(save_queries_dir, eval_workload.typed_queries)

    reset_cuda_peak_memory_stats()
    with _phase(f"train-model ({config.model.epochs} epochs)"):
        trained = train_model(
            train_trajectories=train_traj,
            train_boundaries=train_boundaries,
            workload=train_workload,
            model_config=config.model,
            seed=seeds.torch_seed,
            train_mix=train_mix,
            validation_trajectories=selection_traj,
            validation_boundaries=selection_boundaries,
            validation_workload=selection_workload,
            validation_mix=eval_mix if selection_workload is not None else None,
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
                train_workload_mix=train_mix,
                eval_workload_mix=eval_mix,
            )
            save_checkpoint(save_model, artifacts)
            print(
                f"  saved checkpoint to {save_model}  "
                f"(epochs_trained={trained.epochs_trained}, "
                f"best_epoch={trained.best_epoch}, best_loss={trained.best_loss:.8f}, "
                f"train_mix={_mix_name(train_mix)}, eval_mix={_mix_name(eval_mix)})",
                flush=True,
            )
    methods = [
        MLQDSMethod(
            name="MLQDS",
            trained=trained,
            workload=eval_workload,
            workload_mix=eval_mix,
            temporal_fraction=config.model.mlqds_temporal_fraction,
            diversity_bonus=config.model.mlqds_diversity_bonus,
        ),
        NewUniformTemporalMethod(),
        DouglasPeuckerMethod(),
    ]

    matched: dict[str, Any] = {}
    save_masks = bool(save_simplified_dir)
    with _phase("evaluate-matched"):
        eval_query_cache = EvaluationQueryCache.for_workload(
            test_points,
            test_boundaries,
            eval_workload.typed_queries,
        )
        for method in methods:
            with _phase(f"  eval {method.name}"):
                matched[method.name] = evaluate_method(
                    method=method,
                    points=test_points,
                    boundaries=test_boundaries,
                    typed_queries=eval_workload.typed_queries,
                    workload_mix=eval_mix,
                    compression_ratio=config.model.compression_ratio,
                    return_mask=method.name == "MLQDS" or save_masks,
                    query_cache=eval_query_cache,
                )

        eval_labels, _ = compute_typed_importance_labels(
            points=test_points,
            boundaries=test_boundaries,
            typed_queries=eval_workload.typed_queries,
            seed=seeds.eval_query_seed,
        )
        oracle = OracleMethod(labels=eval_labels, workload_mix=eval_mix)
        with _phase(f"  eval {oracle.name}"):
            matched[oracle.name] = evaluate_method(
                method=oracle,
                points=test_points,
                boundaries=test_boundaries,
                typed_queries=eval_workload.typed_queries,
                workload_mix=eval_mix,
                compression_ratio=config.model.compression_ratio,
                query_cache=eval_query_cache,
            )

    matched_table = print_method_comparison_table(matched)
    geometric_table = print_geometric_distortion_table(matched)

    with _phase("evaluate-shift"):
        train_name = _mix_name(train_mix)
        eval_name = _mix_name(eval_mix)
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
                        workload_mix=train_mix,
                        temporal_fraction=config.model.mlqds_temporal_fraction,
                        diversity_bonus=config.model.mlqds_diversity_bonus,
                    ),
                    points=test_points,
                    boundaries=test_boundaries,
                    typed_queries=train_workload.typed_queries,
                    workload_mix=train_mix,
                    compression_ratio=config.model.compression_ratio,
                    query_cache=train_query_cache,
                ).aggregate_f1
            )
    shift_table = print_shift_table(shift_pairs)

    dump = {
        "config": config.to_dict(),
        "train_mix": train_mix,
        "eval_mix": eval_mix,
        "train_query_count": len(train_workload.typed_queries),
        "eval_query_count": len(eval_workload.typed_queries),
        "train_query_coverage": train_workload.coverage_fraction,
        "eval_query_coverage": eval_workload.coverage_fraction,
        "matched": {
            name: {
                "aggregate_f1": m.aggregate_f1,
                "per_type_f1": m.per_type_f1,
                "compression_ratio": m.compression_ratio,
                "latency_ms": m.latency_ms,
                "avg_retained_point_gap": m.avg_retained_point_gap,
                "avg_retained_point_gap_norm": m.avg_retained_point_gap_norm,
                "max_retained_point_gap": m.max_retained_point_gap,
                "geometric_distortion": m.geometric_distortion,
                "avg_length_preserved": m.avg_length_preserved,
                "avg_length_loss": m.avg_length_loss,
                "combined_query_shape_score": m.combined_query_shape_score,
            }
            for name, m in matched.items()
        },
        "shift": shift_pairs,
        "training_history": trained.history,
        "best_epoch": trained.best_epoch,
        "best_loss": trained.best_loss,
        "best_f1": trained.best_f1,
        "data_audit": data_audit,
        "workload_diagnostics": range_diagnostics_summary,
        "torch_runtime": torch_runtime_snapshot(),
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
        (out_dir / "range_workload_diagnostics.json").write_text(
            json.dumps(range_diagnostics_summary, indent=2) + "\n",
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
                    workload_mix=eval_mix,
                    temporal_fraction=config.model.mlqds_temporal_fraction,
                    diversity_bonus=config.model.mlqds_diversity_bonus,
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
                ref_mask = matched.get(ref_name).retained_mask if matched.get(ref_name) is not None else None
                if ref_mask is not None:
                    write_simplified_csv(
                        str(out_dir / csv_name),
                        test_points,
                        test_boundaries,
                        ref_mask,
                        trajectory_mmsis=test_mmsis,
                    )

        with _phase("trajectory-length-loss"):
            report_trajectory_length_loss(test_points, test_boundaries, eval_mask, top_k=25, trajectory_mmsis=test_mmsis)

    print(f"[pipeline] total runtime {time.perf_counter() - pipeline_t0:.2f}s", flush=True)
    return ExperimentOutputs(
        matched_table=matched_table,
        shift_table=shift_table,
        metrics_dump=dump,
        geometric_table=geometric_table,
    )


def _workload_keyword_to_mix(keyword: str | None) -> dict[str, float] | None:
    """Translate a --workload keyword to a concrete mix, or return None.

    - "mixed"        -> all 4 types (range-heavy: 0.4/0.2/0.2/0.2).
    - "local_mixed"  -> local/point-based types (range=0.6, knn=0.4).
                        These use small boxes / neighbourhood lookups, cheap to eval.
    - "global_mixed" -> trajectory-global types (similarity=0.5, clustering=0.5).
                        DBSCAN + DTW; expensive, needs long wall time.
    - "range"/"knn"/"similarity"/"clustering" -> 100% that type.
    - anything else -> None (fall back to caller default).
    """
    if not keyword:
        return None
    k = keyword.strip().lower()
    if k == "mixed":
        return {"range": 0.4, "knn": 0.2, "similarity": 0.2, "clustering": 0.2}
    if k == "local_mixed":
        return {"range": 0.6, "knn": 0.4}
    if k == "global_mixed":
        return {"similarity": 0.5, "clustering": 0.5}
    if k in {"range", "knn", "similarity", "clustering"}:
        return {k: 1.0}
    return None


def resolve_workload_mixes(
    train_workload_mix_arg: str | None,
    eval_workload_mix_arg: str | None,
    workload_keyword: str | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Parse and normalize train/eval workload mix strings. See src/experiments/README.md for details.

    Priority: explicit --train_workload_mix / --eval_workload_mix strings win.
    Otherwise, if --workload is a recognised keyword, both mixes follow it.
    Otherwise, fall back to the historical mixed-shift defaults.
    """
    keyword_mix = _workload_keyword_to_mix(workload_keyword)
    if keyword_mix is not None:
        default_train = keyword_mix
        default_eval = keyword_mix
    else:
        default_train = {"range": 0.8, "knn": 0.2}
        default_eval = {"range": 0.2, "clustering": 0.8}
    train_mix = parse_workload_mix(train_workload_mix_arg, default=default_train)
    eval_mix = parse_workload_mix(eval_workload_mix_arg, default=default_eval)
    return train_mix, eval_mix
