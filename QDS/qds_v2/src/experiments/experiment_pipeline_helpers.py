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
    OracleMethod,
    QueryBlindMLMethod,
    RandomMethod,
    UniformTemporalMethod,
)
from src.evaluation.evaluate_methods import evaluate_method, print_method_comparison_table, print_shift_table
from src.experiments.experiment_config import ExperimentConfig, TypedQueryWorkload, derive_seed_bundle
from src.queries.query_generator import generate_typed_query_workload
from src.queries.query_types import parse_workload_mix
from src.training.train_model import train_model


@dataclass
class ExperimentOutputs:
    """Experiment run output payload. See src/experiments/README.md for details."""

    matched_table: str
    shift_table: str
    metrics_dump: dict


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


def run_experiment_pipeline(
    config: ExperimentConfig,
    trajectories: list[torch.Tensor],
    train_mix: dict[str, float],
    eval_mix: dict[str, float],
    results_dir: str,
) -> ExperimentOutputs:
    """Run training, matched evaluation, and shifted evaluation tables. See src/experiments/README.md for details."""
    pipeline_t0 = time.perf_counter()
    print(f"[pipeline] {len(trajectories)} trajectories, train_mix={_mix_name(train_mix)}, eval_mix={_mix_name(eval_mix)}", flush=True)

    seeds = derive_seed_bundle(config.data.seed)
    with _phase("split"):
        train_traj, _val_traj, test_traj = split_trajectories(
            trajectories,
            train_fraction=config.data.train_fraction,
            val_fraction=config.data.val_fraction,
            seed=seeds.split_seed,
        )
        print(f"  train={len(train_traj)}  test={len(test_traj)}", flush=True)

    with _phase("build-datasets"):
        train_ds = TrajectoryDataset(train_traj)
        test_ds = TrajectoryDataset(test_traj)
        train_points = train_ds.get_all_points()
        test_points = test_ds.get_all_points()
        train_boundaries = train_ds.get_trajectory_boundaries()
        test_boundaries = test_ds.get_trajectory_boundaries()

    with _phase("generate-workloads"):
        train_workload = generate_typed_query_workload(
            trajectories=train_traj,
            n_queries=config.query.n_queries,
            workload_mix=train_mix,
            seed=seeds.train_query_seed,
        )
        eval_workload = generate_typed_query_workload(
            trajectories=test_traj,
            n_queries=config.query.n_queries,
            workload_mix=eval_mix,
            seed=seeds.eval_query_seed,
        )

    with _phase(f"train-model ({config.model.epochs} epochs)"):
        trained = train_model(
            train_trajectories=train_traj,
            train_boundaries=train_boundaries,
            workload=train_workload,
            model_config=config.model,
            seed=seeds.torch_seed,
            query_blind=False,
        )
    with _phase(f"train-query-blind ({config.model.epochs} epochs)"):
        query_blind = train_model(
            train_trajectories=train_traj,
            train_boundaries=train_boundaries,
            workload=train_workload,
            model_config=config.model,
            seed=seeds.torch_seed + 11,
            query_blind=True,
        )

    methods = [
        MLQDSMethod(name="MLQDS", trained=trained, workload=eval_workload, workload_mix=eval_mix),
        QueryBlindMLMethod(name="QueryBlindML", trained=query_blind, workload=eval_workload, workload_mix=eval_mix),
        RandomMethod(seed=config.data.seed),
        UniformTemporalMethod(),
        DouglasPeuckerMethod(),
    ]

    matched: dict[str, Any] = {}
    with _phase("evaluate-matched"):
        for method in methods:
            with _phase(f"  eval {method.name}"):
                matched[method.name] = evaluate_method(
                    method=method,
                    points=test_points,
                    boundaries=test_boundaries,
                    typed_queries=eval_workload.typed_queries,
                    workload_mix=eval_mix,
                    compression_ratio=config.model.compression_ratio,
                )

        oracle = OracleMethod(labels=trained.labels, workload_mix=eval_mix)
        with _phase(f"  eval {oracle.name}"):
            matched[oracle.name] = evaluate_method(
                method=oracle,
                points=train_points,
                boundaries=train_boundaries,
                typed_queries=train_workload.typed_queries,
                workload_mix=eval_mix,
                compression_ratio=config.model.compression_ratio,
            )

    matched_table = print_method_comparison_table(matched)

    with _phase("evaluate-shift"):
        shift_pairs = {
            _mix_name(train_mix): {
                _mix_name(train_mix): float(
                    evaluate_method(
                        method=MLQDSMethod(name="MLQDS", trained=trained, workload=train_workload, workload_mix=train_mix),
                        points=test_points,
                        boundaries=test_boundaries,
                        typed_queries=train_workload.typed_queries,
                        workload_mix=train_mix,
                        compression_ratio=config.model.compression_ratio,
                    ).aggregate_error
                ),
                _mix_name(eval_mix): float(
                    evaluate_method(
                        method=MLQDSMethod(name="MLQDS", trained=trained, workload=eval_workload, workload_mix=eval_mix),
                        points=test_points,
                        boundaries=test_boundaries,
                        typed_queries=eval_workload.typed_queries,
                        workload_mix=eval_mix,
                        compression_ratio=config.model.compression_ratio,
                    ).aggregate_error
                ),
            }
        }
    shift_table = print_shift_table(shift_pairs)

    dump = {
        "config": config.to_dict(),
        "train_mix": train_mix,
        "eval_mix": eval_mix,
        "matched": {
            name: {
                "aggregate_error": m.aggregate_error,
                "per_type_error": m.per_type_error,
                "compression_ratio": m.compression_ratio,
                "latency_ms": m.latency_ms,
            }
            for name, m in matched.items()
        },
        "shift": shift_pairs,
        "training_history": trained.history,
    }

    with _phase("write-results"):
        out_dir = Path(results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "matched_table.txt").write_text(matched_table + "\n", encoding="utf-8")
        (out_dir / "shift_table.txt").write_text(shift_table + "\n", encoding="utf-8")
        with open(out_dir / "example_run.json", "w", encoding="utf-8") as f:
            json.dump(dump, f, indent=2)
        print(f"  wrote results to {out_dir}", flush=True)

    print(f"[pipeline] total runtime {time.perf_counter() - pipeline_t0:.2f}s", flush=True)
    return ExperimentOutputs(matched_table=matched_table, shift_table=shift_table, metrics_dump=dump)


def resolve_workload_mixes(
    train_workload_mix_arg: str | None,
    eval_workload_mix_arg: str | None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Parse and normalize train/eval workload mix strings. See src/experiments/README.md for details."""
    train_mix = parse_workload_mix(train_workload_mix_arg, default={"range": 0.8, "knn": 0.2})
    eval_mix = parse_workload_mix(eval_workload_mix_arg, default={"range": 0.2, "clustering": 0.8})
    return train_mix, eval_mix
