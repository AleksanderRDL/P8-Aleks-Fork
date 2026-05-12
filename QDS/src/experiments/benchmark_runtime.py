"""Runtime benchmark wrapper for AIS-QDS experiments.

The benchmark intentionally shells out to the existing experiment and
inference entrypoints so timing artifacts cover the real CLI path users run.
It records environment metadata, git state, child commands, parsed phase and
epoch timings, and final F1 metrics into a stable JSON file.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from src.experiments.benchmark_profiles import DEFAULT_PROFILE, PROFILE_CHOICES, benchmark_profile_args
from src.experiments.torch_runtime import (
    AMP_MODE_CHOICES,
    FLOAT32_MATMUL_PRECISION_CHOICES,
    amp_runtime_snapshot,
    apply_torch_runtime_settings,
)


PHASE_DONE_RE = re.compile(r"^\[(?P<name>[^\]]+)\]\s+done in (?P<seconds>[0-9.]+)s")
EPOCH_RE = re.compile(
    r"epoch\s+(?P<epoch>\d+)/(?P<total>\d+).*?\((?P<seconds>[0-9.]+)s\)"
)
INFERENCE_STEP_RE = re.compile(
    r"^\[(?P<name>eval|workload|load-data|trajectory-length-loss)\].*?(?:done|generated|in)\s+"
    r"(?P<seconds>[0-9.]+)s"
)
REAL_USECASE_PROFILE_ARGS = benchmark_profile_args(
    DEFAULT_PROFILE,
    include_workload=True,
    include_checkpoint_selection=True,
    include_f1_diagnostic=True,
)


def _qds_root() -> Path:
    """Return the QDS package root."""
    return Path(__file__).resolve().parents[2]


def _repo_root() -> Path:
    """Return the repository root."""
    return _qds_root().parent


def _run_capture(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run a command and capture text output."""
    env = os.environ.copy()
    started = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    proc.elapsed_seconds = time.perf_counter() - started  # type: ignore[attr-defined]
    return proc


def _git_text(args: list[str]) -> str | None:
    """Run a git command in the repository root and return stripped stdout."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(_repo_root()),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _git_metadata() -> dict[str, Any]:
    """Collect git commit and dirty-status metadata."""
    status = _git_text(["status", "--short"]) or ""
    return {
        "commit": _git_text(["rev-parse", "HEAD"]),
        "branch": _git_text(["branch", "--show-current"]),
        "dirty": bool(status),
        "status_short": status.splitlines(),
    }


def _optional_version(module_name: str) -> str | None:
    """Return module __version__ when importable."""
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def _nvidia_smi_metadata() -> dict[str, Any]:
    """Collect GPU telemetry metadata through nvidia-smi when available."""
    if shutil.which("nvidia-smi") is None:
        return {
            "available": False,
            "unavailable_reason": "nvidia-smi not found on PATH",
            "gpus": [],
        }
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,utilization.gpu,utilization.memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        return {
            "available": False,
            "unavailable_reason": str(exc),
            "gpus": [],
        }
    if proc.returncode != 0:
        return {
            "available": False,
            "unavailable_reason": proc.stderr.strip() or "nvidia-smi returned non-zero exit status",
            "gpus": [],
        }
    gpus = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        gpus.append(
            {
                "name": parts[0],
                "driver_version": parts[1],
                "memory_total_mb": _safe_float(parts[2]),
                "gpu_utilization_percent": _safe_float(parts[3]),
                "memory_utilization_percent": _safe_float(parts[4]),
            }
        )
    return {
        "available": bool(gpus),
        "unavailable_reason": None if gpus else "nvidia-smi returned no GPU rows",
        "gpus": gpus,
    }


def _safe_float(value: str) -> float | None:
    """Parse a float or return None."""
    try:
        return float(value)
    except ValueError:
        return None


def _torch_cuda_metadata() -> dict[str, Any]:
    """Collect Torch/CUDA details without requiring a visible GPU."""
    cuda_available = bool(torch.cuda.is_available())
    devices: list[dict[str, Any]] = []
    if cuda_available:
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            devices.append(
                {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "total_memory_mb": int(props.total_memory // (1024 * 1024)),
                    "major": int(props.major),
                    "minor": int(props.minor),
                }
            )
    return {
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "cuda_devices": devices,
        "triton_version": _optional_version("triton"),
        "tf32_matmul_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn_allowed": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def _environment_metadata(amp_mode: str) -> dict[str, Any]:
    """Collect benchmark environment metadata."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "working_directory": str(Path.cwd()),
        "qds_root": str(_qds_root()),
        "repo_root": str(_repo_root()),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "version_info": list(sys.version_info[:3]),
            "platform": platform.platform(),
        },
        "torch": _torch_cuda_metadata(),
        "gpu_telemetry": _nvidia_smi_metadata(),
        "amp": {
            "mode": amp_mode,
            "bf16_enabled": amp_mode == "bf16",
            "fp16_enabled": amp_mode == "fp16",
            "note": "Forwarded to child train/inference entrypoints; autocast is CUDA-only.",
        },
        "git": _git_metadata(),
    }


def _parse_timings(output: str) -> dict[str, Any]:
    """Parse phase and epoch timings from child stdout."""
    phases: list[dict[str, Any]] = []
    epochs: list[dict[str, Any]] = []
    inference_steps: list[dict[str, Any]] = []
    for line in output.splitlines():
        phase_match = PHASE_DONE_RE.search(line)
        if phase_match:
            phases.append(
                {
                    "name": phase_match.group("name").strip(),
                    "seconds": float(phase_match.group("seconds")),
                }
            )
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            epochs.append(
                {
                    "epoch": int(epoch_match.group("epoch")),
                    "total_epochs": int(epoch_match.group("total")),
                    "seconds": float(epoch_match.group("seconds")),
                    "line": line.strip(),
                }
            )
        inference_match = INFERENCE_STEP_RE.search(line)
        if inference_match:
            inference_steps.append(
                {
                    "name": inference_match.group("name").strip(),
                    "seconds": float(inference_match.group("seconds")),
                    "line": line.strip(),
                }
            )
    return {
        "phase_timings": phases,
        "epoch_timings": epochs,
        "inference_step_timings": inference_steps,
    }


def _load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON file if it exists."""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _matched_summary(run_json: dict[str, Any] | None) -> dict[str, Any]:
    """Extract compact final metrics from an experiment JSON payload."""
    if not run_json:
        return {}
    matched = run_json.get("matched", {})
    summary: dict[str, Any] = {
        "best_epoch": run_json.get("best_epoch"),
        "best_loss": run_json.get("best_loss"),
        "best_selection_score": run_json.get("best_selection_score", run_json.get("best_f1")),
        "best_f1": run_json.get("best_f1"),
        "workload": run_json.get("workload"),
        "train_query_count": run_json.get("train_query_count"),
        "eval_query_count": run_json.get("eval_query_count"),
        "torch_runtime": run_json.get("torch_runtime"),
        "cuda_memory": run_json.get("cuda_memory"),
        "methods": {},
    }
    for name, payload in matched.items():
        summary["methods"][name] = {
            "aggregate_f1": payload.get("aggregate_f1"),
            "per_type_f1": payload.get("per_type_f1"),
            "latency_ms": payload.get("latency_ms"),
            "compression_ratio": payload.get("compression_ratio"),
            "avg_length_preserved": payload.get("avg_length_preserved"),
            "combined_query_shape_score": payload.get("combined_query_shape_score"),
            "range_point_f1": payload.get("range_point_f1"),
            "range_ship_f1": payload.get("range_ship_f1"),
            "range_entry_exit_f1": payload.get("range_entry_exit_f1"),
            "range_boundary_f1": payload.get("range_boundary_f1"),
            "range_temporal_coverage": payload.get("range_temporal_coverage"),
            "range_gap_coverage": payload.get("range_gap_coverage"),
            "range_shape_score": payload.get("range_shape_score"),
            "range_usefulness_score": payload.get("range_usefulness_score"),
            "range_usefulness_schema_version": payload.get("range_usefulness_schema_version"),
        }
    config = run_json.get("config", {})
    model_config = config.get("model", {}) if isinstance(config, dict) else {}
    if model_config:
        summary["batch_size"] = {
            "train_batch_size": model_config.get("train_batch_size"),
            "inference_batch_size": model_config.get("inference_batch_size"),
            "window_length": model_config.get("window_length"),
            "window_stride": model_config.get("window_stride"),
            "query_chunk_size": model_config.get("query_chunk_size"),
        }
    return summary


def _profile_train_args(profile: str, seed: int, results_dir: Path, checkpoint: Path) -> list[str]:
    """Return stable training command args for a runtime benchmark profile."""
    common = [
        "--seed",
        str(seed),
        "--results_dir",
        str(results_dir),
        "--save_model",
        str(checkpoint),
    ]
    if profile == DEFAULT_PROFILE:
        return [*REAL_USECASE_PROFILE_ARGS, *common]
    raise ValueError(f"Unknown benchmark profile: {profile}")


def _split_extra_args(raw: str | None) -> list[str]:
    """Split optional extra CLI args with shell-like quoting."""
    return shlex.split(raw) if raw else []


def _extra_args_include_training_data_source(raw: str | None) -> bool:
    """Return whether train extra args provide real CSV training data."""
    tokens = _split_extra_args(raw)
    data_flags = {"--csv_path", "--train_csv_path", "--train_csv"}
    return any(token in data_flags or any(token.startswith(f"{flag}=") for flag in data_flags) for token in tokens)


def _parse_train_batch_sizes(raw: str | None) -> list[int] | None:
    """Parse comma-separated training batch sizes for benchmark sweeps."""
    if raw is None or not raw.strip():
        return None
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("--train_batch_sizes values must be positive integers.")
        values.append(value)
    if not values:
        raise ValueError("--train_batch_sizes did not contain any positive integer values.")
    return values


def _batch_size_sweep_summary(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build compact comparison rows for training batch-size sweeps."""
    rows: list[dict[str, Any]] = []
    for step in steps:
        if not str(step.get("name", "")).startswith("train_bs"):
            continue
        timings = step.get("timings", {})
        epoch_times = [float(row["seconds"]) for row in timings.get("epoch_timings", [])]
        metrics = step.get("metrics", {})
        methods = metrics.get("methods", {})
        mlqds = methods.get("MLQDS", {}) if isinstance(methods, dict) else {}
        cuda_memory = metrics.get("cuda_memory", {}) or {}
        training_memory = cuda_memory.get("training", {}) if isinstance(cuda_memory, dict) else {}
        configured = metrics.get("batch_size", {}) if isinstance(metrics, dict) else {}
        batch_size = step.get("train_batch_size") or configured.get("train_batch_size")
        rows.append(
            {
                "train_batch_size": batch_size,
                "returncode": step.get("returncode"),
                "elapsed_seconds": step.get("elapsed_seconds"),
                "epoch_time_mean_seconds": (
                    float(sum(epoch_times) / len(epoch_times)) if epoch_times else None
                ),
                "epoch_time_min_seconds": min(epoch_times) if epoch_times else None,
                "epoch_time_max_seconds": max(epoch_times) if epoch_times else None,
                "peak_allocated_mb": training_memory.get("max_allocated_mb"),
                "peak_reserved_mb": training_memory.get("max_reserved_mb"),
                "best_selection_score": metrics.get("best_selection_score", metrics.get("best_f1")),
                "best_f1": metrics.get("best_f1"),
                "mlqds_aggregate_f1": mlqds.get("aggregate_f1") if isinstance(mlqds, dict) else None,
                "mlqds_range_usefulness_score": (
                    mlqds.get("range_usefulness_score") if isinstance(mlqds, dict) else None
                ),
                "mlqds_range_gap_coverage": (
                    mlqds.get("range_gap_coverage") if isinstance(mlqds, dict) else None
                ),
            }
        )
    return rows


def _runtime_child_args(float32_matmul_precision: str, allow_tf32: bool, amp_mode: str) -> list[str]:
    """Return precision args forwarded to benchmark child entrypoints."""
    return [
        "--float32_matmul_precision",
        str(float32_matmul_precision),
        "--allow_tf32" if allow_tf32 else "--no-allow_tf32",
        "--amp_mode",
        str(amp_mode),
    ]


def _write_text(path: Path, text: str) -> None:
    """Write text, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _run_child_step(
    name: str,
    command: list[str],
    results_dir: Path,
    run_json_name: str,
) -> dict[str, Any]:
    """Run one benchmark child command and build its artifact payload."""
    print(f"[benchmark] {name}: {' '.join(shlex.quote(part) for part in command)}", flush=True)
    proc = _run_capture(command, cwd=_qds_root())
    stdout_path = results_dir / f"{name}_stdout.log"
    _write_text(stdout_path, proc.stdout)
    run_json = _load_json(results_dir / run_json_name)
    payload = {
        "name": name,
        "command": command,
        "returncode": proc.returncode,
        "elapsed_seconds": float(getattr(proc, "elapsed_seconds", 0.0)),
        "stdout_path": str(stdout_path),
        "timings": _parse_timings(proc.stdout),
        "metrics": _matched_summary(run_json),
    }
    if proc.returncode != 0:
        payload["error"] = f"{name} command failed; see {stdout_path}"
    return payload


def _build_parser() -> argparse.ArgumentParser:
    """Build benchmark CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run stable AIS-QDS runtime benchmarks and write a JSON artifact.",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "inference", "both"],
        default="train",
        help="Benchmark training, saved-checkpoint inference, or both.",
    )
    parser.add_argument(
        "--profile",
        choices=PROFILE_CHOICES,
        default=DEFAULT_PROFILE,
        help="Training profile for runtime benchmarking. Training modes require CSV data in --train_extra_args.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed recorded and passed to default profile commands.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="artifacts/benchmarks/runtime",
        help="Directory for benchmark JSON, child stdout, and child run artifacts.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint for inference mode. In both mode, defaults to the checkpoint produced by training.",
    )
    parser.add_argument(
        "--inference_csv_path",
        type=str,
        default=None,
        help="Cleaned AIS CSV for inference mode. Required for --mode inference and --mode both.",
    )
    parser.add_argument(
        "--train_extra_args",
        type=str,
        default=None,
        help="Quoted extra args appended to run_ais_experiment.py, e.g. \"--csv_path ../AISDATA/cleaned/x.csv\".",
    )
    parser.add_argument(
        "--train_batch_sizes",
        type=str,
        default=None,
        help="Comma-separated train_batch_size sweep, e.g. '16,32,64,128'. Only valid with --mode train.",
    )
    parser.add_argument(
        "--sweep_continue_on_failure",
        action="store_true",
        help="Continue a train_batch_sizes sweep after a failed child run. Default stops at first failure.",
    )
    parser.add_argument(
        "--inference_extra_args",
        type=str,
        default=None,
        help="Quoted extra args appended to run_inference.py.",
    )
    parser.add_argument(
        "--amp_mode",
        choices=AMP_MODE_CHOICES,
        default="off",
        help="Optional CUDA autocast mode forwarded to training and saved-checkpoint inference.",
    )
    parser.add_argument(
        "--float32_matmul_precision",
        choices=FLOAT32_MATMUL_PRECISION_CHOICES,
        default="highest",
        help="Torch float32 matmul precision. Use 'high' with --allow_tf32 for TF32 benchmarking.",
    )
    parser.add_argument(
        "--allow_tf32",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow TF32 for CUDA float32 matmul in the wrapper and child runs.",
    )
    parser.add_argument(
        "--artifact_name",
        type=str,
        default="benchmark_runtime.json",
        help="Benchmark artifact filename written under results_dir.",
    )
    return parser


def main() -> None:
    """Run the benchmark wrapper."""
    args = _build_parser().parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint) if args.checkpoint else results_dir / "benchmark_model.pt"

    if args.mode in {"inference", "both"} and not args.inference_csv_path:
        raise SystemExit("--inference_csv_path is required for --mode inference and --mode both.")
    if args.mode == "inference" and not args.checkpoint:
        raise SystemExit("--checkpoint is required for --mode inference.")
    if args.mode in {"train", "both"} and not _extra_args_include_training_data_source(args.train_extra_args):
        raise SystemExit(
            "--mode train/both with --profile range_real_usecase requires --train_extra_args "
            "containing --csv_path or --train_csv_path/--eval_csv_path."
        )
    try:
        train_batch_sizes = _parse_train_batch_sizes(args.train_batch_sizes)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if train_batch_sizes is not None and args.mode != "train":
        raise SystemExit("--train_batch_sizes is only supported with --mode train.")

    runtime_settings = apply_torch_runtime_settings(
        float32_matmul_precision=args.float32_matmul_precision,
        allow_tf32=args.allow_tf32,
    )
    runtime_child_args = _runtime_child_args(args.float32_matmul_precision, bool(args.allow_tf32), args.amp_mode)

    wrapper_command = [sys.executable, "-m", "src.experiments.benchmark_runtime", *sys.argv[1:]]
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "wrapper_command": wrapper_command,
        "mode": args.mode,
        "profile": args.profile,
        "seed": int(args.seed),
        "environment": _environment_metadata(args.amp_mode),
        "torch_runtime": {
            **runtime_settings,
            "amp": amp_runtime_snapshot(args.amp_mode),
        },
        "train_batch_sizes": train_batch_sizes,
        "steps": [],
    }

    failures = 0
    if args.mode == "train" and train_batch_sizes is not None:
        for batch_size in train_batch_sizes:
            train_results = results_dir / f"train_bs{batch_size}"
            checkpoint_for_step = checkpoint.with_name(f"{checkpoint.stem}_bs{batch_size}{checkpoint.suffix or '.pt'}")
            train_command = [
                sys.executable,
                "-m",
                "src.experiments.run_ais_experiment",
                *_profile_train_args(args.profile, int(args.seed), train_results, checkpoint_for_step),
                "--train_batch_size",
                str(batch_size),
                *runtime_child_args,
                *_split_extra_args(args.train_extra_args),
            ]
            step = _run_child_step(f"train_bs{batch_size}", train_command, train_results, "example_run.json")
            step["train_batch_size"] = int(batch_size)
            artifact["steps"].append(step)
            failed = int(step["returncode"] != 0)
            failures += failed
            if failed and not args.sweep_continue_on_failure:
                break
        artifact["train_batch_size_sweep"] = _batch_size_sweep_summary(artifact["steps"])
    elif args.mode in {"train", "both"}:
        train_results = results_dir / "train"
        train_command = [
            sys.executable,
            "-m",
            "src.experiments.run_ais_experiment",
            *_profile_train_args(args.profile, int(args.seed), train_results, checkpoint),
            *runtime_child_args,
            *_split_extra_args(args.train_extra_args),
        ]
        step = _run_child_step("train", train_command, train_results, "example_run.json")
        artifact["steps"].append(step)
        failures += int(step["returncode"] != 0)

    if args.mode in {"inference", "both"}:
        inference_results = results_dir / "inference"
        inference_checkpoint = checkpoint if args.mode == "both" else Path(args.checkpoint or "")
        inference_command = [
            sys.executable,
            "-m",
            "src.experiments.run_inference",
            "--checkpoint",
            str(inference_checkpoint),
            "--csv_path",
            str(args.inference_csv_path),
            "--seed",
            str(args.seed),
            "--results_dir",
            str(inference_results),
            *runtime_child_args,
            *_split_extra_args(args.inference_extra_args),
        ]
        step = _run_child_step("inference", inference_command, inference_results, "inference_run.json")
        artifact["steps"].append(step)
        failures += int(step["returncode"] != 0)

    artifact_path = results_dir / args.artifact_name
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    print(f"[benchmark] wrote artifact: {artifact_path}", flush=True)
    if not artifact["environment"]["gpu_telemetry"]["available"]:
        reason = artifact["environment"]["gpu_telemetry"]["unavailable_reason"]
        print(f"[benchmark] GPU telemetry unavailable: {reason}", flush=True)
    if failures:
        raise SystemExit(f"{failures} benchmark step(s) failed. See {artifact_path}.")


if __name__ == "__main__":
    main()
