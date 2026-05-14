"""Child process execution helpers for benchmark runs."""

from __future__ import annotations

import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.benchmark_runtime import EPOCH_RE, INFERENCE_STEP_RE, PHASE_DONE_RE

DEFAULT_CHILD_STDOUT_TAIL_CHARS = 1_000_000


@dataclass
class BenchmarkChildResult:
    """Completed child process result with retained stdout tail, timings, and elapsed time."""

    returncode: int
    stdout: str
    stdout_truncated: bool
    timings: dict[str, Any]
    elapsed_seconds: float

def _append_stdout_tail(tail_chunks: deque[str], tail_chars: int, line: str, max_chars: int) -> tuple[int, bool]:
    """Append a line to the retained stdout tail and trim old chunks past max_chars."""
    if max_chars <= 0:
        return 0, True

    truncated = False
    if len(line) > max_chars:
        tail_chunks.clear()
        tail_chunks.append(line[-max_chars:])
        return max_chars, True

    tail_chunks.append(line)
    tail_chars += len(line)
    while tail_chars > max_chars and tail_chunks:
        overflow = tail_chars - max_chars
        first = tail_chunks[0]
        truncated = True
        if len(first) <= overflow:
            tail_chars -= len(first)
            tail_chunks.popleft()
        else:
            tail_chunks[0] = first[overflow:]
            tail_chars -= overflow
            break
    return tail_chars, truncated

def _append_timing_line(timings: dict[str, list[dict[str, Any]]], line: str) -> None:
    """Parse one child stdout line into the benchmark timing accumulator."""
    phase_match = PHASE_DONE_RE.search(line)
    if phase_match:
        timings["phase_timings"].append(
            {
                "name": phase_match.group("name").strip(),
                "seconds": float(phase_match.group("seconds")),
            }
        )

    epoch_match = EPOCH_RE.search(line)
    if epoch_match:
        timings["epoch_timings"].append(
            {
                "epoch": int(epoch_match.group("epoch")),
                "total_epochs": int(epoch_match.group("total")),
                "seconds": float(epoch_match.group("seconds")),
                "line": line.strip(),
            }
        )

    inference_match = INFERENCE_STEP_RE.search(line)
    if inference_match:
        timings["inference_step_timings"].append(
            {
                "name": inference_match.group("name").strip(),
                "seconds": float(inference_match.group("seconds")),
                "line": line.strip(),
            }
        )

def _run_capture_streaming(
    command: list[str],
    cwd: Path,
    stdout_path: Path,
    *,
    max_stdout_chars: int = DEFAULT_CHILD_STDOUT_TAIL_CHARS,
) -> BenchmarkChildResult:
    """Run a child command while streaming stdout to console and a log file."""
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    tail_chunks: deque[str] = deque()
    tail_chars = 0
    stdout_truncated = False
    timings: dict[str, list[dict[str, Any]]] = {
        "phase_timings": [],
        "epoch_timings": [],
        "inference_step_timings": [],
    }
    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    try:
        with open(stdout_path, "w", encoding="utf-8") as log:
            if proc.stdout is None:
                raise RuntimeError("benchmark child stdout pipe was not created.")
            for line in proc.stdout:
                tail_chars, line_truncated = _append_stdout_tail(tail_chunks, tail_chars, line, max_stdout_chars)
                stdout_truncated = stdout_truncated or line_truncated
                _append_timing_line(timings, line)
                log.write(line)
                log.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
        returncode = int(proc.wait())
    except BaseException:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        raise
    elapsed = time.perf_counter() - started
    return BenchmarkChildResult(
        returncode=returncode,
        stdout="".join(tail_chunks),
        stdout_truncated=stdout_truncated,
        timings=timings,
        elapsed_seconds=float(elapsed),
    )
