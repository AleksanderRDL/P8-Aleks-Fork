"""Experiment output payloads and artifact writing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.metrics import MethodEvaluation
from experiments.range_diagnostics import _evaluation_metrics_payload


@dataclass
class ExperimentOutputs:
    """Experiment run output payload."""

    matched_table: str
    shift_table: str
    metrics_dump: dict
    geometric_table: str = ""
    range_usefulness_table: str = ""
    range_compression_audit_table: str = ""


def write_experiment_results(
    *,
    results_dir: str,
    matched_table: str,
    shift_table: str,
    geometric_table: str,
    range_usefulness_table: str,
    learned_fill_table: str,
    learned_fill_diagnostics: dict[str, MethodEvaluation],
    range_learned_fill_summary: dict[str, Any],
    range_compression_audit: dict[str, dict[str, Any]],
    range_compression_audit_table: str,
    range_diagnostics_summary: dict[str, Any],
    workload_distribution_comparison: dict[str, Any],
    range_diagnostics_rows: list[dict[str, Any]],
    dump: dict[str, Any],
) -> Path:
    """Write the standard experiment result artifact set and return its directory."""
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
    return out_dir
