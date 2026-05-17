"""Print a compact human-readable summary for one Range_QDS run artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table


def _status(value: object) -> str:
    if value is True:
        return "PASS"
    if value is False:
        return "FAIL"
    return "N/A"


def _format_float(value: object) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.6f}"
    return ""


def _json_compact(value: object) -> str:
    if value in (None, [], {}):
        return ""
    return json.dumps(value, sort_keys=True)


def _nested(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _score_table(run: dict[str, Any]) -> Table:
    table = Table(title="QueryUsefulV1")
    table.add_column("Method")
    table.add_column("Score", justify="right")
    table.add_row("MLQDS", _format_float(_nested(run, "matched", "MLQDS", "query_useful_v1_score")))
    table.add_row("uniform", _format_float(_nested(run, "matched", "uniform", "query_useful_v1_score")))
    table.add_row(
        "DouglasPeucker",
        _format_float(_nested(run, "matched", "DouglasPeucker", "query_useful_v1_score")),
    )
    return table


def _gate_table(run: dict[str, Any]) -> Table:
    table = Table(title="Gates")
    table.add_column("Gate")
    table.add_column("Status")
    table.add_column("Failures")
    rows = [
        (
            "workload_stability",
            _nested(run, "workload_stability_gate", "gate_pass"),
            _nested(run, "workload_stability_gate", "failed_checks"),
        ),
        (
            "support_overlap",
            _nested(run, "support_overlap_gate", "gate_pass"),
            _nested(run, "support_overlap_gate", "failed_checks"),
        ),
        (
            "predictability",
            _nested(run, "predictability_audit", "gate_pass"),
            _nested(run, "predictability_audit", "gate_checks"),
        ),
        (
            "prior_alignment",
            _nested(run, "predictability_audit", "prior_predictive_alignment_gate", "gate_pass"),
            _nested(run, "predictability_audit", "prior_predictive_alignment_gate", "failed_checks"),
        ),
        (
            "target_diffusion",
            _nested(run, "target_diffusion_gate", "gate_pass"),
            _nested(run, "target_diffusion_gate", "failed_checks"),
        ),
        (
            "workload_signature",
            _nested(run, "workload_distribution_comparison", "workload_signature_gate", "all_pass"),
            _nested(run, "workload_distribution_comparison", "workload_signature_gate", "failed_pairs"),
        ),
        (
            "learning_causality",
            _nested(run, "learning_causality_summary", "learning_causality_gate_pass"),
            _nested(run, "learning_causality_summary", "learning_causality_failed_checks"),
        ),
        (
            "prior_sample",
            _nested(run, "learning_causality_summary", "prior_sample_gate_pass"),
            _nested(run, "learning_causality_summary", "prior_sample_gate_failures"),
        ),
        (
            "global_sanity",
            _nested(run, "global_sanity_gate", "gate_pass"),
            _nested(run, "global_sanity_gate", "failed_checks"),
        ),
    ]
    for name, passed, failures in rows:
        table.add_row(name, _status(passed), _json_compact(failures))
    return table


def _causality_table(run: dict[str, Any]) -> Table:
    table = Table(title="Causality Deltas")
    table.add_column("Ablation")
    table.add_column("Delta", justify="right")
    summary = run.get("learning_causality_summary") or {}
    rows = [
        ("shuffled_score", summary.get("shuffled_score_ablation_delta")),
        ("untrained", summary.get("untrained_score_ablation_delta")),
        ("shuffled_prior", summary.get("shuffled_prior_field_ablation_delta")),
        ("no_query_prior", summary.get("no_query_prior_field_ablation_delta")),
        ("no_behavior_head", summary.get("no_behavior_head_ablation_delta")),
        ("no_segment_budget_head", summary.get("no_segment_budget_head_ablation_delta")),
        ("no_fairness_preallocation", summary.get("no_trajectory_fairness_preallocation_ablation_delta")),
    ]
    for name, value in rows:
        table.add_row(name, _format_float(value))
    return table


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_json", type=Path, help="Path to example_run.json")
    parser.add_argument("--plain", action="store_true", help="Disable terminal styling")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run = json.loads(args.run_json.read_text(encoding="utf-8"))
    console = Console(force_terminal=False if args.plain else None)
    console.rule(f"Range_QDS Run Summary: {args.run_json}")
    console.print(_score_table(run))
    console.print(_gate_table(run))
    console.print(_causality_table(run))


if __name__ == "__main__":
    main()
