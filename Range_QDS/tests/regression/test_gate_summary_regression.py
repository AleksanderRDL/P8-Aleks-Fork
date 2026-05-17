from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.regression


def _normalize_run_for_regression(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "final_claim_summary": run.get("final_claim_summary"),
        "scores": {
            "has_mlqds": "MLQDS" in (run.get("matched") or {}),
            "has_uniform": "uniform" in (run.get("matched") or {}),
            "has_douglas_peucker": "DouglasPeucker" in (run.get("matched") or {}),
        },
        "gates": {
            "has_workload_stability": "workload_stability_gate" in run,
            "has_support_overlap": "support_overlap_gate" in run,
            "has_predictability": "predictability_audit" in run,
            "has_learning_causality": "learning_causality_summary" in run,
            "has_global_sanity": "global_sanity_gate" in run,
        },
        "config": {
            "model_type": (run.get("config") or {}).get("model", {}).get("model_type"),
            "workload_profile_id": (run.get("config") or {}).get("query", {}).get("workload_profile_id"),
            "selector_type": (run.get("config") or {}).get("model", {}).get("selector_type"),
        },
    }


def test_gate_summary_regression(data_regression: Any) -> None:
    run = {
        "matched": {
            "MLQDS": {"query_useful_v1_score": 0.7},
            "uniform": {"query_useful_v1_score": 0.6},
            "DouglasPeucker": {"query_useful_v1_score": 0.5},
        },
        "config": {
            "model": {
                "model_type": "workload_blind_range_v2",
                "selector_type": "learned_segment_budget_v1",
            },
            "query": {"workload_profile_id": "range_workload_v1"},
        },
        "final_claim_summary": {
            "status": "single_cell_blocked",
            "primary_metric": "QueryUsefulV1",
            "final_success_allowed": False,
            "blocking_gates": ["learning_causality"],
        },
        "workload_stability_gate": {"gate_pass": True},
        "support_overlap_gate": {"gate_pass": True},
        "predictability_audit": {
            "gate_pass": True,
            "prior_predictive_alignment_gate": {"gate_pass": True},
        },
        "learning_causality_summary": {"learning_causality_gate_pass": False},
        "global_sanity_gate": {"gate_pass": True},
    }

    data_regression.check(_normalize_run_for_regression(run))
