from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from experiments.benchmark_report import (
    QUERY_DRIVEN_FINAL_COMPRESSION_RATIOS,
    QUERY_DRIVEN_FINAL_COVERAGE_TARGETS,
    _audit_ratio_prefix,
    _row_from_run,
    query_driven_final_grid_summary,
)

pytestmark = pytest.mark.regression


def _final_grid_row(coverage: float) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_label": f"coverage_{coverage:.2f}",
        "returncode": 0,
        "query_target_coverage": coverage,
        "mlqds_primary_metric": "query_useful_v1",
        "workload_stability_gate_pass": True,
        "support_overlap_gate_pass": True,
        "predictability_gate_pass": True,
        "prior_predictive_alignment_gate_pass": True,
        "target_diffusion_gate_pass": True,
        "workload_signature_gate_pass": True,
        "learning_causality_gate_pass": True,
        "prior_sample_gate_pass": True,
        "global_sanity_gate_pass": True,
    }
    for ratio in QUERY_DRIVEN_FINAL_COMPRESSION_RATIOS:
        prefix = _audit_ratio_prefix(ratio)
        base = 0.50 + coverage + ratio
        row[f"{prefix}_mlqds_query_useful_v1"] = base + 0.05
        row[f"{prefix}_uniform_query_useful_v1"] = base
        row[f"{prefix}_douglas_peucker_query_useful_v1"] = base - 0.05
    return row


def _minimal_query_driven_run_json_fixture() -> dict[str, Any]:
    return {
        "matched": {
            "MLQDS": {
                "query_useful_v1_score": 0.70,
                "range_usefulness_score": 0.65,
                "range_point_f1": 0.60,
            },
            "uniform": {
                "query_useful_v1_score": 0.60,
                "range_usefulness_score": 0.55,
                "range_point_f1": 0.50,
            },
            "DouglasPeucker": {
                "query_useful_v1_score": 0.50,
                "range_usefulness_score": 0.45,
                "range_point_f1": 0.40,
            },
        },
        "config": {
            "data": {"validation_split_mode": "temporal"},
            "model": {
                "model_type": "workload_blind_range_v2",
                "selector_type": "learned_segment_budget_v1",
                "compression_ratio": 0.05,
                "checkpoint_score_variant": "query_useful_v1",
            },
            "query": {
                "workload_profile_id": "range_workload_v1",
                "target_coverage": 0.10,
                "n_queries": 48,
                "max_queries": 256,
            },
            "baselines": {"final_metrics_mode": "diagnostic"},
        },
        "final_claim_summary": {
            "status": "single_cell_pass",
            "primary_metric": "QueryUsefulV1",
            "final_success_allowed": False,
            "blocking_gates": [],
        },
        "workload_blind_protocol": {
            "enabled": True,
            "primary_masks_frozen_before_eval_query_scoring": True,
            "audit_masks_frozen_before_eval_query_scoring": True,
            "eval_geometry_blend_allowed": False,
        },
        "workload_stability_gate": {"gate_pass": True, "configured_target_coverage": 0.10},
        "support_overlap_gate": {"gate_pass": True},
        "predictability_audit": {
            "gate_pass": True,
            "metrics": {"spearman": 0.2, "lift_at_5_percent": 1.5},
            "prior_predictive_alignment_gate": {"gate_pass": True, "failed_checks": []},
        },
        "target_diffusion_gate": {"gate_pass": True, "failed_checks": []},
        "workload_distribution_comparison": {
            "workload_signature_gate": {"all_pass": True, "all_available": True, "pairs": {}}
        },
        "learning_causality_summary": {
            "learning_causality_gate_pass": True,
            "learning_causality_failed_checks": [],
            "prior_sample_gate_pass": True,
            "learned_controlled_retained_slot_fraction": 0.30,
            "planned_learned_controlled_retained_slot_fraction": 0.30,
            "actual_learned_controlled_retained_slot_fraction": 0.30,
            "learning_causality_delta_gate": {
                "min_material_query_useful_delta": 0.01,
                "mlqds_uniform_query_useful_gap": 0.10,
            },
            "learned_segment_selector_config": {
                "geometry_gain_weight": 0.2,
                "segment_score_blend_weight": 0.5,
                "fairness_preallocation_enabled": True,
                "length_repair_fraction": 0.0,
            },
        },
        "global_sanity_gate": {
            "gate_pass": True,
            "failed_checks": [],
            "avg_length_preserved": 0.80,
            "avg_sed_ratio_vs_uniform": 1.0,
        },
        "final_metrics_mode": "diagnostic",
    }


def test_query_driven_final_grid_summary_regression(data_regression: Any) -> None:
    rows = [_final_grid_row(coverage) for coverage in QUERY_DRIVEN_FINAL_COVERAGE_TARGETS]
    run_config = {
        "profile_settings": {
            "final_product_candidate": True,
            "range_coverage_sweep_targets": list(QUERY_DRIVEN_FINAL_COVERAGE_TARGETS),
            "range_compression_sweep_ratios": list(QUERY_DRIVEN_FINAL_COMPRESSION_RATIOS),
        }
    }

    summary = query_driven_final_grid_summary(rows, run_config)

    data_regression.check(
        {
            "status": summary["status"],
            "final_success_allowed": summary["final_success_allowed"],
            "required_cell_count": summary["required_cell_count"],
            "observed_cell_count": summary["observed_cell_count"],
            "required_single_run_gate_names": summary["required_single_run_gate_names"],
            "numeric_success_bars_pass": summary["numeric_success_bars_pass"],
            "failed_checks": summary["failed_checks"],
        }
    )


def test_benchmark_row_field_set_regression(data_regression: Any, tmp_path: Path) -> None:
    stdout_path = tmp_path / "stdout.log"
    stdout_path.write_text("", encoding="utf-8")
    run_json_path = tmp_path / "example_run.json"
    run_json_path.write_text("{}", encoding="utf-8")

    row = _row_from_run(
        workload="range",
        run_label="fixture",
        command=["uv", "run", "--group", "dev", "--", "python", "-m", "experiments.run_ais_experiment"],
        returncode=0,
        elapsed_seconds=1.0,
        run_dir=tmp_path,
        stdout_path=stdout_path,
        run_json_path=run_json_path,
        timings={"phase_timings": [], "epoch_timings": [], "inference_step_timings": []},
        run_json=_minimal_query_driven_run_json_fixture(),
    )

    data_regression.check(
        {
            "field_count": len(row),
            "fields": sorted(row.keys()),
            "required_query_driven_fields_present": {
                key: key in row
                for key in (
                    "prior_predictive_alignment_gate_pass",
                    "workload_signature_gate_pass",
                    "learning_causality_gate_pass",
                    "prior_sample_gate_pass",
                    "learned_segment_fairness_preallocation_enabled",
                    "train_range_acceptance_attempts",
                    "eval_range_acceptance_attempts",
                )
            },
        }
    )
