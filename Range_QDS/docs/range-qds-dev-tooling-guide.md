# Developer Tooling Guide For Range_QDS

This guide explains how to introduce and use four development tools in the Range_QDS query-driven rework workflow:

1. `jq`
2. `hypothesis`
3. `pytest-regressions`
4. `rich`

The goal is not to add tooling for its own sake. The goal is to make the current redesign workflow safer, faster, and easier to reason about.

The current project depends heavily on JSON artifacts, strict gates, workload signatures, predictor/selector ablations, and benchmark summaries. The best current candidate is promising but still blocked by learning-causality and global-sanity issues. Tooling should help expose those problems clearly, not hide them behind prettier logs or brittle snapshots.

---

## 1. Tooling principles

### Use tools to enforce invariants, not to decorate the project

The most valuable checks in this project are not ordinary unit tests like “function returns x.” They are workflow and protocol invariants:

```text
eval queries must not affect compression
retained masks must freeze before eval scoring
final workloads must be healthy and signature-stable
prior fields must be train-derived only
zero-prior ablations must preserve extent and metadata
selector diagnostics must prove learned control is material
benchmark summaries must expose all required gates
```

The tools below should serve those invariants.

### Do not turn experiment metrics into brittle tests

Training runs are noisy. Do not snapshot full `example_run.json` metrics from trained models and expect exact equality. Use regression testing for stable schema/report shape, not stochastic learning results.

### Keep heavy tooling out of hot model paths

None of these tools should slow down model forward passes, target construction, or selector execution in production-like experiment runs.

### Prefer small, readable checks

A useful tool integration is one that makes failure clearer. Avoid giant snapshots, huge generated test cases, or excessive rich formatting.

---

## 2. Recommended rollout order

Use this order:

```text
Phase 1: jq
Phase 2: hypothesis
Phase 3: pytest-regressions
Phase 4: rich
```

Reasoning:

1. `jq` gives immediate artifact-inspection value.
2. `hypothesis` catches subtle generator/selector/prior edge cases.
3. `pytest-regressions` protects stable report and schema shape.
4. `rich` improves human readability once correctness checks are in place.

Do not start with `rich`. Pretty output is useful only after the artifact contracts are reliable.

---

## 3. jq

### Role

Use `jq` for quick inspection of JSON artifacts.

Most Range_QDS experiment artifacts are JSON-heavy:

```text
example_run.json
benchmark_report.json
query_generation_diagnostics
workload_stability_gate
support_overlap_gate
predictability_audit
learning_causality_summary
global_sanity_gate
final_claim_summary
```

`jq` should be the default way to answer:

```text
Which gate failed?
Did MLQDS beat uniform and Douglas-Peucker?
Did prior predictability pass?
Did learning causality fail because of shuffled scores, untrained model, or no-prior ablation?
Did generation exhaust?
Did workload signature drift?
```

### Install and dependency policy

Treat `jq` as a system/dev command-line tool, not a Python dependency.

Good:

```text
apt install jq
brew install jq
devcontainer package
CI image package
```

Avoid:

```text
adding jq as a Python runtime dependency
requiring jq inside model/training code
```

### Recommended file layout

Add reusable filters here:

```text
Range_QDS/scripts/jq/run_summary.jq
Range_QDS/scripts/jq/gates.jq
Range_QDS/scripts/jq/scores.jq
Range_QDS/scripts/jq/causality.jq
Range_QDS/scripts/jq/generator_health.jq
Range_QDS/scripts/jq/predictability.jq
```

This keeps shell commands short and prevents every agent from reinventing filters.

### Suggested Makefile targets

Add these targets to `Range_QDS/Makefile` or the root `Makefile`.

```make
RUN ?= artifacts/results/latest/example_run.json

inspect-run:
	jq -f scripts/jq/run_summary.jq $(RUN)

inspect-gates:
	jq -f scripts/jq/gates.jq $(RUN)

inspect-scores:
	jq -f scripts/jq/scores.jq $(RUN)

inspect-causality:
	jq -f scripts/jq/causality.jq $(RUN)

inspect-generator:
	jq -f scripts/jq/generator_health.jq $(RUN)

inspect-predictability:
	jq -f scripts/jq/predictability.jq $(RUN)
```

### Example filters

#### `scripts/jq/run_summary.jq`

```jq
{
  final_claim_summary,
  scores: {
    mlqds_query_useful_v1: .matched.MLQDS.query_useful_v1_score,
    uniform_query_useful_v1: .matched.uniform.query_useful_v1_score,
    douglas_peucker_query_useful_v1: .matched.DouglasPeucker.query_useful_v1_score,
    mlqds_vs_uniform: (
      .matched.MLQDS.query_useful_v1_score
      - .matched.uniform.query_useful_v1_score
    ),
    mlqds_vs_douglas_peucker: (
      .matched.MLQDS.query_useful_v1_score
      - .matched.DouglasPeucker.query_useful_v1_score
    )
  },
  gates: {
    workload_stability: .workload_stability_gate.gate_pass,
    support_overlap: .support_overlap_gate.gate_pass,
    predictability: .predictability_audit.gate_pass,
    prior_predictive_alignment: .predictability_audit.prior_predictive_alignment_gate.gate_pass,
    target_diffusion: .target_diffusion_gate.gate_pass,
    workload_signature: .workload_distribution_comparison.workload_signature_gate.all_pass,
    learning_causality: .learning_causality_summary.learning_causality_gate_pass,
    global_sanity: .global_sanity_gate.gate_pass
  }
}
```

#### `scripts/jq/gates.jq`

```jq
{
  final_claim_blocking_gates: .final_claim_summary.blocking_gates,
  workload_stability: {
    pass: .workload_stability_gate.gate_pass,
    failed: .workload_stability_gate.failed_checks
  },
  support_overlap: {
    pass: .support_overlap_gate.gate_pass,
    failed: .support_overlap_gate.failed_checks
  },
  predictability: {
    pass: .predictability_audit.gate_pass,
    checks: .predictability_audit.gate_checks,
    metrics: .predictability_audit.metrics
  },
  prior_predictive_alignment: .predictability_audit.prior_predictive_alignment_gate,
  target_diffusion: {
    pass: .target_diffusion_gate.gate_pass,
    failed: .target_diffusion_gate.failed_checks
  },
  workload_signature: .workload_distribution_comparison.workload_signature_gate,
  learning_causality: {
    pass: .learning_causality_summary.learning_causality_gate_pass,
    failed: .learning_causality_summary.learning_causality_failed_checks,
    deltas: {
      shuffled_score: .learning_causality_summary.shuffled_score_ablation_delta,
      untrained: .learning_causality_summary.untrained_score_ablation_delta,
      shuffled_prior: .learning_causality_summary.shuffled_prior_field_ablation_delta,
      no_query_prior: .learning_causality_summary.no_query_prior_field_ablation_delta,
      no_behavior_head: .learning_causality_summary.no_behavior_head_ablation_delta,
      no_segment_budget_head: .learning_causality_summary.no_segment_budget_head_ablation_delta,
      no_fairness_preallocation: .learning_causality_summary.no_trajectory_fairness_preallocation_ablation_delta
    }
  },
  global_sanity: {
    pass: .global_sanity_gate.gate_pass,
    failed: .global_sanity_gate.failed_checks,
    length: .global_sanity_gate.avg_length_preserved,
    sed_ratio: .global_sanity_gate.avg_sed_ratio_vs_uniform
  }
}
```

#### `scripts/jq/generator_health.jq`

```jq
{
  workload_stability_gate,
  train_generation: .query_generation_diagnostics.train.query_generation,
  train_acceptance: .query_generation_diagnostics.train.range_acceptance,
  eval_generation: .query_generation_diagnostics.eval.query_generation,
  eval_acceptance: .query_generation_diagnostics.eval.range_acceptance,
  selection_generation: .query_generation_diagnostics.selection.query_generation,
  selection_acceptance: .query_generation_diagnostics.selection.range_acceptance,
  workload_signature_gate: .workload_distribution_comparison.workload_signature_gate
}
```

### One-line checks

Useful pass/fail command for a strict single-cell run:

```bash
jq -e '
  .workload_stability_gate.gate_pass == true and
  .support_overlap_gate.gate_pass == true and
  .predictability_audit.gate_pass == true and
  .predictability_audit.prior_predictive_alignment_gate.gate_pass == true and
  .target_diffusion_gate.gate_pass == true and
  .workload_distribution_comparison.workload_signature_gate.all_pass == true and
  .learning_causality_summary.learning_causality_gate_pass == true and
  .global_sanity_gate.gate_pass == true
' artifacts/results/.../example_run.json
```

Useful score check:

```bash
jq '{
  mlqds: .matched.MLQDS.query_useful_v1_score,
  uniform: .matched.uniform.query_useful_v1_score,
  dp: .matched.DouglasPeucker.query_useful_v1_score,
  beats_uniform: (.matched.MLQDS.query_useful_v1_score > .matched.uniform.query_useful_v1_score),
  beats_dp: (.matched.MLQDS.query_useful_v1_score > .matched.DouglasPeucker.query_useful_v1_score)
}' artifacts/results/.../example_run.json
```

### What not to do with jq

Do not use `jq` as a replacement for Python tests. It is an inspection tool and lightweight assertion tool.

Do not bury complex acceptance logic in shell scripts. Acceptance logic belongs in Python code and tests; `jq` should expose fields clearly.

---

## 4. hypothesis

### Role

Use `hypothesis` for property-based tests around invariants.

This project has several failure modes that fixed example tests can miss:

```text
query generation edge cases
profile quota drift
coverage guard corner cases
selector budget rounding
short trajectory budgets
zero/flat prior fields
out-of-extent prior sampling
shape/metadata preservation in ablations
```

`hypothesis` should generate many small cases and prove invariants.

### Good targets

Use Hypothesis for:

```text
queries/query_generator.py
queries/workload_profiles.py
training/query_prior_fields.py
training/model_features.py
simplification/learned_segment_budget.py
evaluation/query_useful_v1.py
evaluation/evaluate_methods.py
experiments/range_diagnostics.py
```

Do not use Hypothesis for:

```text
full training loops
GPU-heavy tests
large benchmark runs
tests with real AIS files
tests with stochastic model convergence
```

### Recommended test location

Add:

```text
Range_QDS/tests/property/
Range_QDS/tests/property/test_workload_profile_properties.py
Range_QDS/tests/property/test_query_prior_field_properties.py
Range_QDS/tests/property/test_learned_segment_selector_properties.py
Range_QDS/tests/property/test_query_useful_properties.py
```

Or, if you prefer flat tests:

```text
Range_QDS/tests/test_property_workload_profiles.py
Range_QDS/tests/test_property_query_prior_fields.py
Range_QDS/tests/test_property_learned_segment_selector.py
```

### pytest marker

Add a marker to `pytest.ini` or `pyproject.toml` if not already present:

```toml
[tool.pytest.ini_options]
markers = [
  "property: property-based tests using hypothesis",
]
```

Then use:

```python
import pytest

pytestmark = pytest.mark.property
```

### Recommended Hypothesis settings

Use conservative defaults:

```python
from hypothesis import settings

FAST_PROPERTY_SETTINGS = settings(
    max_examples=50,
    deadline=None,
)

DEEP_PROPERTY_SETTINGS = settings(
    max_examples=200,
    deadline=None,
)
```

Use `deadline=None` because PyTorch tensor construction can have variable timing.

For CI, start with `max_examples=50`. Run deeper locally or in nightly CI.

### Property test examples

#### Profile plans preserve exact requested count

```python
from hypothesis import given, settings, strategies as st

from queries.query_generator import _profile_query_plan
from queries.workload_profiles import range_workload_profile


@given(
    requested=st.integers(min_value=1, max_value=512),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=100, deadline=None)
def test_profile_query_plan_counts_sum_to_requested(requested: int, seed: int) -> None:
    profile = range_workload_profile("range_workload_v1")

    plan = _profile_query_plan(profile, requested_queries=requested, workload_seed=seed)

    assert len(plan["anchor_family_sequence"]) == requested
    assert len(plan["footprint_family_sequence"]) == requested
    assert sum(plan["anchor_family_planned_counts"].values()) == requested
    assert sum(plan["footprint_family_planned_counts"].values()) == requested
```

#### Prefixes stay close to configured family mix

```python
@given(
    requested=st.integers(min_value=16, max_value=256),
    prefix=st.integers(min_value=8, max_value=256),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=100, deadline=None)
def test_profile_query_plan_prefixes_remain_balanced(
    requested: int,
    prefix: int,
    seed: int,
) -> None:
    prefix = min(prefix, requested)
    profile = range_workload_profile("range_workload_v1")
    plan = _profile_query_plan(profile, requested_queries=requested, workload_seed=seed)

    prefix_families = plan["anchor_family_sequence"][:prefix]
    counts = {name: prefix_families.count(name) for name in profile.anchor_family_weights}
    assert sum(counts.values()) == prefix

    # Largest-remainder prefix plans should not completely drop a positive-weight
    # family once the prefix is large enough.
    if prefix >= 32:
        for family, weight in profile.anchor_family_weights.items():
            if weight > 0.0:
                assert counts[family] > 0
```

#### Zero-prior ablation preserves extent and tensor shape

```python
from hypothesis import given, settings, strategies as st
import torch

from training.query_prior_fields import zero_query_prior_field_like


@given(
    grid_bins=st.integers(min_value=2, max_value=32),
    time_bins=st.integers(min_value=1, max_value=16),
)
@settings(max_examples=50, deadline=None)
def test_zero_prior_field_preserves_metadata_shape(grid_bins: int, time_bins: int) -> None:
    prior = {
        "field_names": ("spatial_query_hit_probability",),
        "extent": {
            "t_min": 0.0,
            "t_max": 10.0,
            "lat_min": 1.0,
            "lat_max": 2.0,
            "lon_min": 3.0,
            "lon_max": 4.0,
        },
        "grid_bins": grid_bins,
        "time_bins": time_bins,
        "spatial_query_hit_probability": torch.ones((grid_bins, grid_bins)),
        "contains_eval_queries": False,
        "contains_validation_queries": False,
    }

    zeroed = zero_query_prior_field_like(prior)

    assert zeroed["extent"] == prior["extent"]
    assert zeroed["grid_bins"] == prior["grid_bins"]
    assert zeroed["time_bins"] == prior["time_bins"]
    assert zeroed["spatial_query_hit_probability"].shape == prior["spatial_query_hit_probability"].shape
    assert torch.count_nonzero(zeroed["spatial_query_hit_probability"]).item() == 0
```

#### Selector budget never exceeds total budget

```python
from hypothesis import given, settings, strategies as st
import torch

from simplification.learned_segment_budget import simplify_with_learned_segment_budget_v1


@given(
    trajectory_count=st.integers(min_value=1, max_value=12),
    points_per_trajectory=st.integers(min_value=4, max_value=128),
    compression_ratio=st.floats(min_value=0.01, max_value=0.50, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100, deadline=None)
def test_learned_segment_selector_respects_budget(
    trajectory_count: int,
    points_per_trajectory: int,
    compression_ratio: float,
) -> None:
    total = trajectory_count * points_per_trajectory
    scores = torch.linspace(0.0, 1.0, steps=total)
    boundaries = [
        (idx * points_per_trajectory, (idx + 1) * points_per_trajectory)
        for idx in range(trajectory_count)
    ]

    mask = simplify_with_learned_segment_budget_v1(
        scores,
        boundaries,
        compression_ratio,
    )

    expected_budget = sum(
        min(points_per_trajectory, max(2, int(__import__("math").ceil(compression_ratio * points_per_trajectory))))
        for _ in range(trajectory_count)
    )
    assert int(mask.sum().item()) <= expected_budget
```

### What Hypothesis should enforce for this project

High-value properties:

```text
profile query plans sum exactly to requested count
prefix-balanced plans do not create avoidable family drift
coverage guard never allows final coverage above target + tolerance
range_workload_v1 does not silently use uncovered_anchor_chasing
zeroed prior fields preserve extent, bins, metadata, and shape
out-of-extent zero mode returns zero sampled prior features
nearest mode only clamps when explicitly configured
selector never exceeds retained budget
selector reports learned/skeleton/fallback attribution consistently
endpoint sanity is preserved where configured
query-local interpolation does not reward outside-query anchors without local retained evidence
final grid summary blocks if any child gate is false
```

### What not to do with Hypothesis

Do not generate arbitrary floating tensors into training loops.

Do not write tests that depend on stochastic convergence.

Do not use large `max_examples` by default. Start with 50-100.

Do not create property tests that require internet, GPU, real AIS files, or long benchmark artifacts.

---

## 5. pytest-regressions

### Role

Use `pytest-regressions` to protect stable report and schema outputs.

This project has many structured outputs that can silently drift:

```text
benchmark row fields
final grid summary fields
gate summary fields
normalized artifact summaries
query-generation diagnostic schemas
```

Regression tests should catch accidental field removal, renaming, or shape drift.

### Good uses

Use regression snapshots for:

```text
small normalized dictionaries
benchmark row field sets
final_claim_summary shape
query_driven_final_grid_summary output from synthetic fixture rows
workload_signature_gate summary shape
predictability_audit summary shape
learning_causality_summary selected fields
```

### Bad uses

Do not snapshot:

```text
full example_run.json from a trained model
full stdout logs
runtime seconds
timestamps
absolute paths
GPU memory values
random seeds unless relevant
large raw query arrays
raw floating model metrics from stochastic training
```

### Recommended file layout

```text
Range_QDS/tests/regression/
Range_QDS/tests/regression/test_benchmark_report_regression.py
Range_QDS/tests/regression/test_gate_summary_regression.py
Range_QDS/tests/regression/test_query_generation_regression.py
```

Snapshot baselines will usually live beside the tests.

### Normalization helper

Create a helper before snapshotting artifacts:

```python
def normalize_run_for_regression(run: dict) -> dict:
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
```

### Example regression test for final grid summary

```python
def test_query_driven_final_grid_summary_regression(data_regression):
    rows = [
        _final_grid_row(coverage)
        for coverage in (0.05, 0.10, 0.15, 0.30)
    ]
    run_config = {
        "profile_settings": {
            "final_product_candidate": True,
            "range_coverage_sweep_targets": [0.05, 0.10, 0.15, 0.30],
            "range_compression_sweep_ratios": [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30],
        }
    }

    summary = query_driven_final_grid_summary(rows, run_config)

    stable = {
        "status": summary["status"],
        "final_success_allowed": summary["final_success_allowed"],
        "required_cell_count": summary["required_cell_count"],
        "observed_cell_count": summary["observed_cell_count"],
        "required_single_run_gate_names": summary["required_single_run_gate_names"],
        "numeric_success_bars_pass": summary["numeric_success_bars_pass"],
        "failed_checks": summary["failed_checks"],
    }
    data_regression.check(stable)
```

### Example regression test for benchmark row field set

```python
def test_benchmark_row_field_set_regression(data_regression, tmp_path):
    row = _row_from_run(
        workload="range",
        run_label="fixture",
        command=["python", "-m", "experiments.run_ais_experiment"],
        returncode=0,
        elapsed_seconds=1.0,
        run_dir=tmp_path,
        stdout_path=tmp_path / "stdout.log",
        run_json_path=tmp_path / "example_run.json",
        timings={"phase_timings": [], "epoch_timings": [], "inference_step_timings": []},
        run_json=minimal_query_driven_run_json_fixture(),
    )

    data_regression.check(
        {
            "field_count": len(row),
            "fields": sorted(row.keys()),
        }
    )
```

### Snapshot update policy

Only update regression snapshots when the artifact contract intentionally changes.

Use a commit message like:

```text
Update benchmark report regression after adding prior alignment fields
```

Do not update snapshots just to make tests pass without reviewing field changes.

### What pytest-regressions should protect

High-value artifact contracts:

```text
final_claim_summary has status/final_success_allowed/blocking_gates
query_driven_final_grid_summary includes all required child gates
benchmark rows include prior_predictive_alignment fields
benchmark rows include generator health fields
benchmark rows include learned-segment selector config
benchmark rows include fairness-preallocation ablation delta
workload_signature_gate includes query-count and normalized-hit diagnostics
predictability audit includes per-head diagnostics
```

---

## 6. rich

### Role

Use `rich` for human-readable terminal summaries and optional CLI presentation.

This is useful because Range_QDS runs produce many gates and nested diagnostics. A readable table can save time and prevent misreading an artifact.

Use `rich` for:

```text
summarizing one example_run.json
printing gate pass/fail tables
printing score comparisons
printing causality deltas
printing generator health diagnostics
printing final-grid summary
```

Do not use `rich` for correctness. Tests and schemas enforce correctness.

### Recommended file layout

```text
Range_QDS/scripts/summarize_run.py
Range_QDS/experiments/rich_report.py
```

Keep `rich` code separate from core experiment logic where possible.

### Suggested command

```bash
python -m scripts.summarize_run artifacts/results/.../example_run.json
```

Or:

```bash
python scripts/summarize_run.py artifacts/results/.../example_run.json
```

### Example `rich` summary script

```python
from __future__ import annotations

import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table


def _status(value: object) -> str:
    return "PASS" if value is True else "FAIL" if value is False else "N/A"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: summarize_run.py <example_run.json>")

    path = Path(sys.argv[1])
    run = json.loads(path.read_text(encoding="utf-8"))
    console = Console()

    console.rule(f"Range_QDS Run Summary: {path}")

    scores = Table(title="QueryUsefulV1")
    scores.add_column("Method")
    scores.add_column("Score", justify="right")
    scores.add_row("MLQDS", f"{run['matched']['MLQDS'].get('query_useful_v1_score', 0.0):.6f}")
    scores.add_row("uniform", f"{run['matched']['uniform'].get('query_useful_v1_score', 0.0):.6f}")
    scores.add_row("DouglasPeucker", f"{run['matched']['DouglasPeucker'].get('query_useful_v1_score', 0.0):.6f}")
    console.print(scores)

    gates = Table(title="Gates")
    gates.add_column("Gate")
    gates.add_column("Status")
    gates.add_column("Failures")

    gate_rows = [
        ("workload_stability", run.get("workload_stability_gate", {}).get("gate_pass"), run.get("workload_stability_gate", {}).get("failed_checks")),
        ("support_overlap", run.get("support_overlap_gate", {}).get("gate_pass"), run.get("support_overlap_gate", {}).get("failed_checks")),
        ("predictability", run.get("predictability_audit", {}).get("gate_pass"), run.get("predictability_audit", {}).get("gate_checks")),
        ("prior_alignment", run.get("predictability_audit", {}).get("prior_predictive_alignment_gate", {}).get("gate_pass"), run.get("predictability_audit", {}).get("prior_predictive_alignment_gate", {}).get("failed_checks")),
        ("target_diffusion", run.get("target_diffusion_gate", {}).get("gate_pass"), run.get("target_diffusion_gate", {}).get("failed_checks")),
        ("workload_signature", run.get("workload_distribution_comparison", {}).get("workload_signature_gate", {}).get("all_pass"), None),
        ("learning_causality", run.get("learning_causality_summary", {}).get("learning_causality_gate_pass"), run.get("learning_causality_summary", {}).get("learning_causality_failed_checks")),
        ("global_sanity", run.get("global_sanity_gate", {}).get("gate_pass"), run.get("global_sanity_gate", {}).get("failed_checks")),
    ]

    for name, passed, failures in gate_rows:
        gates.add_row(name, _status(passed), json.dumps(failures) if failures else "")

    console.print(gates)


if __name__ == "__main__":
    main()
```

### CI behavior

Avoid making CI logs too fancy.

Recommended:

```python
from rich.console import Console

console = Console(
    force_terminal=False,
    no_color=None,
)
```

If you need plain output in CI, use:

```bash
NO_COLOR=1
```

or add a CLI flag:

```text
--plain
```

### What not to do with rich

Do not replace JSON artifacts with rich text output.

Do not remove existing machine-readable reports.

Do not make tests depend on ANSI-colored output.

Do not put `rich` imports in model files, selector files, metric functions, or training inner loops.

---

## 7. Combined workflow after introducing tools

### After an experiment run

Use:

```bash
make inspect-run RUN=artifacts/results/.../example_run.json
make inspect-gates RUN=artifacts/results/.../example_run.json
make inspect-causality RUN=artifacts/results/.../example_run.json
```

Then optionally:

```bash
python scripts/summarize_run.py artifacts/results/.../example_run.json
```

### After changing generator/profile code

Run:

```bash
pytest tests/test_query_driven_rework.py tests/test_query_coverage_generation.py
pytest tests/property/test_workload_profile_properties.py
```

Then inspect a generated artifact with:

```bash
make inspect-generator RUN=artifacts/results/.../example_run.json
```

### After changing query-prior fields

Run:

```bash
pytest tests/test_query_driven_rework.py
pytest tests/property/test_query_prior_field_properties.py
make inspect-predictability RUN=artifacts/results/.../example_run.json
```

### After changing selector logic

Run:

```bash
pytest tests/test_query_driven_rework.py
pytest tests/property/test_learned_segment_selector_properties.py
make inspect-causality RUN=artifacts/results/.../example_run.json
```

### After changing benchmark report fields

Run:

```bash
pytest tests/test_benchmark_runner.py
pytest tests/regression/test_benchmark_report_regression.py
```

---

## 8. Suggested Makefile additions

Add only if these fit the existing Makefile style.

```make
RUN ?= artifacts/results/latest/example_run.json

inspect-run:
	jq -f scripts/jq/run_summary.jq $(RUN)

inspect-gates:
	jq -f scripts/jq/gates.jq $(RUN)

inspect-scores:
	jq -f scripts/jq/scores.jq $(RUN)

inspect-causality:
	jq -f scripts/jq/causality.jq $(RUN)

inspect-generator:
	jq -f scripts/jq/generator_health.jq $(RUN)

inspect-predictability:
	jq -f scripts/jq/predictability.jq $(RUN)

summarize-run:
	$(PYTHON) scripts/summarize_run.py $(RUN)

test-property:
	$(PYTHON) -m pytest tests/property

test-regression:
	$(PYTHON) -m pytest tests/regression
```

If `tests/property` or `tests/regression` do not exist yet, add the targets only after creating the directories.

---

## 9. Suggested implementation checkpoint

Use one focused tooling checkpoint.

### Scope

```text
Add jq filters and Make targets.
Add initial Hypothesis property tests for workload profile, query prior fields, and learned segment selector.
Add normalized pytest-regressions snapshots for benchmark final-grid summary and benchmark row field set.
Add optional Rich run-summary script.
```

### Expected files

```text
Range_QDS/scripts/jq/run_summary.jq
Range_QDS/scripts/jq/gates.jq
Range_QDS/scripts/jq/scores.jq
Range_QDS/scripts/jq/causality.jq
Range_QDS/scripts/jq/generator_health.jq
Range_QDS/scripts/jq/predictability.jq
Range_QDS/scripts/summarize_run.py
Range_QDS/tests/property/test_workload_profile_properties.py
Range_QDS/tests/property/test_query_prior_field_properties.py
Range_QDS/tests/property/test_learned_segment_selector_properties.py
Range_QDS/tests/regression/test_benchmark_report_regression.py
Range_QDS/tests/regression/test_gate_summary_regression.py
Range_QDS/Makefile
Range_QDS/docs/dev-tooling-guide.md
```

### Tests to run

```bash
cd Range_QDS

python -m ruff check data evaluation experiments models queries simplification training scripts tests
python -m pyright data evaluation experiments models queries simplification training tests

python -m pytest tests/property -q
python -m pytest tests/regression -q
python -m pytest tests/test_query_driven_rework.py -q
python -m pytest tests/test_benchmark_runner.py -q
python -m pytest tests -q
```

### Stop condition

Stop after tooling is integrated and tests pass. Do not mix this checkpoint with model/selector/generator research changes.

---

## 10. Risks and mitigations

### Risk: Hypothesis creates flaky tests

Mitigation:

```text
Use deterministic seeds where needed.
Avoid full training loops.
Avoid GPU.
Limit max_examples.
Disable timing deadlines.
Only test invariants that should always hold.
```

### Risk: regression snapshots become noise

Mitigation:

```text
Snapshot normalized summaries only.
Strip runtime, paths, timestamps, stochastic metrics.
Keep snapshots small.
Require intentional review when updating snapshots.
```

### Risk: rich output replaces machine-readable artifacts

Mitigation:

```text
Keep JSON artifacts authoritative.
Use Rich only for display.
Do not parse Rich output in tests.
```

### Risk: jq filters become hidden acceptance logic

Mitigation:

```text
Keep acceptance in Python code/tests.
Use jq only for inspection and lightweight manual checks.
Document filters as diagnostic helpers.
```

### Risk: too much tooling distracts from research blockers

Mitigation:

```text
Implement tooling in one checkpoint.
Then return to the active scientific blocker.
Do not keep adding tools after the core workflow is improved.
```

---

## 11. Definition of done

This tooling introduction is done when:

```text
jq filters make one-run gate diagnosis easy
Hypothesis property tests cover at least workload profile, prior fields, and selector budget invariants
pytest-regressions protects benchmark/report schema shape without snapshotting stochastic metrics
Rich summary script prints a readable gate/score summary without replacing JSON artifacts
Make targets exist and work
all tests pass
docs explain when to use each tool and when not to
```

The tooling should reduce time spent interpreting artifacts and reduce accidental regressions in gates/reports. It should not change the scientific objective or acceptance criteria.
