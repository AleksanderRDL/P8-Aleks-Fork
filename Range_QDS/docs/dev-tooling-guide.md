# Range_QDS Developer Tooling And uv Environment Guide

This guide explains how to introduce and use five development tools in the Range_QDS query-driven rework workflow:

1. `uv`
2. `jq`
3. `hypothesis`
4. `pytest-regressions`
5. `rich`

The project now uses `uv` with `[dependency-groups].dev` in `pyproject.toml`. Active project commands should therefore use `uv sync --group dev` and `uv run --group dev -- ...`, not `pip`, `.venv/bin/python`, or `uv --extra dev`.

The goal is not to add tooling for its own sake. The goal is to make the redesign workflow safer, faster, reproducible, and easier to reason about.

The current project depends heavily on JSON artifacts, strict gates, workload signatures, predictor/selector ablations, benchmark summaries, and repeatable experiment commands. Tooling should help expose problems clearly, not hide them behind prettier logs or brittle snapshots.

---

## 1. Tooling principles

### Use tools to enforce invariants, not to decorate the project

The most valuable checks in this project are workflow and protocol invariants:

```text
eval queries must not affect compression
retained masks must freeze before eval scoring
final workloads must be healthy and signature-stable
prior fields must be train-derived only
zero-prior ablations must preserve extent and metadata
selector diagnostics must prove learned control is material
benchmark summaries must expose all required gates
experiment commands must be reproducible
```

### Use uv as the single Python execution layer

After adopting `uv`, do not mix command styles.

Use:

```bash
uv sync --group dev
uv run --group dev -- pytest ...
uv run --group dev -- ruff check ...
uv run --group dev -- pyright ...
uv run --group dev -- python -m experiments.run_ais_experiment ...
```

Do not use:

```bash
python -m pytest
python -m pip install -e ".[dev]"
pip install ...
.venv/bin/python -m pytest
../.venv/bin/python -m pytest
uv run --extra dev -- pytest
```

Because the project now uses `[dependency-groups].dev`, `--extra dev` is stale and should be removed from active docs, Makefiles, and scripts.

### Do not turn experiment metrics into brittle tests

Training runs are noisy. Do not snapshot full `example_run.json` metrics from trained models and expect exact equality. Use regression testing for stable schema/report shape, not stochastic learning results.

### Keep heavy tooling out of hot model paths

None of these tools should slow down model forward passes, target construction, selector execution, or benchmark inner loops.

### Prefer small, readable checks

A useful tool integration is one that makes failure clearer. Avoid giant snapshots, huge generated test cases, or excessive Rich formatting.

---

## 2. Recommended rollout order

Use this order:

```text
Phase 1: uv command migration
Phase 2: jq filters and inspection targets
Phase 3: hypothesis property tests
Phase 4: pytest-regressions snapshots
Phase 5: rich run summaries
```

Reasoning:

1. `uv` makes the environment and commands reproducible.
2. `jq` gives immediate artifact-inspection value.
3. `hypothesis` catches subtle generator/selector/prior edge cases.
4. `pytest-regressions` protects stable report and schema shape.
5. `rich` improves human readability once correctness checks are in place.

Do not start with `rich`. Pretty output is useful only after the artifact contracts and command layer are reliable.

---

## 3. uv

### Role

Use `uv` as the project's Python package manager, lockfile manager, environment manager, and Python command runner.

`uv` should replace:

```text
manual virtualenv setup
pip install -e ".[dev]"
hard-coded .venv/bin/python calls
bare python -m pytest / python -m ruff / python -m pyright commands
ad hoc dependency setup in docs
```

The current dependency convention is:

```toml
[dependency-groups]
dev = [
    ...
]
```

Therefore, active dev commands should include:

```bash
--group dev
```

Use explicit `--group dev` even if a local uv configuration happens to include the dev group by default. Explicit commands are easier for future agents and CI to copy correctly.

### Recommended repository files to update

The uv migration should update these active files:

```text
pyproject.toml              # already updated by user
uv.lock                     # generated and committed after uv lock/sync
README.md
Makefile
Range_QDS/README.md
Range_QDS/Makefile
Range_QDS/docs/query-driven-rework-guide.md
Range_QDS/docs/query-driven-rework-progress.md
Range_QDS/docs/dev-tooling-guide.md
Range_QDS/scripts/*.sh      # only if they hard-code PYTHON paths
CI config files             # if present
```

Do not rewrite historical command examples in archived logs unless they are active instructions. It is fine for historical progress logs to mention old `.venv` commands, but active guides, README setup sections, Make targets, and new checkpoint templates should use uv.

### Environment setup

Recommended first-time setup from repo root:

```bash
uv --version
uv python install 3.14
uv sync --group dev
uv lock --check
```

If the machine already has a compatible Python, `uv python install 3.14` may not be necessary, but the command documents the project expectation.

### Lockfile policy

Commit `uv.lock`.

Use this for local development:

```bash
uv sync --group dev
```

Use this for CI / reproducibility checks:

```bash
uv lock --check
uv sync --frozen --group dev
uv run --frozen --group dev -- pytest Range_QDS/tests
```

Do not use `uv lock --upgrade` casually. Upgrades should be intentional and checkpointed.

### Command policy

Use `uv run` for project Python commands.

Preferred:

```bash
uv run --group dev -- pytest Range_QDS/tests
uv run --group dev -- ruff check Range_QDS
uv run --group dev -- pyright Range_QDS/data Range_QDS/evaluation Range_QDS/experiments Range_QDS/models Range_QDS/queries Range_QDS/simplification Range_QDS/training Range_QDS/tests
uv run --group dev -- python -m experiments.run_ais_experiment ...
```

Avoid:

```bash
python -m pytest
../.venv/bin/python -m pytest
.venv/bin/python -m experiments.run_ais_experiment
python -m pip install -e ".[dev]"
pip install ...
uv run --extra dev -- pytest
```

### Root Makefile pattern

Recommended root `Makefile` structure:

```make
SHELL := /bin/bash

REPO_ROOT := $(abspath .)
UV ?= uv
UV_GROUP ?= dev
UV_GROUP_FLAGS ?= --group $(UV_GROUP)
UV_RUN := cd $(REPO_ROOT) && $(UV) run $(UV_GROUP_FLAGS) --
UV_RUN_FROZEN := cd $(REPO_ROOT) && $(UV) run --frozen $(UV_GROUP_FLAGS) --

.PHONY: help sync lock-check check-env lint test typecheck qds-lint qds-test qds-typecheck

sync:
	cd $(REPO_ROOT) && $(UV) sync $(UV_GROUP_FLAGS)

lock-check:
	cd $(REPO_ROOT) && $(UV) lock --check

check-env:
	cd $(REPO_ROOT) && $(UV) --version
	$(UV_RUN) python -V
	$(UV_RUN) python -m pip check

lint:
	$(UV_RUN) ruff check Range_QDS data ais_pipeline

test:
	$(UV_RUN) pytest Range_QDS/tests

typecheck:
	$(UV_RUN) pyright Range_QDS/data Range_QDS/evaluation Range_QDS/experiments Range_QDS/models Range_QDS/queries Range_QDS/simplification Range_QDS/training Range_QDS/tests

qds-lint:
	$(MAKE) -C Range_QDS lint UV="$(UV)" UV_GROUP="$(UV_GROUP)"

qds-test:
	$(MAKE) -C Range_QDS test UV="$(UV)" UV_GROUP="$(UV_GROUP)"

qds-typecheck:
	$(MAKE) -C Range_QDS typecheck UV="$(UV)" UV_GROUP="$(UV_GROUP)"
```

Adjust the root lint/typecheck paths to match the actual package layout. The important part is: do not call `.venv/bin/python`.

### Range_QDS/Makefile pattern

Because `Range_QDS/Makefile` lives below the root `pyproject.toml`, make it run uv from the root.

Recommended structure:

```make
SHELL := /bin/bash

REPO_ROOT := $(abspath ..)
UV ?= uv
UV_GROUP ?= dev
UV_GROUP_FLAGS ?= --group $(UV_GROUP)
UV_RUN := cd $(REPO_ROOT) && $(UV) run $(UV_GROUP_FLAGS) --
UV_RUN_FROZEN := cd $(REPO_ROOT) && $(UV) run --frozen $(UV_GROUP_FLAGS) --

TYPECHECK_PATHS ?= Range_QDS/data Range_QDS/evaluation Range_QDS/experiments Range_QDS/models Range_QDS/queries Range_QDS/simplification Range_QDS/training Range_QDS/scripts Range_QDS/tests
TEST_PATHS ?= Range_QDS/tests
LINT_PATHS ?= Range_QDS/data Range_QDS/evaluation Range_QDS/experiments Range_QDS/models Range_QDS/queries Range_QDS/simplification Range_QDS/training Range_QDS/scripts Range_QDS/tests

.PHONY: help sync lock-check check-env lint test typecheck smoke smoke-csv

sync:
	cd $(REPO_ROOT) && $(UV) sync $(UV_GROUP_FLAGS)

lock-check:
	cd $(REPO_ROOT) && $(UV) lock --check

check-env:
	cd $(REPO_ROOT) && $(UV) --version
	$(UV_RUN) python -V
	$(UV_RUN) python -m pip check
	$(UV_RUN) python -c 'from importlib.metadata import version; import numpy, pandas, pyarrow, pyright, pyspark, pytest, torch; print("torch", torch.__version__); print("numpy", numpy.__version__); print("pandas", pandas.__version__); print("pyarrow", pyarrow.__version__); print("pyspark", pyspark.__version__); print("python-dotenv", version("python-dotenv")); print("psycopg", version("psycopg")); print("pytest", pytest.__version__); print("pyright", pyright.__version__); print("ruff", version("ruff"))'

lint:
	$(UV_RUN) ruff check $(LINT_PATHS)

test:
	$(UV_RUN) pytest $(TEST_PATHS)

typecheck:
	$(UV_RUN) pyright $(TYPECHECK_PATHS)

smoke:
	$(UV_RUN) python -m experiments.run_ais_experiment \
		--n_ships 4 \
		--n_points 24 \
		--n_queries 4 \
		--epochs 1 \
		--workload range \
		--compression_ratio 0.5 \
		--results_dir "Range_QDS/artifacts/results/smoke_synthetic"
```

Note the path difference: because the command runs from repo root, paths should normally be prefixed with `Range_QDS/`.

### Running experiments

Preferred from repo root:

```bash
uv run --group dev -- python -m experiments.run_ais_experiment   --results_dir Range_QDS/artifacts/results/query_driven_probe   --n_ships 64   --n_points 256   --synthetic_route_families 4   --n_queries 48   --query_coverage 0.10   --max_queries 256   --workload_profile_id range_workload_v1   --coverage_calibration_mode profile_sampled_query_count   --workload_stability_gate_mode final   --model_type workload_blind_range_v2   --range_training_target_mode query_useful_v1_factorized   --selector_type learned_segment_budget_v1   --checkpoint_score_variant query_useful_v1   --checkpoint_selection_metric uniform_gap   --compression_ratio 0.05   --final_metrics_mode diagnostic
```

If running from `Range_QDS/`, either use the Makefile target or explicitly run from the root:

```bash
cd ..
uv run --group dev -- python -m experiments.run_ais_experiment ...
```

Avoid:

```bash
cd Range_QDS
../.venv/bin/python -m experiments.run_ais_experiment ...
```

### Shell scripts and tmux benchmark scripts

Many scripts historically receive a `PYTHON` variable. Moving to uv means you should avoid treating the Python executable as a simple path.

Recommended approach:

```text
Use UV and UV_GROUP variables in shell scripts.
Build commands as arrays where possible.
Call: uv run --group dev -- python -m ...
```

Example shell style:

```bash
UV="${UV:-uv}"
UV_GROUP="${UV_GROUP:-dev}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd -P)}"

cd "$REPO_ROOT"
"$UV" run --group "$UV_GROUP" -- python -m experiments.run_ais_experiment "$@"
```

If a script must accept a command override, use `PYTHON_CMD` instead of `PYTHON`:

```bash
PYTHON_CMD="${PYTHON_CMD:-uv run --group dev -- python}"
```

Then be careful with shell word splitting. Command arrays are safer.

### README command replacements

Replace active setup instructions like this:

```text
Old:
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

New:
uv python install 3.14
uv sync --group dev
```

Replace active test instructions:

```text
Old:
cd Range_QDS
../.venv/bin/python -m pytest tests
../.venv/bin/ruff check ...

New:
uv run --group dev -- pytest Range_QDS/tests
uv run --group dev -- ruff check Range_QDS
```

Replace active experiment commands:

```text
Old:
../.venv/bin/python -m experiments.run_ais_experiment ...

New:
uv run --group dev -- python -m experiments.run_ais_experiment ...
```

### Grep checklist

Before considering the uv migration complete, run:

```bash
grep -R "\.venv/bin/python\|python -m pip install\|pip install -e\|pip install \|--extra dev\|\[project.optional-dependencies\]" -n   README.md Makefile Range_QDS   --exclude-dir=.venv   --exclude-dir=artifacts   --exclude-dir=.git || true
```

Review every hit.

Allowed hits:

```text
archived historical progress logs
explicit migration notes explaining old commands to avoid
```

Not allowed in active docs/scripts:

```text
new setup instructions
new Makefile commands
new benchmark commands
new progress-log checkpoint templates
```

### uv and jq

`jq` is not a Python package. Keep it as a system tool.

Good:

```bash
jq -f Range_QDS/scripts/jq/run_summary.jq Range_QDS/artifacts/results/.../example_run.json
```

No need:

```bash
uv run --group dev -- jq ...
```

### uv and Python dev tools

These should run through uv:

```bash
uv run --group dev -- pytest ...
uv run --group dev -- ruff check ...
uv run --group dev -- pyright ...
uv run --group dev -- python Range_QDS/scripts/summarize_run.py ...
```

Do not use `uvx` for tools that are part of the project dev environment. `uvx` is fine for one-off external tools, but for this project you want locked versions from `uv.lock`.

---

## 4. jq

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
Did learning causality fail because of shuffled scores, untrained model, no-prior ablation, or fairness preallocation?
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

### Suggested Makefile targets

These targets may live in `Range_QDS/Makefile`.

```make
RUN ?= Range_QDS/artifacts/results/latest/example_run.json

inspect-run:
	jq -f Range_QDS/scripts/jq/run_summary.jq $(RUN)

inspect-gates:
	jq -f Range_QDS/scripts/jq/gates.jq $(RUN)

inspect-scores:
	jq -f Range_QDS/scripts/jq/scores.jq $(RUN)

inspect-causality:
	jq -f Range_QDS/scripts/jq/causality.jq $(RUN)

inspect-generator:
	jq -f Range_QDS/scripts/jq/generator_health.jq $(RUN)

inspect-predictability:
	jq -f Range_QDS/scripts/jq/predictability.jq $(RUN)
```

If the Makefile runs from inside `Range_QDS/`, drop the `Range_QDS/` prefixes.

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

Strict single-cell gate check:

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
' Range_QDS/artifacts/results/.../example_run.json
```

Score check:

```bash
jq '{
  mlqds: .matched.MLQDS.query_useful_v1_score,
  uniform: .matched.uniform.query_useful_v1_score,
  dp: .matched.DouglasPeucker.query_useful_v1_score,
  beats_uniform: (.matched.MLQDS.query_useful_v1_score > .matched.uniform.query_useful_v1_score),
  beats_dp: (.matched.MLQDS.query_useful_v1_score > .matched.DouglasPeucker.query_useful_v1_score)
}' Range_QDS/artifacts/results/.../example_run.json
```

### What not to do with jq

Do not use `jq` as a replacement for Python tests. It is an inspection and lightweight manual assertion tool.

Do not bury complex acceptance logic in shell scripts. Acceptance logic belongs in Python code and tests; `jq` should expose fields clearly.

---

## 5. hypothesis

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

### pytest marker

Add a marker to `pyproject.toml` if not already present:

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
import math
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
        min(points_per_trajectory, max(2, int(math.ceil(compression_ratio * points_per_trajectory))))
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

## 6. pytest-regressions

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
        command=["uv", "run", "--group", "dev", "--", "python", "-m", "experiments.run_ais_experiment"],
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

## 7. rich

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

From repo root:

```bash
uv run --group dev -- python Range_QDS/scripts/summarize_run.py Range_QDS/artifacts/results/.../example_run.json
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

### What not to do with Rich

Do not replace JSON artifacts with Rich text output.

Do not remove existing machine-readable reports.

Do not make tests depend on ANSI-colored output.

Do not put `rich` imports in model files, selector files, metric functions, or training inner loops.

---

## 8. Combined workflow after introducing tools

### After cloning or changing dependencies

```bash
uv sync --group dev
uv lock --check
```

### Before code changes

```bash
uv run --group dev -- pytest Range_QDS/tests/test_query_driven_rework.py -q
uv run --group dev -- pytest Range_QDS/tests/test_benchmark_runner.py -q
```

### After code changes

```bash
git diff --check
uv run --group dev -- ruff check Range_QDS
uv run --group dev -- pyright Range_QDS/data Range_QDS/evaluation Range_QDS/experiments Range_QDS/models Range_QDS/queries Range_QDS/simplification Range_QDS/training Range_QDS/tests
uv run --group dev -- pytest Range_QDS/tests -q
```

### After an experiment run

Use:

```bash
make inspect-run RUN=Range_QDS/artifacts/results/.../example_run.json
make inspect-gates RUN=Range_QDS/artifacts/results/.../example_run.json
make inspect-causality RUN=Range_QDS/artifacts/results/.../example_run.json
```

Then optionally:

```bash
uv run --group dev -- python Range_QDS/scripts/summarize_run.py Range_QDS/artifacts/results/.../example_run.json
```

### After changing generator/profile code

Run:

```bash
uv run --group dev -- pytest Range_QDS/tests/test_query_driven_rework.py Range_QDS/tests/test_query_coverage_generation.py
uv run --group dev -- pytest Range_QDS/tests/property/test_workload_profile_properties.py
```

Then inspect a generated artifact with:

```bash
make inspect-generator RUN=Range_QDS/artifacts/results/.../example_run.json
```

### After changing query-prior fields

Run:

```bash
uv run --group dev -- pytest Range_QDS/tests/test_query_driven_rework.py
uv run --group dev -- pytest Range_QDS/tests/property/test_query_prior_field_properties.py
make inspect-predictability RUN=Range_QDS/artifacts/results/.../example_run.json
```

### After changing selector logic

Run:

```bash
uv run --group dev -- pytest Range_QDS/tests/test_query_driven_rework.py
uv run --group dev -- pytest Range_QDS/tests/property/test_learned_segment_selector_properties.py
make inspect-causality RUN=Range_QDS/artifacts/results/.../example_run.json
```

### After changing benchmark report fields

Run:

```bash
uv run --group dev -- pytest Range_QDS/tests/test_benchmark_runner.py
uv run --group dev -- pytest Range_QDS/tests/regression/test_benchmark_report_regression.py
```

---

## 9. Suggested Makefile additions

Add only if these fit the existing Makefile style.

```make
RUN ?= Range_QDS/artifacts/results/latest/example_run.json
UV ?= uv
UV_GROUP ?= dev
REPO_ROOT ?= $(abspath .)
UV_GROUP_FLAGS ?= --group $(UV_GROUP)
UV_RUN := cd $(REPO_ROOT) && $(UV) run $(UV_GROUP_FLAGS) --

sync:
	cd $(REPO_ROOT) && $(UV) sync $(UV_GROUP_FLAGS)

lock-check:
	cd $(REPO_ROOT) && $(UV) lock --check

inspect-run:
	jq -f Range_QDS/scripts/jq/run_summary.jq $(RUN)

inspect-gates:
	jq -f Range_QDS/scripts/jq/gates.jq $(RUN)

inspect-scores:
	jq -f Range_QDS/scripts/jq/scores.jq $(RUN)

inspect-causality:
	jq -f Range_QDS/scripts/jq/causality.jq $(RUN)

inspect-generator:
	jq -f Range_QDS/scripts/jq/generator_health.jq $(RUN)

inspect-predictability:
	jq -f Range_QDS/scripts/jq/predictability.jq $(RUN)

summarize-run:
	$(UV_RUN) python Range_QDS/scripts/summarize_run.py $(RUN)

test-property:
	$(UV_RUN) pytest Range_QDS/tests/property

test-regression:
	$(UV_RUN) pytest Range_QDS/tests/regression
```

If `tests/property` or `tests/regression` do not exist yet, add the targets only after creating the directories.

---

## 10. Suggested implementation checkpoint

Use one focused tooling checkpoint.

### Scope

```text
Migrate active commands to uv using dependency group syntax.
Add jq filters and Make targets.
Add initial Hypothesis property tests for workload profile, query-prior fields, and learned segment selector.
Add normalized pytest-regressions snapshots for benchmark final-grid summary and benchmark row field set.
Add optional Rich run-summary script.
```

### Expected files

```text
README.md
Makefile
Range_QDS/README.md
Range_QDS/Makefile
Range_QDS/docs/dev-tooling-guide.md
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
```

### Tests to run

```bash
uv sync --group dev
uv lock --check
git diff --check

uv run --group dev -- ruff check Range_QDS
uv run --group dev -- pyright Range_QDS/data Range_QDS/evaluation Range_QDS/experiments Range_QDS/models Range_QDS/queries Range_QDS/simplification Range_QDS/training Range_QDS/tests

uv run --group dev -- pytest Range_QDS/tests/property -q
uv run --group dev -- pytest Range_QDS/tests/regression -q
uv run --group dev -- pytest Range_QDS/tests/test_query_driven_rework.py -q
uv run --group dev -- pytest Range_QDS/tests/test_benchmark_runner.py -q
uv run --group dev -- pytest Range_QDS/tests -q
```

### Stop condition

Stop after tooling is integrated and tests pass. Do not mix this checkpoint with model/selector/generator research changes.

---

## 11. Risks and mitigations

### Risk: uv command drift

If some docs/scripts use uv and others use `.venv/bin/python`, future agents will copy stale commands.

Mitigation:

```text
Grep for old commands.
Update active README/Makefile/docs/scripts.
Leave old commands only in clearly historical archived logs.
```

### Risk: stale `--extra dev` commands remain after dependency-group migration

The project uses `[dependency-groups].dev`, not `[project.optional-dependencies].dev`.

Mitigation:

```text
Grep for --extra dev.
Replace it with --group dev in active docs and commands.
```

### Risk: uv lockfile is not committed

Without `uv.lock`, the project does not gain reproducibility.

Mitigation:

```text
Commit uv.lock.
Run uv lock --check in CI.
Use uv sync --frozen --group dev in CI.
```

### Risk: dev dependency group is not included

If commands omit `--group dev`, dev tools may not be available depending on uv settings and project configuration.

Mitigation:

```text
Use uv sync --group dev.
Use uv run --group dev -- ...
```

### Risk: jq filters become hidden acceptance logic

Mitigation:

```text
Keep acceptance in Python code/tests.
Use jq only for inspection and lightweight manual checks.
Document filters as diagnostic helpers.
```

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

### Risk: Rich output replaces machine-readable artifacts

Mitigation:

```text
Keep JSON artifacts authoritative.
Use Rich only for display.
Do not parse Rich output in tests.
```

### Risk: too much tooling distracts from research blockers

Mitigation:

```text
Implement tooling in one checkpoint.
Then return to the active scientific blocker.
Do not keep adding tools after the core workflow is improved.
```

---

## 12. Definition of done

This tooling introduction is done when:

```text
uv is the documented and Makefile-backed Python execution layer
active Makefiles no longer hard-code .venv/bin/python
active README/docs no longer instruct pip install -e ".[dev]"
active commands use --group dev, not --extra dev
uv.lock is created and committed
jq filters make one-run gate diagnosis easy
Hypothesis property tests cover at least workload profile, prior fields, and selector budget invariants
pytest-regressions protects benchmark/report schema shape without snapshotting stochastic metrics
Rich summary script prints a readable gate/score summary without replacing JSON artifacts
Make targets exist and work
all tests pass through uv run
docs explain when to use each tool and when not to
```

The tooling should reduce time spent interpreting artifacts and reduce accidental regressions in gates/reports. It should not change the scientific objective or acceptance criteria.
