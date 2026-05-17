# QDS Code Layout

This is the top-down map for the active `Range_QDS` codebase. It is not an
exhaustive file index; each package README carries local details.

## Main Flow

```text
data -> queries -> training -> simplification -> evaluation -> experiments/reports
```

Main entry point:

```bash
uv run --group dev -- python -m experiments.run_ais_experiment
```

Operational tooling reference: `docs/dev-tooling-guide.md`.

## Package Responsibilities

| Path | Owns | Should not own |
| --- | --- | --- |
| `data/` | AIS loading, segmentation, source/day combination, trajectory caches, flattened boundaries. | Query workload policy, model behavior, benchmark gates. |
| `queries/` | Typed query data, range geometry, workload profiles, query generation, query execution, workload diagnostics. | Training labels, model scoring, retained-mask selection. |
| `models/` | PyTorch model definitions and model-local utilities. | Training loops, checkpoint policy, experiment metrics. |
| `training/` | Feature builders, target builders, priors, losses, batching, checkpoint persistence, inference helpers. | Experiment orchestration, benchmark reporting, final-claim gates. |
| `simplification/` | Query-free score-to-mask selectors and selector diagnostics. | Query generation, model training, benchmark reporting. |
| `evaluation/` | Metrics, baseline methods, query caches, range/query-useful scoring, printable evaluation tables. | Training target construction or experiment command assembly. |
| `experiments/` | Config, CLI parsing, workload assembly, pipeline orchestration, artifact writing, benchmark profiles, reports, runtime helpers. | Low-level model/selector/query primitives that need to be reused without orchestration. |
| `scripts/` | Small operational tools over existing artifacts or profiles. | Scientific logic not already owned by packages above. |
| `tests/` | Guardrails, regression tests, property tests, and focused unit/integration coverage. | Production helpers used only to make tests pass. |

Current layering exception:

- `training.train_model` imports `experiments.experiment_config.ModelConfig`
  and `experiments.torch_runtime`. This works, but it makes `training/`
  depend upward on `experiments/`. If this becomes painful, move shared config
  dataclasses/runtime controls to a neutral package before adding more imports
  in that direction.

## Current Pressure Points

These are the files that most weaken top-down reasoning today. Line counts are
approximate and should be treated as refactor signals, not automatic defects.

| File | Current issue | Recommended split |
| --- | --- | --- |
| `experiments/experiment_pipeline.py` (~5k lines) | Mixes orchestration, phase timing, gates, causality ablations, segment audits, support checks, artifact assembly, and run output. | Extract in this order: `experiments/gates.py`, `experiments/causality.py`, `experiments/segment_audits.py`; leave `run_experiment_pipeline` as the orchestrator. |
| `training/training_targets.py` (~3k lines) | Multiple generations of range target builders and aggregation paths live together. | Split by target family: scalar/legacy diagnostics, set-utility targets, local-swap targets, aggregation/balancing. Keep public wrappers until call sites move. |
| `experiments/benchmark_report.py` (~2k lines) | Final-grid logic, child-run row flattening, audit extraction, and table formatting are interleaved. | Separate `final_grid_summary`, `row_fields`, and `table_formatting`. |
| `simplification/learned_segment_budget.py` (~1.5k lines) | Budget allocation, length repair, trace payloads, and diagnostics are tightly packed. | Split allocation, length repair, and diagnostics/trace once selector behavior is stable. |
| `queries/query_generator.py` (~1.3k lines) | Anchor weighting, profile planning, acceptance filtering, signature generation, and workload assembly share one module. | Split range anchor sampling/profile planning from acceptance/signature diagnostics. |

## Refactor Rules

- Preserve the public experiment commands and artifact field names unless the
  checkpoint explicitly says it is changing them.
- Move behavior only with focused tests around the moved boundary. Do not do
  broad file splits during scientific probe checkpoints.
- Prefer extraction of pure helpers first: gates, row-field builders, signature
  builders, allocation diagnostics.
- Avoid permanent compatibility shims. If a temporary facade is needed during a
  split, mark its removal checkpoint in the progress log.
- Keep final-claim gates close to their tests. The gate code should eventually
  be importable without importing the full experiment pipeline.
