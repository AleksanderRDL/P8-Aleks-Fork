# Archived Analysis: Model Collapse And Random Baseline

This file used to contain an April 29, 2026 chat-derived investigation of model
collapse, the old `Random` baseline, and early query-coverage behavior.

That investigation is no longer current operating documentation. The relevant
findings have since been resolved or superseded:

- checkpoint selection defaults to held-out query F1 instead of raw training loss;
- coverage-targeted generation treats `n_queries` as a minimum when `max_queries`
  is higher;
- matched evaluation uses the eval-set Oracle;
- range evaluation is point-retention F1 rather than trajectory-ID answer F1;
- experiment entrypoints train pure query workloads, not mixed workload labels;
- the old `Random` baseline is not part of the main matched comparison tables.

Use `QDS/README.md` and the module READMEs under `QDS/src/` for current commands,
evaluation semantics, and benchmark workflow.
