# Model Training Progress Log

## 2026-05-13

- Current redesign reference is `Aleks-Sprint/range-training-redesign.md`.
- Reclassified the strong `range_aware` results as workload-aware diagnostic
  and teacher/upper-bound evidence, not final workload-blind success.
- Archived the old long progress log and workload-aware sweep reports under
  `Aleks-Sprint/archive/`.
- Removed stale local generated benchmark outputs and range-aware diagnostic
  caches from `QDS/artifacts/`; retained generic segmented trajectory caches
  that can still speed future work.
- Next real implementation step remains the workload-blind protocol/model:
  train from generated/historical workload supervision, compress without eval
  queries, freeze retained masks, then score held-out queries across the target
  coverage/compression grid.
- Renamed the active workload-aware profile and default artifact/cache family
  from `range_testing_baseline` to `range_workload_aware_diagnostic`. Renamed
  multi-budget output files from `range_objective_audit` to
  `range_compression_audit`, and renamed the learned residual-fill summary to
  `range_learned_fill_summary`.
- Replaced misleading checkpoint/runtime knobs that said `f1` with score-based
  names: `validation_score_every`, `checkpoint_full_score_every`, and
  `checkpoint_score_variant`.
- Renamed the temporal-base label knob to `temporal_residual_label_mode` so it
  describes the actual learned-fill residual behavior.
- Ran a broader naming cleanup beyond the workload-blind pivot. Replaced vague
  local names in workload generation, inference, query execution, simplification,
  GeoJSON export, trajectory caching, and split handling with names that expose
  the actual domain object: bounds, query type, anchor point, candidate indices,
  normalized points/queries, point score accumulators, keep counts, and trajectory
  counts.
- Cleaned stale and duplicated documentation: removed missing frontend references,
  updated artifact docs away from `range_testing_baseline`, condensed QDS module
  READMEs, and kept the workload-blind redesign rationale centralized in
  `range-training-redesign.md`.
- Removed stale compatibility shims for old F1/checkpoint naming, legacy
  `best_f1` artifacts, duplicate benchmark index fields, and old range-label
  cache-key migration. Config deserialization now rejects unknown keys instead
  of silently dropping stale artifact fields.
