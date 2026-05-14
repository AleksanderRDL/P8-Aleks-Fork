# Query-Driven Range_QDS Rework Progress

## Checkpoint 0 - Pre-rework cleanup

Status: completed

Scope:
- Fix stale legacy repository paths.
- Mark old RangeUseful/scalar-target profiles as legacy diagnostics.
- Add clean extension points for range_workload_v1, QueryUsefulV1, factorized targets, query-prior fields, and learned segment-budget selector.
- Do not implement the redesign yet.

Changes:
- Root `README.md`, root `Makefile`, `pyproject.toml`, `Range_QDS/README.md`, and `Range_QDS/Makefile` now use `Range_QDS/` paths.
- Copied the redesign guide to `Range_QDS/docs/query-driven-rework-guide.md` and made the in-project copy canonical in docs.
- Marked legacy benchmark profiles as not final-success eligible and added reserved `range_workload_v1_*` profile names that fail until implemented.
- Added report sections for `final_claim_summary`, `diagnostic_summary`, `legacy_range_useful_summary`, and `learning_causality_summary`.
- Marked `RangeUseful` as `RangeUsefulLegacy` documentation-wise and added the `evaluation/query_useful_v1.py` placeholder.
- Added query-driven scaffolding stubs for factorized targets, query-prior fields, factorized target diagnostics, workload profiles, `workload_blind_range_v2`, `learned_segment_budget_v1`, selector diagnostics, and legacy temporal-hybrid isolation.
- Added model and target metadata that blocks legacy scalar targets and historical-prior KNN paths from final success claims.
- Restored the archived legacy range-aware queue plan under `Range_QDS/benchmark_plans/archive/` so existing queue-plan tests validate against the current project path.
- Added pre-rework cleanup tests covering path hygiene, legacy profile guards, target-mode separation, historical-prior metadata, and report final-claim separation.

Tests run:
- `make test PYTHON="/home/aleks_dev/dev_projects/P8/.venv/bin/python"`: passed, 305 tests.
- `make lint PYTHON="/home/aleks_dev/dev_projects/P8/.venv/bin/python"`: passed.
- `make typecheck PYTHON="/home/aleks_dev/dev_projects/P8/.venv/bin/python"`: passed.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest tests/test_pre_rework_cleanup.py`: passed, 10 tests.
- Stale-path grep: no stale hyphenated project path, old change-directory command, or old slash-prefixed project path references. The broad directory-variable grep still matches the required root Makefile variable name.
- Misleading-final-success grep: clean after excluding the intended canonical guide warnings.

Known issues:
- `workload_profile_id` is not wired into query config yet. Runs without it should be treated as `legacy_generator` and not final-success eligible.
- The old temporal-hybrid selector code remains in `simplify_trajectories.py`; it is documented and stub-isolated but not physically split.

Next recommended checkpoint:
- Implement `range_workload_v1` and `QueryUsefulV1` before any new final benchmark work.
- Then implement factorized targets, train-derived query-prior fields, `workload_blind_range_v2`, and `learned_segment_budget_v1` in that order.
