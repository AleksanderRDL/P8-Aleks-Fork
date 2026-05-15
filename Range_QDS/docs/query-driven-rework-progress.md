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

## Checkpoint 1 - Query-driven workload-blind v2 path

Status: implemented as a candidate path, not a final benchmark claim

Scope:
- Replace the reserved stubs with a runnable query-driven, workload-blind Range-QDS path.
- Keep legacy `RangeUseful` available only as a diagnostic.
- Train from train-only future-query workload samples and freeze retained masks before eval query scoring.
- Add enough diagnostics to detect leakage, stale scalar-target metadata, temporal scaffolding, and weak learning.

Changes:
- Implemented `range_workload_v1` with profile metadata, weighted anchor families, weighted footprint families, acceptance defaults, workload signatures, and cache-key separation by profile.
- Added `QueryUsefulV1` scoring to method evaluation and result dumps. It reweights range-query audit components toward query point mass, query-local behavior, ship coverage, boundary evidence, and small global sanity guardrails.
- Added factorized `QueryUsefulV1` target construction with heads for query-hit probability, conditional behavior utility, boundary/event utility, marginal replacement gain, and segment-budget target.
- Added train-only query-prior fields sampled into `workload_blind_range_v2` point features. Metadata records `built_from_split=train_only`, train workload seed, and explicit `contains_eval_queries=false` / `contains_validation_queries=false`.
- Implemented trainable `WorkloadBlindRangeV2Model` with query-free inference and factorized auxiliary heads.
- Wired factorized loss, checkpoint scoring on `query_useful_v1`, checkpoint persistence of query-prior fields, inference feature reconstruction, and validation scoring.
- Implemented `learned_segment_budget_v1` and selected it through `selector_type`. Diagnostics report learned budget share and explicitly reject a fixed 85% temporal scaffold.
- Added the final-candidate benchmark profile using `range_workload_v1`, `query_useful_v1_factorized`, `workload_blind_range_v2`, and `learned_segment_budget_v1`.
- Updated workload-blind protocol dumps with the required no-leakage flags:
  `masks_frozen_before_eval_query_scoring=true`, `eval_queries_seen_by_model=false`,
  `eval_queries_seen_by_feature_builder=false`, `eval_queries_seen_by_selector=false`,
  `checkpoint_selected_on_eval_queries=false`, and
  `query_conditioned_range_aware_used_for_product_acceptance=false`.
- Added budget-level learning-causality fields for learned-controlled retained slots, learned-slot fraction, skeleton cap, and explicit `learning_causality_ablation_status=not_run`.
- Added focused query-driven tests and updated benchmark/model tests for the new public surface.

Smoke result:
- Command:
  `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m experiments.run_ais_experiment --n_ships 4 --n_points 32 --n_queries 4 --epochs 1 --workload range --model_type workload_blind_range_v2 --range_training_target_mode query_useful_v1_factorized --workload_profile_id range_workload_v1 --selector_type learned_segment_budget_v1 --checkpoint_score_variant query_useful_v1 --checkpoint_selection_metric loss --compression_ratio 0.20 --range_max_point_hit_fraction 1.0 --range_duplicate_iou_threshold 1.0 --range_acceptance_max_attempts 200 --final_metrics_mode core --results_dir artifacts/results/query_driven_v2_smoke`
- Artifact: `artifacts/results/query_driven_v2_smoke/example_run.json`.
- The run trains and evaluates end to end. `training_target_diagnostics.target_family=QueryUsefulV1Factorized`; no stale legacy target-family claim remains.
- Learning diagnostics show the model is not dead: train target Kendall tau `+0.0748`, matched MLQDS-vs-uniform target recall delta `+0.5419`, and low-budget delta `+0.1075`.
- Learning-causality summary records planned learned-controlled retained slots `6/7` at the smoke eval compression ratio, but ablation deltas are explicitly not run.
- Eval `QueryUsefulV1` on this tiny one-epoch smoke is weak: MLQDS `0.1956`, uniform `0.3748`, Douglas-Peucker `0.2822`. This is not acceptable as a final result. It only proves the candidate path runs, learns some train-target signal, and obeys the workload-blind protocol.

Tests run:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest tests`: passed, 311 tests, 1 PyTorch nested-tensor warning.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m ruff check data evaluation experiments models queries simplification training scripts tests`: passed.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pyright data evaluation experiments models queries simplification training tests`: passed.
- Focused smoke command above: passed.

Known issues and gates:
- No full benchmark grid has been run. The final-candidate profile deliberately reports `candidate_requires_full_grid_and_causality_ablations`.
- Required causality ablations are not implemented yet: shuffled scores, untrained scores, no query-prior field, and no behavior head.
- The smoke loses badly to uniform on eval `QueryUsefulV1`. Tiny data and one epoch explain some of that, but it is still a real warning. Do not treat the implementation as accepted until full-grid results beat baselines under stable train/eval workload signatures.
- `QueryUsefulV1` currently derives many components from the existing range audit rather than from a fully independent query-local evaluator. This is acceptable for a v1 metric bridge, but it should be tightened before making strong claims.
- `learned_segment_budget_v1` allocates segment slots from final point scores rather than directly consuming a separately exported segment-budget head at inference. That is simpler, but it leaves useful supervision on the table.
- `range_workload_v1` is still a heuristic workload model. Without real product query logs, the distribution is defensible but not empirically validated.

## Checkpoint 2 - Gate diagnostics and partial causality ablations

Status: implemented diagnostics; current smoke remains blocked

Scope:
- Add explicit gates required by the guide so a candidate run cannot be mistaken for final acceptance.
- Add a predictability audit for train-derived query-prior fields.
- Add workload-signature gate reporting.
- Add partial learning-causality ablations that freeze diagnostic masks before eval query scoring.

Changes:
- Added `predictability_audit` to result dumps. It compares train-query-prior-only scores against held-out eval `QueryUsefulV1` targets after retained masks are frozen. It reports Spearman, sampled Kendall tau, AUC, PR-AUC, PR-AUC lift over base rate, NDCG at 1/2/5/10%, and lift at 1/2/5/10%.
- Added the guide’s minimum predictability gate thresholds to the artifact:
  lift at 1% `>=1.10`, lift at 2% `>=1.15`, lift at 5% `>=1.20`, Spearman `>=0.15`, and PR-AUC lift over base rate `>=1.25`.
- Added `workload_signature_gate` under `workload_distribution_comparison`. It checks anchor-family L1 distance, footprint-family L1 distance, point/ship hit-distribution KS distance, near-duplicate rate, and broad-query rate.
- Persisted raw per-query point/ship/trajectory hit counts in `workload_signature`; older artifacts still fall back to quantile-distance proxies.
- Changed `final_claim_summary.final_success_allowed` so a single final-candidate run is blocked unless required gates pass. It now reports concrete `blocking_gates`.
- Added partial learning-causality ablations:
  `MLQDS_shuffled_scores`, `MLQDS_untrained_model`, `MLQDS_prior_field_only_score`,
  `MLQDS_shuffled_prior_fields`, and `MLQDS_without_query_prior_features`.
- Added `learning_causality_ablations` result payloads and corresponding deltas in `learning_causality_summary`.
- Added benchmark-row fields for `QueryUsefulV1`, predictability gate metrics, workload-signature gate metrics, final-claim blocking gates, and partial causality-ablation deltas. Grid reports can now surface the actual acceptance evidence instead of only legacy `RangeUseful`.

Smoke result:
- Same smoke command and artifact as checkpoint 1:
  `artifacts/results/query_driven_v2_smoke/example_run.json`.
- `final_claim_summary.status=candidate_blocked_by_required_gates`.
- Blocking gates in the smoke: `predictability_gate`, `workload_signature_gate`, `full_coverage_compression_grid`, `learning_causality_ablations`, and `global_sanity_gates`.
- Predictability gate fails:
  Spearman `-0.0297`, lift at 1% `0.8250`, lift at 2% `0.8250`, lift at 5% `0.5563`, PR-AUC lift over base rate `0.8857`.
- Workload-signature gate fails on the tiny smoke because train/eval sampled query signatures drift heavily:
  anchor-family L1 `0.5`, footprint-family L1 `1.0`, point-hit KS `0.5`.
- Causality ablations are mixed and still not acceptable:
  shuffled-score QueryUsefulV1 `0.6221` beats trained MLQDS `0.5628`;
  untrained model is worse (`0.1488`);
  prior-field-only score is worse (`0.4640`) but still competitive;
  shuffled prior fields and removing query-prior features have no effect in the smoke (`delta 0.0`).
- This is exactly why the gate layer matters. The path runs, but the current tiny trained model has not proved learned query-driven behavior.

Tests run:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest tests`: passed, 313 tests, 1 PyTorch nested-tensor warning.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m ruff check data evaluation experiments models queries simplification training scripts tests`: passed.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pyright data evaluation experiments models queries simplification training tests`: passed.
- Focused smoke command: passed.

Known issues and gates:
- Full 4x7 coverage/compression grid has not been run.
- Full causality report is still incomplete. Missing: behavior-head removal and segment-budget-head removal. Also bad: shuffled scores still beat trained scores, and query-prior feature ablations have no effect in the smoke.
- The current smoke fails the guide’s stop/continue rules. Do not continue architecture tuning as if the model is accepted; fix predictability and workload signature stability first.
- Older artifacts without raw per-query hit-count lists still use proxy checks.

## Checkpoint 3 - Selector point attribution and fallback guard fix

Status: implemented attribution; current smoke remains blocked

Scope:
- Replace budget-only learned-slot accounting with actual per-retained-point attribution.
- Detect whether retained points came from skeleton, learned segment allocation, or fallback fill.
- Remove a selector guard bug that let `max_budget_share_per_ship=0.20` force fallback dominance on small eval splits.

Changes:
- Added `simplify_with_learned_segment_budget_v1_with_trace`.
- Persisted `selector_trace_diagnostics.eval_primary` with retained-mask agreement, skeleton count, actual learned count, fallback count, trajectory learned-decision counts, and segment-budget entropy.
- Updated `learning_causality_summary` to prefer actual point attribution over planned budget rows. It now reports planned versus actual learned slot fractions.
- Fixed the trajectory guard in `learned_segment_budget_v1`: the per-ship cap now respects at least an equal-share cap, so a one-trajectory eval split can spend the learned budget through learned segment allocation instead of fallback fill.
- Replaced the stale `selector_diagnostics.py` placeholder with a small shared diagnostic vocabulary.

Smoke result:
- Same focused smoke artifact:
  `artifacts/results/query_driven_v2_smoke/example_run.json`.
- MLQDS now beats uniform and Douglas-Peucker on smoke `QueryUsefulV1`: MLQDS `0.5628`, uniform `0.3748`, Douglas-Peucker `0.2822`.
- Actual selector attribution: 7 retained points, 1 skeleton point, 6 learned-controlled points, 0 fallback points. Trace mask matches the frozen primary mask.
- This improvement does not make the candidate accepted. Final claim is still blocked by `predictability_gate`, `workload_signature_gate`, `full_coverage_compression_grid`, `learning_causality_ablations`, and `global_sanity_gates`.
- Predictability gate still fails: Spearman `-0.0297`, lift at 1% `0.8250`, lift at 2% `0.8250`, lift at 5% `0.5563`, PR-AUC lift over base rate `0.8857`.
- Workload-signature gate still fails on the tiny smoke: anchor-family L1 `0.5`, footprint-family L1 `1.0`, point-hit KS `0.5`.
- Causality still fails because shuffled scores beat trained scores (`0.6221` versus `0.5628`) and query-prior feature ablations have no effect.

Tests run:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest tests/test_query_driven_rework.py tests/test_benchmark_runner.py`: passed, 44 tests.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m ruff check simplification experiments tests/test_query_driven_rework.py tests/test_benchmark_runner.py`: passed.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pyright simplification/learned_segment_budget.py experiments/experiment_pipeline.py experiments/benchmark_report.py tests/test_query_driven_rework.py tests/test_benchmark_runner.py`: passed.
- Focused smoke command: passed.

## Checkpoint 4 - Segment-budget head selection and complete smoke ablation set

Status: implemented smoke-level causality coverage; gates still fail

Scope:
- Make `learned_segment_budget_v1` allocate segment budget from the factorized `segment_budget_target` head instead of reusing final point scores for segment ranking.
- Keep final learned point scores for within-segment point selection.
- Add the remaining required frozen-mask causality ablations for the smoke path.

Changes:
- Added `windowed_predict_with_heads` so workload-blind factorized models can return averaged per-point head logits during inference.
- `MLQDSMethod` now caches factorized head logits and passes `sigmoid(segment_budget_target)` to `learned_segment_budget_v1` as segment-allocation scores.
- Added `WorkloadBlindRangeV2Model.final_logit_from_head_logits`, allowing diagnostic neutralization of a factorized head without feeding eval queries into inference.
- Added `MLQDS_without_segment_budget_head` and `MLQDS_without_behavior_utility_head` frozen-mask ablations.
- Added explicit `learning_causality_gate_pass`, `learning_causality_failed_checks`, and `causality_ablation_missing` fields. The ablation set can now be complete while still failing the gate.
- Preserved the existing `evaluation.baselines.windowed_predict` monkeypatch seam for non-factorized tests and callers; the head-logit inference path is used only when the model exposes `forward_with_heads`.

Smoke result:
- Same focused smoke artifact:
  `artifacts/results/query_driven_v2_smoke/example_run.json`.
- `learning_causality_ablation_status=complete`, `causality_ablation_missing=[]`, but `learning_causality_gate_pass=false`.
- Failed causality checks:
  `shuffled_scores_should_lose`, `shuffled_prior_fields_should_lose`,
  `without_query_prior_features_should_lose`, and `without_segment_budget_head_should_lose`.
- Deltas on smoke `QueryUsefulV1`:
  shuffled scores `-0.0593`, untrained model `+0.4140`, shuffled prior fields `0.0`,
  no query-prior features `0.0`, no behavior head `+0.0470`, no segment-budget head `0.0`,
  prior-field-only `+0.0989`.
- Selector trace confirms `segment_score_source=segment_budget_head_mean`, 6/7 learned-controlled retained points, and 0 fallback points.
- The zero segment-head delta is not proof that the head is useless; this tiny smoke has one eval trajectory and one segment, so the ablation cannot exercise inter-segment allocation. The full grid is still required.

Tests run:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest tests/test_query_driven_rework.py tests/test_benchmark_runner.py tests/test_model_features.py`: passed, 65 tests, 1 PyTorch nested-tensor warning.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m ruff check evaluation/baselines.py training/inference.py models/workload_blind_range_v2.py simplification/learned_segment_budget.py experiments/experiment_pipeline.py experiments/benchmark_report.py tests/test_query_driven_rework.py tests/test_benchmark_runner.py`: passed.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pyright evaluation/baselines.py training/inference.py models/workload_blind_range_v2.py simplification/learned_segment_budget.py experiments/experiment_pipeline.py experiments/benchmark_report.py tests/test_query_driven_rework.py tests/test_benchmark_runner.py`: passed.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest tests`: passed, 315 tests, 1 PyTorch nested-tensor warning.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m ruff check data evaluation experiments models queries simplification training scripts tests`: passed.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pyright data evaluation experiments models queries simplification training tests`: passed.
- `git diff --check`: passed.
- Focused smoke command: passed.

## Checkpoint 5 - Workload stability gate and final-profile replicate defaults

Status: implemented guardrails; current smoke remains blocked

Scope:
- Prevent final-candidate artifacts from passing based on tiny fixed-count query samples.
- Require final-candidate workload generation to look like the guide's calibrated workload protocol, not just `workload_profile_id=range_workload_v1`.
- Make the final benchmark profile train from multiple held-out workload samples.

Changes:
- Added `workload_stability_gate` to run artifacts.
- Final-candidate runs now block on `workload_stability_gate` unless:
  coverage target is one of `[0.05, 0.10, 0.15, 0.30]`;
  query generation mode is `target_coverage`;
  coverage guard is enabled;
  configured overshoot is no looser than the guide tolerance;
  each workload has at least 8 queries;
  train labels use at least 4 train workload replicates;
  each workload reaches target coverage without exhausting generation.
- Added benchmark report fields for workload-stability gate pass/fail, failed checks, train replicate count, and configured target coverage.
- Updated the `range_workload_v1_workload_blind_v2` benchmark profile to pass `--range_train_workload_replicates 4`.
- Tightened the final profile's 10% coverage overshoot to `0.0075`, matching the guide.

Smoke result:
- Fixed-count smoke artifact:
  `artifacts/results/query_driven_v2_smoke/example_run.json`.
  It now blocks on `workload_stability_gate` with failures for no grid coverage target, one train replicate, no coverage guard, fixed-count generation, and too few queries.
- Coverage-target replicate smoke artifact:
  `artifacts/results/query_driven_v2_stability_smoke/example_run.json`.
  It uses four train workload replicates and target-coverage generation, but blocks on `coverage_overshoot_tolerance_too_loose` because the smoke intentionally used `--range_max_coverage_overshoot 0.50`.
- This is intentional. A huge overshoot makes "target coverage" meaningless; it should not be accepted as final workload evidence.

Tests run:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest tests/test_query_driven_rework.py tests/test_benchmark_runner.py tests/test_pre_rework_cleanup.py`: passed, 56 tests.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m ruff check experiments/experiment_pipeline.py experiments/benchmark_profiles.py experiments/benchmark_report.py tests/test_query_driven_rework.py tests/test_benchmark_runner.py tests/test_pre_rework_cleanup.py`: passed.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pyright experiments/experiment_pipeline.py experiments/benchmark_profiles.py experiments/benchmark_report.py tests/test_query_driven_rework.py tests/test_benchmark_runner.py tests/test_pre_rework_cleanup.py`: passed.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest tests`: passed, 317 tests, 1 PyTorch nested-tensor warning.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m ruff check data evaluation experiments models queries simplification training scripts tests`: passed.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pyright data evaluation experiments models queries simplification training tests`: passed.
- `git diff --check`: passed.
- Both smoke commands above: passed.

## Checkpoint 6 - Train workload replicate signature diagnostics

Status: implemented; current smoke remains blocked

Scope:
- Stop hiding train-workload replicate drift behind the first train workload.
- Make workload-signature diagnostics cover every train-label workload used by the query-prior field and factorized labels.

Changes:
- `range-diagnostics` now emits summaries for `train_r1`, `train_r2`, etc. when `range_train_workload_replicates > 1`.
- `workload_distribution_comparison.workload_signature_gate.pairs` now includes every train replicate against eval.
- Benchmark rows now expose `workload_signature_pair_count` and `workload_signature_failed_pairs`.

Smoke result:
- Refreshed `artifacts/results/query_driven_v2_stability_smoke/example_run.json`.
- Signature pair labels are now `train`, `train_r1`, `train_r2`, `train_r3`.
- All four train/eval signature pairs fail in the synthetic stability smoke. That is the correct artifact behavior: the replicate workloads are visible and cannot be silently ignored.

Tests run:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest tests/test_benchmark_runner.py tests/test_query_driven_rework.py`: passed, 46 tests.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m ruff check experiments/experiment_pipeline.py experiments/benchmark_report.py tests/test_benchmark_runner.py`: passed.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pyright experiments/experiment_pipeline.py experiments/benchmark_report.py tests/test_benchmark_runner.py`: passed.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest tests`: passed, 317 tests, 1 PyTorch nested-tensor warning.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m ruff check data evaluation experiments models queries simplification training scripts tests`: passed.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pyright data evaluation experiments models queries simplification training tests`: passed.
- `git diff --check`: passed.
- Stability smoke command: passed.

## Checkpoint 7 - Final-grid summary and global sanity gate

Status: implemented acceptance machinery; current smokes remain blocked

Scope:
- Replace the static full-grid blocker with benchmark-level evidence that can actually validate the 4x7 final grid.
- Surface `QueryUsefulV1` wins across the compression audit, not just old `RangeUseful`.
- Add an explicit global geometry sanity gate to single-run artifacts.

Changes:
- Benchmark rows now expose per-compression `QueryUsefulV1` scores and MLQDS-vs-uniform / MLQDS-vs-Douglas-Peucker deltas from `range_compression_audit`.
- `benchmark_report.json` now includes `query_driven_final_grid_summary`, with required coverage targets `[0.05, 0.10, 0.15, 0.30]`, required compression ratios `[0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]`, missing-cell reporting, numeric success bars, child-gate failures, and final-grid pass/block status.
- Single-run `final_claim_summary` no longer unconditionally blocks `learning_causality_ablations` and `global_sanity_gates`; those blockers now depend on actual gate evidence. A single run still blocks on `full_coverage_compression_grid`.
- Added `global_sanity_gate` with endpoint-sanity, length-preservation, and average-SED-ratio checks. Catastrophic outlier fraction is recorded as unavailable/report-only because no defensible thresholded outlier metric exists yet.
- `QueryUsefulV1` now consumes actual endpoint sanity from retained masks instead of assuming endpoint sanity is always perfect.
- Bumped benchmark artifact schema to version 5.

Smoke result:
- Refreshed `artifacts/results/query_driven_v2_smoke/example_run.json`.
  MLQDS still beats uniform and Douglas-Peucker on smoke `QueryUsefulV1`: MLQDS `0.5428`, uniform `0.3748`, Douglas-Peucker `0.2822`.
  It remains blocked by workload stability, predictability, workload signature, causality, global sanity, and full-grid gates.
- Refreshed `artifacts/results/query_driven_v2_stability_smoke/example_run.json`.
  It remains blocked. MLQDS loses to uniform on `QueryUsefulV1`: MLQDS `0.3586`, uniform `0.3840`, Douglas-Peucker `0.3299`.
  The stability gate still fails on `coverage_overshoot_tolerance_too_loose`; global sanity also fails because endpoint/length sanity is poor.

Tests run:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest tests/test_benchmark_runner.py tests/test_query_driven_rework.py`: passed, 49 tests.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m ruff check evaluation/evaluate_methods.py experiments/experiment_pipeline.py experiments/benchmark_report.py experiments/benchmark_runner.py experiments/benchmark_artifacts.py tests/test_benchmark_runner.py tests/test_query_driven_rework.py`: passed.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pyright evaluation/evaluate_methods.py experiments/experiment_pipeline.py experiments/benchmark_report.py experiments/benchmark_runner.py experiments/benchmark_artifacts.py tests/test_benchmark_runner.py tests/test_query_driven_rework.py`: passed.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest tests`: passed, 320 tests, 1 PyTorch nested-tensor warning.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m ruff check data evaluation experiments models queries simplification training scripts tests`: passed.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pyright data evaluation experiments models queries simplification training tests`: passed.
- `git diff --check`: passed.
- Fixed-count smoke command: passed.
- Stability smoke command: passed.

## Checkpoint 8 - Mandatory endpoint skeleton floor

Status: implemented; global sanity still fails on length preservation

Scope:
- Fix a selector violation of the guide's global sanity policy.
- Endpoint retention must be a floor when trajectory budget permits, not a best-effort candidate constrained away by the skeleton cap.

Changes:
- `learned_segment_budget_v1` now retains both trajectory endpoints whenever the trajectory has at least two retained slots available.
- The skeleton cap remains in diagnostics, but the trace now reports `skeleton_cap_exceeded_for_endpoint_sanity` when endpoint retention necessarily exceeds the guide cap on tiny budgets or small trajectory counts.
- Updated selector tests to assert endpoint retention and actual skeleton/learned attribution.

Smoke result:
- Refreshed `artifacts/results/query_driven_v2_smoke/example_run.json`.
  Endpoint sanity now passes (`endpoint_sanity=1.0`), MLQDS `QueryUsefulV1` is `0.5556` versus uniform `0.3748` and Douglas-Peucker `0.2822`.
  Global sanity still fails on length preservation (`avg_length_preserved=0.4097`).
- Refreshed `artifacts/results/query_driven_v2_stability_smoke/example_run.json`.
  Endpoint sanity now passes (`endpoint_sanity=1.0`), but MLQDS loses to uniform on `QueryUsefulV1`: MLQDS `0.2826`, uniform `0.3840`, Douglas-Peucker `0.3299`.
  Global sanity still fails on length preservation (`avg_length_preserved=0.5970`).

Interpretation:
- The endpoint issue was a real bug and is fixed.
- The remaining length-preservation failure should not be patched by reintroducing a large temporal scaffold. That would violate the rework objective. The model/selector need better learned query-value plus geometry-aware non-redundancy, not a hidden uniform fallback.

Tests run:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest tests/test_query_driven_rework.py tests/test_benchmark_runner.py`: passed, 49 tests.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m ruff check simplification/learned_segment_budget.py tests/test_query_driven_rework.py`: passed.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pyright simplification/learned_segment_budget.py tests/test_query_driven_rework.py`: passed.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest tests`: passed, 320 tests, 1 PyTorch nested-tensor warning.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m ruff check data evaluation experiments models queries simplification training scripts tests`: passed.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pyright data evaluation experiments models queries simplification training tests`: passed.
- `git diff --check`: passed.
- Fixed-count smoke command: passed.
- Stability smoke command: passed.

## Checkpoint 9 - Prior-field path cleanup and geometry tie-breaker

Status: implemented diagnostics and model-path cleanup; current smokes remain blocked

Scope:
- Add a light geometry-aware non-redundancy tie-breaker inside learned segment-budget selection without reintroducing a temporal scaffold.
- Fix the query-prior field path so `behavior_utility_prior` can be built from the conditional behavior target rather than silently duplicating query-hit probability.
- Add smoothed train-derived prior fields so tiny train workloads do not produce a sparse KNN-like grid.
- Add a dedicated prior-field encoder branch to `workload_blind_range_v2` while preserving the original initialization order and making the new residual branch neutral at initialization.

Changes:
- `learned_segment_budget_v1` now accepts raw points and uses a small geometry-gain tie-breaker during within-segment point selection. The weight is intentionally low; a higher geometry weight hurt query-local metrics.
- `MLQDSMethod`, frozen-mask generation, selector trace generation, and causality ablations now pass points into the learned selector so the tie-breaker is available consistently before eval query scoring.
- `build_train_query_prior_fields` now accepts explicit `behavior_values`; the training path passes the `conditional_behavior_utility` factorized head.
- Prior fields now use a fixed 3x3 binomial smoothing kernel by default. Artifacts report raw and smoothed nonzero spatial-query cell counts.
- `WorkloadBlindRangeV2Model` now has a dedicated `prior_feature_encoder` and trainable `prior_feature_scale`. The branch is constructed after the original model modules so adding it does not perturb the original initialization sequence. Its output layer starts at zero, so the branch must learn a contribution instead of injecting random prior-field noise.
- Checkpoint loading accepts older `workload_blind_range_v2` checkpoints that do not have the new prior branch keys, but still rejects unrelated missing/unexpected state.
- Updated stale module docs that still described `workload_blind_range_v2` as a placeholder.

Smoke result:
- Fixed-count smoke artifact:
  `artifacts/results/query_driven_v2_smoke/example_run.json`.
  MLQDS beats uniform and Douglas-Peucker on `QueryUsefulV1`: MLQDS `0.4408`, uniform `0.3748`, Douglas-Peucker `0.2822`.
  This is below the checkpoint-8 fixed-count smoke (`0.5556`), so the change is not a quality win on that tiny run.
  Smoothed prior diagnostics: raw nonzero spatial-query cells `3`, smoothed nonzero cells `20`.
  Causality still fails because shuffled prior fields, removing prior fields, and removing the segment-budget head have zero effect.
- Stability smoke artifact:
  `artifacts/results/query_driven_v2_stability_smoke/example_run.json`.
  MLQDS still loses narrowly to uniform on `QueryUsefulV1`: MLQDS `0.3774`, uniform `0.3840`, Douglas-Peucker `0.3299`.
  Smoothed prior diagnostics: raw nonzero spatial-query cells `11`, smoothed nonzero cells `80`.
  Predictability is partially better: Spearman `0.2685`, lift at 1% `2.2207`, lift at 2% `2.2207`, lift at 5% `3.3143`; PR-AUC lift is still too low (`0.9359`), so the predictability gate fails.
  Causality still fails. Prior-field ablations remain zero-delta, untrained beats trained in this one-epoch smoke, and removing the behavior head improves the score. This is a real blocker.

Interpretation:
- The prior field itself is less sparse after smoothing and has some held-out predictability signal in the stability smoke.
- The trained model is still not demonstrably using prior fields. This fails the guide's stop rule for `query_prior_field_ablation_no_effect`.
- The geometry tie-breaker is only a tie-breaker. It does not fix length preservation, and increasing it becomes a selector-side geometry hack that hurts the query objective.

Tests run:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest tests/test_query_driven_rework.py tests/test_model_features.py`: passed, 38 tests, 1 PyTorch nested-tensor warning.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m ruff check models/workload_blind_range_v2.py training/train_model.py training/query_prior_fields.py training/checkpoints.py tests/test_query_driven_rework.py tests/test_model_features.py`: passed.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pyright models/workload_blind_range_v2.py training/train_model.py training/query_prior_fields.py training/checkpoints.py tests/test_query_driven_rework.py tests/test_model_features.py`: passed.
- Fixed-count smoke command: passed.
- Stability smoke command: passed.

## Checkpoint 10 - Prior-ablation sensitivity and replacement-target sparsity

Status: implemented diagnostics and a target cleanup; current smokes remain blocked

Scope:
- Distinguish "prior ablation does not change retained masks" from "prior ablation does not even change model scores."
- Reduce one target-diffusion source in `marginal_replacement_gain`; it was assigning replacement value to every queried point, not query-local marginal representatives.
- Avoid leaving an over-sparse target change that damages the fixed smoke.

Changes:
- `MLQDSMethod` now caches raw model predictions separately from post-processed selector scores.
- `learning_causality_summary.prior_sensitivity_diagnostics` now reports raw-prediction sensitivity and selector-score sensitivity for shuffled-prior and no-prior ablations:
  mean absolute score delta, max absolute score delta, retained-mask Jaccard, retained-mask Hamming fraction, and top-k Jaccard at the retained count.
- `marginal_replacement_gain` now uses query-local representative support per contiguous query/trajectory run instead of adding every positive query point to replacement mass.
- The representative keep fraction is `0.50`. A stricter `0.25` probe reduced label diffusion more aggressively but collapsed the fixed smoke below uniform, so it was rejected.
- Added a focused test proving replacement support is sparse on a simple all-in-query run.

Smoke result:
- Fixed-count smoke artifact:
  `artifacts/results/query_driven_v2_smoke/example_run.json`.
  MLQDS beats uniform and Douglas-Peucker on `QueryUsefulV1`: MLQDS `0.4408`, uniform `0.3748`, Douglas-Peucker `0.2822`.
  Training positive fraction is now `0.3906` instead of the earlier diffuse fixed-count label, but prior causality still fails.
  Prior raw-prediction sensitivity remains tiny: no-prior mean absolute raw delta `0.00168`; shuffled-prior mean absolute raw delta `0.00103`. Retained-mask Jaccard remains `1.0`.
- Stability smoke artifact:
  `artifacts/results/query_driven_v2_stability_smoke/example_run.json`.
  MLQDS still loses narrowly to uniform on `QueryUsefulV1`: MLQDS `0.3758`, uniform `0.3840`, Douglas-Peucker `0.3299`.
  Training positive fraction dropped from `0.9115` to `0.7240`, which is progress but still too diffuse for low-budget ranking.
  Prior raw-prediction sensitivity is effectively absent: shuffled-prior mean absolute raw delta `0.0`; no-prior mean absolute raw delta `0.00034`. Retained-mask Jaccard remains `1.0`.
  Predictability remains mixed: Spearman `0.3564`, lift at 1% `1.5903`, lift at 5% `3.1606`, but PR-AUC lift `0.9120`, so the gate fails.

Interpretation:
- The prior field has some held-out predictability in the stability smoke, but the trained model barely uses it. This is now proven at raw-logit level, not just retained-mask level.
- The replacement target is less obviously wrong, but the stability target is still too diffuse and model training still does not produce a credible learned prior-dependent compressor.
- Do not count the fixed-count smoke win as success. It still fails workload stability, predictability, causality, global sanity, and full-grid gates.

Tests run:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest tests/test_query_driven_rework.py tests/test_benchmark_runner.py`: passed, 53 tests.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest tests/test_query_driven_rework.py tests/test_training_losses.py`: passed, 22 tests.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m ruff check evaluation/baselines.py experiments/experiment_pipeline.py tests/test_query_driven_rework.py`: passed.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m ruff check training/query_useful_targets.py tests/test_query_driven_rework.py`: passed.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pyright evaluation/baselines.py experiments/experiment_pipeline.py tests/test_query_driven_rework.py`: passed.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pyright training/query_useful_targets.py tests/test_query_driven_rework.py`: passed.
- Fixed-count smoke command: passed.
- Stability smoke command: passed.

## Checkpoint 11 - Prior-field input audit and query-box prior rasterization

Status: implemented diagnostics and corrected prior-field semantics; current smokes remain blocked

Scope:
- Determine whether prior-field ablations fail because the model ignores prior inputs or because the ablations do not actually perturb sampled eval features.
- Correct the query-hit prior fields so they represent the train workload query distribution, not only train points hit by train queries.
- Add explicit failure reporting when eval points cannot sample meaningful train-derived prior features.

Changes:
- `learning_causality_summary.prior_sensitivity_diagnostics` now includes `sampled_prior_features` for shuffled-prior and no-prior ablations:
  sampled feature delta, primary/nonzero fractions, per-channel deltas, and the fraction of eval compression points outside the prior extent.
- `spatial_query_hit_probability` and `spatiotemporal_query_hit_probability` now rasterize train query boxes into grid cells. They no longer depend solely on train points occupying those cells.
- Query-prior field schema is now version `3`, with `spatial_query_field_source=train_query_box_density` and `out_of_extent_sampling=zero`.
- Prior-field diagnostics now report raw/nonzero query-box cells separately from raw point-hit cells.
- Added `prior_sample_gate_pass` and `prior_sample_gate_failures` to the learning-causality summary. This blocks misleading prior-ablation evidence when sampled priors are all zero, unchanged by shuffling, or mostly out of extent.
- Fixed a misleading sampler behavior: out-of-extent eval points are no longer clamped into edge grid cells where they could pick up fake route/query-prior mass.
- Benchmark rows now expose prior-sample gate fields and sampled-prior support metrics, so a full grid cannot hide absent train-prior support inside a generic causality failure.
- Added an opt-in shared-route synthetic generator mode, `--synthetic_route_families`, for same-support query-prior smoke/probe runs. The default synthetic generator remains independent random routes.
- Added focused tests for query-box rasterization, sampled-prior sensitivity, and prior-sample gate failure explanations.

Smoke result:
- Fixed-count smoke artifact:
  `artifacts/results/query_driven_v2_smoke/example_run.json`.
  MLQDS still beats uniform and Douglas-Peucker on `QueryUsefulV1`: MLQDS `0.4408`, uniform `0.3748`, Douglas-Peucker `0.2822`.
  The prior field is schema `3` and query-box sourced. The prior sample gate fails because all eval points are outside the train-prior extent; sampled train-prior features are now correctly zero rather than clamped into edge-cell route density.
  Failed causality checks:
  `shuffled_prior_fields_should_lose`, `without_query_prior_features_should_lose`, `without_segment_budget_head_should_lose`, and `eval_points_mostly_outside_query_prior_extent`.
- Stability smoke artifact:
  `artifacts/results/query_driven_v2_stability_smoke/example_run.json`.
  MLQDS loses to uniform and slightly to Douglas-Peucker on `QueryUsefulV1`: MLQDS `0.3275`, uniform `0.3840`, Douglas-Peucker `0.3299`.
  Sampled train-prior features at eval points are entirely zero: primary nonzero fraction `0.0`, shuffled sampled-feature delta `0.0`, and outside-prior-extent fraction `1.0`.
  Failed causality checks include `sampled_query_prior_features_all_zero`, `shuffled_prior_fields_did_not_change_sampled_inputs`, and `eval_points_mostly_outside_query_prior_extent`.
  The target is still too diffuse in this smoke: training positive fraction `0.8594`; train target fit Kendall tau is negative (`-0.013`).

Interpretation:
- The earlier "model ignores prior fields" conclusion was incomplete. In the current smokes, the model cannot use query-prior fields because the sampled prior inputs are zero at every eval point.
- The corrected query-box prior is the right workload-prior object, but these synthetic splits still violate the stable-distribution assumption: eval trajectories sit outside the train-prior support.
- The next fix is not another model branch. The next fix is workload/profile support: train-prior support must cover the held-out eval domain under the same future-query distribution, or the run is not valid final evidence.

Tests run:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest tests/test_query_driven_rework.py -q`: passed, 21 tests.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest tests/test_benchmark_runner.py tests/test_query_driven_rework.py -q`: passed, 57 tests.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m ruff check training/query_prior_fields.py experiments/experiment_pipeline.py tests/test_query_driven_rework.py`: passed.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pyright training/query_prior_fields.py experiments/experiment_pipeline.py tests/test_query_driven_rework.py`: passed.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m ruff check experiments/benchmark_report.py tests/test_benchmark_runner.py training/query_prior_fields.py tests/test_query_driven_rework.py`: passed.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pyright experiments/benchmark_report.py tests/test_benchmark_runner.py training/query_prior_fields.py tests/test_query_driven_rework.py`: passed.
- Fixed-count smoke command: passed.
- Stability smoke command: passed.

Support probe:
- Command:
  `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m experiments.run_ais_experiment --n_ships 12 --n_points 48 --n_queries 8 --max_queries 64 --epochs 1 --workload range --query_coverage 0.30 --range_max_coverage_overshoot 0.50 --range_train_workload_replicates 4 --model_type workload_blind_range_v2 --range_training_target_mode query_useful_v1_factorized --workload_profile_id range_workload_v1 --selector_type learned_segment_budget_v1 --checkpoint_score_variant query_useful_v1 --checkpoint_selection_metric loss --compression_ratio 0.20 --range_max_point_hit_fraction 1.0 --range_duplicate_iou_threshold 1.0 --range_acceptance_max_attempts 800 --final_metrics_mode core --results_dir artifacts/results/query_driven_v2_support_probe`
- Result: MLQDS `QueryUsefulV1` `0.2910`, uniform `0.2702`, Douglas-Peucker `0.3182`.
- Prior sample gate still fails: primary sampled prior nonzero fraction `0.0`; outside-prior-extent fraction `0.6667`.
- This confirms the train-prior support failure is not only the 4/6-ship tiny smoke. The synthetic generator/split is not a valid same-support workload test for query-prior learning.

Shared-route support smoke:
- Command:
  `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m experiments.run_ais_experiment --n_ships 6 --n_points 48 --synthetic_route_families 1 --n_queries 8 --max_queries 64 --epochs 1 --workload range --query_coverage 0.30 --range_max_coverage_overshoot 0.50 --range_train_workload_replicates 4 --model_type workload_blind_range_v2 --range_training_target_mode query_useful_v1_factorized --workload_profile_id range_workload_v1 --selector_type learned_segment_budget_v1 --checkpoint_score_variant query_useful_v1 --checkpoint_selection_metric loss --compression_ratio 0.20 --range_max_point_hit_fraction 1.0 --range_duplicate_iou_threshold 1.0 --range_acceptance_max_attempts 800 --final_metrics_mode core --results_dir artifacts/results/query_driven_v2_shared_route_smoke`
- Result: MLQDS `QueryUsefulV1` `0.3651`, uniform `0.2646`, Douglas-Peucker `0.2753`.
- Prior sample gate passes: primary sampled prior nonzero fraction `0.9583`; outside-prior-extent fraction `0.0`.
- Predictability is no longer dead: Spearman `0.1502`, PR-AUC lift `1.3678`, lift@5% `1.5196`, lift@10% `1.6995`. It still fails the strict lift@1% and lift@2% gates.
- Causality still fails because removing query-prior features does not hurt the retained mask and removing the segment-budget head does not hurt. This is not final learned success.

Shared-route 5-epoch probe:
- Command:
  `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m experiments.run_ais_experiment --n_ships 6 --n_points 48 --synthetic_route_families 1 --n_queries 8 --max_queries 64 --epochs 5 --workload range --query_coverage 0.30 --range_max_coverage_overshoot 0.50 --range_train_workload_replicates 4 --model_type workload_blind_range_v2 --range_training_target_mode query_useful_v1_factorized --workload_profile_id range_workload_v1 --selector_type learned_segment_budget_v1 --checkpoint_score_variant query_useful_v1 --checkpoint_selection_metric loss --compression_ratio 0.20 --range_max_point_hit_fraction 1.0 --range_duplicate_iou_threshold 1.0 --range_acceptance_max_attempts 800 --final_metrics_mode core --results_dir artifacts/results/query_driven_v2_shared_route_5epoch_probe`
- Result: MLQDS `QueryUsefulV1` `0.3664`, uniform `0.2646`, Douglas-Peucker `0.2753`.
- Prior field ablations now have material effect: shuffled-prior and no-prior both score `0.3160`, below MLQDS; raw-prediction mean absolute deltas are about `0.0036` and `0.0041`; retained-mask Jaccard is `0.8182`.
- Causality still fails only on `without_segment_budget_head_should_lose` in this probe. This is weak evidence because the eval split has one held-out trajectory and limited segment-allocation pressure.
- Training target is still too diffuse: positive label fraction `0.9740`. The probe proves the model can learn prior-dependent behavior when support exists, but not that the final target/selector is accepted.

## Completion audit - Current state versus objective

Status: not complete

Objective restated as concrete deliverables:
- Read and follow `query-driven-rework-guide.md`.
- Train from a statistically stable future-query workload distribution.
- Compress trajectories before future eval queries are known.
- Preserve points likely to matter for future range-query answers under that distribution.
- Use `QueryUsefulV1` as the primary product metric and keep legacy `RangeUseful` diagnostic-only.
- Make the model actually learn; reject wins caused by temporal scaffolding, query-conditioned inference, checkpoint leakage, KNN lookup, retained-frequency hacks, or selector tricks.
- Work in checkpoints and keep this progress log current.

Prompt-to-artifact checklist:
- Canonical guide read and used: yes. This log and implementation reference `docs/query-driven-rework-guide.md`.
- Versioned workload profile: implemented. Evidence: `queries/workload_profiles.py`, `queries/query_generator.py`, workload signatures in smoke artifact.
- Train/eval same workload profile with held-out seeds: implemented in config path. Evidence: smoke artifact has train/eval `profile_id=range_workload_v1`; held-out seeds are used. Stability gate fails in the tiny smoke.
- Workload statistical-stability gate: implemented. Evidence: `workload_stability_gate` blocks fixed-count/tiny/loose-overshoot runs and the final benchmark profile now uses 4 train workload replicates.
- Workload-blind eval compression: implemented for the candidate path. Evidence: smoke `workload_blind_protocol` has `masks_frozen_before_eval_query_scoring=true` and all eval-query-seen flags false.
- `QueryUsefulV1`: implemented and reported. Evidence: `evaluation/query_useful_v1.py`, `evaluation/evaluate_methods.py`, benchmark-report fields, smoke `matched.*.query_useful_v1_score`.
- Legacy `RangeUseful` diagnostic-only: implemented in final-claim/report separation.
- Factorized `QueryUsefulV1` labels: implemented. Evidence: `training/query_useful_targets.py`, `training/training_epoch.py`, smoke `training_target_diagnostics.target_family=QueryUsefulV1Factorized`.
- Train-only query-prior fields: implemented. Evidence: `training/query_prior_fields.py`; smoke metadata says train-only and no eval/validation queries.
- Trainable workload-blind model: implemented. Evidence: `models/workload_blind_range_v2.py`; smoke trains for one epoch.
- Learned segment-budget selector: implemented. Evidence: `simplification/learned_segment_budget.py`; smoke selector type is `learned_segment_budget_v1`.
- Temporal scaffold avoidance: implemented for the candidate path. Evidence: selector diagnostics show no fixed 85% scaffold; smoke has per-retained-point attribution with 6/7 learned-controlled retained points and 0 fallback points; segment allocation uses the factorized segment-budget head.
- Predictability audit: implemented. Evidence: `training/predictability_audit.py`; smoke audit exists. Gate fails.
- Workload signature gate: implemented. Evidence: `experiments/range_diagnostics.py`; smoke gate exists. Gate fails.
- Train replicate signature visibility: implemented. Evidence: stability smoke signature gate includes `train`, `train_r1`, `train_r2`, and `train_r3` pairs against eval.
- Learning causality report: implemented for smoke-level required ablations, but it fails. Evidence: smoke has shuffled-score, untrained-model, prior-only, shuffled-prior-field, no-query-prior-feature, no-behavior-head, and no-segment-budget-head ablations, plus point-level selector attribution. Current artifacts also prove sampled prior inputs are absent or mostly out of extent in the synthetic smokes, so prior-field ablations are not valid learning evidence.
- Prior-field encoder and smoothed query-box prior fields: implemented. Evidence: `models/workload_blind_range_v2.py` has a dedicated prior branch and `training/query_prior_fields.py` reports schema-2 query-box prior diagnostics. This is not accepted evidence because eval points in the current smokes do not sample useful train-prior support.
- Global sanity gate: implemented for endpoint sanity, length preservation, and average SED ratio. Current smokes now pass endpoint sanity but fail length preservation. Catastrophic outlier fraction remains report-only until a real outlier metric is defined.
- Full final grid acceptance summary: implemented at benchmark level. Evidence: `query_driven_final_grid_summary` in benchmark artifacts. Full grid not run.
- Numeric success bars: not met. Fixed-count smoke beats uniform on `QueryUsefulV1`, but workload stability/signature/predictability/global-sanity gates fail; stability smoke loses to uniform; full grid is not run; causality checks still fail; train-prior support does not cover eval compression points in the synthetic smokes.

Completion decision:
- Do not mark the active goal complete.
- The implementation now prevents false final claims and exposes the blocking evidence.
- The next work should not be architecture tuning. Per the guide, fix workload stability, train-prior support, and predictability first, then rerun the grid and causality gates.

## Checkpoint 12 - Causality ablation cleanup and selector allocation fix

Changes:
- Fixed the `MLQDS_without_segment_budget_head` ablation. It now uses neutral constant segment scores instead of passing `segment_scores=None`, which previously fell back to `point_score_top20_mean` and still used learned point scores for segment allocation.
- Replaced the learned-segment allocator's sorted round-robin behavior with score-weighted diminishing-priority allocation. Equal segment scores still degrade to uniform allocation, so the no-head ablation is now a real neutral control.
- Fixed factorized head ablations in `workload_blind_range_v2`: disabled multiplicative heads now use neutral probability `0.5` rather than `1.0`. The old behavior made `without_behavior_utility_head` an advantaged ablation, not a removal.
- Exposed `segment_budget_head_ablation_mode` in benchmark rows.
- Tried a bounded query-free geometry guardrail for length preservation and removed it after measurement. It still failed the length gate and broke causality (`prior_field_only` beat the trained model), so keeping it would have been misleading.

Tests run:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest tests/test_query_driven_rework.py tests/test_benchmark_runner.py -q`: passed, 60 tests.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m ruff check simplification/learned_segment_budget.py models/workload_blind_range_v2.py experiments/experiment_pipeline.py experiments/benchmark_report.py tests/test_query_driven_rework.py tests/test_benchmark_runner.py`: passed.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest tests`: passed, 332 tests, 1 PyTorch nested-tensor warning.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m ruff check data evaluation experiments models queries simplification training scripts tests`: passed.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pyright data evaluation experiments models queries simplification training tests`: passed.
- `git diff --check`: passed.

Shared-route 5-epoch probe after cleanup:
- Command:
  `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m experiments.run_ais_experiment --n_ships 6 --n_points 48 --synthetic_route_families 1 --n_queries 8 --max_queries 64 --epochs 5 --workload range --query_coverage 0.30 --range_max_coverage_overshoot 0.50 --range_train_workload_replicates 4 --model_type workload_blind_range_v2 --range_training_target_mode query_useful_v1_factorized --workload_profile_id range_workload_v1 --selector_type learned_segment_budget_v1 --checkpoint_score_variant query_useful_v1 --checkpoint_selection_metric loss --compression_ratio 0.20 --range_max_point_hit_fraction 1.0 --range_duplicate_iou_threshold 1.0 --range_acceptance_max_attempts 800 --final_metrics_mode core --results_dir artifacts/results/query_driven_v2_shared_route_5epoch_probe`
- Result: MLQDS `QueryUsefulV1` `0.3676`, uniform `0.2646`, Douglas-Peucker `0.2753`.
- Learning causality gate now passes on this same-support probe:
  - shuffled score delta `+0.1589`
  - untrained model delta `+0.1385`
  - shuffled/no query-prior delta `+0.0511`
  - no behavior-head delta `+0.0724`
  - no segment-budget-head delta `+0.0011`
  - prior sample gate passes.
- This is useful evidence, not final acceptance. The no-segment-head delta is positive but tiny, so segment-budget learning is still weak.
- Global sanity still fails: length preservation `0.5400` versus required minimum `0.80`; SED ratio versus uniform is within limit (`1.213 <= 1.50`).
- Workload stability still fails in this probe because `range_max_coverage_overshoot=0.50` is intentionally loose for the smoke; the gate requires `0.02`.
- Training target remains too diffuse: positive label fraction `0.9740`.

Current decision:
- Keep the ablation and allocator fixes.
- Do not claim final success. The candidate now has one clean same-support learning-causality probe, but it still fails global sanity, strict workload stability, target diffusion, predictability-at-low-budget, and full-grid acceptance.

Strict workload-stability probes:
- A strict shared-route probe with `--range_max_coverage_overshoot 0.02` can satisfy the workload-stability gate.
- Three-epoch strict command:
  `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m experiments.run_ais_experiment --n_ships 6 --n_points 48 --synthetic_route_families 1 --n_queries 8 --max_queries 64 --epochs 3 --workload range --query_coverage 0.30 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 4 --model_type workload_blind_range_v2 --range_training_target_mode query_useful_v1_factorized --workload_profile_id range_workload_v1 --selector_type learned_segment_budget_v1 --checkpoint_score_variant query_useful_v1 --checkpoint_selection_metric loss --compression_ratio 0.20 --range_max_point_hit_fraction 1.0 --range_duplicate_iou_threshold 1.0 --range_acceptance_max_attempts 2000 --final_metrics_mode core --results_dir artifacts/results/query_driven_v2_shared_route_strict_overshoot_probe`
- Ten-epoch strict command:
  `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m experiments.run_ais_experiment --n_ships 6 --n_points 48 --synthetic_route_families 1 --n_queries 8 --max_queries 64 --epochs 10 --workload range --query_coverage 0.30 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 4 --model_type workload_blind_range_v2 --range_training_target_mode query_useful_v1_factorized --workload_profile_id range_workload_v1 --selector_type learned_segment_budget_v1 --checkpoint_score_variant query_useful_v1 --checkpoint_selection_metric loss --compression_ratio 0.20 --range_max_point_hit_fraction 1.0 --range_duplicate_iou_threshold 1.0 --range_acceptance_max_attempts 2000 --final_metrics_mode core --results_dir artifacts/results/query_driven_v2_shared_route_strict_10epoch_probe`
- Ten-epoch strict result: MLQDS `QueryUsefulV1` `0.1664`, uniform `0.1981`, Douglas-Peucker `0.2895`.
- Workload-stability gate passes, but predictability still fails, learning causality fails, and global sanity fails length preservation (`0.5913 < 0.80`).
- Longer training does not fix the strict-profile failure. The target remains diffuse (`positive_label_fraction=0.9583`), and ablations show the learned model is not reliably using the prior under strict coverage.
- This narrows the next blocker: the accepted workload profile and current factorized target are still not producing a robust learnable held-out signal at this small scale. Do not paper this over with geometry guardrails or selector tricks.

## Checkpoint 13 - Target diffusion gate

Status: implemented blocker reporting; training behavior unchanged

Scope:
- Make the guide's target-diffusion warning an explicit artifact gate.
- Distinguish practical target support (`>0.01`) from tiny nonzero label noise.
- Keep failed ad hoc target sparsification out of the accepted code path.

Changes:
- Added `support_fraction_by_threshold_by_head` to factorized target diagnostics.
- Added `final_label_support_fraction_by_threshold` for the combined `QueryUsefulV1` scalar target.
- Added `target_diffusion_gate` to single-run artifacts and final-claim blocking gates.
- Added `target_diffusion_gate_pass` to benchmark rows and to the benchmark-level final-grid required child gates.
- The gate fails when the final label or any factorized head has practical support above `0.50`, or when the top `5%` of points captures less than `0.10` of a head's label mass.

Rejected experiment:
- Tried more aggressive replacement-target sparsification after checkpoint 12: repeated-support gating, lower replacement keep fraction, and top active segment support.
- Reverted it. It reduced positive support in some probes, but broke same-support causality and still did not make the strict workload-stability profile pass. That is selector/label tuning without a robust learned signal.

Strict gate smoke:
- Command:
  `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m experiments.run_ais_experiment --n_ships 6 --n_points 48 --synthetic_route_families 1 --n_queries 8 --max_queries 64 --epochs 1 --workload range --query_coverage 0.30 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 4 --model_type workload_blind_range_v2 --range_training_target_mode query_useful_v1_factorized --workload_profile_id range_workload_v1 --selector_type learned_segment_budget_v1 --checkpoint_score_variant query_useful_v1 --checkpoint_selection_metric loss --compression_ratio 0.20 --range_max_point_hit_fraction 1.0 --range_duplicate_iou_threshold 1.0 --range_acceptance_max_attempts 2000 --final_metrics_mode core --results_dir artifacts/results/query_driven_v2_target_diffusion_gate_smoke`
- Result: `target_diffusion_gate.gate_pass=false`.
- Final-label support above `0.01` is `0.7917`; the gate fails `final_label_support_fraction_above_max`.
- Failing heads include `query_hit_probability`, `conditional_behavior_utility`, `boundary_event_utility`, `marginal_replacement_gain`, and `segment_budget_target`.
- Final-claim blockers now include `target_diffusion_gate` along with predictability, workload signature, learning causality, global sanity, and full-grid gates.

Tests run:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest tests/test_query_driven_rework.py tests/test_benchmark_runner.py -q`: passed, 62 tests.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m ruff check training/factorized_target_diagnostics.py training/query_useful_targets.py experiments/experiment_pipeline.py experiments/benchmark_report.py tests/test_query_driven_rework.py tests/test_benchmark_runner.py`: passed.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pyright training/factorized_target_diagnostics.py training/query_useful_targets.py experiments/experiment_pipeline.py experiments/benchmark_report.py tests/test_query_driven_rework.py tests/test_benchmark_runner.py`: passed.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest tests`: passed, 334 tests, 1 PyTorch nested-tensor warning.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m ruff check data evaluation experiments models queries simplification training scripts tests`: passed.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pyright data evaluation experiments models queries simplification training tests`: passed.
- `git diff --check`: passed.

Current decision:
- The target is now formally blocked as too diffuse for the product claim.
- This does not solve the learning problem. It prevents a misleading final claim while the next real fix is designed.

## Checkpoint 14 - Factorized target semantic cleanup

Status: implemented; strict profile still blocked

Scope:
- Reduce label diffusion through the target definition itself, not through selector post-processing.
- Align the model's interpretable final-score formula with the revised target formula.
- Refine the diffusion gate so broad query-hit probability and per-point segment-budget labels are reported but do not falsely fail point-ranking target support.

Changes:
- `conditional_behavior_utility` now uses sparse high-change trajectory behavior. Per-trajectory behavior weights are high-passed above the `0.70` quantile instead of treating every tiny motion variation as behavior utility.
- `marginal_replacement_gain` is now expected per-query marginal gain (`replacement_mass / query_count`) instead of max-normalized retained-frequency-like support.
- `boundary_event_utility` now uses squared event probability, matching stable repeated boundary evidence rather than one-off query-boundary hits.
- The final scalar target is now:
  `q_hit * replacement * (0.5 + behavior) + 0.25 * boundary_event_utility`.
- `workload_blind_range_v2` now uses the same interpretable score formula before calibration.
- `target_diffusion_gate` now blocks on final-label support and true point-rank utility heads:
  `conditional_behavior_utility`, `boundary_event_utility`, and `marginal_replacement_gain`.
  It reports `query_hit_probability` and `segment_budget_target` diffusion without treating their point support as a final blocker.

Strict 10-epoch target-redesign probe:
- Command:
  `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m experiments.run_ais_experiment --n_ships 6 --n_points 48 --synthetic_route_families 1 --n_queries 8 --max_queries 64 --epochs 10 --workload range --query_coverage 0.30 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 4 --model_type workload_blind_range_v2 --range_training_target_mode query_useful_v1_factorized --workload_profile_id range_workload_v1 --selector_type learned_segment_budget_v1 --checkpoint_score_variant query_useful_v1 --checkpoint_selection_metric loss --compression_ratio 0.20 --range_max_point_hit_fraction 1.0 --range_duplicate_iou_threshold 1.0 --range_acceptance_max_attempts 2000 --final_metrics_mode core --results_dir artifacts/results/query_driven_v2_target_redesign_strict_10epoch_probe`
- Result: MLQDS `QueryUsefulV1` `0.1823`, uniform `0.1981`, Douglas-Peucker `0.2895`.
- Target diffusion gate passes. Final support above `0.01` is `0.2813`, versus `0.7917` before the semantic cleanup.
- Workload-stability gate passes.
- Still blocked by predictability, learning causality, global sanity, and full-grid gates.
- Causality failures are now `untrained_model_should_lose` and `prior_field_only_should_not_match_trained`. This is narrower than the earlier strict failure set, but still fatal.
- Global sanity still fails length preservation (`0.5878 < 0.80`).

Loose 5-epoch same-support probe:
- Command:
  `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m experiments.run_ais_experiment --n_ships 6 --n_points 48 --synthetic_route_families 1 --n_queries 8 --max_queries 64 --epochs 5 --workload range --query_coverage 0.30 --range_max_coverage_overshoot 0.50 --range_train_workload_replicates 4 --model_type workload_blind_range_v2 --range_training_target_mode query_useful_v1_factorized --workload_profile_id range_workload_v1 --selector_type learned_segment_budget_v1 --checkpoint_score_variant query_useful_v1 --checkpoint_selection_metric loss --compression_ratio 0.20 --range_max_point_hit_fraction 1.0 --range_duplicate_iou_threshold 1.0 --range_acceptance_max_attempts 800 --final_metrics_mode core --results_dir artifacts/results/query_driven_v2_target_redesign_5epoch_probe`
- Result: MLQDS `QueryUsefulV1` `0.3511`, uniform `0.2646`, Douglas-Peucker `0.2753`.
- This still is not final evidence: workload overshoot is intentionally loose, target diffusion fails under that loose workload (`final support >0.01 = 0.5885`), and the no-query-prior ablation beats the trained model.

Tests run:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest tests/test_query_driven_rework.py -q`: passed, 26 tests.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m ruff check training/query_useful_targets.py models/workload_blind_range_v2.py experiments/experiment_pipeline.py tests/test_query_driven_rework.py`: passed.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pyright training/query_useful_targets.py models/workload_blind_range_v2.py experiments/experiment_pipeline.py tests/test_query_driven_rework.py`: passed.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest tests`: passed, 334 tests, 1 PyTorch nested-tensor warning.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m ruff check data evaluation experiments models queries simplification training scripts tests`: passed.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pyright data evaluation experiments models queries simplification training tests`: passed.
- `git diff --check`: passed.

Current decision:
- Keep the semantic target cleanup. It fixes the strict target-diffusion blocker and slightly improves strict `QueryUsefulV1` (`0.1664 -> 0.1823`), but it does not solve the product claim.
- The remaining hard blocker is learned transfer: strict-profile trained MLQDS still loses to uniform and Douglas-Peucker, and untrained/prior-only controls still beat or match it.

## Checkpoint 15 - Validation checkpoint scoring uses segment-budget head

Status: implemented; strict validation-selected probe improves but remains blocked

Scope:
- Make validation checkpoint scoring match final workload-blind inference for `learned_segment_budget_v1`.
- Stop selecting checkpoints with a validation selector that ignores the factorized segment-budget head.

Problem found:
- Training-time validation used `simplify_mlqds_predictions(predictions, ...)` with only final point logits.
- For `selector_type=learned_segment_budget_v1`, that meant validation selection did not pass `segment_scores` from the `segment_budget_target` head.
- Final evaluation did pass segment scores through `MLQDSMethod`, so validation checkpoint selection and final inference were inconsistent.

Changes:
- `simplify_mlqds_predictions` now accepts optional `segment_scores` and `points`, and forwards them to `learned_segment_budget_v1`.
- Validation scoring now uses `windowed_predict_with_heads` when available.
- Validation scoring extracts `sigmoid(segment_budget_target)` and passes it to the learned segment-budget selector.
- Added a regression test proving validation checkpoint scoring forwards segment-head scores into the learned selector.

Strict 10-epoch validation-selected probe:
- Command:
  `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m experiments.run_ais_experiment --n_ships 6 --n_points 48 --synthetic_route_families 1 --n_queries 8 --max_queries 64 --epochs 10 --workload range --query_coverage 0.30 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 4 --model_type workload_blind_range_v2 --range_training_target_mode query_useful_v1_factorized --workload_profile_id range_workload_v1 --selector_type learned_segment_budget_v1 --checkpoint_score_variant query_useful_v1 --checkpoint_selection_metric uniform_gap --validation_score_every 1 --checkpoint_full_score_every 1 --checkpoint_candidate_pool_size 1 --compression_ratio 0.20 --range_max_point_hit_fraction 1.0 --range_duplicate_iou_threshold 1.0 --range_acceptance_max_attempts 2000 --final_metrics_mode core --results_dir artifacts/results/query_driven_v2_validation_segment_head_strict_10epoch_probe`
- Result: MLQDS `QueryUsefulV1` `0.2527`, uniform `0.1981`, Douglas-Peucker `0.2895`.
- This is a real improvement over strict loss-selected target-redesign MLQDS `0.1823`.
- Best checkpoint is epoch 1 by held-out validation uniform-gap selection. Later epochs overfit train loss and degrade validation score.
- Target diffusion gate passes.
- Causality still fails: `untrained_model_should_lose` and `prior_field_only_should_not_match_trained`.
- Global sanity still fails length preservation (`0.5605 < 0.80`).
- Predictability gate still fails low-budget lift.
- Workload-stability gate fails in this tiny probe because the held-out selection workload has only 2 queries and misses target coverage. That is a small-probe artifact, not final evidence.

Rejected experiment:
- Tried increasing behavior-head influence by changing the final formula to
  `q_hit * replacement * (0.25 + 0.75 * behavior) + 0.25 * boundary`.
- Reverted it. Strict validation-selected MLQDS dropped to `QueryUsefulV1=0.1652`, below uniform `0.1981`, so it damaged the main accepted-workload probe instead of fixing causality.
- The behavior-head ablation remains weak. That is a real open issue, not something to paper over by lowering the ablation standard.

Tests run:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest tests/test_training_does_not_collapse.py tests/test_query_driven_rework.py -q`: passed, 63 tests, 1 PyTorch nested-tensor warning.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m ruff check training/training_validation.py simplification/mlqds_scoring.py tests/test_training_does_not_collapse.py`: passed.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pyright training/training_validation.py simplification/mlqds_scoring.py tests/test_training_does_not_collapse.py`: passed.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest tests`: passed, 335 tests, 1 PyTorch nested-tensor warning.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m ruff check data evaluation experiments models queries simplification training scripts tests`: passed.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pyright data evaluation experiments models queries simplification training tests`: passed.
- `git diff --check`: passed.

Current decision:
- Keep the validation-segment fix. It closes a real train/eval mismatch and improves strict QueryUsefulV1 enough to beat uniform in this probe.
- Do not claim success. DP still wins, global sanity fails, predictability fails, and causality remains invalid because untrained/prior-only controls are still too strong.

## Checkpoint 16 - Material learning-causality delta gate

Status: implemented gate tightening; training behavior unchanged

Scope:
- Prevent final-candidate artifacts from passing learning causality on microscopic positive ablation deltas.
- Align the causality gate with the guide's requirement that trained scores beat shuffled/untrained/prior controls by a material amount, not just by numerical noise.

Problem found:
- `learning_causality_gate_pass` treated every positive `QueryUsefulV1` delta as a pass.
- That was too weak. A run where the no-segment-head or prior-only control loses by `0.0001` could pass despite not proving that the learned heads or train-derived priors materially control retained masks.

Changes:
- Added a material delta floor of `0.005` `QueryUsefulV1` for causality ablation checks.
- Added the guide's shuffled-score threshold: shuffled-score ablation must lose by at least `0.60` of the positive MLQDS-vs-uniform `QueryUsefulV1` gap, with the same `0.005` minimum.
- Artifacts now record `learning_causality_delta_gate` with the per-check thresholds, the MLQDS-vs-uniform gap, and the configured threshold constants.
- Benchmark rows now expose the material-delta threshold fields and threshold map.

Current decision:
- Keep the stricter gate. It will make some previously "passing" same-support probes fail on weak segment-head or prior-control separation. That is correct. Tiny ablation deltas are not learning evidence.
- This does not improve model quality. It reduces the chance of a false final claim.

## Checkpoint 17 - Trajectory-anchored geometry tie-breaker

Status: implemented selector bug fix; smoke behavior unchanged

Scope:
- Fix a flaw in the light geometry-aware non-redundancy tie-breaker inside `learned_segment_budget_v1`.
- Keep the fix query-free and selector-local. This is not a temporal scaffold and does not use eval queries.

Problem found:
- Within-segment learned point selection only passed retained anchors from the current segment into the length-gain tie-breaker.
- For an interior segment with no already retained point, length-gain scores collapsed to zero even when trajectory endpoints had already been retained globally.
- That made the "geometry-aware" tie-breaker silently degenerate into pure learned-score ranking for many interior segments.

Changes:
- Segment point selection now evaluates candidates in trajectory-local coordinates.
- Candidate scores remain finite only inside the allocated segment, but existing retained anchors are taken from the full trajectory.
- Added a regression test where an interior segment has no local retained anchor; the selector now chooses the candidate with real trajectory-level length gain.

Smoke result:
- Refreshed `artifacts/results/query_driven_v2_causality_gate_smoke/example_run.json`.
- MLQDS `RangeUseful` and length preservation are unchanged in this tiny smoke because the held-out trajectory is too short to exercise an anchorless interior segment under the default segment size.
- The artifact now includes `learning_causality_delta_gate` from checkpoint 16.

Current decision:
- Keep the selector fix. It corrects a real implementation bug without giving the selector extra future-query information.
- It does not solve global sanity in current strict probes. Length preservation remains a blocker.

## Checkpoint 18 - Support-valid strict single-cell correction

Status: implemented checkpoint scope; strict support-valid debug cell remains blocked

Scope:
- Fix no-prior ablation confounding by preserving the train prior extent and metadata while zeroing query-prior channels.
- Add a first-class train/eval support-overlap gate and block final claims on it.
- Penalize validation checkpoint selection for bad global sanity instead of letting validation `QueryUsefulV1` assume optimistic geometry.
- Tighten `QueryUsefulV1` with one true query-local interpolation-fidelity component.
- Clean misleading workload-profile semantics and make final workload coverage calibration profile-sampled instead of uncovered-anchor chasing.
- Add segment-level/listwise training pressure for the segment-budget head.
- Run one strict support-valid shared-route debug cell before any full-grid work.

Changes:
- Added `zero_query_prior_field_like`, and the no-query-prior ablation now uses a zeroed train prior field instead of removing the field entirely.
- Added `support_overlap_gate` with extent, sampled-prior, route-density, query-prior-support, and train/eval spatial-intersection checks. Final-claim and benchmark reporting now include the gate.
- Validation `query_useful_v1` scoring now includes geometry sanity and returns a penalized selection score when length, SED, or endpoint sanity are bad.
- Added `range_query_local_interpolation_fidelity`, based on reconstruction of query-local removed points from retained trajectory anchors. `QueryUsefulV1` schema is now version 2 and records `query_useful_v1_metric_maturity=bridge_with_true_query_local_interpolation_component`.
- Added `coverage_calibration_mode`. `range_workload_v1` defaults to `profile_sampled_query_count`; legacy generation remains explicitly marked as `uncovered_anchor_chasing`.
- Implemented actual corridor-like query boxes and distinct `port_or_approach_zone` anchor weighting.
- Renamed the misleading `marginal_replacement_gain` head to `replacement_representative_value` because it is not true counterfactual marginal gain.
- Added segment-pooled BCE and pairwise segment-rank loss for the segment-budget head. Training diagnostics now report segment-head tau and top-k target-mass recall.

Strict support-valid shared-route probe:
- Command:
  `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m experiments.run_ais_experiment --results_dir artifacts/results/query_driven_v2_checkpoint18_strict_support_probe --n_ships 8 --n_points 128 --synthetic_route_families 1 --seed 1818 --n_queries 48 --query_coverage 0.10 --max_queries 512 --range_max_coverage_overshoot 0.0075 --range_train_workload_replicates 4 --workload_profile_id range_workload_v1 --coverage_calibration_mode profile_sampled_query_count --model_type workload_blind_range_v2 --range_training_target_mode query_useful_v1_factorized --selector_type learned_segment_budget_v1 --checkpoint_score_variant query_useful_v1 --checkpoint_selection_metric uniform_gap --validation_score_every 1 --checkpoint_full_score_every 1 --epochs 3 --embed_dim 32 --num_heads 2 --num_layers 1 --train_batch_size 8 --inference_batch_size 8 --compression_ratio 0.05 --mlqds_temporal_fraction 0.0 --mlqds_hybrid_mode fill --mlqds_score_mode rank_confidence --range_acceptance_max_attempts 6000 --final_metrics_mode diagnostic`
- Artifact:
  `artifacts/results/query_driven_v2_checkpoint18_strict_support_probe/example_run.json`.
- Support overlap passes: outside train-prior extent `0.0000`, sampled prior nonzero `0.9922`, primary sampled prior nonzero `0.6914`, route-density overlap `0.9492`, query-prior support overlap `0.6914`, spatial extent intersection `1.0000`.
- Workload stability passes under strict settings: target coverage `0.10`, overshoot `0.0075`, four train workload replicates, profile-sampled coverage calibration. Some workloads stop by acceptance or guard exhaustion, but their final coverage satisfies the target.
- Target diffusion passes: final label support above `0.01` is `0.0109`.
- `QueryUsefulV1`: MLQDS `0.2394`, uniform `0.2430`, Douglas-Peucker `0.3113`. MLQDS loses to both baselines.
- `RangePointF1`: MLQDS `0.2589`, uniform `0.2758`, Douglas-Peucker `0.3458`.
- `RangeUseful`: MLQDS `0.2011`, uniform `0.1980`, Douglas-Peucker `0.2536`.
- Global sanity still fails length preservation: MLQDS `0.5370` versus required `0.80`; uniform is `0.5898`, Douglas-Peucker is `0.7000`. Endpoint sanity passes and SED ratio versus uniform is within limit.
- Predictability fails: Spearman `-0.3708`, lift@1/2/5% all `0.0`, PR-AUC lift `1.0641`.
- Learning causality fails every material-delta check. Shuffled scores and prior-field-only controls beat trained MLQDS; shuffled/zeroed priors and disabled heads do not materially change retained masks or score.
- Segment-head diagnostics are weak: segment-head Kendall tau `0.0950`, top-5% segment target-mass recall `0.1070`.

Focused tests run:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest tests/test_query_driven_rework.py tests/test_benchmark_runner.py::test_query_driven_final_grid_summary_accepts_complete_passing_grid tests/test_benchmark_runner.py::test_benchmark_row_records_effective_child_torch_runtime`: passed, 40 tests.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest tests/test_query_driven_rework.py::test_query_useful_v1_has_true_query_local_interpolation_component tests/test_training_does_not_collapse.py::test_validation_range_usefulness_matches_final_audit tests/test_training_does_not_collapse.py::test_validation_query_score_matches_final_mlqds_scoring`: passed, 9 tests.

Full verification:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest tests`: passed, 348 tests, 1 PyTorch nested-tensor warning.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m ruff check data evaluation experiments models queries simplification training scripts tests`: passed.
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pyright data evaluation experiments models queries simplification training tests`: passed.
- `git diff --check`: passed.

Current decision:
- Keep the checkpoint 18 fixes. They remove real confounds and make the artifact harder to misread.
- Do not claim final success. The first strict support-valid cell is now a cleaner failure: support exists and strict workload generation is acceptable, but learned transfer still is not convincing.
- Next work should target actual learned signal and global geometry preservation, not a full 4x7 grid. Running the grid now would mostly document a failure already visible in the single-cell probe.
