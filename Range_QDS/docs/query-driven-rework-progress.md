# Query-Driven Rework Progress

This is the short checkpoint log required by `docs/query-driven-rework-guide.md`.
Detailed stdout and raw metrics are kept in `artifacts/results/`.

## High-Value Summary

The redesign has made real progress, but it is not complete. The project has moved from broad structural uncertainty to a narrower candidate-level blocker. The current best strict synthetic/debug cell beats both final baselines on `QueryUsefulV1`, while workload stability, support overlap, target diffusion, prior predictability, prior-predictive alignment, and workload signature gates pass. The remaining blockers are learning-causality materiality and global sanity, especially length preservation.

Current best single-cell evidence is promising but not final success:

```text
MLQDS QueryUsefulV1:           0.1669032451715525
uniform QueryUsefulV1:         0.14223795796380634
Douglas-Peucker QueryUsefulV1: 0.16362459837911367
length preservation:           0.7938149625265364
```

Interpretation:
- This is the best current candidate because it beats both uniform and Douglas-Peucker in one strict synthetic/debug cell while keeping the workload/prior gates healthy.
- It is not a final success claim because learning causality still fails and length preservation is below the active `0.80` gate.
- The full 4x7 grid should remain unrun until the strict single-cell gates pass.
- The next useful work is not more broad sweeping. It is targeted work on selector/length allocation and material learned causality from the current best candidate.

Major durable discoveries so far:
- Balanced synthetic split cardinalities were necessary to make workload-signature diagnostics meaningful. The old default `70/15/15` synthetic split created misleading raw hit-count and query-count drift.
- Prior predictability became healthy after target/predictability fixes. The current blocker is no longer generic prior support or target diffusion.
- Raw factorized scalar targets plus factorized head base-rate initialization materially improved model calibration and produced the first strict-cell MLQDS win over Douglas-Peucker in this sequence.
- `route_density_prior` is harmful under the current raw-factorized/head-initialized setup. It should stay available for diagnostics/support overlap, but be excluded from v2 model inputs. Do not generalize this finding to older target/model states.
- `learned_segment_length_repair_fraction=0.6` is material to the current best candidate. Removing repair improves `QueryUsefulV1` and some causality signs, but invalidates global geometry. Full repair or stronger geometry repair weakens learned control or loses to Douglas-Peucker.
- Training-fit improvements are not enough. Several changes improved fit diagnostics but worsened retained-mask quality.

Current research question:

```text
Can the selector/model make train-derived prior, behavior, and score perturbations materially affect frozen retained masks while preserving at least 0.80 length and the current MLQDS win over uniform and Douglas-Peucker?
```

If a future checkpoint does not answer that question more clearly, it is probably low-value.

## Current State — 2026-05-17

Status: active, not complete

Best current code candidate:
- `workload_blind_range_v2`
- `route_density_prior` excluded from v2 model inputs
- hidden prior residual scale `0.25`
- no direct prior-to-head residual
- `learned_segment_score_blend_weight=0.05`
- `learned_segment_length_repair_fraction=0.6`

Best current strict artifact:
- path: `artifacts/results/query_driven_v2_checkpoint04_no_route_density_strict_probe_c10_r05`

Best current strict result:
- MLQDS QueryUsefulV1: `0.1669032451715525`
- uniform QueryUsefulV1: `0.14223795796380634`
- Douglas-Peucker QueryUsefulV1: `0.16362459837911367`
- length preservation: `0.7938149625265364`
- gates passed: workload stability, support overlap, predictability, prior-predictive alignment, target diffusion, workload signature
- gates failed: learning causality, global sanity

Current blockers:
- Learning causality still fails. In the best strict artifact, key deltas are correct-sign but below material thresholds: shuffled scores `0.008957581030671818` versus required `0.014799172324647697`; untrained model `0.002338397270806869` versus required `0.005`; shuffled prior fields `0.0028898208710833317` versus required `0.005`; without query-prior features `0.0028898208710833317` versus required `0.005`.
- Segment-budget-head materiality is already useful in the best strict artifact: `0.010472792329425523`, above the `0.005` material threshold. Do not treat all heads as equally weak.
- Length preservation is close but still below the guide's active `0.80` gate: `0.7938149625265364`.
- No-length-repair improves MLQDS QueryUsefulV1 to `0.1759846099523811`, but length collapses to `0.6790996203798462` and learning causality still fails. It is a diagnostic, not a candidate.
- Full 4x7 grid remains intentionally unrun because strict single-cell gates still fail.

Current decision:
- Do not run the full grid.
- Do not increase workload/caps yet; current standard strict cell already has healthy accepted query counts.
- Do not lower gates for a success claim while learning causality still fails.
- Do not lower the length gate to `0.75`; that would still leave learning causality failed.
- Keep `learned_segment_length_repair_fraction=0.6` in all summaries of the current candidate. It is material to the best-candidate trade-off.
- Next scientific checkpoint should target either selector/length allocation or material causality from the Checkpoint 4.74 candidate.

Current extra discoveries:
- The best candidate depends materially on `learned_segment_length_repair_fraction=0.6`; summaries must carry this knob because no-repair has stronger score causality but invalid global geometry.
- The score-protected length frontier in the best/no-repair artifacts only clears the `0.80` length gate while protecting about `10%` of budget for top learned-score points. At the guide's `25%` learned-slot materiality floor, the length upper bound is about `0.7911`, so the current selector/score distribution has a real learned-control-vs-length tension.
- `max_budget_share_per_ship` in `simplification/learned_segment_budget.py` is not a strict per-ship cap when the fair-share cap is larger; it is effectively `max(share_cap, fair_share_cap)`. Treat the name as misleading when reasoning about selector allocation caps.

Why this candidate is current best:
- Earlier route-density exclusion failed under the Checkpoint 3.x target/model state, so route density should not be treated as generically bad across all historical runs.
- Checkpoint 4.72 later isolated `route_density_prior` as the dominant harmful prior channel under the newer raw-factorized/head-initialized setup: zeroing only route density improved QueryUsefulV1 to `0.16718745914649327`, while other prior channels were neutral or slightly helpful.
- Checkpoint 4.73 made the narrow code change: keep `route_density_prior` in prior fields for support diagnostics, but zero it for v2 model features.
- Checkpoint 4.74 restored the strict-cell MLQDS win over Douglas-Peucker while keeping the standard workload/prior gates healthy.
- Checkpoint 4.83 showed the current length-repair path suppresses some score/causality upside, but removing it destroys global geometry and still does not pass learning causality. Therefore `learned_segment_length_repair_fraction=0.6` remains part of the best current candidate.
- The current problem is not workload health or generic prior harm. The remaining problem is making useful prior/behavior/score perturbations material enough in retained masks while preserving length.

Evidence boundary:
- A strict single-cell win is not a final success claim. Final acceptance still requires all strict single-cell gates plus the full 4x7 coverage/compression grid.
- Any future change must be judged against Checkpoint 4.74 unless it intentionally redefines the candidate baseline.
- Checkpoint 4.83 is useful evidence about the repair-vs-causality trade-off, but it does not replace Checkpoint 4.74 as the best candidate because its length is invalid.
- Raw training-fit improvements are not enough. Checkpoint 4.79 showed better fit diagnostics can still worsen retained-mask quality and lose the Douglas-Peucker comparison.
- Length-only improvements are not enough. Checkpoints 4.65, 4.66, and 4.81 improved length slightly or nearly cleared it but weakened MLQDS, learned control, or causality.
- A no-repair score win is not enough. Checkpoint 4.83 beat both baselines on QueryUsefulV1 but failed global sanity badly and still failed learning causality.

Rejected-path memory:

| Path | Best observed effect | Rejection reason |
|---|---:|---|
| no length repair, `learned_segment_length_repair_fraction=0.0` | MLQDS `0.1759846099523811`; learned-controlled slot fraction `0.8461538461538461` | length collapsed to `0.6790996203798462`; learning causality still failed |
| full length repair | length `0.7980194800294772` | learned-controlled slot fraction collapsed to `0.203125`; MLQDS lost to Douglas-Peucker |
| geometry gain `0.25` | length `0.797193150044111` | MLQDS regressed and causality worsened |
| full prior residual scale `1.0` after route removal | length `0.7939141083394758` | MLQDS `0.16109363670733973`, lost to Douglas-Peucker; shuffled-score causality failed by sign |
| semantic prior-to-head residual | improved training fit | retained-mask result worsened; MLQDS `0.16054051959902663`, lost to Douglas-Peucker; prior ablations became harmful |
| point-score blend `0.15` | length `0.7943720026689473` | MLQDS `0.1581758366351451`, lost to Douglas-Peucker; shuffled and untrained causality failed by sign |

Next-checkpoint guardrails:
- Prefer narrow changes that preserve Checkpoint 4.74's DP win and healthy workload/prior gates.
- For length work, preserve learned-controlled slots; do not spend the budget with query-free repair that crowds out learned selection.
- For causality work, focus on making prior/behavior/score perturbations move retained masks materially, not merely improving per-head fit.
- Score-protected length filling is a plausible diagnostic direction, but it must respect the observed frontier: protecting `25%` learned-score budget currently appears incompatible with the `0.80` length gate.
- Do not re-test blunt prior-strength escalation unless there is a new mechanism that explains why it will avoid the Checkpoint 4.76 and 4.79 failures.
- Do not add temporal scaffold or change acceptance thresholds to manufacture a success claim.

Minimum pass condition for the next scientific candidate update:
- Keep the Checkpoint 4.74 baseline comparable unless there is an explicit reason to reset the baseline.
- Preserve the MLQDS win over uniform and Douglas-Peucker on `QueryUsefulV1`.
- Clear `global_sanity_gate`, especially length preservation `>=0.80`.
- Clear `learning_causality_gate` with material deltas, not only correct signs.
- Keep `workload_stability`, `workload_signature`, `support_overlap`, `target_diffusion`, `predictability`, and `prior_predictive_alignment` passing.
- Report whether the change affects learned-controlled slot fraction, segment-budget-head delta, shuffled-score delta, no-prior delta, no-behavior-head delta, and length.

## Checkpoint 1 — Workload Generator And Profile Health

Status: completed

Goal:
- Make range workload generation healthy enough for standard strict single-cell probes.

Changes:
- Aligned `range_workload_v1` footprints and query-plan behavior with the guide.
- Added workload-signature checks for query-count mismatch.
- Added prefix-balanced profile-plan behavior so expanded workloads preserve family mix.

Tests:
- Focused workload/profile tests in `tests/test_query_driven_rework.py`
- `tests/test_query_coverage_generation.py`
- ruff/static checks on changed workload modules

Experiment artifact:
- path: `artifacts/results/query_driven_v2_checkpoint01_*`
- command: guide-aligned strict synthetic/debug single-cell probes

Key results:
- Early small probes exposed workload signature and query-count drift.
- Later strict cells generated healthy train/eval/selection workloads with accepted query counts above standard strict diagnostic minimum.

Decision:
- Continue from healthy strict synthetic cells; do not tune model from unhealthy workload evidence.

## Checkpoint 2 — Prior Predictability And Target Alignment

Status: completed

Goal:
- Make train-derived priors and QueryUsefulV1 targets measurable under healthy workloads.

Changes:
- Added prior predictability and target-diffusion diagnostics.
- Added support-overlap and prior-alignment gates.
- Added factorized QueryUsefulV1 target diagnostics.

Tests:
- Focused predictability, prior-field, target, and gate tests in `tests/test_query_driven_rework.py`

Experiment artifact:
- path: `artifacts/results/query_driven_v2_checkpoint02_*`

Key results:
- Workload, support, predictability, prior-alignment, target-diffusion, and signature gates pass in the retained strict cell.
- Remaining failures moved to model/selector causality and global sanity.

Decision:
- Continue to model and selector checkpoints.

## Checkpoint 3 — Factorized Model And Selector Diagnostics

Status: completed

Goal:
- Make the learned workload-blind model interpretable and causally diagnosable.

Changes:
- Added `workload_blind_range_v2`.
- Added factorized QueryUsefulV1 heads.
- Added `learned_segment_budget_v1`.
- Added frozen-mask protocol and causality ablations.
- Added selector trace diagnostics, learned-slot accounting, head ablation sensitivity, and length feasibility audits.

Tests:
- Full project pytest currently passes.
- Focused model/selector tests in `tests/test_query_driven_rework.py`

Experiment artifact:
- path: `artifacts/results/query_driven_v2_checkpoint03_*`

Key results:
- Early learned runs beat uniform inconsistently but failed causality and global sanity.
- Diagnostics showed retained-mask quality, segment allocation, and prior-path behavior needed targeted fixes.

Decision:
- Continue with targeted model/selector fixes only after gate-level diagnostics identify the component.

## Checkpoint 4.61 — Raw Factorized Scalar Target

Status: completed

Goal:
- Train factorized mode against the raw QueryUsefulV1 scalar target instead of legacy scaled labels.

Changes:
- Added raw scalar target handling for `query_useful_v1_factorized`.
- Kept legacy target scaling for legacy modes.

Tests:
- Focused target and factorized diagnostics tests.
- ruff, pyright, `git diff --check`.

Experiment artifact:
- path: not generated in this code checkpoint

Key results:
- Code checks passed.
- Strict replay still needed after the change.

Decision:
- Continue to strict replay.

## Checkpoint 4.62 — Raw Factorized Strict Replay

Status: completed; diagnostic failed

Goal:
- Test whether raw scalar targets fix factorized learning.

Experiment artifact:
- path: `artifacts/results/query_driven_v2_checkpoint04_raw_factorized_scalar_strict_probe_c10_r05`
- path: `artifacts/results/query_driven_v2_checkpoint04_raw_factorized_scalar_diagnostic/raw_factorized_scalar_summary.json`

Key results:
- MLQDS QueryUsefulV1: `0.16057549994768916`
- uniform QueryUsefulV1: `0.14223795796380634`
- Douglas-Peucker QueryUsefulV1: `0.16362459837911367`
- length: `0.7933048661024167`
- gates failed: learning causality, global sanity
- factorized heads were badly calibrated against low base-rate targets.

Decision:
- Fix factorized head initialization.

## Checkpoint 4.63 — Factorized Head Base-Rate Initialization

Status: completed

Goal:
- Initialize factorized output-head biases from empirical training target base rates.

Changes:
- Added factorized head output-bias initialization from target means.
- Added diagnostics and focused regression test.

Tests:
- Focused factorized-head tests.
- ruff, pyright, `git diff --check`.

Experiment artifact:
- path: not generated in this code checkpoint

Key results:
- Focused checks passed.

Decision:
- Continue to strict replay.

## Checkpoint 4.64 — Head-Bias Initialization Strict Replay

Status: completed; improved but still blocked

Goal:
- Test whether base-rate initialization fixes factorized calibration and causality.

Experiment artifact:
- path: `artifacts/results/query_driven_v2_checkpoint04_head_bias_init_strict_probe_c10_r05`
- path: `artifacts/results/query_driven_v2_checkpoint04_head_bias_init_diagnostic/head_bias_init_summary.json`

Key results:
- MLQDS QueryUsefulV1: `0.16512927110095915`
- uniform QueryUsefulV1: `0.14223795796380634`
- Douglas-Peucker QueryUsefulV1: `0.16362459837911367`
- length: `0.7931550386328327`
- first strict-cell MLQDS QueryUsefulV1 win over Douglas-Peucker in this sequence
- gates failed: learning causality, global sanity
- prior-feature removal slightly improved score, suggesting harmful prior integration.

Decision:
- Keep head-bias initialization.
- Diagnose global sanity and prior causality separately.

## Checkpoint 4.65 — Full Length Repair Diagnostic

Status: completed; diagnostic failed

Goal:
- Test whether full existing length repair clears the length gate.

Experiment artifact:
- path: `artifacts/results/query_driven_v2_checkpoint04_full_length_repair_strict_probe_c10_r05`
- path: `artifacts/results/query_driven_v2_checkpoint04_full_length_repair_diagnostic/full_length_repair_summary.json`

Key results:
- MLQDS QueryUsefulV1: `0.16243558593475863`
- length: `0.7980194800294772`
- learned-controlled slot fraction dropped to `0.203125`
- gates failed: learning causality, global sanity

Decision:
- Reject full repair. It weakens learned control and loses to Douglas-Peucker.

## Checkpoint 4.66 — Higher Geometry-Gain Diagnostic

Status: completed; diagnostic failed

Goal:
- Test whether stronger geometry gain clears length without collapsing learned slots.

Experiment artifact:
- path: `artifacts/results/query_driven_v2_checkpoint04_geometry_gain025_strict_probe_c10_r05`
- path: `artifacts/results/query_driven_v2_checkpoint04_geometry_gain025_diagnostic/geometry_gain025_summary.json`

Key results:
- MLQDS QueryUsefulV1: `0.16022420264941584`
- length: `0.797193150044111`
- learned-controlled slot fraction stayed healthy
- causality worsened

Decision:
- Reject geometry gain `0.25`.

## Checkpoint 4.67 — Query-Prior Branch Initialization

Status: completed

Goal:
- Fix the near-zero prior branch output initialization.

Changes:
- Changed prior output initialization from `std=1e-3` to Xavier.
- Added focused prior-branch tests.

Tests:
- Focused prior-branch tests.
- ruff, pyright, `git diff --check`.

Experiment artifact:
- path: not generated in this code checkpoint

Key results:
- Focused checks passed.

Decision:
- Continue to strict replay.

## Checkpoint 4.68 — Prior-Init Strict Replay

Status: completed; diagnostic failed

Goal:
- Test whether stronger prior initialization makes prior features causally useful.

Experiment artifact:
- path: `artifacts/results/query_driven_v2_checkpoint04_prior_init_strict_probe_c10_r05`
- path: `artifacts/results/query_driven_v2_checkpoint04_prior_init_diagnostic/prior_init_summary.json`

Key results:
- MLQDS QueryUsefulV1: `0.14892550519596737`
- uniform QueryUsefulV1: `0.14223795796380634`
- Douglas-Peucker QueryUsefulV1: `0.16362459837911367`
- length: `0.7872785562747836`
- prior features became influential but harmful.

Decision:
- Reject full-strength prior path.

## Checkpoint 4.69 — Bounded Prior Residual Scale

Status: completed

Goal:
- Bound the prior residual scale without returning to near-zero suppression.

Changes:
- Set prior residual scale initialization/reset to `0.25`.
- Updated focused tests.

Tests:
- Focused prior-scale tests.
- ruff, pyright, `git diff --check`.

Experiment artifact:
- path: not generated in this code checkpoint

Key results:
- Focused checks passed.

Decision:
- Continue to strict replay.

## Checkpoint 4.70 — Bounded-Prior Strict Replay

Status: completed; diagnostic failed

Goal:
- Test whether bounded prior scale recovers useful prior sensitivity.

Experiment artifact:
- path: `artifacts/results/query_driven_v2_checkpoint04_bounded_prior_strict_probe_c10_r05`
- path: `artifacts/results/query_driven_v2_checkpoint04_bounded_prior_diagnostic/bounded_prior_summary.json`

Key results:
- MLQDS QueryUsefulV1: `0.16008457877061275`
- length: `0.7936343750367146`
- prior ablations still improved score
- lost to Douglas-Peucker

Decision:
- Stop scale guessing. Diagnose prior channels.

## Checkpoint 4.71 — Per-Channel Prior Ablation Diagnostics

Status: completed

Goal:
- Add optional per-channel prior ablation diagnostics.

Changes:
- Added `zero_query_prior_field_channels`.
- Added optional per-channel prior ablation diagnostics under `learning_causality_summary.prior_channel_ablation_diagnostics`.
- Added focused tests.

Tests:
- Focused prior-channel tests.
- ruff, pyright, `git diff --check`.

Experiment artifact:
- path: not generated in this code checkpoint

Key results:
- Focused checks passed.

Decision:
- Continue to strict diagnostic replay.

## Checkpoint 4.72 — Prior-Channel Diagnostic Replay

Status: completed; diagnostic succeeded

Goal:
- Identify which prior channel causes harmful prior behavior.

Experiment artifact:
- path: `artifacts/results/query_driven_v2_checkpoint04_prior_channel_diag_strict_probe_c10_r05`
- path: `artifacts/results/query_driven_v2_checkpoint04_prior_channel_diag_diagnostic/prior_channel_summary.json`

Key results:
- Base MLQDS QueryUsefulV1: `0.16008457877061275`
- zeroing `route_density_prior` alone improved QueryUsefulV1 to `0.16718745914649327`
- other prior channels were neutral or slightly helpful

Decision:
- Remove `route_density_prior` from v2 model inputs while keeping it available for support diagnostics.

## Checkpoint 4.73 — Exclude Route Density From V2 Model Input

Status: completed

Goal:
- Exclude the harmful route-density channel from v2 model features.

Changes:
- Added `WORKLOAD_BLIND_RANGE_V2_MODEL_DISABLED_PRIOR_FIELDS = ("route_density_prior",)`.
- Zeroed disabled prior channels in v2 feature construction.
- Bumped v2 schema to `6`.
- Added focused test proving route density remains in prior sampling but not v2 model features.

Tests:
- Focused route-density exclusion tests.
- ruff, pyright, `git diff --check`.

Experiment artifact:
- path: not generated in this code checkpoint

Key results:
- Focused checks passed.

Decision:
- Continue to strict replay.

## Checkpoint 4.74 — No-Route-Density Strict Replay

Status: completed; best current candidate

Goal:
- Test whether route-density exclusion fixes prior causality and restores the DP comparison.

Experiment artifact:
- path: `artifacts/results/query_driven_v2_checkpoint04_no_route_density_strict_probe_c10_r05`
- path: `artifacts/results/query_driven_v2_checkpoint04_no_route_density_diagnostic/no_route_density_summary.json`

Key results:
- MLQDS QueryUsefulV1: `0.1669032451715525`
- uniform QueryUsefulV1: `0.14223795796380634`
- Douglas-Peucker QueryUsefulV1: `0.16362459837911367`
- length: `0.7938149625265364`
- prior/no-head deltas became positive but remained below material thresholds
- gates failed: learning causality, global sanity

Decision:
- Keep route-density exclusion.
- Continue from this candidate.

## Checkpoint 4.75 — Restore Prior Scale After Route Removal

Status: completed

Goal:
- Test a full prior residual scale after removing route density.

Changes:
- Temporarily set prior residual scale back to `1.0`.
- Updated focused tests.

Tests:
- Focused prior-scale tests.
- ruff, pyright, `git diff --check`.

Experiment artifact:
- path: not generated in this code checkpoint

Key results:
- Focused checks passed.

Decision:
- Continue to strict replay.

## Checkpoint 4.76 — No-Route-Density Scale-1 Replay

Status: completed; diagnostic failed

Goal:
- Test whether full prior scale works after route-density removal.

Experiment artifact:
- path: `artifacts/results/query_driven_v2_checkpoint04_no_route_density_scale1_strict_probe_c10_r05`
- path: `artifacts/results/query_driven_v2_checkpoint04_no_route_density_scale1_diagnostic/no_route_density_scale1_summary.json`

Key results:
- MLQDS QueryUsefulV1: `0.16109363670733973`
- lost to Douglas-Peucker
- shuffled-score causality failed by sign
- length: `0.7939141083394758`

Decision:
- Reject scale `1.0`.

## Checkpoint 4.77 — Revert Failed Prior Scale

Status: completed

Goal:
- Restore the best current code candidate after failed scale test.

Changes:
- Reverted prior scale to `0.25`.
- Reverted schema to `6`.

Tests:
- Focused prior-scale and route-density tests.
- ruff, pyright, `git diff --check`.

Experiment artifact:
- path: not generated in this code checkpoint

Key results:
- Focused checks passed.

Decision:
- Continue from Checkpoint 4.74.

## Checkpoint 4.78 — Semantic Prior-To-Head Residual

Status: completed

Goal:
- Test a direct interpretable prior-to-head residual.

Changes:
- Temporarily added semantic direct prior-to-head residuals.
- Kept route density at zero influence.

Tests:
- Focused prior-branch tests.
- ruff, pyright, `git diff --check`.

Experiment artifact:
- path: not generated in this code checkpoint

Key results:
- Focused checks passed.

Decision:
- Continue to strict replay.

## Checkpoint 4.79 — Semantic Prior Residual Strict Replay

Status: completed; diagnostic failed

Goal:
- Test whether semantic prior residuals make causality material.

Experiment artifact:
- path: `artifacts/results/query_driven_v2_checkpoint04_semantic_prior_residual_strict_probe_c10_r05`
- path: `artifacts/results/query_driven_v2_checkpoint04_semantic_prior_residual_diagnostic/semantic_prior_residual_summary.json`

Key results:
- MLQDS QueryUsefulV1: `0.16054051959902663`
- lost to Douglas-Peucker
- training fit improved, but retained-mask result worsened
- prior ablations became harmful again

Decision:
- Reject semantic prior residuals.

## Checkpoint 4.80 — Revert Semantic Residual

Status: completed

Goal:
- Remove the failed semantic residual path.

Changes:
- Removed direct prior-head residual code.
- Restored schema `6`.

Tests:
- Focused prior/route-density tests.
- ruff, pyright, `git diff --check`.

Experiment artifact:
- path: not generated in this code checkpoint

Key results:
- Focused checks passed.

Decision:
- Continue from Checkpoint 4.74.

## Checkpoint 4.81 — Higher Point-Score Blend Diagnostic

Status: completed; diagnostic failed

Goal:
- Test whether point scores are underweighted inside learned segments.

Experiment artifact:
- path: `artifacts/results/query_driven_v2_checkpoint04_score_blend015_strict_probe_c10_r05`
- path: `artifacts/results/query_driven_v2_checkpoint04_score_blend015_diagnostic/score_blend015_summary.json`

Key results:
- MLQDS QueryUsefulV1: `0.1581758366351451`
- lost to Douglas-Peucker
- length: `0.7943720026689473`
- shuffled and untrained causality failed by sign

Decision:
- Reject `learned_segment_score_blend_weight=0.15`.
- Continue from Checkpoint 4.74.

## Checkpoint 4.82 — Commit-Prep Cleanup

Status: completed

Goal:
- Prepare the current codebase work for a checkpointed git save.

Changes:
- Condensed this progress log from a long raw checkpoint journal into a short guide-compliant ledger.
- Verified rejected scale-1, semantic-residual, and score-blend escalation paths are not active production code.
- Kept the best current candidate active: route-density excluded from v2 model inputs, prior scale `0.25`, no direct prior-head residual.

Tests:
- `../.venv/bin/ruff check evaluation/baselines.py experiments/benchmark_report.py experiments/experiment_cli.py experiments/experiment_config.py experiments/experiment_data.py experiments/experiment_methods.py experiments/experiment_pipeline.py experiments/range_diagnostics.py experiments/run_ais_experiment.py experiments/run_inference.py models/workload_blind_range_v2.py queries/query_generator.py queries/workload_profiles.py simplification/learned_segment_budget.py simplification/mlqds_scoring.py tests/test_benchmark_runner.py tests/test_experiment_data.py tests/test_query_coverage_generation.py tests/test_query_driven_rework.py tests/test_torch_runtime_controls.py tests/test_training_does_not_collapse.py training/checkpoints.py training/model_features.py training/predictability_audit.py training/query_prior_fields.py training/query_useful_targets.py training/train_model.py training/training_epoch.py training/training_validation.py`
- `../.venv/bin/python -m pyright evaluation/baselines.py experiments/benchmark_report.py experiments/experiment_cli.py experiments/experiment_config.py experiments/experiment_data.py experiments/experiment_methods.py experiments/experiment_pipeline.py experiments/range_diagnostics.py experiments/run_ais_experiment.py experiments/run_inference.py models/workload_blind_range_v2.py queries/query_generator.py queries/workload_profiles.py simplification/learned_segment_budget.py simplification/mlqds_scoring.py tests/test_benchmark_runner.py tests/test_experiment_data.py tests/test_query_coverage_generation.py tests/test_query_driven_rework.py tests/test_torch_runtime_controls.py tests/test_training_does_not_collapse.py training/checkpoints.py training/model_features.py training/predictability_audit.py training/query_prior_fields.py training/query_useful_targets.py training/train_model.py training/training_epoch.py training/training_validation.py`
- `git diff --check`
- `../.venv/bin/python -m pytest tests/test_query_driven_rework.py`
- `../.venv/bin/python -m pytest tests/test_training_does_not_collapse.py tests/test_experiment_data.py tests/test_query_coverage_generation.py`
- `../.venv/bin/python -m pytest tests/test_benchmark_runner.py tests/test_torch_runtime_controls.py`
- `../.venv/bin/python -m pytest`

Experiment artifact:
- path: not generated in this checkpoint
- command: no scientific probe was run; this was a cleanup and verification checkpoint.

Key results:
- ruff passed.
- pyright passed.
- `git diff --check` passed.
- Focused pytest batches passed.
- Full pytest passed: `408 passed, 1 warning`.

Decision:
- Codebase is ready for a checkpoint commit.
- Remaining rework blockers after the save are learning-causality materiality and length preservation.

## Checkpoint 4.83 — No-Length-Repair Causality Diagnostic

Status: completed; diagnostic failed

Goal:
- Test whether the current query-free length-repair swaps are suppressing material learned-score causality.

Changes:
- No code changes.
- Removed one aborted non-comparable artifact before rerunning with the same split geometry as Checkpoint 4.74.

Tests:
- Not run; this checkpoint was experiment-only.

Experiment artifact:
- path: `artifacts/results/query_driven_v2_checkpoint04_no_length_repair_causality_diag_c10_r05`
- command: strict synthetic single-cell matching Checkpoint 4.74 scale/seed/workload, with `learned_segment_length_repair_fraction=0.0`.

Key results:
- MLQDS QueryUsefulV1: `0.1759846099523811`
- uniform QueryUsefulV1: `0.14223795796380634`
- Douglas-Peucker QueryUsefulV1: `0.16362459837911367`
- length: `0.6790996203798462`
- learned-controlled slot fraction: `0.8461538461538461`
- gates passed: workload stability, support overlap, predictability, prior-predictive alignment, target diffusion, workload signature
- gates failed: learning causality, global sanity
- causality passed for shuffled scores, untrained model, prior-field-only, and segment-budget-head ablation.
- causality failed for shuffled prior fields, no query-prior features, and no behavior head.

Decision:
- Reject no-repair as a candidate: it destroys global sanity and still does not pass learning causality.
- Do not increase workload/caps for this blocker; the strict cell already has healthy workload scale.
- Next checkpoint should target either score-protected length filling as a query-free selector diagnostic, or a model/prior-path change that makes train-derived priors materially affect retained masks without reintroducing harmful route density.

## Checkpoint 4.84 — Discovery Log Hygiene

Status: completed

Goal:
- Make sure relevant extra discoveries are preserved in log and summary outputs.

Changes:
- Added a current extra-discoveries section near the top of this log.
- Promoted the material length-repair knob, score-protected length frontier conflict, and per-ship-cap naming issue into durable notes.

Tests:
- `git diff --check`

Experiment artifact:
- path: not generated
- command: no probe was run; this was documentation hygiene.

Key results:
- Relevant extra discoveries are now recorded in this log instead of only in chat summaries.

Decision:
- Continue future checkpoints from the Checkpoint 4.74 candidate and keep extra discoveries in both progress-log updates and final summaries.

## Checkpoint 4.85 — Developer Tooling

Status: partial

Goal:
- Implement the tooling guide without touching scientific model, selector, or generator behavior.
- Migrate active commands to `uv --group dev`.
- Add jq filters, property tests, regression snapshots, Rich summaries, and yamllint.

Changes:
- Reworked root and `Range_QDS` Makefiles around `uv sync --group dev` and `uv run --group dev -- ...`.
- Updated active README and experiment command examples away from `.venv/bin/python` and pip install flows.
- Migrated benchmark preflight/tmux launchers from `PYTHON` executable paths to `UV` and `UV_GROUP`.
- Added jq artifact filters under `scripts/jq/`.
- Added `scripts/summarize_run.py` Rich run summary.
- Added Hypothesis property tests for workload-profile plans, zero-prior fields, and learned-segment selector budget accounting.
- Added pytest-regressions snapshots for final-grid summary, benchmark row fields, and gate summary shape.
- Added `yamllint==1.38.0`, `.yamllint`, and `make lint-yaml`.
- Added pytest markers for `property` and `regression`.
- Suppressed Pyright `reportPrivateImportUsage` to remove Torch-stub false positives and make the configured typecheck usable.

Tests:
- `uv sync --group dev`
- `uv lock --check`
- `git diff --check`
- `uv run --group dev -- yamllint .`
- `uv run --group dev -- pyright Range_QDS/data Range_QDS/evaluation Range_QDS/experiments Range_QDS/models Range_QDS/queries Range_QDS/simplification Range_QDS/training Range_QDS/scripts Range_QDS/tests`
- `uv run --group dev -- ruff check Range_QDS/scripts/summarize_run.py Range_QDS/tests/property Range_QDS/tests/regression Range_QDS/experiments/run_inference.py`
- `uv run --group dev -- pytest Range_QDS/tests/property Range_QDS/tests/regression -q`
- `uv run --group dev -- pytest Range_QDS/tests/test_query_driven_rework.py Range_QDS/tests/test_benchmark_runner.py Range_QDS/tests/property Range_QDS/tests/regression -q`
- `uv run --group dev -- pytest Range_QDS/tests -q`
- `bash -n Range_QDS/scripts/benchmark_preflight.sh Range_QDS/scripts/run_range_benchmark_tmux.sh Range_QDS/scripts/run_benchmark_queue_tmux.sh`

Experiment artifact:
- path: not generated
- command: no scientific probe was run; this was tooling-only.

Key results:
- Full pytest passed: `415 passed, 1 warning`.
- Full Pyright passed after removing Torch-stub private-export noise.
- yamllint passed.
- jq filters parse.
- Full `ruff check Range_QDS` still does not pass: `195` pre-existing lint findings remain outside this tooling patch.

Extra discoveries:
- Active `experiments/README.md` and `experiments/run_inference.py` still had stale `.venv/bin/python` examples; fixed.
- Default yamllint indentation does not fit pytest-regressions generated snapshot YAML, so those generated snapshots are excluded from YAML linting.
- The full Ruff gate is not yet a reliable project-wide save gate until the existing lint debt is either fixed or intentionally scoped.

Decision:
- Tooling is implemented and usable.
- Treat the checkpoint as partial because the guide's full Ruff check remains blocked by existing lint debt.
- Continue scientific iterations only after the user decides whether to commit this tooling checkpoint with the documented Ruff debt or spend a separate cleanup checkpoint on project-wide Ruff.

## Checkpoint 4.86 — Documentation Cleanup

Status: completed

Goal:
- Remove or update clearly stale Range_QDS documentation.
- Deduplicate active docs and condense long prose so high-value guidance is easier to find.

Changes:
- Condensed `docs/dev-tooling-guide.md` from rollout essay to compact operating reference.
- Condensed `experiments/README.md` and `training/README.md` to active commands, active profiles, final-candidate settings, and current mode classifications.
- Updated stale statements that described `QueryUsefulV1`, `workload_profile_id`, and `query_useful_v1_factorized` as future/unimplemented.
- Updated model, query, evaluation, simplification, and code-layout READMEs for the current workload-blind v2 path.
- Added clearer warnings that the tmux benchmark Makefile defaults still point at legacy diagnostic artifact families unless profile/family/cache variables are overridden.

Tests:
- `git diff --check`
- `uv run --group dev -- yamllint .`
- stale-doc grep for old active command styles and known obsolete placeholder phrases

Experiment artifact:
- path: not generated
- command: no scientific probe was run; this was documentation-only.

Key results:
- Active non-historical docs no longer claim key rework components are unimplemented placeholders.
- Markdown line count dropped from about `5146` to `3479` lines.
- Remaining `.venv` references are historical entries in this progress log, not active instructions.

Extra discoveries:
- `Range_QDS/Makefile` still defaults benchmark profile/family/cache variables to legacy diagnostic paths. The docs now warn about this, but a future tooling cleanup should consider changing defaults or adding explicit query-driven benchmark targets.
- The canonical rework guide remains intentionally long because it is still the source of truth for protocol gates and evidence levels; this checkpoint avoided rewriting acceptance criteria.

Decision:
- Documentation is clean enough for the checkpoint save.
- Continue scientific iterations from the current candidate after committing the tooling/docs cleanup.

## Checkpoint 4.87 — Tooling Guide Conceptual Restoration

Status: completed

Goal:
- Restore durable developer-tooling principles that were over-condensed from `docs/dev-tooling-guide.md`.
- Keep rollout prose removed while preserving conceptual usage rules for Hypothesis, pytest-regressions, and tooling risks.

Changes:
- Restored tooling principles around invariant enforcement, uv command consistency, noisy experiment metrics, hot-path isolation, and small readable checks.
- Added compact Hypothesis good targets, good properties, bad uses, and default settings guidance.
- Added compact pytest-regressions good uses, bad uses, snapshot update policy, and schema-protection purpose.
- Added concise tooling risks: uv drift, dependency syntax drift, lockfile drift, jq-as-acceptance, flaky property tests, noisy snapshots, Rich replacing JSON, and tooling distraction.

Tests:
- `git diff --check`
- `uv run --group dev -- yamllint .`

Experiment artifact:
- path: not generated
- command: no scientific probe was run; this was documentation-only.

Key results:
- `docs/dev-tooling-guide.md` remains compact at about `250` lines instead of reverting to the old rollout-length guide.
- Durable conceptual guidance is back in active docs.

Decision:
- Documentation correction is complete.
- Continue from the documentation/tooling checkpoint state.

## Checkpoint 4.88 — Code Cleanup

Status: completed

Goal:
- Remove clearly stale or unused compatibility code from active production paths.
- Improve misleading names where the current meaning is clear and covered by tests.
- Keep intentional diagnostic legacy paths that still have a real use case.

Changes:
- Removed unused compatibility modules: `training/training_pipeline.py`, `simplification/selector_diagnostics.py`, and `simplification/legacy_temporal_hybrid.py`.
- Removed the unused `training.query_useful_targets.build` wrapper; active code imports `build_query_useful_v1_targets` directly.
- Dropped unimplemented benchmark-profile stubs from `PROFILE_CHOICES` so CLIs no longer advertise profiles that immediately fail.
- Renamed historical-prior route-context feature constants/functions away from misleading `legacy`/`old` wording.
- Renamed benchmark-profile settings from `profile_legacy_diagnostic` / `legacy_reason` to `profile_diagnostic_only` / `profile_note`.
- Renamed the learning-causality artifact flag from `legacy_temporal_hybrid_selector` to `selector_final_candidate`.
- Changed missing range-query metadata family counts from `legacy_or_unspecified` to `unspecified`.

Tests:
- `git diff --check`
- `uv run --group dev -- ruff check --select F401,F821,F822,F823 ...` on edited Python files
- `uv run --group dev -- pyright ...` on edited production modules
- `uv run --group dev -- pyright Range_QDS/data Range_QDS/evaluation Range_QDS/experiments Range_QDS/models Range_QDS/queries Range_QDS/simplification Range_QDS/training Range_QDS/scripts Range_QDS/tests`
- `uv run --group dev -- pytest Range_QDS/tests/test_model_features.py Range_QDS/tests/test_pre_rework_cleanup.py Range_QDS/tests/test_benchmark_runner.py -q`
- `uv run --group dev -- pytest Range_QDS/tests/test_query_driven_rework.py -q`
- `uv run --group dev -- pytest Range_QDS/tests -q`

Experiment artifact:
- path: not generated
- command: no scientific probe was run; this was code cleanup only.

Key results:
- Full pytest passed: `415 passed, 1 warning`.
- Full Pyright passed.
- No deleted module had in-repository imports.
- Broad Ruff on the edited large files still hits existing project lint debt; focused undefined/unused checks passed.

Extra discoveries:
- `workload_blind_range_v2.calibration_head` is still retained only for checkpoint-state compatibility and is frozen/unused in final score composition. It may be removable later, but doing so needs an explicit checkpoint-loading policy decision rather than a cleanup guess.
- The benchmark/runtime Makefile defaults still point at legacy diagnostic profiles; this checkpoint cleaned profile definitions but did not change run defaults.
- Intentional legacy diagnostics remain: `RangeUsefulLegacy`, legacy generator profiles, and non-final scalar-target modes. They are still used for comparability and guardrail tests, so deleting them would be wrong right now.

Decision:
- Code cleanup is safe to save.
- Continue scientific iterations from the existing candidate; this checkpoint does not change model evidence or gate status.

## Checkpoint 4.89 — Test Cleanup and Coverage

Status: completed

Goal:
- Remove or update stale, outdated, or misleading test logic.
- Identify important behavior coverage gaps in the current test suite and add focused tests where the gap is concrete.

Changes:
- Renamed `tests/test_pre_rework_cleanup.py` to `tests/test_rework_guardrails.py` and updated its stale pre-rework module description.
- Renamed the v2 checkpoint compatibility test from a vague legacy-prior name to `test_workload_blind_range_v2_checkpoint_accepts_missing_prior_feature_encoder`.
- Added guardrails that removed compatibility shims stay removed and that the removed `query_useful_targets.build` alias does not return.
- Added a profile-choice guardrail: every advertised benchmark profile must be implemented and loadable.
- Added assertions that profile settings use current `profile_diagnostic_only` / `profile_note` keys instead of stale `profile_legacy_diagnostic` / `legacy_reason` keys.
- Added coverage that missing range workload family metadata is counted as `unspecified`.
- Added pipeline-smoke coverage for the renamed `learning_causality_summary.selector_final_candidate` key and absence of the stale `legacy_temporal_hybrid_selector` key.

Tests:
- `git diff --check`
- `uv run --group dev -- ruff check --select F401,F821,F822,F823 Range_QDS/tests/test_beats_random_in_distribution.py Range_QDS/tests/test_rework_guardrails.py Range_QDS/tests/test_model_features.py Range_QDS/tests/test_query_coverage_generation.py`
- `uv run --group dev -- pyright Range_QDS/tests/test_rework_guardrails.py Range_QDS/tests/test_model_features.py Range_QDS/tests/test_query_coverage_generation.py`
- `uv run --group dev -- pyright Range_QDS/data Range_QDS/evaluation Range_QDS/experiments Range_QDS/models Range_QDS/queries Range_QDS/simplification Range_QDS/training Range_QDS/scripts Range_QDS/tests`
- `uv run --group dev -- pytest Range_QDS/tests/test_rework_guardrails.py Range_QDS/tests/test_model_features.py Range_QDS/tests/test_query_coverage_generation.py -q`
- `uv run --group dev -- pytest Range_QDS/tests/test_beats_random_in_distribution.py::test_pipeline_reports_f1_scores -q`
- `uv run --group dev -- pytest Range_QDS/tests -q`

Experiment artifact:
- path: not generated
- command: no scientific probe was run; this was tests-only.

Key results:
- Full pytest passed: `421 passed, 1 warning`.
- Full Pyright passed.
- Focused undefined/unused Ruff checks passed.
- The test suite now covers the main cleanup outcomes from Checkpoint 4.88 instead of only relying on grep/manual review.

Extra discoveries:
- Remaining `legacy` references in tests are mostly intentional comparability or diagnostic guardrails: `RangeUsefulLegacy`, legacy generator behavior, scalar-target separation, and checkpoint backward-loading tests.
- The suite already has broad coverage for workload gates, protocol flags, benchmark row guardrails, and selector learned-slot accounting. The concrete missing coverage was around stale cleanup regressions and renamed artifact/profile keys, which this checkpoint added.
- Full Ruff remains unsuitable as a project-wide test cleanup gate until existing lint debt is addressed; focused correctness selectors are still the practical save gate.

Decision:
- Test cleanup is safe to save.
- Continue scientific iterations from the existing candidate; this checkpoint does not change model evidence or gate status.

## Checkpoint 4.90 — Code Organization Audit

Status: completed

Goal:
- Identify structural and modularization improvements that would make `Range_QDS` easier to reason about from the top down.
- Avoid speculative behavior-changing refactors while the scientific candidate is still unresolved.

Changes:
- Expanded `CODE_LAYOUT.md` from a terse directory list into a top-down architecture map.
- Added package ownership boundaries and "should not own" guidance.
- Recorded the current layering exception where `training.train_model` imports experiment-owned config/runtime helpers.
- Recorded concrete modularization pressure points and recommended split order.
- Added refactor rules for future structure work.

Tests:
- `git diff --check`
- No Python tests were run; this checkpoint changed documentation only.

Experiment artifact:
- path: not generated
- command: no scientific probe was run; this was structure-audit documentation only.

Key results:
- Biggest maintainability pressure points by approximate line count:
  - `experiments/experiment_pipeline.py`: `4965` lines
  - `training/training_targets.py`: `3090` lines
  - `experiments/benchmark_report.py`: `1957` lines
  - `simplification/learned_segment_budget.py`: `1485` lines
  - `queries/query_generator.py`: `1280` lines
- The highest-value first extraction is `experiments/gates.py`: support overlap, workload stability, target diffusion, and global sanity gates are pure enough to move later and are already heavily tested.
- A full split of `experiment_pipeline.py` is not safe as a drive-by cleanup because its private helpers are imported by tests and several helpers are tied to artifact schemas.

Extra discoveries:
- `training/` currently depends upward on `experiments/` through `ModelConfig` and torch runtime helpers. This is an architectural smell. The right fix is a neutral config/runtime package, not more experiment imports from lower layers.
- `training_targets.py` mixes old scalar diagnostic target families with newer query-driven/factorized target paths. That is intentional historically, but it is a readability cost and should be split by target family after scientific behavior stabilizes.
- Future refactors should preserve artifact field names unless the checkpoint explicitly changes the schema; report and gate fields are part of the debugging protocol.

Decision:
- Do not perform a broad module split now.
- Use the documented extraction order for future cleanup: gates first, then causality diagnostics, segment audits, benchmark row/report helpers, target-family splits, selector allocation/repair splits, and query generator planning/acceptance splits.
