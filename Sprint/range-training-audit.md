# Workload-Blind Range-QDS Completion Audit

This audit checks the active redesign goal against the current implementation
and benchmark artifacts. It is not a success report. The protocol is largely
implemented, but the final model-quality bar is not met.

## Objective Restated

Concrete deliverables:

1. Implement a workload-blind Range-QDS path that trains from historical or
   generated range workloads, then compresses validation/eval trajectories
   before future eval queries are scored.
2. Freeze retained masks before held-out eval queries affect scoring,
   checkpoint selection, feature building, or retained-set decisions.
3. Evaluate the same blind model across coverage targets `5%,10%,15%,30%` and
   compression targets `1%,2%,5%,10%,15%,20%,30%`.
4. Report `RangeUseful` and all required component, geometry, length,
   runtime, and latency metrics.
5. Show the blind model beats uniform temporal sampling and Douglas-Peucker on
   `RangeUseful` across most or all grid cells, especially `1%,2%,5%`.
6. Show generalization across AIS days and held-out workload-generator
   seeds/settings.
7. Keep `range_aware` as a diagnostic/teacher only, not as final blind
   evidence.
8. If the model fails, diagnose by component metrics, label mass,
   teacher/student fit, workload generation, and runtime bottlenecks.

## Prompt-To-Artifact Checklist

| Requirement | Evidence inspected | Status |
| --- | --- | --- |
| Start from `Sprint/range-training-redesign.md` | Design file defines the hard rule, grid, metrics, model direction, teacher/student constraints, and runtime notes. Progress log tracks experiments against it. | Met |
| Eval protocol freezes masks before held-out eval query scoring | Best-branch artifacts record `workload_blind_protocol.enabled=True`, `primary_masks_frozen_before_eval_query_scoring=True`, `audit_masks_frozen_before_eval_query_scoring=True`, `eval_geometry_blend_allowed=False`. Code wraps frozen masks before `eval-query-cache-prep`. | Met |
| Eval queries are not passed into blind compression feature builder/model | `historical_prior` is marked workload blind; blind evaluation uses query-free features. Tests cover NaN query features for blind simplify/validation. | Met |
| Coverage targets `5%,10%,15%,30%` evaluated | Artifacts: `multitrain0205_c05...`, `c10...`, `c15...`, `c30...scorecache...`. | Met |
| Compression targets `1%,2%,5%,10%,15%,20%,30%` evaluated | Each coverage artifact contains `range_compression_audit` for the full grid. | Met |
| Required metrics reported | Benchmark rows and `example_run.json` report `RangePointF1`, `ShipF1`, `ShipCov`, `EntryExitF1`, `CrossingF1`, `TemporalCov`, `GapCov`, `GapCovTime`, `GapCovDistance`, `TurnCov`, `ShapeScore`, SED/PED geometry, length preservation, runtime, latency. Benchmark rows also flatten per-compression `RangeUseful` audit cells and deltas for MLQDS, uniform, Douglas-Peucker, and TemporalRandomFill. | Met |
| Beats uniform across most/all target cells, especially low compression | Current best wins `17/28` cells vs uniform and only `4/12` low-budget cells (`1%,2%,5%`). `1%` and `2%` are temporal ties, not learned wins. | Failed |
| Beats Douglas-Peucker across grid | Current best wins `28/28` cells vs DP on the main coverage grid. | Met |
| Wins come from learned behavior, not temporal scaffold | Current best uses `local_swap` with `mlqds_temporal_fraction=0.85`; low-budget `1%` and `2%` cells are protected temporal ties. Clean stratified and lower-scaffold variants fail. | Failed |
| Sensible retained trajectories | Retained CSVs are written; length preservation is near uniform or slightly better, but global SED/PED distortion remains worse than uniform. | Partial |
| Generalizes across AIS days | Main split trains `2026-02-02..05`, validates `2026-02-06`, evaluates `2026-02-07`. It produces matched-cell gains, but robustness is not strong enough. | Partial |
| Generalizes across held-out workload seeds/settings | Seed-47 mixed-density works at `5/7` audit wins vs uniform; seed-47 dense eval drops to `2/7`. | Failed |
| `range_aware` used only as diagnostic/teacher | Teacher-distillation artifact is labeled diagnostic and is not used as final success. | Met |
| Failure diagnosed clearly | Progress log and this audit cover components, label mass, teacher fit, workload-generation coverage, selector behavior, runtime, and latency. | Met |
| Tests and hygiene | Full suite `../.venv/bin/python -m pytest -q tests` -> `260 passed, 1 warning`; `git diff --check` passed. | Met |

## Current Best Blind Branch

Configuration:

- `model_type=historical_prior`
- train days `2026-02-02..05`, validation `2026-02-06`, eval `2026-02-07`
- `range_label_mode=usefulness_ship_balanced`
- `range_train_workload_replicates=4`
- `range_replicate_target_aggregation=label_mean`
- `mlqds_hybrid_mode=local_swap`
- `mlqds_temporal_fraction=0.85`
- `mlqds_diversity_bonus=0.02`

Artifacts:

- `artifacts/manual/multitrain0205_c05_historicalprior_shipbalanced_localswap085_20260515`
- `artifacts/manual/multitrain0205_c10_historicalprior_shipbalanced_localswap085_20260515`
- `artifacts/manual/multitrain0205_c15_historicalprior_shipbalanced_localswap085_20260515`
- `artifacts/manual/multitrain0205_c30_historicalprior_shipbalanced_localswap085_flatpredict_20260515`

Matched `5%` compression:

| Coverage target | Actual train/eval coverage | MLQDS | Uniform | DP | TemporalRandomFill | Wins vs uniform | Low-budget wins | Wins vs DP | Latency |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `5%` | `5.23% / 5.89%` | `0.304912` | `0.300945` | `0.221774` | `0.270232` | `3/7` | `1/3` | `7/7` | `870.10ms` |
| `10%` | `10.42% / 10.03%` | `0.289699` | `0.267976` | `0.203119` | `0.263826` | `5/7` | `1/3` | `7/7` | `902.36ms` |
| `15%` | `15.04% / 15.35%` | `0.263460` | `0.244701` | `0.188116` | `0.242981` | `5/7` | `1/3` | `7/7` | `868.52ms` |
| `30%` | `30.01% / 30.13%` | `0.218596` | `0.211751` | `0.185568` | `0.216216` | `4/7` | `1/3` | `7/7` | `393.42ms` |

Aggregate grid:

- Versus uniform: `17/28` wins.
- Versus DP: `28/28` wins.
- Versus TemporalRandomFill: `18/28` wins.
- Low-budget versus uniform (`1%,2%,5%`): `4/12` wins.

The low-budget result is not good enough. `1%` and `2%` are ties caused by the
heavy temporal scaffold. They are not learned wins.

## Component And Geometry Diagnosis

At `30%` coverage / `5%` compression, current best versus uniform:

- `RangePointF1`: `0.091175` vs `0.089271`
- `ShipF1`: `0.667008` vs `0.655058`
- `ShipCov`: `0.109885` vs `0.104853`
- `EntryExitF1`: `0.199307` vs `0.193753`
- `CrossingF1`: `0.101842` vs `0.096723`
- `TemporalCov`: `0.274301` vs `0.260320`
- `GapCov`: `0.252489` vs `0.247608`
- `TurnCov`: `0.099750` vs `0.090915`
- `ShapeScore`: `0.171505` vs `0.159903`

The matched-cell improvement is broad across components, but small. Geometry
is worse:

- MLQDS `AvgSED=0.5703 km`; uniform `0.5391 km`.
- MLQDS length preservation `0.9305`; uniform `0.9276`.

This is a sensible retained trajectory set, but not a clearly superior one.

## Generalization Checks

Held-out workload-generator settings at `30%` coverage:

| Artifact | Setting | MLQDS | Uniform | DP | TemporalRandomFill | Wins vs uniform | Low-budget wins | Wins vs DP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `multitrain0205_c30_historicalprior_shipbalanced_localswap085_seed47_20260515` | seed `47`, mixed-density train/eval | `0.236355` | `0.226328` | `0.233481` | `0.225552` | `5/7` | `1/3` | `7/7` |
| `multitrain0205_c30_historicalprior_shipbalanced_localswap085_seed47_evaldense_20260515` | train mixed-density, eval/checkpoint dense | `0.218542` | `0.211209` | `0.206258` | `0.210162` | `2/7` | `1/3` | `6/7` |

The mixed held-out seed is acceptable as a diagnostic. The dense held-out
setting fails broad-grid robustness.

## Failed Variants That Matter

| Variant | Artifact | Outcome |
| --- | --- | --- |
| Clean learned stratified selector | `multitrain0205_c30_historicalprior_shipbalanced_stratified_20260515` | `0/7` wins vs uniform; confirms gains depend on temporal scaffold. |
| Lower scaffold with `local_delta_swap=0.66` | `multitrain0205_c30_historicalprior_shipbalanced_localdelta066_20260515` | Matched `0.208588` vs uniform `0.211751`; low-budget `0/3`. |
| Local-swap utility labels | `multitrain0205_c30_historicalprior_localdelta066_localswaputility_point32_20260515` | Forces a `1%` win but loses `2%`, `5%`, and broader grid; target build `181.31s`. |
| Continuity-retained labels | `multitrain0205_c30_historicalprior_continuity_localdelta066_20260515` | Helps some medium budgets, still loses `2%` and matched `5%`; worse geometry. |
| Anchor-prior mixing | `multitrain0205_c30_historicalprior_shipbalanced_localswap085_anchor3_20260515` | Matched `0.215155`; only `1/7` wins vs uniform. |
| `range_aware` retained-frequency teacher | `multitrain0205_c30_historicalprior_teacherretfreq_localswap085_20260515` | Matched `0.212709`; weaker than direct ship-balanced target; labels diffuse (`62.97%` positive). |
| Training-footprint diversity | `multitrain0205_c30_historicalprior_shipbalanced_localswap085_footmix6_20260515` | Matched `0.211856`, uniform `0.211751`, TemporalRandomFill `0.216216`; wins uniform `5/7` but low-budget remains `1/3` and target prep grows to `100.98s`. |
| Support cap `0.30` | `multitrain0205_c30_historicalprior_shipbalanced_localswap085_supp030_20260515` | Latency improves to `526.73ms`, but matched `0.209186` loses uniform. |
| Support cap `0.50` | `multitrain0205_c30_historicalprior_shipbalanced_localswap085_supp050_20260515` | Latency improves to `641.37ms`, but matched `0.209155` loses uniform. |
| KNN neighbor count `k=8` | `multitrain0205_c30_historicalprior_shipbalanced_localswap085_k8_20260515` | Matched `0.216053`; worse than default `k=32`, fewer wins vs uniform (`3/7`) and TRF (`2/7`). |
| KNN neighbor count `k=64` | `multitrain0205_c*_historicalprior_shipbalanced_localswap085_k64_20260515` | Full grid improves uniform wins only from `17/28` to `18/28`, keeps low-budget wins `4/12`, lowers matched scores at every coverage, and reduces mean audit delta. |
| Train-source KNN agreement | `multitrain0205_c30_historicalprior_shipbalanced_localswap085_sourcemean_20260515`, `..._sourcemedian_20260515` | Source-mean matched `0.213399`; source-median matched `0.214525`; both keep low-budget wins at `1/3` and trail pooled `k=32`. |
| MMSI identity historical prior | `multitrain0205_c30_historicalmmsi_shipbalanced_localswap085_w2_20260515`, `..._w05_20260515` | Legal query-free feature, but not useful enough. Weight `2.0` drops matched to `0.212172` and wins uniform `3/7`; weight `0.5` recovers `4/7` but matched stays `0.213956`, below current best, with low-budget wins still `1/3`. |
| Minimum learned swaps | `multitrain0205_c30_historicalprior_shipbalanced_localswap085_minswap1_20260515`, `..._minswap2_20260515` | Diagnostic selector fix for low-budget temporal ties. Min1 opens a real `1%` win and improves wins to `5/7`, but still loses `2%` and trails current-best matched score. Min2 worsens matched quality and geometry despite better train-target recall. |
| Trainable `range_prior` with minswap1 | `multitrain0205_c30_rangeprior_shipbalanced_localswap085_minswap1_20260515` | Fails to fit/use exposed learned-swap target: matched `0.211065` loses uniform, uniform wins only `1/7`, low-budget wins `0/3`, train-target tau `+0.063`, low-budget target-recall delta `-0.0131`. Latency is better (`93.19ms`) but quality is not. |
| `historical_prior_student` with minswap1 | `multitrain0205_c30_historicalpriorstudent_shipbalanced_localswap085_minswap1_20260515` | Fits target signal better (`tau=+0.791`, low-budget target-recall lift `+0.0895`) and wins uniform `5/7`, but matched `0.215283` trails current best `0.218596`, trails TemporalRandomFill at matched `5%`, still loses the `2%` uniform cell, worsens geometry vs uniform, and regresses latency to `751.72ms`. |
| `range_prior_clock_density` with minswap1 | `multitrain0205_c30_rangepriorclockdens_shipbalanced_localswap085_minswap1_20260515` | Adding clock-time and local-density features does not fix neural target fit: matched `0.211236` loses uniform, wins uniform `0/7`, low-budget wins `0/3`, train-target tau `+0.013`, matched target-recall delta `-0.0150`, low-budget delta `-0.0304`. |
| MLP pointwise `historical_prior_student` with minswap1 | `multitrain0205_c30_historicalpriorstudent_mlp_pointwise_shipbalanced_localswap085_minswap1_20260515` | Direct pointwise fit is not enough. Train fit is decent (`tau=+0.745`, low-budget target-recall lift `+0.0894`), but matched `0.210241` loses uniform, wins uniform only `1/7`, low-budget wins `0/3`, and latency remains KNN-dominated at `706.60ms`. |
| MLSimp methodology review | `Sprint/MLSimp-Paper.pdf` | Useful next ideas are structural and protocol-constrained: graph/segment query-free scoring, globality/uniqueness regularization, training-only amplified low-budget labels, and a diagnostic database-global budget selector. Its query-adjusted inference path must not be copied as final success because it uses simulated query importance during simplification. |
| MLSimp-inspired segment-context scorer | `multitrain0205_c30_segmentcontext_shipbalanced_localswap085_minswap1_20260515` | First structural architecture check failed. Matched `0.210407` loses uniform `0.211751` and TemporalRandomFill `0.216053`; audit wins uniform `0/7`, low-budget uniform `0/3`, DP `6/7`, TemporalRandomFill `1/7`. Train-target tau is negative (`-0.031`) and target-recall lift is negative, so segment attention/globality scalars alone do not make current labels learnable. |
| Structural retained-frequency target | `multitrain0205_c30_segmentcontext_structural010_shipbalanced_localswap085_minswap1_20260515`, `...structural025...`, `...structural050...` | Partial but insufficient. Blend `0.25` is the best check: matched `0.212538` barely beats uniform `0.211751`, opens `2%` and `5%`, but still loses `1%`, wins only `2/7` uniform cells, trails TemporalRandomFill at matched `5%`, and worsens geometry (`AvgSED=0.5596 km` vs uniform `0.5391`). Blends `0.10` and `0.50` lose every uniform audit cell. |
| Lower scaffold with `local_swap=0.66` | `multitrain0205_c30_historicalprior_shipbalanced_localswap066_20260515` | Matched `0.208252` loses uniform, low-budget wins `0/3`, and geometry collapses (`AvgSED=1.0303 km`). |
| Lower scaffold with low-budget target | `multitrain0205_c30_historicalprior_shipbalanced_localswap066_lowbudgets_20260515` | Forces a `1%` win but still loses `2%`, `5%`, and the broader grid: matched `0.204214`, uniform wins `1/7`, low-budget wins `1/3`, with worse geometry (`AvgSED=0.8502 km`). |
| Low-budget target with `local_delta_swap=0.66` | `multitrain0205_c30_historicalprior_shipbalanced_localdelta066_lowbudgets_20260515` | Worse gate diagnostic: matched `0.195771`, uniform wins `0/7`, low-budget wins `0/3`, TemporalRandomFill wins `0/7`, and runtime regresses (`40.88s`). |
| Local-window replacement selector | `multitrain0205_c30_historicalprior_shipbalanced_localwindow066_20260515` | Temporary diagnostic removed from public CLI after measurement; matched `0.203202`, low-budget wins `0/3`; localizing swaps reduces geometry damage but worsens `RangeUseful`. |
| Local-delta gain/cost labels | `multitrain0205_c30_historicalprior_localdelta066_gaincost_gain32_20260515` | Better aligned with the selector, but worse in practice: matched `0.196546`, uniform wins `0/7`, low-budget wins `0/3`; target build `216.53s` and train-target lift still misleads. |
| Current-best raw score mode | `multitrain0205_c30_historicalprior_shipbalanced_localswap085_rawscore_20260515` | Neutral, not useful: matched `0.218596` and audit wins `4/7` mirror the rank-score current best; low-budget cells remain temporal ties and `10%` still loses uniform. |
| Matrix-multiply exact distance kernel | `multitrain0205_c30_historicalprior_shipbalanced_localswap085_matmulknn_20260515` | Reverted; slower (`936.50ms`) and worse (`0.212090`). |

## Runtime Diagnosis

Useful implementation improvement:

- `MLQDSMethod` now reuses the same budget-independent score vector across
  compression-ratio mask freezes.
- Current-best audit freeze time dropped from `4.50s` to `0.37s` with identical
  quality on the `30%` branch.
- Standalone `historical_prior` now uses a flat pointwise inference path instead
  of scoring overlapping windows. Current-best `30%` quality is unchanged while
  matched MLQDS latency drops from about `866ms` to `393ms`, and train-fit
  diagnostics drop from about `23s` to `7.5s`.

Remaining bottleneck:

- First-mask latency for full-support historical-prior KNN remains about
  `393ms`, still far above uniform and DP.
- Pruning support reduces latency but loses quality.
- Matrix-multiply exact distances were slower and changed retained ordering.

The remaining credible runtime path is an indexed or approximate KNN
implementation with quality ablations, not support pruning.

## Final Assessment

The redesign implementation is credible. The final model is not.

Passes:

- Workload-blind protocol and artifact guards.
- Full coverage/compression audit.
- Required metrics and diagnostics.
- DP wins across the main grid.
- A meaningful partial branch at matched `5%` across all coverage targets.

Fails:

- Uniform baseline is not beaten strongly enough: `17/28` cells, with only
  `4/12` low-budget wins.
- `1%` and `2%` low-budget cells are temporal ties, not learned improvements.
- Current best depends on `mlqds_temporal_fraction=0.85`, which makes model
  learning less central than desired.
- Dense held-out workload setting is weak (`2/7` wins vs uniform).
- Geometry remains worse than uniform.
- First-mask latency is still high, though materially improved.

Do not mark the active goal complete as a final successful Range-QDS model.
The current system supports a clear failure diagnosis and a partial
workload-blind baseline, but it does not satisfy the acceptance bar.

## Next Credible Work

1. Add an indexed or approximate historical-prior KNN path that preserves broad
   support and benchmark quality/runtime jointly.
2. Replace scalar historical-frequency targets with a structural blind model
   that predicts when to override uniform temporal structure. Existing target
   reshuffles do not transfer.
3. Do not treat more workload-footprint mixing as the default next move. The
   six-footprint replicate run increased target-prep cost and still failed the
   matched and low-budget acceptance bar.
4. Do not treat KNN neighbor-count tuning as a model-quality solution. `k=64`
   raises the raw win count by one cell but weakens matched scores and leaves
   the low-budget failure untouched.
5. Do not treat source-day agreement aggregation as a quality solution for the
   current nonparametric prior. It is cleaner than pooled memorization, but it
   weakens matched quality and leaves low-budget cells unchanged.
6. Do not spend more time on selector-only variants around the current
   historical-prior score. The local-window diagnostic contained geometry damage
   but made `RangeUseful` worse.
7. Do not assume pairwise local marginal labels will fix replacement confidence.
   Gain/cost labels align with `local_delta_swap`, fit train targets strongly,
   and still lose every uniform audit cell.
8. Treat train target-recall lift as a weak diagnostic only. It missed the
   quality collapse under support capping and lower-scaffold `local_swap`.
9. Do not pursue retained-frequency budget weighting around the current
   historical-prior/local-swap branch. It can force an isolated `1%` win, but
   it damages `2%`, `5%`, and the broader grid.
10. Do not treat MMSI identity as the missing blind signal for the current
   historical-prior KNN branch. It is protocol-legal, but the measured hash
   prior lowers matched quality or merely recovers the old win count while
   leaving low-budget cells unchanged.
11. Do not use minimum learned swaps as a final-success mechanism around the
   current KNN score. It exposes selector-rounding artifacts and oracle
   headroom, but the blind score still cannot win the `2%` cell or beat the
   current matched score robustly.
12. Do not assume the existing trainable `range_prior` model becomes useful
   once learned slots are exposed. Under minswap1 it fails target fit and loses
   uniform; the next trainable path needs a stronger objective/model, not just
   the selector diagnostic.
13. Do not treat `historical_prior_student` as a solved trainable replacement.
   It fits the current labels better than `range_prior`, but the held-out gain
   is still below the current-best KNN branch, misses the `2%` cell, and is too
   slow to justify broader sweeps before architecture/objective changes.
14. Do not assume missing clock/density features explain neural failure. The
   `range_prior_clock_density` minswap1 run still cannot fit the target and
   loses every uniform compression cell.
15. Do not treat pointwise MLP fitting of the KNN prior as a rescue path. It
   improves train fit but worsens held-out usefulness and keeps KNN latency.
16. Use MLSimp as methodology input, not as a protocol template. Its
    query-based importance adjustment is useful as a training-prior idea, but
    using generated/eval-time query adjustment during final compression would
    weaken the workload-blind claim.
17. Do not assume the MLSimp-inspired segment-context architecture is enough by
    itself. The first `segment_context_range` benchmark cannot fit the current
    labels and loses every uniform audit cell; the next MLSimp-derived step
    needs objective-level signal such as explicit globality/uniqueness pressure
    or training-only low-budget oracle labels.
18. Do not treat the first structural-retained target as final success. It
    creates a weak low-budget signal only at blend `0.25`, but the target
    positive mass diffuses to more than half the points, the `1%` cell still
    loses, and geometry worsens versus uniform.
19. Do not treat structural boost as the fix for additive target diffusion.
    `range_structural_target_source_mode=boost` correctly preserves
    train-usefulness support, but the benchmark loses every uniform audit cell
    and all low-budget cells. The small additive-blend gains came from broader
    support, not from a robust learned ranking signal.
20. Do not add a duplicate stratified retained-frequency target. The existing
    retained-frequency transform already follows `mlqds_hybrid_mode`; with
    `stratified` it is target/selector aligned. `segment_context_range` under
    this aligned stratified setup loses every uniform audit cell and worsens
    geometry, so the problem is not merely target/selector mismatch.
21. Treat MLSimp-style global budget allocation as a diagnostic, not a result.
    `GlobalRandomBudget` loses every uniform audit cell, while
    `GlobalOracleBudget` shows very large headroom from perfect global ranking
    (`0.737246` matched `RangeUseful`, and `0.995275` at `30%`). This says the
    budget allocator is cheap and promising, but only when the score is already
    excellent. The final path needs training-only global-scarcity labels or
    teacher-student distillation, not eval-label global allocation.
22. Do not treat training-only global-scarcity labels as sufficient. The
    `global_budget_retained_frequency` target fits strongly under the
    historical prior, but with the existing `local_swap` selector it only gets
    `0.212994` matched `RangeUseful`, below the current-best `0.218596`, and
    still misses low-budget robustness.
23. Do not treat the blind `global_budget` selector as a final-success path in
    its current form. It raises point-hit F1 but loses every uniform audit cell,
    drops matched `RangeUseful` to `0.173004`, and worsens `AvgSED` to
    `6.2173 km`. Endpoint skeleton plus global top score is too
    continuity-blind for sensical retained trajectories.
24. The proposed week-to-week AIS split had not been run before. The code now
    supports comma-separated validation/eval CSV lists and a capped smoke run
    over train `2026-02-02..07`, validation `2026-02-08`, eval
    `2026-02-09..15` completed. The smoke result still fails the core pattern:
    only `2/7` wins vs uniform, `1/3` low-budget wins, `1%` and `2%` temporal
    ties, and worse geometry. A full week-level run is now possible, but the
    current model should not be expected to pass without a stronger learned
    signal.
25. A larger week-to-week mid-cap run (`60` segments/day) confirms the same
    failure. Current-best historical-prior retained-frequency gets only
    `0.170825` matched `RangeUseful` vs uniform `0.169161`, wins `3/7` audit
    cells vs uniform, has only `1/3` low-budget wins, ties uniform at `1%` and
    `2%`, and worsens `AvgSED` (`0.7885 km` vs `0.7367 km`). Do not run a full
    week-level acceptance sweep on this branch unchanged.
26. Per-training-day validation holdout has not been tested as a controlled
    experiment. The current serious runs use independent validation days. The
    code can split validation from combined training trajectories when no
    validation CSV is provided, but that split is not source-stratified by day.
    Treat source-stratified same-day validation as a selector diagnostic, not as
    stronger final evidence than independent-day validation.
27. Benchmark rows now include explicit train/validation/eval CSV paths,
    split-specific file counts, and selected cleaned-file lists. This closes a
    report-level audit gap for multi-day/week-to-week experiments; model quality
    is unchanged.
28. Source-stratified same-training-day validation is now implemented and tested
    as an explicit diagnostic. On the week-to-week mid-cap current-best branch,
    it improves matched `5%` `RangeUseful` (`0.173938` vs the independent-day
    run's `0.170825`) but worsens the audit grid (`2/7` wins vs uniform instead
    of `3/7`) and still ties uniform at `1%` and `2%`. It should not replace
    independent-day validation for final claims.
29. Selector-capacity diagnostics now report how much retained budget can be
    affected by learned scores. The refreshed source-stratified week artifact
    shows `0` learned slots at `1%` and `2%`; at matched `5%`, only `8.70%` of
    retained slots are learned-score decisions and `42.14%` of eval trajectories
    have no learned slot. This confirms the current local-swap branch is too
    scaffold-dominated for final workload-blind success.
30. Opening learned low-budget slots does not rescue the current branch. The
    longer-trajectory smoke (`max_points_per_segment=512`,
    `mlqds_min_learned_swaps=1`) gives the model learned slots at `1%`, `2%`,
    and `5%`, but loses every uniform audit cell (`0/7`) and trails
    TemporalRandomFill in every cell. Strong train target fit
    (`tau=+0.9983`, low-budget target-recall delta `+0.0834`) does not
    transfer to held-out week `RangeUseful`.
