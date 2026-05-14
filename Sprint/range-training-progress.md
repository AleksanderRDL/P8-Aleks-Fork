# Range Training Progress

Working log for the workload-blind Range-QDS redesign. `Sprint/range-training-redesign.md`
remains the design reference.

## 2026-05-14

### Protocol implementation checkpoint

- Added explicit workload-blind model types: `workload_blind_range` and `range_prior`.
- Added point-only blind features and a query-free inference path.
- Updated evaluation so workload-blind runs freeze `MLQDS`, `uniform`, and
  `DouglasPeucker` retained masks before held-out eval query scoring.
- Blocked eval-time geometry-label blending for workload-blind methods.
- Added regression tests proving blind simplification and validation scoring do
  not read query feature tensors.
- Added an expected-usefulness workload-blind benchmark profile.

Conclusion: the main leakage risk in the old path is now explicitly guarded.
This does not prove the trained model is useful.

### Focused experiment results

Baseline real AIS setup unless noted:

- train day: `2026-02-02`
- validation day: `2026-02-03`
- eval day: `2026-02-04`
- `max_segments=120`, `max_points_per_segment=256`
- target compression near `5%`

| Artifact | Change | Result |
| --- | --- | --- |
| `blind_real_c10_usefulness_20260514` | simple expected-usefulness labels, temporal fill `10%` | Failed. `MLQDS RangeUseful=0.1886`, uniform `0.2664`, DP `0.2339`. |
| `blind_real_c10_temporal25_residual_20260514` | temporal residual labels, temporal fill `25%` | Failed. `MLQDS=0.2209`, uniform `0.2664`, DP `0.2339`. Point F1 improved, aggregate still poor. |
| `blind_real_c10_balanced_temporal25_20260514` | component-balanced labels | Failed. `MLQDS=0.2213`. No meaningful lift. |
| `blind_real_c10_temporal25_div1_20260514` | diversity bonus `1.0` | Failed. `MLQDS=0.2240`. Slight 1% compression lift, but not a broad win. |
| `blind_real_c10_ranking_bce_20260514` | ranking+BCE objective | Failed. `MLQDS=0.1884`. |
| `blind_real_c10_features_temporal25_20260514` | added blind local features | Failed. `MLQDS=0.2166`. |
| `blind_real_c10_features_temporal25_500seg_20260514` | larger 500 segment cap | Failed. `MLQDS=0.2446`, uniform `0.3094`, DP `0.2930`. |
| `range_aware_real_c10_temporal25_20260514` | query-aware diagnostic | Succeeded diagnostically only. `range_aware MLQDS=0.2986`, uniform `0.2664`, DP `0.2339`. |
| `blind_real_true_c10_n24_20260514` | lower actual workload coverage using 24 queries | Failed. `MLQDS=0.2259`, uniform `0.2713`, DP `0.2472`; point F1 beat baselines. |
| `blind_real_true_c10_swap50_20260514` | swap mode with `50%` temporal scaffold | Partial diagnostic. `MLQDS=0.2566`, uniform `0.2713`, DP `0.2472`. It beat DP and temporal random fill at 5%, but still lost uniform and relies too much on scaffold. |

### Failed assumptions

- More local point features alone did not make the expected-usefulness target
  learnable enough.
- Component-balanced labels did not fix the temporal/gap/shape regressions.
- Ranking loss did not fix student fit.
- More segments did not fix the issue.
- Large temporal scaffolding can hide model weakness. It is useful as a
  diagnostic, not final evidence.

### Current diagnosis

The blind model can learn some point-hit signal: several runs improve
`RangePointF1`. It does not preserve enough temporal continuity, gap coverage,
and shape to win `RangeUseful`.

The query-aware diagnostic beats uniform/DP on the same workload. The workload
and labels are not hopeless; the bottleneck is compressing the workload-specific
signal into a query-blind retained set.

### Next decision

Stop spending time on selector tricks until the training target improves.

Next target path: teacher-student distillation or a cleaner marginal target that
teaches retained-set usefulness rather than isolated point usefulness. The first
distillation variant should aggregate a `range_aware` teacher into query-blind
per-point targets, then verify whether the blind student can beat temporal
random fill and uniform at low compression without increasing temporal scaffold.

### Follow-up implementation checkpoint

- Added query-aware teacher-student distillation as a first-class option:
  `range_teacher_distillation_mode={rank_percentile,retained_frequency}`.
- Added retained-set frequency training targets:
  `range_training_target_mode=retained_frequency`.
- Added benchmark profiles for teacher distillation and retained-frequency
  workload-blind training.
- Kept final evaluation protocol unchanged: workload-blind retained masks are
  still frozen before eval query scoring.

### Follow-up experiment results

Same real AIS low-coverage setup as above unless noted:

| Artifact | Change | Result |
| --- | --- | --- |
| `blind_real_true_c10_teacherfreq_20260514` | range-aware teacher retained-frequency labels, temporal fill `25%` | Failed. `MLQDS=0.2544`, uniform `0.2713`, DP `0.2472`; lost temporal random fill. |
| `blind_real_true_c10_teacherrank_20260514` | range-aware teacher rank-percentile labels | Failed. `MLQDS=0.2360`, worse than DP. Dense teacher labels were too weak/noisy. |
| `blind_real_true_c10_retfreq_20260514` | explicit label retained-frequency target, temporal fill `25%`, residual training | Partial. `MLQDS=0.2763`, uniform `0.2713`, DP `0.2472`; wins at 5%, but grid still mixed and loses temporal random fill at 5%. |
| `blind_real_true_c10_retfreq_swap25_20260514` | retained-frequency target with `swap` mode and temporal `25%` | Partial but no meaningful grid improvement. `MLQDS=0.2786`. |
| `blind_real_true_c10_retfreq_fulltarget_20260514` | retained-frequency target, temporal fill `25%`, no residual masking | Strongest 25% run. `MLQDS=0.2870`, uniform `0.2713`, DP `0.2472`, temporal random fill `0.2853`. Grid wins uniform at 1/5/10/15/30, loses at 2/20; DP remains mixed. |
| `blind_real_true_c10_retfreq_fulltarget_t30_20260514` | retained-frequency full target, temporal fill `30%` | Best single 5% result. `MLQDS=0.3001`, uniform `0.2713`, DP `0.2472`, temporal random fill `0.2880`. Grid improves 5/10/20, nearly ties uniform at 2, but still loses DP at 1/15/30. |
| `blind_real_true_c10_retfreq_fulltarget_t35_20260514` | retained-frequency full target, temporal fill `35%` | Failed. Validation checkpoint selected poorly; eval `MLQDS=0.2322`. Treat as evidence against further temporal-fraction tuning. |
| `blind_real_true_c10_retfreq_fulltarget_t30_features25_20260514` | added larger blind continuity feature vector | Failed. `MLQDS=0.2598`, worse than the 17-feature retained-frequency run. Removed the expanded feature vector rather than keeping stale experimental code. |

### Updated diagnosis

The retained-frequency target is a real improvement. It produces learned wins at
5% compression and beats temporal random fill there, so the win is not purely a
uniform scaffold. It still does not satisfy final acceptance: the audit grid is
not consistently ahead of uniform and DP, and global geometry distortion remains
worse than both baselines.

The core remaining weakness is not point-hit learning. It is continuity and
shape preservation under a blind scoring model. The next technical step should
be a model or feature change that makes continuity learnable directly, not more
temporal-fraction tuning.

### Additional implementation checkpoint

- Added optional `temporal_distribution_loss_weight` for a budget-aware temporal
  CDF regularizer. It is disabled by default and reported in benchmark rows.
- Added training-workload replicate generation via
  `range_train_workload_replicates`.
- Added explicit replicate aggregation modes:
  `range_replicate_target_aggregation={label_mean,frequency_mean}`.
  `label_mean` averages raw usefulness labels before retained-frequency target
  selection. `frequency_mean` averages per-workload retained-frequency targets.
- Recorded training-workload replicate counts, coverages, and aggregation
  diagnostics in run JSON.

### Additional experiment results

Same real AIS setup unless noted. Compression target is still `5%`; audit
tables cover `1/2/5/10/15/20/30%`.

| Artifact | Change | Result |
| --- | --- | --- |
| `blind_real_true_c10_retfreq_fulltarget_t30_tempcdf005_20260514` | temporal CDF loss `0.05` | Mostly neutral. `MLQDS=0.3005`, uniform `0.2713`, DP `0.2472`; worsened 2% and did not fix 15/20/30. Do not tune this further without a stronger reason. |
| `blind_real_true_c10_retfreq_fulltarget_t30_trainrep2_20260514` | 2 train workload replicates, `frequency_mean` | Failed. `MLQDS=0.2816`; below temporal random fill and weaker than single-workload target. |
| `blind_real_true_c10_retfreq_fulltarget_t30_trainrep4_20260514` | 4 train workload replicates, `frequency_mean` | Mixed. `MLQDS=0.2949`; best 1% cell (`0.1606` vs uniform `0.1366`, DP `0.1452`) but worse at 5%+ than single-workload. |
| `blind_real_true_c10_retfreq_fulltarget_t30_trainrep4_labelmean_20260514` | 4 train workload replicates, `label_mean` | Mixed/weak. `MLQDS=0.2954`; better than uniform at 5/10/30, still loses 2/15/20 and DP at 1/15/30. |

### Coverage-axis diagnostic

Using the current best single-workload retained-frequency setup:

| Artifact | Actual train/eval coverage | Result |
| --- | --- | --- |
| `blind_real_true_c05_retfreq_fulltarget_t30_20260514` | train `10.09%`, eval `6.77%` | Failed vs uniform. `MLQDS=0.2599`, uniform `0.2791`, DP `0.2300`. The fixed 24-query floor overshot the nominal 5% train target. |
| `blind_real_true_c10_retfreq_fulltarget_t30_20260514` | train `10.09%`, eval `12.78%` | Best current cell. `MLQDS=0.3001`, uniform `0.2713`, DP `0.2472`. |
| `blind_real_true_c15_retfreq_fulltarget_t30_20260514` | train `15.11%`, eval `15.19%` | Failed vs uniform. `MLQDS=0.2662`, uniform `0.2762`, DP `0.2460`. |
| `blind_real_true_c30_retfreq_fulltarget_t30_20260514` | train `30.04%`, eval `30.12%` | Failed vs uniform. `MLQDS=0.2421`, uniform `0.2820`, DP `0.2263`. |

### Updated diagnosis after coverage sweep

- The model is not ready for final success claims. It only clears the matched
  10% coverage / 5% compression cell.
- The current target reliably improves `RangePointF1`, but uniform keeps winning
  `TemporalCov`, `GapCov`, and often shape as coverage broadens.
- Multi-workload target averaging is not automatically better. `frequency_mean`
  diffuses label mass across too many points. `label_mean` is cleaner, but still
  does not fix continuity.
- The 5% coverage run is not cleanly calibrated because `n_queries=24` already
  overshoots the train target. Future coverage-grid runs need per-coverage
  query-count calibration, not a fixed query floor.
- Next useful direction is not more temporal fraction tuning. The likely missing
  piece is a set-aware/marginal target or a continuity-aware model/selector that
  learns why uniform wins temporal and gap coverage without making temporal
  scaffolding dominate the result.

### Component-target checkpoint

- Tightened the workload-blind audit protocol so retained masks for every
  audited compression ratio are frozen before held-out eval query scoring, not
  only the matched 5% ratio.
- Added `component_retained_frequency`, which builds retained-frequency targets
  independently for each `RangeUseful` component, plus
  `range_component_target_blend` for interpolation with the base target.

Same real AIS 10% coverage setup:

| Artifact | Change | Result |
| --- | --- | --- |
| `blind_real_true_c10_component_retfreq_t30_20260514` | component-only retained-frequency target | Failed. Matched `MLQDS=0.2742`, uniform `0.2713`, DP `0.2472`; it barely beat uniform at 5% and was worse than the base retained-frequency target. |
| `blind_real_true_c10_componentblend035_retfreq_t30_20260514` | `35%` component target blended with base target | Mixed. Matched `MLQDS=0.2992`, uniform `0.2713`, DP `0.2472`; near the best base target at 5%, but still loses uniform at 2/15/20% and DP at 1/15/30%. |

### Diagnosis after component target

- Component-only labels diffuse target mass across too many points. They do not
  create a useful low-budget ranking.
- The 35% blend slightly helps 15% and 30% compression, but the gain is too small
  to change the conclusion.
- The dominant failure remains continuity: `TemporalCov`, `GapCov`, and
  `ShapeScore` lag uniform/DP in the cells where `RangeUseful` loses.
- More target interpolation is unlikely to satisfy acceptance. The next
  worthwhile experiment should test whether a modest diversity-aware selector
  helps the retained-frequency model without turning into a uniform sampler. If
  not, the target needs a genuinely set-aware marginal construction.

### Diversity and coverage sweep

- Added `mlqds_diversity_bonus` into retained-frequency target experiments.
  This changes the training target and final selector together; it is still a
  selector-sensitive path and not proof by itself.
- Added `range_temporal_target_blend` as a diagnostic retained-frequency target
  interpolation with pure uniform temporal anchors. It changes training labels
  only and leaves inference `mlqds_temporal_fraction` unchanged.

10% coverage diversity sweep:

| Artifact | Change | Result |
| --- | --- | --- |
| `blind_real_true_c10_retfreq_fulltarget_t30_div002_20260514` | diversity `0.02` | Best grid balance. Matched `MLQDS=0.2996`, uniform `0.2713`, DP `0.2472`; wins uniform `6/7`, DP `5/7`, including 1/2/5% compression. |
| `blind_real_true_c10_retfreq_fulltarget_t30_div005_20260514` | diversity `0.05` | Best matched 10% result. `MLQDS=0.3075`, but wins uniform only `4/7` because 2/15/20% regress. |
| `blind_real_true_c10_retfreq_fulltarget_t30_div015_20260514` | diversity `0.15` | Too diffuse. `MLQDS=0.2762`; target positive fraction rose to `56.9%` and the 5% cell degraded. |

Coverage sweep using diversity `0.02`:

| Artifact | Actual train/eval coverage | Result |
| --- | --- | --- |
| `blind_real_true_c05_n8_retfreq_t30_div002_20260514` | train `5.19%`, eval `5.39%` | Good. Matched `MLQDS=0.2888`, uniform `0.2541`, DP `0.2192`; wins uniform and DP `5/7` audit cells. This required lowering `n_queries` to `8`; fixed `24` queries was not a valid 5% coverage setup. |
| `blind_real_true_c10_retfreq_fulltarget_t30_div002_20260514` | train `10.09%`, eval `12.78%` | Good compression-grid result. Matched `MLQDS=0.2996`; wins uniform `6/7`, DP `5/7`. |
| `blind_real_true_c15_retfreq_t30_div002_20260514` | train `15.11%`, eval `15.19%` | Failed vs uniform. Matched `MLQDS=0.2583`, uniform `0.2762`, DP `0.2460`; wins uniform only `1/7`. |
| `blind_real_true_c30_retfreq_t30_div002_20260514` | train `30.04%`, eval `30.12%` | Failed vs uniform. Matched `MLQDS=0.2350`, uniform `0.2820`, DP `0.2263`; wins uniform `0/7`, DP `1/7`. |
| `blind_real_true_c30_retfreq_t30_div002_temporalblend020_20260514` | train `30.04%`, eval `30.12%` | Worse. Temporal target blend `0.20` gave `MLQDS=0.2174`; target positive fraction jumped to `77.1%`. |

### Updated diagnosis after diversity and temporal-target tests

- The workload-blind redesign now has a credible low-coverage result: calibrated
  5% coverage and 10% coverage can beat uniform/DP across most compression cells.
- It still fails final acceptance because the same method collapses at 15% and
  30% coverage. Those cells are not close.
- The failure is not a lack of query signal or a runtime issue. The diagnostic
  oracle gaps remain large, runtime is small enough for iteration, and the low
  coverage cells learn useful point priors.
- The failure is target alignment under broad workloads. Retained-frequency
  labels still train a hotspot ranking, while broad workloads reward reusable
  temporal/shape coverage. A blunt temporal target blend diffuses label mass and
  makes the model worse.
- Next useful implementation is a real set-aware marginal target that estimates
  incremental `RangeUseful` gain under the retained set, or a model/selector that
  can learn redundancy and coverage directly. More scalar blends are unlikely to
  satisfy acceptance.

### Marginal-target and high-coverage failure checkpoint

- Added `range_training_target_mode=marginal_coverage_frequency`.
  It greedily converts range labels into retained-set frequency targets by
  erasing local label neighborhoods after each selected point. This is a
  training-only target transform; eval masks are still frozen before held-out
  query scoring.
- Added `range_marginal_target_radius_scale` to config, CLI, benchmark profiles,
  and benchmark reports.
- Added tests for marginal target construction and multi-workload aggregation.

Focused 30% coverage experiments, all on the same 2026-02-02/03/04 real AIS
setup with frozen audit ratios `1/2/5/10/15/20/30%`:

| Artifact | Change | Result |
| --- | --- | --- |
| `blind_real_true_c30_marginal_t30_div002_r050_20260514` | marginal target, radius scale `0.50` | Failed. Matched `MLQDS=0.2295`, uniform `0.2820`, DP `0.2263`; target positives rose to `57.8%`. It still loses temporal, gap, and shape components badly. |
| `blind_real_true_c30_retfreq_t30_div002_residual_20260514` | retained-frequency target with temporal-residual budget loss | Worse. `MLQDS=0.2201`; residual masking did not fix the train/inference mismatch. |
| `blind_real_true_c30_retfreq_t30_div002_trainrep4_labelmean_20260514` | 4 train workloads, label-mean aggregation | Failed. `MLQDS=0.2298`; extra workload seeds did not transfer into broad held-out query usefulness. |
| `blind_real_true_c30_retfreq_swap30_div002_20260514` | retained-frequency target with `swap` selector | Failed. `MLQDS=0.2282`; swap preserved no useful advantage and worsened global distortion. |

Conclusion: the broad-coverage failure is not solved by marginal neighborhood
erasure, residual loss masking, more same-setting train workloads, or swap-mode
selector changes. The learned fill is still weaker than temporal random fill at
30% coverage, so the model is not extracting a reusable blind prior for broad
future range workloads. Treat the current low-coverage wins as real but
insufficient. The next credible path is a deeper target/model change: either
multi-day historical training with day-anchored workloads, or a task-aligned
continuity/shape curriculum that proves the student can beat temporal random
fill before claiming baseline wins.

### Gap metric ablation checkpoint

- Added diagnostic gap variants without changing `RangeUseful` weights:
  `range_gap_time_coverage` and `range_gap_distance_coverage`.
- Count `GapCov` remains the aggregate component. The new fields are reported
  in range tables, run JSON, benchmark rows, runtime summaries, and inference
  outputs.

Reran the 30% coverage retained-frequency/diversity baseline with new fields:

| Artifact | Matched result | Gap diagnosis |
| --- | --- | --- |
| `blind_real_true_c30_retfreq_t30_div002_gapdiag_20260514` | `MLQDS=0.2350`, uniform `0.2820`, DP `0.2263` | The failure is not an artifact of count-normalized `GapCov`. At 5% compression, MLQDS trails uniform on count gap (`0.2079` vs `0.3545`), time gap (`0.1920` vs `0.2931`), and distance gap (`0.1927` vs `0.3315`). |

Conclusion: the broad-coverage weakness is real temporal/spatial continuity
loss, not only a metric-definition problem. The next useful step is not another
minor selector or target-radius sweep. It should change the information the
blind student can learn, starting with day-anchored multi-day historical
training or an explicit continuity/shape curriculum.

### Day-anchored workload-generation checkpoint

- Added `range_time_domain_mode` with default `dataset` and explicit
  `anchor_day`.
- `anchor_day` samples the same range anchors as before, but clamps each query
  to the 24-hour source/calendar day containing the anchor. For relative-second
  tensors it uses 24-hour chunks from the dataset lower bound; for epoch-like
  seconds it aligns to calendar day boundaries.
- Wired the mode through config, CLI, workload cache keys, benchmark profiles,
  benchmark reports, inference, coverage estimation, and query diagnostics.
- Benchmark profiles now pass `--range_time_domain_mode anchor_day`, while the
  raw CLI default remains `dataset` so older ad hoc runs are reproducible.
- Added tests for boundary-clamping behavior, workload diagnostics, cache-key
  separation, config round-trip/backward defaults, and profile CLI args.

Validation:

- `py_compile` passed for touched query/config/benchmark/inference files.
- `git diff --check` passed.
- Targeted tests passed:
  `QDS/tests/test_query_coverage_generation.py`,
  `QDS/tests/test_torch_runtime_controls.py`,
  `QDS/tests/test_benchmark_runner.py` (`47 passed`).
- Full test suite passed: `QDS/tests` (`199 passed`, one PyTorch nested-tensor
  warning).
- Synthetic pipeline smoke passed with
  `--model_type workload_blind_range --range_time_domain_mode anchor_day`; the
  artifact is `artifacts/manual/anchor_day_smoke_20260514`. The artifact records
  `anchor_day` in both config and train/eval workload generation diagnostics.

Conclusion: this does not solve the broad-coverage model failure by itself. It
removes a real protocol mismatch before multi-day historical training. The next
experiment should train on a combined multi-day historical CSV with
`anchor_day` workloads and evaluate on a distinct held-out day/settings seed.

### Multi-day `anchor_day` retained-frequency experiment

Built a combined historical train CSV:

- Source days: `2026-02-02`, `2026-02-05`, `2026-02-06`
- Validation day: `2026-02-03`
- Eval day: `2026-02-04`
- Artifact: `artifacts/cache/aisdk-2026-02-02_05_06_train_combined.csv`
  (`13,090,035` rows, `2,254` unique MMSIs, about `1.1G`)
- Loaded train cache after `max_segments=120`, `max_points_per_segment=256`:
  `120` segments, `15,669` points. Source-day distribution after loading:
  day chunks `0/3/4` with `37/43/40` segments, so the capped train set does
  actually span all three source days.

Focused 30% coverage run:

- Artifact: `blind_multiday_anchor_c30_retfreq_t30_div002_cr005_20260514`
- Target/method: retained-frequency, `mlqds_temporal_fraction=0.30`,
  diversity `0.02`, fixed `2.2 km / 5 h`, `anchor_day`
- Train/eval coverage: `30.19%` / `30.12%`
- Label target: positive fraction `51.31%`, mass `1965.14`

Result: failed. It is slightly worse than the single-day train baseline.

| Compression | Single-day MLQDS | Multi-day MLQDS | Uniform | DP | Multi-day vs uniform |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1%` | `0.1012` | `0.0988` | `0.1119` | `0.1111` | `-0.0131` |
| `2%` | `0.1373` | `0.1132` | `0.1937` | `0.1556` | `-0.0805` |
| `5%` | `0.2350` | `0.2252` | `0.2820` | `0.2263` | `-0.0568` |
| `10%` | `0.3310` | `0.3210` | `0.3742` | `0.3330` | `-0.0533` |
| `15%` | `0.3825` | `0.3882` | `0.4470` | `0.4060` | `-0.0587` |
| `20%` | `0.4730` | `0.4684` | `0.5249` | `0.4745` | `-0.0565` |
| `30%` | `0.5495` | `0.5503` | `0.6045` | `0.5788` | `-0.0542` |

Component diagnosis at `5%`:

- Multi-day MLQDS loses to uniform on point F1 (`0.1013` vs `0.1110`),
  ship F1 (`0.6855` vs `0.8411`), temporal coverage (`0.2414` vs `0.4036`),
  count gap (`0.1661` vs `0.3545`), time gap (`0.1717` vs `0.2931`),
  distance gap (`0.1663` vs `0.3315`), shape (`0.1396` vs `0.2178`), and
  length preservation (`0.8498` vs `0.8881`).
- It only meaningfully improves entry/exit against DP, not enough to affect
  `RangeUseful`.

Conclusion: simply adding historical days with day-anchored workloads does not
make retained-frequency labels learn the broad-workload prior. This weakens the
"not enough historical data" assumption. The failure still points at target and
objective mismatch: broad workloads reward continuity and shape coverage, while
the current learned ranking over-selects query-likely hotspots and loses the
uniform temporal spine. The next useful variation should be an explicit
continuity/shape curriculum or a set-level training objective, not another
same-label data-scale run.

Runtime note: the initial combined CSV parse took `35.48s`; the cached rerun
loaded in `0.13s`. If multi-day training remains in scope, full CSV concatenates
are the wrong long-term mechanism. Per-day cached composition would avoid the
1.1G intermediate and the full parse cost.

### Temporal-distribution objective check

Tested the existing budgeted temporal-CDF regularizer as an objective-side
continuity pressure, not an inference-time scaffold:

- Artifact: `blind_real_true_c30_retfreq_t30_div002_tcdf050_20260514`
- Change: `temporal_distribution_loss_weight=0.50`
- Otherwise same 30% retained-frequency setup as
  `blind_real_true_c30_retfreq_t30_div002_20260514`, with `anchor_day` enabled.

Result: failed. It did not close the broad-workload gap.

| Compression | Baseline MLQDS | Temporal-CDF MLQDS | Uniform | DP | Temporal-CDF vs uniform |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1%` | `0.1012` | `0.0993` | `0.1119` | `0.1111` | `-0.0125` |
| `2%` | `0.1373` | `0.1359` | `0.1937` | `0.1556` | `-0.0578` |
| `5%` | `0.2350` | `0.2295` | `0.2820` | `0.2263` | `-0.0525` |
| `10%` | `0.3310` | `0.3202` | `0.3742` | `0.3330` | `-0.0540` |
| `15%` | `0.3825` | `0.3874` | `0.4470` | `0.4060` | `-0.0596` |
| `20%` | `0.4730` | `0.4708` | `0.5249` | `0.4745` | `-0.0541` |
| `30%` | `0.5495` | `0.5515` | `0.6045` | `0.5788` | `-0.0530` |

Conclusion: a soft CDF regularizer is too indirect. It slightly stabilizes the
high-compression tail but hurts the critical `2%/5%/10%` cells. The model needs
training signal that values retained sets by actual range continuity/shape
utility, not just less-clustered logits.

### Query-spine, feature, and capacity checks

Implementation changes:

- Added `range_training_target_mode=query_spine_frequency`. It builds
  training-only temporal spine anchors from train range queries, then converts
  them into retained-frequency labels. Eval compression remains query-blind and
  frozen before held-out query scoring.
- Restored the default workload-blind feature vector to the stronger 17-column
  set. The attempted 24-column context vector is kept only for checkpoint
  compatibility; it was empirically worse as a default.
- Exposed model capacity controls in the CLI:
  `--embed_dim`, `--num_heads`, `--num_layers`, `--dropout`.

Validation:

- `py_compile` passed for touched training/config/CLI files.
- `git diff --check` passed.
- Full tests passed: `QDS/tests` (`202 passed`, one PyTorch nested-tensor
  warning).

Focused 30% coverage experiments on the same 2026-02-02/03/04 real AIS setup:

| Artifact | Change | Result |
| --- | --- | --- |
| `blind_real_true_c30_component_retfreq_t30_div002_20260514` | component-only retained-frequency target at 30% coverage | Failed. Matched `MLQDS=0.2234`, uniform `0.2820`, DP `0.2263`; loses uniform in every audit cell. |
| `blind_real_true_c30_retfreq_t30_div002_featctx_20260514` | 24-column blind context features | Failed. Matched `MLQDS=0.2305`; worse than the 17-column retained-frequency baseline and worse at higher compression. |
| `blind_real_true_c30_queryspine_f010_t30_div002_20260514` | query-spine target, spine fraction `0.10`, 24-column context features | Failed. Matched `MLQDS=0.2240`; source positives were only `1008/18669`, but densifying was needed to test the target fairly. |
| `blind_real_true_c30_queryspine_f030_t30_div002_20260514` | query-spine target, spine fraction `0.30`, 24-column context features | Failed. Matched `MLQDS=0.2330`; still loses uniform by `0.0490`. |
| `blind_real_true_c30_queryspine_f030_t30_div002_basicfeat_20260514` | query-spine target, spine fraction `0.30`, restored 17-column features | Failed. Matched `MLQDS=0.2249`; loses uniform in every audit cell and loses DP except near the matched cell. |
| `blind_real_true_c30_retfreq_t30_div005_20260514` | retained-frequency baseline, diversity `0.05` | Failed. Matched `MLQDS=0.2244`; diversity did not recover continuity. |
| `blind_real_true_c30_retfreq_t30_div002_pw0_20260514` | retained-frequency baseline, no pointwise BCE term | Failed. Matched `MLQDS=0.2178`; pointwise BCE is not the blocker. |
| `blind_real_true_c30_retfreq_t30_div002_wide16_20260514` | retained-frequency baseline, `embed_dim=128`, `num_layers=4`, `epochs=16` | Failed, but diagnostic. Matched `MLQDS=0.2325`; beats DP only at 5%, loses uniform in every audit cell, and worsens global distortion. |

Best of these on the 30% grid (`wide16`):

| Compression | MLQDS | Uniform | DP | MLQDS vs uniform |
| --- | ---: | ---: | ---: | ---: |
| `1%` | `0.0940` | `0.1119` | `0.1111` | `-0.0179` |
| `2%` | `0.1495` | `0.1937` | `0.1556` | `-0.0443` |
| `5%` | `0.2325` | `0.2820` | `0.2263` | `-0.0495` |
| `10%` | `0.3196` | `0.3742` | `0.3330` | `-0.0546` |
| `15%` | `0.3818` | `0.4470` | `0.4060` | `-0.0652` |
| `20%` | `0.4626` | `0.5249` | `0.4745` | `-0.0622` |
| `30%` | `0.5552` | `0.6045` | `0.5788` | `-0.0494` |

Component diagnosis at 5% remains unchanged in substance: the learned model is
competitive on point F1 and sometimes entry/exit, but loses the useful broad
workload signal through temporal coverage, gap coverage, shape, and length
preservation. The learned fill is still worse than temporal random fill in the
30% coverage setting.

Student-fit diagnosis:

- The 30% retained-frequency target is dense (`~51%` positive labels), but the
  student's train-window Kendall tau remains weak (`roughly -0.1` to `+0.15`)
  across baseline and capacity runs.
- Wider/deeper training improves matched `RangePointF1` but does not close the
  validation uniform gap and increases score-scale instability/global
  distortion.

Conclusion: the broad-coverage failure is not caused by the 24-column feature
choice, sparse query-spine labels, a too-small diversity bonus, BCE blending, or
minor model undercapacity. The current target is not sufficiently learnable or
not sufficiently aligned for broad workloads. The next credible step is a
genuinely set-level continuity/shape objective or curriculum that directly
optimizes residual fill against temporal-random/uniform baselines. More scalar
target blends and selector tweaks are now low-value.

### Data-scale, objective, and component-mass checks

Focused follow-up from the same 30% coverage failure:

| Artifact | Change | Result |
| --- | --- | --- |
| `blind_real_true_c30_retfreq_t30_div002_500seg_20260514` | raised train/validation/eval caps to `500` segments | Failed. Matched `MLQDS=0.2246`, uniform `0.2657`, DP `0.2655`; loses every audit ratio. It also worsens length preservation (`0.769` vs uniform `0.853`, DP `0.906`). |
| `blind_real_true_c30_retfreq_t30_div002_rankbce_20260514` | retained-frequency labels with `ranking_bce`, `ranking_pairs_per_type=256` | Failed badly. Matched `MLQDS=0.2016`; pairwise ranking is not better than budget-top-k here. |
| `blind_real_true_c30_retfreq_balanced_t30_div002_20260514` | `range_label_mode=usefulness_balanced` before retained-frequency target | Failed. Matched `MLQDS=0.2186`; component mass balancing does not transfer. |

500-segment audit grid:

| Compression | MLQDS | Uniform | DP | MLQDS vs uniform |
| --- | ---: | ---: | ---: | ---: |
| `1%` | `0.1281` | `0.1483` | `0.1417` | `-0.0202` |
| `2%` | `0.1511` | `0.2030` | `0.1906` | `-0.0519` |
| `5%` | `0.2246` | `0.2657` | `0.2655` | `-0.0411` |
| `10%` | `0.3233` | `0.3707` | `0.3544` | `-0.0474` |
| `15%` | `0.3901` | `0.4368` | `0.4240` | `-0.0467` |
| `20%` | `0.4553` | `0.4972` | `0.4902` | `-0.0419` |
| `30%` | `0.5650` | `0.5843` | `0.5946` | `-0.0193` |

Additional diagnosis:

- The earlier multi-day test did diversify source days, but because it still
  used `max_segments=120`, it did not test training-data scale. The 500-segment
  run closes that loophole and still fails.
- Validation often peaks at the first one or two epochs and then degrades while
  score scale grows. This looks like target/objective misalignment, not simply
  undertraining.
- Raw train usefulness labels are point/ship dominated. At 30% coverage,
  temporal+gap label mass is only about `13%`, and turn+shape about `10%`.
  However, balancing component mass made results worse, so the issue is not
  solved by scalar reweighting.

Conclusion: broad coverage still needs a different training formulation. The
next implementation should make the learned residual directly compete with
temporal-random/uniform continuity, likely through a set-level residual-fill
curriculum or loss. Data scale, pairwise ranking, and component mass balancing
are not enough.

### Query-residual target and blind-time feature checks

Implementation changes:

- Added `range_training_target_mode=query_residual_frequency`. It is
  training-only query-aware supervision: for each train range query and budget,
  it simulates the query-blind temporal base, then labels residual in-query
  anchors for boundary, gap, turn, and shape support. Eval compression remains
  workload-blind and freezes masks before held-out query scoring.
- Added `range_query_residual_multiplier` and
  `range_query_residual_mass_mode={query,point}`. `query` gives each train
  query unit target mass. `point` keeps selected residual-anchor frequency
  mass before averaging queries.
- Fixed a real cross-day feature issue in the workload-blind feature builder:
  the first blind base feature is now trajectory-local time fraction, not raw
  absolute timestamp. Training on one AIS day and evaluating another should not
  force the student to extrapolate epoch time under a train-day scaler.

Validation:

- Targeted tests passed:
  `test_model_features.py`, `test_teacher_distillation.py`,
  `test_torch_runtime_controls.py`, `test_benchmark_runner.py`
  (`51 passed`, one PyTorch nested-tensor warning before mass-mode; `45 passed`
  for the target/config subset after mass-mode wiring).
- The synthetic query-residual smoke run passed and recorded frozen masks before
  scoring.

Focused 30% coverage experiments on the same 2026-02-02/03/04 split:

| Artifact | Change | Result |
| --- | --- | --- |
| `blind_real_true_c30_queryresidual_m100_t30_div002_20260514` | query-residual target, multiplier `1.0`, query-equal mass | Failed. Matched `MLQDS=0.2061`, uniform `0.2820`, DP `0.2263`; target positives `11.0%`, target mass `1.0`. |
| `blind_real_true_c30_queryresidual_m300_t30_div002_20260514` | query-residual target, multiplier `3.0`, query-equal mass | Failed. Matched `MLQDS=0.2128`; target positives rose to `28.1%` but mass stayed `1.0`. |
| `blind_real_true_c30_queryresidual_point_m100_t30_div002_20260514` | query-residual target, multiplier `1.0`, point-frequency mass | Failed. Matched `MLQDS=0.2069`; mass rose to `6.266`, but validation/eval stayed far below uniform. |
| `blind_real_true_c30_retfreq_reltime_t30_div002_20260514` | retained-frequency baseline after replacing absolute timestamp with trajectory-local time fraction | Failed. Matched `MLQDS=0.2346`, essentially unchanged from old retained-frequency (`0.2350`). Geometry distortion improved somewhat, but `RangeUseful` did not. |
| `blind_real_true_c30_retfreq_reltime_t30_swap_div002_20260514` | retained-frequency with `mlqds_hybrid_mode=swap` | Failed. Matched `MLQDS=0.2104`, worse than fill mode; global geometry also worsened. |

Best query-residual grid (`m300`, query-equal mass):

| Compression | MLQDS | Uniform | DP | MLQDS vs uniform |
| --- | ---: | ---: | ---: | ---: |
| `1%` | `0.0934` | `0.1119` | `0.1111` | `-0.0185` |
| `2%` | `0.1076` | `0.1937` | `0.1556` | `-0.0861` |
| `5%` | `0.2128` | `0.2820` | `0.2263` | `-0.0692` |
| `10%` | `0.3185` | `0.3742` | `0.3330` | `-0.0558` |
| `15%` | `0.3923` | `0.4470` | `0.4060` | `-0.0547` |
| `20%` | `0.4557` | `0.5249` | `0.4745` | `-0.0692` |
| `30%` | `0.5379` | `0.6045` | `0.5788` | `-0.0667` |

Component diagnosis for query-residual `m300` at `5%`:

- MLQDS trails uniform on ship F1 (`0.662` vs `0.841`), temporal coverage
  (`0.212` vs `0.404`), gap coverage (`0.188` vs `0.355`), shape
  (`0.130` vs `0.218`), crossing (`0.103` vs `0.126`), and length
  preservation (`0.844` vs `0.888`).
- It only barely matches turn coverage and remains worse than DP on
  `RangeUseful`.

Conclusion: the residual-anchor target family is not the missing piece. Raising
anchor density, changing target mass normalization, fixing absolute-time
feature shift, and using a temporal-swap selector all fail the broad-coverage
cell. The broad workload problem still looks like a learned-score problem:
current blind scores cannot improve temporal continuity enough to beat uniform.
Further work should not claim success from `swap` or higher temporal scaffolding.
The next credible step is either an actual set-utility teacher/optimizer for
train queries or a more fundamental model/feature rethink; more anchor
heuristics are low-value.

### Set-utility target checkpoint

Implementation changes:

- Added `range_training_target_mode=set_utility_frequency`. It is a training-only
  query-aware target that starts from the query-blind temporal base and assigns
  labels by one-step marginal `RangeUseful` gain on train queries.
- Added `range_set_utility_multiplier`,
  `range_set_utility_candidate_limit`, and
  `range_set_utility_mass_mode={gain,point,query}`.
- Kept eval unchanged: retained masks are frozen before held-out eval query
  scoring.

Validation:

- Targeted tests passed for teacher/target/config plumbing:
  `test_teacher_distillation.py`, `test_torch_runtime_controls.py`,
  `test_benchmark_runner.py` (`46 passed`, one PyTorch nested-tensor warning).
- Synthetic smoke run passed:
  `set_utility_smoke_20260514`.

Focused 30% coverage experiment:

| Artifact | Change | Result |
| --- | --- | --- |
| `blind_real_true_c30_setutility_gain_t30_div002_20260514` | set-utility target, gain mass, candidate limit `128`, temporal fill `30%`, diversity `0.02` | Failed badly. Matched `MLQDS=0.1914`, uniform `0.2820`, DP `0.2263`; target positives `8.14%`, target mass `0.4762`. |

Component diagnosis at `5%` compression:

- MLQDS trails uniform on `RangePointF1` (`0.099` vs `0.111`),
  `ShipF1` (`0.561` vs `0.841`), `TemporalCov` (`0.187` vs `0.404`),
  `GapCov` (`0.148` vs `0.355`), `ShapeScore` (`0.120` vs `0.218`),
  and length preservation (`0.832` vs `0.888`).
- It only beats DP on `RangePointF1`; it loses DP on aggregate usefulness and
  most continuity components.
- Target construction took `38.71s` on only `120` segments. That is already a
  meaningful iteration bottleneck, and the quality does not justify optimizing
  this exact path yet.

Conclusion: exact one-step marginal `RangeUseful` labels are too sparse and
too weak for the current blind student. This result is worse than ordinary
retained-frequency, so the next experiment should be a single broad-coverage
teacher-distillation diagnostic, not another set-utility mass-normalization
tweak.

### Teacher-distillation broad-coverage diagnostic

Protocol note:

- The comparable run must use the cached kilometer/hour workload profile:
  `range_spatial_km=2.2`, `range_time_hours=5.0`, `max_queries=512`, and
  `cache_dir=artifacts/cache`. Re-running with only fraction-based footprints
  produced a non-comparable fixed 24-query workload around `51%` eval coverage.
  Treat those accidental runs as invalid diagnostics.

Focused 30% coverage experiment:

| Artifact | Change | Result |
| --- | --- | --- |
| `blind_real_true_c30_teacherfreq_t30_div002_exact_20260514` | query-aware `range_aware` teacher, retained-frequency distillation, full-target student labels, temporal fill `30%`, diversity `0.02` | Failed. Matched `MLQDS=0.2130`, uniform `0.2820`, DP `0.2263`; teacher labels positive fraction `39.15%`, target mass `2309.43`. |

Audit grid:

| Compression | MLQDS | Uniform | DP | MLQDS vs uniform |
| --- | ---: | ---: | ---: | ---: |
| `1%` | `0.1131` | `0.1119` | `0.1111` | `+0.0012` |
| `2%` | `0.1366` | `0.1937` | `0.1556` | `-0.0571` |
| `5%` | `0.2130` | `0.2820` | `0.2263` | `-0.0690` |
| `10%` | `0.3201` | `0.3742` | `0.3330` | `-0.0542` |
| `15%` | `0.3787` | `0.4470` | `0.4060` | `-0.0683` |
| `20%` | `0.4610` | `0.5249` | `0.4745` | `-0.0639` |
| `30%` | `0.5399` | `0.6045` | `0.5788` | `-0.0647` |

Component diagnosis at `5%` compression:

- MLQDS trails uniform on `RangePointF1` (`0.101` vs `0.111`),
  `ShipF1` (`0.652` vs `0.841`), `TemporalCov` (`0.218` vs `0.404`),
  `GapCov` (`0.198` vs `0.355`), `ShapeScore` (`0.126` vs `0.218`),
  and length preservation (`0.859` vs `0.888`).
- It trails TemporalRandomFill on both aggregate usefulness (`0.213` vs
  `0.263`) and point F1 (`0.101` vs `0.106`).
- The student overfits the teacher-distilled objective relative to validation:
  training loss keeps improving, while validation `RangeUseful` peaks early and
  remains below uniform.

Conclusion: teacher-retained labels do not solve the broad blind compression
problem. They improve the 1% cell slightly, but the learned retained set still
has the same continuity failure. At this point, aggregated point labels,
retained-frequency labels, component labels, residual anchors, exact one-step
set utility, and teacher-retained distillation all fail the 30% coverage cell.
The remaining issue is probably representational or architectural: the current
pointwise blind scorer does not learn a retained-set structure that competes
with uniform temporal continuity at broad range workloads.

### Stratified selector checkpoint

Implementation changes:

- Added `mlqds_hybrid_mode=stratified`.
- The selector splits each trajectory into retained-budget temporal/index
  strata and chooses the highest learned-score point inside each stratum, with
  endpoints fixed when possible.
- This is still workload-blind: eval queries are not passed into the model,
  feature builder, or selector before compression. Retained masks are still
  frozen before held-out query scoring.

Validation:

- Focused tests passed for simplification, CLI/config profile plumbing, and
  benchmark rows: `test_metrics.py`, `test_training_losses.py`,
  `test_torch_runtime_controls.py`, `test_benchmark_runner.py`
  (`76 passed` after removing the bad stratified-loss branch).

Focused experiments:

| Artifact | Change | Result |
| --- | --- | --- |
| `blind_real_true_c30_retfreq_t30_div002_tempcdf020_20260514` | retained-frequency baseline, temporal-CDF loss `0.20` | Failed. Matched `MLQDS=0.2321`, uniform `0.2820`, DP `0.2263`; better geometry, still not useful enough. |
| `blind_real_true_c30_retfreq_stratified_div002_20260514` | retained-frequency target with stratified selector | Best 30% broad result so far but still failed. Matched `MLQDS=0.2634`, uniform `0.2820`, DP `0.2263`; length preservation `0.885`, close to uniform `0.888`. |
| `blind_real_true_c30_retfreq_stratified_loss_div002_20260514` | experimental stratified per-bin soft loss | Failed and removed from code. Matched `MLQDS=0.2547`; training runtime rose to `51.9s` because the loss was Python-loop heavy. |
| `blind_real_true_c15_retfreq_stratified_div002_20260514` | 15% coverage, stratified selector | Failed. Matched `MLQDS=0.2565`, uniform `0.2762`, DP `0.2460`; no real improvement over the old 15% retained-frequency run. |
| `blind_real_true_c30_teacherfreq_stratified_div002_20260514` | query-aware teacher retained-frequency labels with stratified selector | Failed. Matched `MLQDS=0.2658`, uniform `0.2820`, DP `0.2263`; teacher signal helps only marginally. |
| `blind_real_true_c30_pointvalue_stratified_div002_20260514` | raw point-value usefulness labels with stratified selector | Failed. Matched `MLQDS=0.2447`, uniform `0.2820`, DP `0.2263`; validation peaked below uniform and `43/120` zero-positive training windows were filtered. |
| `blind_real_true_c30_retfreq_stratified_binloss015_div002_20260514` | vectorized local stratum cross-entropy, weight `0.15` | Failed and removed from code. Matched `MLQDS=0.2598`, uniform `0.2820`, DP `0.2263`; slightly improved the 2% cell but regressed the matched 5% cell versus plain stratified retained-frequency. |
| `blind_real_true_c30_retfreq_stratified_binloss005_div002_20260514` | vectorized local stratum cross-entropy, weight `0.05` | Failed and removed from code. Matched `MLQDS=0.2530`, uniform `0.2820`, DP `0.2263`; lower weight was worse, so the bin-local CE idea is not worth keeping as a stale knob. |
| `blind_real_true_c30_retfreq_stratified_rankbce512_div002_20260514` | retained-frequency target with `ranking_bce`, `512` sampled pairs/window | Failed. Matched `MLQDS=0.2609`, uniform `0.2820`, DP `0.2263`; pairwise ranking improved crossing at 5% but lost temporal/gap coverage and produced worse global geometry than plain budget-top-k. |
| `blind_real_true_c30_rangeprior_retfreq_stratified_div002_20260514` | `range_prior` full query-free context features with retained-frequency stratified selector | Strong single-workload-label result. Matched `MLQDS=0.2789`, uniform `0.2820`, DP `0.2263`; wins 1%, 10%, and 15% compression at 30% coverage, but still loses 2%, 5%, 20%, and 30%. |
| `blind_real_true_c30_rangeprior_teacherfreq_stratified_div002_20260514` | `range_prior` with query-aware teacher retained-frequency labels | Failed. Matched `MLQDS=0.2740`, uniform `0.2820`, DP `0.2263`; teacher labels degrade the stronger blind feature model. |
| `blind_real_true_c15_rangeprior_retfreq_stratified_div002_20260514` | 15% coverage with `range_prior`, retained-frequency target, stratified selector | Failed narrowly. Matched `MLQDS=0.2732`, uniform `0.2762`, DP `0.2460`; wins the 2% and 10% compression cells versus uniform but not enough for a broad claim. |
| `blind_real_true_c30_rangeprior_componentblend035_stratified_div002_20260514` | `range_prior` with component-retained target blend `0.35` | Failed. Matched `MLQDS=0.2719`, uniform `0.2820`, DP `0.2263`; component blending diffused target mass and reduced the useful local ranking. |
| `blind_real_true_c30_rangeprior_retfreq_stratified_trainrep4_labelmean_div002_20260514` | `range_prior` retained-frequency target after label-mean aggregation across 4 train workload seeds | Current best 30% result, but still not acceptance. Matched `MLQDS=0.2824`, uniform `0.2820`, DP `0.2263`; wins the 5%, 10%, and 15% cells, loses 1%, 2%, 20%, and 30%. The matched win is only `+0.0004`, so it is not a confident model win. |
| `blind_real_true_c15_rangeprior_retfreq_stratified_trainrep4_labelmean_div002_20260514` | 15% coverage with the same `range_prior` trainrep4 label-mean target as the current 30% best | Failed. Matched `MLQDS=0.2735`, uniform `0.2762`, DP `0.2460`; wins 2%, 10%, and 30% versus uniform, loses 1%, 5%, 15%, and 20%. Replicate label-mean does not close the 15% coverage gap. |
| `blind_real_true_c30_rangeprior_retfreq_stratified_trainrep4_labelmean_rankconf015_div002_20260514` | same current 30% best setup, but inference score mode `rank_confidence` with weight `0.15` | Neutral. Matched `MLQDS=0.2824`, uniform `0.2820`, DP `0.2263`; audit grid is effectively unchanged from pure rank. The raw logit margin is not carrying a useful local-choice signal. |
| `blind_real_true_c30_rangeprior_retfreq_stratified_trainrep4_labelmean_lowbudgets_div002_20260514` | same as trainrep4 label mean, but retained-frequency target/loss only used `1%,2%,5%` budgets | Failed. Matched `MLQDS=0.2768`, uniform `0.2820`, DP `0.2263`; low-budget target was very sparse (`5.7%` positives) and still lost the 1%, 2%, and 5% cells. The all-ratio target is not failing merely because low budgets are underweighted. |
| `blind_real_true_c30_rangeprior_retfreq_stratified_trainrep4_frequencymean_div002_20260514` | same as above, but aggregates per-workload retained-frequency targets by `frequency_mean` | Failed. Matched `MLQDS=0.2720`, uniform `0.2820`, DP `0.2263`; positive target fraction rose from `43.5%` to `63.7%`, which looks like target diffusion rather than useful robustness. |
| `blind_real_true_c30_rangeprior_retfreq_stratified_trainrep8_labelmean_div002_20260514` | same as trainrep4 label mean, but with 8 train workload seeds | Failed. Matched `MLQDS=0.2701`, uniform `0.2820`, DP `0.2263`; more historical workloads did not improve transfer and made local ranking worse despite a lower positive target fraction (`40.2%`). |
| `blind_real_true_c30_rangeprior_retfreq_stratified_trainrep4_labelmean_pointblend005_div002_20260514` | diagnostic retained-frequency target blended with `5%` normalized expected-usefulness value | Failed and removed from code. Matched `MLQDS=0.2775`, uniform `0.2820`, DP `0.2263`; the blend inflated positive target fraction to `78.1%` and degraded the best trainrep4 result. |
| `blind_real_true_c30_rangeprior_retfreq_stratified_trainrep4_labelmean_pointblend015_div002_20260514` | diagnostic retained-frequency target blended with `15%` normalized expected-usefulness value | Failed and removed from code. Matched `MLQDS=0.2816`, uniform `0.2820`, DP `0.2263`; close, but still worse than the no-blend trainrep4 label-mean target and not worth keeping as a stale knob. |
| `blind_real_true_c30_rangeprior_rich_retfreq_stratified_trainrep4_labelmean_none_div002_20260514` | added query-free heading sine/cosine, adjacent movement bearing, normalized curvature, and gap-balance features to `range_prior` | Failed and reverted from code. Matched `MLQDS=0.2591`, uniform `0.2820`, DP `0.2263`; geometry also worsened (`LengthPres=0.874` vs uniform `0.888`). The extra directional features increased capacity/noise without improving blind transfer. |
| `blind_real_true_c30_rangeprior_rich_retfreq_stratified_trainrep4_labelmean_TEMPRESID_WRONGMODE_div002_20260514` | accidental rich-feature run with CLI default `temporal_residual_label_mode=temporal` | Invalid for comparison and renamed to avoid stale interpretation. It failed anyway: matched `MLQDS=0.2638`, uniform `0.2820`, DP `0.2263`. |
| `blind_real_true_c30_rangeprior_retfreq_swap30_trainrep4_labelmean_div002_20260514` | same trainrep4 label-mean target, but `mlqds_hybrid_mode=swap` with `30%` protected uniform share | Failed badly. Matched `MLQDS=0.2081`, uniform `0.2820`, DP `0.2263`; global distortion also worsened (`LengthPres=0.851`). This rules out swap as a legitimate fix and reinforces that temporal scaffold tricks are not enough. |
| `blind_real_true_c30_rangeprior_retfreq_stratified_trainrep4_labelmean_power120_div002_20260514` | temporary retained-frequency target power `1.20` to sharpen high-frequency budget labels | Failed and removed from code. Matched `MLQDS=0.2820`, just below uniform and below the no-power target; length preservation improved slightly but entry/exit and point F1 stayed weak. |
| `blind_real_true_c30_rangeprior_retfreq_stratified_trainrep4_labelmean_power150_div002_20260514` | temporary retained-frequency target power `1.50` | Failed and removed from code. Matched `MLQDS=0.2817`, uniform `0.2820`, DP `0.2263`; target mass dropped from `2309.43` to `1479.60` without improving low-budget ranking. |
| `blind_real_true_c30_rangeprior_retfreq_stratified_trainrep4_labelmean_selectc02_div002_20260514` | same best trainrep4 label-mean target, but checkpoint selection and matched scoring at `2%` compression | Failed. Validation briefly beat uniform at `2%`, but held-out eval was `MLQDS=0.1608`, uniform `0.1937`, DP `0.1556`. The low-budget failure is not mainly a 5%-checkpoint mismatch. |
| `blind_real_true_c30_rangedensity_retfreq_stratified_trainrep4_labelmean_div002_20260514` | temporary `range_density_prior` model with query-free spatial, trajectory-diversity, temporal, and spatiotemporal density features | Failed and removed from code. Validation overfit badly (`+0.0144` over uniform at epoch 2), but held-out eval matched `MLQDS=0.2720`, uniform `0.2820`, DP `0.2263`; density context reduced point F1 and did not generalize across AIS days. |

Cleanup:

- Removed the temporary `range_density_prior` code path after the failed
  diagnostic artifact. The progress log keeps the negative result; production
  feature builders and CLI choices no longer expose it.
- Updated direct CLI/config defaults for `budget_loss_ratios` and
  `budget_loss_temperature` to the active audit grid and `0.25`. The previous
  direct CLI defaults silently ran only `1%,2%,5%,10%` with temperature `0.10`,
  which was a stale-footgun for manual benchmarks.

30% stratified retained-frequency audit grid:

| Compression | MLQDS | Uniform | DP | MLQDS vs uniform |
| --- | ---: | ---: | ---: | ---: |
| `1%` | `0.0935` | `0.1119` | `0.1111` | `-0.0183` |
| `2%` | `0.1633` | `0.1937` | `0.1556` | `-0.0305` |
| `5%` | `0.2634` | `0.2820` | `0.2263` | `-0.0187` |
| `10%` | `0.3505` | `0.3742` | `0.3330` | `-0.0238` |
| `15%` | `0.4240` | `0.4470` | `0.4060` | `-0.0230` |
| `20%` | `0.4846` | `0.5249` | `0.4745` | `-0.0403` |
| `30%` | `0.5934` | `0.6045` | `0.5788` | `-0.0112` |

30% `range_prior` trainrep4 label-mean audit grid:

| Compression | MLQDS | Uniform | DP | MLQDS vs uniform |
| --- | ---: | ---: | ---: | ---: |
| `1%` | `0.1111` | `0.1119` | `0.1111` | `-0.0008` |
| `2%` | `0.1875` | `0.1937` | `0.1556` | `-0.0062` |
| `5%` | `0.2824` | `0.2820` | `0.2263` | `+0.0004` |
| `10%` | `0.3840` | `0.3742` | `0.3330` | `+0.0098` |
| `15%` | `0.4499` | `0.4470` | `0.4060` | `+0.0030` |
| `20%` | `0.4936` | `0.5249` | `0.4745` | `-0.0312` |
| `30%` | `0.5974` | `0.6045` | `0.5788` | `-0.0072` |

Component diagnosis at 30% coverage / 5% compression:

- Stratified MLQDS is now close to uniform on temporal coverage (`0.385` vs
  `0.404`) and gap coverage (`0.333` vs `0.355`), and it beats uniform on
  shape (`0.230` vs `0.218`).
- It still loses on `RangePointF1` (`0.099` vs `0.111`), entry/exit
  (`0.206` vs `0.244`), crossing (`0.104` vs `0.126`), and aggregate
  `RangeUseful`.
- TemporalRandomFill is still slightly better than MLQDS at the matched row
  (`0.2727` vs `0.2634`), so learned local choices inside bins are not yet
  reliably better than random local choices.

Conclusion: stratified selection is the first change that materially addresses
the broad continuity failure without eval-query leakage. It also keeps
trajectories much more sensible. It is still not final success because it does
not beat uniform at 15% or 30% coverage and does not beat TemporalRandomFill at
the 30% matched row. The next useful work should improve learned local choice
inside strata, not add more temporal structure. Raw point-value labels, pairwise
ranking, and a vectorized local stratum cross-entropy loss did not fix that
local-choice problem; all were weaker than retained-frequency with the plain
budget-top-k loss. Candidate directions should now move beyond simple
point-label/loss reweighting: features or pretraining that distinguish
entry/exit/crossing points inside a temporal bin, or a more structural model
objective.

### Full blind context checkpoint

Implementation change:

- `model_type="range_prior"` now uses the full 24-dimensional query-free
  context feature set. `model_type="workload_blind_range"` remains on the old
  17-dimensional feature slice for checkpoint compatibility.
- The added features are still workload-blind: trajectory-local distance
  position, reverse distance position, local time span around a point, local
  curvature, next heading/speed deltas, and max adjacent gap. No range query
  boxes or eval labels enter compression.

Conclusion: the 24-column `range_prior` feature set plus stratified selection is
the current useful blind path. Aggregating four training workload seeds by
label mean can barely clear uniform at the matched 30% coverage / 5%
compression cell, but the win is too small and the low-budget cells still fail.
Teacher labels, component-retained blending, pairwise ranking, bin-local losses,
low-budget-only retained targets, point-value blending, and richer direction
features, temporary density priors, swap selection, retained-target power
sharpening, and 2%-targeted checkpoint selection do not fix the remaining
problem. The remaining failure is not protocol
leakage; retained masks are frozen before eval scoring. It is a blind
local-ranking problem: the student still under-selects future range point hits,
entry/exit points, and crossings at low budgets.

### Held-out workload-generator setting checkpoint

Implementation change:

- Added explicit range anchor priors via
  `range_anchor_mode={mixed_density,dense,uniform,sparse}`.
- `mixed_density` preserves the historical generator behavior: a 70% spatial
  density-biased / 30% uniform anchor mix.
- `dense`, `uniform`, and `sparse` expose held-out generator settings for
  generalization checks without changing the workload-blind compression
  protocol.
- Wired the mode through config, CLI, workload cache keys, benchmark profiles,
  benchmark reports, inference, coverage estimation, and query diagnostics.
- Inference artifacts now record both `query_config` and workload-generation
  diagnostics so held-out generator settings are auditable.

Validation:

- Focused tests passed:
  `test_query_coverage_generation.py`, `test_torch_runtime_controls.py`,
  `test_benchmark_runner.py` (`48 passed`).
- Re-ran the current best 30% `range_prior` trainrep4 label-mean setup with
  explicit `range_anchor_mode=mixed_density` and saved a checkpoint:
  `blind_real_true_c30_rangeprior_retfreq_stratified_trainrep4_labelmean_anchorctl_div002_20260514`.
  It reproduced the previous matched result: `MLQDS=0.2824`, uniform
  `0.2820`, DP `0.2263`.
- Evaluated that saved mixed-density model on the held-out eval day with
  `range_anchor_mode=sparse`, seed `47`, and 30% coverage:
  `blind_real_true_c30_rangeprior_retfreq_stratified_trainrep4_labelmean_anchorctl_sparseeval_20260514`.

Sparse-anchor held-out result at 5% compression:

| Method | RangeUseful | RangePointF1 | ShipF1 | ShipCov | EntryExitF1 | CrossingF1 | TemporalCov | GapCov | GapTime | GapDist | TurnCov | ShapeScore | LengthPres |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MLQDS | `0.1747` | `0.1222` | `0.4996` | `0.1286` | `0.1585` | `0.1075` | `0.1340` | `0.1391` | `0.1469` | `0.1579` | `0.1185` | `0.0889` | `0.8935` |
| uniform | `0.1788` | `0.1242` | `0.5326` | `0.1299` | `0.1622` | `0.1169` | `0.1249` | `0.1370` | `0.1558` | `0.1628` | `0.1092` | `0.0837` | `0.8881` |
| DP | `0.1760` | `0.1251` | `0.4783` | `0.1331` | `0.1751` | `0.1213` | `0.1235` | `0.1197` | `0.1323` | `0.1450` | `0.1572` | `0.0866` | `0.9306` |

Conclusion: the current best blind model does not generalize to a sparse
background-anchor workload prior. It loses to uniform by `0.0042` and to DP by
`0.0013` on `RangeUseful`. This is not a protocol leak or selector artifact;
compression is still query-blind. It is a real held-out workload-generator
failure. The model has learned a useful mixed-density local ranking, but that
ranking is brittle when future queries are less density-centered.

### Train anchor-prior mixture checkpoint

Implementation change:

- Added `range_train_anchor_modes`, an optional comma-separated/list config that
  cycles anchor priors across train label workload replicates.
- Empty `range_train_anchor_modes` preserves current behavior. Eval and
  checkpoint-selection workloads still use `range_anchor_mode`.
- Workload cache keys use the effective anchor mode for each generated
  workload, so mixed train priors do not collide with same-seed mixed-density
  workloads.
- Benchmark rows and run configs now record `range_train_anchor_modes`.

Validation:

- Focused tests passed:
  `test_query_coverage_generation.py`, `test_torch_runtime_controls.py`,
  `test_benchmark_runner.py` (`49 passed`).

Focused experiment:

- Artifact:
  `blind_real_true_c30_rangeprior_retfreq_stratified_trainrep4_anchormix_labelmean_div002_20260514`
- Change: same current-best 30% `range_prior` retained-frequency stratified
  setup, but four training workloads used
  `mixed_density,sparse,uniform,dense`; eval stayed `mixed_density`.
- Train workload query counts changed from `116/132/122/124` under repeated
  mixed-density seeds to `116/350/157/107` under the anchor-prior mixture.
- Raw aggregated training labels became broader: positive raw-label fraction
  `65.7%` versus `55.3%` for repeated mixed-density train seeds.
- Retained-frequency target positives became slightly narrower:
  `40.8%` versus `43.5%`.

Mixed-density eval result:

| Compression | MLQDS | Uniform | DP | MLQDS vs uniform |
| --- | ---: | ---: | ---: | ---: |
| `1%` | `0.1078` | `0.1119` | `0.1111` | `-0.0041` |
| `2%` | `0.1868` | `0.1937` | `0.1556` | `-0.0070` |
| `5%` | `0.2821` | `0.2820` | `0.2263` | `+0.0001` |
| `10%` | `0.3844` | `0.3742` | `0.3330` | `+0.0101` |
| `15%` | `0.4475` | `0.4470` | `0.4060` | `+0.0005` |
| `20%` | `0.4935` | `0.5249` | `0.4745` | `-0.0313` |
| `30%` | `0.6001` | `0.6045` | `0.5788` | `-0.0044` |

Sparse-anchor held-out eval for the mixed-prior-trained checkpoint:

| Method | RangeUseful | RangePointF1 | ShipF1 | ShipCov | EntryExitF1 | CrossingF1 | TemporalCov | GapCov | GapTime | GapDist | TurnCov | ShapeScore | LengthPres |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MLQDS | `0.1733` | `0.1211` | `0.4983` | `0.1278` | `0.1560` | `0.1049` | `0.1313` | `0.1381` | `0.1467` | `0.1567` | `0.1184` | `0.0881` | `0.8933` |
| uniform | `0.1788` | `0.1242` | `0.5326` | `0.1299` | `0.1622` | `0.1169` | `0.1249` | `0.1370` | `0.1558` | `0.1628` | `0.1092` | `0.0837` | `0.8881` |
| DP | `0.1760` | `0.1251` | `0.4783` | `0.1331` | `0.1751` | `0.1213` | `0.1235` | `0.1197` | `0.1323` | `0.1450` | `0.1572` | `0.0866` | `0.9306` |

Conclusion: anchor-prior mixture is not a robustness fix. It preserves the same
tiny mixed-density 5% matched win, loses the low-budget cells, and makes sparse
held-out eval slightly worse (`0.1733` vs the previous sparse `0.1747`). This
rules out the simple "train on more generator settings" hypothesis for the
current retained-frequency/stratified model. The problem remains local ranking
under blind compression, not merely train-workload prior coverage.

### Sparse-specific and aggregation diagnostics

Focused sparse-specific experiment:

- Artifact:
  `blind_real_true_c30_rangeprior_retfreq_stratified_trainrep4_sparse_labelmean_div002_20260514`
- Change: same `range_prior` retained-frequency stratified setup, but train,
  validation, and eval workloads all used `range_anchor_mode=sparse`.
- Train/eval query counts rose sharply because sparse anchors cover less
  densely: train label workloads used `385/350/380/359` queries, eval used
  `257`, and selection used `439`.
- Target positive fraction was `40.2%`; raw aggregated sparse labels covered
  `74.1%` of points.
- Runtime was `47.9s`; range label prep plus diagnostics dominated because
  sparse workloads have many more queries.

Sparse eval audit:

| Compression | MLQDS | Uniform | DP | MLQDS vs uniform |
| --- | ---: | ---: | ---: | ---: |
| `1%` | `0.0969` | `0.0938` | `0.0909` | `+0.0031` |
| `2%` | `0.1302` | `0.1280` | `0.1263` | `+0.0022` |
| `5%` | `0.1922` | `0.1839` | `0.1938` | `+0.0083` |
| `10%` | `0.2839` | `0.2857` | `0.2763` | `-0.0018` |
| `15%` | `0.3706` | `0.3637` | `0.3552` | `+0.0069` |
| `20%` | `0.4292` | `0.4409` | `0.4149` | `-0.0118` |
| `30%` | `0.5565` | `0.5500` | `0.5330` | `+0.0064` |

Cross-prior check:

- Evaluating this sparse-trained checkpoint on the mixed-density eval workload
  produced `MLQDS=0.2786`, uniform `0.2820`, DP `0.2263`.
- The sparse-specific model therefore learns a useful sparse prior but gives up
  the already fragile mixed-density result.

Aggregation check:

- Artifact:
  `blind_real_true_c30_rangeprior_retfreq_stratified_trainrep4_anchormix_frequencymean_div002_20260514`
- Change: same anchor-prior mixture as above, but
  `range_replicate_target_aggregation=frequency_mean`.
- Target positive fraction jumped to `67.9%`.
- Mixed-density eval failed: matched `MLQDS=0.2740`, uniform `0.2820`, DP
  `0.2263`.
- Audit wins versus uniform only at `10%`; it loses `1%,2%,5%,15%,20%,30%`.

Conclusion: the current model can learn prior-specific sparse behavior, but the
single blind scorer does not reconcile sparse and density-centered priors under
the current label aggregation. `label_mean` is too brittle; `frequency_mean`
diffuses target mass and weakens mixed-density ranking. A robust final model
likely needs an explicit multi-prior objective or architecture that learns
stable retained-set roles across priors, not more averaging of per-point labels.

### Rank-mean aggregation check

- Added a temporary `rank_mean` aggregation diagnostic that averaged
  per-workload positive-label percentile ranks before retained-frequency target
  selection.
- Artifact:
  `blind_real_true_c30_rangeprior_retfreq_stratified_trainrep4_anchormix_rankmean_div002_20260514`
- Mixed-density eval failed narrowly: matched `MLQDS=0.2818`, uniform
  `0.2820`, DP `0.2263`.
- Audit wins versus uniform only at `10%` and barely at `15%`; it loses
  `1%,2%,5%,20%,30%`.
- Sparse held-out eval also failed: `MLQDS=0.1745`, uniform `0.1788`, DP
  `0.1760`.

Conclusion: rank-normalizing labels does not solve the multi-prior problem. It
behaves like a weaker version of `label_mean`: target positives stay moderate
(`40.3%`), but low-budget and sparse generalization still fail. Removed the
temporary `rank_mean` code path after recording this result to avoid keeping a
stale experiment knob.

### Held-out anchor-prior matrix notes

Evaluated the current best mixed-density checkpoint
`blind_real_true_c30_rangeprior_retfreq_stratified_trainrep4_labelmean_anchorctl_div002_20260514`
against additional held-out anchor settings at 30% coverage / 5% compression,
seed `47`:

| Held-out anchor mode | Query count | MLQDS | Uniform | DP | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| `dense` | `64` | `0.2893` | `0.2893` | `0.2683` | Tiny MLQDS win over uniform: `+0.00009`. |
| `uniform` | `97` | `0.2264` | `0.2255` | `0.1880` | Small MLQDS win over uniform: `+0.0009`. |
| `sparse` | `251` | `0.1747` | `0.1788` | `0.1760` | Clear failure versus both baselines. |

Conclusion: the held-out generator failure is concentrated in sparse/background
anchor workloads. The model mostly ties or barely beats uniform on dense and
uniform-anchor settings, but sparse anchors expose the lack of robust
background-coverage ranking. This is also the most expensive generator setting,
because it needs far more queries to reach the same coverage target.

### Raw-label max, component, ensemble, and teacher diagnostics

Implementation changes:

- Added `range_replicate_target_aggregation=label_max`, matching the design
  reference's max-usefulness aggregation variant.
- Added multi-replicate support for `component_retained_frequency` targets.
  Component streams are aggregated separately before retained-frequency target
  conversion, or per-replicate component targets can be averaged.
- Added multi-replicate teacher-student distillation support. Query-aware
  `range_aware` teachers are trained only on training workload replicates, then
  their retained-frequency labels are aggregated for the workload-blind
  student. Eval queries are still not passed into compression.

Focused experiments:

| Artifact / diagnostic | Change | Result |
| --- | --- | --- |
| `blind_real_true_c30_rangeprior_retfreq_stratified_trainrep4_anchormix_labelmax_div002_20260514` | Anchor-prior mixture with raw-label max before retained-frequency target selection. | Mixed-density matched `RangeUseful=0.282120`, uniform `0.282023`, DP `0.226293`; effectively the same tiny tie as `label_mean`. Low-budget audit still loses `1%` and `2%`. |
| `blind_real_true_c30_rangeprior_retfreq_stratified_trainrep4_anchormix_labelmax_sparseeval_20260514` | Same checkpoint on held-out sparse-anchor eval, seed `47`. | Failed: MLQDS `0.174454`, uniform `0.178830`, DP `0.175960`. Loss is driven by lower `ShipF1` (`0.5023` vs uniform `0.5326`) and lower crossing/boundary scores. |
| Query-blind checkpoint ensemble diagnostic | Combined current mixed-density and sparse-specific checkpoint rank scores before frozen-mask selection. | Failed sparse held-out: best tested sparse score was `0.176913` (`70%` sparse-specific / `30%` mixed), still below uniform `0.178830`. This suggests a multi-head scorer will not automatically fix sparse/background ship presence. |
| `blind_real_true_c30_rangeprior_componentrf_stratified_trainrep4_anchormix_labelmean_div002_20260514` | Multi-prior component-retained-frequency target with raw component-label mean. | Failed matched eval: MLQDS `0.271640`, uniform `0.282023`, DP `0.226293`. Target positives diffused to `73.6%` of points. |
| `blind_real_true_c30_rangeprior_teacherretfreq_stratified_trainrep4_anchormix_frequencymean_div002_20260514` | Four `range_aware` teachers, one per train anchor prior, retained-frequency distillation, mean target aggregation. | Failed matched eval: MLQDS `0.271371`, uniform `0.282023`, DP `0.226293`. Distilled positives diffused to `85.5%` of points. |

Conclusion: raw-label upper envelopes, component-separated targets, simple
score ensembling, and multi-prior retained-frequency teachers all fail in the
same way: they broaden the target or rank signal without improving the
query-blind local choice enough to beat uniform. The strongest current evidence
is now against more independent point-label aggregation. The bottleneck is a
structural objective/model issue: sparse/background range workloads reward
per-trajectory coverage of future query boxes, while the current single scorer
cannot identify those future hit points reliably from the available blind
features.

Validation:

- Focused target/protocol tests passed after the implementation changes.
- Full suite passed: `213 passed, 1 warning`.
- `git diff --check` passed.

### Target-fit and direct pointwise objective diagnostics

Diagnostics added:

- `artifacts/manual/range_label_stratified_oracle_diagnostic_20260514/diagnostic.json`
  uses eval usefulness labels as query-aware diagnostic scores, then applies the
  same query-blind selector forms. This is not a valid final method, but it
  tests selector capacity.
- `artifacts/manual/current_best_label_fit_diagnostic_20260514.json` compares
  the current best mixed-density checkpoint score to raw held-out usefulness
  labels across anchor priors.
- `artifacts/manual/current_best_exact_target_fit_diagnostic_20260514.json`
  reconstructs the exact four-replicate retained-frequency training target for
  the current best checkpoint and measures target-mass capture under the same
  frozen selector.
- Added `loss_objective=pointwise_bce`, a direct soft-label objective using all
  valid supervised points. This is a fit diagnostic, not a new target family.
- For this objective, zero-label windows with valid labels are kept instead of
  being filtered by the positive-window prefilter. Otherwise the diagnostic
  would silently stop being "all valid supervised points" on sparse targets.

Selector-capacity diagnostic at 30% coverage / 5% compression:

| Held-out anchor mode | Eval-label stratified | Eval-label fill | Uniform | DP |
| --- | ---: | ---: | ---: | ---: |
| `mixed_density` | `0.453552` | `0.529221` | `0.257056` | `0.240323` |
| `sparse` | `0.373817` | `0.360254` | `0.178830` | `0.175960` |
| `uniform` | `0.461815` | `0.543552` | `0.225512` | `0.1880` |
| `dense` | `0.485376` | `0.544086` | `0.289250` | `0.2683` |

Conclusion: the selector can beat uniform and DP if supplied with useful
scores. The bottleneck is learned blind score quality, not retained-mask
freezing or selector capacity.

Current-best exact training-target fit:

- Correlation between current-best rank scores and the exact retained-frequency
  target is weak: Pearson all-active `0.0851`, Spearman all-active `0.0396`,
  sampled Kendall all-active `0.0430`.
- At 5% compression, current-best model target mass is `327.57`, uniform is
  `332.43`, while the same selector using exact target scores gets `757.29`.
- At 1% compression, current-best model mass is `250.43`, uniform is `249.86`,
  target-score selector is `308.86`.
- At 30% compression, current-best model mass is `876.43`, uniform is
  `828.43`, target-score selector is `1932.86`.

Conclusion: the current best model barely fits its own training target at low
budgets. That is a stronger diagnosis than the held-out sparse failure: the
student is not merely failing to transfer; it is failing to learn the retained
target well enough on the training day.

Focused pointwise objective run:

| Artifact | Change | Result |
| --- | --- | --- |
| `blind_real_true_c30_rangeprior_retfreq_stratified_trainrep4_labelmean_pointwisebce_div002_20260514` | Current best 30% setup, but `loss_objective=pointwise_bce` | Failed. Matched `RangeUseful=0.268446`, uniform `0.282023`, DP `0.226293`. It wins only the 15% cell versus uniform and essentially ties at 30%; it loses 1/2/5/10/20%. |

Pointwise exact-target fit:

- Pearson all-active `0.0812`, Spearman all-active `0.0334`, sampled Kendall
  all-active `0.0287`.
- At 5% compression, target mass improves only to `336.86` versus uniform
  `332.43`, still far below target-score selector `757.29`.
- Eval component losses at 5% are not subtle: MLQDS trails uniform on
  `RangePointF1` (`0.1047` vs `0.1110`), entry/exit (`0.2018` vs `0.2439`),
  crossing (`0.0954` vs `0.1263`), temporal coverage (`0.3490` vs `0.4036`),
  and shape (`0.2058` vs `0.2178`).

Conclusion: direct pointwise BCE does not fix target fit or final usefulness.
The failure is not just the budget-top-k loss. The current query-free feature
set and pointwise/windowed scorer are too weak for the broad retained-frequency
target. Further independent scalar label aggregation or BCE-style objectives
are low-value.

Validation:

- Targeted pointwise/filter tests passed.
- Full suite passed: `216 passed, 1 warning`.
- `git diff --check` passed.

### Historical-prior workload-blind model diagnostic

Implementation changes:

- Added `model_type=historical_prior`, a query-free KNN scorer over normalized
  train-day route-context features and retained-frequency targets.
- Integrated it into checkpoint save/load, CLI/config, model-feature inference,
  and the normal frozen-mask evaluation path.
- The model compresses eval trajectories before eval query scoring. Eval
  queries are not passed into model features, scoring, checkpoint state, or
  retained-mask construction.

Focused exact-target diagnostic:

- `artifacts/manual/historical_prior_knn_diagnostic_20260514.json` showed that a
  route-context KNN prior can recover signal the neural student misses: at 30%
  coverage / 5% compression, KNN scored `RangeUseful=0.287499`, uniform
  `0.282023`, DP `0.226293`.
- The same diagnostic across coverage targets gave wins versus uniform in
  `19/28` cells and versus DP in `21/28` cells. Low-budget wins versus uniform
  were `3/4` at 1%, `1/4` at 2%, and `4/4` at 5%.

Clean pipeline results with `historical_prior_k=32`:

| Coverage target | Eval coverage | 5% MLQDS | 5% uniform | 5% DP | Audit wins vs uniform | Audit wins vs DP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5% | 6.77% | `0.285687` | `0.279080` | `0.230046` | `5/7` | `7/7` |
| 10% | 12.78% | `0.304379` | `0.271277` | `0.247207` | `6/7` | `6/7` |
| 15% | 15.19% | `0.286377` | `0.276161` | `0.245976` | `5/7` | `5/7` |
| 30% | 30.12% | `0.289026` | `0.282023` | `0.226293` | `4/7` | `6/7` |

Aggregate grid result: wins versus uniform in `20/28` cells and versus DP in
`24/28` cells. Low-budget cells (`1%,2%,5%`) win versus uniform in `9/12` and
versus DP in `11/12`. This is materially better than the neural range-prior
checkpoints, but it still fails important cells: 30% coverage at 1% and 2%
compression loses to uniform, and several 20%/30% high-ratio cells lose because
uniform preserves ship/gap coverage better.

Held-out seed/settings results at 30% coverage, seed `47`, `k=32`:

| Eval anchor mode | 5% MLQDS | 5% uniform | 5% DP | Audit wins vs uniform | Audit wins vs DP |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mixed_density` | `0.265202` | `0.268905` | `0.242044` | `2/7` | `6/7` |
| `dense` | `0.276691` | `0.285668` | `0.243850` | `1/7` | `6/7` |
| `uniform` | `0.196135` | `0.207935` | `0.188440` | `3/7` | `5/7` |
| `sparse` | `0.188897` | `0.165128` | `0.171306` | `4/7` | `6/7` |

Training-label anchor cycling across
`mixed_density,dense,uniform,sparse` did not solve robustness. It made
seed-47 mixed nearly tie uniform (`0.268488` vs `0.268905`) but hurt sparse and
still lost the dense/uniform anchor modes.

KNN smoothing diagnostic:

- Seed-47 mixed with `k=64` improves the 5% cell to `0.271710` versus uniform
  `0.268905` and DP `0.242044`, with audit wins `4/7` versus uniform and `7/7`
  versus DP.
- The same `k=64` is not a universal fix: seed-42 mixed drops from `0.289026`
  to `0.282540`, seed-47 dense loses badly (`0.260022` vs uniform `0.285668`),
  and seed-47 uniform also loses (`0.186682` vs uniform `0.207935`).

Selector variation:

- `mlqds_hybrid_mode=swap` is a dead end for this prior. Seed-42 mixed drops
  from `0.289026` to `0.220416`, loses every audit ratio versus uniform and DP,
  and raises MLQDS latency from about `548 ms` to `2191 ms`.
- Seed-47 mixed/dense/uniform also fail under `swap` (`RangeUseful=0.208075`,
  `0.214788`, and `0.165222` respectively). The failure is not subtle:
  geometry distortion jumps above `3 km` average SED on seed-47 mixed/dense.

Train-workload and target follow-ups:

- Added `range_train_footprints`, a train-only footprint-family list such as
  `1.1:2.5,2.2:5.0,4.4:10.0`, cycled across train workload replicates. Eval and
  checkpoint-selection workloads keep the configured final footprint, so this
  does not alter the blind evaluation protocol.
- Seed-47 mixed with six train footprint replicates nearly ties uniform at 5%
  (`0.268356` vs `0.268905`) but still loses. Dense gets worse
  (`0.265895` vs uniform `0.285668`). Footprint diversity alone is not the
  missing robustness.
- Historical-prior raw point-value targets are worse than retained-frequency:
  seed-47 mixed drops to `0.248863`, dense to `0.255877`. The raw expected
  usefulness target diffuses label mass and weakens the local ranking.
- The first two-day historical training check used `max_segments=240` globally,
  which changed train, validation, and eval size together. That comparison was
  not a valid answer to the training-day-diversity question.
- Added split-specific segment caps:
  `--train_max_segments`, `--validation_max_segments`, and
  `--eval_max_segments`. Explicit split-CSV runs now use train/eval caps
  independently while `--csv_path` keeps the global cap.
- Corrected two-day historical training check with train cap `240` and fixed
  validation/eval caps `120`:

| Eval anchor | Train source | 5% MLQDS | 5% uniform | 5% DP | Audit wins vs uniform | Audit wins vs DP | Low-budget wins vs uniform | Low-budget wins vs DP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mixed_density` | day 02 | `0.263483` | `0.268905` | `0.242044` | `2/7` | `7/7` | `1/3` | `3/3` |
| `mixed_density` | days 02+03 | `0.276636` | `0.268905` | `0.242044` | `5/7` | `7/7` | `2/3` | `3/3` |
| `dense` | day 02 | `0.277409` | `0.285668` | `0.243850` | `3/7` | `6/7` | `0/3` | `2/3` |
| `dense` | days 02+03 | `0.263860` | `0.285668` | `0.243850` | `0/7` | `5/7` | `0/3` | `2/3` |

  Mixed-density now supports the idea that extra historical train-day diversity
  can help when eval is fixed. Dense contradicts it: the two-day prior reduces
  `RangePointF1`, `ShipF1`, `EntryExitF1`, crossing, gap coverage, and
  geometry. This is still not final success.
- Two-day train-anchor cycling across
  `mixed_density,dense,uniform,sparse` is not a fix. Mixed falls back below
  uniform at 5% (`0.265667` vs `0.268905`) with only `3/7` audit wins versus
  uniform, while dense remains far below uniform at 5% (`0.267805` vs
  `0.285668`). The sparse train replicate also failed the requested 30%
  coverage cap (`24.57%` after `512` queries), so this setting is both weaker
  and more expensive.
- Query-blind temporal target blending for dense day-02 training does not fix
  the ship/point deficit. Blend `0.20` scores `0.272265` at 5% versus uniform
  `0.285668`; blend `0.40` drops to `0.267471`. It spreads label positives to
  `75.15%` of train points and dilutes the retained-frequency signal.
- Historical-prior set-utility targets also fail. Seed-47 mixed drops to
  `0.250852`; dense improves slightly to `0.282289` but still loses uniform
  `0.285668`. The set-utility target has very low mass (`~0.46-0.49`) and does
  not transfer robustly from query-free context.

Historical-prior density/sparsity diagnostic:

- Added two query-free spatial context features to `historical_prior`: local
  grid density and normalized sparsity. Old 19-dimensional checkpoints remain
  loadable; new checkpoints use 21 dimensions. Added
  `historical_prior_density_weight` to control how strongly the KNN distance
  uses those two dimensions.
- Default density weight `1.0` is only a small change. Dense day-02 improves
  from `0.277409` to `0.278607` at 5% compression, still below uniform
  `0.285668`. Mixed days 02+03 drops slightly from `0.276636` to `0.274817`,
  still above uniform `0.268905`.
- Weight `4.0` is worse. Dense falls to `0.274843`; mixed days 02+03 falls to
  `0.272817`. Overweighting density moves the prior away from useful geometry
  and does not fix robustness.
- Weight `2.0` is the best dense variant measured so far: dense day-02 reaches
  `0.281235` at 5% versus uniform `0.285668` and DP `0.243850`. It still loses
  uniform because `RangePointF1` remains lower (`0.095325` vs uniform
  `0.100327`), despite better `EntryExitF1` (`0.249158` vs `0.227432`) and
  length preservation (`0.9003` vs `0.8881`). Latency remains high at about
  `620 ms`.
- Smaller neighbor counts under weight `2.0` are not a fix: dense `k=8` scores
  `0.279932`, and `k=16` scores `0.275357`.

Conclusion from this branch: density/sparsity helps diagnose the dense failure
but does not solve it. The remaining gap is still point/ship coverage at low
budgets, not entry/exit behavior. More KNN distance tuning is unlikely to meet
the acceptance bar.

Set-utility mass-mode follow-up:

- Re-ran dense set-utility with the new density features and weight `2.0`.
  `gain` mass is still weak: target positives `2167` (`7.12%`), target mass
  `0.4395`, matched 5% `RangeUseful=0.280722` versus uniform `0.285668`.
  It wins only high-ratio uniform cells (`20%`, `30%`) and loses the important
  `1%`, `2%`, and `5%` cells.
- `range_set_utility_mass_mode=point` fixes label mass numerically
  (`11.0188`) and improves the dense matched 5% cell:
  `MLQDS=0.288992`, uniform `0.285668`, DP `0.243850`. It improves
  `RangePointF1` (`0.103626` vs uniform `0.100327`) and `EntryExitF1`
  (`0.283205` vs `0.227432`), but still loses uniform at `1%`, `2%`, `10%`,
  and `15%`. The low-budget losses are ship/gap/temporal coverage, not absence
  of point labels.
- `range_set_utility_mass_mode=query` is worse on dense: matched 5%
  `0.280122` versus uniform `0.285668`.
- The point-mass gain does not generalize to mixed-density. With the same
  day-02 train cap and seed-47 mixed eval, matched 5% falls to `0.244629`
  versus uniform `0.268905` and barely above DP `0.242044`.

Conclusion from this branch: point-mass set-utility is a useful diagnostic but
not a final path. It can overfit the dense anchor prior enough to win one
matched cell, but it does not beat the low-budget grid and it breaks the mixed
held-out setting. The target builder also costs about `60 s` per run on this
setup, so further blind tuning here needs a stronger reason than another mass
knob.

Target mass and selector follow-up:

- Tested existing `mlqds_hybrid_mode=fill` with retained-frequency historical
  prior on the dense seed-47 day-02 setup. It is not a safe fallback. Matched
  5% falls to `0.229124` versus uniform `0.285668` and DP `0.243850`, with
  worse global geometry (`AvgSED=1.6372 km`). The learned fill is harming the
  temporal base, not adding useful residual behavior.
- Added `range_target_balance_mode=trajectory_unit_mass`, a training-only
  target transform that rescales each train trajectory's positive range-target
  mass to one after target construction. This tests whether the historical
  prior is dominated by a few dense routes while eval budgets remain per
  trajectory. It does not change eval compression or use eval queries.
- Trajectory-unit balancing worsens dense retained-frequency:
  `0.276352` versus uniform `0.285668`, DP `0.243850`. Target mass changes from
  `3814.1428` to exactly `240.0000` across `240` positive train trajectories,
  but the low-budget point/ship deficit remains.
- Trajectory-unit balancing also fails on mixed point-mass set-utility:
  `0.246896` versus uniform `0.268905`, DP `0.242044`. It is slightly better
  than the unbalanced mixed point-mass result (`0.244629`) but nowhere near
  useful.
- `range_replicate_target_aggregation=frequency_mean` is worse for dense
  historical retained-frequency (`0.268722` vs uniform `0.285668`) and diffuses
  positives to `60.26%`.
- `range_replicate_target_aggregation=label_max` is also not enough:
  dense matched 5% `0.280451` versus uniform `0.285668`.

Conclusion from this branch: simple target mass normalization, per-workload
frequency averaging, label upper envelopes, and fill-mode temporal scaffolding
do not solve robustness. The remaining problem is structural: the blind prior
can find entry/exit-like points, but cannot reliably choose low-budget points
that preserve ship, temporal, and gap coverage across generator priors.

Stratified center-regularizer diagnostic:

- Added `mlqds_stratified_center_weight`, default `0.0`, to the stratified
  selector. When nonzero, it subtracts a normalized center-distance penalty
  inside each retained-budget stratum. This is query-blind and is wired through
  validation, teacher/target retained-frequency construction, inference,
  benchmark profiles, saved simplified CSV fallback, and reporting.
- This is not a final-product mechanism by itself. It risks making the learned
  selector more uniform-like, so it must only count if learned scoring still
  beats temporal/random fill and the low-budget audit improves.
- Dense seed-47 day-02 historical prior with density weight `2.0`,
  retained-frequency labels, and `center_weight=0.10` gets worse:
  matched 5% `RangeUseful=0.270407` versus uniform `0.285668` and DP
  `0.243850`. It wins only the `1%` audit cell and loses `2%`, `5%`,
  `10%`, `15%`, `20%`, and `30%` to uniform. At 5%, TemporalRandomFill also
  beats MLQDS (`0.279328` vs `0.270407`), so the learned ranking is not helping
  enough under this selector.
- A smaller penalty is worse, not better: `center_weight=0.02` scores
  `0.264187` at matched 5%, below both the no-center density-weight-2 result
  (`0.281235`) and uniform (`0.285668`). It also worsens geometry:
  `AvgSED=0.6226 km` versus uniform `0.4843 km`.

Conclusion from this branch: center regularization is a dead end. It confirms
that the low-budget failure is not fixed by nudging selections toward stratum
centers; the model signal itself remains insufficient for dense held-out
workloads.

Clock-time and two-day prior diagnostics:

- Found a plausible implementation weakness: the historical-prior feature slice
  inherited trajectory-local normalized time and discarded absolute clock time.
  Since `anchor_day` range workloads are time-windowed and timestamps are known
  before compression, bounded clock-time features are workload-blind and legal.
- Added circular clock-time features to `historical_prior` and kept old 19-dim
  and density-only 21-dim checkpoints loadable. Added
  `historical_prior_clock_weight`, default `0.0`, so the old density-only
  behavior remains the default. With `clock_weight=0.0`, the dense day-02
  density-weight-2 result exactly recovers the previous matched score:
  `0.281235` vs uniform `0.285668`.
- Clock time does not help this dense held-out setting. Full clock weighting
  scores `0.275288`, and `clock_weight=0.25` scores `0.270649`; both worsen
  `RangePointF1`, entry/exit, and geometry versus the clock-disabled result.
- Re-tested two-day train data (`/tmp/aisdk-2026-02-02_2026-02-03_combined.csv`)
  with the density features and `historical_prior_density_weight=2.0`.
  Dense matched 5% reaches only `0.278122` versus uniform `0.285668`, worse
  than one-day day-02 training (`0.281235`). The extra day changes the train
  workload distribution but does not improve transfer to the held-out eval day.

Conclusion from this branch: the absence of clock-time was a real modeling
blind spot, but exploiting it through KNN distance is not useful here. More
historical-prior distance-feature weighting is low-value unless a new target or
model can improve low-budget point/ship coverage directly.

Historical-prior support filtering:

- Added `historical_prior_min_target`, default `0.0`, to optionally store only
  train points whose retained-frequency target is at least a configured
  threshold. This is a training-support filter, not query-conditioned eval
  inference. It also reduces KNN latency by shrinking the stored prior.
- Dense seed-47 day-02 with density weight `2.0`, `k=32`, and
  `min_target=0.28` improves the matched 5% cell from `0.281235` to
  `0.283803`, still below uniform `0.285668` but well above DP `0.243850`.
  Freeze latency drops from roughly `612 ms` to `430 ms`. Geometry is still
  worse than uniform (`AvgSED=0.5915 km` vs `0.4843 km`), though length
  preservation matches uniform (`0.8881`).
- The 5% component picture improves but still does not clear the bar:
  `RangePointF1=0.098768` vs uniform `0.100327`, `ShipF1=0.857639` vs
  `0.879018`, `ShipCov=0.130091` vs `0.125241`, `EntryExitF1=0.241946` vs
  `0.227432`, `CrossingF1=0.153137` vs `0.106029`, `TemporalCov=0.412581` vs
  `0.437502`, `GapCov=0.350315` vs `0.397476`, `TurnCov=0.121162` vs
  `0.097923`, `ShapeScore=0.214886` vs `0.221969`.
- The audit grid still loses uniform at every measured compression target:
  `1%` `0.092739` vs `0.105707`, `2%` `0.179349` vs `0.181583`,
  `5%` `0.283803` vs `0.285668`, `10%` `0.371593` vs `0.385741`,
  `15%` `0.446415` vs `0.452170`, `20%` `0.509970` vs `0.514366`,
  `30%` `0.597062` vs `0.598160`.
- Positive-support-only filtering (`min_target=0.14`) is worse:
  matched 5% `0.278473`. Reducing smoothing to `k=16` at `min_target=0.28`
  is also worse: matched 5% `0.279983`. The useful threshold is a runtime and
  sharpness improvement, not a final success.
- Combining the useful threshold with even a tiny stratified center penalty is
  still bad: `min_target=0.28`, `k=32`, `center_weight=0.01` drops matched 5%
  to `0.251960`. This confirms the center-regularizer branch should stay dead.

Conclusion: the historical-prior path proves the workload-blind protocol and
retained-frequency labels contain useful signal, but it is not final success.
It beats uniform and DP on the main seed/coverage grid, yet it is too brittle
across held-out seeds and anchor settings. Further work should address the
prior/selector structure or add genuinely robust historical training data; more
pointwise neural losses, scalar blends, or KNN distance knobs are low-value
unless target fit and low-budget coverage improve first.

Global-fill allocation diagnostic:

- Added `mlqds_hybrid_mode="global_fill"` as a query-blind diagnostic selector.
  It preserves the existing total retained count, keeps a temporal base in each
  trajectory, then allocates the remaining learned-score fill slots globally
  across trajectories. This tests whether the blind prior can spend scarce
  residual budget across vessels instead of being forced to keep the same fill
  quota in every trajectory.
- Dense seed-47 day-02 historical prior with density weight `2.0`,
  retained-frequency labels, and `global_fill` is a clear failure:
  matched 5% `RangeUseful=0.229731` versus uniform `0.285668` and DP
  `0.243850`. It is also worse than TemporalRandomFill (`0.252199`), so the
  learned score is actively harmful under global allocation.
- Component failure is broad: at 5%, `RangePointF1=0.086862`,
  `ShipF1=0.695802`, `TemporalCov=0.296149`, `GapCov=0.220697`, and
  `ShapeScore=0.147849`, all much worse than uniform (`0.100327`, `0.879018`,
  `0.437502`, `0.397476`, `0.221969`). Geometry also collapses:
  `AvgSED=1.9455 km` and length preservation `0.8582` versus uniform
  `0.4843 km` and `0.8881`.
- Audit grid result: MLQDS loses uniform at `1%`, `2%`, `5%`, `10%`, `15%`,
  and `20%`, and only wins at `30%` (`0.621020` vs `0.598160`). The selector
  over-concentrates the fill budget and destroys low-budget ship/temporal/gap
  coverage. This invalidates global learned allocation for the current prior.
- The first `global_fill` diagnostic failed badly, so it must not be treated as
  a final selector. The code path has since been restored as an explicit,
  tested diagnostic because it is the cleaner continuity-preserving counterpart
  to raw `global_budget`, but it remains a negative prior until a new target
  proves otherwise.

Historical-prior support capping:

- Added `historical_prior_support_ratio`, default `1.0`, to cap stored
  historical-prior support to the top retained-frequency points per training
  trajectory before the optional `historical_prior_min_target` filter. This is
  still workload-blind at eval: it only changes which train points are stored
  in the prior.
- With dense seed-47 day-02, density weight `2.0`, `min_target=0.28`, and
  `support_ratio=0.30`, the result is effectively identical to the uncapped
  thresholded run: matched 5% `RangeUseful=0.283803` versus uniform
  `0.285668`. Diagnostics show the cap keeps `9224` pre-threshold train points
  (`30.3%`) and the min-target filter stores `5735` points (`18.9%`), so the
  threshold is already the tighter useful support decision.
- Tightening to `support_ratio=0.10` stores only `3147` train points (`10.3%`)
  and makes the model worse: matched 5% `RangeUseful=0.282770` versus uniform
  `0.285668`. It raises `RangePointF1` to `0.103378` but loses too much
  coverage structure: `ShipF1=0.862989` vs uniform `0.879018`, `GapCov=0.345955`
  vs `0.397476`, and geometry is worse (`AvgSED=0.6205 km` vs `0.4843 km`).
- The 10% support-cap audit loses uniform at every compression target:
  `1%` `0.084789` vs `0.105707`, `2%` `0.174980` vs `0.181583`,
  `5%` `0.282770` vs `0.285668`, `10%` `0.365477` vs `0.385741`,
  `15%` `0.437807` vs `0.452170`, `20%` `0.501453` vs `0.514366`,
  `30%` `0.592487` vs `0.598160`.

Conclusion: support capping is not a missing selector knob. The current
historical-prior signal can slightly re-rank range-relevant points, but tighter
support removes temporal/gap/ship coverage faster than it improves point-level
range hits. This branch should not receive more KNN/support tuning unless the
training signal or prior structure changes.

Stratified temporal-fraction audit:

- Ran the dense seed-47 historical-prior `min_target=0.28` setup with
  `mlqds_temporal_fraction=0.35` instead of `0.30`. The matched and audit
  metrics are identical to the 30% run: matched 5% `RangeUseful=0.283803` vs
  uniform `0.285668`, and the audit still loses uniform in every compression
  cell.
- Root cause: `mlqds_hybrid_mode="stratified"` deliberately ignores
  `mlqds_temporal_fraction`; it picks one learned-score point inside each
  retained-budget stratum rather than reserving a temporal base. The README
  already said this, but the training code still allowed
  `temporal_residual_label_mode="temporal"` to shrink budget-loss ratios as if
  a temporal base existed.
- Fixed that mismatch: stratified selection now forces the effective
  temporal-residual training mode to `none`. This does not change the current
  best `range_prior` artifact because it already used
  `temporal_residual_label_mode="none"`, and it does not change the historical
  prior result because that path stores labels directly. It does prevent future
  stratified neural runs from optimizing a residual candidate set that
  inference never constructs.

Prototype-prior diagnostic:

- Tested a temporary workload-blind prototype prior that compressed train-day
  retained-frequency labels into 512 query-free feature-space prototypes before
  scoring eval points. It was meant to smooth the brittle KNN historical prior
  without using eval queries.
- Dense seed-47, day-02 train/day-04 eval, `prototype_count=512`, six k-means
  iterations, no min-target filter: matched 5% `RangeUseful=0.272387` versus
  uniform `0.285668` and raw historical-prior `0.281235`. It improved length
  preservation (`0.9044`) but lost point, ship, temporal, and gap coverage. The
  audit lost uniform at `1%`, `2%`, `5%`, `10%`, `15%`, and `20%`, winning only
  at `30%` (`0.599945` vs `0.598160`).
- Adding the useful KNN threshold (`min_target=0.28`) made the prototype prior
  worse: matched 5% `RangeUseful=0.266545`, below uniform and below temporal
  random fill (`0.271562`). It lost uniform across every audit compression
  target.
- The temporary prototype model code was removed after the diagnostic. Keeping
  a public model type that is worse than raw KNN and uniform would add a stale
  path without moving the acceptance criteria.

Implementation bug fixed while testing this branch:

- Saved-model inference treated every workload-blind checkpoint as if it used
  the neural blind feature dimensions. That excludes `historical_prior`, whose
  saved `point_dim` uses the historical route/clock/density feature set.
  `forward_predict` now resolves query-free features by saved point dimension,
  so historical-prior checkpoints can run without query tensors.

Validation:

- `py_compile` passed for the historical-prior model path.
- Targeted split-cap/query-generation tests passed: `30 passed`.
- Full suite passed after the support-cap, stratified residual-mode, and
  historical-prior saved-inference changes: `228 passed, 1 warning`.
- `git diff --check` passed.

Longer pointwise-BCE training check:

- Ran the current `range_prior` retained-frequency, stratified, train-replicate-4
  setup for 24 epochs with `loss_objective=pointwise_bce` to test whether the
  prior 8-epoch pointwise result was simply under-trained.
- The run did optimize the pointwise objective: loss dropped from `0.5872` to
  about `0.2754`, and prediction standard deviation increased from `0.1000` to
  `0.7001`. The best checkpoint was epoch `11` under the validation
  `uniform_gap` selector, with best loss `0.337296` and validation selection
  score `0.238645` against validation uniform `0.240360`.
- The retained-set result got worse, not better. Matched 5% eval:
  `RangeUseful=0.261446` and `RangePointF1=0.098997` versus uniform
  `0.282023` / `0.111028`, DP `0.226293` / `0.087228`, and the earlier
  8-epoch pointwise run's `RangeUseful=0.268446`.
- The learned fill also loses to temporal random fill: `0.261446` versus
  `0.272689` on `RangeUseful`, and `0.098997` versus `0.105327` on
  `RangePointF1`. Temporal oracle fill remains far above both at
  `RangeUseful=0.476062`, so the budget has useful range signal that the blind
  student is not capturing.
- Component losses versus uniform at 5% are broad: `EntryExitF1=0.194817`
  versus `0.243910`, `CrossingF1=0.090168` versus `0.126345`,
  `TemporalCov=0.339393` versus `0.403586`, and `ShapeScore=0.197192` versus
  `0.217789`. Geometry is also worse: `AvgSED=0.5361 km` and length
  preservation `0.8858` versus uniform `0.4843 km` and `0.8881`.
- Audit grid result: MLQDS loses uniform at every compression target:
  `1%` `0.107171` vs `0.111879`, `2%` `0.168556` vs `0.193744`,
  `5%` `0.261446` vs `0.282023`, `10%` `0.357510` vs `0.374241`,
  `15%` `0.444188` vs `0.446965`, `20%` `0.498975` vs `0.524876`,
  and `30%` `0.599282` vs `0.604541`.

Conclusion: simple pointwise under-training is not the blocker. Longer BCE
training makes the scalar ranking more confident but less useful under the
frozen retained-mask protocol. The next credible direction is a genuinely
stronger workload-blind representation or set-aware retained-set objective, not
more epochs, pointwise losses, or selector tweaks around the same scalar model.

MLP-only blind student diagnostic:

- Added a clean `num_layers=0` path for `WorkloadBlindRangeQDSModel`. For
  workload-blind neural models this skips both the local transformer and the
  sinusoidal window-position encoding, leaving point features -> point encoder
  -> score head. This tests whether overlapping-window position encodings are
  corrupting blind target fit without adding a new model type.
- Focused 8-epoch run:
  `blind_real_true_c30_rangeprior_mlp_retfreq_stratified_trainrep4_labelmean_div002_20260514`.
  Matched 5% eval: `RangeUseful=0.275325`, `RangePointF1=0.108750`, uniform
  `0.282023` / `0.111028`, DP `0.226293` / `0.087228`. It beats DP and temporal
  random fill at 5% (`0.272689`) by a small amount, but still loses uniform.
- The 8-epoch MLP preserves geometry slightly better than the transformer:
  `AvgSED=0.4833 km` and length preservation `0.8978` versus uniform
  `0.4843 km` / `0.8881`, and versus the 24-epoch pointwise transformer
  `0.5361 km` / `0.8858`. The useful-score loss is still structural:
  entry/exit and crossing remain weak (`EntryExitF1=0.199272` versus uniform
  `0.243910`; `CrossingF1=0.101723` versus `0.126345`).
- 8-epoch audit loses uniform at every compression target:
  `1%` `0.093188` vs `0.111879`, `2%` `0.174417` vs `0.193744`,
  `5%` `0.275325` vs `0.282023`, `10%` `0.368712` vs `0.374241`,
  `15%` `0.446546` vs `0.446965`, `20%` `0.499129` vs `0.524876`,
  and `30%` `0.590626` vs `0.604541`.
- Longer 48-epoch MLP:
  `blind_real_true_c30_rangeprior_mlp48_retfreq_stratified_trainrep4_labelmean_div002_20260514`.
  Training loss fell from `0.9018` to a best selected loss of `0.600294`, and
  prediction standard deviation rose to `3.6967` at the selected epoch. The best
  validation score improved late to `0.234501`, still below validation uniform
  `0.240360`.
- Held-out eval got worse after longer MLP training: matched 5%
  `RangeUseful=0.272409`, below uniform `0.282023`, below temporal random fill
  `0.272689`, and below the 8-epoch MLP. It loses uniform at every audit target:
  `1%` `0.100107` vs `0.111879`, `2%` `0.181254` vs `0.193744`,
  `5%` `0.272409` vs `0.282023`, `10%` `0.361743` vs `0.374241`,
  `15%` `0.433502` vs `0.446965`, `20%` `0.504506` vs `0.524876`,
  and `30%` `0.597050` vs `0.604541`.

Conclusion: inconsistent transformer window-position encoding is not the main
blocker. Removing it gives a cleaner and faster diagnostic model, but the
current query-free feature/label setup still cannot learn a retained-set policy
that beats uniform. The failure remains strongest in ship presence,
entry/exit/crossing, and gap/temporal coverage, not just global geometry.

Stratified-budget loss diagnostic:

- Identified a real train/inference mismatch: the successful blind recipes use
  `mlqds_hybrid_mode="stratified"`, whose retained-mask selector keeps
  endpoints and then selects one learned-score point per temporal/index stratum.
  The default `budget_topk` loss optimizes global soft top-k target mass, not
  per-stratum choices.
- Added an explicit `loss_objective="stratified_budget_topk"` diagnostic. It
  uses stratum-local softmax target capture for stratified selection and is
  opt-in only. It is not the default because the first focused run was both
  slower and worse on held-out eval.
- Focused run:
  `blind_real_true_c30_rangeprior_stratloss_retfreq_stratified_trainrep4_labelmean_div002_20260514`.
  Same current-best `range_prior` retained-frequency trainrep4 setup, but with
  the stratum-local loss. Validation improved close to uniform: best selected
  epoch `7/8`, validation `RangeUseful=0.239700` versus validation uniform
  `0.240360`.
- Held-out eval failed: matched 5% `RangeUseful=0.265821` and
  `RangePointF1=0.104397` versus uniform `0.282023` / `0.111028`, DP
  `0.226293` / `0.087228`, and temporal random fill `0.272689` /
  `0.105327`.
- Component diagnosis at 5%: the loss improves `EntryExitF1` relative to the
  prior 24-epoch pointwise run (`0.220775`), but it still loses uniform
  (`0.243910`) and loses core coverage: `TemporalCov=0.356444` versus uniform
  `0.403586`, `GapCov=0.319809` versus `0.354542`, and
  `ShapeScore=0.195094` versus `0.217789`. Geometry also worsens:
  `AvgSED=0.6054 km` versus uniform `0.4843 km`.
- Audit grid loses uniform at every target except `30%`:
  `1%` `0.093764` vs `0.111879`, `2%` `0.165124` vs `0.193744`,
  `5%` `0.265821` vs `0.282023`, `10%` `0.360172` vs `0.374241`,
  `15%` `0.444395` vs `0.446965`, `20%` `0.491098` vs `0.524876`,
  and `30%` `0.610807` vs `0.604541`.
- Runtime regression is material: training took `55.08s`, with epoch loss time
  roughly `3.1s` after warmup, versus about `6s` total for the previous full
  test suite and sub-second epoch loss time in the ordinary MLP run. The Python
  stratum loop would need vectorization before broader sweeps.

Conclusion: the stratified train/inference mismatch is real, but matching the
selector locally does not solve generalization. It mostly overfits validation
and weakens ship/gap/shape behavior on held-out eval. Keep the loss as an
explicit diagnostic only; do not replace default `budget_topk` with it.

Benchmark-report failure diagnostics:

- Added report-row fields that make a blind run's failure mode explicit instead
  of relying on manual artifact notes. Each row now records
  `single_cell_range_status`, workload-blind protocol freeze flags, strict
  `RangeUseful` win booleans versus uniform/DP/temporal-random-fill, audit-grid
  win counts, low-compression audit win counts, and min/mean low-compression
  deltas versus uniform.
- Added component-level MLQDS-vs-uniform deltas and
  `worst_uniform_component_delta_metric`, plus flattened geometry/runtime fields
  (`AvgSED`, `AvgPED`, length preservation, latency, dominant parsed phase, and
  per-epoch training subphase means). This directly exposes whether a failed run
  is losing by entry/exit, temporal/gap coverage, shape, geometry distortion, or
  runtime.
- Added teacher-distillation summary fields to benchmark rows:
  teacher mode, teacher model type, replicate count, positive-label fraction,
  and label mass. Combined with existing target-label mass and best/final loss
  fields, this is enough to diagnose label mass and teacher/student fit without
  opening several child artifacts by hand.
- Regression coverage: `test_benchmark_runner.py` now checks diagnostic status
  classification, audit low-compression counts, protocol-fail handling for a
  leaky blind run that otherwise has good scores, geometry deltas, component
  deltas, teacher fields, and runtime bottleneck extraction.

Validation:

- `PYTHONPATH=QDS .venv/bin/python -m pytest -q QDS/tests/test_benchmark_runner.py`
  -> `22 passed`.
- `PYTHONPATH=QDS .venv/bin/python -m pytest -q QDS/tests`
  -> `233 passed, 1 warning`.
- `git diff --check` -> clean.

Conclusion: no new model success was claimed. This checkpoint improves the
failure-diagnosis acceptance path and should make the next benchmark variation
less ambiguous, especially at `1%`, `2%`, and `5%` compression.

Range-prior clock/density representation diagnostic:

- Added `model_type="range_prior_clock_density"` as a separate workload-blind
  neural model type. It keeps old `range_prior` intact and adds query-free
  clock-time plus current-day spatial density/sparsity features to the full
  route-context feature set. Checkpoint inference is keyed by saved point
  dimension, so old `range_prior` checkpoints still use the old 24-column
  features.
- First attempted run accidentally omitted the current-best data caps and
  started training on full 4M-point days. Stopped it before result artifacts
  were written. This was not comparable.
- Comparable run:
  `blind_real_true_c30_rangeprior_clockdens_retfreq_stratified_trainrep4_labelmean_div002_20260514`.
  Same retained-frequency, stratified, train-replicate-4 recipe, but with the
  clock/density neural feature set.
- Because current workload generation now overshoots the nominal `30%`
  coverage for this footprint, reran the old `range_prior` model under the same
  current generated workloads:
  `blind_real_true_c30_currentgen_rangeprior_retfreq_stratified_trainrep4_labelmean_div002_20260514`.

Current-generator comparison, eval coverage `50.95%` / `24` eval queries:

| Model | Matched RangeUseful | Uniform | DP | 1% delta vs uniform | 2% delta | 5% delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `range_prior` | `0.383255` | `0.394612` | `0.377531` | `-0.025711` | `-0.008585` | `-0.011358` |
| `range_prior_clock_density` | `0.378709` | `0.394612` | `0.377531` | `-0.010086` | `-0.010968` | `-0.015904` |

Component diagnosis at matched 5%:

- Clock/density improves `EntryExitF1` versus uniform (`+0.009159`) and narrows
  the crossing gap relative to plain current-gen range_prior, but it damages the
  components that already matter most: `TemporalCov` delta `-0.048504`,
  `GapCov` delta `-0.079616`, and `ShapeScore` delta `-0.047953` versus
  uniform.
- Geometry is worse: clock/density `AvgSED=0.6937 km`, length preservation
  `0.8760`; current-gen `range_prior` `AvgSED=0.5308 km`, length preservation
  `0.8886`; uniform `0.4843 km`, `0.8881`.
- The feature change does not solve low compression. It improves the `1%`
  uniform delta but worsens `2%` and `5%`, and it still loses uniform at the
  acceptance-critical low targets.

Extra discovery:

- The older `c30` current-best artifact is not reproducible under current
  workload generation without its old workload cache. The old artifact used
  cached workloads with `eval_coverage=30.12%` and `76` eval queries. The same
  nominal settings under current generation stop after the minimum `24` queries
  with `eval_coverage=50.95%`.
- Fixed the warning path so train and eval overshoot now warns, not only
  selection overshoot. This is still only a warning. The real calibration issue
  remains: target coverage is currently a lower-bound stop criterion with
  `n_queries` as a hard minimum, so a large footprint can silently exceed the
  requested coverage by a lot unless warnings are watched or the coverage
  estimator is used.

Validation:

- Focused tests after adding `range_prior_clock_density` and overshoot warnings:
  `68 passed`.
- Full suite: `235 passed, 1 warning`.

Conclusion: clock/density features are not the missing representation. They
shift component behavior but worsen the matched cell and do not produce a
low-compression win. The more urgent systems issue is workload calibration and
cache provenance: future coverage-grid claims need actual coverage printed and
compared, not just nominal `query_coverage`.

Coverage overshoot guard checkpoint:

- Added `range_max_coverage_overshoot` as an opt-in workload generation guard.
  When `query_coverage` is set, candidate range boxes that would push union
  point coverage above `target_coverage + tolerance` are rejected. The option
  accepts fractions or percentages.
- The guard is wired through `QueryConfig`, CLI parsing, `run_ais_experiment`,
  `run_inference`, workload cache keys, the sampled coverage estimator, and
  benchmark profiles. Current benchmark profiles now pass `0.02` absolute
  overshoot tolerance, so nominal coverage grid cells should not silently drift
  to much broader workloads.
- Query generation diagnostics now record
  `range_max_coverage_overshoot`, `coverage_guard_enabled`, and
  `max_allowed_coverage`; rejected boxes are counted under
  `range_acceptance.rejection_reasons.coverage_overshoot`.
- If the requested footprint and minimum query count cannot satisfy the upper
  guard, generation may return fewer than `n_queries` and stop with
  `range_coverage_guard_exhausted`. This is deliberate. A benchmark should fail
  clearly rather than relabel a 51% workload as a 30% workload.

Validation:

- Focused coverage/profile/protocol tests:
  `PYTHONPATH=QDS .venv/bin/python -m pytest -q QDS/tests/test_query_coverage_generation.py QDS/tests/test_torch_runtime_controls.py QDS/tests/test_benchmark_runner.py`
  -> `55 passed`.
- Focused diagnostics/training/protocol tests:
  `PYTHONPATH=QDS .venv/bin/python -m pytest -q QDS/tests/test_range_workload_diagnostics.py QDS/tests/test_training_does_not_collapse.py QDS/tests/test_workload_blind_protocol.py`
  -> `52 passed, 1 warning`.
- Full suite:
  `PYTHONPATH=QDS .venv/bin/python -m pytest -q QDS/tests`
  -> `236 passed, 1 warning`.
- `git diff --check` -> clean.

Practical calibration check on the capped `2026-02-04` eval day
(`max_segments=120`, `max_points_per_segment=256`, eval seed `2424757607`,
`2.2km/5h`, `anchor_day`, `n_queries=24`, `max_queries=512`,
`range_max_coverage_overshoot=0.02`):

| Target coverage | Generated queries | Actual coverage | Stop reason | Overshoot rejections |
| ---: | ---: | ---: | --- | --- |
| `5%` | `24` | `6.77%` | `target_coverage_reached` | `0` |
| `10%` | `24` | `11.98%` | `target_coverage_reached` | `14` |
| `15%` | `29` | `15.19%` | `target_coverage_reached` | `0` |
| `30%` | `76` | `30.12%` | `target_coverage_reached` | `0` |

Conclusion: this does not create a successful blind model. It removes a
protocol ambiguity that was strong enough to invalidate coverage-grid claims.
Next benchmark variations should rerun the coverage/compression grid with the
guard enabled, and treat guard exhaustion as a footprint/query-count calibration
failure rather than a model result.

Guarded coverage/compression grid checkpoint:

- Fixed benchmark-runner profile argument resolution. `benchmark_runner` exposed
  the blind profiles in `PROFILE_CHOICES`, but `_profile_args` still rejected
  every non-default profile and always used `DEFAULT_PROFILE_ARGS`. Blind
  profiles are now runnable through the main benchmark path.
- Fixed benchmark profile child args to actually pass
  `--range_audit_compression_ratios 0.01,0.02,0.05,0.10,0.15,0.20,0.30`.
  Before this, profile metadata advertised the compression grid but the child
  run did not necessarily evaluate it.
- Ran the guarded `range_prior` retained-frequency recipe across the required
  coverage targets using real three-day AIS splits:
  - train: `2026-02-02`
  - validation/checkpoint: `2026-02-03`
  - eval: `2026-02-04`
  - caps: `max_segments=120`, `max_points_per_segment=256`
  - query footprint: `2.2km/5h`, `anchor_day`
  - guard: `range_max_coverage_overshoot=0.02`
  - compression audit: `1%,2%,5%,10%,15%,20%,30%`

Artifacts:

- `artifacts/manual/guarded_c05_rangeprior_retfreq_20260514`
- `artifacts/manual/guarded_c10_rangeprior_retfreq_20260514`
- `artifacts/manual/guarded_c15_rangeprior_retfreq_20260514`
- `artifacts/manual/guarded_c30_rangeprior_retfreq_20260514`
- Component target diagnostic:
  `artifacts/manual/guarded_c30_rangeprior_componentfreq_20260514`

Matched 5% compression results:

| Coverage target | Actual eval coverage | Eval queries | Status | MLQDS | Uniform | DP | Delta vs uniform | Low-compression uniform wins |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `5%` | `6.77%` | `24` | fails uniform | `0.259937` | `0.279080` | `0.230046` | `-0.019143` | `1/3` |
| `10%` | `11.98%` | `24` | beats both at matched cell | `0.282581` | `0.268608` | `0.232056` | `+0.013974` | `2/3` |
| `15%` | `15.19%` | `29` | fails uniform | `0.264545` | `0.276161` | `0.245976` | `-0.011617` | `1/3` |
| `30%` | `30.12%` | `76` | fails uniform | `0.275764` | `0.282023` | `0.226293` | `-0.006260` | `0/3` |

Compression-grid diagnosis:

- The model beats DP in most cells, but that is not enough. It beats uniform in
  only `1/7`, `3/7`, `4/7`, and `1/7` cells for the `5%`, `10%`, `15%`, and
  `30%` coverage targets respectively.
- Low-compression behavior is not acceptable. Across `1%`, `2%`, and `5%`
  budgets, uniform wins most cells: `5%` coverage has only `1/3` wins, `10%`
  has `2/3`, `15%` has `1/3`, and `30%` has `0/3`.
- Failure components vary by coverage, but continuity is the recurring problem.
  `GapCov` deltas are usually negative across the grid and become severe at
  `20%`/`30%` budgets. At matched `30%` coverage, the worst single matched
  component is `EntryExitF1` (`-0.030152`), while `TemporalCov` is `-0.021506`.
- Training label mass is not aligned with the failures: `GapCov` contributes
  only about `6.1%` of component label mass and `ShapeScore` about `4.8-5.0%`;
  point and ship-coverage mass dominate. This explains why the learned masks
  can improve some presence-like behavior while still losing continuity and
  boundary evidence to uniform.
- Runtime is not the blocker on the capped benchmark. Training is the largest
  phase, but only about `3.3-4.0s`; full child runs complete in roughly
  `10-12s` with cache hits.
- Teacher/student fit was not measured in this checkpoint because
  `range_teacher_distillation_mode=none`. This result diagnoses the blind
  retained-frequency target, not the teacher path.

Component-retained-frequency diagnostic:

- Ran the same guarded `30%` setup with
  `range_training_target_mode=component_retained_frequency`.
- It made the matched result worse: MLQDS `RangeUseful=0.262287` versus uniform
  `0.282023`, delta `-0.019736`.
- It also worsened the main weak components:
  `TemporalCov -0.056876`, `GapCov -0.023967`, `ShapeScore -0.012387`, and
  `EntryExitF1 -0.034425` versus uniform.
- Conclusion: simple component-retained-frequency is not the next useful
  direction for this recipe. The target needs a more explicit continuity/boundary
  correction or a selector/scoring change that preserves runs without becoming
  temporal scaffolding.

Validation:

- Profile wiring tests:
  `PYTHONPATH=QDS .venv/bin/python -m pytest -q QDS/tests/test_benchmark_runner.py QDS/tests/test_torch_runtime_controls.py`
  -> `37 passed`.
- Full suite:
  `PYTHONPATH=QDS .venv/bin/python -m pytest -q QDS/tests`
  -> `237 passed, 1 warning`.
- `git diff --check` -> clean.

Conclusion: the workload-blind model still fails the acceptance requirement.
The calibrated grid is now meaningful, and the failure is no longer ambiguous:
the learned model is not reliably better than uniform, especially at low
compression, and a naive component-frequency target worsens continuity.

Continuity-target and positive-only retained-frequency checkpoint:

- Added `range_training_target_mode="continuity_retained_frequency"`. It builds
  retained-frequency targets from the boundary/continuity components only:
  `EntryExitF1`, `CrossingF1`, `TemporalCov`, `GapCov`, `TurnCov`, and
  `ShapeScore`. `range_component_target_blend` blends it with ordinary
  retained-frequency targets.
- Fixed a target-construction flaw: retained-frequency labels no longer count
  zero-usefulness points selected only because a per-trajectory budget has spare
  slots. Temporal anchors must now come from explicit temporal target blending,
  not accidental zero-label filler.

Focused artifacts:

- `artifacts/manual/guarded_c10_rangeprior_retfreq_posonly_20260514`
- `artifacts/manual/guarded_c30_rangeprior_retfreq_posonly_20260514`
- `artifacts/manual/guarded_c10_rangeprior_continuityblend050_20260514`
- `artifacts/manual/guarded_c30_rangeprior_continuityblend050_20260514`

Key comparison:

| Run | Status | MLQDS | Uniform | Delta vs uniform | Audit wins | Low-budget wins | Target mass |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| old retained-frequency `10%` | old matched win | `0.282581` | `0.268608` | `+0.013974` | `3/7` | `2/3` | `2309.429` |
| positive-only retained-frequency `10%` | fails uniform | `0.238046` | `0.268608` | `-0.030562` | `2/7` | `1/3` | `711.429` |
| continuity-blend `10%` | fails uniform | `0.238233` | `0.268608` | `-0.030374` | `4/7` | `1/3` | `570.757` |
| old retained-frequency `30%` | fails uniform | `0.275764` | `0.282023` | `-0.006260` | `1/7` | `0/3` | `2309.428` |
| positive-only retained-frequency `30%` | fails uniform | `0.274989` | `0.282023` | `-0.007035` | `2/7` | `0/3` | `1408.857` |
| continuity-blend `30%` | fails uniform | `0.275889` | `0.282023` | `-0.006134` | `2/7` | `0/3` | `1195.359` |

Diagnosis:

- The previous calibrated `10%` matched win was not clean evidence. It depended
  on zero-utility filler labels, which acted like accidental temporal
  scaffolding. That violates the spirit of the final acceptance requirement.
- Suppressing zero filler makes the retained-frequency target much sparser and
  exposes a real blind-model gap: the student no longer beats uniform at the
  matched cell.
- Continuity-blend is directionally useful for some audit cells (`30%` audit
  wins improve from `1/7` to `2/7`; `10%` from `2/7` positive-only to `4/7`),
  but it collapses the matched `10%` result and still fails every
  low-compression acceptance slice that matters.
- Conclusion: the next viable path is not more temporal filler or component
  reweighting alone. The blind student needs either a stronger query-blind
  representation of historically useful boundary/continuity points or a teacher
  signal that transfers those points without relying on implicit temporal
  selection.

Validation:

- Focused target/profile tests:
  `PYTHONPATH=QDS .venv/bin/python -m pytest -q QDS/tests/test_teacher_distillation.py QDS/tests/test_training_does_not_collapse.py QDS/tests/test_torch_runtime_controls.py QDS/tests/test_benchmark_runner.py`
  -> `90 passed, 1 warning`.
- Full suite:
  `PYTHONPATH=QDS .venv/bin/python -m pytest -q QDS/tests`
  -> `239 passed, 1 warning`.
- `git diff --check` -> clean.

Low-budget target-weighting checkpoint:

- Added `range_target_budget_weight_power`, a training-only retained-frequency
  target knob. `0.0` preserves uniform budget averaging. Positive values weight
  target budgets as `compression_ratio ** -power` before normalization. This is
  explicit and auditable; it does not alter eval compression, checkpoint
  selection protocol, or query access.
- Ran focused guarded slices with `range_target_budget_weight_power=1.0`:
  - `artifacts/manual/guarded_c30_rangeprior_retfreq_lowbudgetw1_20260514`
  - `artifacts/manual/guarded_c10_rangeprior_retfreq_lowbudgetw1_20260514`

Results:

| Coverage | Run | MLQDS | Uniform | Delta | Audit wins | Low-budget wins | Target mass |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `30%` | positive-only retained-frequency | `0.274989` | `0.282023` | `-0.007035` | `2/7` | `0/3` | `1408.857` |
| `30%` | low-budget weight `1.0` | `0.274068` | `0.282023` | `-0.007955` | `2/7` | `0/3` | `498.718` |
| `10%` | positive-only retained-frequency | `0.238046` | `0.268608` | `-0.030562` | `2/7` | `1/3` | `711.429` |
| `10%` | low-budget weight `1.0` | `0.241481` | `0.268608` | `-0.027127` | `2/7` | `1/3` | `273.308` |

Diagnosis:

- Low-budget weighting does not solve the acceptance failure. It barely changes
  `1%`/`2%` at `30%` coverage, still loses `5%`, and worsens matched `30%`.
- At `10%` coverage it improves the matched cell by only `+0.003435` versus
  positive-only retained-frequency, still far below uniform and still only
  `1/3` wins in the `1%`/`2%`/`5%` slice.
- Target mass shrinks sharply, but the learned model still under-preserves
  ship, temporal, and gap coverage. The problem is not just equal weighting of
  high-compression target masks.
- This keeps the low-budget weighting knob as a documented diagnostic, but it
  is not a promising final path.

Validation:

- Focused tests:
  `.venv/bin/pytest -q QDS/tests/test_teacher_distillation.py QDS/tests/test_torch_runtime_controls.py QDS/tests/test_benchmark_runner.py`
  -> `57 passed, 1 warning`.
- Full suite:
  `.venv/bin/pytest -q QDS/tests`
  -> `240 passed, 1 warning`.
- `git diff --check` -> clean.

Selector/loss-alignment checkpoint:

- Tested `mlqds_score_mode=rank_confidence` on the `30%` guarded positive-only
  retained-frequency slice:
  `artifacts/manual/guarded_c30_rangeprior_retfreq_posonly_rankconf015_20260514`.
- Tested `loss_objective=stratified_budget_topk` on the same slice:
  `artifacts/manual/guarded_c30_rangeprior_retfreq_posonly_stratloss_20260514`.

Results:

| Run | MLQDS | Uniform | Delta | Audit wins | Low-budget wins | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| positive-only retained-frequency | `0.274989` | `0.282023` | `-0.007035` | `2/7` | `0/3` | `9.65s` |
| rank-confidence selector | `0.274770` | `0.282023` | `-0.007253` | `2/7` | `0/3` | `8.27s` |
| stratified-budget loss | `0.267584` | `0.282023` | `-0.014439` | `2/7` | `0/3` | `44.49s` |

Diagnosis:

- Rank-confidence scoring is effectively neutral. It improves `10%` and `15%`
  audit cells slightly, but does not move the critical `1%`/`2%`/`5%` cells and
  is not a path to acceptance.
- `stratified_budget_topk` aligns the loss with the stratified selector, but it
  makes the low-budget cells worse (`1%`: `0.094013` vs uniform `0.111879`,
  `2%`: `0.166281` vs `0.193744`, `5%`: `0.267584` vs `0.282023`) and is about
  `4.6x` slower on this focused run. It does improve some high-budget
  continuity/boundary components, which is useful diagnostically but not enough.
- The remaining failure is not a trivial score-mode or loss/selector mismatch.
  The next useful work should diagnose target learnability and representation:
  whether the neural blind student can fit the retained-frequency labels at all,
  and whether the features expose the historical prior signal that KNN can use.

Train-target fit diagnostic checkpoint:

- Added `training_fit_diagnostics` to the training output and benchmark report.
  It runs after restoring the selected checkpoint, scores train points only, and
  compares MLQDS retained-mask target-mass recall against uniform across the
  configured budget grid. It is diagnostic only: no eval queries, no checkpoint
  selection input, no retained-set decision changes.
- Added focused tests for the diagnostic and report columns.
- Reran the `30%` guarded positive-only retained-frequency slice with fit
  diagnostics:
  `artifacts/manual/guarded_c30_rangeprior_retfreq_posonly_fitdiag_20260514`.

Key fit results:

| Compression | MLQDS train-target recall | Uniform train-target recall | Delta |
| ---: | ---: | ---: | ---: |
| `1%` | `0.4653` | `0.4620` | `+0.0033` |
| `2%` | `0.3166` | `0.3052` | `+0.0113` |
| `5%` | `0.2483` | `0.2383` | `+0.0100` |
| `10%` | `0.2369` | `0.2290` | `+0.0080` |
| `15%` | `0.2744` | `0.2413` | `+0.0331` |
| `20%` | `0.2967` | `0.2734` | `+0.0233` |
| `30%` | `0.3697` | `0.3458` | `+0.0239` |

Associated eval result for that rerun:

- matched `RangeUseful`: MLQDS `0.276037`, uniform `0.282023`, DP `0.226293`.
- audit wins vs uniform: `1/7`; low-budget wins: `0/3`.
- `score_target_kendall_tau=-0.059`; matched train-target recall delta only
  `+0.0100`; low-budget mean train-target recall delta only `+0.0082`.

Diagnosis:

- The neural student barely captures more of its own training target than
  uniform, even before held-out eval queries are considered. This is a
  student-fit/representation problem, not only generalization noise.
- Negative target tau plus small target-mass deltas mean the transformer score
  is not ordering retained-frequency targets well. The low-budget eval failure
  is therefore expected.
- Next useful direction: compare against the historical-prior/KNN diagnostic on
  the same train-target fit metric and then test whether `range_prior` needs
  clock/density features, a simpler non-transformer scorer, or a target that is
  more learnable from query-free features.

Historical-prior and feature/architecture checkpoint:

- Reran the fixed positive-only `30%` guarded slice with `historical_prior`:
  `artifacts/manual/guarded_c30_historicalprior_retfreq_posonly_20260514`.
- Reran neural `range_prior_clock_density` with the same target and fit
  diagnostics:
  `artifacts/manual/guarded_c30_rangepriorclockdens_retfreq_posonly_fitdiag_20260514`.
- Reran `range_prior_clock_density` with `num_layers=0` and `dropout=0.0`:
  `artifacts/manual/guarded_c30_rangepriorclockdens_mlp_retfreq_posonly_fitdiag_20260514`.

Summary:

| Run | Matched MLQDS | Uniform | Delta | Audit wins | Low-budget wins | Train-fit tau | Fit delta @5% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `range_prior` | `0.276037` | `0.282023` | `-0.005987` | `1/7` | `0/3` | `-0.059` | `+0.0100` |
| `range_prior_clock_density` | `0.278217` | `0.282023` | `-0.003806` | `2/7` | `1/3` | `-0.050` | `-0.0100` |
| `range_prior_clock_density`, MLP | `0.269048` | `0.282023` | `-0.012975` | `0/7` | `0/3` | `+0.002` | `+0.0055` |
| `historical_prior` KNN | `0.283991` | `0.282023` | `+0.001968` | `5/7` | `2/3` | n/a | n/a |

Diagnosis:

- The current fixed positive-only target is not useless. KNN on historical
  query-free features beats uniform on `5/7` audit cells and `2/3` low-budget
  cells, including a small matched `5%` win. This remains a diagnostic/baseline,
  not final learned-model success.
- Adding clock/density features to the neural student helps a little on eval
  (`2/7` audit wins instead of `1/7`), but it still fails matched `5%` and
  fails train-target fit. The student is not exploiting the signal that the KNN
  prior can use.
- Removing transformer layers is worse. The `num_layers=0` MLP has almost zero
  target tau, loses all audit cells, and should not be expanded.
- Next useful direction is not more shallow architecture toggles. Either add a
  learned historical-prior feature/kernel head that can imitate the KNN signal,
  or use the KNN prior as an explicit training feature/teacher while preserving
  workload-blind inference.

Pointwise-fit checkpoint:

- Ran direct pointwise BCE on `range_prior_clock_density`:
  `artifacts/manual/guarded_c30_rangepriorclockdens_pointwise_retfreq_posonly_fitdiag_20260514`.

Result:

| Run | Matched MLQDS | Uniform | Delta | Audit wins | Low-budget wins | Train-fit tau | Fit delta @5% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `range_prior_clock_density`, budget top-k | `0.278217` | `0.282023` | `-0.003806` | `2/7` | `1/3` | `-0.050` | `-0.0100` |
| `range_prior_clock_density`, pointwise BCE | `0.255489` | `0.282023` | `-0.026534` | `1/7` | `0/3` | `-0.120` | `-0.0168` |

Diagnosis:

- Direct soft-label fitting is worse. It drives down BCE but collapses useful
  ranking for retained masks; the selected checkpoint has negative target tau,
  worse train-target recall than uniform, and much worse eval `RangeUseful`.
- This confirms the problem is not just the differentiable top-k objective.
  The current neural scorer is failing to learn a stable historical-prior
  ordering from these point features.

Historical-prior distillation and prior-assisted student checkpoint:

- Added `range_training_target_mode=historical_prior_retained_frequency`.
  It builds the existing retained-frequency target, scores train points with a
  leave-one-out query-free historical KNN teacher, then converts those teacher
  scores back into retained-frequency labels. This is train-only distillation;
  eval compression still uses frozen query-blind masks.
- Added `model_type=historical_prior_student`, which appends the stored
  train-derived historical KNN prior score to each point's blind features and
  trains a normal blind neural scorer on top. This remains workload-blind at
  eval, but it must beat the standalone `historical_prior` diagnostic to count
  as useful learned behavior.
- Added benchmark-report fields for target-transform mass and historical
  teacher score sharpness.

Focused `30%` guarded-slice results:

| Artifact | Change | Matched MLQDS | Uniform | Audit wins | Low-budget wins | Train-fit tau | Fit delta @5% |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `guarded_c30_histteacher_rangepriorclockdens_20260514` | neural `range_prior_clock_density`, historical-prior teacher target | `0.272448` | `0.282023` | `1/7` | `1/3` | `+0.064` | `+0.0091` |
| `guarded_c30_histteacher_rangepriorclockdens_mlp_20260514` | same target, MLP-only scorer | `0.267235` | `0.282023` | `0/7` | `0/3` | `+0.135` | `+0.0296` |
| `guarded_c30_historicalpriorstudent_retfreq_mlp_20260514` | explicit historical-prior feature, base retained-frequency target, MLP-only scorer | `0.264931` | `0.282023` | `0/7` | `0/3` | `-0.065` | `+0.1287` |

Diagnosis:

- Historical-prior teacher distillation diffuses the target. The transformed
  target has `5728` positives and mass `2079.43`, versus base retained-frequency
  `3873` positives and mass `1408.86`; teacher score `p95` is only `0.201`.
  The neural student captures little useful ranking and loses the matched cell.
- The MLP-only scorer fits the train target better than the transformer on
  target-recall diagnostics, but eval is worse. Better train-target recall is
  not sufficient because the target itself is broad and misaligned.
- Feeding the KNN prior as an explicit feature does not rescue the learned
  scorer. It improves train retained-target recall over uniform, but eval
  `RangeUseful` falls further below uniform and every audit cell loses uniform.
  This suggests the useful part of `historical_prior` is the nonparametric KNN
  retained ranking itself, not a signal the current neural scorer can
  productively recalibrate.

Conclusion: this branch does not meet acceptance and should not be promoted as
final learned success. The next useful step is not another neural wrapper around
the same scalar target. Either the final method must explicitly accept a
train-derived nonparametric historical prior as the model class, or the target
needs a different structural objective that teaches temporal/ship/gap coverage
instead of re-ranking hotspot priors.

Historical-prior train-fit diagnostic:

- Added train-target fit diagnostics to the `historical_prior` early-return
  path and marked them with `model_fits_stored_train_support=true`, because the
  KNN scorer stores train points and therefore has an optimistic train fit.
- Reran the fixed `30%` guarded historical-prior slice:
  `artifacts/manual/guarded_c30_historicalprior_retfreq_fitdiag_20260514`.

Result:

- matched `RangeUseful`: MLQDS `0.283991`, uniform `0.282023`, DP `0.226293`.
- audit wins vs uniform: `5/7`; low-budget wins vs uniform: `2/3`.
- train-fit tau: `+0.999`; matched train-target recall delta vs uniform:
  `+0.5726`; low-budget mean delta: `+0.4424`.
- freeze latency: about `517 ms`; worse than uniform and DP, but still usable
  for focused iteration.

Diagnosis:

- The historical prior can almost perfectly recover its stored train target,
  unlike the neural students. The remaining gap is transfer to held-out day and
  held-out query workload, not fitting the train target.
- The matched eval win is real under the frozen-mask protocol but small, and
  geometry is worse than both baselines (`AvgSED=0.5679 km` vs uniform
  `0.4843 km`, DP `0.5208 km`). This still does not satisfy final acceptance.

Held-out AIS-day validation for historical prior:

- Ran the same `historical_prior` retained-frequency setup on a different
  train/validation/eval day triplet:
  `artifacts/manual/heldoutdays_050607_c30_historicalprior_retfreq_20260514`
  with train `2026-02-05`, validation `2026-02-06`, eval `2026-02-07`.

Result:

- matched `RangeUseful`: MLQDS `0.208324`, uniform `0.211751`, DP `0.185568`.
- audit wins vs uniform: `0/7`; low-budget wins vs uniform: `0/3`.
- audit wins vs DP: `6/7`, so it is still above DP, but the acceptance bar is
  uniform and DP.
- train-fit tau remains very high (`+0.995`) and matched train-target recall
  delta is `+0.5811`, confirming that the model fits stored train support but
  does not generalize enough to the held-out AIS day.

Conclusion: `historical_prior` is not final success evidence. It is the best
diagnostic signal so far, but the day-shift run fails the explicit
generalization requirement. Treat future work as target/objective redesign,
not KNN polishing.

Multi-train historical-prior support check:

- Added first-class comma-separated `--train_csv_path` support in the benchmark
  runner and AIS experiment entrypoint. The runner now records every selected
  train CSV and prewarms each cache; `run_ais_experiment` loads the train CSVs
  separately, applies the train split cap per source, concatenates train
  trajectories, and keeps validation/eval CSVs distinct. This replaces the
  earlier `/tmp/*_combined.csv` workaround with explicit provenance.
- Focused benchmark:
  `artifacts/manual/multitrain0205_c30_historicalprior_retfreq_stratified_20260514`
  with train days `2026-02-02` through `2026-02-05`, validation day
  `2026-02-06`, eval day `2026-02-07`, `30%` coverage, `historical_prior`,
  retained-frequency labels, four train workload replicates, and stratified
  frozen-mask compression.

Result:

- matched `RangeUseful`: MLQDS `0.205750`, uniform `0.211751`, DP `0.185568`.
- audit wins vs uniform: `0/7`; low-budget wins vs uniform: `0/3`.
- audit wins vs DP: `6/7`.
- low-budget RangeUseful cells:
  - `1%`: MLQDS `0.071679`, uniform `0.074838`, DP `0.071081`.
  - `2%`: MLQDS `0.117750`, uniform `0.133392`, DP `0.119772`.
  - `5%`: MLQDS `0.205750`, uniform `0.211751`, DP `0.185568`.
- train-fit remains optimistic: tau `+0.993`, matched train-target recall
  delta `+0.5530`, low-budget mean delta `+0.4172`, with
  `model_fits_stored_train_support=true`.
- Runtime bottleneck moved to range training-label prep: `67.19s` of `89.75s`
  total. KNN freeze latency is about `940 ms`.
- Component failure at `5%` is still broad:
  `RangePointF1` `0.085026` vs uniform `0.089271`, `ShipF1` `0.616555` vs
  `0.655058`, `TemporalCov` `0.232868` vs `0.260320`, `GapCov` `0.215785` vs
  `0.247608`, and `ShapeScore` `0.140018` vs `0.159903`. MLQDS only beats
  uniform on `ShipCov`, `EntryExitF1`, `CrossingF1`, `TurnCov`, and length
  preservation at the matched cell. Geometry is worse: `AvgSED=0.7126 km` vs
  uniform `0.5391 km`, DP `0.3660 km`.

Conclusion:

- More historical train-day support does not repair held-out day `2026-02-07`.
  The stored-support prior still fits train labels very well and still fails
  uniform on all held-out compression cells. This weakens the hypothesis that
  the remaining issue is just insufficient train-day diversity.
- The next credible change should be a genuinely different structural target
  for ship/query-hit coverage or a different model family. Another KNN support,
  density, or selector variant is now low-value unless it directly addresses
  the observed ship/point/temporal/gap deficits.

Saved-checkpoint confidence gate diagnostic:

- Saved the multi-train historical-prior checkpoint to
  `artifacts/manual/multitrain0205_c30_historicalprior_retfreq_stratified_saved_20260514/model.pt`
  and reconstructed `5%` retained masks for eval day `2026-02-07`.
- Eval-oracle per-trajectory swaps show latent useful local choices: starting
  from uniform, swapping only the trajectories where MLQDS is individually
  better reaches `RangeUseful=0.237674`; a greedy eval-oracle gate reaches
  `0.238500` using `28` trajectories. This is a diagnostic upper bound only,
  because it uses eval queries to choose swaps.
- Query-free confidence features selected on validation day `2026-02-06` did
  not transfer. Best validation-selected gate was `length <= 148`, which gave
  eval `RangeUseful=0.211774` versus uniform `0.211751`, effectively noise.
  Eval-oracle single-threshold gates can reach about `0.2180`, so the missing
  piece is a transferable confidence signal, not proof that a gate is valid.

Conclusion: do not add a trajectory confidence gate as a final path yet. It
would currently be either an eval-oracle leak or a validation-selected trick
that does not generalize.

Query-normalized spine target check:

- Added `range_query_spine_mass_mode={hit_group,query}`. `hit_group` preserves
  the old behavior, where each `(query, trajectory-hit)` group gets unit mass
  before averaging over queries. `query` gives each train query unit mass split
  across its hit trajectories. This fixes the previous target-mass bias where
  busy queries contributed more total label mass than sparse queries.
- Focused held-out-day benchmarks used train `2026-02-05`, validation
  `2026-02-06`, eval `2026-02-07`, `30%` coverage, `5%` matched compression,
  `0.30` query-spine fraction, query-normalized mass, and frozen audit masks
  over `1%,2%,5%,10%,15%,20%,30%`.

Results:

| Run | Model/loss | Matched RangeUseful | Uniform | DP | Audit wins vs uniform | Low-budget wins vs uniform | Train-target fit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `heldout050607_c30_queryspine_querymass_f030_t30_div002_20260514` | neural `budget_topk` | `0.188843` | `0.243518` | `0.221775` | `0/7` | `0/3` | tau `-0.036`, matched recall delta `-0.0111` |
| `heldout050607_c30_queryspine_querymass_f030_t30_pointwise_20260514` | neural `pointwise_bce` | `0.202843` | `0.243518` | `0.221775` | `0/7` | `0/3` | tau `-0.077`, matched recall delta `-0.0293` |
| `heldout050607_c30_historicalprior_queryspine_querymass_f030_t30_20260514` | `historical_prior` KNN | `0.213622` | `0.243518` | `0.221775` | `1/7` | `1/3` | tau `+0.998`, matched recall delta `+0.6198` |

Component diagnosis at matched `5%`:

- Neural `budget_topk` is worse than uniform on the main components:
  `ShipF1=0.544039` vs `0.700292`, `TemporalCov=0.212074` vs `0.321659`,
  `GapCov=0.147948` vs `0.285507`, `ShapeScore=0.096952` vs `0.171532`.
  Geometry is also nonsensical: `AvgSED=2.7899 km` vs uniform `0.5391 km`.
- `pointwise_bce` improves the aggregate to `0.202843` but still loses every
  audit cell versus uniform and still has bad geometry (`AvgSED=2.4145 km`).
- `historical_prior` proves the target can be fit on stored train support but
  still loses eval except the `1%` cell (`0.116962` vs uniform `0.110466`).
  This separates student-fit failure from target-transfer failure: even a
  near-perfect train-target prior does not make query-spine labels generalize.

Conclusion:

- Query-normalized spine labels are not the missing structural objective. They
  overemphasize train-query support anchors in a way that damages ship/temporal,
  gap, shape, and geometry metrics on held-out day `2026-02-07`.
- Further query-spine mass/loss/selector variants are low-value unless they add
  a genuinely new signal for transferable trajectory coverage.

Multi-train neural range-prior check:

- Ran the current retained-frequency `range_prior` recipe with first-class
  multi-train CSV support instead of the previous combined-file workaround:
  train days `2026-02-02` through `2026-02-05`, validation `2026-02-06`, eval
  `2026-02-07`, `30%` coverage, four train workload replicates, label-mean
  retained-frequency targets, stratified blind selector, frozen audit ratios
  `1%,2%,5%,10%,15%,20%,30%`.
- Focused artifact:
  `artifacts/manual/multitrain0205_c30_rangeprior_retfreq_stratified_trainrep4_labelmean_20260514`.
- Ran the same setup with the transformer removed (`num_layers=0`,
  `dropout=0.0`) to test a simpler model family:
  `artifacts/manual/multitrain0205_c30_rangeprior_mlp_retfreq_stratified_trainrep4_labelmean_20260514`.

Results:

| Run | Matched RangeUseful | Uniform | DP | TemporalRandomFill | Audit wins vs uniform | Low-budget wins vs uniform | Train-target fit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `range_prior` transformer | `0.233120` | `0.243518` | `0.221775` | `0.243083` | `0/7` | `0/3` | tau `+0.092`, matched recall delta `+0.0182` |
| `range_prior` MLP-only | `0.226894` | `0.243518` | `0.221775` | `0.243083` | `0/7` | `0/3` | tau `+0.169`, matched recall delta `-0.0140` |

Transformer audit grid:

| Compression | MLQDS | Uniform | DP | TemporalRandomFill |
| ---: | ---: | ---: | ---: | ---: |
| `1%` | `0.104036` | `0.110466` | `0.111041` | `0.107278` |
| `2%` | `0.150855` | `0.164363` | `0.154629` | `0.158326` |
| `5%` | `0.233120` | `0.243518` | `0.221775` | `0.243083` |
| `10%` | `0.322360` | `0.336102` | `0.311079` | `0.330375` |
| `15%` | `0.399348` | `0.406774` | `0.380443` | `0.401149` |
| `20%` | `0.462032` | `0.464926` | `0.446399` | `0.460452` |
| `30%` | `0.562112` | `0.576317` | `0.550964` | `0.570977` |

Component diagnosis at matched `5%` for the transformer:

- MLQDS is close to temporal random fill but still below uniform. It beats DP
  and is competitive on geometry (`AvgSED=0.6519 km`, length preservation
  `0.9246`), but the acceptance bar is uniform.
- The largest deficits versus uniform are still ship and continuity:
  `ShipF1=0.657634` vs `0.700292`, `TemporalCov=0.308819` vs `0.321659`,
  `GapCov=0.248502` vs `0.285507`, and `RangePointF1=0.094546` vs
  `0.097447`. MLQDS only beats uniform on crossing, turn, and shape by small
  amounts.
- MLP-only is not a rescue. It lowers memory and preserves length slightly
  better (`0.9309`), but it loses more target recall and drops matched
  `RangeUseful` to `0.226894`.

Conclusion:

- Multi-day train support helps enough to beat DP on the held-out day, but it
  does not beat uniform or temporal random fill. The learned signal is still
  too weak at `1%`, `2%`, and `5%`.
- The failure is now sharply identified: for broad `30%` coverage, uniform
  temporal spacing is a very strong blind prior, and the learned model's local
  choices mostly trade away ship/gap coverage for small crossing/turn/shape
  gains. A different model family without the transformer does not fix that.

### Multi-train neural range-prior coverage sweep

Completed the same `range_prior` transformer recipe across the required
coverage targets using train days `2026-02-02` through `2026-02-05`,
validation `2026-02-06`, and eval `2026-02-07`.

- Model/target: `range_prior`, retained-frequency labels, four train workload
  replicates, `label_mean` aggregation, stratified selector, diversity `0.02`.
- Eval remains workload-blind: retained masks are frozen before held-out eval
  query scoring for every audited compression ratio.
- The `5%` coverage run used `--n_queries 8`; the profile default
  `n_queries=160` is a minimum and would over-cover low targets.

Matched `5%` compression summary:

| Target coverage | Actual train/eval coverage | MLQDS | Uniform | DP | TemporalRandomFill | Wins vs uniform | Low-budget wins vs uniform | Train-target fit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `5%` | `5.23% / 5.89%` | `0.265777` | `0.300945` | `0.221774` | `0.298508` | `1/7` | `0/3` | tau `+0.179`, matched delta `+0.0299` |
| `10%` | `12.00% / 12.00%` | `0.372872` | `0.358494` | `0.374403` | `0.378922` | `5/7` | `1/3` | tau `-0.160`, matched delta `+0.0214` |
| `15%` | `17.00% / 17.00%` | `0.348129` | `0.333190` | `0.325748` | `0.340592` | `5/7` | `1/3` | tau `+0.089`, matched delta `+0.0367` |
| `30%` | `30.01% / 31.99%` | `0.233120` | `0.243518` | `0.221775` | `0.243083` | `0/7` | `0/3` | tau `+0.092`, matched delta `+0.0182` |

Coverage sweep audit grids:

| Coverage | Compression | MLQDS | Uniform | DP | TemporalRandomFill |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `5%` | `1%` | `0.134683` | `0.179347` | `0.159288` | `0.164384` |
| `5%` | `2%` | `0.167974` | `0.223040` | `0.181939` | `0.200833` |
| `5%` | `5%` | `0.265777` | `0.300945` | `0.221774` | `0.298508` |
| `5%` | `10%` | `0.386361` | `0.426356` | `0.328137` | `0.388527` |
| `5%` | `15%` | `0.434481` | `0.493875` | `0.384553` | `0.460166` |
| `5%` | `20%` | `0.510297` | `0.496718` | `0.442629` | `0.524768` |
| `5%` | `30%` | `0.620414` | `0.640910` | `0.534452` | `0.618184` |
| `10%` | `1%` | `0.233339` | `0.247060` | `0.260353` | `0.242528` |
| `10%` | `2%` | `0.275339` | `0.305466` | `0.306196` | `0.302385` |
| `10%` | `5%` | `0.372872` | `0.358494` | `0.374403` | `0.378922` |
| `10%` | `10%` | `0.438766` | `0.432626` | `0.470510` | `0.437366` |
| `10%` | `15%` | `0.507354` | `0.494465` | `0.536859` | `0.502081` |
| `10%` | `20%` | `0.574731` | `0.550831` | `0.594147` | `0.544405` |
| `10%` | `30%` | `0.652769` | `0.632518` | `0.685924` | `0.620127` |
| `15%` | `1%` | `0.184539` | `0.204261` | `0.216322` | `0.201591` |
| `15%` | `2%` | `0.235295` | `0.264956` | `0.261675` | `0.264543` |
| `15%` | `5%` | `0.348129` | `0.333190` | `0.325748` | `0.340592` |
| `15%` | `10%` | `0.420068` | `0.415733` | `0.426856` | `0.414994` |
| `15%` | `15%` | `0.485140` | `0.473888` | `0.497788` | `0.480698` |
| `15%` | `20%` | `0.536559` | `0.521729` | `0.558034` | `0.523719` |
| `15%` | `30%` | `0.633092` | `0.617385` | `0.655003` | `0.618107` |
| `30%` | `1%` | `0.104036` | `0.110466` | `0.111041` | `0.107278` |
| `30%` | `2%` | `0.150855` | `0.164363` | `0.154629` | `0.158326` |
| `30%` | `5%` | `0.233120` | `0.243518` | `0.221775` | `0.243083` |
| `30%` | `10%` | `0.322360` | `0.336102` | `0.311079` | `0.330375` |
| `30%` | `15%` | `0.399348` | `0.406774` | `0.380443` | `0.401149` |
| `30%` | `20%` | `0.462032` | `0.464926` | `0.446399` | `0.460452` |
| `30%` | `30%` | `0.562112` | `0.576317` | `0.550964` | `0.570977` |

Component diagnosis:

- The calibrated `5%` coverage cell fails cleanly versus uniform. At matched
  compression, MLQDS trails uniform on `RangePointF1` (`0.089324` vs
  `0.100008`), entry/exit (`0.266498` vs `0.349356`), temporal coverage
  (`0.303114` vs `0.431409`), gap coverage (`0.279145` vs `0.384105`), and
  shape (`0.146550` vs `0.256194`). It also loses TemporalRandomFill
  (`0.265777` vs `0.298508`), so the learned within-bin choices are actively
  worse than random local fill for this workload.
- The `10%` and `15%` cells are partial wins over uniform at matched
  compression, but not robust wins. Both lose `1%` and `2%` compression to
  uniform, and they mostly lose DP at low budgets. The `10%` cell loses DP at
  every audited ratio; the `15%` cell beats DP only at matched `5%`.
- The `30%` cell remains a broad-coverage failure. It loses uniform and
  TemporalRandomFill at matched `5%`, and loses uniform in every audit cell.

Conclusion:

- This is not final workload-blind success. The model has learned some useful
  local behavior, but the win is coverage-specific and not reliable at the
  important `1%`, `2%`, and `5%` compression targets.
- The current path does not satisfy the acceptance bar across coverage targets.
  More checkpoint/selector tricks would be misleading here. The next credible
  step must improve the learned score itself: either a stronger model that fits
  retained-frequency targets on train support, or a new target that directly
  rewards low-budget ship/temporal/gap preservation without collapsing into a
  uniform scaffold.

### Query-floor robustness check

The `10%` and `15%` partial wins above used the benchmark profile's
`n_queries=160` floor. That creates many duplicate/near-duplicate range queries
after the coverage target has already been reached. To test whether the best
partial result was robust to a different held-out workload-generator setting,
reran the `10%` cell with `--n_queries 8`, letting generation stop as soon as
target coverage was reached.

Artifact:
`artifacts/manual/multitrain0205_c10_rangeprior_retfreq_stratified_trainrep4_labelmean_n8_20260514`.

Result:

| Setting | Eval coverage | Eval queries | MLQDS | Uniform | DP | TemporalRandomFill | Wins vs uniform | Low-budget wins vs uniform |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `10%`, `n_queries=160` | `12.00%` | `160` | `0.372872` | `0.358494` | `0.374403` | `0.378922` | `5/7` | `1/3` |
| `10%`, `n_queries=8` | `10.03%` | `19` | `0.248207` | `0.267976` | `0.203119` | `0.274145` | `0/7` | `0/3` |

Calibrated `10%` audit grid:

| Compression | MLQDS | Uniform | DP | TemporalRandomFill |
| ---: | ---: | ---: | ---: | ---: |
| `1%` | `0.103685` | `0.140792` | `0.118559` | `0.133553` |
| `2%` | `0.142964` | `0.197449` | `0.152526` | `0.174273` |
| `5%` | `0.248207` | `0.267976` | `0.203119` | `0.274145` |
| `10%` | `0.352542` | `0.388761` | `0.329504` | `0.380282` |
| `15%` | `0.441181` | `0.445505` | `0.411931` | `0.435455` |
| `20%` | `0.484591` | `0.487122` | `0.483975` | `0.489488` |
| `30%` | `0.609280` | `0.615274` | `0.580826` | `0.603991` |

Component diagnosis at matched `5%`:

- MLQDS beats uniform on `ShipCov`, `ShipF1`, `TurnCov`, and shape-adjacent
  local choices, but it loses the components that dominate the aggregate:
  `RangePointF1=0.090172` vs uniform `0.094305`, `EntryExitF1=0.242903` vs
  `0.282151`, `TemporalCov=0.283282` vs `0.368852`, `GapCov=0.264492` vs
  `0.356454`, and `ShapeScore=0.154805` vs `0.224318`.
- Learned fill also loses to TemporalRandomFill (`0.248207` vs `0.274145`).
  That is a direct signal that the learned within-stratum choices are worse
  than random local choices for this calibrated workload.
- Train-target fit remains weak: tau `+0.120`, matched target-recall delta
  `+0.0209`, low-budget delta `+0.0223`.

Conclusion:

- The earlier `10%` partial win is not robust. It depends on the
  high-query-floor workload setting, not a stable workload-blind retained-set
  policy.
- This strengthens the failure diagnosis: the current neural `range_prior`
  recipe does not generalize across held-out workload-generator settings even
  when AIS train/eval days are held fixed.

### Benchmark profile query-floor cleanup

Implementation cleanup:

- Changed workload-blind benchmark profile defaults from `n_queries=160` to
  `n_queries=8`. `n_queries` is a minimum in coverage-targeted generation, not
  a pure hint, so the old default could force extra duplicate/near-duplicate
  queries after target coverage had already been reached.
- Kept the generator semantics unchanged. Existing tests deliberately require
  `n_queries` to remain a minimum.
- Added a regression test that all workload-blind profiles use the small query
  floor and still report `workload_blind=true`.
- Updated `QDS/experiments/README.md` to stop saying the workload-blind
  protocol is unfinished, to list the blind profiles, and to warn that
  high-query floors are a held-out workload-generator setting rather than a
  neutral coverage sweep default.

Validation:

- Targeted tests:
  `../.venv/bin/python -m pytest -q tests/test_benchmark_runner.py tests/test_torch_runtime_controls.py tests/test_query_coverage_generation.py`
  -> `59 passed`.
- Full suite:
  `../.venv/bin/python -m pytest -q tests`
  -> `247 passed, 1 warning`.

Conclusion: this is a protocol hygiene fix, not a model improvement. It should
prevent future benchmark runs from accidentally reproducing the invalid
high-query-floor partial win.

### Query-floor diagnostics in artifacts

Implementation cleanup:

- Added coverage-target reach diagnostics to range workload generation:
  `target_reached_query_count`, `coverage_at_target_reached`, and
  `extra_queries_after_target_reached`.
- Added benchmark row/report fields for train/eval/selection query generation:
  final query count, final coverage, extra queries after the coverage target was
  already reached, floor-dominated flags, stop reason, and near-duplicate query
  rate.
- Fixed `historical_prior_student` classification in benchmark reports so it is
  treated as a workload-blind candidate.

Validation:

- `../.venv/bin/python -m py_compile queries/query_generator.py experiments/benchmark_report.py tests/test_benchmark_runner.py tests/test_query_coverage_generation.py`
  passed.
- Targeted tests:
  `../.venv/bin/python -m pytest -q tests/test_query_coverage_generation.py tests/test_benchmark_runner.py`
  -> `45 passed`.
- Full suite:
  `../.venv/bin/python -m pytest -q tests`
  -> `247 passed, 1 warning`.
- Hygiene:
  `git diff --check` passed.

Conclusion: this does not improve the model. It makes workload-generator
failure modes first-class evidence in benchmark artifacts, which is necessary
after the query-floor robustness failure above.

### Ship-balanced usefulness labels

Implementation:

- Added `range_label_mode=usefulness_ship_balanced`.
- It keeps the existing usefulness components but normalizes point, entry/exit,
  and crossing support labels by query-hit ship before aggregation. This is
  train-only supervision; eval compression is still workload-blind.
- Fixed range label/diagnostic cache handling for all component-label modes so
  stale cached summaries with unavailable component mass are recomputed instead
  of silently dropping component-label evidence.

Validation:

- Targeted label/cache tests:
  `../.venv/bin/python -m pytest -q tests/test_f1_importance_labels.py tests/test_range_workload_diagnostics.py tests/test_torch_runtime_controls.py`
  -> `40 passed`.
- Cache regression after stale-component fix:
  `../.venv/bin/python -m pytest -q tests/test_range_workload_diagnostics.py`
  -> `15 passed`.
- Full suite:
  `../.venv/bin/python -m pytest -q tests`
  -> `249 passed, 1 warning`.

Focused benchmark:
`artifacts/manual/multitrain0205_c10_rangeprior_shipbalanced_stratified_trainrep4_labelmean_n8_20260514`.

Configuration:

- Train: AIS `2026-02-02` through `2026-02-05`; validation:
  `2026-02-06`; eval: `2026-02-07`.
- `n_queries=8`, `query_coverage=10%`, `range_prior`,
  `range_training_target_mode=retained_frequency`,
  `range_label_mode=usefulness_ship_balanced`,
  `range_train_workload_replicates=4`,
  `range_replicate_target_aggregation=label_mean`,
  `mlqds_hybrid_mode=stratified`.
- Protocol flags stayed valid:
  `workload_blind_protocol_enabled=True`,
  `primary_masks_frozen_before_eval_query_scoring=True`, and
  `audit_masks_frozen_before_eval_query_scoring=True`.
- Eval workload was not floor-dominated:
  `19` eval queries, `10.03%` final coverage, stop reason
  `target_coverage_reached`, near-duplicate rate `0.0`.

Result:

| Compression | MLQDS | Uniform | DP | TemporalRandomFill |
| ---: | ---: | ---: | ---: | ---: |
| `1%` | `0.103685` | `0.140792` | `0.118559` | `0.133553` |
| `2%` | `0.152188` | `0.197449` | `0.152526` | `0.174273` |
| `5%` | `0.232686` | `0.267976` | `0.203119` | `0.274145` |
| `10%` | `0.324784` | `0.388761` | `0.329504` | `0.380282` |
| `15%` | `0.428651` | `0.445505` | `0.411931` | `0.435455` |
| `20%` | `0.504799` | `0.487122` | `0.483975` | `0.489488` |
| `30%` | `0.606174` | `0.615274` | `0.580826` | `0.603991` |

Audit: wins vs uniform `1/7`, wins vs both uniform and DP `1/7`,
low-budget wins vs uniform `0/3`, low-budget wins vs both `0/3`.

Matched `5%` diagnosis:

- `RangeUseful`: MLQDS `0.232686`, uniform `0.267976`, DP `0.203119`,
  TemporalRandomFill `0.274145`.
- Component losses vs uniform remain concentrated in the components the
  redesign is supposed to preserve: `RangePointF1=0.081317` vs `0.094305`,
  `EntryExitF1=0.217655` vs `0.282151`,
  `TemporalCov=0.265656` vs `0.368852`,
  `GapCov=0.263185` vs `0.356454`, and
  `ShapeScore=0.156464` vs `0.224318`.
- Geometry is also worse: MLQDS `AvgSED=0.7087 km` vs uniform `0.5391 km`
  and DP `0.3660 km`; length preservation is `0.9257` vs uniform `0.9276`
  and DP `0.9607`.
- Runtime was not the blocker: benchmark elapsed `18.36s`; MLQDS latency
  `100.21ms`.

Training signal diagnosis:

- Train-fit Kendall tau stayed negative: `-0.0488`.
- Target-recall lift was tiny: matched `+0.0237`; low-budget mean `+0.0225`.
- Target positive label fraction stayed sparse at `0.1035` with positive mass
  `2703.1426`.
- Pre-clamp component contribution fractions were:
  `RangePointF1=0.2640`, `ShipF1=0.1406`, `ShipCov=0.1560`,
  `EntryExitF1=0.0938`, `CrossingF1=0.1054`,
  `TemporalCov=0.0679`, `GapCov=0.0611`, `TurnCov=0.0464`,
  `ShapeScore=0.0649`.

Conclusion:

- This failed. Ship-balanced support labels make the label semantics less
  dominated by dense-hit ships, but the student still does not learn a useful
  low-budget ranking.
- Scaling this label mode across the full coverage grid would be wasted compute.
  The failure now points at model/objective structure, not another scalar label
  mass reshuffle. The next credible branch is a stronger blind model or a
  teacher/student objective that directly improves train-support fit before
  spending more benchmark time.

### Historical-prior ship-balanced diagnostics

Focused benchmark:
`artifacts/manual/guarded_c30_historicalprior_shipbalanced_retfreq_20260514`.

Configuration:

- Train: AIS `2026-02-02`; validation: `2026-02-03`; eval:
  `2026-02-04`.
- `historical_prior`, `query_coverage=30%`, `n_queries=24`,
  `max_queries=512`, `range_label_mode=usefulness_ship_balanced`,
  `range_train_workload_replicates=4`,
  `range_replicate_target_aggregation=label_mean`,
  `mlqds_hybrid_mode=stratified`, `mlqds_temporal_fraction=0.30`.
- Protocol flags stayed valid:
  `workload_blind_protocol_enabled=True`,
  `primary_masks_frozen_before_eval_query_scoring=True`, and
  `audit_masks_frozen_before_eval_query_scoring=True`.
- Eval workload was not floor-dominated: `76` eval queries, `30.12%`
  final coverage.

Result:

| Compression | MLQDS | Uniform | DP |
| ---: | ---: | ---: | ---: |
| `1%` | `0.119286` | `0.111879` | `0.111075` |
| `2%` | `0.193291` | `0.193744` | `0.155647` |
| `5%` | `0.284063` | `0.282023` | `0.226293` |
| `10%` | `0.390158` | `0.374241` | n/a |
| `15%` | `0.452764` | `0.446965` | n/a |
| `20%` | `0.516238` | `0.524876` | n/a |
| `30%` | `0.608561` | `0.604541` | n/a |

Audit: wins vs uniform `5/7`, wins vs both uniform and DP `5/7`,
low-budget wins vs uniform `2/3`, low-budget wins vs both `2/3`.

Matched `5%` diagnosis:

- `RangeUseful`: MLQDS `0.284063`, uniform `0.282023`, DP `0.226293`,
  TemporalRandomFill `0.272689`.
- Useful gains are small and mostly come from ship/turn coverage:
  `ShipF1=0.844391` vs uniform `0.841076`,
  `ShipCov=0.142834` vs `0.139868`, and
  `TurnCov=0.208437` vs `0.110202`.
- The weak components remain weak: `EntryExitF1=0.220509` vs uniform
  `0.243910`, `CrossingF1=0.124932` vs `0.126345`,
  `TemporalCov=0.382536` vs `0.403586`, and
  `GapCov=0.340193` vs `0.354542`.
- Geometry remains worse than uniform: MLQDS `AvgSED=0.5836 km` vs
  uniform `0.4843 km`; length preservation is better than uniform
  (`0.9029` vs `0.8881`) but worse than DP (`0.9306`).

Training signal:

- Component contribution fractions: `RangePointF1=0.2801`,
  `ShipF1=0.1109`, `ShipCov=0.1655`, `EntryExitF1=0.0976`,
  `CrossingF1=0.1125`, `TemporalCov=0.0704`, `GapCov=0.0633`,
  `TurnCov=0.0485`, `ShapeScore=0.0512`.

Variants:

- `artifacts/manual/guarded_c30_historicalprior_shipbalanced_mint028_retfreq_20260514`
  with `historical_prior_min_target=0.28` failed: matched `RangeUseful`
  `0.267124` vs uniform `0.282023`. It reduced latency but removed too much
  useful point/ship support.
- `artifacts/manual/guarded_c30_historicalprior_shipbalanced_traincap240_retfreq_20260514`
  with `train_max_segments=240` failed: matched `RangeUseful` `0.276583`
  vs uniform `0.282023`, with higher latency (`620.08ms`). More same-day
  support worsened transfer under this setup.

Conclusion:

- Historical prior remains the strongest diagnostic path because it can fit
  train support and freeze masks before eval queries. It still is not final
  acceptance: it loses one low-budget cell, loses the `20%` cell, has brittle
  held-out seed/settings/day evidence from earlier runs, and carries worse
  geometry than uniform.
- Ship-balanced labels are a small target-semantics improvement for this KNN
  path. They are not a robust fix.

### Stratified diversity reporting cleanup

Implementation cleanup:

- Added `mlqds_effective_diversity_bonus` to benchmark rows and profile
  settings.
- It reports `0.0` when `mlqds_hybrid_mode=stratified`, because the stratified
  selector does not consume `mlqds_diversity_bonus`.
- Updated CLI help to say `mlqds_diversity_bonus` applies to fill/swap and is
  ignored by stratified mode.

Validation:

- `../.venv/bin/python -m py_compile experiments/benchmark_report.py experiments/benchmark_profiles.py experiments/experiment_cli.py tests/test_benchmark_runner.py`
  passed.
- Targeted tests:
  `../.venv/bin/python -m pytest -q tests/test_benchmark_runner.py tests/test_torch_runtime_controls.py`
  -> `41 passed`.
- Full suite:
  `../.venv/bin/python -m pytest -q tests`
  -> `250 passed, 1 warning`.
- Hygiene:
  `git diff --check` passed.

Conclusion: this changes reporting only. It fixes misleading `div*`
stratified artifact metadata; it does not change retained masks or scores.

### Benchmark profile max-query cleanup

Implementation cleanup:

- Split `BenchmarkProfile.max_queries` from `BenchmarkProfile.query_chunk_size`.
  The old profile code used `query_chunk_size` as the emitted `--max_queries`
  value. Current defaults were both `2048`, so behavior did not change, but the
  coupling was wrong: workload generation caps and inference batching are
  unrelated settings.
- Added configured workload fields to benchmark rows:
  `n_queries`, `max_queries`, `query_target_coverage`,
  `range_spatial_km`, and `range_time_hours`. Realized split coverage was
  already reported, but the configured target was not first-class in the row.

Validation:

- `../.venv/bin/python -m py_compile experiments/benchmark_profiles.py experiments/benchmark_inputs.py experiments/benchmark_runner.py tests/test_benchmark_runner.py tests/test_torch_runtime_controls.py`
  passed.
- `../.venv/bin/python -m py_compile experiments/benchmark_report.py experiments/benchmark_profiles.py tests/test_benchmark_runner.py`
  passed.
- Targeted tests:
  `../.venv/bin/python -m pytest -q tests/test_benchmark_runner.py tests/test_torch_runtime_controls.py`
  -> `41 passed`.
- Full suite:
  `../.venv/bin/python -m pytest -q tests`
  -> `250 passed, 1 warning`.
- Hygiene:
  `git diff --check` passed.

Conclusion: protocol hygiene only. It prevents future profile changes from
silently changing the workload-generator query cap when tuning inference chunk
size, and makes coverage-grid report slicing less fragile.

### Held-out ship-balanced historical prior

Focused benchmark:
`artifacts/manual/heldout050607_c30_historicalprior_shipbalanced_retfreq_20260514`.

Configuration:

- Train: AIS `2026-02-05`; validation: `2026-02-06`; eval:
  `2026-02-07`.
- `historical_prior`, `query_coverage=30%`, `n_queries=24`,
  `max_queries=512`, `range_label_mode=usefulness_ship_balanced`,
  `range_train_workload_replicates=4`,
  `range_replicate_target_aggregation=label_mean`,
  `mlqds_hybrid_mode=stratified`, `mlqds_temporal_fraction=0.30`.
- Protocol flags stayed valid; eval workload was not floor-dominated:
  `107` eval queries, `30.13%` final coverage, near-duplicate rate `0.0`.

Result:

| Compression | MLQDS | Uniform | DP | TemporalRandomFill |
| ---: | ---: | ---: | ---: | ---: |
| `1%` | `0.071331` | `0.074838` | `0.071081` | `0.073021` |
| `2%` | `0.127188` | `0.133392` | `0.119772` | `0.125038` |
| `5%` | `0.208181` | `0.211751` | `0.185568` | `0.211369` |
| `10%` | `0.295345` | `0.315642` | `0.273035` | `0.312540` |
| `15%` | `0.377508` | `0.387494` | `0.343327` | `0.384329` |
| `20%` | `0.438875` | `0.456034` | `0.414393` | `0.448841` |
| `30%` | `0.554834` | `0.569004` | `0.517197` | `0.570399` |

Audit: wins vs uniform `0/7`, wins vs DP `7/7`, low-budget wins vs uniform
`0/3`, low-budget wins vs DP `3/3`.

Matched `5%` diagnosis:

- `RangeUseful`: MLQDS `0.208181`, uniform `0.211751`, DP `0.185568`.
- The model fits the train target almost perfectly but still loses eval:
  target tau `+0.995`, matched target-recall delta `+0.5827`, low-budget
  delta `+0.4414`.
- Losses vs uniform are broad: `RangePointF1=0.087565` vs `0.089271`,
  `ShipF1=0.651624` vs `0.655058`,
  `EntryExitF1=0.172259` vs `0.193753`,
  `CrossingF1=0.087357` vs `0.096723`,
  `TemporalCov=0.244292` vs `0.260320`,
  `GapCov=0.225971` vs `0.247608`, and
  `ShapeScore=0.153274` vs `0.159903`.
- Only `ShipCov` and `TurnCov` improve. That is not enough.
- Geometry is worse than both baselines: MLQDS `AvgSED=0.7552 km` vs
  uniform `0.5391 km` and DP `0.3660 km`. Length preservation is slightly
  above uniform (`0.9297` vs `0.9276`) but below DP (`0.9607`).
- MLQDS latency is still high at `517.02ms`.

Conclusion:

- Ship-balanced labels do not generalize across AIS days for the strongest
  historical-prior branch.
- The failure is now clear: train-day historical support can be fitted, but it
  is not a stable future-day retained-set policy. Treating this as final
  workload-blind success would be wrong.

### Conservative swap residual diagnostic

Focused benchmark:
`artifacts/manual/heldout050607_c30_historicalprior_shipbalanced_swap085_20260514`.

Configuration:

- Same held-out AIS split and ship-balanced historical-prior target as above.
- Changed selector to `mlqds_hybrid_mode=swap`,
  `mlqds_temporal_fraction=0.85`, and `mlqds_diversity_bonus=0.02`.
  This starts from full uniform temporal sampling and swaps only the
  unprotected budget share by learned score.

Result:

| Compression | MLQDS | Uniform | DP | TemporalRandomFill |
| ---: | ---: | ---: | ---: | ---: |
| `1%` | `0.074838` | `0.074838` | `0.071081` | `0.074838` |
| `2%` | `0.133392` | `0.133392` | `0.119772` | `0.133392` |
| `5%` | `0.209946` | `0.211751` | `0.185568` | `0.213749` |
| `10%` | `0.312233` | `0.315642` | `0.273035` | `0.321374` |
| `15%` | `0.378627` | `0.387494` | `0.343327` | `0.387910` |
| `20%` | `0.440780` | `0.456034` | `0.414393` | `0.447858` |
| `30%` | `0.552194` | `0.569004` | `0.517197` | `0.559323` |

Audit: wins vs uniform `0/7`, low-budget wins vs uniform `0/3`.

Diagnosis:

- Conservative swap reduces the damage versus the stratified selector, but it
  still loses uniform at every compression target.
- At `5%`, `10%`, and `15%`, random residual swaps beat both uniform and the
  learned historical-prior swaps. The learned residual ranking is worse than
  random on held-out day `2026-02-07`.
- Train-target fit under swap remains excellent as an ordering diagnostic
  (`tau=+0.998`), but target-recall lift falls to `+0.0867` at matched `5%`
  because the selector is mostly protected uniform points.
- Geometry improves relative to stratified historical prior (`AvgSED=0.6034`
  vs `0.7552 km`) but remains worse than uniform (`0.5391 km`).

Conclusion: temporal scaffolding can make the method less bad, but the useful
part is the scaffold. The learned residual choices are not adding reliable
held-out range usefulness. This is not a valid final-success path.

### Local paired-swap selector diagnostic

Implementation:

- Added `mlqds_hybrid_mode=local_swap`.
- It is still query-blind. It starts from the full uniform temporal retained
  set, chooses learned-score additions from non-base points, and removes the
  nearest removable non-endpoint temporal-base point for each addition.
- Added CLI support and a focused selector unit test.
- Updated `mlqds_diversity_bonus` help to state that it applies to
  `fill`, `swap`, and `local_swap`, but not `stratified`.

Validation so far:

- `../.venv/bin/python -m py_compile experiments/benchmark_report.py experiments/experiment_cli.py simplification/simplify_trajectories.py tests/test_benchmark_runner.py tests/test_metrics.py tests/test_torch_runtime_controls.py`
  passed.
- Targeted tests:
  `../.venv/bin/python -m pytest -q tests/test_benchmark_runner.py tests/test_metrics.py tests/test_torch_runtime_controls.py`
  -> `85 passed`.
- Full suite:
  `../.venv/bin/python -m pytest -q tests`
  -> `251 passed, 1 warning`.
- Hygiene:
  `git diff --check` passed.

Held-out 2026-02-05/06/07, coverage target `30%`,
`historical_prior`, ship-balanced labels, `local_swap`, temporal fraction
`0.85`:

| Compression | MLQDS | Uniform | DP | TemporalRandomFill |
| ---: | ---: | ---: | ---: | ---: |
| `1%` | `0.074838` | `0.074838` | `0.071081` | `0.074838` |
| `2%` | `0.133392` | `0.133392` | `0.119772` | `0.133392` |
| `5%` | `0.215024` | `0.211751` | `0.185568` | `0.216216` |
| `10%` | `0.310196` | `0.315642` | `0.273035` | `0.317771` |
| `15%` | `0.381925` | `0.387494` | `0.343327` | `0.382734` |
| `20%` | `0.457207` | `0.456034` | `0.414393` | `0.458854` |
| `30%` | `0.558208` | `0.569004` | `0.517197` | `0.563904` |

Matched `5%`: MLQDS `0.215024`, uniform `0.211751`, DP `0.185568`.
This is the first held-out matched-cell win for this branch, but it is not
enough. Audit wins vs uniform are only `2/7`; low-budget wins vs uniform are
`1/3`; wins vs DP are `7/7`. It also trails TemporalRandomFill at `5%`,
`10%`, `15%`, `20%`, and `30%`.

Matched `5%` component diagnosis:

- Improves over uniform on `ShipCov`, `EntryExitF1`, `TemporalCov`,
  `TurnCov`, and `ShapeScore`.
- Still loses `RangePointF1`, `ShipF1`, `CrossingF1`, and `GapCov`.
- Geometry is close but still worse than uniform:
  `AvgSED=0.5568 km` vs uniform `0.5391 km`. Length preservation is slightly
  better: `0.9308` vs `0.9276`.
- Latency remains high at `521.65ms`.

Other local-swap checks:

- `local_swap` temporal fraction `0.70` on the same held-out split failed:
  matched MLQDS `0.208409` vs uniform `0.211751`; audit wins vs uniform
  `0/7`. More learned residual budget made the retained set worse.
- `local_swap` temporal fraction `0.85` with the original retained-frequency
  usefulness labels was weaker than ship-balanced labels: matched MLQDS
  `0.214311`, audit wins vs uniform `1/7`.
- The earlier guarded 2026-02-02/03/04 split also failed with local swap:
  matched MLQDS `0.279959` vs uniform `0.282023`; audit wins vs uniform
  `2/7`, low-budget wins `0/3`. This is worse than the previous stratified
  ship-balanced run on that split.

Conclusion:

- `local_swap` is a useful selector diagnostic. It reduces continuity damage
  enough to expose a small held-out matched-cell gain.
- It is not a final solution. It relies on heavy uniform protection, loses most
  audit cells, does not consistently beat TemporalRandomFill, and gets worse
  when more budget is delegated to learned residual choices.
- The next serious change should improve the residual ranking itself. More
  selector tuning is low expected value unless it is tied to a new target or
  model signal.

### TemporalRandomFill audit reporting

Implementation cleanup:

- Added audit counts and deltas versus `TemporalRandomFill` across the
  compression grid:
  `audit_beats_temporal_random_fill_range_usefulness_count`,
  `audit_low_beats_temporal_random_fill_range_usefulness_count`,
  `audit_min_vs_temporal_random_fill_range_usefulness`,
  `audit_mean_vs_temporal_random_fill_range_usefulness`,
  `audit_min_low_vs_temporal_random_fill_range_usefulness`, and
  `audit_mean_low_vs_temporal_random_fill_range_usefulness`.
- Added `audit_missing_temporal_random_fill_count` so missing random-fill
  diagnostics are explicit instead of silently absent.
- Added the important TemporalRandomFill audit fields to the compact markdown
  benchmark table.

Reason:

- Recent local-swap and swap failures show that learned residual choices are
  often worse than random residual choices. The benchmark report needs to make
  that visible without manually opening child `example_run.json` files.

### Local-swap utility target diagnostic

Implementation:

- Added `range_training_target_mode=local_swap_utility_frequency`.
- This is training-only query-aware supervision for the actual `local_swap`
  action: start from the full uniform temporal retained set, pair each candidate
  with the nearest removable non-endpoint temporal-base point, score the
  replacement against train queries, and label only positive `RangeUseful`
  gains.
- The target requires `mlqds_hybrid_mode=local_swap`. Using it with another
  selector would be a mismatched target.
- Reused the existing `range_set_utility_*` candidate/mass controls and updated
  CLI help to say those knobs also apply to `local_swap_utility_frequency`.
- Added benchmark-report fields for local-swap utility candidate counts,
  positive-gain counts, selected counts, and source mass.

Validation:

- `../.venv/bin/python -m py_compile training/training_targets.py experiments/experiment_pipeline.py experiments/experiment_cli.py experiments/benchmark_report.py tests/test_teacher_distillation.py tests/test_benchmark_runner.py`
  passed.
- Targeted tests:
  `../.venv/bin/python -m pytest -q tests/test_teacher_distillation.py tests/test_benchmark_runner.py tests/test_torch_runtime_controls.py tests/test_metrics.py`
  -> `108 passed, 1 warning`.
- Full suite:
  `../.venv/bin/python -m pytest -q tests`
  -> `252 passed, 1 warning`.
- Hygiene:
  `git diff --check` passed.

Focused held-out 2026-02-05/06/07 diagnostics, coverage target `30%`,
`historical_prior`, `local_swap`, temporal fraction `0.85`, ship-balanced
labels, one train workload replicate:

| Run | Matched MLQDS | Uniform | DP | TemporalRandomFill | Wins vs uniform | Low-budget wins | Wins vs TRF | Target mass | Target time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `heldout050607_c30_historicalprior_localswaputility_point32_20260514` | `0.208867` | `0.211751` | `0.185568` | `0.216216` | `3/7` | `0/3` | `2/7` | `5.8496` | `20.09s` |
| `heldout050607_c30_historicalprior_localswaputility_point32_loww1_20260514` | `0.207973` | `0.211751` | `0.185568` | `0.216216` | `1/7` | `0/3` | `2/7` | `4.8820` | `20.51s` |

Unweighted audit grid:

| Compression | MLQDS | Uniform | DP | TemporalRandomFill |
| ---: | ---: | ---: | ---: | ---: |
| `1%` | `0.074838` | `0.074838` | `0.071081` | `0.074838` |
| `2%` | `0.133392` | `0.133392` | `0.119772` | `0.133392` |
| `5%` | `0.208867` | `0.211751` | `0.185568` | `0.216216` |
| `10%` | `0.315761` | `0.315642` | `0.273035` | `0.317771` |
| `15%` | `0.390854` | `0.387494` | `0.343327` | `0.382734` |
| `20%` | `0.445237` | `0.456034` | `0.414393` | `0.458854` |
| `30%` | `0.571075` | `0.569004` | `0.517197` | `0.563904` |

Diagnosis:

- The target is not empty. It scored `11818` candidate replacements, found
  `5633` positive-gain candidates, selected `3305`, and accumulated selected
  train-query gain mass `145.29`.
- Train-target fit is excellent (`tau=+0.998`, matched target-recall delta
  `+0.1356`), but eval `5%` still loses uniform and TemporalRandomFill.
- The unweighted target improves some medium/high compression cells
  (`10%`, `15%`, `30%`) but loses the important `5%` cell and all low-budget
  cells. Weighting low budgets with `range_target_budget_weight_power=1.0`
  makes the matched cell worse, not better.
- Target construction is now the runtime bottleneck at about `20s` even with
  candidate limit `32`.

Conclusion:

- This was the right alignment test for `local_swap`, and it failed where it
  matters. The train-query local replacement gains do not transfer into a
  useful held-out 5% retained policy.
- More `local_swap_utility_frequency` mass or budget knobs are low expected
  value. The branch is not final-success evidence.

### Local-delta swap selector diagnostic

Implementation:

- Added `mlqds_hybrid_mode=local_delta_swap`.
- It starts from full uniform temporal sampling like `swap` and `local_swap`.
  For each learned candidate it pairs the nearest removable non-endpoint
  temporal-base point, but it performs the replacement only when the candidate
  score is higher than that paired base-point score.
- This is query-blind and deterministic. It is a confidence gate on learned
  replacement decisions, not eval-query conditioning.
- Allowed `local_swap_utility_frequency` targets to run with
  `local_delta_swap`, because they score the same paired replacement action.

Validation:

- `../.venv/bin/python -m py_compile simplification/simplify_trajectories.py experiments/experiment_cli.py training/training_targets.py tests/test_metrics.py`
  passed.
- Targeted tests:
  `../.venv/bin/python -m pytest -q tests/test_metrics.py tests/test_torch_runtime_controls.py tests/test_teacher_distillation.py`
  -> `83 passed, 1 warning`.
- Full suite:
  `../.venv/bin/python -m pytest -q tests`
  -> `254 passed, 1 warning`.
- Hygiene:
  `git diff --check` passed.

Focused held-out 2026-02-05/06/07 diagnostics, coverage target `30%`,
`historical_prior`, temporal fraction `0.85`:

| Run | Target | Matched MLQDS | Uniform | DP | TemporalRandomFill | Wins vs uniform | Low-budget wins | Wins vs TRF |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `heldout050607_c30_historicalprior_shipbalanced_localdelta085_20260515` | ship-balanced retained-frequency, 4 train workloads | `0.210386` | `0.211751` | `0.185568` | `0.219589` | `3/7` | `0/3` | `2/7` |
| `heldout050607_c30_historicalprior_localdeltautility_point32_20260515` | local-swap utility, 1 train workload, point mass, candidate limit `32` | `0.208786` | `0.211751` | `0.185568` | `0.219589` | `3/7` | `0/3` | `3/7` |

Retained-frequency local-delta audit grid:

| Compression | MLQDS | Uniform | DP | TemporalRandomFill |
| ---: | ---: | ---: | ---: | ---: |
| `1%` | `0.074838` | `0.074838` | `0.071081` | `0.074838` |
| `2%` | `0.133392` | `0.133392` | `0.119772` | `0.133392` |
| `5%` | `0.210386` | `0.211751` | `0.185568` | `0.219589` |
| `10%` | `0.316215` | `0.315642` | `0.273035` | `0.316431` |
| `15%` | `0.395369` | `0.387494` | `0.343327` | `0.386703` |
| `20%` | `0.452509` | `0.456034` | `0.414393` | `0.452501` |
| `30%` | `0.572658` | `0.569004` | `0.517197` | `0.574628` |

Diagnosis:

- The score-delta gate improves geometry and some medium/high cells versus
  earlier local-swap variants, but it still misses the matched `5%` cell and
  has zero low-budget wins.
- Combining the gate with the local-swap utility target does not rescue the
  low-budget failure. It improves `10%` and `30%`, but still loses `5%` and
  `20%`.
- Latency remains high: retained-frequency local-delta MLQDS latency is
  `584.37ms`.

Conclusion:

- Local replacement confidence is a valid diagnostic, but not the missing
  ingredient. It confirms the same failure: the learned score can identify some
  useful medium/high-budget replacements, but it cannot reliably improve the
  low-budget retained set over uniform temporal spacing.
- Stop spending effort on local-swap selector variants unless the underlying
  target/model changes first.

### Multi-train historical-prior local-swap coverage sweep

Setup:

- Train days: `2026-02-02` through `2026-02-05`.
- Validation day: `2026-02-06`; eval day: `2026-02-07`.
- Model/target: `historical_prior`, ship-balanced retained-frequency labels,
  four train workload replicates, `label_mean` aggregation.
- Selector: `mlqds_hybrid_mode=local_swap`, `mlqds_temporal_fraction=0.85`,
  `mlqds_diversity_bonus=0.02`.
- Eval protocol remains workload-blind: MLQDS, uniform, and DP masks are frozen
  before held-out eval query scoring for every audited compression ratio.

Matched `5%` compression summary:

| Coverage target | Actual train/eval coverage | MLQDS | Uniform | DP | TemporalRandomFill | Wins vs uniform | Low-budget wins | Wins vs TRF | Latency |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `5%` | `5.23% / 5.89%` | `0.304912` | `0.300945` | `0.221774` | `0.270232` | `3/7` | `1/3` | `4/7` | `870.10ms` |
| `10%` | `10.42% / 10.03%` | `0.289699` | `0.267976` | `0.203119` | `0.263826` | `5/7` | `1/3` | `5/7` | `902.36ms` |
| `15%` | `15.04% / 15.35%` | `0.263460` | `0.244701` | `0.188116` | `0.242981` | `5/7` | `1/3` | `5/7` | `868.52ms` |
| `30%` | `30.01% / 30.13%` | `0.218596` | `0.211751` | `0.185568` | `0.216216` | `4/7` | `1/3` | `4/7` | `882.78ms` |

Compression-grid results:

| Coverage | Compression | MLQDS | Uniform | DP | TemporalRandomFill |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `5%` | `1%` | `0.179347` | `0.179347` | `0.159288` | `0.179347` |
| `5%` | `2%` | `0.223040` | `0.223040` | `0.181939` | `0.223040` |
| `5%` | `5%` | `0.304912` | `0.300945` | `0.221774` | `0.270232` |
| `5%` | `10%` | `0.422577` | `0.426356` | `0.328137` | `0.419124` |
| `5%` | `15%` | `0.500853` | `0.493875` | `0.384553` | `0.496129` |
| `5%` | `20%` | `0.495013` | `0.496718` | `0.442629` | `0.495329` |
| `5%` | `30%` | `0.646638` | `0.640910` | `0.534452` | `0.634951` |
| `10%` | `1%` | `0.140792` | `0.140792` | `0.118559` | `0.140792` |
| `10%` | `2%` | `0.197449` | `0.197449` | `0.152526` | `0.197449` |
| `10%` | `5%` | `0.289699` | `0.267976` | `0.203119` | `0.263826` |
| `10%` | `10%` | `0.402313` | `0.388761` | `0.329504` | `0.377287` |
| `10%` | `15%` | `0.456131` | `0.445505` | `0.411931` | `0.451391` |
| `10%` | `20%` | `0.513000` | `0.487122` | `0.483975` | `0.482709` |
| `10%` | `30%` | `0.628778` | `0.615274` | `0.580826` | `0.611539` |
| `15%` | `1%` | `0.094439` | `0.094439` | `0.075757` | `0.094439` |
| `15%` | `2%` | `0.159050` | `0.159050` | `0.137839` | `0.159050` |
| `15%` | `5%` | `0.263460` | `0.244701` | `0.188116` | `0.242981` |
| `15%` | `10%` | `0.364434` | `0.360089` | `0.288725` | `0.350844` |
| `15%` | `15%` | `0.419627` | `0.417688` | `0.367964` | `0.405420` |
| `15%` | `20%` | `0.499201` | `0.475298` | `0.433824` | `0.469858` |
| `15%` | `30%` | `0.603939` | `0.595752` | `0.521309` | `0.588727` |
| `30%` | `1%` | `0.074838` | `0.074838` | `0.071081` | `0.074838` |
| `30%` | `2%` | `0.133392` | `0.133392` | `0.119772` | `0.133392` |
| `30%` | `5%` | `0.218596` | `0.211751` | `0.185568` | `0.216216` |
| `30%` | `10%` | `0.315237` | `0.315642` | `0.273035` | `0.317771` |
| `30%` | `15%` | `0.393674` | `0.387494` | `0.343327` | `0.382734` |
| `30%` | `20%` | `0.463240` | `0.456034` | `0.414393` | `0.458854` |
| `30%` | `30%` | `0.577452` | `0.569004` | `0.517197` | `0.563904` |

Diagnostics:

- This is the first branch that beats uniform and DP at the matched `5%`
  compression cell for all four required coverage targets on the
  `2026-02-02..07` split.
- It is still not final success. Across the full coverage/compression grid it
  wins `17/28` cells versus uniform, ties uniform at every `1%` and `2%` cell,
  and loses several medium/high-budget cells. The low-budget story is therefore
  not strong enough: the `5%` cells improve, but `1%` and `2%` are protected
  temporal ties rather than learned wins.
- The `30%` matched cell has real component gains over uniform:
  `RangePointF1=0.091175` vs `0.089271`, `ShipF1=0.667008` vs `0.655058`,
  `ShipCov=0.109885` vs `0.104853`, `EntryExitF1=0.199307` vs `0.193753`,
  `CrossingF1=0.101842` vs `0.096723`, `TemporalCov=0.274301` vs `0.260320`,
  `GapCov=0.252489` vs `0.247608`, `TurnCov=0.099750` vs `0.090915`, and
  `ShapeScore=0.171505` vs `0.159903`.
- Geometry remains worse than uniform. At `30%` coverage / `5%` compression,
  MLQDS `AvgSED=0.5703 km` versus uniform `0.5391 km`, while length
  preservation is slightly better (`0.9305` vs `0.9276`).
- Runtime is a practical problem. MLQDS freeze latency is around
  `870-902ms`, while uniform is single-digit milliseconds. Cold or changed
  train workload label prep can dominate total runtime (`66-68s` at `30%`
  coverage); cache-hit reruns are much cheaper.

Selector control:

- Same multi-train ship-balanced target with `mlqds_hybrid_mode=stratified`
  failed at `30%` coverage:
  `artifacts/manual/multitrain0205_c30_historicalprior_shipbalanced_stratified_20260515`.
- Matched `RangeUseful`: MLQDS `0.199061`, uniform `0.211751`, DP `0.185568`,
  TemporalRandomFill `0.211369`.
- Audit wins vs uniform: `0/7`; low-budget wins: `0/3`.
- Geometry worsened: `AvgSED=0.7086 km` versus uniform `0.5391 km`.
- This is important. The broad gains above rely on the conservative local-swap
  temporal scaffold. The learned KNN score is adding useful swaps in some
  cells, but the clean learned-stratified selector is not strong enough.

Held-out workload-generator checks:

| Artifact | Setting | Matched MLQDS | Uniform | DP | TemporalRandomFill | Wins vs uniform | Low-budget wins | Wins vs DP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `multitrain0205_c30_historicalprior_shipbalanced_localswap085_seed47_20260515` | seed `47`, mixed-density train/eval | `0.236355` | `0.226328` | `0.233481` | `0.225552` | `5/7` | `1/3` | `7/7` |
| `multitrain0205_c30_historicalprior_shipbalanced_localswap085_seed47_evaldense_20260515` | train mixed-density, eval/checkpoint dense | `0.218542` | `0.211209` | `0.206258` | `0.210162` | `2/7` | `1/3` | `6/7` |

Seed `47` mixed-density supports a real matched-cell win and broad DP wins, but
the dense held-out setting exposes the same weakness: matched `5%` wins, but the
audit grid is only `2/7` versus uniform and the mean audit delta is slightly
negative (`-0.00038`). This is not robust final acceptance.

Conclusion:

- The workload-blind protocol is now doing the right thing, and this branch
  produces sensible retained trajectories with matched-cell gains across the
  required coverage targets.
- It still fails the acceptance bar. The result depends on a heavy temporal
  scaffold (`0.85`), low-budget `1%` and `2%` cells are ties rather than learned
  improvements, the dense held-out setting is not a broad grid win, and latency
  is far above simple baselines.
- Treat `historical_prior + ship-balanced retained-frequency + local_swap` as
  the current best diagnostic/partial-success branch, not as final
  workload-blind Range-QDS success.

### Low-budget local-delta diagnostics

Reason:

- The `local_swap` `0.85` branch ties uniform at `1%` and `2%` because
  `ceil(total_keep * 0.85)` consumes the whole budget at those sizes. No learned
  replacement is attempted.
- To test whether the low-budget failure is only this rounding artifact, ran a
  lower-scaffold variant with `mlqds_hybrid_mode=local_delta_swap` and
  `mlqds_temporal_fraction=0.66`. This allows learned replacements at `1%` and
  `2%`, but keeps a query-blind score-delta gate.

Focused `30%` coverage diagnostics on train days `2026-02-02..05`,
validation `2026-02-06`, eval `2026-02-07`:

| Artifact | Target | Matched MLQDS | Uniform | DP | TemporalRandomFill | Wins vs uniform | Low-budget wins | Runtime bottleneck |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `multitrain0205_c30_historicalprior_shipbalanced_localdelta066_20260515` | ship-balanced retained-frequency, 4 train workloads | `0.208588` | `0.211751` | `0.185568` | `0.211599` | `2/7` | `0/3` | train fit `22.99s` |
| `multitrain0205_c30_historicalprior_shipbalanced_localdelta066_rawscore_20260515` | same target, but `mlqds_score_mode=raw` | `0.206221` | `0.211751` | `0.185568` | `0.211599` | `2/7` | `0/3` | train fit `22.81s` |
| `multitrain0205_c30_historicalprior_localdelta066_localswaputility_point32_20260515` | local-swap utility, 1 train workload, point mass, candidate limit `32` | `0.198615` | `0.211751` | `0.185568` | `0.211599` | `1/7` | `1/3` | target build `181.31s` |
| `multitrain0205_c30_historicalprior_continuity_localdelta066_20260515` | continuity-retained target, 4 train workloads | `0.210109` | `0.211751` | `0.185568` | `0.211599` | `3/7` | `1/3` | target build `74.33s` |

Audit grids:

| Artifact | Compression | MLQDS | Uniform | DP | TemporalRandomFill |
| --- | ---: | ---: | ---: | ---: | ---: |
| retained-frequency local-delta `0.66` | `1%` | `0.072106` | `0.074838` | `0.071081` | `0.073021` |
| retained-frequency local-delta `0.66` | `2%` | `0.130258` | `0.133392` | `0.119772` | `0.128149` |
| retained-frequency local-delta `0.66` | `5%` | `0.208588` | `0.211751` | `0.185568` | `0.211599` |
| retained-frequency local-delta `0.66` | `10%` | `0.322988` | `0.315642` | `0.273035` | `0.326294` |
| retained-frequency local-delta `0.66` | `15%` | `0.387708` | `0.387494` | `0.343327` | `0.383078` |
| retained-frequency local-delta `0.66` | `20%` | `0.454175` | `0.456034` | `0.414393` | `0.443481` |
| retained-frequency local-delta `0.66` | `30%` | `0.555129` | `0.569004` | `0.517197` | `0.566597` |
| raw-score local-delta `0.66` | `1%` | `0.072106` | `0.074838` | `0.071081` | `0.073021` |
| raw-score local-delta `0.66` | `2%` | `0.123920` | `0.133392` | `0.119772` | `0.128149` |
| raw-score local-delta `0.66` | `5%` | `0.206221` | `0.211751` | `0.185568` | `0.211599` |
| raw-score local-delta `0.66` | `10%` | `0.324545` | `0.315642` | `0.273035` | `0.326294` |
| raw-score local-delta `0.66` | `15%` | `0.383361` | `0.387494` | `0.343327` | `0.383078` |
| raw-score local-delta `0.66` | `20%` | `0.459717` | `0.456034` | `0.414393` | `0.443481` |
| raw-score local-delta `0.66` | `30%` | `0.560554` | `0.569004` | `0.517197` | `0.566597` |
| local-swap utility local-delta `0.66` | `1%` | `0.081662` | `0.074838` | `0.071081` | `0.073021` |
| local-swap utility local-delta `0.66` | `2%` | `0.118164` | `0.133392` | `0.119772` | `0.128149` |
| local-swap utility local-delta `0.66` | `5%` | `0.198615` | `0.211751` | `0.185568` | `0.211599` |
| local-swap utility local-delta `0.66` | `10%` | `0.302894` | `0.315642` | `0.273035` | `0.326294` |
| local-swap utility local-delta `0.66` | `15%` | `0.382995` | `0.387494` | `0.343327` | `0.383078` |
| local-swap utility local-delta `0.66` | `20%` | `0.443487` | `0.456034` | `0.414393` | `0.443481` |
| local-swap utility local-delta `0.66` | `30%` | `0.549919` | `0.569004` | `0.517197` | `0.566597` |
| continuity local-delta `0.66` | `1%` | `0.076332` | `0.074838` | `0.071081` | `0.073021` |
| continuity local-delta `0.66` | `2%` | `0.129376` | `0.133392` | `0.119772` | `0.128149` |
| continuity local-delta `0.66` | `5%` | `0.210109` | `0.211751` | `0.185568` | `0.211599` |
| continuity local-delta `0.66` | `10%` | `0.324995` | `0.315642` | `0.273035` | `0.326294` |
| continuity local-delta `0.66` | `15%` | `0.399299` | `0.387494` | `0.343327` | `0.383078` |
| continuity local-delta `0.66` | `20%` | `0.453912` | `0.456034` | `0.414393` | `0.443481` |
| continuity local-delta `0.66` | `30%` | `0.552990` | `0.569004` | `0.517197` | `0.566597` |

Diagnosis:

- Lowering the scaffold proves the problem is not just budget rounding. Learned
  low-budget replacements can be attempted, but they do not transfer. The
  retained-frequency target has excellent train fit (`tau=+0.998`, low-budget
  target-recall delta `+0.2061`) while held-out `1%`, `2%`, and `5%` all lose
  uniform.
- Using raw historical-prior values for the delta gate is worse than ordinal
  rank. It raises the train target-recall delta (`+0.3178` matched,
  `+0.2615` low-budget) but lowers held-out matched `RangeUseful` to
  `0.206221` and badly damages geometry (`AvgSED=0.9897 km` vs uniform
  `0.5391 km`). The absolute KNN target value is not a transferable confidence
  signal.
- The paired-swap utility target can force a learned `1%` win, but it destroys
  `2%`, `5%`, and the broader grid. It scored `34,863` replacement candidates,
  found `22,056` positive train-query gains, and selected `9,899`, but the
  resulting target has only `5.0%` positives and does not generalize. It is also
  too slow for routine iteration (`181.31s` target build on this slice).
- Continuity-retained labels improve some medium-budget cells (`10%`, `15%`)
  and produce a small `1%` win, but still fail matched `5%`, fail `2%`, and
  damage global geometry (`AvgSED=0.7152 km` vs uniform `0.5391 km`).
- All three low-scaffold variants preserve DP wins but fail the actual uniform
  bar. This is the core failure: uniform temporal spacing is a strong blind
  low-budget prior, and current historical-prior scores overfit train-query
  replacement gains.

Conclusion:

- Do not change selector rounding to make low-budget swaps happen by default.
  The empirical result is worse, not better.
- Do not expand `local_swap_utility_frequency` without a new transfer mechanism
  or a major optimization; it is both slower and lower quality.
- The remaining missing piece is not another scalar target reshuffle. It is a
  transferable confidence model for when a learned replacement should override
  uniform temporal structure, or a fundamentally stronger blind representation
  of future range-useful route context.

### Anchor-prior training mix diagnostics

Reason:

- The current best branch trains the blind historical prior on mixed-density
  generated workloads. Dense held-out eval weakens the grid win. Tested whether
  mixing train anchor priors improves robustness to held-out workload-generator
  settings.

Focused `30%` coverage diagnostics on train days `2026-02-02..05`,
validation `2026-02-06`, eval `2026-02-07`, with the best known
`local_swap` `0.85` selector:

| Artifact | Train anchor modes | Matched MLQDS | Uniform | DP | TemporalRandomFill | Wins vs uniform | Low-budget wins | Wins vs TRF |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `multitrain0205_c30_historicalprior_shipbalanced_localswap085_anchormix_20260515` | mixed-density, dense, uniform, sparse | `0.212494` | `0.211751` | `0.185568` | `0.216216` | `1/7` | `1/3` | `1/7` |
| `multitrain0205_c30_historicalprior_shipbalanced_localswap085_anchor3_20260515` | mixed-density, dense, uniform | `0.215155` | `0.211751` | `0.185568` | `0.216216` | `1/7` | `1/3` | `0/7` |

`anchor3` audit grid:

| Compression | MLQDS | Uniform | DP | TemporalRandomFill |
| ---: | ---: | ---: | ---: | ---: |
| `1%` | `0.074838` | `0.074838` | `0.071081` | `0.074838` |
| `2%` | `0.133392` | `0.133392` | `0.119772` | `0.133392` |
| `5%` | `0.215155` | `0.211751` | `0.185568` | `0.216216` |
| `10%` | `0.311640` | `0.315642` | `0.273035` | `0.317771` |
| `15%` | `0.382230` | `0.387494` | `0.343327` | `0.382734` |
| `20%` | `0.448852` | `0.456034` | `0.414393` | `0.458854` |
| `30%` | `0.563237` | `0.569004` | `0.517197` | `0.563904` |

Diagnosis:

- The four-mode mix includes a sparse train replicate that failed to reach the
  requested `30%` range coverage (`18.21%` with the query cap). That makes the
  supervision distribution suspect.
- Removing sparse avoids the bad coverage replicate, but still dilutes the best
  branch. `anchor3` improves the matched `5%` cell over uniform by only
  `+0.003405`, versus `+0.006845` for the mixed-density-only best branch, and
  loses the audit mean against uniform (`-0.002687`).
- Train fit is high (`tau=+0.993`), but target-recall lift is weak
  (`+0.0700` matched, `+0.0233` low-budget). The model is ranking the mixed
  labels, but those labels no longer create a useful retained-set ordering.

Conclusion:

- Do not spend another held-out dense run on anchor-prior mixing. It weakened
  the known best branch before testing the harder setting.
- The failure mode is consistent with earlier diagnostics: aggregate historical
  workload frequency is too blunt. More generated workload variety does not
  automatically create a stronger blind prior.

### Teacher distillation on the current best selector

Reason:

- Earlier teacher-distillation failures used weaker stratified/selectors. The
  current best branch is `historical_prior` plus ship-balanced labels and
  `local_swap` `0.85`. Tested whether a query-aware `range_aware` teacher gives
  a cleaner blind target under that selector.

Focused `30%` coverage diagnostic:
`artifacts/manual/multitrain0205_c30_historicalprior_teacherretfreq_localswap085_20260515`.

Configuration:

- Train days `2026-02-02..05`, validation `2026-02-06`, eval `2026-02-07`.
- `model_type=historical_prior`, `range_label_mode=usefulness_ship_balanced`.
- `range_train_workload_replicates=4`, `range_replicate_target_aggregation=label_mean`.
- `range_teacher_distillation_mode=retained_frequency`,
  `range_teacher_epochs=4`.
- `mlqds_hybrid_mode=local_swap`, `mlqds_temporal_fraction=0.85`,
  `mlqds_diversity_bonus=0.02`.

Result:

| Compression | MLQDS | Uniform | DP | TemporalRandomFill |
| ---: | ---: | ---: | ---: | ---: |
| `1%` | `0.074838` | `0.074838` | `0.071081` | `0.074838` |
| `2%` | `0.133392` | `0.133392` | `0.119772` | `0.133392` |
| `5%` | `0.212709` | `0.211751` | `0.185568` | `0.216216` |
| `10%` | `0.316227` | `0.315642` | `0.273035` | `0.317771` |
| `15%` | `0.386466` | `0.387494` | `0.343327` | `0.382734` |
| `20%` | `0.457846` | `0.456034` | `0.414393` | `0.458854` |
| `30%` | `0.569163` | `0.569004` | `0.517197` | `0.563904` |

Diagnostics:

- Matched `5%` improves over uniform by only `+0.000958`, versus `+0.006845`
  for the current best ship-balanced retained-frequency branch.
- Audit wins versus uniform are `4/7`, but three of those are tiny
  (`10%`, `20%`, `30%`) and the mean audit delta is only `+0.000355`.
- Low-budget result remains `1/3`, with `1%` and `2%` still temporal ties.
- The distilled labels are diffuse: `46,052` positives, `62.97%` of train
  points, mass `9056.86`.
- Historical-prior fit to the distilled target is high (`tau=+0.979`), but the
  selector-relevant target-recall lift is weak (`+0.0511` matched,
  `+0.0170` low-budget). The student can rank the teacher aggregate, but the
  aggregate does not create a materially better retained-set policy.
- Geometry is slightly closer to uniform than the current best branch
  (`AvgSED=0.5519 km` vs uniform `0.5391 km`), but still worse. Latency also
  regresses to `968.91ms`.

Conclusion:

- Do not promote retained-frequency teacher distillation as the next path. It
  is weaker than direct ship-balanced retained-frequency labels under the same
  selector.
- The query-aware teacher signal is compressing into a broad positive-frequency
  prior, not a sharp blind confidence signal. Improving the teacher is unlikely
  to fix the current failure unless the aggregation becomes much less diffuse
  or the student gains a better query-free route-context representation.

### Score-cache runtime checkpoint

Reason:

- The design runtime notes call for reusing one score vector across the
  compression-ratio audit. The historical-prior branch was recomputing KNN
  scores for each audited compression ratio, making manual iteration slower
  without changing the retained-score ordering.

Change:

- Added `mlqds_simplification_scores(...)` as the budget-independent score
  conversion helper.
- `MLQDSMethod` now caches the converted simplification score vector per method
  instance and invalidates it when points, boundaries, query features,
  scoring/blend config, model/scaler identity, or geometry scores change.
- Retention still freezes masks before eval query scoring; this only avoids
  repeated model inference while freezing multiple budget masks from the same
  score vector.

Validation:

- Targeted tests:
  `../.venv/bin/python -m pytest -q tests/test_workload_blind_protocol.py tests/test_metrics.py tests/test_benchmark_runner.py`
  -> `78 passed, 1 warning`.
- Full suite after the shared evaluation change:
  `../.venv/bin/python -m pytest -q tests`
  -> `255 passed, 1 warning`.
- Compile check:
  `../.venv/bin/python -m py_compile evaluation/baselines.py simplification/mlqds_scoring.py tests/test_workload_blind_protocol.py`
  passed.
- End-to-end rerun:
  `artifacts/manual/multitrain0205_c30_historicalprior_shipbalanced_localswap085_scorecache_20260515`.

Runtime result on the current best `30%` coverage branch:

| Artifact | Matched MLQDS | Uniform | DP | Audit wins vs uniform | Low-budget wins | Freeze audit time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| before cache: `multitrain0205_c30_historicalprior_shipbalanced_localswap085_20260515` | `0.218596` | `0.211751` | `0.185568` | `4/7` | `1/3` | `4.50s` |
| after cache: `multitrain0205_c30_historicalprior_shipbalanced_localswap085_scorecache_20260515` | `0.218596` | `0.211751` | `0.185568` | `4/7` | `1/3` | `0.37s` |

Conclusion:

- This is a real iteration-cost fix and preserves retained-mask quality.
- It does not solve the product-quality failure. Primary MLQDS freeze latency
  is still `865.70ms` on this run because the first historical-prior KNN score
  pass remains expensive.

### Historical-prior support cap runtime diagnostic

Reason:

- The historical-prior model stores every train point by default, including
  low/zero target support. Since first-mask latency is dominated by the KNN
  pass over stored support, tested whether top-target per-trajectory support
  capping gives a usable speed/quality trade-off.

Focused `30%` coverage diagnostics on the current best branch:

| Artifact | Support ratio | Stored support | Matched MLQDS | Uniform | DP | Wins vs uniform | Low-budget wins | MLQDS latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `multitrain0205_c30_historicalprior_shipbalanced_localswap085_scorecache_20260515` | `1.00` | `73,132` | `0.218596` | `0.211751` | `0.185568` | `4/7` | `1/3` | `865.70ms` |
| `multitrain0205_c30_historicalprior_shipbalanced_localswap085_supp030_20260515` | `0.30` | `22,115` | `0.209186` | `0.211751` | `0.185568` | `2/7` | `0/3` | `526.73ms` |
| `multitrain0205_c30_historicalprior_shipbalanced_localswap085_supp050_20260515` | `0.50` | `36,608` | `0.209155` | `0.211751` | `0.185568` | `2/7` | `0/3` | `641.37ms` |

Audit grids:

| Support | Compression | MLQDS | Uniform | DP | TemporalRandomFill |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0.30` | `1%` | `0.074838` | `0.074838` | `0.071081` | `0.074838` |
| `0.30` | `2%` | `0.133392` | `0.133392` | `0.119772` | `0.133392` |
| `0.30` | `5%` | `0.209186` | `0.211751` | `0.185568` | `0.216216` |
| `0.30` | `10%` | `0.319484` | `0.315642` | `0.273035` | `0.317771` |
| `0.30` | `15%` | `0.387678` | `0.387494` | `0.343327` | `0.382734` |
| `0.30` | `20%` | `0.455885` | `0.456034` | `0.414393` | `0.458854` |
| `0.30` | `30%` | `0.566956` | `0.569004` | `0.517197` | `0.563904` |
| `0.50` | `1%` | `0.074838` | `0.074838` | `0.071081` | `0.074838` |
| `0.50` | `2%` | `0.133392` | `0.133392` | `0.119772` | `0.133392` |
| `0.50` | `5%` | `0.209155` | `0.211751` | `0.185568` | `0.216216` |
| `0.50` | `10%` | `0.321978` | `0.315642` | `0.273035` | `0.317771` |
| `0.50` | `15%` | `0.385855` | `0.387494` | `0.343327` | `0.382734` |
| `0.50` | `20%` | `0.452293` | `0.456034` | `0.414393` | `0.458854` |
| `0.50` | `30%` | `0.571313` | `0.569004` | `0.517197` | `0.563904` |

Diagnosis:

- Support caps reduce latency, but they remove too much context. Both capped
  branches lose the matched `5%` cell and all learned low-budget opportunities.
- `support_ratio=0.50` improves geometry relative to full support
  (`AvgSED=0.5505 km` vs `0.5703 km`) but still loses the primary
  `RangeUseful` target. That is not an acceptable trade-off.
- Train target fit tau drops sharply (`+0.998` full support to `+0.412` at
  `0.30` and `+0.477` at `0.50`) while target-recall lift stays superficially
  similar. The target-recall diagnostic is not sensitive enough to detect the
  retained-set quality loss from removing broad support.

Conclusion:

- Do not use historical-prior support capping as a final runtime fix.
- If first-mask latency must be reduced, the credible path is an approximate or
  indexed KNN implementation that keeps broad support, not pruning the support
  distribution.

### Historical-prior distance-kernel diagnostic

Reason:

- Tested whether the exact KNN pass could be sped up without pruning support by
  replacing `torch.cdist(...).square()` with an algebraically equivalent
  squared-L2 matrix-multiply kernel.

Result:

- Artifact:
  `artifacts/manual/multitrain0205_c30_historicalprior_shipbalanced_localswap085_matmulknn_20260515`.
- Matched `RangeUseful`: MLQDS `0.212090`, uniform `0.211751`, DP
  `0.185568`, TemporalRandomFill `0.216216`.
- Audit wins versus uniform fell to `2/7`; low-budget wins stayed `1/3`.
- First-mask MLQDS latency worsened to `936.50ms`, versus `865.70ms` for the
  `torch.cdist` score-cache run.
- Train target fit stayed high (`tau=+0.998`), so the regression is in
  retained ordering/runtime behavior, not a training-target failure.

Conclusion:

- Reverted the matrix-multiply distance kernel. It was both slower and worse
  on retained-set quality.
- Keep `torch.cdist` for the exact historical-prior path unless a real indexed
  or approximate search implementation is added with quality ablations.

Validation after reverting the kernel experiment:

- Targeted tests:
  `../.venv/bin/python -m pytest -q tests/test_model_features.py tests/test_workload_blind_protocol.py tests/test_metrics.py tests/test_benchmark_runner.py`
  -> `93 passed, 1 warning`.
- Full suite:
  `../.venv/bin/python -m pytest -q tests`
  -> `255 passed, 1 warning`.
- Hygiene:
  `git diff --check` passed.

### Training-footprint diversity diagnostic

Reason:

- Tested whether the current best local-swap branch failed because all training
  workloads used the same range footprint. Added six train workload replicates
  across `1.1km/2.5h`, `2.2km/5h`, and `4.4km/10h`, with `label_mean`
  aggregation. Evaluation stayed on the held-out `2.2km/5h` workload.

Result:

- Artifact:
  `artifacts/manual/multitrain0205_c30_historicalprior_shipbalanced_localswap085_footmix6_20260515`.
- Matched `RangeUseful`: MLQDS `0.211856`, uniform `0.211751`, DP
  `0.185568`, TemporalRandomFill `0.216216`.
- Audit wins: uniform `5/7`, low-budget uniform `1/3`, DP `7/7`,
  TemporalRandomFill `4/7`.
- Target: `24,483` positive labels (`33.5%`), mass `5,674.0`.
- Train fit: `tau=+0.996`, matched target-recall lift `+0.0542`,
  low-budget target-recall lift `+0.0181`.
- Geometry and runtime: MLQDS `AvgSED=0.5534 km` vs uniform `0.5391 km`;
  length preservation `0.9317` vs uniform `0.9276`; first-mask latency
  `877.43ms`; `range-training-prep` was the bottleneck at `100.98s`.
- One small-footprint training replicate stopped at `29.64%` coverage before
  the requested `30%`, so this also exposed a workload-generation cost/coverage
  trade-off.

Audit grid:

| Compression | MLQDS | Uniform | DP | TemporalRandomFill |
| ---: | ---: | ---: | ---: | ---: |
| `1%` | `0.074838` | `0.074838` | `0.071081` | `0.074838` |
| `2%` | `0.133392` | `0.133392` | `0.119772` | `0.133392` |
| `5%` | `0.211856` | `0.211751` | `0.185568` | `0.216216` |
| `10%` | `0.325085` | `0.315642` | `0.273035` | `0.317771` |
| `15%` | `0.391462` | `0.387494` | `0.343327` | `0.382734` |
| `20%` | `0.461336` | `0.456034` | `0.414393` | `0.458854` |
| `30%` | `0.576320` | `0.569004` | `0.517197` | `0.563904` |

Conclusion:

- Footprint diversity slightly improves high-budget audit cells but does not
  improve the matched `5%` cell in a meaningful way and still leaves low-budget
  wins at `1/3`.
- It is worse than the current best direct branch
  (`0.211856` vs `0.218596` matched `RangeUseful`) and still loses
  TemporalRandomFill at the matched cell.
- Do not spend more time on workload-footprint mixing unless paired with a
  different target or model. The failure is still target/model structure, not
  lack of workload variety.

Follow-up hygiene:

- Added `range_train_footprints` to row-level benchmark reports. The artifact
  already stored it in `run_config.json` and `example_run.json`, but omitting it
  from `benchmark_report.json` rows made run comparisons unnecessarily brittle.
- Added `historical_prior_k` to row-level benchmark reports for the same
  reason. It is a core model parameter for the KNN prior and should not require
  opening child configs during ablations.

### Historical-prior KNN neighbor-count diagnostic

Reason:

- The current best branch uses the nonparametric `historical_prior` KNN scorer
  with default `historical_prior_k=32`. Tested whether sharpening or smoothing
  the train-derived prior changes the grid enough to matter before adding a
  more invasive model family.

Focused `30%` coverage results:

| Artifact | `k` | Matched MLQDS | Uniform | DP | TemporalRandomFill | Wins vs uniform | Low-budget wins | Wins vs TRF | Mean audit delta vs uniform |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `multitrain0205_c30_historicalprior_shipbalanced_localswap085_k8_20260515` | `8` | `0.216053` | `0.211751` | `0.185568` | `0.216216` | `3/7` | `1/3` | `2/7` | `+0.000656` |
| `multitrain0205_c30_historicalprior_shipbalanced_localswap085_scorecache_20260515` | `32` | `0.218596` | `0.211751` | `0.185568` | `0.216216` | `4/7` | `1/3` | `4/7` | `+0.004039` |
| `multitrain0205_c30_historicalprior_shipbalanced_localswap085_k64_20260515` | `64` | `0.216484` | `0.211751` | `0.185568` | `0.216216` | `5/7` | `1/3` | `4/7` | `+0.004828` |

Full coverage sweep for `k=64`, same train/validation/eval days and
`local_swap` `0.85` selector:

| Coverage | Matched MLQDS | Uniform | DP | TemporalRandomFill | Wins vs uniform | Low-budget wins | Wins vs TRF | Mean audit delta vs uniform |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `5%` | `0.301589` | `0.300945` | `0.221774` | `0.270232` | `4/7` | `1/3` | `5/7` | `+0.002248` |
| `10%` | `0.272584` | `0.267976` | `0.203119` | `0.263826` | `5/7` | `1/3` | `4/7` | `+0.007721` |
| `15%` | `0.250934` | `0.244701` | `0.188116` | `0.242981` | `4/7` | `1/3` | `5/7` | `+0.003615` |
| `30%` | `0.216484` | `0.211751` | `0.185568` | `0.216216` | `5/7` | `1/3` | `4/7` | `+0.004828` |

Comparison to the `k=32` grid:

- `k=64` improves uniform win count from `17/28` to `18/28`.
- Low-budget wins do not move: still `4/12`, with `1%` and `2%` protected
  temporal ties rather than learned wins.
- Matched `5%` score drops at all four coverage targets.
- Mean audit delta averaged across coverage drops from about `+0.00650` to
  `+0.00460`.
- DP wins remain `28/28`, so the DP story is unchanged.
- Parallel execution made the `5%`, `10%`, and `15%` latency numbers noisy;
  the serial `30%` runs are the reliable latency comparison. There is no
  evidence that changing `k` solves runtime.

Conclusion:

- Sharpening (`k=8`) is worse. Smoothing (`k=64`) slightly improves the audit
  count but weakens matched scores and does not touch the low-budget failure.
- `historical_prior_k` tuning is not a credible path to acceptance. The
  remaining issue is still learned replacement confidence and transferable
  route usefulness, not the nearest-neighbor count.

### Train-source agreement diagnostic

Reason:

- The multi-day historical prior previously discarded train CSV source IDs
  before fitting the KNN scorer. That lets a single train day dominate a future
  point if it is the nearest support, even when other train days disagree.
- Added optional `historical_prior_source_aggregation` for `historical_prior`
  and `historical_prior_student`. `none` preserves pooled KNN behavior;
  `mean`, `min`, and `median` score each train source separately and then
  aggregate. The explicit train CSV loader now passes one source ID per train
  trajectory through split, training, checkpointing, and row-level reports.

Validation:

- Targeted tests:
  `../.venv/bin/python -m pytest -q tests/test_model_features.py tests/test_training_does_not_collapse.py::test_historical_prior_training_returns_fitted_prior_and_diagnostics tests/test_training_does_not_collapse.py::test_historical_prior_training_caps_support_per_trajectory tests/test_training_does_not_collapse.py::test_historical_prior_training_preserves_train_source_ids`
  -> `19 passed`.
- Config/report tests:
  `../.venv/bin/python -m pytest -q tests/test_torch_runtime_controls.py tests/test_benchmark_runner.py::test_benchmark_row_records_effective_child_torch_runtime`
  -> `15 passed`.

Focused `30%` coverage results on the current-best branch:

| Variant | Artifact | Matched MLQDS | Uniform | DP | TemporalRandomFill | Wins vs uniform | Low-budget wins | Wins vs TRF | Mean audit delta vs uniform | Train tau | Train matched target lift | First-mask latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Source mean | `multitrain0205_c30_historicalprior_shipbalanced_localswap085_sourcemean_20260515` | `0.213399` | `0.211751` | `0.185568` | `0.216216` | `5/7` | `1/3` | `2/7` | `+0.002458` | `+0.817` | `+0.0675` | `1027.38ms` |
| Source median | `multitrain0205_c30_historicalprior_shipbalanced_localswap085_sourcemedian_20260515` | `0.214525` | `0.211751` | `0.185568` | `0.216216` | `5/7` | `1/3` | `3/7` | `+0.002413` | `+0.506` | `+0.0089` | `985.21ms` |

Audit grid:

| Variant | 1% | 2% | 5% | 10% | 15% | 20% | 30% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Source mean MLQDS | `0.074838` | `0.133392` | `0.213399` | `0.317020` | `0.390123` | `0.457798` | `0.578791` |
| Source mean uniform | `0.074838` | `0.133392` | `0.211751` | `0.315642` | `0.387494` | `0.456034` | `0.569004` |
| Source median MLQDS | `0.074838` | `0.133392` | `0.214525` | `0.316176` | `0.388191` | `0.462506` | `0.575417` |
| Source median uniform | `0.074838` | `0.133392` | `0.211751` | `0.315642` | `0.387494` | `0.456034` | `0.569004` |

Conclusion:

- Source agreement is a useful diagnostic knob, but it is not a quality path
  here. It lowers matched score versus pooled `k=32` (`0.218596`) and does not
  move the `1%`/`2%` temporal ties.
- The source-mean fit still looks good on the train target, but eval quality
  gets worse. That is another concrete warning that train target-recall lift is
  not a reliable selector.
- Do not sweep source aggregation across coverage targets unless a different
  model uses the source signal more directly. The nonparametric historical
  prior mostly needs a better notion of replacement confidence, not stricter
  day agreement.

### Low-budget swap-count diagnostic

Reason:

- `local_swap` with `mlqds_temporal_fraction=0.85` leaves no learned swaps at
  the `1%` and `2%` audit cells after per-trajectory rounding. For a 256-point
  segment, `1%` keeps `3` points and protects all `3`; `2%` keeps `6` and
  protects all `6`.
- Ran the same current-best branch with `mlqds_hybrid_mode=local_swap` and
  `mlqds_temporal_fraction=0.66` to allow learned replacements at low budgets
  without the `local_delta_swap` positive-delta gate.

Result:

- Artifact:
  `artifacts/manual/multitrain0205_c30_historicalprior_shipbalanced_localswap066_20260515`.
- Matched `RangeUseful`: MLQDS `0.208252`, uniform `0.211751`, DP
  `0.185568`, TemporalRandomFill `0.209914`.
- Audit wins: uniform `1/7`, low-budget uniform `0/3`, DP `7/7`,
  TemporalRandomFill `4/7`.
- Train fit looked excellent but was misleading: `tau=+0.989`, matched
  target-recall lift `+0.2532`, low-budget target-recall lift `+0.2123`.
- Geometry collapses versus the `0.85` branch: `AvgSED=1.0303 km` vs uniform
  `0.5391 km`; length preservation `0.9211` vs uniform `0.9276`.

Audit grid:

| Compression | MLQDS | Uniform | DP | TemporalRandomFill |
| ---: | ---: | ---: | ---: | ---: |
| `1%` | `0.074488` | `0.074838` | `0.071081` | `0.069970` |
| `2%` | `0.132600` | `0.133392` | `0.119772` | `0.133861` |
| `5%` | `0.208252` | `0.211751` | `0.185568` | `0.209914` |
| `10%` | `0.312730` | `0.315642` | `0.273035` | `0.307124` |
| `15%` | `0.384650` | `0.387494` | `0.343327` | `0.383572` |
| `20%` | `0.448831` | `0.456034` | `0.414393` | `0.457188` |
| `30%` | `0.585130` | `0.569004` | `0.517197` | `0.561629` |

Conclusion:

- The low-budget tie is mechanically caused by the `0.85` scaffold, but simply
  allowing learned swaps exposes a worse problem: the learned score chooses
  replacements that hurt held-out `RangeUseful` and geometry.
- This invalidates "just lower the scaffold" for the current historical prior.
  The model needs a better replacement-confidence signal before the selector
  should spend low-budget points on learned swaps.

### Local-window replacement diagnostic

Reason:

- The `local_swap=0.66` run could have failed because global top-score
  candidates were stealing points from distant temporal anchors. I tested that
  explanation with a temporary local-window selector: each removable temporal
  anchor could move only to a better-scored non-base point inside its own
  neighbouring-anchor window.
- The selector was useful as a diagnostic but was removed from the public CLI
  after benchmarking. Keeping a failed one-off selector would make the research
  system harder to trust.

Result:

- Artifact:
  `artifacts/manual/multitrain0205_c30_historicalprior_shipbalanced_localwindow066_20260515`.
- Matched `RangeUseful`: MLQDS `0.203202`, uniform `0.211751`, DP
  `0.185568`, TemporalRandomFill `0.209507`.
- Audit wins: uniform `1/7`, low-budget uniform `0/3`, DP `7/7`,
  TemporalRandomFill `2/7`.
- Train fit again looked strong but did not transfer: `tau=+0.9909`, matched
  target-recall lift `+0.1462`, low-budget lift `+0.1538`.
- Geometry is much less broken than global `local_swap=0.66`, but still not
  better than uniform: `AvgSED=0.5910 km` vs uniform `0.5391 km`; length
  preservation `0.9337` vs uniform `0.9276`.

Audit grid:

| Compression | MLQDS | Uniform | DP | TemporalRandomFill |
| ---: | ---: | ---: | ---: | ---: |
| `1%` | `0.073718` | `0.074838` | `0.071081` | `0.078420` |
| `2%` | `0.129159` | `0.133392` | `0.119772` | `0.130475` |
| `5%` | `0.203202` | `0.211751` | `0.185568` | `0.209507` |
| `10%` | `0.314111` | `0.315642` | `0.273035` | `0.324478` |
| `15%` | `0.388943` | `0.387494` | `0.343327` | `0.386017` |
| `20%` | `0.455407` | `0.456034` | `0.414393` | `0.454302` |
| `30%` | `0.552531` | `0.569004` | `0.517197` | `0.572436` |

Conclusion:

- Global swap overreach is not the root issue. Localizing replacements mostly
  contains geometry damage, but it makes `RangeUseful` worse.
- The current historical-prior score is not a reliable replacement-confidence
  signal, even when replacement scope is constrained locally. More selector
  variants are unlikely to fix this without changing the learned signal.

### Local-delta gain/cost target diagnostic

Reason:

- `local_swap_utility_frequency` labels useful additions, but it does not label
  the cost of removing the temporal anchor that `local_delta_swap` uses as the
  paired comparison point. That is misaligned: the selector accepts a candidate
  only when candidate score exceeds the paired base score.
- Added `range_training_target_mode="local_swap_gain_cost_frequency"` for this
  diagnostic. For each train query and budget, candidate value is
  `score(base - anchor + candidate) - score(base - anchor)` and anchor cost is
  `score(base) - score(base - anchor)`. Thus candidate value beats anchor cost
  exactly when the one-step replacement improves train-query `RangeUseful`.
  Eval compression remains workload-blind.

Result:

- Artifact:
  `artifacts/manual/multitrain0205_c30_historicalprior_localdelta066_gaincost_gain32_20260515`.
- Matched `RangeUseful`: MLQDS `0.196546`, uniform `0.211751`, DP
  `0.185568`, TemporalRandomFill `0.211599`.
- Audit wins: uniform `0/7`, low-budget uniform `0/3`, DP `6/7`,
  TemporalRandomFill `1/7`.
- Target construction is slower than the earlier local-swap utility target:
  `216.53s` for one train workload at candidate limit `32`.
- Target diagnostics: `6,420` positive labels (`8.78%`), target mass
  `0.31596`, `34,863` scored candidates, `22,056` positive-net-gain
  candidates, `8,537` selected pairs, selected candidate-value mass
  `346.95`, selected removal-cost mass `90.97`.
- Train fit again looked excellent and again failed to transfer:
  `tau=+0.977`, matched target-recall lift `+0.1339`, low-budget lift
  `+0.2380`.
- Geometry worsened: `AvgSED=0.6966 km` vs uniform `0.5391 km`; length
  preservation `0.9298` vs uniform `0.9276`.

Audit grid:

| Compression | MLQDS | Uniform | DP | TemporalRandomFill |
| ---: | ---: | ---: | ---: | ---: |
| `1%` | `0.071902` | `0.074838` | `0.071081` | `0.073021` |
| `2%` | `0.119015` | `0.133392` | `0.119772` | `0.128149` |
| `5%` | `0.196546` | `0.211751` | `0.185568` | `0.211599` |
| `10%` | `0.302984` | `0.315642` | `0.273035` | `0.326294` |
| `15%` | `0.371208` | `0.387494` | `0.343327` | `0.383078` |
| `20%` | `0.443963` | `0.456034` | `0.414393` | `0.443481` |
| `30%` | `0.565274` | `0.569004` | `0.517197` | `0.566597` |

Conclusion:

- Pairwise gain/cost supervision is logically better aligned with
  `local_delta_swap`, but it is too train-query-specific for the current
  historical KNN prior. It overfits target recall and loses every uniform
  audit cell.
- This makes the replacement-confidence issue sharper: the problem is not only
  missing removal-cost labels. The nonparametric prior is learning local
  train-query marginal events that do not transfer to held-out day queries.

### Historical-prior flat inference fast path

Reason:

- Runtime was still a material iteration bottleneck for the current-best
  `historical_prior` branch.
- The standalone historical prior is pointwise: each score depends only on the
  point's query-free feature vector and stored train support. It was still
  being evaluated through overlapping trajectory windows, which duplicated
  identical KNN scoring work before averaging.
- Added `window_independent=True` to `HistoricalPriorRangeQDSModel` and a
  `windowed_predict` fast path that scores the flat point tensor once for such
  workload-blind pointwise models. This does not apply to trainable students
  because transformer context can depend on the window.

Validation:

- Focused tests:
  `../.venv/bin/python -m pytest -q tests/test_model_features.py::test_historical_prior_windowed_predict_uses_pointwise_fast_path tests/test_model_features.py::test_historical_prior_model_scores_and_round_trips_checkpoint`
  -> `2 passed`.
- Broader target suite:
  `../.venv/bin/python -m pytest -q tests/test_teacher_distillation.py tests/test_torch_runtime_controls.py tests/test_benchmark_runner.py`
  -> `66 passed, 1 warning`.
- Full suite:
  `../.venv/bin/python -m pytest -q tests`
  -> `260 passed, 1 warning`.
- `git diff --check` passed.

Benchmark:

- Artifact:
  `artifacts/manual/multitrain0205_c30_historicalprior_shipbalanced_localswap085_flatpredict_20260515`.
- Same current-best `30%` branch as the score-cache run:
  `historical_prior`, ship-balanced retained-frequency, 4 train workloads,
  `local_swap`, `mlqds_temporal_fraction=0.85`, `k=32`.
- Quality is unchanged at the audited precision:
  MLQDS `0.218596`, uniform `0.211751`, DP `0.185568`,
  TemporalRandomFill `0.216216`.
- Audit wins remain unchanged: uniform `4/7`, low-budget uniform `1/3`,
  DP `7/7`, TemporalRandomFill `4/7`.
- Matched MLQDS latency improved from about `865.70ms` to `393.42ms`.
- Historical-prior train-fit diagnostics improved from about `23s` to `7.52s`.
- Pipeline runtime with cache hits was `12.61s`; runtime bottleneck moved back
  to train/model diagnostics (`7.57s`) instead of retained-mask freeze.

Conclusion:

- This is a clean exact runtime improvement, not a quality improvement.
- The final acceptance failure remains unchanged: low-budget `1%` and `2%`
  cells are still temporal ties, uniform wins are only `17/28`, and the dense
  held-out generator setting remains weak.

### Current-best raw-score ablation after flat inference

Reason:

- The standalone `historical_prior` outputs target-like values directly. The
  default `rank` score mode discards score scale before `local_swap`, so I
  retested the current-best `0.85` scaffold with `mlqds_score_mode=raw`.

Result:

- Artifact:
  `artifacts/manual/multitrain0205_c30_historicalprior_shipbalanced_localswap085_rawscore_20260515`.
- Same current-best `30%` branch as the flat-predict run:
  `historical_prior`, ship-balanced retained-frequency, 4 train workloads,
  `local_swap`, `mlqds_temporal_fraction=0.85`, `k=32`.
- Matched `RangeUseful` is unchanged at audited precision: MLQDS `0.218596`,
  uniform `0.211751`, DP `0.185568`, TemporalRandomFill `0.216216`.
- Audit wins are unchanged: uniform `4/7`, low-budget uniform `1/3`, DP
  `7/7`, TemporalRandomFill `4/7`.
- Train-fit diagnostics again look strong but do not explain held-out gains:
  `tau=+0.998`, matched target-recall lift `+0.0706`, low-budget lift
  `+0.0235`.
- First-mask latency was `412.88ms`, within noise of the flat-predict
  current-best run.

Audit grid:

| Compression | MLQDS | Uniform | DP | TemporalRandomFill |
| ---: | ---: | ---: | ---: | ---: |
| `1%` | `0.074838` | `0.074838` | `0.071081` | `0.074838` |
| `2%` | `0.133392` | `0.133392` | `0.119772` | `0.133392` |
| `5%` | `0.218596` | `0.211751` | `0.185568` | `0.216216` |
| `10%` | `0.315237` | `0.315642` | `0.273035` | `0.317771` |
| `15%` | `0.393674` | `0.387494` | `0.343327` | `0.382734` |
| `20%` | `0.463240` | `0.456034` | `0.414393` | `0.458854` |
| `30%` | `0.577434` | `0.569004` | `0.517197` | `0.563904` |

Conclusion:

- Raw historical-prior score scale does not add a useful replacement signal on
  the current-best `0.85` scaffold. It mostly reproduces the same masks.
- Do not sweep this across coverage targets. The low-budget cells are still
  temporal ties, and the `10%` cell still loses uniform.

### Historical-prior chunk-size runtime check

Reason:

- After the flat pointwise inference path, the historical-prior KNN scorer still
  chunks query points. The current-best run used `query_chunk_size=512`; I
  tested `2048` as a cheap runtime-only ablation.

Result:

- Artifact:
  `artifacts/manual/multitrain0205_c30_historicalprior_shipbalanced_localswap085_chunk2048_20260515`.
- Quality is identical to current best at audited precision:
  MLQDS `0.218596`, uniform `0.211751`, DP `0.185568`.
- First-mask latency did not improve: `424.09ms` versus about `393-413ms` on
  the flat-predict/raw-score runs.
- Train-fit diagnostics were slightly faster (`6.54s`), but total runtime was
  essentially unchanged (`11.56s` with cache hits).

Conclusion:

- Larger KNN chunks do not solve the remaining runtime bottleneck. Keep
  `query_chunk_size=512` for the current profile unless a later backend changes
  the memory/runtime trade-off.

### Low-budget retained-frequency target under lower scaffold

Reason:

- The current-best `local_swap=0.85` scaffold cannot learn at the `1%` cell:
  a 256-point trajectory keeps `3` points and protects all `3`.
- A lower `local_swap=0.66` scaffold allows learned replacements at `1%` and
  `2%`, but the all-budget target failed. I tested whether restricting the
  retained-frequency target to `1%,2%,5%` budgets improves the explicit
  low-budget acceptance cells.

Result:

- Artifact:
  `artifacts/manual/multitrain0205_c30_historicalprior_shipbalanced_localswap066_lowbudgets_20260515`.
- Same `30%` current branch ingredients: `historical_prior`,
  ship-balanced labels, 4 train workloads, `label_mean`, but
  `mlqds_temporal_fraction=0.66` and `budget_loss_ratios=0.01,0.02,0.05`.
- Target became much sharper: `2,995` positives (`4.10%`) and target mass
  `1403.00`, versus `21,603` positives and mass `5034.29` for the all-budget
  current-best target.
- Train target fit still looked strong: `tau=+1.000`, matched target-recall
  lift `+0.1689`, low-budget lift `+0.1546`.
- Held-out quality worsened: matched `RangeUseful=0.204214`, uniform
  `0.211751`, DP `0.185568`, TemporalRandomFill `0.209914`.
- Audit wins: uniform `1/7`, low-budget uniform `1/3`, DP `7/7`,
  TemporalRandomFill `2/7`.
- Geometry also worsened: `AvgSED=0.8502 km` vs uniform `0.5391 km`;
  length preservation `0.9258` vs uniform `0.9276`.

Audit grid:

| Compression | MLQDS | Uniform | DP | TemporalRandomFill |
| ---: | ---: | ---: | ---: | ---: |
| `1%` | `0.078447` | `0.074838` | `0.071081` | `0.069970` |
| `2%` | `0.130946` | `0.133392` | `0.119772` | `0.133861` |
| `5%` | `0.204214` | `0.211751` | `0.185568` | `0.209914` |
| `10%` | `0.308068` | `0.315642` | `0.273035` | `0.307124` |
| `15%` | `0.368775` | `0.387494` | `0.343327` | `0.383572` |
| `20%` | `0.438398` | `0.456034` | `0.414393` | `0.457188` |
| `30%` | `0.552500` | `0.569004` | `0.517197` | `0.561629` |

Conclusion:

- Low-budget target sharpening can force a learned `1%` win, but it damages
  `2%`, `5%`, and the broader grid. The failure components are the same:
  weaker `ShipF1`, `GapCov`, `TemporalCov`, and `ShapeScore`.
- Do not pursue more retained-frequency budget weighting for the current
  historical-prior/local-swap branch. It is optimizing train target recall while
  making held-out retained sets less useful.

### Low-budget target with local-delta gate

Reason:

- The low-budget retained-frequency target plus `local_swap=0.66` proved that
  the sparse target can find some useful `1%` replacements, but unrestricted
  swaps hurt later cells.
- Tested the same low-budget target with `local_delta_swap=0.66` to accept only
  learned replacements whose score beats the paired local temporal anchor.

Result:

- Artifact:
  `artifacts/manual/multitrain0205_c30_historicalprior_shipbalanced_localdelta066_lowbudgets_20260515`.
- Target stayed sparse: `2,957` positives (`4.04%`) and target mass
  `1586.67`.
- Train fit again looked excellent and again lied: `tau=+1.000`, matched
  target-recall lift `+0.1998`, low-budget lift `+0.2270`.
- Held-out quality was worse than both the low-budget `local_swap` run and the
  all-budget lower-scaffold run: matched `RangeUseful=0.195771`, uniform
  `0.211751`, DP `0.185568`, TemporalRandomFill `0.211599`.
- Audit wins: uniform `0/7`, low-budget uniform `0/3`, DP `6/7`,
  TemporalRandomFill `0/7`.
- Runtime also regressed: pipeline `40.88s`; train/model diagnostics
  `24.01s`; audit mask freeze `4.11s`; first-mask latency `648.12ms`.

Audit grid:

| Compression | MLQDS | Uniform | DP | TemporalRandomFill |
| ---: | ---: | ---: | ---: | ---: |
| `1%` | `0.068358` | `0.074838` | `0.071081` | `0.073021` |
| `2%` | `0.125258` | `0.133392` | `0.119772` | `0.128149` |
| `5%` | `0.195771` | `0.211751` | `0.185568` | `0.211599` |
| `10%` | `0.310714` | `0.315642` | `0.273035` | `0.326294` |
| `15%` | `0.365392` | `0.387494` | `0.343327` | `0.383078` |
| `20%` | `0.443207` | `0.456034` | `0.414393` | `0.443481` |
| `30%` | `0.541010` | `0.569004` | `0.517197` | `0.566597` |

Conclusion:

- The local-delta gate does not rescue low-budget retained-frequency
  sharpening. It removes the only useful low-budget win and still damages the
  full grid.
- This closes the current retained-frequency/scaffold branch. The failure is
  not budget emphasis, raw score scale, or a missing positive-delta gate.

### Historical-prior MMSI identity diagnostic

Reason:

- MMSI is known before compression and is independent of future eval queries.
  Train/eval overlap is nontrivial on this AIS slice (`94/120` eval segments
  have an MMSI seen in the four train days), so I tested whether a legal
  workload-blind vessel-identity prior improves the current historical KNN
  branch.
- Implemented a separate `model_type="historical_prior_mmsi"` rather than
  mutating the current best. It appends a deterministic 4-dim MMSI hash between
  route-context and clock/density historical-prior features. Eval masks still
  freeze before query scoring.

Implementation:

- Added MMSI-aware query-free feature construction, CLI/config/report/checkpoint
  plumbing, and `MLQDSMethod`/inference support for passing trajectory MMSIs
  before compression.
- Added `historical_prior_mmsi_weight` to scale only the MMSI hash slice in the
  KNN distance.
- Targeted tests pass:
  `../.venv/bin/python -m pytest -q tests/test_model_features.py tests/test_torch_runtime_controls.py tests/test_benchmark_runner.py`
  -> `61 passed`.

Benchmarks:

| Artifact | MMSI weight | MLQDS | Uniform | DP | TemporalRandomFill | Wins vs uniform | Low-budget wins | Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `multitrain0205_c30_historicalmmsi_shipbalanced_localswap085_w2_20260515` | `2.0` | `0.212172` | `0.211751` | `0.185568` | `0.216216` | `3/7` | `1/3` | `447.19ms` |
| `multitrain0205_c30_historicalmmsi_shipbalanced_localswap085_w05_20260515` | `0.5` | `0.213956` | `0.211751` | `0.185568` | `0.216216` | `4/7` | `1/3` | `395.16ms` |

Best MMSI audit grid (`weight=0.5`):

| Compression | MLQDS | Uniform | DP | TemporalRandomFill |
| ---: | ---: | ---: | ---: | ---: |
| `1%` | `0.074838` | `0.074838` | `0.071081` | `0.074838` |
| `2%` | `0.133392` | `0.133392` | `0.119772` | `0.133392` |
| `5%` | `0.213956` | `0.211751` | `0.185568` | `0.216216` |
| `10%` | `0.319555` | `0.315642` | `0.273035` | `0.317771` |
| `15%` | `0.390799` | `0.387494` | `0.343327` | `0.382734` |
| `20%` | `0.460288` | `0.456034` | `0.414393` | `0.458854` |
| `30%` | `0.566479` | `0.569004` | `0.517197` | `0.563904` |

Diagnosis:

- The MMSI signal does not improve the target-recall diagnostic. It leaves
  train fit essentially unchanged: `tau=+0.998`, matched target-recall lift
  `+0.0709`, low-budget lift `+0.0236`.
- It does not create learned low-budget wins. The `1%` and `2%` cells are still
  temporal ties, and the `5%` cell still loses TemporalRandomFill.
- Strong identity weighting (`2.0`) overfits and drops held-out usefulness.
  Weak identity weighting (`0.5`) recovers the old win count but remains below
  the current best `historical_prior` matched score (`0.218596`).
- Do not sweep MMSI weighting across coverage targets. The branch is legal and
  clean, but it does not address the core failure: the current blind score
  cannot confidently choose useful low-budget replacements.

### Minimum learned-swap diagnostic

Reason:

- The current best `local_swap=0.85` branch cannot make learned replacements at
  `1%` and `2%` because temporal-fraction rounding consumes the whole retained
  budget. This made the low-budget cells temporal ties.
- Added `mlqds_min_learned_swaps` as an explicit diagnostic selector knob.
  Default `0` preserves existing behavior. Values `1` or `2` force at least
  that many learned replacements per trajectory when swap capacity exists, so
  the run tests score quality rather than the rounding artifact.

Implementation:

- Added `min_learned_swaps` to `simplify_with_temporal_score_hybrid` and wired
  it through MLQDS scoring, `MLQDSMethod`, `ScoreHybridMethod`, validation,
  target construction, teacher-distillation frequency targets, config, CLI, and
  benchmark reporting.
- Added selector/config/report tests. Focused tests pass:
  `../.venv/bin/python -m pytest -q tests/test_metrics.py tests/test_torch_runtime_controls.py tests/test_benchmark_runner.py`
  -> `89 passed`.

Benchmarks:

| Artifact | Min swaps | MLQDS | Uniform | DP | TemporalRandomFill | Wins vs uniform | Low-budget wins vs uniform | Low-budget wins vs TRF | AvgSED |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `multitrain0205_c30_historicalprior_shipbalanced_localswap085_minswap1_20260515` | `1` | `0.216122` | `0.211751` | `0.185568` | `0.216053` | `5/7` | `2/3` | `2/3` | `0.5715` |
| `multitrain0205_c30_historicalprior_shipbalanced_localswap085_minswap2_20260515` | `2` | `0.213524` | `0.211751` | `0.185568` | `0.221114` | `5/7` | `2/3` | `1/3` | `0.6814` |

Audit grid:

| Compression | Min1 MLQDS | Min1 vs uniform | Min1 vs TRF | Min2 MLQDS | Min2 vs uniform | Min2 vs TRF | Oracle fill |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1%` | `0.076520` | `+0.001682` | `+0.006550` | `0.075635` | `+0.000797` | `+0.005665` | `0.142866` |
| `2%` | `0.130115` | `-0.003277` | `-0.001042` | `0.128180` | `-0.005212` | `-0.005898` | `0.309222` |
| `5%` | `0.216122` | `+0.004371` | `+0.000069` | `0.213524` | `+0.001773` | `-0.007590` | `0.386070` |
| `10%` | `0.311576` | `-0.004066` | `-0.006195` | `0.315267` | `-0.000375` | `-0.002499` | `0.491658` |
| `15%` | `0.393280` | `+0.005785` | `+0.010609` | `0.391369` | `+0.003874` | `+0.008698` | `0.599228` |
| `20%` | `0.457513` | `+0.001478` | `-0.001634` | `0.456781` | `+0.000747` | `-0.001796` | `0.664478` |
| `30%` | `0.577090` | `+0.008086` | `+0.013370` | `0.575304` | `+0.006300` | `+0.011619` | `0.717490` |

Diagnosis:

- `minswap1` proves the old `1%` tie was partly a selector-rounding artifact:
  one learned replacement creates a real `1%` win.
- It does not solve the low-budget acceptance problem. The `2%` cell still
  loses uniform and TemporalRandomFill, and matched `5%` remains below the
  current best `0.218596`.
- `minswap2` is worse despite stronger train-target fit: matched target-recall
  lift rises to `+0.1451` and low-budget lift to `+0.1762`, but held-out
  `RangeUseful`, TRF wins, and geometry degrade. This is another case where
  train target capture is not reliable evidence of held-out query usefulness.
- Oracle fill with the same min-swap selector has large headroom at every low
  budget (`2%` oracle fill `0.309222` vs uniform `0.133392`), so the selector
  can express useful replacements. The blind historical-prior score is the
  weak link.
- Do not use `mlqds_min_learned_swaps` as a final-success mechanism around the
  current KNN score. It is useful for diagnosis, but it is still a selector
  scaffold and does not produce a broad, robust learned win.

### Trainable `range_prior` under min-swap exposure

Reason:

- The min-swap diagnostic showed the selector can express useful low-budget
  replacements if the score is good enough, and oracle fill has large headroom.
- Tested whether the trainable workload-blind `range_prior` model can use the
  exposed learned-swap slots better than the nonparametric historical KNN.

Result:

- Artifact:
  `artifacts/manual/multitrain0205_c30_rangeprior_shipbalanced_localswap085_minswap1_20260515`.
- Same train/eval days, ship-balanced retained-frequency target, 4 train
  workload replicates, `label_mean`, `local_swap`, `mlqds_temporal_fraction=0.85`,
  `mlqds_min_learned_swaps=1`.
- Matched held-out `RangeUseful=0.211065`, below uniform `0.211751`, DP
  `0.185568`, and TemporalRandomFill `0.216053`.
- Audit wins: uniform `1/7`, low-budget uniform `0/3`, DP `7/7`,
  TemporalRandomFill `4/7`.
- Runtime is much better than KNN (`93.19ms` latency vs KNN minswap1
  `443.43ms`), but quality is not acceptable.
- Train target fit is weak: `tau=+0.063`, matched target-recall lift `+0.0074`,
  low-budget target-recall delta `-0.0131`.

Audit grid:

| Compression | `range_prior` MLQDS | vs uniform | vs TRF | KNN minswap1 MLQDS |
| ---: | ---: | ---: | ---: | ---: |
| `1%` | `0.073464` | `-0.001374` | `+0.003494` | `0.076520` |
| `2%` | `0.131555` | `-0.001837` | `+0.000398` | `0.130115` |
| `5%` | `0.211065` | `-0.000686` | `-0.004988` | `0.216122` |
| `10%` | `0.314656` | `-0.000986` | `-0.003115` | `0.311576` |
| `15%` | `0.391632` | `+0.004138` | `+0.008961` | `0.393280` |
| `20%` | `0.452372` | `-0.003662` | `-0.006775` | `0.457513` |
| `30%` | `0.565284` | `-0.003720` | `+0.001565` | `0.577090` |

Conclusion:

- Current trainable `range_prior` is not the missing structural model. It does
  not fit the exposed min-swap target well enough, especially at low budgets.
- The bottleneck splits cleanly now:
  - historical KNN fits train labels but does not transfer well enough;
  - `range_prior` transfers cheaply but cannot fit the target.
- Next credible work should be model-capacity/objective work on a trainable
  blind scorer, not more selector rounding or KNN context knobs.

### Historical-prior student under min-swap exposure

Reason:

- The plain trainable `range_prior` cannot fit the min-swap target, while the
  nonparametric historical prior fits train labels but transfers weakly.
- Tested the existing `historical_prior_student` path to see whether a
  trainable blind scorer can use the historical-prior signal under the same
  exposed learned-swap selector.

Result:

- Artifact:
  `artifacts/manual/multitrain0205_c30_historicalpriorstudent_shipbalanced_localswap085_minswap1_20260515`.
- Same train/eval days, ship-balanced retained-frequency target, 4 train
  workload replicates, `label_mean`, `local_swap`, `mlqds_temporal_fraction=0.85`,
  `mlqds_min_learned_swaps=1`.
- Matched held-out `RangeUseful=0.215283`, above uniform `0.211751` and DP
  `0.185568`, but below TemporalRandomFill `0.216053` and below the
  current-best KNN branch `0.218596`.
- Audit wins: uniform `5/7`, low-budget uniform `2/3`, DP `7/7`,
  TemporalRandomFill `5/7`, low-budget TemporalRandomFill `2/3`.
- Train target fit is much better than `range_prior`: `tau=+0.791`, matched
  target-recall lift `+0.0560`, low-budget target-recall lift `+0.0895`.
- Runtime is worse than both trainable `range_prior` and current-best flat KNN:
  matched latency `751.72ms`.
- Geometry is still worse than uniform: AvgSED `0.5577 km` vs uniform
  `0.5391 km`.

Audit grid:

| Compression | Student MLQDS | vs uniform | vs TRF | TemporalOracleFill |
| ---: | ---: | ---: | ---: | ---: |
| `1%` | `0.080487` | `+0.005649` | `+0.010518` | `0.142866` |
| `2%` | `0.131752` | `-0.001640` | `+0.000595` | `0.240030` |
| `5%` | `0.215283` | `+0.003532` | `-0.000770` | `0.316119` |
| `10%` | `0.315560` | `-0.000082` | `-0.002211` | `0.491677` |
| `15%` | `0.388504` | `+0.001010` | `+0.005833` | `0.598372` |
| `20%` | `0.462266` | `+0.006232` | `+0.003119` | `0.663274` |
| `30%` | `0.583286` | `+0.014282` | `+0.019567` | `0.717158` |

Conclusion:

- The student can fit the target signal better than `range_prior`, but the
  held-out win is too small and too scaffold-dependent.
- It does not beat the current-best matched score and it does not clear the
  `2%` uniform cell.
- Because latency regresses and quality remains weak, this is not the final
  model path. It is evidence that useful target signal exists, but the current
  student architecture/objective still does not turn it into robust blind
  compression behavior.

### `range_prior_clock_density` under min-swap exposure

Reason:

- Plain `range_prior` failed target fit under min-swap exposure.
- Tested whether adding clock-time and local spatial-density features gives the
  trainable blind scorer enough context to use the exposed learned slots.

Result:

- Artifact:
  `artifacts/manual/multitrain0205_c30_rangepriorclockdens_shipbalanced_localswap085_minswap1_20260515`.
- Same train/eval days, ship-balanced retained-frequency target, 4 train
  workload replicates, `label_mean`, `local_swap`, `mlqds_temporal_fraction=0.85`,
  `mlqds_min_learned_swaps=1`.
- Matched held-out `RangeUseful=0.211236`, below uniform `0.211751`, DP
  `0.185568`, and TemporalRandomFill `0.216053`.
- Audit wins: uniform `0/7`, low-budget uniform `0/3`, DP `6/7`,
  TemporalRandomFill `1/7`.
- Train target fit is essentially absent: `tau=+0.013`, matched target-recall
  delta `-0.0150`, low-budget target-recall delta `-0.0304`.
- Latency is acceptable for a neural model (`104.81ms`), but quality is not.

Audit grid:

| Compression | Clock+density MLQDS | vs uniform | vs TRF | TemporalOracleFill |
| ---: | ---: | ---: | ---: | ---: |
| `1%` | `0.067134` | `-0.007703` | `-0.002835` | `0.142866` |
| `2%` | `0.132287` | `-0.001105` | `+0.001129` | `0.240030` |
| `5%` | `0.211236` | `-0.000515` | `-0.004817` | `0.316119` |
| `10%` | `0.312198` | `-0.003444` | `-0.005573` | `0.491677` |
| `15%` | `0.382625` | `-0.004869` | `-0.000046` | `0.598372` |
| `20%` | `0.450223` | `-0.005812` | `-0.008925` | `0.663274` |
| `30%` | `0.557668` | `-0.011336` | `-0.006052` | `0.717158` |

Conclusion:

- Missing clock/density features are not the reason the neural blind scorer
  fails. The stronger point context still does not fit the target.
- This reinforces the split: cheap neural scorers are fast but fail the target;
  KNN-like historical context captures target structure but transfers too
  weakly and costs too much.

### MLP pointwise historical-prior student

Reason:

- The transformer `historical_prior_student` fit the target better than
  `range_prior`, but it was slow and still weaker than the standalone KNN
  branch.
- Tested whether an MLP-only student with direct pointwise BCE can preserve the
  useful KNN prior ranking without transformer overhead.

Result:

- Artifact:
  `artifacts/manual/multitrain0205_c30_historicalpriorstudent_mlp_pointwise_shipbalanced_localswap085_minswap1_20260515`.
- Same train/eval days, ship-balanced retained-frequency target, 4 train
  workload replicates, `label_mean`, `local_swap`, `mlqds_temporal_fraction=0.85`,
  `mlqds_min_learned_swaps=1`, but `num_layers=0`, `dropout=0.0`,
  `loss_objective=pointwise_bce`.
- Matched held-out `RangeUseful=0.210241`, below uniform `0.211751`, DP
  `0.185568`, and TemporalRandomFill `0.216053`.
- Audit wins: uniform `1/7`, low-budget uniform `0/3`, DP `7/7`,
  TemporalRandomFill `3/7`.
- Train target fit is respectable (`tau=+0.745`, matched target-recall lift
  `+0.0543`, low-budget lift `+0.0894`), but it does not transfer.
- Latency remains bad (`706.60ms`) because KNN prior scoring still dominates.

Conclusion:

- The student wrapper is not rescuing the KNN prior. Better pointwise target fit
  is not enough, and removing the transformer makes held-out quality worse.
- This closes the obvious "student is overcomplicated" explanation.

### MLSimp paper methodology review

Source:

- `Sprint/MLSimp-Paper.pdf`, PVLDB 2024, "Quantifying Point Contributions:
  A Lightweight Framework for Efficient and Effective Query-Driven Trajectory
  Simplification."

Useful ideas for Range-QDS:

- MLSimp's central split is the right one for our failure: learn a
  query-free structural importance model, then use query/workload information
  only as training or prior signal. Their final lightweight model predicts
  point importance once and samples by score.
- Their "globality" and "uniqueness" definitions are more structural than our
  scalar retained-frequency targets. Globality measures how representative a
  point is of the full trajectory; uniqueness measures how different it is from
  semantically similar neighbors. This maps directly to our low-budget failure:
  current labels fit train queries but do not learn a transferable notion of
  critical point vs redundant point.
- Their trajectory graph is a practical architecture hint: segment long
  trajectories, add segment-summary nodes, and let point nodes attend to
  trajectory-level context. Our current blind neural scorer is mostly windowed
  point context and has repeatedly failed target fit.
- Their mutual-learning stage uses a heavy generative model only during
  training, then deploys the lightweight scorer. For us, the analogous teacher
  should be a training-only low-budget oracle/local-swap teacher that produces
  amplified labels for missed `1%`/`2%`/`5%` points. The deployed model should
  remain query-free.
- Their ablation says query adjustment helps, but too many generated queries
  add noise. That supports our own failed workload-replicate/footprint sweeps:
  just adding more generated range workloads is not the path.
- Their global simplification case study is directly relevant. They report
  that database-global budget allocation beats simplifying each trajectory
  independently because it adapts retention rates per trajectory. Our selector
  currently allocates the same compression fraction per trajectory, which may
  waste points on low-utility trajectories and starve high-utility ones.

Ideas to reject or constrain:

- MLSimp's query-based importance adjustment at simplification time is not
  acceptable for our final claim if it is driven by eval/test query workloads.
  At most, adapt it as a train-derived workload prior or diagnostic upper bound.
- A diffusion model is probably overkill as the next implementation. The useful
  takeaway is "training-only amplified low-budget labels," not diffusion itself.
- Their reported query-driven wins rely partly on synthetic query adjustment.
  We need a stricter workload-blind protocol, so copying their inference path
  would weaken the claim.

Concrete next implementation direction:

1. Add a lightweight graph/segment blind scorer: point encoder plus
   segment-summary tokens, with point-to-segment attention or GAT-lite message
   passing.
2. Add a self-supervised structural regularizer inspired by globality and
   uniqueness, kept alongside the retained-frequency/RangeUseful supervision.
3. Add a training-only low-budget amplification target from TemporalOracleFill
   or local-swap oracle labels, then train the graph scorer to absorb that
   signal without passing eval queries into compression.
4. Add a diagnostic global-budget selector with endpoint/minimum-retention
   safeguards and evaluate whether database-level allocation improves
   `RangeUseful` without destroying `ShipCov`, length preservation, or visual
   sensibility.

### MLSimp-inspired segment-context scorer check

Implementation:

- Added `model_type="segment_context_range"`, a workload-blind neural scorer
  using the 28-column `range_prior_clock_density` features, fixed
  trajectory-order segment summaries, point-to-segment attention, local
  uniqueness, and trajectory globality scalars.
- Wired it through training, checkpoint loading, benchmark CLI/profile
  metadata, and documentation.
- Added regression coverage proving query tensors do not affect its forward
  output and that checkpoints round-trip.

Validation:

- Focused tests:
  `../.venv/bin/python -m pytest -q tests/test_model_features.py tests/test_torch_runtime_controls.py tests/test_benchmark_runner.py tests/test_workload_blind_protocol.py`
  -> `67 passed, 1 warning`.
- `git diff --check` passed.

Artifact:

- `artifacts/manual/multitrain0205_c30_segmentcontext_shipbalanced_localswap085_minswap1_20260515`

Result:

- Matched held-out `RangeUseful=0.210407`, below uniform `0.211751`,
  DP `0.185568`, and TemporalRandomFill `0.216053`.
- Audit wins: uniform `0/7`, low-budget uniform `0/3`, DP `6/7`,
  TemporalRandomFill `1/7`.
- Low-budget deltas vs uniform are all negative: `1%=-0.0077`,
  `2%=-0.0049`, `5%=-0.0013`.
- Train target fit is bad: Kendall tau `-0.031`, matched target-recall lift
  `-0.0232`, low-budget lift `-0.0276`.
- Matched geometry is still slightly worse than uniform (`AvgSED=0.5457 km`
  vs `0.5391 km`), while length preservation is effectively tied
  (`0.9281` vs `0.9276`).
- Latency improves relative to KNN students (`108.05ms`) but not enough to
  matter because quality fails.

Conclusion:

- Segment context by itself did not make the retained-frequency labels
  learnable. The useful MLSimp takeaway is not "add segment attention"; it is
  training the lightweight scorer with explicit structural/globality/uniqueness
  pressure or training-only low-budget oracle signal.
- Do not sweep this architecture further without changing the objective.

### Structural retained-frequency target check

Implementation:

- Added `range_training_target_mode="structural_retained_frequency"`.
- The target blends train-workload usefulness with query-free structural scores
  before retained-frequency target selection. The structural score uses
  trajectory-local uniqueness, turn score, time-gap prominence, endpoint flags,
  and centroid-based globality. It is supervision only; eval compression still
  uses query-free model features and frozen retained masks.
- Added `range_structural_target_blend` to config/CLI/reporting and benchmark
  rows for structural score mass and p95.

Validation:

- Focused tests:
  `../.venv/bin/python -m pytest -q tests/test_teacher_distillation.py tests/test_torch_runtime_controls.py tests/test_benchmark_runner.py tests/test_workload_blind_protocol.py tests/test_model_features.py`
  -> `94 passed, 1 warning`.
- `git diff --check` passed before benchmarking.

Artifacts:

- `artifacts/manual/multitrain0205_c30_segmentcontext_structural010_shipbalanced_localswap085_minswap1_20260515`
- `artifacts/manual/multitrain0205_c30_segmentcontext_structural025_shipbalanced_localswap085_minswap1_20260515`
- `artifacts/manual/multitrain0205_c30_segmentcontext_structural050_shipbalanced_localswap085_minswap1_20260515`

Focused result on the 30% held-out split:

| Structural blend | Matched MLQDS | Uniform | DP | TRF | Uniform wins | Low-budget wins | TRF wins | Train tau | Low-budget target-recall lift | AvgSED |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.10` | `0.210951` | `0.211751` | `0.185568` | `0.216053` | `0/7` | `0/3` | `0/7` | `+0.003` | `-0.0279` | `0.5418` |
| `0.25` | `0.212538` | `0.211751` | `0.185568` | `0.216053` | `2/7` | `2/3` | `3/7` | `+0.203` | `-0.0452` | `0.5596` |
| `0.50` | `0.211289` | `0.211751` | `0.185568` | `0.216053` | `0/7` | `0/3` | `1/7` | `+0.063` | `-0.0551` | `0.5381` |

Details for the only partial branch, blend `0.25`:

- Matched `RangeUseful` barely beats uniform: `+0.000787`.
- Low-budget deltas vs uniform: `1%=-0.0053`, `2%=+0.000018`,
  `5%=+0.000787`.
- The target positive fraction jumps to `51.8%`, with target mass `9056.86`
  versus base retained-frequency mass `5051.14`.
- Geometry is worse than uniform: `AvgSED=0.5596 km` versus `0.5391 km`.
- The fit diagnostic is mixed: rank tau improves, but target-recall lift is
  negative at matched and low budgets.

Conclusion:

- Structural/globality/uniqueness supervision creates a real but weak signal.
  It can open the `2%` and `5%` cells for `segment_context_range`, but it still
  loses `1%`, most of the grid, TemporalRandomFill at the matched cell, and
  global geometry.
- This is not final success. It is evidence that structural pressure is worth
  keeping as a diagnostic, but the current retained-frequency formulation
  diffuses too much mass and does not produce a robust low-budget ranker.

### Structural retained-frequency boost check

Implementation:

- Added `range_structural_target_source_mode`.
- `blend` keeps the previous additive structural support behavior.
- `boost` preserves train-usefulness support and only re-ranks useful points by
  query-free structural prominence. This was the clean MLSimp-inspired
  diagnostic for whether globality/uniqueness helps once it stops adding broad
  structural-only positives.

Validation:

- Focused tests:
  `../.venv/bin/python -m pytest -q tests/test_teacher_distillation.py::test_structural_retained_frequency_boost_preserves_label_support tests/test_torch_runtime_controls.py::test_parser_accepts_structural_target_blend`
  -> `2 passed`.
- Wider tests:
  `../.venv/bin/python -m pytest -q tests/test_teacher_distillation.py tests/test_torch_runtime_controls.py tests/test_benchmark_runner.py tests/test_workload_blind_protocol.py tests/test_model_features.py`
  -> `95 passed, 1 warning`.
- `git diff --check` passed.

Artifact:

- `artifacts/manual/multitrain0205_c30_segmentcontext_structural025boost_shipbalanced_localswap085_minswap1_20260515`

Result:

- Matched `RangeUseful=0.211256`, below uniform `0.211751`, above DP
  `0.185568`, and below TemporalRandomFill `0.216053`.
- Audit wins: uniform `0/7`, low-budget uniform `0/3`, DP `6/7`,
  TemporalRandomFill `2/7`.
- Low-budget deltas vs uniform: `1%=-0.0065`, `2%=-0.0037`,
  `5%=-0.0005`.
- Train target fit remains weak: Kendall tau `+0.058`, matched target-recall
  lift `-0.0241`, low-budget lift `-0.0301`.
- Positive target fraction drops back to `29.5%`; target mass `5052.86` is
  essentially the base retained-frequency mass `5051.14`. This confirms boost
  avoided the additive target diffusion, but it also removed the small gains.
- Component losses versus uniform are broad rather than isolated:
  `TemporalCov`, `GapCov`, and `ShapeScore` all trail uniform at the matched
  point; `EntryExitF1`, `CrossingF1`, and `TurnCov` are slightly better but not
  enough to move `RangeUseful`.
- Geometry is still worse than uniform: `AvgSED=0.5451 km` vs `0.5391 km`;
  length preservation is tied (`0.9278` vs `0.9276`).
- Latency is `108.9ms`, far above uniform `2.3ms` and DP `13.9ms`.

Conclusion:

- Boost is cleaner but worse. The earlier additive structural target won a
  couple of cells because it broadened support, not because the model learned a
  robust useful-point ranking.
- Do not sweep `range_structural_target_source_mode=boost` around this
  architecture. The next credible step is objective-level: either train on
  explicit low-budget oracle/teacher distributions or change the student so it
  predicts sparse replacements against a temporal scaffold instead of trying to
  globally rank all useful points.

### Segment-context stratified retained-frequency check

Reason:

- Before adding a new stratum-specific target, inspected the target plumbing.
  `range_retained_frequency_training_labels` already honors
  `mlqds_hybrid_mode`, so `mlqds_hybrid_mode=stratified` trains on the same
  query-blind per-bin selection rule used at inference. Adding another
  "stratified retained-frequency" target would have duplicated existing
  behavior.
- Historical-prior stratified had already failed. This run checks whether the
  trainable MLSimp-inspired `segment_context_range` scorer can use the aligned
  stratified target better than the KNN prior.

Artifact:

- `artifacts/manual/multitrain0205_c30_segmentcontext_stratified_shipbalanced_20260515`

Result:

- Matched `RangeUseful=0.198479`, well below uniform `0.211751`,
  DP `0.185568`, and TemporalRandomFill `0.211369`.
- Audit wins: uniform `0/7`, low-budget uniform `0/3`, DP `5/7`,
  TemporalRandomFill `0/7`.
- Low-budget mean delta vs uniform is `-0.01636`; full-grid mean delta vs
  uniform is `-0.02200`.
- Target positive fraction is moderate (`21.45%`, mass `5364.0`), and train
  fit looks mildly positive (`tau=+0.103`, matched target-recall lift
  `+0.0143`, low-budget lift `+0.0258`), but this does not transfer.
- Matched component losses versus uniform are broad: `RangePointF1`, `ShipCov`,
  `EntryExitF1`, `CrossingF1`, `TemporalCov`, `GapCov`, and `ShapeScore` all
  lose. Only `TurnCov` improves.
- Geometry is bad: `AvgSED=0.6881 km` vs uniform `0.5391 km`; length
  preservation is also lower (`0.9238` vs `0.9276`).
- Latency is `111.9ms`, reasonable for this neural path but irrelevant because
  quality fails.

Conclusion:

- The issue is not just target/selector mismatch. The aligned stratified target
  makes the segment-context scorer worse than uniform in every audited
  compression cell.
- Do not add a duplicate stratified target or sweep `segment_context_range`
  under `stratified` around this setup. The current neural context model is not
  learning transferable per-bin range usefulness.

### Benchmark audit-cell flattening cleanup

Reason:

- The acceptance grid is coverage by compression, but benchmark rows only
  exposed aggregate audit win counts. Exact per-compression `RangeUseful`
  values were buried in each child artifact's `range_compression_audit` table.
  That made low-budget diagnosis slower and made it too easy to talk about a
  run by win counts without showing the actual cell deltas.

Implementation:

- `benchmark_report.py` now flattens each audit compression ratio into JSON/CSV
  row fields:
  `audit_ratio_<ratio>_mlqds_range_usefulness`,
  `audit_ratio_<ratio>_uniform_range_usefulness`,
  `audit_ratio_<ratio>_douglas_peucker_range_usefulness`,
  `audit_ratio_<ratio>_temporal_random_fill_range_usefulness`, and the
  corresponding MLQDS deltas.
- The compact markdown table is unchanged. It keeps aggregate counts so reports
  remain readable; the full machine-readable report carries the detailed grid.

Validation:

- Focused tests:
  `../.venv/bin/python -m pytest -q tests/test_benchmark_runner.py::test_benchmark_row_records_effective_child_torch_runtime tests/test_benchmark_runner.py::test_benchmark_markdown_table_is_compact`
  -> `2 passed`.
- Wider report/config tests:
  `../.venv/bin/python -m pytest -q tests/test_benchmark_runner.py tests/test_torch_runtime_controls.py`
  -> `43 passed`.
- `git diff --check` passed.

Conclusion:

- This does not improve model quality, but it removes a real audit weakness.
  Future benchmark reports can be inspected against the compression-grid
  acceptance criterion without opening child artifacts.

### MLSimp-style global budget diagnostic

Reason:

- MLSimp emphasizes database-level importance ranking and global allocation.
  The final Range-QDS model still must be workload-blind, but a diagnostic
  global budget allocator can test whether the current per-trajectory budget
  rule is hiding headroom.

Implementation:

- Added an endpoint-protected `simplify_with_global_score_budget` helper.
  It keeps a minimal temporal skeleton per trajectory, then spends the
  remaining retained-point budget globally by score.
- Added `GlobalRandomBudget` and `GlobalOracleBudget` to the learned-fill
  diagnostics. These are diagnostic-only methods. `GlobalOracleBudget` uses
  held-out eval labels and cannot be counted as workload-blind success.

Validation:

- Focused tests:
  `../.venv/bin/python -m pytest -q tests/test_metrics.py::test_global_score_budget_preserves_skeleton_and_spends_remaining_budget_globally tests/test_metrics.py::test_score_global_budget_method_validates_score_count tests/test_beats_random_in_distribution.py::test_pipeline_reports_f1_scores`
  -> `3 passed`.
- Wider diagnostic/report/protocol tests:
  `../.venv/bin/python -m pytest -q tests/test_metrics.py tests/test_beats_random_in_distribution.py tests/test_benchmark_runner.py tests/test_workload_blind_protocol.py`
  -> `83 passed`.
- `git diff --check` passed before the benchmark.

Artifact:

- `artifacts/manual/multitrain0205_c30_historicalprior_shipbalanced_localswap085_globaldiag_20260515`

Result:

- Primary current-best branch reproduced the known partial result:
  matched `RangeUseful=0.218596` vs uniform `0.211751`, DP `0.185568`,
  TemporalRandomFill `0.216216`.
- Audit wins vs uniform are still weak: `4/7` overall and `1/3` at
  `1%,2%,5%`. The low-budget `1%` and `2%` cells are temporal ties, not
  learned wins.
- `GlobalRandomBudget` is worse than uniform in every audited compression
  cell. Matched `RangeUseful=0.185698`; latency is only `2.52ms`.
- `GlobalOracleBudget` exposes large headroom when labels are perfect:
  matched `RangeUseful=0.737246`; at `5%,10%,15%,20%,30%` it reaches
  `0.737246`, `0.857323`, `0.920339`, `0.957736`, `0.995275`.
  It is also much better than uniform at `1%` and `2%`
  (`0.235382`, `0.510319`), though the existing per-trajectory oracle remains
  stronger at those two extreme budgets.
- The cheap global allocator is not the bottleneck. The bottleneck is learning
  a transferable workload-blind ranking with enough confidence to spend
  scarce budget globally.

Conclusion:

- MLSimp's global ranking/allocation idea is useful as a diagnostic and likely
  as a training target design cue. It does not rescue the current score.
- The next credible model-side move is not another selector trick. It is a
  training-only target that teaches global scarcity: expected usefulness
  normalized by point/trajectory competition, or teacher-student distillation
  from global oracle allocations generated on training workloads only.

### Training-only global-scarcity target

Reason:

- The global oracle diagnostic showed large headroom, but global random was
  bad. Tested whether train-workload global oracle membership can produce a
  better query-blind score.

Implementation:

- Added `range_training_target_mode="global_budget_retained_frequency"`.
  It uses train-only range labels, keeps the same endpoint skeleton as the
  global diagnostic, then labels useful points that win the remaining
  database-level score competition across configured budgets.
- Added `mlqds_hybrid_mode="global_budget"` as a diagnostic blind selector:
  endpoint skeleton per trajectory, remaining budget spent globally by learned
  score. This is workload-blind, but not a final success mechanism unless it
  beats baselines and preserves trajectory geometry.

Validation:

- Focused target tests:
  `../.venv/bin/python -m pytest -q tests/test_teacher_distillation.py::test_global_budget_retained_frequency_uses_database_level_competition tests/test_teacher_distillation.py::test_global_budget_retained_frequency_aggregates_replicates_after_selection`
  -> `2 passed`.
- Global selector focused tests:
  `../.venv/bin/python -m pytest -q tests/test_metrics.py::test_temporal_score_hybrid_global_budget_delegates_to_global_allocator tests/test_torch_runtime_controls.py::test_parser_accepts_global_budget_target_and_selector tests/test_training_does_not_collapse.py::test_temporal_residual_budget_ratios_match_learned_fill_budget`
  -> `3 passed`.
- Wider relevant suites:
  `../.venv/bin/python -m pytest -q tests/test_metrics.py tests/test_teacher_distillation.py tests/test_torch_runtime_controls.py tests/test_benchmark_runner.py tests/test_workload_blind_protocol.py tests/test_training_does_not_collapse.py`
  -> `164 passed`.

Target-only benchmark:

- Artifact:
  `artifacts/manual/multitrain0205_c30_historicalprior_globalbudgetfreq_shipbalanced_localswap085_20260515`
- Setup: historical-prior scorer, ship-balanced labels, 4 train workload
  replicates, `frequency_mean`, global-budget retained-frequency target, but
  existing `local_swap` selector with `mlqds_temporal_fraction=0.85`.
- Matched `RangeUseful=0.212994` vs uniform `0.211751`, DP `0.185568`,
  TemporalRandomFill `0.216216`.
- Audit wins vs uniform: `4/7`, low-budget wins `1/3`; this is weaker than the
  current-best retained-frequency branch (`0.218596` matched).
- Train target fit is strong (`tau=+0.984`, matched target-recall delta
  `+0.0426`), but the held-out gain is small and misses TemporalRandomFill.
- Geometry is still worse than uniform: `AvgSED=0.5808 km` vs uniform
  `0.5391 km`.

Aligned global-selector benchmark:

- Artifact:
  `artifacts/manual/multitrain0205_c30_historicalprior_globalbudgetfreq_globalselector_20260515`
- Setup: same global-budget target, `mlqds_hybrid_mode=global_budget`,
  `mlqds_temporal_fraction=0.0`, `mlqds_diversity_bonus=0.0`.
- Matched `RangeUseful=0.173004`, below uniform `0.211751`, DP `0.185568`,
  and GlobalRandomBudget/TemporalRandomFill `0.185698`.
- Audit wins vs uniform: `0/7`; low-budget wins `0/3`.
- It improves matched `RangePointF1` (`0.094828` vs uniform `0.089271`) but
  destroys continuity and route quality: `ShipF1=0.438206` vs uniform
  `0.655058`, `TemporalCov=0.198374` vs `0.260320`, `GapCov=0.141557` vs
  `0.247608`, `ShapeScore=0.116618` vs `0.159903`.
- Global geometry is unacceptable: `AvgSED=6.2173 km` vs uniform `0.5391 km`;
  length preserved `0.8553` vs uniform `0.9276`.
- Train target fit looks excellent (`tau=+0.984`, matched target-recall delta
  `+0.5341`, low-budget delta `+0.3624`), so target-fit metrics are actively
  misleading here.

Conclusion:

- Global scarcity is real, but raw global allocation is too continuity-blind.
  The current train-derived score learns point hits and global support at the
  expense of per-ship continuity, gap coverage, route shape, and geometry.
- Do not use `global_budget` as a final selector. It remains a diagnostic for
  measuring how much global allocation damages sensical retained trajectories.
- The next target should constrain global competition within route/ship
  continuity units, not across the whole database with only endpoint skeletons.

### Week-to-week AIS split support and smoke run

Reason:

- The previous real-AIS experiments only used train days `2026-02-02..05`,
  validation `2026-02-06`, and eval `2026-02-07`. That is not enough evidence
  for week-level generalization.
- Cleaned AIS files exist for `2026-02-02` through `2026-02-15`, so the
  proposed protocol is feasible.

Implementation:

- Extended explicit CSV loading so `--validation_csv_path` and
  `--eval_csv_path` accept comma-separated CSV lists, matching the existing
  multi-train behavior.
- Distinct-source checks now inspect each train/validation/eval file in those
  comma lists.
- Multi-day validation/eval runs write combined audit payloads and log combined
  split counts.
- Added `mlqds_hybrid_mode="global_fill"` as a future diagnostic selector:
  temporal base per trajectory, residual learned-fill budget allocated
  globally. This is not yet benchmarked and is not success evidence.

Validation:

- Focused input-resolution tests:
  `../.venv/bin/python -m pytest -q tests/test_benchmark_runner.py::test_resolve_data_sources_accepts_multi_validation_and_eval_csvs tests/test_benchmark_runner.py::test_resolve_data_sources_rejects_duplicate_multi_eval_csv tests/test_benchmark_runner.py::test_resolve_data_sources_accepts_multiple_train_csvs tests/test_benchmark_runner.py::test_resolve_data_sources_rejects_duplicate_multi_train_csv`
  -> `4 passed`.
- Focused `global_fill`/multi-eval tests:
  `../.venv/bin/python -m pytest -q tests/test_metrics.py::test_temporal_score_hybrid_global_fill_preserves_base_and_spends_residual_globally tests/test_torch_runtime_controls.py::test_parser_accepts_global_fill_selector tests/test_benchmark_runner.py::test_resolve_data_sources_accepts_multi_validation_and_eval_csvs`
  -> `3 passed`.
- Broader relevant suites:
  `../.venv/bin/python -m pytest -q tests/test_metrics.py tests/test_benchmark_runner.py tests/test_torch_runtime_controls.py tests/test_workload_blind_protocol.py`
  -> `103 passed`.
- `git diff --check` passed.

Week smoke artifact:

- `artifacts/manual/week0202_0208_eval0209_0215_smoke_20260515`

Setup:

- Train: `2026-02-02..07`, capped to `24` segments per day.
- Checkpoint validation: `2026-02-08`, capped to `24` segments.
- Eval: `2026-02-09..15`, capped to `24` segments per day.
- Historical-prior retained-frequency branch, 2 train workload replicates,
  `query_coverage=0.30`, `n_queries=8`, `max_queries=128`.

Result:

- Loader path worked: train combined `6` sources, `144` segments, `10038`
  points; eval combined `7` sources, `168` segments, `10605` points.
- Matched `RangeUseful=0.297622` vs uniform `0.288702`, DP `0.259732`,
  TemporalRandomFill `0.286177`.
- Audit wins vs uniform: `2/7`; low-budget wins `1/3`; wins vs DP `6/7`;
  wins vs TemporalRandomFill `3/7`.
- Low-budget `1%` and `2%` are still ties with uniform. The matched `5%` cell
  wins, but `15%`, `20%`, and `30%` lose to uniform.
- Geometry remains worse than uniform: `AvgSED=1.1846 km` vs uniform
  `0.8871 km`; DP is better at `0.6959 km`.
- First run built small per-day caches at about `11-13s` per day. Rerunning the
  same capped week split should be much cheaper.

Conclusion:

- The exact week-to-next-week protocol had not been tried before. It is now
  supported and smoke-tested.
- The smoke result is not acceptance evidence because it uses small caps and
  only one coverage target. It does show the same pattern as the day-level
  runs: a matched `5%` gain, low-budget temporal ties, weak full-grid wins, and
  worse geometry.
- A full week-to-week acceptance run is now possible, but the current model is
  unlikely to pass without a stronger learned signal.

### Week-to-week mid-cap current-best run

Reason:

- The first week-to-week smoke used only `24` segments per day. Before paying
  for a full week-level acceptance run, test whether the current-best branch
  keeps its pattern at a larger cap.

Artifact:

- `artifacts/manual/week0202_0208_eval0209_0215_midcap_currentbest_20260515`

Setup:

- Train: `2026-02-02..07`, `60` segments per day, `42960` train points.
- Checkpoint validation: `2026-02-08`, `60` segments, `7081` points.
- Eval: `2026-02-09..15`, `60` segments per day, `48872` eval points.
- Historical-prior retained-frequency branch, ship-balanced labels, 4 train
  workload replicates, `label_mean`, `local_swap`,
  `mlqds_temporal_fraction=0.85`, `query_coverage=0.30`, `n_queries=16`,
  `max_queries=256`.

Result:

- Matched `RangeUseful=0.170825` vs uniform `0.169161`, DP `0.164088`,
  TemporalRandomFill `0.170273`.
- Audit wins vs uniform: `3/7`; low-budget wins `1/3`; wins vs DP `4/7`;
  wins vs TemporalRandomFill `5/7`.
- `1%` and `2%` are again exact temporal ties with uniform.
- Component differences at matched `5%` are tiny and mixed:
  `RangePointF1=0.101916` vs uniform `0.100818`,
  `ShipF1=0.544247` vs `0.541192`,
  `GapCov=0.119362` vs `0.123176`,
  `ShapeScore=0.076256` vs `0.073716`.
- Geometry is still worse than uniform and DP:
  `AvgSED=0.7885 km` vs uniform `0.7367 km` and DP `0.5186 km`.
- Train fit is strong but weakly predictive: `tau=+0.995`,
  matched target-recall delta `+0.0809`, low-budget delta `+0.0270`.
- Runtime bottleneck after cache warmup was range training prep (`48.50s`);
  MLQDS first-mask latency was `601.7ms`.

Conclusion:

- The week-to-week mid-cap result confirms the smoke result was not hiding a
  strong model. The branch barely beats uniform at matched `5%`, loses or ties
  important low-budget cells, and keeps worse geometry.
- Do not spend a full week-level acceptance sweep on this branch as-is. The
  likely outcome is a more expensive version of the same failure.

### Validation split design note and report metadata cleanup

Question checked:

- Whether checkpoint validation should use a holdout split from each training
  day rather than an independent validation day/dataset.

Current evidence:

- Serious day-level and week-to-week runs used independent validation days
  (`2026-02-06` in the main multi-train split, `2026-02-08` in the
  week-to-week split).
- The runner can omit `--validation_csv_path` when separate train/eval CSVs are
  provided; in that mode, validation is randomly split from the combined
  training trajectories. That is not source-stratified by day, so it should not
  be counted as a tested per-training-day validation design.
- A per-day holdout is a valid next A/B only if implemented as source-stratified
  validation. A naive random holdout can overweight dense days and leak
  day/vessel context into checkpoint selection, which would make validation
  easier but less convincing for week-to-week generalization.

Recommendation:

- Keep independent-day validation as the honest generalization gate.
- Add source-stratified within-training-day validation as a selector diagnostic
  or checkpoint-smoothing variant, then compare it directly against the
  independent-day selector on the same week-to-week eval data. Do not treat a
  same-day validation gain as final evidence unless held-out week eval improves.

Code cleanup:

- Added row-level CSV source metadata to benchmark reports:
  `csv_path`, `train_csv_path`, `validation_csv_path`, `eval_csv_path`,
  split-specific file counts, `selected_cleaned_csv_file_count`, and
  `selected_cleaned_csv_files`.
- This fixes an audit gap where `run_config.json` recorded multi-day sources
  correctly, but each top-level benchmark row did not independently show which
  train/validation/eval files produced the score.

Validation:

- `.venv/bin/python -m py_compile QDS/experiments/benchmark_report.py QDS/experiments/benchmark_runner.py`
- `.venv/bin/python -m pytest -q QDS/tests/test_benchmark_runner.py -k "data_source_metadata or concrete_family_root or resolve_data_sources_accepts_multi_validation_and_eval_csvs"`
- `.venv/bin/python -m pytest -q QDS/tests/test_benchmark_runner.py QDS/tests/test_torch_runtime_controls.py`

### Source-stratified training-day validation A/B

Reason:

- The independent-day week-to-week selector may be too strict or too different
  from the training distribution. Test the user's proposed alternative as an
  explicit diagnostic: hold out validation trajectories from each train day,
  select checkpoints on that same-day validation workload, and still score the
  final frozen masks on the held-out eval week.

Implementation:

- Added `DataConfig.validation_split_mode` and CLI
  `--validation_split_mode {random,source_stratified}`.
- Default remains `random`, preserving the old fallback when
  `--train_csv_path`/`--eval_csv_path` are supplied without
  `--validation_csv_path`.
- `source_stratified` requires train trajectory source ids and holds out
  validation trajectories per train CSV source without emptying any source.
- Added `data_split_diagnostics` to `example_run.json` so the held-out source
  counts are auditable.
- Benchmark rows now include `validation_split_mode` and `val_fraction`.

Validation:

- `.venv/bin/python -m py_compile QDS/experiments/experiment_config.py QDS/experiments/experiment_cli.py QDS/experiments/experiment_data.py QDS/experiments/experiment_pipeline.py QDS/experiments/run_ais_experiment.py QDS/experiments/benchmark_report.py`
- `.venv/bin/python -m pytest -q QDS/tests/test_experiment_data.py QDS/tests/test_torch_runtime_controls.py -k "source_stratified or validation_split_mode or roundtrips or exposes"`
- `.venv/bin/python -m pytest -q QDS/tests/test_experiment_data.py QDS/tests/test_benchmark_runner.py QDS/tests/test_torch_runtime_controls.py QDS/tests/test_workload_blind_protocol.py`

Diagnostic artifact:

- `artifacts/manual/week0202_0207_sourceval_eval0209_0215_midcap_currentbest_20260515`

Setup:

- Train source CSVs: `2026-02-02..07`, `60` segments per source.
- No independent validation CSV.
- `validation_split_mode=source_stratified` held out `9` trajectories from each
  of the `6` train sources: `54` validation trajectories total, leaving `306`
  train trajectories.
- Eval unchanged: `2026-02-09..15`, `60` segments per source.
- Same current-best historical-prior retained-frequency branch as the mid-cap
  independent-validation run.

Result:

- Matched `RangeUseful=0.173938` vs uniform `0.169161`, DP `0.164088`,
  TemporalRandomFill `0.170273`.
- Independent-day validation on the comparable mid-cap run was
  `RangeUseful=0.170825`, so same-day source-stratified validation improves the
  matched `5%` cell by `+0.003113`.
- The audit grid gets worse: wins vs uniform drop from `3/7` to `2/7`, and
  wins vs TemporalRandomFill drop from `5/7` to `4/7`.
- Low-budget behavior is unchanged: `1%` and `2%` still tie uniform exactly;
  low-budget wins remain `1/3`.
- High-budget cells degrade materially:
  - `15%`: `0.341706` vs uniform `0.346127`
  - `20%`: `0.398845` vs uniform `0.404274`
  - `30%`: `0.517878` vs uniform `0.526265`
- Geometry remains worse than uniform and DP:
  `AvgSED=0.8075 km` vs uniform `0.7367 km` and DP `0.5186 km`.
- Train target fit is still strong:
  `tau=+0.9888`, matched target-recall delta `+0.0899`, low-budget delta
  `+0.0300`.
- Added selector-capacity diagnostics to refreshed artifacts. For this run's
  eval split:
  - `1%`: learned slots `0`, zero-learned trajectories `100%`
  - `2%`: learned slots `0`, zero-learned trajectories `100%`
  - `5%`: learned slots `243`, only `8.70%` of retained budget;
    `42.14%` of trajectories still have no learned slot

Conclusion:

- Same-day source-stratified validation is useful as a selector diagnostic, but
  it does not fix the acceptance failure. It makes the matched 5% number look
  better while hurting the broader compression grid.
- The low-budget ties are not merely weak labels; under the current capped
  week-to-week setup, the selector gives the learned model no decision capacity
  at `1%` and `2%`. This branch cannot be accepted as a learned-model win.
- Keep independent-day validation as the primary generalization gate. Do not
  treat same-day validation gains as final evidence unless held-out eval-week
  grid wins improve.

### Longer-trajectory low-budget capacity smoke

Reason:

- The `192` point cap made the `1%` and `2%` cells non-diagnostic for the
  current `local_swap` branch: the learned model had zero retained slots to
  influence. Test whether opening learned slots fixes low-budget eval.

Artifact:

- `artifacts/manual/week0202_0207_sourceval_eval0209_0215_p512_minswap1_smoke_20260515`

Setup:

- Train source CSVs: `2026-02-02..07`, `24` segments per source,
  `max_points_per_segment=512`.
- Source-stratified validation held out `3` trajectories per train source:
  `18` validation trajectories, leaving `126` train trajectories.
- Eval: `2026-02-09..15`, `24` segments per source.
- Same historical-prior retained-frequency branch, but with
  `mlqds_min_learned_swaps=1` so low-budget cells can use learned replacements.

Result:

- Matched `RangeUseful=0.244852` vs uniform `0.248133`, DP `0.227763`,
  TemporalRandomFill `0.254525`.
- Audit wins vs uniform: `0/7`; low-budget wins `0/3`; wins vs DP `6/7`;
  wins vs TemporalRandomFill `0/7`.
- Per-ratio deltas vs uniform:
  - `1%`: `-0.001758`
  - `2%`: `-0.002178`
  - `5%`: `-0.003280`
  - `10%`: `-0.005274`
  - `15%`: `-0.002066`
  - `20%`: `-0.006981`
  - `30%`: `-0.002996`
- Selector capacity was no longer zero:
  - `1%`: `61` learned slots, `11.01%` of retained budget
  - `2%`: `70` learned slots, `8.21%` of retained budget
  - `5%`: `191` learned slots, `10.80%` of retained budget
- Train-fit diagnostics look strong despite eval failure:
  `tau=+0.9983`, matched target-recall delta `+0.1006`,
  low-budget delta `+0.0834`.
- Geometry is still worse than uniform and DP:
  `AvgSED=0.1906 km` vs uniform `0.1434 km`, DP `0.1507 km`.

Conclusion:

- The low-budget failure is not only a selector-capacity artifact. Once learned
  slots are opened at `1%` and `2%`, the current historical-prior target still
  fails every uniform audit cell.
- The train target is now actively misleading: strong target recall does not
  transfer to held-out eval-week `RangeUseful`. The next useful target work
  needs either a different teacher/label family or explicit anti-overfit
  validation against independent-day/week workloads, not more scaffold tuning.

### Source-mean historical-prior aggregation smoke

Reason:

- Test whether the retained-frequency target is overfitting pooled historical
  source support. If source pooling is the issue, averaging support by source
  should transfer better to the held-out eval week.
- This is still a capped smoke, not acceptance evidence.

Artifact:

- `artifacts/manual/week0202_0207_sourceval_eval0209_0215_p512_minswap1_sourcemean_20260515`

Setup:

- Same setup as the longer-trajectory low-budget capacity smoke:
  train `2026-02-02..07`, eval `2026-02-09..15`, `24` segments per source,
  `max_points_per_segment=512`, source-stratified validation, retained-frequency
  target, `local_swap`, `mlqds_min_learned_swaps=1`.
- Changed only `historical_prior_source_aggregation=mean`.

Result:

- Matched `RangeUseful=0.239040` vs uniform `0.248133`, DP `0.227763`,
  TemporalRandomFill `0.254525`.
- Audit wins vs uniform: `1/7`; low-budget wins `1/3`; wins vs DP `7/7`;
  wins vs TemporalRandomFill `1/7`.
- Per-ratio `RangeUseful` deltas vs uniform:
  - `1%`: `+0.001592`
  - `2%`: `-0.001894`
  - `5%`: `-0.009092`
  - `10%`: `-0.013821`
  - `15%`: `-0.013171`
  - `20%`: `-0.018123`
  - `30%`: `-0.013777`
- Train-fit weakened but remained positive:
  `tau=+0.7641`, matched target-recall delta `+0.0828`, low-budget delta
  `+0.0691`.
- Component losses vs uniform at matched 5% are broad, especially
  `ShipF1=-0.0460`, `EntryExitF1=-0.0119`, `TemporalCov=-0.0041`,
  `GapCov=-0.0174`; `TurnCov` and `ShapeScore` improve but not enough.
- Geometry remains worse than uniform and DP:
  `AvgSED=0.1932 km` vs uniform `0.1434 km`, DP `0.1507 km`.

Conclusion:

- Source-mean aggregation does not fix transfer. It improves the `1%` cell but
  degrades the matched cell and almost every higher-budget cell.
- The failure is not explained by source pooling alone. The retained-frequency
  target remains misaligned with held-out eval-week `RangeUseful` under blind
  compression.

### Benchmark scale caveat

Current iteration status:

- Most runs so far are focused capped benchmarks, not full acceptance runs.
- They do score the full compression-ratio audit grid (`1%,2%,5%,10%,15%,20%,30%`)
  and the component metrics, but they cap segment count and sometimes segment
  length to keep iteration cost manageable.
- These runs are useful for rejecting weak candidates. They are not sufficient
  for claiming success.

Decision rule:

- Do not spend full-week/full-cap benchmark time on a candidate that loses the
  capped week-to-week filter, especially at `1%,2%,5%`.
- Once a candidate beats uniform and DP across the capped week-to-week grid with
  sane geometry, rerun it as a materially larger/full benchmark before making
  any acceptance claim.
- If capped and full runs disagree, trust the full run and treat the smoke setup
  as biased.

### RangeUseful gap-ablation diagnostics

Reason:

- Validate whether current conclusions depend on `GapCov` being
  count-normalized. Irregular AIS sampling can make count-gap continuity a bad
  proxy for elapsed-time or along-track continuity.

Implementation:

- Kept primary `RangeUseful` schema `7` unchanged.
- Added diagnostic aggregate variants for new runs:
  `range_usefulness_gap_time_score`,
  `range_usefulness_gap_distance_score`, and
  `range_usefulness_gap_min_score`.
- `range_usefulness_gap_min_score` replaces the gap term with
  `min(GapCovTime, GapCovDistance)`.
- Benchmark rows now expose matched-cell gap-ablation scores/deltas and
  audit-grid win counts versus uniform for the ablation variants.

Validation:

- `PYTHONPATH=QDS .venv/bin/python -m py_compile QDS/evaluation/range_usefulness.py QDS/evaluation/evaluate_methods.py QDS/evaluation/metrics.py QDS/experiments/range_diagnostics.py QDS/experiments/benchmark_report.py QDS/experiments/benchmark_runtime.py QDS/experiments/run_inference.py`
- `PYTHONPATH=QDS .venv/bin/python -m pytest -q QDS/tests/test_metrics.py::test_range_usefulness_gap_time_detects_irregular_sampling_gap QDS/tests/test_benchmark_runner.py::test_benchmark_row_records_effective_child_torch_runtime`
- `PYTHONPATH=QDS .venv/bin/python -m pytest -q QDS/tests/test_metrics.py QDS/tests/test_benchmark_runner.py QDS/tests/test_torch_runtime_controls.py QDS/tests/test_range_point_evaluation.py`

Post-hoc ablation on existing 512-point artifacts:

- `week0202_0207_sourceval_eval0209_0215_p512_minswap1_smoke`:
  base/time/distance/min gap variants all keep wins vs uniform at `0/7`;
  low-budget wins remain `0/3`.
- `week0202_0207_sourceval_eval0209_0215_p512_minswap1_sourcemean`:
  base/time/distance/min gap variants all keep wins vs uniform at `1/7`;
  low-budget wins remain `1/3`.

Conclusion:

- The current failure is not an artifact of the count-normalized `GapCov`
  aggregate. Time-span and distance-span gap variants slightly change deltas
  but do not change the decision.
- The next useful work remains target/workload signal quality, not changing
  the primary metric.

### Sparse 5% coverage workload smoke

Reason:

- Test whether the current 512-point local-swap retained-frequency branch only
  fails on broad `30%` coverage workloads, or also fails when future range
  coverage is sparse.
- This is still a capped smoke, not acceptance evidence.

Artifact:

- `artifacts/manual/week0202_0207_sourceval_eval0209_0215_p512_minswap1_c05_smoke_20260515`

Setup:

- Same train/eval data as the 512-point local-swap smoke:
  train `2026-02-02..07`, eval `2026-02-09..15`, `24` segments/source,
  `max_points_per_segment=512`, source-stratified validation.
- Changed `query_coverage` from `0.30` to `0.05`.
- Kept fixed footprint `2.2km/5h`, `n_queries=8`, `max_queries=128`,
  retained-frequency target, `local_swap`, `mlqds_temporal_fraction=0.85`,
  and `mlqds_min_learned_swaps=1`.

Result:

- Matched `RangeUseful=0.319333` vs uniform `0.346695`, DP `0.323960`,
  TemporalRandomFill `0.344900`.
- Matched `RangePointF1=0.101699` vs uniform `0.138826`; this is a large
  point-hit failure under sparse workloads.
- Audit wins vs uniform: `4/7`; low-budget wins `2/3`; wins vs both uniform
  and DP: `3/7`.
- The matched `5%` cell still fails uniform and DP:
  - `MLQDS=0.319333`
  - uniform `0.346695`
  - DP `0.323960`
- Gap-ablation variants do not change the matched decision:
  - time-gap delta vs uniform `-0.0276`
  - distance-gap delta vs uniform `-0.0296`
  - min-gap delta vs uniform `-0.0276`
- Geometry is worse than uniform and DP:
  `AvgSED=0.2117 km` vs uniform `0.1434 km`, DP `0.1507 km`.
- Train target fit is again near-perfect:
  `tau=+1.0000`, matched target-recall delta `+0.1513`, low-budget delta
  `+0.1056`.
- Workload-generation diagnostics are weak for this sparse smoke:
  eval target `5%` ended at `7.00%`, `8` queries, `7/8` queries after target
  coverage was reached, and `eval_query_floor_dominated=True`.
  Eval near-duplicate rate was `12.5%`.

Conclusion:

- The candidate does not become acceptable at sparse coverage. It wins some
  audit ratios, but the matched `5%` compression cell loses uniform and DP.
- Sparse coverage exposes a stronger `RangePointF1` failure, not a hidden model
  win.
- The `5%` coverage workload generator setting itself is noisy with the current
  fixed footprint and `n_queries=8` floor. Final coverage-grid claims need
  multiple held-out seeds/settings or calibrated lower-footprint workloads, not
  one sparse smoke.

### Coverage-sweep runner support

Reason:

- Coverage targets are part of the acceptance grid, but running them required
  manual repeated `benchmark_runner` invocations. That is error-prone and makes
  artifact provenance weaker.

Implementation:

- Added `benchmark_runner.py --coverage_targets`, accepting fractions or
  percents, for example `--coverage_targets 0.05,0.10,0.15,0.30`.
- The runner creates one child run per coverage target and appends compact
  suffixes such as `c05`, `c10`, and `c30` to the child run label.
- The runner rejects combining `--coverage_targets` with `--query_coverage`
  inside `--extra_args`, because that would make coverage precedence ambiguous.
- `run_config.json` records the requested coverage target list.
- Documented the option in `QDS/experiments/README.md`.

Validation:

- `PYTHONPATH=QDS .venv/bin/python -m py_compile QDS/experiments/benchmark_runner.py`
- `PYTHONPATH=QDS .venv/bin/python -m pytest -q QDS/tests/test_benchmark_runner.py::test_benchmark_coverage_target_parser_accepts_fractions_and_percents QDS/tests/test_benchmark_runner.py::test_run_config_records_profile_checkpoint_selection_metric`
- `PYTHONPATH=QDS .venv/bin/python -m pytest -q QDS/tests/test_benchmark_runner.py::test_benchmark_report_records_concrete_family_root`
- `PYTHONPATH=QDS .venv/bin/python -m pytest -q QDS/tests/test_benchmark_runner.py QDS/tests/test_metrics.py QDS/tests/test_torch_runtime_controls.py QDS/tests/test_range_point_evaluation.py`
- `PYTHONPATH=QDS .venv/bin/python -m py_compile QDS/experiments/benchmark_runner.py QDS/experiments/benchmark_report.py QDS/evaluation/range_usefulness.py QDS/evaluation/evaluate_methods.py QDS/evaluation/metrics.py QDS/experiments/range_diagnostics.py QDS/experiments/benchmark_runtime.py QDS/experiments/run_inference.py`
- `git diff --check`

Conclusion:

- This does not make the current model better. It removes a procedural gap for
  final coverage-grid evaluation once a candidate actually clears the focused
  filters.

### Filtered sparse-workload smoke

Artifact:

- `artifacts/manual/week0202_0207_sourceval_eval0209_0215_p512_minswap1_c05_filtered_smoke_20260515`

Setup:

- Same capped week split as the unfiltered sparse smoke:
  train `2026-02-02..07`, eval `2026-02-09..15`, `24` segments/source,
  `max_points_per_segment=512`, source-stratified validation.
- Kept `query_coverage=0.05`, `n_queries=8`, `max_queries=128`,
  fixed `2.2km/5h` footprint, retained-frequency target, historical prior,
  `local_swap`, `mlqds_temporal_fraction=0.85`, and `mlqds_min_learned_swaps=1`.
- Added query acceptance filters:
  `range_max_coverage_overshoot=0.005`,
  `range_max_point_hit_fraction=0.02`,
  `range_max_trajectory_hit_fraction=0.08`,
  `range_duplicate_iou_threshold=0.50`,
  `range_acceptance_max_attempts=1000`.

Workload diagnostics:

- Train workload: `15` accepted queries, coverage `5.10%`,
  no empty, broad, or near-duplicate queries.
- Eval workload: `13` accepted queries, coverage `5.08%`,
  no empty, broad, or near-duplicate queries.
- Train-vs-eval median hit mismatch shrank from the unfiltered failure case:
  point-hit p50 delta `-9`, trajectory-hit p50 delta `-1`.
- Oracle gap remained large:
  train `+0.5777`, eval `+0.5332`.

Result:

- Matched `5%` compression:
  `MLQDS RangeUseful=0.269803`, uniform `0.262994`, DP `0.205225`,
  TemporalRandomFill `0.264042`.
- Matched `RangePointF1=0.128763` vs uniform `0.119644`, DP `0.107076`,
  TemporalRandomFill `0.119918`.
- Audit grid wins vs uniform: `3/7`; low-budget wins `2/3`;
  wins vs both uniform and DP: `3/7`; wins vs TemporalRandomFill: `5/7`.
- Low-budget audit is mixed:
  - `1%`: wins uniform and DP.
  - `2%`: loses uniform, beats DP.
  - `5%`: wins uniform and DP.
- Gap-ablation variants keep the same low-budget conclusion:
  low wins vs uniform `2/3` for primary, time-gap, distance-gap, and min-gap
  scores.
- Geometry is still worse than the baselines at matched `5%`:
  `AvgSED=0.1724 km` vs uniform `0.1434 km`, DP `0.1507 km`;
  length preservation `0.9786` vs uniform `0.9816`, DP `0.9898`.
- Runtime was small for the capped smoke: `12.1s` wall time, with the
  compression audit the largest phase at `4.09s`.

Conclusion:

- Query workload filtering materially changes the sparse-coverage result.
  The unfiltered `5%` smoke was not reliable because the generator overshot and
  produced a train/eval workload mismatch.
- This filtered run is still only a capped smoke, not acceptance evidence. It
  clears the matched `5%` cell on `RangeUseful`, but it does not clear most of
  the compression grid and still has worse global geometry distortion.
- Next benchmarking should use the new `--coverage_targets` path with the same
  query filters, then repeat across held-out workload seeds/settings before any
  claim of model success.

### Filtered capped coverage sweep

Artifact:

- `artifacts/manual/week0202_0207_sourceval_eval0209_0215_p512_minswap1_filtered_cov_sweep_smoke_20260515`

Setup:

- Same capped week split and candidate as the filtered sparse smoke:
  train `2026-02-02..07`, eval `2026-02-09..15`, `24` segments/source,
  `max_points_per_segment=512`, source-stratified validation.
- Used `benchmark_runner.py --coverage_targets 0.05,0.10,0.15,0.30`.
- Kept filtered workload generation:
  `range_max_coverage_overshoot=0.005`,
  `range_max_point_hit_fraction=0.02`,
  `range_max_trajectory_hit_fraction=0.08`,
  `range_duplicate_iou_threshold=0.50`,
  `range_acceptance_max_attempts=1000`.
- Kept `max_queries=128` for this sweep.

Result summary:

| Coverage | Eval cov | Stop | MLQDS | Uniform | DP | TRF | vs uniform | Audit wins vs uniform | Low wins vs uniform |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `5%` | `5.08%` | target reached | `0.269803` | `0.262994` | `0.205225` | `0.264042` | `+0.006809` | `3/7` | `2/3` |
| `10%` | `10.03%` | target reached | `0.295574` | `0.299819` | `0.249379` | `0.299943` | `-0.004245` | `0/7` | `0/3` |
| `15%` | `15.32%` | target reached | `0.283432` | `0.280875` | `0.244724` | `0.275445` | `+0.002558` | `1/7` | `1/3` |
| `30%` | `25.10%` | max queries | `0.241152` | `0.245873` | `0.217262` | `0.237774` | `-0.004721` | `0/7` | `0/3` |

Component diagnosis:

- `5%`: wins matched `RangeUseful`, but worst component is CrossingF1
  (`-0.0273` vs uniform) and geometry is worse.
- `10%`: loses on `GapCov` (`-0.0223`), shape, and ShipF1. This is a real
  failure, not a gap-aggregate artifact.
- `15%`: small matched win, mostly from temporal coverage and shape, but only
  `1/7` audit ratios beat uniform.
- `30%`: this row is invalid as a 30% coverage test because train/eval stopped
  at `max_queries=128` before reaching target coverage.

Conclusion:

- The filtered workload fix is necessary but not sufficient. The candidate is
  unstable across coverage and compression targets.
- The current retained-frequency/historical-prior/local-swap candidate cannot
  be called successful. It wins isolated cells, not the acceptance grid.
- `max_queries=128` is too low for filtered `30%` coverage with this footprint.

### Coverage target miss reporting

Reason:

- The first filtered coverage sweep exposed a reporting flaw: a target-coverage
  miss was visible only through raw `final_coverage` and
  `generation_stop_reason` fields. That is easy to miss in a large benchmark
  report.

Implementation:

- Added explicit benchmark row fields for train/eval/selection query
  generation:
  `*_query_target_reached`, `*_query_target_shortfall`,
  `*_query_target_overshoot`, and
  `*_query_target_missed_by_max_queries`.
- Added these fields to the compact benchmark report table for train/eval and
  selection where they matter for acceptance interpretation.

Validation:

- `PYTHONPATH=QDS .venv/bin/python -m py_compile QDS/experiments/benchmark_report.py`
- `PYTHONPATH=QDS .venv/bin/python -m pytest -q QDS/tests/test_benchmark_runner.py::test_benchmark_row_records_effective_child_torch_runtime QDS/tests/test_benchmark_runner.py::test_query_floor_fields_flags_coverage_target_miss`
- `PYTHONPATH=QDS .venv/bin/python -m pytest -q QDS/tests/test_benchmark_runner.py`
- `PYTHONPATH=QDS .venv/bin/python -m pytest -q QDS/tests/test_benchmark_runner.py QDS/tests/test_metrics.py QDS/tests/test_torch_runtime_controls.py QDS/tests/test_range_point_evaluation.py`
- `git diff --check`

Conclusion:

- This is a research-system hygiene fix, not a model improvement.

### Filtered 30% coverage rerun with larger query cap

Artifact:

- `artifacts/manual/week0202_0207_sourceval_eval0209_0215_p512_minswap1_c30_mq512_filtered_smoke_20260515`

Setup:

- Same capped week split and candidate as the filtered coverage sweep.
- Reran only `query_coverage=0.30`.
- Raised `max_queries` from `128` to `512` and
  `range_acceptance_max_attempts` from `1000` to `3000`.

Workload diagnostics:

- Train coverage reached target: `30.05%`, `148` accepted queries.
- Eval coverage reached target: `30.05%`, `190` accepted queries.
- Train-vs-eval workload distribution was much cleaner than the invalid c30
  sweep row: point-hit p50 delta `-3`, trajectory-hit p50 delta `0`.

Result:

- Matched `5%` compression:
  `MLQDS RangeUseful=0.226018`, uniform `0.230274`, DP `0.212091`,
  TemporalRandomFill `0.226814`.
- Matched deltas:
  vs uniform `-0.004256`, vs DP `+0.013927`, vs TemporalRandomFill
  `-0.000796`.
- Audit wins vs uniform: `2/7`; low-budget wins `2/3`.
  The only uniform wins are `1%` and `2%`; MLQDS loses every ratio from `5%`
  through `30%`.
- Worst matched component loss vs uniform is ShipF1:
  `0.751015` vs `0.782652`, delta `-0.031637`.
- Gap coverage is also worse:
  `0.253887` vs `0.265342`, delta `-0.011455`.
- Geometry remains worse than uniform and DP:
  `AvgSED=0.1713 km` vs uniform `0.1434 km`, DP `0.1507 km`.
- Runtime bottleneck moved to range label preparation:
  `13.49s` of `33.84s` wall time.

Conclusion:

- The c30 failure is not just a `max_queries=128` artifact. Once target
  coverage is actually reached, the candidate still loses uniform at the
  matched `5%` compression cell and across most of the compression grid.
- The useful signal is concentrated in very low compression (`1%`, `2%`), while
  the target `5%+` behavior remains too uniform-like or worse.
- Next model-side step should target the ShipF1/GapCov losses at moderate and
  high coverage. The label fit is already high, so simply fitting the current
  retained-frequency label harder is unlikely to fix the acceptance gap.

### Cleanup pass after accumulated redesign changes

Reason:

- Paused benchmark iteration to look for bugs, stale contracts, and mechanical
  errors after many uncommitted implementation passes.

Checks:

- Full Python bytecode compile:
  `PYTHONPATH=QDS .venv/bin/python -m compileall -q QDS`.
- Full test suite:
  `PYTHONPATH=QDS .venv/bin/python -m pytest -q QDS/tests`.
- Static type check:
  `cd QDS && ../.venv/bin/pyright --stats`.
- Whitespace/conflict-marker check:
  `git diff --check`.
- Searched changed surfaces for obvious debugger/TODO/error patterns.

Fixes:

- Added `_model_point_dim()` in `training/inference.py` and used it in
  inference, evaluation baselines, and validation scoring. This makes the
  `point_dim` contract explicit for historical-prior and workload-blind models
  that are plain `torch.nn.Module` implementations rather than
  `TrajectoryQDSModel` subclasses.
- Relaxed training and validation helper type contracts from
  `TrajectoryQDSModel` to `torch.nn.Module` where the runtime code supports all
  scorer models. This removes a stale type assumption introduced by adding
  historical-prior models.
- Added an explicit historical-prior model guard in `train_model.py` before
  calling `set_prior()` and reading stored prior buffers.
- Tightened benchmark/profile typing:
  `_assert_distinct_csv_sources()` now accepts a read-only `Mapping`, and
  profile settings allow `list[str]`.
- Made `attach_range_geometry_scores()` accept a `Sequence[Method]`, matching
  how frozen retained-mask method lists are passed.
- Hardened historical-prior retained-frequency diagnostics reads with a numeric
  diagnostics helper instead of indexing `dict[str, object]` as if all values
  were already numeric.
- Cleaned type noise in tests with explicit model-type assertions and
  diagnostics access helpers.

Validation result:

- `compileall`: passed.
- Full tests: `284 passed`, `1` PyTorch nested-tensor warning.
- Pyright: `0 errors`, `0 warnings`, `104` checked source files.
- `git diff --check`: passed.

Extra discovery:

- The runtime behavior was already covered by tests, but Pyright exposed a real
  stale design assumption: much of the training/evaluation code still typed the
  active model as `TrajectoryQDSModel` even though the redesign added
  non-transformer scorer modules. That was not crashing because Python is
  duck-typed, but it made the type contracts misleading and would have hidden
  future integration mistakes.

### Cleanup pass: production asserts and extra static checks

Reason:

- Continued the cleanup pause with checks aimed at bugs tests may not catch:
  stale runtime assertions, duplicate literal dictionary keys, mutable default
  arguments, and lint-style hazards.

Checks:

- `ruff` and `pyflakes` are not installed in the environment.
- Custom AST duplicate literal dictionary key scan: passed.
- Custom AST mutable literal default scan for production code: passed.
- Production assert scan:
  `rg -n "^\\s*assert\\s|\\bassert\\s" QDS/{evaluation,experiments,models,queries,simplification,training} --glob '*.py'`.

Fixes:

- Replaced production `assert` guards with explicit runtime errors in:
  - `training/train_model.py`
  - `queries/query_generator.py`
  - `experiments/range_cache.py`
  - `experiments/range_diagnostics.py`
  - `training/training_targets.py`
  - `experiments/experiment_pipeline.py`
  - `experiments/benchmark_process.py`
- Added `_require_validation_inputs()` in `train_model.py` so checkpoint
  validation scoring fails with a clear error if called without complete
  validation inputs.
- Replaced component-label narrowing asserts with explicit `RuntimeError`s,
  preserving behavior under optimized Python.

Validation result:

- Production assert scan: no remaining production asserts in the scanned
  modules.
- `PYTHONPATH=QDS .venv/bin/python -m compileall -q QDS`: passed.
- `cd QDS && ../.venv/bin/pyright`: `0 errors`, `0 warnings`.
- `PYTHONPATH=QDS .venv/bin/python -m pytest -q QDS/tests`:
  `284 passed`, `1` PyTorch nested-tensor warning.
- `git diff --check`: passed.

Extra discovery:

- The production asserts were mostly type-narrowing guards and would usually
  work in normal runs, but relying on them is still wrong. Running with
  `python -O` would remove those checks and turn clear invariant failures into
  later, noisier crashes.

### Cleanup pass: targeted Ruff correctness lint

Reason:

- Previous cleanup found that the repo did not have Ruff or Pyflakes installed.
  Installed Ruff into the local virtualenv for a one-off correctness lint pass
  without changing repository dependencies.

Checks:

- Installed local tool:
  `.venv/bin/python -m pip install ruff`
  installed `ruff==0.15.13`.
- Ran correctness-only lint:
  `.venv/bin/ruff check QDS --select F,E9`.
- Re-ran changed-Python compile:
  `git diff --name-only -- '*.py' | sort | xargs -r .venv/bin/python -m py_compile`.
- Re-ran Pyright:
  `cd QDS && ../.venv/bin/python -m pyright data evaluation experiments models queries simplification training scripts tests`.
- Re-ran full tests:
  `PYTHONPATH=QDS .venv/bin/python -m pytest -q QDS/tests`.
- Re-ran dependency and diff checks:
  `cd QDS && ../.venv/bin/python -m pip check`;
  `git diff --check`.

Fix:

- Removed one unused import from `experiments/benchmark_inputs.py`:
  `DEFAULT_PROFILE` was imported but not used.

Validation result:

- Ruff `F,E9`: passed.
- Changed Python compile: passed.
- Pyright: `0 errors`, `0 warnings`.
- Full tests: `284 passed`, `1` PyTorch nested-tensor warning.
- `pip check`: no broken requirements.
- `git diff --check`: passed.

Extra discovery:

- Ruff is useful enough here to justify adding it as a tracked dev dependency
  later, but I did not add it to `requirements.txt` in this cleanup pass to
  avoid changing the project's declared environment without a broader tooling
  decision.

### Cleanup pass: final-claim and selector-scaffold reporting guard

Reason:

- The latest filtered high-coverage artifact exposed a reporting risk: with
  `mlqds_temporal_fraction=0.85` and `local_swap`, only `10.8%` of the matched
  5% eval budget came from learned slots. A benchmark could previously be
  labeled as a workload-blind RangeUseful success even if most of the retained
  set came from temporal scaffolding.

Fixes:

- Added a benchmark-report claim guard:
  `selector_claim_status`, `selector_claim_has_material_learned_budget`, and
  `selector_claim_min_learned_slot_fraction`.
- `single_cell_range_status` now refuses to report
  `beats_uniform_and_douglas_peucker` when the matched selector budget is
  scaffold dominated or missing selector-budget evidence.
- Changed benchmark profile metadata so workload-blind profiles are recorded as
  `final_product_candidate=True` but `final_product_claim=False`; the claim must
  come from protocol, selector-evidence, and coverage/compression report gates,
  not the chosen profile name.

Validation result:

- `PYTHONPATH=QDS .venv/bin/python -m pytest -q QDS/tests/test_benchmark_runner.py -q`:
  passed.
- `PYTHONPATH=QDS .venv/bin/python -m py_compile` for changed report/profile/test
  files: passed.
- `ruff F,E9` on changed report/profile/test files: passed.

Extra discovery:

- The previous low-coverage "good" results should be re-read skeptically if
  they used high temporal scaffold settings. Wins with a learned-slot fraction
  below the guard threshold are diagnostic, not final evidence that model
  learning caused the improvement.

### Cleanup pass: pyproject dependency migration and Ruff dev dependency

Reason:

- The correctness lint pass was useful, but Ruff was only installed manually.
  Dependency management was also split between `requirements.txt` and
  `pyproject.toml`, which made stale dependency declarations likely.

Fixes:

- Moved runtime dependencies into `[project.dependencies]` in `pyproject.toml`.
- Moved dev/test dependencies into `[project.optional-dependencies].dev`.
- Removed `requirements.txt`; there is no compatibility wrapper to go stale.
- Set `requires-python = ">=3.14"` because this repo is currently validated on
  Python 3.14.
- Added Ruff configuration to `pyproject.toml`; it enforces the
  correctness-focused rule set `F,E9` and infers Python 3.14 from
  `requires-python`.
- Added `make lint` at the repo root and in `QDS/Makefile`.
- Included Ruff in `make qds-check-env` version reporting.
- Fixed stale `Aleks-Sprint` documentation links to point at `Sprint`.

Validation result:

- `make install`: passed using editable install `pip install -e ".[dev]"`.
- `make lint`: passed.
- `make qds-check-env`: passed and reports `ruff 0.15.13`.
- Ruff settings show target version `3.14`.
- `make test`: `286 passed`, `1` PyTorch nested-tensor warning.
- `make typecheck`: `0 errors`, `0 warnings`.
- `git diff --check` on changed dependency/tooling files: passed.

Decision:

- `pyproject.toml` is now the dependency source of truth. Install with
  `pip install -e ".[dev]"` from the repo root.

### Cleanup pass: targeted footgun lint cleanup

Reason:

- Continued the non-experimental cleanup pass with targeted Ruff checks for
  common correctness and maintainability hazards beyond the default `F,E9`
  gate: mutable defaults, suspicious raises, redundant branches, unnecessary
  temporary returns, and simple context-manager nesting.

Fixes:

- Simplified range-table pure-range detection and shift-table key iteration in
  `QDS/evaluation/tables.py`.
- Removed redundant temporary-return assignments in
  `QDS/training/training_setup.py` and AIS cleaning helpers.
- Normalized EOF newlines and spacing in AIS cleaning step modules.
- Flattened nested DB connection/cursor context managers in `db/smoke_test_db.py`.
- Removed the stale compatibility `db/run_range_query.py` entry point in the
  follow-up cleanup; it should not be referenced as a supported script.

Validation result:

- Targeted Ruff hazard set: passed.
- `compileall` for `ais_pipeline`, `db`, `QDS`, and `main.py`: passed.
- `make lint`: passed.
- `make typecheck`: `0 errors`, `0 warnings`.
- `make test`: `286 passed`, `1` PyTorch nested-tensor warning.
- `git diff --check`: passed.

Extra discovery:

- Remaining broad exception handlers are intentional in the inspected paths:
  transaction rollback, child-process cleanup, optional environment telemetry,
  benchmark failure status writing, and non-fatal training diagnostics. I left
  them alone because narrowing them would make those cleanup/diagnostic paths
  less reliable without a concrete failure case.

### Cleanup pass: QDS duplicate logic and shared helpers

Reason:

- Several Range-QDS implementation iterations left repeated low-level logic in
  the benchmark, training, and model paths. These are exactly the kind of small
  inconsistencies that can later skew protocol claims or make smoke runs differ
  from benchmark runs.

Fixes:

- Centralized workload-blind model-type classification in
  `training.model_features.is_workload_blind_model_type`; benchmark profiles and
  benchmark reports now use the same predicate as training/checkpoint code.
- Moved cached sinusoidal positional encoding into
  `models.positional_encoding.CachedSinusoidalPositionalEncodingMixin`; the
  query-aware, workload-blind, and segment-context models no longer carry three
  copies of the same cache/build logic.
- Added `experiments.cli_utils` for shared CLI normalization:
  `normalized_gap_arg` and `split_csv_path_list`.
- Removed duplicate gap-normalization and CSV-list parsing helpers from
  experiment, inference, and benchmark input code.
- Added `QDS/tests/test_cli_utils.py` for the shared CLI behavior.

Validation result:

- Focused touched-path tests:
  `39 passed`.
- `PYTHONPATH=QDS .venv/bin/python -m compileall -q QDS`: passed.
- Extended Ruff hazard set for QDS: passed.
- Duplicate function-body scan across QDS for functions with at least three
  top-level statements: no duplicates found.
- `make lint`: passed.
- `make typecheck`: `0 errors`, `0 warnings`.
- `make test`: `289 passed`, `1` PyTorch nested-tensor warning.
- `git diff --check`: passed.

Extra discovery:

- `training.model_features` still intentionally contains old feature-shape
  compatibility paths for checkpoint loading. That is not dead code by current
  tests, but it is a real maintenance cost. If old checkpoints are no longer
  valuable, delete those compatibility branches in a separate, explicit cleanup.

### Cleanup pass: AIS geospatial helper consolidation

Reason:

- The AIS cleaning step and its validation tools each carried an identical
  PySpark `haversine_km` implementation plus duplicated distance constants.
  That can make outlier removal and post-cleaning diagnostics disagree if one
  copy changes.

Fixes:

- Added `ais_pipeline.geo` with shared `EARTH_RADIUS_KM`, `KNOTS_TO_KMH`, and
  `haversine_km`.
- Updated `ais_pipeline.steps.remove_outliers`,
  `ais_pipeline.tools.check_gaps`, and
  `ais_pipeline.tools.validate_cleaning` to use the shared helper.
- Fixed the progress log reference to the removed `db/run_range_query.py`
  compatibility entry point so it is not described as a supported script.

Validation result:

- Root Ruff correctness profile: passed.
- Extended Ruff hazard set across `QDS`, `ais_pipeline`, and `db`: passed.
- Duplicate function-body scan across Python source roots: no duplicates found
  for functions with at least three top-level statements.
- `compileall` for `ais_pipeline`, `db`, `QDS`, and `main.py`: passed.
- `make lint`: passed.
- `make typecheck`: `0 errors`, `0 warnings`.
- `make test`: `289 passed`, `1` PyTorch nested-tensor warning.
- `git diff --check`: passed.

Extra discovery:

- There are no Spark integration tests for the AIS cleaning scripts. This pass
  validates imports and static behavior, not a live Spark run over AIS input.

### Cleanup pass: QDS default footguns and model registry cleanup

Reason:

- The direct CLI/config default for `temporal_residual_label_mode` was still
  `temporal`, even though a prior experiment was explicitly marked invalid
  because the manual run accidentally used that default. Residual-only training
  should be opt-in, not the default.
- Model-type strings were duplicated across the parser, feature builder,
  training path, and checkpoint loader.
- Saved point-feature reconstruction had parallel query-free `if` ladders and a
  redundant old/current workload-blind dimension branch.

Fixes:

- Changed the direct CLI/config default
  `temporal_residual_label_mode` from `temporal` to `none`.
- Kept benchmark profiles explicit, so diagnostic residual training still runs
  only when the profile asks for it.
- Added a regression test that direct config and CLI defaults are non-residual.
- Added `SUPPORTED_MODEL_TYPES`, `QUERY_AWARE_MODEL_TYPES`,
  `WORKLOAD_BLIND_MODEL_TYPE_CHOICES`, `HISTORICAL_PRIOR_MODEL_TYPES`, and
  `NONPARAMETRIC_HISTORICAL_PRIOR_MODEL_TYPES` in `training.model_features`.
- Updated parser choices, training model selection, and checkpoint loading to
  use the shared model-type registry constants.
- Centralized query-free saved-feature reconstruction through one helper used by
  both generic saved-model inference and workload-blind inference.

Validation result:

- Focused touched-path tests:
  `45 passed`, `1` PyTorch nested-tensor warning.
- `PYTHONPATH=QDS .venv/bin/python -m compileall -q QDS`: passed.
- Extended Ruff hazard set for QDS: passed.
- Duplicate function-body scan across QDS for functions with at least three
  top-level statements: no duplicates found.
- `make lint`: passed.
- `make typecheck`: `0 errors`, `0 warnings`.
- `make test`: `291 passed`, `1` PyTorch nested-tensor warning.
- `git diff --check`: passed.

Extra discovery:

- There is still unavoidable dispatch branching for model classes in training
  and checkpoint loading because historical-prior models need checkpoint
  backfill and prior-specific constructor arguments. Centralizing the model-type
  names fixes the stale-string risk without hiding that behavior behind a
  fragile generic factory.

### Cleanup pass: QDS checkpoint and workload-generation behavior

Reason:

- Checkpoint loading filtered retired query-config keys but still allowed stale
  model/data/baseline config keys to break old checkpoints. That is wrong for
  checkpoint compatibility while normal config loading should remain strict.
- Checkpoint loading silently fell back to the baseline query-aware model for an
  unknown persisted `model_type`. That can make a removed or misspelled model
  look loadable while changing inference behavior.
- Fixed-count range workload generation stopped after `n_queries` candidate
  attempts even when acceptance filters rejected candidates and
  `range_acceptance_max_attempts` allowed more retries. That can quietly shrink
  train/validation/eval workloads.

Fixes:

- Centralized checkpoint config-section filtering for `data`, `query`, `model`,
  and `baselines`, while preserving strict `ExperimentConfig.from_dict` behavior
  outside checkpoint loading.
- Added checkpoint-loader regression coverage for stale section keys, missing
  checkpoint config sections, and unknown persisted model types.
- Changed fixed-count range workload generation to retry until the requested
  query count is accepted or the configured acceptance attempt budget is
  exhausted.
- Added a regression where one rejected fixed-count range candidate previously
  returned 7/8 queries despite an 80-attempt budget.

Validation result:

- Focused query/checkpoint tests: `84 passed`, `1` PyTorch nested-tensor warning.
- `PYTHONPATH=QDS .venv/bin/python -m compileall -q QDS`: passed.
- Extended Ruff hazard set for QDS: passed.
- `make lint`: passed.
- `make typecheck`: `0 errors`, `0 warnings`.
- `make test`: `295 passed`, `1` PyTorch nested-tensor warning.
