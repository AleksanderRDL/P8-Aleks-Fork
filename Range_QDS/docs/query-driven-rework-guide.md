# Range_QDS Query-Driven Rework Guide

This is the active operating guide for the Range_QDS redesign. It is written for a new engineer or implementation agent continuing the work from the current repository state.

The project is **not** trying to build generic trajectory simplification. The goal is:

> Train from a stable future range-query workload distribution, then compress validation/eval AIS trajectories **before future eval queries are known**, while preserving the points and trajectory evidence most likely to matter for those future queries.

The final result must come from learned workload-blind model behavior. A win caused mostly by query-conditioned inference, temporal scaffolding, checkpoint leakage, historical KNN lookup, or selector tricks is not acceptable.

---

## 1. End-state objective

The desired final system is a query-driven, workload-blind AIS compressor.

At deployment/eval time, the system receives only trajectories and train-derived artifacts. It must produce retained masks before future range queries are known. Later, those future queries should still be answered well from the compressed data.

The final system should satisfy four things at once:

1. **Workload-blind compression**
   - No eval query boxes, query tensors, query/point containment labels, query boundary distances, or eval-query-derived features before retained masks are frozen.

2. **Query-driven learned behavior**
   - The model is trained from generated/historical training workloads.
   - The model learns stable workload priors and query-local behavior value.
   - The learned model materially affects retained masks.

3. **Future-query usefulness**
   - Compressed trajectories preserve likely in-query point mass.
   - Within likely query ranges, retained points should explain ship behavior: presence, entry/exit, crossings, temporal span, turns, local shape, speed/heading changes, and enough local evidence to reconstruct movement.

4. **Sensible global trajectories**
   - Global geometry is not the primary goal, but retained trajectories must not become nonsensical.
   - Endpoint sanity, rough length preservation, and bounded geometry distortion are guardrails.

The target product result is not “best possible geometric simplification.” It is “best useful data retained for likely future range-query workloads.”

---

## 2. Current implementation state

The candidate path exists and runs end to end:

```text
workload_profile_id               = range_workload_v1
model_type                        = workload_blind_range_v2
range_training_target_mode         = query_useful_v1_factorized
selector_type                     = learned_segment_budget_v1
checkpoint_score_variant           = query_useful_v1
primary metric                    = QueryUsefulV1
legacy diagnostic metric           = RangeUsefulLegacy
```

The following components are implemented and should be preserved unless a targeted replacement is justified:

```text
Range_QDS/queries/workload_profiles.py
Range_QDS/queries/query_generator.py
Range_QDS/evaluation/query_useful_v1.py
Range_QDS/training/query_useful_targets.py
Range_QDS/training/query_prior_fields.py
Range_QDS/models/workload_blind_range_v2.py
Range_QDS/simplification/learned_segment_budget.py
Range_QDS/experiments/experiment_pipeline.py
Range_QDS/experiments/benchmark_report.py
```

The repository includes gates and diagnostics for:

```text
workload stability
workload signature stability
support overlap
query-prior predictability
prior-predictive alignment
target diffusion
learning causality
prior sample sensitivity
global sanity
final 4x7 grid acceptance
```

Integration verification currently passes:

```text
git diff --check
ruff
pyright
tests/test_query_driven_rework.py
tests/test_query_coverage_generation.py
tests/test_benchmark_runner.py
full test suite
```

The remaining PyTorch nested-tensor warning is not a project blocker unless CI treats warnings as errors.

---

## 3. Current evidence and active blocker

The most recent strict debug probe failed. It is useful evidence, not a success claim.

Result:

```text
QueryUsefulV1:
  MLQDS:           0.0645
  uniform:         0.1190
  DouglasPeucker:  0.1478

RangeUsefulLegacy:
  MLQDS:           0.0369
  uniform:         0.0936
  DouglasPeucker:  0.1228
```

Final success was blocked by:

```text
workload_stability_gate
predictability_gate
prior_predictive_alignment_gate
workload_signature_gate
learning_causality_ablations
global_sanity_gates
full_coverage_compression_grid
```

Support overlap and target diffusion passed. That means the model had nonzero prior-field input support and labels were not obviously too diffuse. The first active blocker is **workload generation health and signature stability**, not model architecture.

The strict probe showed unhealthy workload generation:

```text
train accepted:      11 / 16 requested queries
selection accepted:  11 / 16 requested queries
eval accepted:       16 / 16 requested queries
train/selection exhausted 6000 attempts
eval required 3305 attempts and rejected 3289 candidates
dominant rejection reason: too_broad
```

The accepted workloads were not clean samples from the intended profile. They were heavily filtered subsets. Planned family quotas matched, but accepted train/eval signatures diverged.

Train-vs-eval signature failure examples:

```text
point-hit distribution KS:  1.0000
ship-hit distribution KS:   0.5966
anchor-family L1:           0.3295
footprint-family L1:        0.1477
```

Predictability also failed:

```text
aggregate Spearman:           0.0332
aggregate Kendall tau:       -0.0316
PR-AUC lift:                  0.9440
lift@1%,2%,5%,10%:            0.0
```

Per-head diagnostics showed one useful clue:

```text
segment_budget_target:
  Spearman: 0.3177
  lift@5%:  1.1791

query_hit_probability:
  lift@5%: 0.0
```

Interpretation: the current train-derived prior has weak segment-structure signal, but it does not predict held-out future query-hit mass. Query-hit transfer must improve before model tuning is likely to matter.

Learning causality failed decisively:

```text
MLQDS:                          0.0645
MLQDS_shuffled_scores:           0.2281
MLQDS_untrained_model:           0.1167
MLQDS_without_segment_budget:    0.1209
MLQDS_prior_field_only_score:    0.0652
MLQDS_without_query_prior:       0.0645
```

Interpretation: the trained score is actively harmful in this probe. Removing the segment-budget head improves the result, and removing query-prior features changes nothing. This is not learned success.

Global sanity also failed:

```text
MLQDS length preservation: 0.5907
required minimum:          0.8000
```

Do not run the full 4x7 grid until a strict single-cell probe passes or gives a narrower diagnosis.

---

## 4. Design contract

The redesign is a contract between five components:

```text
1. versioned future-query workload profile
2. query-local metric
3. factorized train labels and train-only priors
4. trainable workload-blind model
5. selector with material learned control and sanity guardrails
```

All five must be aligned. If one is broken, a downstream model sweep will waste time.

### Correct order of work

1. Stabilize workload generation and signatures.
2. Verify train-derived priors predict held-out query usefulness.
3. Improve target/model only after the predictability audit says useful signal exists.
4. Improve selector only when learned scores are useful but retained masks are poor.
5. Run real-AIS probes only after synthetic/debug probes have clean workload health.
6. Run the full grid only after a strict support-valid single-cell probe passes.

---

## 5. Protocol rules

### Allowed

- Training labels from training workloads.
- Query-aware teachers on training workloads.
- Historical/train-derived priors built before eval compression.
- Validation workloads for checkpoint selection, if validation retained masks are blind and validation scoring does not use final eval queries.
- Eval workloads for scoring only after eval masks are frozen.

### Forbidden for final claims

- Eval queries passed into the model, feature builder, selector, retained-set decision, checkpoint selector, or query-prior builder before compression.
- Eval point/query containment features.
- Eval query boundary-distance features.
- Query cross-attention at eval compression time.
- Checkpoint selection using final eval-query performance.
- Treating query-conditioned `range_aware` as workload-blind success.
- Treating historical-prior KNN lookup as final learned-model success.
- Using eval geometry labels or geometry-label blending for a final workload-blind mask.
- Using a large temporal scaffold that makes learned scores mostly irrelevant.

### Required artifact flags

Every serious run should record:

```yaml
workload_blind_protocol:
  enabled: true
  masks_frozen_before_eval_query_scoring: true
  eval_queries_seen_by_model: false
  eval_queries_seen_by_feature_builder: false
  eval_queries_seen_by_selector: false
  checkpoint_selected_on_eval_queries: false
  query_conditioned_range_aware_used_for_product_acceptance: false
```

Any run that violates these rules is diagnostic only.

---

## 6. Workload profile requirements

The query workload is the product prior. It must be stable enough for the model to learn and broad enough to represent expected future use.

### Active profile

```text
workload_profile_id = range_workload_v1
```

### Recommended anchor-family weights

Use these unless real product query logs justify a change:

```yaml
anchor_family_weights:
  density_route: 0.40
  boundary_entry_exit: 0.20
  crossing_turn_change: 0.15
  port_or_approach_zone: 0.15
  sparse_background_control: 0.10
```

Rationale:

- `density_route` captures recurring traffic corridors.
- `boundary_entry_exit` supports range-query entry/exit evidence.
- `crossing_turn_change` supports behavior explanation.
- `port_or_approach_zone` captures stable AIS-relevant hotspots.
- `sparse_background_control` prevents overfitting to only dense areas, but should not dominate.

Do not increase sparse/background weight merely to make the benchmark broader. That makes uniform temporal sampling close to minimax and undermines the query-driven premise.

### Recommended footprint-family weights

Start with:

```yaml
footprint_family_weights:
  small_local: 0.25
  medium_operational: 0.45
  large_context: 0.20
  route_corridor_like: 0.10
```

Recommended nominal shapes:

```yaml
small_local:
  spatial_radius_km: 1.1
  time_half_window_hours: 2.5

medium_operational:
  spatial_radius_km: 2.2
  time_half_window_hours: 5.0

large_context:
  spatial_radius_km: 4.0
  time_half_window_hours: 8.0

route_corridor_like:
  spatial_radius_km: 2.2
  time_half_window_hours: 5.0
  elongation_allowed: true
```

Current blocker indicates the profile/acceptance settings may still be too hard to sample cleanly on small synthetic splits. Before changing the model, make accepted train/eval workload signatures stable.

### Coverage calibration

Final candidate workloads should use:

```text
coverage_calibration_mode = profile_sampled_query_count
workload_stability_gate_mode = final
```

Do not use `uncovered_anchor_chasing` for final claims unless it is explicitly declared part of the product workload. It can distort the query distribution by chasing uncovered points.

### Query count

Final-profile workloads must not pass with too few queries. A query workload can technically reach coverage with a small number of broad boxes, but that is not enough evidence that the workload profile is stable or learnable.

Hard gate:

```text
minimum accepted queries per workload for final-mode gates: 8
```

Recommended practical sizes:

```text
tiny smoke only:                         4-8 accepted queries
minimum strict diagnostic:               16 accepted queries
standard strict synthetic diagnostic:    32-64 accepted queries
standard real-AIS single-cell probe:     64-128 accepted queries
multi-seed / multi-day confirmation:     64-128 accepted queries per workload
final grid evidence:                     64-256 accepted queries per workload where runtime/data scale permits
```

Use the hard gate only as a lower bound. Do not interpret an 8-query workload as strong scientific evidence. At low query counts, anchor-family proportions, footprint-family proportions, point-hit distributions, ship-hit distributions, and top-k predictability metrics are too noisy.

If strict overshoot plus acceptance filters cannot generate at least 8 healthy accepted queries, the workload profile or footprint settings are wrong for that dataset scale. If it can generate 8 but not at least 16-32 without exhaustion or severe rejection pressure, treat the run as a small diagnostic only.

### Generator health requirements

A final-candidate workload should fail if:

```text
range_acceptance.exhausted == true
stop_reason in {range_acceptance_exhausted, range_coverage_guard_exhausted}
query_count < 8
rejection_rate > 0.85
coverage_guard_rejection_pressure > 2.0
coverage target below/above guard
profile id mismatch
coverage calibration mode not profile_sampled_query_count
```

Recommended diagnostic fields:

```text
accepted
rejected
attempts
rejection_rate
coverage_guard_rejection_pressure
rejection_reasons
rejection_reasons_by_anchor_family
rejection_reasons_by_footprint_family
planned_anchor_family_counts
accepted_anchor_family_counts
planned_footprint_family_counts
accepted_footprint_family_counts
```

### Workload signature gate

Train/eval signatures should compare:

```text
anchor-family distribution
footprint-family distribution
point hits per query
ship hits per query
near-duplicate rate
broad-query rate
query count
profile id
```

Recommended thresholds:

```yaml
anchor_family_l1_distance_max: 0.12
footprint_family_l1_distance_max: 0.12
point_hit_distribution_ks_max: 0.20
ship_hit_distribution_ks_max: 0.20
near_duplicate_rate_max: 0.05
broad_query_rate_max: 0.05
min_signature_query_count: 8
```

If planned family quotas match but accepted signatures fail, acceptance filters are skewing the generated workload. Fix the generator/profile first.

---

## 7. Metric requirements

### Primary metric

```text
QueryUsefulV1
```

It should emphasize:

```text
query-local point mass
query-local behavior explanation
ship presence and coverage inside query windows
entry/exit and crossing evidence
turn/shape/local interpolation
small global sanity guardrail
```

### Legacy metric

```text
RangeUsefulLegacy
```

Keep it for comparability and diagnostics. Do not use it as the final product metric.

### Current metric caveat

`QueryUsefulV1` now includes a true query-local interpolation component, but it is still partly a bridge over old range-audit components. It is acceptable as the active primary metric for current work, but future improvements should continue making it more query-local and less dependent on global proxies.

### Recommended future metric improvements

Add or strengthen:

```text
query-local interpolation fidelity
query-local speed/heading reconstruction
query-local retained evidence factor
query-local point-mass distribution preservation
per-ship query-local behavior explanation
```

Important rule:

> A trajectory should not get high query-local behavior score solely from outside-query anchors. At least one retained in-query point, or a clearly justified retained bracket/evidence rule, should be required for nonzero local evidence credit.

---

## 8. Target, prior, model, and selector requirements

### Factorized target

Active target:

```text
query_useful_v1_factorized
```

Required heads:

```text
query_hit_probability
conditional_behavior_utility
boundary_event_utility
replacement_representative_value
segment_budget_target
```

Recommended interpretation:

```text
final point score ≈ P(future query hits point)
                  × behavior usefulness if queried
                  × replacement/non-redundancy value
                  + boundary/event bonus
```

Keep the heads separate. Do not collapse everything prematurely into one retained-frequency scalar.

### Query-prior fields

Query-prior fields must be built from training workloads only.

Fields should include:

```text
spatial_query_hit_probability
spatiotemporal_query_hit_probability
boundary_entry_exit_likelihood
crossing_likelihood
behavior_utility_prior
route_density_prior
```

They must record:

```yaml
built_from_split: train_only
contains_eval_queries: false
contains_validation_queries: false
profile_id: range_workload_v1
train_workload_seed: ...
extent: ...
out_of_extent_sampling: ...
```

Current recommendation:

```text
out_of_extent_sampling = nearest
```

Use `nearest` for debug/real probes where train/eval route support overlaps but eval points may slightly exceed train bounds. Use `zero` only when explicitly testing support failure.

### Prior predictability

Before tuning the neural model, the train-derived prior must predict held-out eval usefulness.

Required diagnostics:

```text
aggregate Spearman / Kendall
PR-AUC lift over base rate
lift@1%, 2%, 5%, 10%
per-head predictability
prior-channel predictability
top-k eval target mass
sampled prior nonzero fraction
out-of-extent fraction
```

Per-head diagnostics are mandatory because aggregate failure can hide useful sub-signals.

Focus first on:

```text
query_hit_probability
segment_budget_target
```

If `query_hit_probability` fails, the future query-hit prior is not transferring. Fix workload generation, prior fields, or query-profile semantics before model tuning.

If `query_hit_probability` works but `segment_budget_target` fails, fix target/selector alignment.

If `segment_budget_target` works but query-hit fails, the selector may preserve structurally useful points in wrong query regions.

### Model

Active model:

```text
workload_blind_range_v2
```

The model must:

```text
score points query-free at eval compression time
use train-derived prior features only
produce factorized heads
support ablations by disabling heads
generalize across held-out days/seeds under the same workload profile
```

Recommended defaults for debug probes:

```yaml
embed_dim: 32
num_heads: 2
num_layers: 1
epochs: 3-5
loss_objective: budget_topk
mlqds_score_mode: rank_confidence
```

For real AIS probes, increase capacity only after workload health and predictability gates pass.

### Selector

Active selector:

```text
learned_segment_budget_v1
```

The selector should:

```text
retain a minimal rough skeleton
allocate most budget by learned segment value
select points within segments by learned value + non-redundancy
report learned-controlled retained slots
avoid fixed high temporal scaffold
```

Recommended default:

```yaml
learned_segment_geometry_gain_weight: 0.12
learned_segment_score_blend_weight: 0.05
learned_segment_fairness_preallocation: true
```

These are query-free selector guardrails. They must be reported and ablated.

Causality ablation must include:

```text
without_trajectory_fairness_preallocation
without_segment_budget_head
without_behavior_utility_head
without_query_prior_features
shuffled_prior_fields
shuffled_scores
untrained_model
prior_field_only_score
```

If fairness preallocation or geometry tie-breaker drives the win more than learned heads, the result is not clean learned success.

---

## 9. Learning-causality requirements

A final candidate must prove that learning caused the retained-mask improvement.

Required evidence:

```text
trained model beats shuffled scores
trained model beats untrained model
trained model beats prior-field-only score
trained model beats shuffled prior fields
trained model beats no-query-prior-features
trained model beats no-behavior-head
trained model beats no-segment-budget-head
selector learned-controlled slot fraction is material
retained masks change when relevant learned signals are ablated
```

Default material delta:

```text
min QueryUsefulV1 delta: 0.005
```

For shuffled scores, if MLQDS beats uniform, require:

```text
shuffled score loss >= max(0.005, 0.60 × MLQDS-vs-uniform gap)
```

Recommended learned-slot thresholds:

```text
compression >= 10%: learned-controlled slot fraction >= 0.35
compression >= 5%:  learned-controlled slot fraction >= 0.25
```

Low-compression budgets may have limited learned slots, but the model still must matter where slots exist.

If an untrained model or shuffled score beats the trained model, stop model tuning and diagnose target/predictability/selector.

---

## 10. Required acceptance gates

A strict single-cell probe must pass all of these before running the full grid:

```text
workload_stability_gate_pass = true
support_overlap_gate_pass = true
predictability_gate_pass = true
prior_predictive_alignment_gate_pass = true
target_diffusion_gate_pass = true
workload_signature_gate_pass = true
learning_causality_gate_pass = true
global_sanity_gate_pass = true
MLQDS QueryUsefulV1 > uniform
MLQDS QueryUsefulV1 > DouglasPeucker
```

### Support-overlap gate schema

Required fields:

```json
{
  "gate_pass": false,
  "eval_points_outside_train_prior_extent_fraction": 0.0,
  "sampled_prior_nonzero_fraction": 0.0,
  "primary_sampled_prior_nonzero_fraction": 0.0,
  "route_density_overlap": 0.0,
  "query_prior_support_overlap": 0.0,
  "train_eval_spatial_extent_intersection_fraction": 0.0,
  "failed_checks": []
}
```

Recommended thresholds:

```text
eval_points_outside_train_prior_extent_fraction <= 0.10
sampled_prior_nonzero_fraction >= 0.50
primary_sampled_prior_nonzero_fraction >= 0.30
route_density_overlap >= 0.25
query_prior_support_overlap >= 0.25
```

Support overlap is necessary but not sufficient. It proves that priors are sampled nontrivially, not that they predict usefulness.

### Global sanity gate

Current hard checks:

```text
avg_length_preserved between 0.80 and 1.20
endpoint_sanity = 1.0 for eligible trajectories
avg_sed_ratio_vs_uniform <= threshold
```

SED ratio threshold:

```text
compression <= 1%:  2.00
compression <= 2%:  1.75
otherwise:          1.50
```

If this gate repeatedly fails, do not hide the problem with a large temporal scaffold. Add query-free learned/sanity-aware selector constraints or revise the metric/threshold only with justification.

---


## 11. Probe scale policy and evidence levels

Small probes are useful for code correctness and fast failure localization. They are dangerous for scientific conclusions.

A tiny run can pass or fail for reasons that disappear at realistic scale:

```text
one query family dominates by chance
one rejected query shifts family L1 distances
KS distances are meaningless with too few queries
lift@1% selects only a handful of points
1% and 2% compression mostly test endpoints and rounding
causality ablations have too few learned-controlled slots
global sanity swings because of one trajectory
```

Use the levels below. Do not make conclusions that exceed the evidence level.

### Level 0 — static verification and unit tests

Purpose:

```text
verify code integration
verify no-leakage invariants
verify report fields and gates exist
verify masks freeze before eval scoring
```

Recommended scope:

```text
no model-quality experiment required
unit tests and focused smoke tests only
```

Allowed conclusions:

```text
code path works or is broken
metadata/gates are present or missing
protocol invariants are enforced or violated
```

Forbidden conclusions:

```text
model learns
workload profile is stable
prior predictability is useful
selector is effective
final candidate is promising
```

### Level 1 — tiny smoke experiment

Purpose:

```text
confirm the end-to-end command runs
confirm artifacts are emitted
confirm no obvious tensor/shape/config bug
```

Recommended scale:

```text
ships:               4-12
points/ship:          32-96
synthetic families:   1-2
accepted queries:     4-8
train replicates:     1-2
epochs:               1-2
compression:          one ratio, usually 5% or 20%
coverage:             one target, usually 10%
```

Allowed conclusions:

```text
implementation path runs
artifact schema is valid
gates are emitted
obvious bugs exist or do not exist
```

Forbidden conclusions:

```text
learning is real
predictability is good
generator is healthy
workload signatures are stable
uniform/DP wins are meaningful
```

### Level 2 — minimum strict diagnostic single-cell

Purpose:

```text
localize the current blocker under final-mode gates
separate generator/profile failure from prior/target/model/selector failure
```

Minimum recommended scale:

```text
ships:               24-32
points/ship:          128-192
synthetic families:   2-3
accepted queries:     16-32
train replicates:     4
epochs:               3-5
compression:          5%
coverage:             10%
acceptance attempts:  20,000+
```

Allowed conclusions:

```text
which gate is currently blocking
whether generator health is obviously bad
whether support overlap exists
whether prior channels have nonzero sample support
whether target diffusion is obviously bad
```

Still forbidden:

```text
final model-quality claim
full predictability claim
low-budget robustness claim
coverage-grid claim
```

If this level fails because of generator exhaustion, too few accepted queries, signature drift, or tiny learned-slot counts, do not tune the model. Increase scale or fix the workload profile first.

### Level 3 — standard strict diagnostic single-cell

Purpose:

```text
make a serious one-cell claim about signal, learning causality, and selector behavior
```

Recommended scale:

```text
ships:               48-96
points/ship:          192-384
synthetic families:   3-6
accepted queries:     32-64
train replicates:     4-8
epochs:               5-10
compression:          5%
coverage:             10% or 15%
acceptance attempts:  30,000-60,000
```

Required evidence:

```text
workload_stability_gate_pass = true
workload_signature_gate_pass = true
support_overlap_gate_pass = true
target_diffusion_gate_pass = true
predictability_gate_pass = true
prior_predictive_alignment_gate_pass = true
learning_causality_gate_pass = true
global_sanity_gate_pass = true
MLQDS QueryUsefulV1 > uniform
MLQDS QueryUsefulV1 > DouglasPeucker
```

Allowed conclusion:

```text
candidate is worth testing on real AIS
candidate is not worth testing on real AIS
```

### Level 4 — real AIS strict single-cell

Purpose:

```text
show the same strict single-cell behavior transfers to real held-out AIS data
```

Recommended scale depends on data availability, but do not use tiny caps for evidence:

```text
train days/sources:       at least 2-4 historical days when available
validation day/source:    separate from train and eval
eval day/source:          separate held-out day/source
trajectories:             at least 48-128 after segmentation
points/trajectory cap:    192-512 where runtime permits
accepted queries:         64-128 per workload
train replicates:         4-8
epochs:                   5-10
compression:              5%
coverage:                 10% or 15%
```

Required evidence:

```text
same gates as Level 3
route/support overlap is real but not identical leakage
per-head predictability is stable
causality ablations remain meaningful
global sanity remains within thresholds
```

Allowed conclusion:

```text
candidate is worth multi-seed/multi-day confirmation
candidate needs target/prior/model/selector work
```

### Level 5 — multi-seed / multi-day confirmation

Purpose:

```text
verify the single-cell result is not a seed/day accident
```

Recommended scale:

```text
seeds:                    3-5
real train/eval splits:    2-4 when data is available
coverage targets:          at least 10% and 15%
compression ratios:        at least 2%, 5%, and 10%
accepted queries:          64-128 per workload
train replicates:          4-8
```

Required evidence:

```text
mean and worst-case QueryUsefulV1 vs uniform/DP
gate pass rate
per-head predictability stability
learning-causality stability
global sanity stability
runtime/latency stability
```

Allowed conclusion:

```text
ready for full 4x7 grid
not ready for full 4x7 grid
```

### Level 6 — final 4x7 grid

Purpose:

```text
final acceptance
```

Required grid:

```text
coverage targets:     5%, 10%, 15%, 30%
compression ratios:   1%, 2%, 5%, 10%, 15%, 20%, 30%
cells:                28
```

Required evidence:

```text
no missing cells
all child gates pass
numeric success bars pass
all component/runtime/latency fields reported
real AIS held-out-day evidence included
```

### Scale-specific interpretation rules

Use these rules when deciding what to do next:

```text
If Level 1 passes, run Level 2. Do not claim learning.
If Level 2 fails on generator/signature, fix workload generation or increase scale.
If Level 2 passes but predictability fails, fix prior fields / workload profile / targets.
If Level 3 fails causality, diagnose target/loss/model/selector.
If Level 3 passes, run Level 4 on real AIS.
If Level 4 passes, run Level 5.
If Level 5 passes, run Level 6.
```

Never run the full grid to “see what happens” when the standard strict single-cell evidence is already failing. The grid is expensive and mostly useful after the one-cell gates show the candidate is coherent.

---

## 12. Full final grid requirements

Run the full final grid only after a strict single-cell probe passes.

Coverage targets:

```text
0.05, 0.10, 0.15, 0.30
```

Compression ratios:

```text
0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30
```

Required cells:

```text
4 coverage targets × 7 compression ratios = 28 cells
```

### Numeric success bars

The benchmark-level final grid should pass:

```text
MLQDS beats uniform on QueryUsefulV1 in at least 19 / 28 cells
MLQDS beats Douglas-Peucker on QueryUsefulV1 in at least 24 / 28 cells
MLQDS beats uniform in at least 7 / 12 low-budget cells
MLQDS beats uniform in at least 3 / 4 matched 5% compression cells
```

Low-budget cells are:

```text
compression ratios 0.01, 0.02, 0.05 across all 4 coverage targets
```

These thresholds are a practical minimum for claiming “most grid cells” without requiring impossible perfection. If the project later demands a stricter standard, raise these thresholds, do not lower them to fit weak results.

### Required report fields

Every final-grid run should report:

```text
QueryUsefulV1
RangeUsefulLegacy
RangePointF1
ShipF1
ShipCov
EntryExitF1
CrossingF1
TemporalCov
GapCov
GapCovTime
GapCovDistance
TurnCov
ShapeScore
query-local interpolation fidelity
global SED/PED distortion
length preservation
runtime
latency
workload stability gate
workload signature gate
support overlap gate
predictability gate
prior-predictive alignment gate
target diffusion gate
learning causality gate
global sanity gate
selector learned-slot attribution
per-head predictability
prior-channel predictability
generation rejection diagnostics
```

Do not report only aggregate wins. Component regressions are often where false success hides.

---

## 13. Ambiguity-resolution recommendations

When a value or strategy is ambiguous, use these defaults unless evidence says otherwise.

### Workload profile

```text
Use range_workload_v1.
Use profile_sampled_query_count.
Use final gate mode for any acceptance evidence.
Use 4-8 train workload replicates.
Use strict overshoot tolerance by coverage:
  5%  -> 0.005
  10% -> 0.0075
  15% -> 0.010
  30% -> 0.020
```

Recommended query scale by evidence level:

```text
tiny smoke:                       4-8 accepted queries
minimum strict diagnostic:         16-32 accepted queries
standard strict diagnostic:        32-64 accepted queries
real AIS single-cell evidence:     64-128 accepted queries
multi-seed confirmation:           64-128 accepted queries
final grid:                        64-256 accepted queries where data/runtime permits
```

A run with only 8 accepted queries can verify gates exist. It should not be used to claim workload stability or model learning.

### Data

For synthetic debug:

```text
Use shared-route support: synthetic_route_families >= 2 for minimum strict diagnostics.
Use synthetic_route_families = 3-6 for standard strict diagnostics.
Use enough scale that the profile can generate healthy accepted queries.
Minimum strict diagnostic: n_ships >= 24-32 and n_points >= 128-192.
Standard strict diagnostic: n_ships >= 48-96 and n_points >= 192-384.
```

For real evidence:

```text
Use real AIS held-out days.
Use separate train, validation, and eval CSVs.
Prefer at least 2-4 train days/sources when available.
Use at least 48-128 trajectories after segmentation for evidence runs.
Use point caps of 192-512 where runtime permits.
Require route/support overlap but not identical query sets.
```

Synthetic shared-route success is not final evidence. It is a debugging step.

### Model and training

Start small until gates pass:

```text
embed_dim = 32
num_heads = 2
num_layers = 1
epochs = 3-5
train_batch_size = 8
inference_batch_size = 8
loss_objective = budget_topk
checkpoint_selection_metric = uniform_gap
checkpoint_score_variant = query_useful_v1
```

Increase model capacity only after:

```text
workload health passes
signature stability passes
prior predictability has useful signal
trained model is not worse than controls
```

### Selector

Keep:

```text
mlqds_temporal_fraction = 0.0
selector_type = learned_segment_budget_v1
```

Do not reintroduce high temporal scaffolding to improve scores. Use query-free sanity constraints and ablations instead.

### Metric

Use `QueryUsefulV1` as primary. Use `RangeUsefulLegacy` only as diagnostic.

If QueryUsefulV1 rewards behavior that conflicts with the actual product goal, improve the metric explicitly and bump/report the schema version. Do not silently change interpretation.

---

## 14. Risk register

### Risk: drawing scientific conclusions from undersized probes

Small probes can prove code correctness, but they cannot prove workload stability, predictability, learning causality, or final success. This project is especially sensitive because query-family distributions, top-k predictability, learned-slot attribution, and global geometry all become unstable at low query/trajectory counts.

Mitigation:

```text
Use Level 0-1 only for implementation verification.
Use Level 2 only for blocker localization.
Use Level 3+ before claiming useful signal.
Use Level 4+ before trusting real-data behavior.
Use Level 6 only after earlier levels pass.
```

If a result changes dramatically when moving from tiny smoke to standard diagnostic scale, trust the larger run and treat the smaller one as biased.

### Risk 1 — Workload profile is not learnable

Symptom:

```text
workload signatures pass
support overlap passes
but predictability/prior alignment fails
```

Meaning:

The workload family may not have stable cross-day spatial/temporal signal, or the train-derived prior fields are too weak.

Fix direction:

```text
inspect per-head predictability
improve query-prior raster/field construction
use more historical train days
improve route/hotspot representation
narrow or clarify workload profile
```

### Risk 2 — Accepted workloads are rejection-biased

Symptom:

```text
planned family quotas match
accepted anchor/footprint distributions differ
range_acceptance exhausted
too_broad/coverage_overshoot dominates
workload signature fails
```

Fix direction:

```text
adjust footprint sizes
reduce overly broad families
improve acceptance filters
calibrate query count and footprint scale separately
report rejection skew by family
```

### Risk 3 — Model fits target but target is wrong

Symptom:

```text
train target fit good
predictability maybe acceptable
eval QueryUsefulV1 poor
ablations show learned heads not useful
```

Fix direction:

```text
compare per-head target fit against eval usefulness
improve replacement/segment target
add true counterfactual marginal gain samples
avoid diffuse labels
```

### Risk 4 — Selector dominates learning

Symptom:

```text
fairness/geometry ablation drives gains
trained-vs-shuffled delta small
learned-slot fraction low
untrained model competitive
```

Fix direction:

```text
increase learned decision authority carefully
improve model score reliability first
report and ablate every query-free selector heuristic
```

### Risk 5 — Global sanity blocks useful query behavior

Symptom:

```text
MLQDS improves QueryUsefulV1 but fails length/SED
```

Fix direction:

```text
add query-free sanity head or selector constraint
preserve endpoints and rough path-length without high temporal scaffold
penalize bad validation sanity more strongly
```

### Risk 6 — Uniform is close to optimal

Symptom:

```text
query workload broad/random
query-hit predictability low
uniform beats learned model consistently
```

Fix direction:

```text
revisit product workload assumptions
do not force ML success under arbitrary broad workloads
narrow workload profile only if it matches real use
```

---

## 15. Failure diagnosis tree

Use this order after every strict probe.

### Step 1 — Workload generation

If any fail:

```text
workload_stability_gate
workload_signature_gate
generator health fields
```

then do not tune model. Fix generator/profile/acceptance.

Primary questions:

```text
Are at least 8 queries accepted per workload?
Did generation exhaust?
Are rejection rates too high?
Are accepted anchor/footprint family distributions close to planned?
Are point/ship-hit distributions close across train/eval?
```

### Step 2 — Support and predictability

If support overlap fails:

```text
fix train/eval route support or prior-field extent/sampling
```

If support passes but predictability fails:

```text
inspect per-head predictability
query_hit_probability must improve first
segment_budget signal alone is not enough
```

### Step 3 — Causality

If trained model loses to untrained/shuffled/prior-only controls:

```text
stop architecture sweeps
inspect target alignment and loss
verify scores materially change retained masks
```

### Step 4 — Global sanity

If QueryUsefulV1 improves but global sanity fails:

```text
add sanity-aware selector/model constraints
do not add large temporal scaffold
```

### Step 5 — Baseline comparison

If gates pass but MLQDS loses uniform/DP:

```text
diagnose component deltas
check whether metric rewards intended behavior
check whether selector is allocating budget to wrong segments
```

---

## 16. Forward roadmap: start checkpoints from here

Use concise checkpoints. Numbering restarts here.

### Checkpoint 1 — Workload generator health and signature stability

Goal:

```text
Produce a strict support-valid synthetic/debug single-cell workload where generation is healthy and train/eval signatures pass at a scale large enough to evaluate workload stability.
```

Use the standard strict diagnostic scale unless runtime makes it impossible. A smaller 24-32 ship / 16-query run is allowed only as a quick preliminary failure-localization step.

Recommended command shape:

```bash
uv run --group dev -- python -m experiments.run_ais_experiment \
  --results_dir Range_QDS/artifacts/results/query_driven_v2_checkpoint01_generator_health_probe_standard_c10_r05 \
  --n_ships 64 \
  --n_points 256 \
  --synthetic_route_families 4 \
  --seed 2324 \
  --n_queries 48 \
  --query_coverage 0.10 \
  --max_queries 256 \
  --range_max_coverage_overshoot 0.0075 \
  --range_train_workload_replicates 4 \
  --workload_profile_id range_workload_v1 \
  --coverage_calibration_mode profile_sampled_query_count \
  --workload_stability_gate_mode final \
  --model_type workload_blind_range_v2 \
  --range_training_target_mode query_useful_v1_factorized \
  --selector_type learned_segment_budget_v1 \
  --checkpoint_score_variant query_useful_v1 \
  --checkpoint_selection_metric uniform_gap \
  --validation_score_every 1 \
  --checkpoint_full_score_every 1 \
  --checkpoint_candidate_pool_size 1 \
  --epochs 3 \
  --embed_dim 32 \
  --num_heads 2 \
  --num_layers 1 \
  --train_batch_size 8 \
  --inference_batch_size 8 \
  --compression_ratio 0.05 \
  --mlqds_temporal_fraction 0.0 \
  --mlqds_hybrid_mode fill \
  --mlqds_score_mode rank_confidence \
  --range_acceptance_max_attempts 40000 \
  --final_metrics_mode diagnostic
```

Pass condition for this checkpoint:

```text
workload_stability_gate_pass = true
workload_signature_gate_pass = true
support_overlap_gate_pass = true
target_diffusion_gate_pass = true
generation not exhausted
accepted query count >= 32 preferred, >= 16 minimum for a diagnostic
rejection rate acceptable
coverage guard rejection pressure acceptable
```

If it fails, change workload/profile/generator settings or increase dataset scale. Do not tune the model.

Possible fixes:

```text
increase dataset scale
increase accepted query target only when generation is healthy
reduce large_context weight
reduce footprint radii
reduce footprint jitter
make route_corridor_like less broad
increase max attempts only if rejection rate is not structurally high
separate debug profile from final profile if needed
```

### Checkpoint 2 — Prior predictability and target alignment

Goal:

```text
Make train-derived priors predict held-out query usefulness under a healthy workload.
```

Pass condition:

```text
predictability_gate_pass = true
prior_predictive_alignment_gate_pass = true
query_hit_probability has positive Spearman and useful lift@5
segment_budget_target has useful lift@5
```

If query-hit fails:

```text
fix prior fields/workload profile
do not tune model
```

If query-hit passes but behavior/replacement/segment fails:

```text
fix factorized targets and segment labels
```

### Checkpoint 3 — Learned model causality

Goal:

```text
Make the trained model beat ablations under one strict healthy single-cell probe.
```

Pass condition:

```text
learning_causality_gate_pass = true
MLQDS > uniform on QueryUsefulV1
MLQDS > DouglasPeucker on QueryUsefulV1
trained model beats untrained/shuffled/prior-only/no-head ablations
```

If not, diagnose target/loss/model capacity only after Checkpoints 1-2 pass.

### Checkpoint 4 — Global sanity correction

Goal:

```text
Pass length/endpoint/SED global sanity without a high temporal scaffold.
```

Possible fixes:

```text
add length/skeleton auxiliary head
add selector constraint for path-length support
increase geometry-gain tie-breaker carefully
add validation sanity hard constraint
```

Do not use `mlqds_temporal_fraction=0.85` or similar scaffolded masking.

### Checkpoint 5 — Real AIS strict single-cell

Goal:

```text
Repeat strict single-cell evidence on real held-out AIS days.
```

Use separate:

```text
train_csv_path
validation_csv_path
eval_csv_path
```

Pass condition:

```text
all single-cell gates pass
MLQDS beats uniform and DP on QueryUsefulV1
learning causality passes
global sanity passes
```

### Checkpoint 6 — Full final grid

Goal:

```text
Run the 4x7 grid only after a strict real-AIS single-cell passes.
```

Pass condition:

```text
all child gates pass
numeric success bars pass
component/reporting checklist complete
```

---

## 17. What not to do

Do not compensate for failures with:

```text
query-conditioned inference
eval-query feature leakage
checkpoint selection on final eval queries
range_aware as final result
historical_prior KNN as final learned success
large temporal scaffold
geometry-label blending disguised as learning
loose coverage overshoot
tiny query counts in final gates
artificially easy workload profiles not matching product intent
```

Do not continue sweeping:

```text
KNN neighbor count
source-day agreement aggregation
local-swap temporal fraction
min learned swaps around current KNN score
retained-frequency budget weighting
pointwise MLP imitation of KNN
old RangeUseful scalar target blends
```

Those were useful historically but are no longer the main path.

---

## 18. Relevant files by task

### Workload profile and generator

```text
Range_QDS/queries/workload_profiles.py
Range_QDS/queries/query_generator.py
Range_QDS/queries/workload_diagnostics.py
Range_QDS/experiments/range_diagnostics.py
Range_QDS/experiments/workload_cache.py
```

### Metric and evaluation

```text
Range_QDS/evaluation/query_useful_v1.py
Range_QDS/evaluation/evaluate_methods.py
Range_QDS/evaluation/metrics.py
Range_QDS/evaluation/query_cache.py
```

### Targets and priors

```text
Range_QDS/training/query_useful_targets.py
Range_QDS/training/query_prior_fields.py
Range_QDS/training/factorized_target_diagnostics.py
Range_QDS/training/predictability_audit.py
```

### Model and training

```text
Range_QDS/models/workload_blind_range_v2.py
Range_QDS/training/train_model.py
Range_QDS/training/training_epoch.py
Range_QDS/training/training_validation.py
Range_QDS/training/model_features.py
```

### Selector

```text
Range_QDS/simplification/learned_segment_budget.py
Range_QDS/simplification/mlqds_scoring.py
```

### Reports and gates

```text
Range_QDS/experiments/experiment_pipeline.py
Range_QDS/experiments/benchmark_report.py
Range_QDS/experiments/benchmark_runner.py
```

### Tests

```text
Range_QDS/tests/test_query_driven_rework.py
Range_QDS/tests/test_query_coverage_generation.py
Range_QDS/tests/test_benchmark_runner.py
```

---

## 19. Progress-log format

Use concise checkpoints. Each checkpoint should record:

```md
## Checkpoint N — <short name>

Status: completed / partial / failed

Goal:
- ...

Changes:
- ...

Tests:
- ...

Experiment artifact:
- path: ...
- command: ...

Key results:
- MLQDS QueryUsefulV1: ...
- uniform QueryUsefulV1: ...
- Douglas-Peucker QueryUsefulV1: ...
- gates passed: ...
- gates failed: ...

Decision:
- continue / pivot / stop and diagnose
```

Keep the progress log short. Detailed stdout and raw metrics belong in artifacts.

---

## 20. Completion definition

The redesign is complete only when:

```text
1. the full 4x7 grid is present
2. QueryUsefulV1 final-grid numeric success bars pass
3. all child gates pass
4. workload-blind protocol flags prove no eval query leakage
5. learning causality proves trained model behavior matters
6. RangeUsefulLegacy and all component metrics are reported
7. real AIS held-out-day evidence passes
8. retained trajectories are globally sane
9. failures and limitations are documented honestly
```

Anything less is either a useful diagnostic, a partial result, or a negative result.
