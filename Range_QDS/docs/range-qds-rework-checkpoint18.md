# Range_QDS Query-Driven Rework Fix Brief

## Purpose

This brief turns the current investigation findings into a concrete implementation plan for the steps.

The current rework is **not complete**. The implementation has useful scaffolding and several real bug fixes, but the candidate still does not satisfy the ambition:

> A trained workload-blind model should compress trajectories before future eval queries are known, preserve points likely to matter for future range-query workloads drawn from the same stable distribution, and beat uniform temporal sampling and Douglas-Peucker on the primary query-driven metric because of learned model behavior, not scaffolding or selector tricks.

Do not claim success until strict evidence passes.

---

## Current State Summary

The following has been implemented and should mostly be preserved:

```text
Range_QDS/docs/query-driven-rework-guide.md
Range_QDS/docs/query-driven-rework-progress.md
Range_QDS/queries/workload_profiles.py
Range_QDS/evaluation/query_useful_v1.py
Range_QDS/training/query_useful_targets.py
Range_QDS/training/query_prior_fields.py
Range_QDS/models/workload_blind_range_v2.py
Range_QDS/simplification/learned_segment_budget.py
```

The current candidate path exists:

```text
workload_profile_id = range_workload_v1
model_type = workload_blind_range_v2
range_training_target_mode = query_useful_v1_factorized
selector_type = learned_segment_budget_v1
checkpoint_score_variant = query_useful_v1
primary metric = QueryUsefulV1
```

Good work already done:

- Legacy `RangeUseful` is separated from final claim logic.
- Frozen-mask workload-blind protocol exists.
- `QueryUsefulV1` exists.
- Train-only query-prior fields exist.
- Factorized model and labels exist.
- Learned segment-budget selector exists.
- Causality, predictability, workload-stability, workload-signature, target-diffusion, prior-sample, global-sanity, and final-grid gates exist.
- Several real bugs were fixed:
  - endpoint retention was not mandatory enough;
  - validation checkpoint scoring did not use the segment-budget head;
  - no-segment-head ablation was not actually neutral;
  - disabled multiplicative heads used advantaged constants;
  - out-of-extent prior sampling was clamped into fake prior mass;
  - geometry tie-breaker ignored full-trajectory anchors in some segments.

But the active goal is still blocked.

---

## Do Not Claim Success

Current evidence does **not** satisfy final success.

Known failed or weak evidence:

- Full 4x7 coverage/compression grid has not passed.
- Strict accepted-workload probes still fail predictability, causality, global sanity, DP comparison, or several of these.
- Loose same-support probes can beat uniform and DP, but they use loose overshoot or artificial route support and are not final evidence.
- Strict profile can beat uniform in one validation-selected probe, but still loses to Douglas-Peucker and fails causality/global sanity.
- Length preservation repeatedly fails the global sanity gate.
- Segment-budget head effect remains weak.
- Query-prior support is absent in many synthetic splits because eval points are outside train-prior extent.

Keep the final-claim gates strict.

---

## Current Best Evidence To Respect

### Encouraging but not accepted

Strict validation-selected probe:

```text
MLQDS QueryUsefulV1:           0.2527
Uniform QueryUsefulV1:         0.1981
Douglas-Peucker QueryUsefulV1: 0.2895
```

Interpretation:

- Beats uniform.
- Loses to Douglas-Peucker.
- Fails causality.
- Fails length preservation.
- Fails predictability.
- Best checkpoint is epoch 1; later epochs overfit train loss.

Loose same-support probe:

```text
MLQDS QueryUsefulV1:           0.3676
Uniform QueryUsefulV1:         0.2646
Douglas-Peucker QueryUsefulV1: 0.2753
```

Interpretation:

- Useful learning evidence when support exists.
- Not final evidence because coverage overshoot is loose and strict workload gates fail.

Strict 10-epoch probe:

```text
MLQDS QueryUsefulV1:           0.1664
Uniform QueryUsefulV1:         0.1981
Douglas-Peucker QueryUsefulV1: 0.2895
```

Interpretation:

- This is the more important warning.
- Strict workload stability can pass, but the trained candidate still loses and fails predictability, causality, and length preservation.

---

## Main Missing Work

### 1. Full 4x7 final grid is still missing

Final acceptance requires coverage targets:

```text
0.05, 0.10, 0.15, 0.30
```

and compression ratios:

```text
0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30
```

Do **not** run the full grid yet. It will likely fail. First get a strict support-valid single-cell probe to pass.

### 2. Strict workload evidence is missing

Loose overshoot probes do not count.

The accepted profile must pass:

```text
workload_stability_gate
workload_signature_gate
prior_sample_gate
predictability_gate
target_diffusion_gate
learning_causality_gate
global_sanity_gate
```

### 3. Predictability is still a hard blocker

Before more architecture tuning, prove train-derived priors predict held-out eval usefulness.

Required evidence:

```text
query_prior_predictability_audit.gate_pass = true
lift@1%, lift@2%, lift@5%, PR-AUC lift, Spearman/Kendall improve materially
sampled prior features are nonzero at eval points
eval points mostly lie inside train-prior support
```

### 4. Learning causality must pass under strict profile

The trained model must materially beat:

```text
MLQDS_shuffled_scores
MLQDS_untrained_model
MLQDS_prior_field_only_score
MLQDS_shuffled_prior_fields
MLQDS_without_query_prior_features
MLQDS_without_behavior_utility_head
MLQDS_without_segment_budget_head
```

Passing by tiny deltas is not enough. Keep the material delta floor.

### 5. Global sanity is unresolved

Endpoint retention now passes, but length preservation keeps failing.

Observed failures include:

```text
avg_length_preserved ~= 0.54 to 0.59
required minimum = 0.80
```

This cannot be patched by adding a huge temporal scaffold.

The selector/model must become better at preserving rough trajectory length and sanity while still being query-driven.

### 6. QueryUsefulV1 is still a bridge metric

Current `QueryUsefulV1` mostly remaps old range-audit components. It is useful, but it is not yet a true query-local behavior explanation metric.

Missing true query-local components:

```text
query-local interpolation error
query-local speed/heading reconstruction
query-local behavior explanation
query-local point-mass distribution preservation
query-local retained trajectory explanation
```

### 7. Segment-budget learning remains weak

The segment-budget head is wired, but ablation deltas are too small in current probes.

Likely reason:

```text
segment_budget_target is trained as repeated per-point BCE
selector uses mean segment scores
there is no explicit segment-level/listwise allocation loss
```

---

## Implemented Wrongly Or Weakly

### Issue A — No-prior ablation is confounded

Problem:

For `workload_blind_range_v2`, absolute features use the train query-prior extent when a prior field exists. If the prior field is removed, feature building falls back to the current eval point-set extent.

That means `MLQDS_without_query_prior_features` likely changes both:

```text
1. query-prior feature channels
2. absolute geo/time normalization frame
```

This makes the ablation ambiguous.

Fix:

Implement no-prior ablation by passing a zeroed prior-field object with the **same train extent and metadata**, not `None`.

Recommended implementation:

```python
def zero_query_prior_field_like(prior_field: dict[str, Any]) -> dict[str, Any]:
    zeroed = copy.deepcopy(prior_field)
    for name in QUERY_PRIOR_FIELD_NAMES:
        value = zeroed.get(name)
        if isinstance(value, torch.Tensor):
            zeroed[name] = torch.zeros_like(value)
    zeroed["ablation"] = "zero_query_prior_features"
    zeroed["contains_eval_queries"] = False
    zeroed["contains_validation_queries"] = False
    return zeroed
```

Then use this zeroed field in the no-prior ablation.

Acceptance test:

```text
No-query-prior ablation preserves:
- grid extent
- absolute feature normalization
- point_dim
- metadata provenance
Only prior channel values are zeroed.
```

### Issue B — Validation QueryUsefulV1 ignores global sanity inputs

Problem:

Validation scoring calls `query_useful_v1_from_range_audit(range_audit)` without passing:

```text
length_preservation
avg_sed_km
endpoint_sanity
```

The function defaults these to optimistic values. This means checkpoint selection can pick a model that looks good on validation `QueryUsefulV1` while failing the final global sanity gate.

Fix options:

Option A:

```text
Include length preservation, endpoint sanity, and SED in validation QueryUsefulV1.
```

Option B, preferred:

```text
Keep QueryUsefulV1 mostly query-local, but add validation checkpoint penalty for global sanity failure.
```

Recommended validation score:

```text
validation_selection_score =
    query_useful_v1
  - 0.10 * max(0, 0.80 - avg_length_preserved)
  - 0.05 * max(0, avg_sed_ratio_vs_uniform - allowed_sed_ratio)
  - 0.10 * endpoint_failure_penalty
```

Do not over-weight global geometry, but do not let checkpoint selection ignore it.

Acceptance test:

```text
A validation checkpoint with bad length preservation must score lower than an otherwise similar checkpoint with acceptable length preservation.
```

### Issue C — QueryUsefulV1 over-claims final metric maturity

Current implementation is a bridge over `RangeUseful` audit components.

Fix:

Either rename internally/documentation-wise:

```text
QueryUsefulV1Bridge
```

or add a clear field:

```json
"query_useful_v1_metric_maturity": "bridge_from_range_audit_components"
```

Then implement true query-local components in a later checkpoint.

### Issue D — Workload family semantics are partly fake

Current `range_workload_v1` names include:

```text
density_route
boundary_entry_exit
crossing_turn_change
port_or_approach_zone
sparse_background_control
route_corridor_like
```

But implementation currently maps some of these to the same density weights, and `route_corridor_like` does not appear to create actual elongated/corridor-like query boxes.

Fix one of two ways:

Option A, implement the semantics:

- `route_corridor_like`: elongated query boxes aligned to local trajectory direction.
- `port_or_approach_zone`: hotspot/endpoint/density zone that is actually distinct from generic density.
- `boundary_entry_exit`: endpoint/entry-exit biased.
- `crossing_turn_change`: turn/change biased.

Option B, rename them honestly:

```text
density_route
endpoint_or_boundary_biased
turn_change_biased
density_hotspot
sparse_background_control
```

Do not keep names that imply behavior not implemented.

### Issue E — Target-coverage generation can distort the intended distribution

Problem:

When target coverage is not yet reached, query generation anchors on uncovered points:

```python
anchor_mask = ~covered
```

This changes the workload distribution to chase uncovered points.

Fix:

For final-candidate workloads, prefer:

```text
sample queries from the profile distribution normally
calibrate query count to reach expected coverage
reject/regenerate workloads that miss coverage or drift signature
do not alter anchor distribution to chase uncovered points
```

If uncovered-point anchoring remains, mark it explicitly as part of the product workload profile and include it in signature diagnostics.

Recommended new config:

```text
coverage_calibration_mode:
  - profile_sampled_query_count
  - uncovered_anchor_chasing
```

Default for final acceptance should be:

```text
profile_sampled_query_count
```

### Issue F — `marginal_replacement_gain` is not true marginal gain

Current target uses a heuristic `query_value` and sparse replacement support. It is not a true counterfactual marginal set gain under `QueryUsefulV1`.

Fix:

Either rename it:

```text
replacement_representative_value
```

or implement true sampled marginal gain:

```text
For sampled train queries and budgets:
  build base retained mask
  for candidate additions/replacements:
    score QueryUsefulV1 delta
  aggregate positive deltas as target
```

### Issue G — Segment-budget head is trained too weakly

Current segment target is repeated per point and trained via auxiliary per-point BCE.

Fix:

Add segment-level/listwise training.

Recommended losses:

```text
segment_budget_bce_loss:
  mean pooled segment logit vs segment target

segment_budget_pairwise_rank_loss:
  high-value segment should outrank low-value segment within trajectory/day

segment_topk_mass_recall_loss:
  selected top-k segment logits should capture segment target mass under budget
```

Do not rely only on per-point BCE.

### Issue H — Global sanity is gated but not learned

The system detects length failures but does not really optimize against them.

Fix options:

1. Add query-free geometry/sanity auxiliary head:
   ```text
   length_preservation_value
   skeleton_value
   local_spacing_value
   ```
2. Add selector constraints:
   ```text
   per-trajectory minimum approximate length span
   min retained path-length ratio
   no large endpoint-to-endpoint shortcut without intermediate supports
   ```
3. Add validation penalty as above.

Do not add a large uniform temporal scaffold.

---

## Changed Assumptions

### Old assumption: same workload profile is enough

New assumption:

```text
Same workload profile is not enough. Final evidence also requires train/eval spatial-support overlap.
```

Add a first-class support-overlap gate.

### Old assumption: synthetic random routes are valid final tests

New assumption:

```text
Independent synthetic random-route splits are OOD diagnostics for spatial-prior learning.
Shared-route synthetic is a debugging proxy.
Final evidence should use real AIS held-out days with overlapping route support.
```

### Old assumption: strict coverage is a minor detail

New assumption:

```text
Strict coverage settings materially change the learning problem. Loose overshoot probes are debugging only.
```

### Old assumption: QueryUsefulV1 is done because weights changed

New assumption:

```text
Current QueryUsefulV1 is a bridge metric. It needs true query-local behavior components before final claims.
```

### Old assumption: global sanity is secondary

New assumption:

```text
Global sanity is either a real constraint or it should be relaxed. It cannot remain an after-the-fact gate the model never optimizes.
```

### Old assumption: a wired segment-budget head is useful

New assumption:

```text
A head is useful only when ablations show material retained-mask and QueryUsefulV1 impact under strict workloads.
```

---

## Required Next Checkpoint

Create:

```md
## Checkpoint 18 - Support-valid strict single-cell correction
```

Update:

```text
Range_QDS/docs/query-driven-rework-progress.md
```

with scope, changes, tests, probe results, failures, and next decision.

### Checkpoint 18 Scope

Do this before any full-grid benchmark.

1. Fix no-prior ablation confounding by preserving train extent with a zeroed prior field.
2. Add a train/eval support-overlap gate.
3. Fix validation checkpoint selection so global sanity is included or penalized.
4. Tighten `QueryUsefulV1` with at least one true query-local behavior component.
5. Clean workload-profile semantics or rename misleading query families.
6. Change final workload coverage generation away from uncovered-anchor chasing, or explicitly mark it as part of the profile.
7. Add segment-level/listwise loss for the segment-budget head.
8. Run one strict, support-valid, real-AIS or shared-route debug cell.

---

## Support-Overlap Gate

Add a gate:

```text
support_overlap_gate
```

Recommended fields:

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

If using real AIS with partial support, tune thresholds only after inspecting distributions. Do not loosen them just to pass a bad synthetic split.

The final claim should block on:

```text
support_overlap_gate
```

---

## True Query-Local Behavior Metric Upgrade

Current `QueryUsefulV1` maps old components. Add at least one real query-local behavior component before running the full grid.

Recommended first component:

```text
query_local_interpolation_fidelity
```

For each query and each ship trajectory segment inside the query:

1. Get original in-query points.
2. Get retained points that are inside the query, plus retained bracket points just outside the query if available.
3. Interpolate retained trajectory at original timestamps.
4. Compute average spatial error.
5. Normalize by query-local path length or mean segment length.
6. Convert to score:
   ```text
   1 / (1 + normalized_error)
   ```

Add this as a component to `QueryUsefulV1`.

Recommended weight change:

```text
query_local_interpolation_fidelity: 0.10
reduce old query_local_shape_score or proxy interpolation score accordingly
```

Do not remove old components yet; add a schema version bump.

---

## Segment-Budget Training Upgrade

Add segment-level training data derived from factorized targets.

Recommended module work:

```text
Range_QDS/training/query_useful_targets.py
Range_QDS/training/training_epoch.py
Range_QDS/models/workload_blind_range_v2.py
```

Implementation sketch:

1. In target builder, produce segment rows:
   ```text
   segment_start
   segment_end
   segment_target_mass
   segment_target_normalized
   trajectory_id
   ```
2. In training windows, map points to segment ids or build window-local segment targets.
3. Add loss:
   ```text
   segment_budget_head_loss =
       BCEWithLogits(mean(segment_budget_logits in segment), segment_target)
   ```
4. Add pairwise loss:
   ```text
   high-target segments should outrank low-target segments
   ```
5. Add diagnostics:
   ```text
   segment_head_tau
   segment_head_topk_mass_recall@budget
   no_segment_head_ablation_delta
   ```

Acceptance for this checkpoint:

```text
no_segment_budget_head_delta >= 0.005 under strict support-valid probe
```

---

## Validation Selection Fix

Current validation can ignore global sanity. Fix checkpoint selection.

Recommended:

```text
checkpoint_score_variant=query_useful_v1
checkpoint_selection_metric=uniform_gap
validation_global_sanity_penalty_enabled=true
```

Add config if needed:

```python
validation_global_sanity_penalty_weight: float = 0.10
validation_length_preservation_min: float = 0.80
```

Validation scoring should record:

```text
validation_query_useful_v1
validation_avg_length_preserved
validation_endpoint_sanity
validation_avg_sed_ratio_vs_uniform
validation_global_sanity_penalty
validation_selection_score_after_penalty
```

---

## Workload Profile Cleanup

### Fix or rename workload families

If implementing real semantics:

```text
route_corridor_like:
  elongated box aligned to local trajectory heading or local segment chord

port_or_approach_zone:
  density hotspot near endpoints / high-stop or low-speed clusters / approach zones

boundary_entry_exit:
  endpoint/box-boundary-biased anchors

crossing_turn_change:
  turn/change-biased anchors
```

If not implementing real semantics, rename families to honest names.

### Fix coverage calibration

Do not chase uncovered points silently in final profile.

Recommended profile field:

```text
coverage_calibration_mode = "profile_sampled_query_count"
```

Behavior:

```text
sample queries from profile distribution
measure coverage
if outside tolerance, regenerate with adjusted query count or fail
```

Keep uncovered-anchor chasing as diagnostic only:

```text
coverage_calibration_mode = "uncovered_anchor_chasing"
final_success_allowed = false
```

---

## Strict Probe To Run After Fixes

Run one strict support-valid probe before full grid.

Preferred: real AIS held-out days with overlapping route support.

If unavailable, shared-route synthetic debug is acceptable only as a debug probe.

Example strict debug shape:

```bash
python -m experiments.run_ais_experiment \
  --n_ships 6 \
  --n_points 48 \
  --synthetic_route_families 1 \
  --n_queries 8 \
  --max_queries 64 \
  --epochs 5 \
  --workload range \
  --query_coverage 0.30 \
  --range_max_coverage_overshoot 0.02 \
  --range_train_workload_replicates 4 \
  --model_type workload_blind_range_v2 \
  --range_training_target_mode query_useful_v1_factorized \
  --workload_profile_id range_workload_v1 \
  --selector_type learned_segment_budget_v1 \
  --checkpoint_score_variant query_useful_v1 \
  --checkpoint_selection_metric uniform_gap \
  --validation_score_every 1 \
  --checkpoint_full_score_every 1 \
  --checkpoint_candidate_pool_size 1 \
  --compression_ratio 0.20 \
  --range_duplicate_iou_threshold 1.0 \
  --range_acceptance_max_attempts 2000 \
  --final_metrics_mode core \
  --results_dir artifacts/results/query_driven_v2_checkpoint18_strict_probe
```

Modify flags as needed for real AIS.

Do not use `--range_max_coverage_overshoot 0.50` for acceptance evidence.

---

## Minimum Pass Condition Before Full Grid

Before running the full 4x7 grid, require a strict support-valid single-cell probe with:

```text
workload_stability_gate_pass = true
support_overlap_gate_pass = true
target_diffusion_gate_pass = true
prior_sample_gate_pass = true
predictability_gate_pass = true or near-pass with clear reason
workload_signature_gate_pass = true
learning_causality_gate_pass = true
global_sanity_gate_pass = true
MLQDS QueryUsefulV1 > uniform
MLQDS QueryUsefulV1 > DouglasPeucker
no-query-prior ablation is unconfounded
no-segment-head ablation has material delta
```

If this fails, do not run the full grid.

---

## Full Grid Acceptance

Only after the strict support-valid probe passes, run the final grid.

Required coverage targets:

```text
0.05, 0.10, 0.15, 0.30
```

Required compression ratios:

```text
0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30
```

Required report evidence:

```text
query_driven_final_grid_summary
final_claim_summary
QueryUsefulV1 scores
legacy RangeUseful diagnostics
uniform and Douglas-Peucker comparisons
predictability gate
workload stability gate
workload signature gate
support overlap gate
prior sample gate
target diffusion gate
learning causality gate
global sanity gate
runtime and latency
component metrics
```

Do not accept a grid if child gates fail.

---

## Tests To Add Or Update

Add focused tests for the discovered issues:

```text
test_no_query_prior_ablation_preserves_train_extent
test_zero_prior_field_like_preserves_metadata_and_shape
test_validation_query_useful_penalizes_bad_global_sanity
test_support_overlap_gate_blocks_out_of_extent_eval_points
test_support_overlap_gate_passes_same_support_eval_points
test_route_corridor_family_has_actual_corridor_semantics_or_is_not_final
test_final_profile_does_not_chase_uncovered_points_unless_declared
test_query_useful_v1_has_true_query_local_interpolation_component
test_segment_budget_head_has_segment_level_loss
test_no_segment_budget_head_ablation_has_neutral_segment_scores
```

Run:

```bash
cd Range_QDS
make lint
make typecheck
make test
git diff --check
```

---

## Files Most Likely To Need Changes

```text
Range_QDS/training/query_prior_fields.py
Range_QDS/training/model_features.py
Range_QDS/experiments/experiment_pipeline.py
Range_QDS/training/training_validation.py
Range_QDS/evaluation/query_useful_v1.py
Range_QDS/evaluation/evaluate_methods.py
Range_QDS/training/query_useful_targets.py
Range_QDS/training/training_epoch.py
Range_QDS/models/workload_blind_range_v2.py
Range_QDS/simplification/learned_segment_budget.py
Range_QDS/queries/workload_profiles.py
Range_QDS/queries/query_generator.py
Range_QDS/experiments/benchmark_report.py
Range_QDS/docs/query-driven-rework-progress.md
```

---

## Stop Conditions

Stop and diagnose instead of sweeping if any of these happen:

```text
strict support-valid probe loses to uniform
strict support-valid probe loses to Douglas-Peucker
predictability gate fails badly
support-overlap gate fails
no-prior ablation is still confounded
untrained or prior-only control beats trained model
no-segment-head delta is below material threshold
length preservation remains below 0.80
```

Do not compensate with:

```text
large temporal scaffold
query-conditioned inference
eval-query feature leakage
checkpoint selection on eval queries
geometry-label blending disguised as learning
KNN historical prior as final learned success
loose coverage overshoot
artificially easier workload profile
```

---

## Progress Log Requirement

At the end of the checkpoint, append to:

```text
Range_QDS/docs/query-driven-rework-progress.md
```

Use this structure:

```md
## Checkpoint 18 - Support-valid strict single-cell correction

Status: completed / partial / failed

Scope:
- ...

Changes:
- ...

Tests run:
- ...

Strict probe:
- command:
- artifact:
- MLQDS QueryUsefulV1:
- uniform QueryUsefulV1:
- Douglas-Peucker QueryUsefulV1:
- gates passed:
- gates failed:

Rejected experiments:
- ...

Current decision:
- ...
```

Be explicit if the checkpoint fails. A clear failure diagnosis is better than a fake success.

---

## Final Instruction To Agent

Prioritize correctness over optimism.

The current implementation already prevents many false claims. Keep that discipline. The next goal is not to get one nicer smoke number. The next goal is to produce a strict, support-valid, causally learned single-cell result that justifies running the full grid.
