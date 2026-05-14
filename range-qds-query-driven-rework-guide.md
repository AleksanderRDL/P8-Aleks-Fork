# Range-QDS Query-Driven Rework Guide

This document synthesizes the current redesign spec, the audit/progress artifacts, and the revised project direction. It is intended to guide a complete rework of the current Range-QDS implementation.

The central change is this:

> The goal is not generic workload-agnostic trajectory simplification. The goal is **query-driven, workload-blind simplification**: train from a statistically stable future-query distribution, compress before future eval queries are known, and preserve the points most likely to matter for future range-query answers under that distribution.

The model must actually learn. A result mostly caused by temporal scaffolding, query-conditioned inference, checkpoint leakage, or nonparametric lookup is not enough.

---

## 0. Revised Conclusion

The current implementation should not be treated as a failed dead end. It built much of the right protocol machinery and found a weak but real signal. But the current branch is optimizing the wrong final target too hard.

The previous setup treated the existing `RangeUseful` aggregate and broad held-out generator settings as fixed truth. That made the task drift toward generic temporal trajectory preservation. Your actual product target is narrower and more learnable:

1. Future query workloads are generated from a known statistical family.
2. Train and eval workloads should follow the same versioned workload profile, with held-out days and held-out seeds.
3. Compression must remain workload-blind at eval time: no eval query boxes, point/query relations, query embeddings, or eval-query-derived features before retained masks are frozen.
4. The learned model should preserve **query-local statistical point mass** and **query-local behavior explanation** much more than generic global geometry.
5. Global trajectory sanity remains important, but as a light guardrail, not the dominant objective.

The right next move is not another sweep over retained-frequency labels, KNN neighbor counts, temporal scaffold ratios, or minor selector tweaks. The right next move is to realign the workload generator, metric, targets, model, selector, and acceptance criteria around query-driven simplification.

---

## 1. Non-Negotiable Protocol Rules

These rules stay from the original redesign and should remain enforced by tests and artifact metadata.

### Allowed

- Train labels from generated or historical **training** range workloads.
- Query-aware teachers on training workloads.
- Historical/train-derived priors built before eval compression.
- Validation workloads for checkpoint selection, but only after blind validation masks are frozen.
- Held-out eval workloads for scoring only after eval masks are frozen.

### Forbidden for final claims

- Eval queries passed into the model, feature builder, selector, retained-set decision, checkpoint selector, or query-prior builder before compression.
- Eval point/query containment features.
- Eval query boundary-distance features.
- Query cross-attention at eval compression time.
- Checkpoint selection using final eval-query performance.
- Treating `range_aware` as final workload-blind success.

### Required artifact flags

Every serious run should record at least:

```yaml
workload_blind_protocol:
  enabled: true
  masks_frozen_before_eval_query_scoring: true
  eval_queries_seen_by_model: false
  eval_queries_seen_by_feature_builder: false
  eval_queries_seen_by_selector: false
  checkpoint_selected_on_eval_queries: false
  range_aware_used_as_final_model: false
```

---

## 2. What the Current Findings Actually Show

The current work produced a credible negative result and a useful baseline, not a final model.

### Current best branch

The strongest branch appears to be roughly:

```yaml
model_type: historical_prior
range_label_mode: usefulness_ship_balanced
range_train_workload_replicates: 4
range_replicate_target_aggregation: label_mean
mlqds_hybrid_mode: local_swap
mlqds_temporal_fraction: 0.85
mlqds_diversity_bonus: 0.02
train_days: 2026-02-02..2026-02-05
validation_day: 2026-02-06
eval_day: 2026-02-07
```

Main result:

| Comparison | Result |
|---|---:|
| Wins vs uniform | `17/28` |
| Wins vs Douglas-Peucker | `28/28` |
| Wins vs TemporalRandomFill | `18/28` |
| Low-budget wins vs uniform, `1%,2%,5%` | `4/12` |

Matched `5%` compression showed small but broad component gains at `30%` coverage:

| Component | MLQDS | Uniform |
|---|---:|---:|
| `RangePointF1` | `0.091175` | `0.089271` |
| `ShipF1` | `0.667008` | `0.655058` |
| `ShipCov` | `0.109885` | `0.104853` |
| `EntryExitF1` | `0.199307` | `0.193753` |
| `CrossingF1` | `0.101842` | `0.096723` |
| `TemporalCov` | `0.274301` | `0.260320` |
| `GapCov` | `0.252489` | `0.247608` |
| `TurnCov` | `0.099750` | `0.090915` |
| `ShapeScore` | `0.171505` | `0.159903` |

This suggests some real workload-blind signal exists. But it is shallow.

### Why this is not final learned success

The evidence does not support the claim that a trainable model has solved the task.

Reasons:

- The strongest branch is a historical-prior KNN-like method, not a clean learned compressor.
- The selector uses a heavy temporal scaffold (`0.85`).
- Clean learned stratified selector variants lose to uniform.
- Lower-scaffold variants degrade quality and geometry.
- Trainable `range_prior` variants fail target fit or fail held-out transfer.
- `historical_prior_student` fits labels better but trails the KNN branch and still misses low-budget cells.
- Pointwise MLP fitting improves training fit but worsens held-out usefulness.
- Opening learned low-budget slots exposes that the current blind score does not transfer.

Correct interpretation:

> Historical/query-prior signal exists, but the current scalar-target and temporal-scaffold formulation does not yet produce robust learned query-driven simplification.

---

## 3. Strategic Reframe

The project should now be framed as a contract between five things:

1. A versioned future-query workload profile.
2. A metric aligned with that workload profile.
3. Factorized train labels that represent query probability and behavior value separately.
4. A trainable workload-blind model that can learn stable query priors.
5. A selector where learned decisions materially control retained masks.

The train/eval query distributions must share statistical structure. Held-out eval should mean:

```text
same workload profile
held-out seed
held-out sampled query set
held-out AIS day/source
mild held-out jitter inside the profile
```

It should not mean arbitrary sparse/dense/background generator settings unless those are explicitly part of the product workload.

---

## 4. Point 1 — Define the Query Workload as a Product Object

The workload generator is not just a benchmark helper. It defines the prior the model learns.

Create a versioned workload profile, for example:

```yaml
workload_profile_id: range_workload_v1
profile_goal: query_driven_ais_simplification
coverage_targets: [0.05, 0.10, 0.15, 0.30]
compression_targets: [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
time_domain_mode: anchor_day
query_count_mode: calibrated_to_coverage
```

### Recommended anchor-family weights

Use this as the first concrete default unless real product query logs suggest otherwise:

```yaml
anchor_family_weights:
  density_route: 0.40
  boundary_entry_exit: 0.20
  crossing_turn_change: 0.15
  port_or_approach_zone: 0.15
  sparse_background_control: 0.10
```

Rationale:

- `density_route` should dominate because real future range queries are likely to concentrate around traffic corridors and high-activity areas.
- `boundary_entry_exit` and `crossing_turn_change` directly support query-local behavior explanations.
- `port_or_approach_zone` captures repeated AIS-relevant structure.
- `sparse_background_control` should exist, but not dominate, unless the product actually expects many sparse exploratory queries.

Do not set sparse/background high just to make the benchmark look broad. That makes uniform near-minimax and weakens the query-driven premise.

### Recommended footprint families

Start with a weighted mixture rather than one fixed box size:

```yaml
footprint_families:
  small_local:
    weight: 0.25
    spatial_radius_km: 1.0
    time_half_window_hours: 1.0
  medium_operational:
    weight: 0.45
    spatial_radius_km: 2.2
    time_half_window_hours: 5.0
  large_context:
    weight: 0.20
    spatial_radius_km: 4.5
    time_half_window_hours: 10.0
  route_corridor_like:
    weight: 0.10
    spatial_radius_km: 3.0
    time_half_window_hours: 6.0
    elongation_allowed: true
```

The existing `2.2 km / 5 h` setting should remain as the medium family, not the whole workload.

### Query-count calibration

Do not use a fixed query floor such as `24` for every coverage target. The audit already showed that this overshoots low coverage.

Recommended policy:

```yaml
coverage_calibration:
  pilot_workloads_per_split: 8
  query_count_search: binary_or_grid
  target_tolerance:
    coverage_05: 0.005
    coverage_10: 0.0075
    coverage_15: 0.010
    coverage_30: 0.020
  calibrate_per:
    - dataset_day
    - workload_profile_id
    - coverage_target
```

Coverage should be reached by changing query count, not by broadening the query distribution until it becomes a generic coverage problem.

### Workload acceptance filters

Recommended first defaults:

```yaml
query_acceptance:
  min_points_per_query: 3
  min_ships_per_query: 1
  max_individual_query_point_fraction:
    coverage_05: 0.020
    coverage_10: 0.025
    coverage_15: 0.030
    coverage_30: 0.050
  max_near_duplicate_hitset_jaccard: 0.65
  max_near_duplicate_fraction: 0.05
  max_empty_query_fraction: 0.00
  max_broad_query_fraction: 0.05
  max_attempt_multiplier: 20
```

Empty queries are not useful for training a compressor. If real product workloads include empty queries, track them separately as query system behavior, not as simplification supervision.

### Workload signature report

Every generated train/val/eval workload should emit:

```yaml
workload_signature:
  profile_id: range_workload_v1
  coverage_actual: ...
  query_count: ...
  anchor_family_counts: ...
  footprint_family_counts: ...
  point_hits_per_query: {p10: ..., p50: ..., p90: ...}
  ship_hits_per_query: {p10: ..., p50: ..., p90: ...}
  trajectory_hits_per_query: {p10: ..., p50: ..., p90: ...}
  time_span_hours_per_query: {p10: ..., p50: ..., p90: ...}
  spatial_radius_km_per_query: {p10: ..., p50: ..., p90: ...}
  near_duplicate_rate: ...
  broad_query_rate: ...
  empty_query_rate: ...
  train_eval_signature_distance: ...
```

Recommended gate:

```yaml
workload_signature_gate:
  anchor_family_l1_distance_max: 0.12
  footprint_family_l1_distance_max: 0.12
  point_hit_distribution_ks_max: 0.20
  ship_hit_distribution_ks_max: 0.20
  near_duplicate_rate_max: 0.05
  broad_query_rate_max: 0.05
```

If train and eval workload signatures drift beyond these bounds, label the result as out-of-profile diagnostic, not final acceptance evidence.

---

## 5. Point 2 — Add a Predictability Audit Before More Model Work

Before training another final model, answer this question:

> Can query-free train-derived features predict held-out eval query usefulness under `range_workload_v1`?

If the answer is no, more model training will mostly produce noise.

### Build diagnostic labels

For each point `p`, estimate from train workloads:

```text
H(p) = probability point p is inside a future query
B(p) = behavior usefulness if point p is queried
Y(p) = expected query usefulness = H(p) * B(p)
```

Build equivalent labels on validation/eval workloads only for diagnostics, after blind masks are frozen or in an offline predictability audit that is not used for checkpoint/final compression.

### Required predictability metrics

```yaml
predictability_audit:
  ranking:
    - spearman
    - kendall_tau
    - ndcg_at_budget_grid
  classification:
    - auc
    - pr_auc
  budget_lift:
    - lift_at_1_percent
    - lift_at_2_percent
    - lift_at_5_percent
    - lift_at_10_percent
  balance:
    - ship_balanced_lift
    - query_family_lift
    - coverage_target_lift
  degradation:
    - same_day_heldout_seed
    - next_day
    - multi_day_holdout
    - mild_profile_jitter
```

### Recommended go/no-go thresholds

Minimum to proceed with model work:

```yaml
minimum_predictability_gate:
  lift_at_1_percent: 1.10
  lift_at_2_percent: 1.15
  lift_at_5_percent: 1.20
  spearman_min: 0.15
  pr_auc_lift_over_base_rate: 1.25
```

Strong signal target:

```yaml
strong_predictability_target:
  lift_at_1_percent: 1.25
  lift_at_2_percent: 1.35
  lift_at_5_percent: 1.45
  spearman_min: 0.30
  pr_auc_lift_over_base_rate: 1.75
```

If the predictability audit fails, fix the workload profile, prior-field features, or target definition before touching model architecture.

---

## 6. Point 3 — The Current `RangeUseful` Is Misaligned

The current `RangeUseful` is useful as a diagnostic, but it is too general for the revised ambition.

It currently rewards global-ish temporal and shape preservation enough that uniform temporal sampling becomes extremely strong. That pushes the system toward temporal scaffolding instead of learned query-driven retention.

Do not silently mutate the existing metric. Create a new versioned primary metric:

```text
QueryUsefulV1
```

Keep reporting old `RangeUseful`, geometry distortion, length preservation, and component metrics as diagnostics.

Checkpointing and final acceptance should move to `QueryUsefulV1` after it is implemented and documented.

---

## 7. Point 4 — Define `QueryUsefulV1`

The new metric should match the actual product target:

1. Preserve statistical point mass inside likely future query ranges.
2. Preserve the retained points that best explain ship behavior inside those queries.
3. Preserve ship presence and boundary/event evidence.
4. Keep rough global trajectory sanity as a guardrail.

### Recommended top-level weights

```yaml
QueryUsefulV1:
  QueryPointMass: 0.40
  QueryLocalBehavior: 0.30
  ShipPresenceAndCoverage: 0.15
  BoundaryAndEventEvidence: 0.10
  GlobalSanity: 0.05
```

### Recommended component breakdown

```yaml
QueryPointMass:                  # total 0.40
  ShipBalancedQueryPointRecall: 0.18
  QueryBalancedPointRecall: 0.10
  QueryPointMassRatio: 0.07
  QueryPointDistributionStability: 0.05

QueryLocalBehavior:              # total 0.30
  QueryLocalInterpolationScore: 0.10
  QueryLocalTurnChangeCoverage: 0.07
  QueryLocalSpeedHeadingCoverage: 0.06
  QueryLocalShapeScore: 0.05
  QueryLocalGapContinuity: 0.02

ShipPresenceAndCoverage:         # total 0.15
  ShipF1: 0.08
  ShipCov: 0.05
  MultiPointShipEvidence: 0.02

BoundaryAndEventEvidence:        # total 0.10
  EntryExitF1: 0.04
  CrossingF1: 0.03
  QueryBoundaryEvidence: 0.03

GlobalSanity:                    # total 0.05
  EndpointOrSkeletonSanity: 0.02
  GlobalShapeGuardrailScore: 0.02
  LengthPreservationGuardrail: 0.01
```

### Query-local behavior definitions

Recommended first implementation:

```text
QueryLocalInterpolationScore:
  For each query and ship, compare original in-query positions against positions
  reconstructed/interpolated from retained points. Convert error to [0,1].

QueryLocalTurnChangeCoverage:
  Retained fraction of high heading-change / curvature / turn-score points inside query.

QueryLocalSpeedHeadingCoverage:
  Retained fraction of high acceleration, deceleration, speed-change, and heading-change points inside query.

QueryLocalShapeScore:
  Query-local version of shape preservation, not full-trajectory shape.

QueryLocalGapContinuity:
  Gap continuity inside query, using time/distance variants rather than only count-normalized gaps.
```

### Guardrail thresholds

Recommended initial final-report gates:

```yaml
global_sanity_gates:
  endpoints_retained_when_trajectory_budget_ge_2: true
  avg_sed_ratio_vs_uniform_max:
    compression_01: 2.00
    compression_02: 1.75
    compression_05_plus: 1.50
  length_preservation_range:
    min: 0.80
    max: 1.20
  catastrophic_geometry_outlier_fraction_max: 0.05
```

These should be reported separately from `QueryUsefulV1`. A model can be allowed to have worse global geometry than uniform, but not nonsense geometry.

---

## 8. Point 5 — Stop Training on One Scalar Retained-Frequency Label

The current scalar retained-frequency family is too compressed. It hides different concepts inside one value:

- probability of being queried;
- usefulness conditional on being queried;
- boundary/event role;
- ship coverage role;
- behavior explanation role;
- redundancy/marginal replacement value;
- segment/trajectory budget allocation.

That is why variants keep showing strong train fit but weak held-out usefulness.

### Replace with factorized labels

Recommended label heads:

```yaml
label_heads:
  query_hit_probability:
    target: P(point inside future query under train workloads)
    loss: bce_or_focal
    weight: 0.30

  conditional_behavior_utility:
    target: usefulness of point given it is inside a query
    loss: smooth_l1_plus_pairwise_rank
    weight: 0.25

  boundary_event_utility:
    target: entry_exit/crossing/query-boundary role
    loss: bce_plus_rank
    weight: 0.15

  marginal_replacement_gain:
    target: gain from replacing local skeleton/uniform candidate with this point
    loss: pairwise_rank_or_listwise_ndcg
    weight: 0.20

  segment_budget_target:
    target: expected query-local value mass for segment/window
    loss: kl_or_smooth_l1
    weight: 0.10
```

### Final scoring formula

Start with an explicit interpretable combination before learning an opaque final head:

```text
score(p) =
  query_hit_probability(p)
  * conditional_behavior_utility(p)
  * replacement_confidence(p)
  + boundary_event_bonus(p)
```

Recommended clipped form:

```text
score(p) = sigmoid(q_hit_logit)
         * sigmoid(behavior_logit)
         * sigmoid(replacement_logit)
         + 0.25 * sigmoid(boundary_event_logit)
```

Then add a learned calibration head only after this baseline is understood.

### Label mass diagnostics

Every target build should report:

```yaml
target_diagnostics:
  positive_fraction_by_head: ...
  label_mass_by_query_family: ...
  label_mass_by_ship: ...
  label_mass_by_segment_position: ...
  topk_label_mass_budget_grid: ...
  entropy_by_head: ...
  train_eval_label_drift: ...
```

If any label head has positive mass above roughly `50%` at low budgets, it is probably too diffuse for ranking.

---

## 9. Point 6 — The Final Model Should Not Be KNN

The historical-prior KNN branch is useful as a diagnostic and teacher. It should not be the final success path unless the project is explicitly renamed to nonparametric historical-prior retrieval.

Use KNN as:

- a teacher;
- a sanity-check prior;
- a feature-ablation reference;
- a source of training labels;
- a ceiling/floor diagnostic for query prior transfer.

Do not use KNN as the final model.

### Recommended final model type

Introduce a trainable model type:

```yaml
model_type: workload_blind_range_v2
```

Recommended architecture:

```yaml
workload_blind_range_v2:
  point_feature_encoder:
    hidden_dim: 128
    layers: 2
    activation: gelu
  local_context_encoder:
    type: temporal_conv_or_small_transformer
    hidden_dim: 128
    window_points: 32
    layers: 2
  segment_context_encoder:
    segment_size_points: 32_or_64
    hidden_dim: 128
    layers: 2
  prior_field_encoder:
    hidden_dim: 64
    layers: 2
  heads:
    - query_hit_probability
    - conditional_behavior_utility
    - boundary_event_utility
    - marginal_replacement_gain
    - segment_budget
    - final_score
```

Keep the first trainable version small. The current failures do not indicate that model size is the main bottleneck.

---

## 10. Point 7 — Preserve Absolute Geospatial Signal

This is one of the strongest implementation suspicions.

For query-driven simplification, the model must learn stable spatial and spatiotemporal priors. If the feature path normalizes away absolute location, the model cannot learn that future queries concentrate around specific ports, routes, corridors, crossings, or traffic zones.

### Required features

Recommended feature groups:

```yaml
absolute_position_features:
  - projected_x_global
  - projected_y_global
  - lat_global
  - lon_global
  - normalized_global_x_in_training_extent
  - normalized_global_y_in_training_extent

trajectory_local_features:
  - relative_position_in_trajectory
  - local_time_delta
  - local_distance_delta
  - speed
  - acceleration
  - course_or_heading
  - heading_change
  - curvature_or_turn_score
  - gap_duration
  - gap_distance
  - endpoint_flags

traffic_context_features:
  - local_point_density_train_field
  - local_ship_density_train_field
  - route_density_train_field
  - historical_query_hit_prior
  - historical_boundary_prior
  - historical_crossing_prior
  - historical_behavior_utility_prior

time_features:
  - hour_of_day_sin_cos
  - day_of_week_sin_cos_if_relevant
  - source_day_relative_time
```

### Normalization rule

Do not normalize lat/lon or projected coordinates per trajectory or per eval split in a way that destroys cross-day comparability.

Recommended policy:

```yaml
normalization:
  coordinate_frame: fixed_training_extent_or_epsg_projection
  fit_on: training_split_only
  apply_to: train_val_eval
  per_trajectory_relative_features: allowed_as_additional_features
  absolute_features_removed: false
```

Add a test that verifies two identical geographic points from different days produce the same absolute geo features.

---

## 11. Point 8 — Add a Train-Derived Query Prior Field

This is probably the highest-value change.

Build a query-prior field from training workloads only. At eval compression time, the model may use this field because it contains no eval-query information.

### Recommended prior fields

```yaml
train_derived_prior_fields:
  spatial_query_hit_probability:
    grid_resolution_km: 0.25
    smoothing_sigma_km: 0.75
  spatiotemporal_query_hit_probability:
    grid_resolution_km: 0.50
    time_bins: 24_hourly_or_6_four_hour_bins
    smoothing_sigma_km: 1.00
  boundary_entry_exit_likelihood:
    grid_resolution_km: 0.25
    smoothing_sigma_km: 0.75
  crossing_likelihood:
    grid_resolution_km: 0.25
    smoothing_sigma_km: 0.75
  behavior_utility_prior:
    grid_resolution_km: 0.25
    smoothing_sigma_km: 0.75
  route_density_prior:
    grid_resolution_km: 0.25
    smoothing_sigma_km: 0.75
```

### Prior-field artifact

Each prior field should have a schema-versioned cache:

```yaml
prior_field_artifact:
  profile_id: range_workload_v1
  built_from_split: train_only
  train_csvs: ...
  train_workload_seed: ...
  grid_projection: ...
  grid_resolution_km: ...
  smoothing: ...
  contains_eval_queries: false
  contains_validation_queries: false_for_final_eval_models
```

### Model usage

For each point, sample the prior fields at that point's coordinate and time bin. These become features, not final hand-coded decisions.

Required ablations:

```yaml
prior_field_ablations:
  trained_model_with_prior_fields: required
  trained_model_without_prior_fields: required
  prior_fields_only_score: diagnostic
  shuffled_prior_fields: diagnostic
```

A real query-driven model should lose query-local performance when prior fields are removed or shuffled.

---

## 12. Point 9 — Replace the 85% Temporal Scaffold with Learned Budget Allocation

The current `mlqds_temporal_fraction=0.85` is too scaffold-dominated. It protects the result but makes learned behavior marginal.

The selector should become:

```text
minimal rough skeleton
+ learned trajectory/segment budget allocation
+ learned within-segment point selection
+ local non-redundancy/diversity control
```

Not:

```text
85% uniform temporal scaffold + tiny learned residual
```

And not:

```text
global top-k points with no trajectory sanity
```

### Recommended selector

Introduce:

```yaml
selector_type: learned_segment_budget_v1
```

Algorithm:

1. Split trajectories into fixed-size or adaptive segments.
2. Predict segment query value from the segment-budget head.
3. Allocate budget across segments globally or per batch using predicted segment value.
4. Within selected segments, pick points by final learned score.
5. Apply non-maximum suppression / spacing to avoid redundant clusters.
6. Retain minimal skeleton points only where needed for rough trajectory sanity.

### Recommended skeleton policy

```yaml
minimal_skeleton_policy:
  endpoint_retention:
    if_trajectory_budget_ge_2: true
    if_trajectory_budget_eq_1: choose_best_of_endpoint_midpoint_or_high_score
  max_skeleton_fraction_by_compression:
    compression_01: 0.50
    compression_02: 0.40
    compression_05: 0.25
    compression_10: 0.20
    compression_15_plus: 0.15
  no_fixed_85_percent_temporal_scaffold: true
```

This gives low-budget trajectories some sanity without allowing temporal scaffolding to dominate the whole retained set.

### Budget allocation details

Recommended first implementation:

```yaml
budget_allocation:
  budget_scope: dataset_global_with_trajectory_guards
  segment_score: predicted_segment_query_value
  segment_score_temperature: 0.75
  minimum_selected_trajectories_policy: query_prior_weighted
  per_ship_fairness_guard:
    enabled: true
    max_budget_share_per_ship: 0.20
  non_redundancy:
    min_temporal_spacing_fraction_within_segment: 0.10
    min_spatial_spacing_km: 0.10
    suppress_same_local_peak: true
```

The key is that the model should reallocate budget toward likely queried regions/ships/segments. Uniform wastes budget equally where future queries are unlikely.

---

## 13. Point 10 — Prove the Model Actually Learned

Baseline comparisons can stay limited to uniform and Douglas-Peucker for the product claim. But causal ablations are still required to prove learning.

These are not extra product baselines. They are learning-evidence checks.

### Required learning-causality report

Every final candidate should report:

```yaml
learning_causality:
  learned_controlled_retained_slots: ...
  learned_controlled_retained_slot_fraction: ...
  trajectories_with_at_least_one_learned_decision: ...
  trajectories_with_zero_learned_decisions: ...
  segment_budget_entropy: ...
  delta_vs_untrained_model: ...
  delta_vs_shuffled_scores: ...
  delta_vs_shuffled_prior_fields: ...
  delta_vs_prior_field_only_score: ...
  delta_without_query_prior_features: ...
  delta_without_behavior_utility_head: ...
  delta_without_segment_budget_head: ...
```

### Recommended gates

Minimum credible learned contribution:

```yaml
learned_contribution_gates:
  compression_01:
    report_only_due_to_budget_rounding: true
    learned_slots_required_when_budget_permits: true
  compression_02:
    trajectories_with_learned_decision_min_when_len_ge_200: 0.25
  compression_05:
    learned_controlled_slot_fraction_min: 0.25
    trajectories_with_learned_decision_min: 0.50
  compression_10_plus:
    learned_controlled_slot_fraction_min: 0.35
    trajectories_with_learned_decision_min: 0.70
  shuffled_scores_should_lose_fraction_of_uniform_gap: 0.60
  untrained_model_should_not_match_trained_model: true
```

A run where the model controls only `8%` of retained slots at `5%` compression should not count as learned success.

---

## 14. Point 11 — Reinterpret Held-Out Workload Generalization

Final evaluation should test in-distribution generalization first:

```text
same workload profile
held-out AIS days
held-out seeds
held-out sampled query sets
mild profile jitter
```

Out-of-distribution settings should remain diagnostics, not final failure criteria, unless they are part of the intended product.

### Recommended eval tiers

```yaml
evaluation_tiers:
  tier_1_final_acceptance:
    - same_profile_heldout_seed
    - same_profile_heldout_day
    - same_profile_multi_day_holdout
  tier_2_robustness:
    - mild_anchor_weight_jitter_plus_minus_10_percent
    - mild_footprint_weight_jitter_plus_minus_10_percent
    - query_count_recalibrated_to_coverage
  tier_3_ood_diagnostic:
    - dense_only
    - sparse_background_heavy
    - broad_query_heavy
    - different_time_window_family
```

A dense-only or sparse/background-heavy result should not define final success if the product workload is `range_workload_v1`.

---

## 15. Point 12 — Concrete Rework Roadmap

Do this in phases. Do not return to open-ended agent sweeps until the gates are in place.

### Phase 1 — Freeze legacy branch

Mark the current best branch as a legacy diagnostic:

```yaml
legacy_branch:
  name: historical_prior_shipbalanced_localswap085
  role:
    - diagnostic_baseline
    - teacher_candidate
    - regression_reference
  not_final_success: true
```

Do not spend more effort tuning:

- KNN neighbor count;
- source-day agreement aggregation;
- local-swap temporal fraction;
- min learned swaps around current KNN score;
- retained-frequency budget weighting;
- pointwise MLP imitation of KNN;
- scalar structural target blends around the current metric.

### Phase 2 — Implement `range_workload_v1`

Deliverables:

- versioned workload profile config;
- anchor-family weights;
- footprint-family weights;
- query-count calibration;
- workload signature reports;
- signature drift gates;
- held-out seed/day generator support;
- mild jitter mode;
- OOD diagnostic modes clearly separated.

### Phase 3 — Implement `QueryUsefulV1`

Deliverables:

- new metric module;
- query-local behavior components;
- query-local gap time/distance support;
- primary aggregate weights;
- global sanity guardrails;
- report both `QueryUsefulV1` and old `RangeUseful`.

### Phase 4 — Predictability audit

Deliverables:

- train-derived labels for `H(p)`, `B(p)`, `Y(p)`;
- held-out diagnostic labels;
- top-k lift reports at all compression targets;
- ship-balanced and query-family-balanced lift;
- fail-fast gate before architecture work.

### Phase 5 — Prior-field builder

Deliverables:

- train-only spatial/spatiotemporal query prior fields;
- boundary/crossing/behavior prior fields;
- schema-versioned caches;
- feature sampling path;
- no eval-query contamination tests.

### Phase 6 — Factorized model

Deliverables:

- `workload_blind_range_v2` model;
- factorized heads;
- multi-head losses;
- learned final score;
- feature ablations;
- KNN teacher optional but not final.

### Phase 7 — Learned selector

Deliverables:

- `learned_segment_budget_v1` selector;
- minimal skeleton policy;
- learned segment budget allocation;
- non-redundancy control;
- learned-slot accounting;
- shuffled-score and untrained-model ablations.

### Phase 8 — Final evaluation

Deliverables:

- full coverage grid: `5%,10%,15%,30%`;
- full compression grid: `1%,2%,5%,10%,15%,20%,30%`;
- uniform and Douglas-Peucker comparisons;
- old `RangeUseful` diagnostic;
- geometry/length/runtime/latency reports;
- learning-causality report;
- held-out seeds/days/profile jitter.

---

## 16. Point 13 — Revised Acceptance Criteria

Recommended new acceptance criteria:

```text
A workload-blind trained model compresses trajectories before eval queries are known.

Train and eval workloads are generated from the same versioned workload profile,
using held-out AIS days and held-out seeds.

The model is evaluated across coverage targets 5%, 10%, 15%, 30% and compression
targets 1%, 2%, 5%, 10%, 15%, 20%, 30%.

Primary score is QueryUsefulV1, which prioritizes query-local point mass and
behavior explanation inside likely future query ranges. Old RangeUseful and
global geometry are reported as diagnostics.

The model beats uniform temporal sampling and Douglas-Peucker on QueryUsefulV1
across most grid cells, with special attention to 1%, 2%, and 5% compression.

The model has causal learning evidence: trained scores outperform shuffled and
untrained scores, learned-controlled retained slots are material, and removing
query-prior features reduces query-local performance.

Global geometry may be rougher than uniform, but must pass explicit sanity
thresholds.
```

### Recommended numeric success bars

Minimum credible success:

```yaml
minimum_success:
  beats_uniform_queryuseful_cells_min: 19   # out of 28
  beats_dp_queryuseful_cells_min: 24        # out of 28
  low_budget_beats_uniform_cells_min: 7     # out of 12 for 1/2/5%
  matched_5_percent_coverage_cells_uniform_min: 3 # out of 4 coverage targets
  trained_vs_shuffled_score_delta_positive: true
  learned_contribution_gates_pass: true
  global_sanity_gates_pass: true
```

Target success:

```yaml
target_success:
  beats_uniform_queryuseful_cells_min: 22   # out of 28
  beats_dp_queryuseful_cells_min: 28        # out of 28
  low_budget_beats_uniform_cells_min: 9     # out of 12
  matched_5_percent_coverage_cells_uniform_min: 4 # out of 4
  old_rangeuseful_reported_not_required_to_win_all: true
  trained_vs_shuffled_score_delta_fraction_of_gain_min: 0.60
  no_query_leakage_audit_pass: true
```

Stretch success:

```yaml
stretch_success:
  beats_uniform_queryuseful_cells_min: 25
  low_budget_beats_uniform_cells_min: 10
  mild_profile_jitter_still_above_uniform: true
  latency_better_than_historical_knn_branch: true
```

---

## 17. Point 14 — What the Previous Work Was Probably Tunnel-Visioned On

The previous work was valuable, but it stayed too long inside the old framing.

### Main tunnel-vision pattern

```text
new scalar label
new retained-frequency variant
new target blend
new temporal scaffold ratio
new KNN knob
new local-swap selector tweak
```

The recurring failures indicate that the bottleneck is not one missing blend coefficient.

### Specific traps to avoid

1. Treating old `RangeUseful` as the final truth even when it over-rewards generic temporal preservation.
2. Treating broad held-out generator settings as final requirements instead of OOD diagnostics.
3. Treating DP wins as meaningful enough. Uniform is the main baseline.
4. Treating temporal scaffold protection as model success.
5. Treating target fit as eval usefulness.
6. Treating KNN historical prior as a final learned model.
7. Treating pointwise scalar labels as sufficient for a set-level coverage objective.
8. Treating low-budget wins as real when learned slots are zero.
9. Treating source/MMSI identity as likely to fix the core problem.
10. Treating selector-only changes as likely to solve weak learned signal.

The next implementation should force the model to win through learned query-prior and behavior-value prediction.

---

## 18. Point 15 — Code Areas to Inspect and Rework

Highest-priority files or equivalent modules:

### Workload generation

```text
QDS/queries/range_workloads.py
QDS/queries/query_generator.py
```

Inspect/rework:

- whether generator defines a stable workload family;
- anchor-family mixing;
- footprint-family mixing;
- query-count calibration;
- acceptance filters;
- duplicate/broad-query control;
- `anchor_day` behavior;
- workload signature reporting;
- held-out seed/day/profile-jitter handling.

### Metrics

```text
QDS/evaluation/range_usefulness.py
QDS/evaluation/range_metrics.py
QDS/evaluation/benchmark_metrics.py
```

Inspect/rework:

- exact old `RangeUseful` weights;
- query-local point mass metrics;
- query-local behavior reconstruction/interpolation;
- time/distance `GapCov` variants;
- global sanity guardrails;
- primary metric switching to `QueryUsefulV1`;
- reporting old `RangeUseful` as diagnostic.

### Target construction

```text
QDS/training/training_targets.py
QDS/training/range_targets.py
QDS/training/teacher_distillation.py
```

Inspect/rework:

- retained-frequency label construction;
- positive label diffusion;
- ship-balanced label behavior;
- target mass by query family;
- factorized label heads;
- marginal replacement labels;
- segment-budget labels;
- train/eval diagnostic label separation.

### Feature builder and prior fields

```text
QDS/training/model_features.py
QDS/training/feature_builder.py
QDS/training/query_prior_fields.py   # recommended new module
```

Inspect/rework:

- absolute geo feature preservation;
- train-only normalization;
- route/density features;
- prior-field cache schema;
- eval-query leakage tests;
- feature ablations.

### Model path

```text
QDS/models/range_prior*.py
QDS/models/workload_blind_range_v2.py # recommended new model
QDS/models/historical_prior*.py
```

Inspect/rework:

- trainable factorized heads;
- local/segment context;
- prior-field encoder;
- KNN used as teacher/diagnostic only;
- pointwise MLP path not treated as final.

### Selector/simplification

```text
QDS/simplification/simplify_trajectories.py
QDS/simplification/mlqds_scoring.py
QDS/simplification/selectors.py
QDS/simplification/learned_segment_budget.py # recommended new module
```

Inspect/rework:

- temporal scaffold ratio;
- endpoint/minimal skeleton handling;
- learned slot accounting;
- segment budget allocation;
- non-redundancy suppression;
- low-budget rounding;
- global vs per-trajectory budget allocation.

### Benchmark/reporting

```text
QDS/experiments/benchmark_runner.py
QDS/experiments/benchmark_profiles.py
QDS/experiments/benchmark_report.py
QDS/experiments/experiment_pipeline.py
```

Inspect/rework:

- profile IDs;
- frozen-mask sequencing;
- primary metric selection;
- query-prior artifact provenance;
- learned-causality reports;
- full grid propagation;
- workload signature drift reports;
- separation of final acceptance vs OOD diagnostics.

---

## 19. First-Principles Risks and Mitigations

### Risk 1 — No stable query signal exists

If future queries are too broad, too random, or too different across days, uniform will be hard to beat.

Mitigation:

- define `range_workload_v1` as a real product prior;
- run predictability audit before model sweeps;
- use train-derived spatial/spatiotemporal prior fields.

### Risk 2 — The metric rewards uniform too much

If the metric overweights temporal coverage and global shape, the model will hide behind scaffolding or lose.

Mitigation:

- move final objective to `QueryUsefulV1`;
- keep global geometry as guardrail;
- report old `RangeUseful` separately.

### Risk 3 — Low budgets have too little discretionary capacity

At `1%` and `2%`, endpoints/sanity can consume most of the budget.

Mitigation:

- use global/segment budget allocation;
- report learned slots explicitly;
- permit rougher global geometry;
- evaluate whether learned decisions exist where budget permits.

### Risk 4 — Pointwise scoring cannot solve a set objective

Multiple high-scoring points can be redundant. A lower point may be better if it covers an uncovered ship, time span, or behavior event.

Mitigation:

- segment-budget head;
- marginal replacement labels;
- non-redundancy selector;
- learned budget allocation.

### Risk 5 — Train labels diffuse across too many points

Averaging many generated workloads can make labels smooth but useless for low-budget ranking.

Mitigation:

- factorized heads;
- query-family-balanced labels;
- top-k/listwise losses;
- target entropy and label mass diagnostics.

### Risk 6 — Nonparametric historical prior looks better than learned models

KNN can memorize spatial priors and still not count as a learned model.

Mitigation:

- use KNN as teacher/reference only;
- require trainable final model;
- require trained vs untrained/shuffled ablations.

---

## 20. Recommended Benchmark Profiles

### Profile A — Predictability audit

```yaml
profile: range_workload_v1_predictability_audit
model: none_or_simple_diagnostic
outputs:
  - train_eval_label_lift
  - topk_lift_grid
  - query_family_lift
  - ship_balanced_lift
  - signature_drift
success_gate: minimum_predictability_gate
```

### Profile B — Prior-field only diagnostic

```yaml
profile: range_workload_v1_priorfield_only
model: prior_field_score
selector: learned_segment_budget_v1_or_simple_topk_with_sanity
purpose: prove train-derived query field contains useful signal
final_success_allowed: false
```

### Profile C — Trainable factorized model

```yaml
profile: range_workload_v1_workload_blind_v2
model: workload_blind_range_v2
selector: learned_segment_budget_v1
primary_metric: QueryUsefulV1
baselines:
  - uniform
  - douglas_peucker
causal_ablations:
  - untrained_model
  - shuffled_scores
  - no_prior_fields
  - no_behavior_head
  - no_segment_budget_head
```

### Profile D — Mild profile jitter

```yaml
profile: range_workload_v1_mild_jitter
jitter:
  anchor_family_weights: +/-0.10
  footprint_family_weights: +/-0.10
  query_count: recalibrated_to_coverage
purpose: robustness inside product family
```

### Profile E — OOD diagnostic only

```yaml
profile: range_workload_v1_ood_diagnostics
settings:
  - dense_only
  - sparse_background_heavy
  - broad_query_heavy
final_success_allowed: false
```

---

## 21. Stop/Continue Rules

### Stop current line if

```yaml
stop_if:
  predictability_lift_at_5_percent_below: 1.20
  trained_model_not_better_than_shuffled_scores: true
  learned_slot_fraction_at_5_percent_below: 0.25
  query_prior_field_ablation_no_effect: true
  only_dp_is_beaten_not_uniform: true
  wins_depend_on_temporal_scaffold_above: 0.50
```

### Continue only if

```yaml
continue_if:
  predictability_gate_passes: true
  QueryUsefulV1_uniform_gap_positive_on_validation: true
  learned_causality_ablation_passes: true
  low_budget_has_real_learned_slots_where_budget_permits: true
  workload_signature_drift_within_gate: true
```

---

## 22. Implementation Checklist

### Workload

- [ ] Add `range_workload_v1` profile.
- [ ] Add anchor-family weights.
- [ ] Add footprint-family weights.
- [ ] Add query-count calibration to coverage.
- [ ] Add acceptance filters.
- [ ] Add workload signature reports.
- [ ] Add train/eval signature drift gates.
- [ ] Separate final, jitter, and OOD profiles.

### Metric

- [ ] Add `QueryUsefulV1`.
- [ ] Add `QueryPointMass` components.
- [ ] Add `QueryLocalBehavior` components.
- [ ] Add query-local interpolation score.
- [ ] Add query-local speed/heading/turn coverage.
- [ ] Add global sanity guardrails.
- [ ] Keep old `RangeUseful` diagnostic.

### Labels

- [ ] Add `query_hit_probability` head target.
- [ ] Add `conditional_behavior_utility` head target.
- [ ] Add `boundary_event_utility` head target.
- [ ] Add `marginal_replacement_gain` head target.
- [ ] Add `segment_budget_target`.
- [ ] Add target mass/entropy diagnostics.

### Features

- [ ] Preserve absolute geo features.
- [ ] Add train-only normalization tests.
- [ ] Add route/density context features.
- [ ] Add train-derived query prior fields.
- [ ] Add prior-field cache provenance.
- [ ] Add no-eval-query contamination tests.

### Model

- [ ] Add `workload_blind_range_v2`.
- [ ] Add factorized heads.
- [ ] Add prior-field encoder.
- [ ] Add local/segment context encoder.
- [ ] Add multi-head losses.
- [ ] Keep KNN only as teacher/diagnostic.

### Selector

- [ ] Add `learned_segment_budget_v1`.
- [ ] Add minimal skeleton policy.
- [ ] Add segment budget allocation.
- [ ] Add non-redundancy suppression.
- [ ] Add learned-slot accounting.
- [ ] Add shuffled/untrained score ablations.

### Evaluation

- [ ] Run full coverage/compression grid.
- [ ] Compare only against uniform and DP for final product claim.
- [ ] Report causal ablations separately.
- [ ] Report old `RangeUseful`, geometry, length, runtime, latency.
- [ ] Report held-out seed/day/profile-jitter results.
- [ ] Report OOD diagnostics separately.

---

## 23. Final Practical Recommendation

The rework should start with workload and metric alignment, not model architecture.

Recommended order:

1. Implement `range_workload_v1`.
2. Implement `QueryUsefulV1`.
3. Run predictability audit.
4. Add train-derived query prior fields.
5. Verify absolute geo features survive the feature path.
6. Train `workload_blind_range_v2` with factorized heads.
7. Use `learned_segment_budget_v1` selector.
8. Prove learned contribution with ablations.
9. Run final uniform/DP comparison.

This order matters. If the workload profile and metric do not expose a stable query-driven signal, the model cannot learn the intended behavior. If the selector still hides behind temporal scaffolding, wins will not prove model learning. If the feature path removes absolute spatial priors, the model cannot learn the query distribution you want it to exploit.

The ambition is realistic only if the future query workload family is explicit and stable. Under that condition, the model can learn to preserve points likely to matter for future queries. Under arbitrary future workloads, uniform temporal sampling will often remain too strong.
