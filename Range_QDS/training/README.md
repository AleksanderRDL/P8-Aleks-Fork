# Training Module

Builds supervision, batches trajectory windows, trains MLQDS, selects
checkpoints, and persists model artifacts.

## Files

| File | Purpose |
| --- | --- |
| `importance_labels.py` | Per-point workload labels and labelled masks. |
| `trajectory_batching.py` | Fixed-length windows that do not cross trajectory boundaries. |
| `scaler.py` | Persisted min-max scaler. |
| `checkpoint_selection.py` | Validation score and uniform-gap selection helpers. |
| `checkpoints.py` | Save/load `ModelArtifacts`. |
| `inference.py` | Deterministic persisted-model prediction. |
| `training_losses.py` | Budget, ranking, pointwise, and temporal-residual losses. |
| `training_targets.py` | Legacy RangeUseful/scalar target transforms and dispatch constants. |
| `query_useful_targets.py` | Placeholder for future factorized QueryUsefulV1 labels. |
| `query_prior_fields.py` | Placeholder for train-derived query-prior fields. |
| `factorized_target_diagnostics.py` | Placeholder for future factorized-label diagnostics. |
| `training_diagnostics.py` | Target diagnostics and rank-correlation helpers. |
| `training_epoch.py` | One-epoch forward/loss/backward optimization pass. |
| `training_validation.py` | Validation scoring for checkpoint selection. |
| `training_setup.py` / `training_windows.py` | Setup, workload, and trajectory-window helpers. |
| `training_outputs.py` | Shared training result dataclass. |
| `train_model.py` | Main training loop and checkpoint selection flow. |
| `training_pipeline.py` | Compatibility re-exports. |

## Current Flow

The implemented strong path is workload-aware:

1. Generate a pure workload and point labels.
2. Fit `FeatureScaler` on training points/query features.
3. Train a query-conditioned model on padded trajectory windows.
4. Select checkpoints by validation score or `uniform_gap`.
5. Restore the best checkpoint for final evaluation.

This is not the final workload-blind protocol. The redesign target is in
[`../docs/query-driven-rework-guide.md`](../docs/query-driven-rework-guide.md).

## Labels

- `range_label_mode="usefulness"`: current range label default. It approximates
  the `RangeUseful` audit components, not only in-box point hits.
- `range_label_mode="usefulness_balanced"`: rescales component mass toward the
  audit weights when raw support frequency dominates.
- `range_label_mode="usefulness_ship_balanced"`: keeps the same component
  family but normalizes point, entry/exit, and crossing support by query-hit
  ship before aggregation. This is a training-only diagnostic for low-budget
  ship and point coverage failures.
- `range_label_mode="point_f1"`: old point-hit ablation.
- `range_boundary_prior_weight`: optional explicit prior; diagnostic profile
  keeps it at `0.0`.

Labels are additive training targets. They are not exact retained-set optimizers.

Legacy training target transforms:

All current `range_training_target_mode` values except
`query_useful_v1_factorized` are legacy RangeUseful/scalar diagnostics. They
are useful for regression and teacher experiments, but they are not
final-success eligible for the query-driven rework.

- `range_training_target_mode="retained_frequency"`: converts usefulness labels
  into oracle retained-set membership frequency across configured budgets.
  Zero-usefulness points are not counted as retained targets when a trajectory
  budget has spare slots; temporal target blending is the explicit way to add
  query-blind temporal anchors.
- `range_training_target_mode="global_budget_retained_frequency"`: converts
  usefulness labels into training-only database-level budget winners. It keeps
  the same endpoint skeleton as the global-budget diagnostic, then labels the
  useful points that win the remaining global score competition. Eval
  compression still receives only query-free point features.
- `range_target_budget_weight_power`: optional training-only weighting for
  retained-frequency-style targets. `0.0` averages budgets uniformly. Positive
  values weight smaller compression ratios as `ratio ** -power` before
  normalization, which is useful for diagnosing low-budget failures without
  changing eval compression or using eval queries.
- `range_training_target_mode="marginal_coverage_frequency"`: consumes nearby
  label mass after each selected point to reduce redundant hotspot targets.
- `range_training_target_mode="structural_retained_frequency"`: blends train
  workload usefulness with query-free structural scores before retained-target
  selection. The structural score uses trajectory-local uniqueness, turn/gap
  signals, endpoint support, and centroid-based globality; it is training-only
  and must be treated as a regularizer, not inference-time geometry blending.
  `range_structural_target_blend` controls the structural weight.
  `range_structural_target_source_mode="blend"` adds structural support before
  retained-set selection. `"boost"` preserves train-usefulness support and only
  re-ranks useful points by structural prominence, which is the cleaner option
  when additive support diffuses labels too broadly.
- `range_training_target_mode="query_spine_frequency"`: training-only
  query-aware target that labels temporal support anchors inside train-query
  slices. `range_query_spine_mass_mode="hit_group"` preserves the legacy
  equal-mass `(query, trajectory-hit)` behavior; `"query"` gives each train
  query unit mass split across its hit trajectories. Eval compression remains
  query-blind.
- `range_training_target_mode="query_residual_frequency"`: training-only
  query-aware target that simulates the temporal base, then labels residual
  anchors inside train-query slices. `range_query_residual_mass_mode="query"`
  gives each train query equal mass; `"point"` keeps selected-anchor frequency
  mass. Eval compression remains query-blind.
- `range_training_target_mode="set_utility_frequency"`: training-only
  query-aware teacher that scores each residual candidate by one-step marginal
  train-query `RangeUseful` gain from the temporal base. This is slower but
  tests whether exact set-utility signal is learnable by the blind student.
- `range_training_target_mode="local_swap_gain_cost_frequency"`: training-only
  query-aware teacher for `mlqds_hybrid_mode="local_delta_swap"`. It labels
  non-base candidate value and paired temporal-anchor removal cost on the same
  scale, so the learned score can decide whether an override should beat the
  uniform anchor it would remove. Eval compression remains query-blind. Treat
  this as diagnostic; the first historical-prior run fit the train target but
  lost every uniform audit cell.
- `range_target_balance_mode="trajectory_unit_mass"`: optional training-only
  diagnostic that rescales each train trajectory's positive range-target mass
  to one after target construction. It tests whether historical priors are
  over-dominated by a few dense routes. It does not change eval compression or
  expose eval queries.

## Loss And Selection

- `loss_objective="budget_topk"`: default range loss. It trains score mass into
  top-k budgets from `budget_loss_ratios`. The direct CLI default is the active
  audit grid: `0.01,0.02,0.05,0.10,0.15,0.20,0.30`.
- `loss_objective="stratified_budget_topk"`: slower diagnostic loss for
  `mlqds_hybrid_mode="stratified"`. It optimizes soft target-mass capture
  within each final selector stratum instead of global top-k.
- `temporal_residual_label_mode="none"`: default. It trains on the configured
  target directly.
- `temporal_residual_label_mode="temporal"`: explicit diagnostic mode. It
  trains only the learned fill left after the temporal base selected by
  `mlqds_temporal_fraction`. It is ignored for
  `mlqds_hybrid_mode="stratified"` because that selector has no reserved
  temporal base.
- `loss_objective="ranking_bce"`: pairwise-ranking ablation.
- `loss_objective="pointwise_bce"`: direct soft-label fit diagnostic. It uses
  all valid supervised points instead of sampled negative balancing or top-k
  recall.
- `pointwise_loss_weight`: auxiliary BCE term.
- `mlqds_hybrid_mode="fill"`: temporal base plus learned score fill.
- `mlqds_hybrid_mode="swap"`: start from uniform temporal sampling and replace
  only the unprotected budget share.
- `mlqds_hybrid_mode="local_swap"` and `"local_delta_swap"`: replace
  unprotected temporal base points with learned-score candidates, either
  unconditionally or only when the learned score improves over the paired base
  point.
- `mlqds_min_learned_swaps`: diagnostic lower bound on learned replacements per
  trajectory for `swap`, `local_swap`, and `local_delta_swap`. Default `0`
  preserves `mlqds_temporal_fraction` rounding exactly. Use it to test whether
  low-budget failures are caused by no learned slots versus bad learned scores;
  do not treat it as final proof by itself.
- `mlqds_hybrid_mode="stratified"`: split each trajectory into retained-budget
  strata and use learned scores to pick inside each stratum. This is still
  workload-blind; future eval queries are not used during compression. This
  mode does not use `mlqds_temporal_fraction` or `mlqds_diversity_bonus`.
  `mlqds_stratified_center_weight` optionally penalizes within-stratum choices
  far from the stratum center; `0.0` keeps pure learned score selection.
- `mlqds_hybrid_mode="global_fill"`: keep a temporal base in every trajectory,
  then allocate the remaining learned-fill slots globally by score. This tests
  whether global scarcity helps after continuity safeguards are preserved.
- `mlqds_hybrid_mode="global_budget"`: keep an endpoint skeleton for every
  trajectory, then allocate the remaining retained-point budget globally by
  learned score. This is workload-blind but should be treated as a diagnostic
  until it beats global random and preserves sensible trajectory geometry.
- `checkpoint_score_variant="range_usefulness"`: legacy range diagnostic target.
  It is retained for old-profile comparability, not final rework acceptance.
  `answer` and `combined` are diagnostics.
- `checkpoint_selection_metric="uniform_gap"`: validation score minus fair
  uniform score, with active-type deficit penalties.
- `training_fit_diagnostics`: post-training, train-data-only student-fit
  report. It scores the restored checkpoint against the scaled training target
  and reports target-mass recall versus uniform across the configured budget
  grid. This is diagnostic only; it does not use eval queries or affect
  checkpoint selection.
- `range_replicate_target_aggregation="label_mean"`: default replicated
  train-workload aggregation before retained-frequency target selection.
  `label_max` keeps points that are strongly useful under any training
  workload prior; `frequency_mean` averages per-workload retained-frequency
  masks instead.
- `range_training_target_mode="component_retained_frequency"` supports
  replicated train workloads by aggregating each RangeUseful component stream
  separately before target selection, or by averaging per-replicate component
  retained-frequency targets.
- `range_training_target_mode="continuity_retained_frequency"` is a narrower
  component target for boundary and continuity failures. It excludes the
  point/ship-presence components and builds retained-frequency targets from
  entry/exit, crossing, temporal coverage, gap coverage, turn coverage, and
  shape. `range_component_target_blend` can blend it back toward ordinary
  retained-frequency targets for ablation.
- `range_teacher_distillation_mode="retained_frequency"` can distill multiple
  training workload replicates. The query-aware teachers use training workloads
  only; the final student remains workload-blind at compression time.
- `range_training_target_mode="historical_prior_retained_frequency"` first
  builds the ordinary retained-frequency target, scores train points with a
  leave-one-out query-free historical KNN teacher, then converts those teacher
  scores back into retained-frequency labels. This is train-only distillation:
  eval compression still uses the configured blind student and frozen masks.
  Treat this as a diagnostic target unless it beats the base retained-frequency
  target on held-out days; the first 30% guarded slice diffused label mass and
  underperformed uniform.
- `range_training_target_mode="query_useful_v1_factorized"` is reserved for the
  future QueryUsefulV1 target family. Selecting it fails deliberately in this
  checkpoint.

## Model Inputs

`model_type="range_aware"` adds point/query relation features before scoring.
That is useful for diagnostics and teacher targets, but it is workload-aware.
A final blind model must score without future eval query features.

`model_type="workload_blind_range"` keeps the original compact query-free
feature slice for checkpoint compatibility. `model_type="range_prior"` uses the
full query-free context feature set: trajectory-local time, local index and
distance position, adjacent time/distance gaps, heading/speed deltas, curvature,
and endpoint flags. `model_type="range_prior_clock_density"` adds query-free
clock-time and current-day spatial density/sparsity features to that context
set. These blind variants do not consume query boxes.
Those density/sparsity features are current-split point-cloud context features.
They are not train-derived query-prior fields.
`model_type="segment_context_range"` is a query-free structural scorer inspired
by MLSimp's globality/uniqueness framing. It uses the same 28-column
`range_prior_clock_density` feature input, then adds fixed trajectory-order
segment summaries, point-to-segment attention, local uniqueness, and trajectory
globality scalars before predicting the retained score. It is a structural
model candidate, not a query-conditioned shortcut.
For these neural blind models, `num_layers=0` uses the point encoder and score
head only. This avoids transformer window-position encodings and is useful when
testing whether overlapping window position noise is hurting target fit.

`model_type="historical_prior"` is a query-free nonparametric diagnostic model.
It stores normalized train-day route-context, circular clock-time, spatial
density/sparsity features, and retained-frequency targets, then scores future
points by inverse-distance KNN. It is workload-blind at compression time, but it
is not a neural student; treat it as a historical-prior KNN diagnostic/teacher,
not proof that a trainable model learned the target. `historical_prior_clock_weight`
and `historical_prior_density_weight` control the KNN distance weights assigned
to clock-time and density/sparsity dimensions without changing the retained-mask
protocol. Clock weighting defaults to `0.0` because the first dense diagnostic
run made it worse. `historical_prior_min_target` optionally filters weak/zero
train support before fitting the KNN prior; `0.0` preserves the full train set.
`historical_prior_support_ratio` optionally caps the stored top-target support
per train trajectory before min-target filtering; `1.0` preserves all support.
`historical_prior_source_aggregation` can combine per-train-CSV KNN scores with
`mean`, `min`, or `median` instead of pooling every historical point. This is a
multi-day transfer diagnostic: it penalizes one-day-only matches while keeping
eval compression query-free. The default `none` preserves the original pooled
prior.

`model_type="historical_prior_mmsi"` is the same workload-blind KNN prior with
an additional deterministic 4-dim MMSI hash in the query-free feature vector.
It may use MMSI because vessel identity is known before compression and does not
depend on eval queries. `historical_prior_mmsi_weight` scales only that identity
slice in the KNN distance. Keep this as a diagnostic feature path: the initial
30% coverage checks did not improve the low-budget acceptance cells.

`model_type="historical_prior_student"` uses the same query-free stored prior
as an explicit neural input feature. It appends the KNN prior score to each
point's blind feature vector and trains a normal blind scorer on top. This is
still workload-blind at eval, but it must be compared against the standalone
`historical_prior` KNN diagnostic to prove the trainable layer is adding value.
The first guarded 30% check did not add value, so keep it in diagnostic status.

Workload-blind point features deliberately replace absolute timestamp with
trajectory-local time fraction. Cross-day train/eval runs should not ask the
student to extrapolate raw epoch time under a scaler fitted on a different day.
The historical-prior variant is the exception: it uses bounded sine/cosine
clock-time features because `anchor_day` workloads make time of day relevant
and the clock is known before compression.

## Runtime Defaults

The active benchmark profile sets:

- `train_batch_size=64`
- `inference_batch_size=64`
- `query_chunk_size=2048`
- `allow_tf32=True`
- `amp_mode="bf16"`

Library defaults are more conservative for direct CLI use.

Standalone `historical_prior` inference is pointwise. `windowed_predict` uses a
flat fast path for that model instead of scoring overlapping trajectory windows;
this preserves exact KNN scores while avoiding duplicate work. Trainable
students still use the normal windowed path because their transformer context
can depend on the window.

## Persistence

Use `checkpoints.py` for save/load and `inference.py` for prediction. Checkpoint
artifacts include model state, scaler stats, config, target diagnostics, epoch
timing, and selected validation scores.
