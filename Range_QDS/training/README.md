# Training Module

Builds supervision, batches trajectory windows, trains scorers, selects
checkpoints, and persists model artifacts.

## Key Files

| File | Purpose |
| --- | --- |
| `query_useful_targets.py` | Factorized QueryUsefulV1 label construction. |
| `query_prior_fields.py` | Train-only query-prior field construction and sampling. |
| `model_features.py` | Query-free and query-conditioned point feature builders. |
| `training_targets.py` | Legacy RangeUseful/scalar target transforms. |
| `training_losses.py` | Budget, ranking, pointwise, and auxiliary losses. |
| `training_validation.py` | Validation scoring and checkpoint selection support. |
| `train_model.py` | Main training loop and checkpoint selection flow. |
| `checkpoints.py` / `inference.py` | Save/load and deterministic prediction. |
| `trajectory_batching.py` / `training_windows.py` | Window construction without crossing trajectory boundaries. |

## Active Query-Driven Flow

The current candidate path is workload-blind at compression time:

1. Generate train-only `range_workload_v1` workloads.
2. Build factorized `QueryUsefulV1` labels.
3. Build train-derived query-prior fields.
4. Train `workload_blind_range_v2` from query-free point/context/prior features.
5. Select checkpoints without final eval-query scoring.
6. Freeze retained masks before held-out eval queries are scored.

The acceptance contract is in
[`../docs/query-driven-rework-guide.md`](../docs/query-driven-rework-guide.md).

## Final-Candidate Settings

| Setting | Active value |
| --- | --- |
| `workload_profile_id` | `range_workload_v1` |
| `model_type` | `workload_blind_range_v2` |
| `range_training_target_mode` | `query_useful_v1_factorized` |
| `selector_type` | `learned_segment_budget_v1` |
| `checkpoint_score_variant` | `query_useful_v1` |
| `checkpoint_selection_metric` | `uniform_gap` |

These settings are necessary but not sufficient. Final claims still require all
single-cell gates and the full final grid.

## Targets

`query_useful_v1_factorized` is the active target family. It predicts separate
query-hit, behavior, replacement, segment-budget, and prior-related signals
rather than one legacy scalar.

Legacy `range_training_target_mode` values are retained for diagnostics,
regression, and teacher experiments. They are not final-success evidence:

- `retained_frequency`
- `global_budget_retained_frequency`
- `marginal_coverage_frequency`
- `structural_retained_frequency`
- `component_retained_frequency`
- `continuity_retained_frequency`
- `query_spine_frequency`
- `query_residual_frequency`
- `set_utility_frequency`
- `local_swap_gain_cost_frequency`
- `historical_prior_retained_frequency`

Use legacy modes only when the checkpoint hypothesis explicitly needs them.

## Loss And Selection

- `loss_objective="budget_topk"` is the standard range loss.
- `ranking_bce` and `pointwise_bce` are diagnostics.
- `checkpoint_selection_metric="uniform_gap"` scores validation performance
  against fair uniform, with penalties for active-type deficits.
- `validation_length_preservation_min` controls checkpoint-selection sanity
  pressure; the strict global sanity gate is enforced later in experiment
  outputs.
- `training_fit_diagnostics` is train-data-only. It is useful for target-fit
  debugging, not final evidence.

## Model Inputs

`range_aware` and older query-conditioned models can see query features and are
diagnostic only. Final blind candidates must score without future eval query
features.

Workload-blind feature families:

- `workload_blind_range`: compact query-free compatibility path.
- `range_prior`: local trajectory context features.
- `range_prior_clock_density`: adds query-free clock and current-day density
  context.
- `segment_context_range`: structural segment/context diagnostic scorer.
- `workload_blind_range_v2`: active trainable QueryUsefulV1 candidate.

Historical-prior paths:

- `historical_prior`: query-free KNN diagnostic/teacher.
- `historical_prior_mmsi`: KNN diagnostic with deterministic MMSI hash
  features.
- `historical_prior_student`: trainable model with KNN prior input.

Historical-prior variants are workload-blind, but they must beat or explain
the standalone KNN teacher before claiming learned value.

## Runtime And Persistence

Benchmark profiles commonly use:

- `train_batch_size=64`
- `inference_batch_size=64`
- `query_chunk_size=2048`
- `allow_tf32=True`
- `amp_mode="bf16"`

Direct CLI defaults are more conservative. Use `checkpoints.py` for save/load
and `inference.py` for prediction. Checkpoints include model state, scaler
stats, config, target diagnostics, epoch timing, and selected validation
scores.
