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
| `train_model.py` | Losses, diagnostics, and training loop. |
| `training_pipeline.py` | Compatibility re-exports. |

## Current Flow

The implemented strong path is workload-aware:

1. Generate a pure workload and point labels.
2. Fit `FeatureScaler` on training points/query features.
3. Train a query-conditioned model on padded trajectory windows.
4. Select checkpoints by validation score or `uniform_gap`.
5. Restore the best checkpoint for final evaluation.

This is not the final workload-blind protocol. The redesign target is in
[`../../../Aleks-Sprint/range-training-redesign.md`](../../../Aleks-Sprint/range-training-redesign.md).

## Labels

- `range_label_mode="usefulness"`: current range label default. It approximates
  the `RangeUseful` audit components, not only in-box point hits.
- `range_label_mode="usefulness_balanced"`: rescales component mass toward the
  audit weights when raw support frequency dominates.
- `range_label_mode="point_f1"`: old point-hit ablation.
- `range_boundary_prior_weight`: optional explicit prior; diagnostic profile
  keeps it at `0.0`.

Labels are additive training targets. They are not exact retained-set optimizers.

## Loss And Selection

- `loss_objective="budget_topk"`: default range loss. It trains score mass into
  top-k budgets from `budget_loss_ratios`.
- `temporal_residual_label_mode="temporal"`: trains only the learned fill left
  after the temporal base selected by `mlqds_temporal_fraction`.
- `loss_objective="ranking_bce"`: pairwise-ranking ablation.
- `pointwise_loss_weight`: auxiliary BCE term.
- `mlqds_hybrid_mode="fill"`: temporal base plus learned score fill.
- `mlqds_hybrid_mode="swap"`: start from uniform temporal sampling and replace
  only the unprotected budget share.
- `checkpoint_score_variant="range_usefulness"`: default range checkpoint
  target. `answer` and `combined` are diagnostics.
- `checkpoint_selection_metric="uniform_gap"`: validation score minus fair
  uniform score, with active-type deficit penalties.

## Model Inputs

`model_type="range_aware"` adds point/query relation features before scoring.
That is useful for diagnostics and teacher targets, but it is workload-aware.
A final blind model must score without future eval query features.

## Runtime Defaults

The active benchmark profile sets:

- `train_batch_size=64`
- `inference_batch_size=64`
- `query_chunk_size=2048`
- `allow_tf32=True`
- `amp_mode="bf16"`

Library defaults are more conservative for direct CLI use.

## Persistence

Use `checkpoints.py` for save/load and `inference.py` for prediction. Checkpoint
artifacts include model state, scaler stats, config, target diagnostics, epoch
timing, and selected validation scores.
