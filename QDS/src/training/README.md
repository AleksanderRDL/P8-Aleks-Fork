# Training Module

This module builds range-usefulness labels, batches trajectory-local windows,
fits the scaler, trains MLQDS, and persists checkpoint artifacts.

## Files

| File | Purpose |
| --- | --- |
| `importance_labels.py` | Per-point workload labels and labelled masks. |
| `trajectory_batching.py` | Fixed-length trajectory windows with padding metadata. |
| `scaler.py` | Persisted min-max scaler for point/query features. |
| `train_model.py` | Loss objectives, diagnostics, and training loop. |
| `training_pipeline.py` | Checkpoint save/load and deterministic prediction helpers. |

## Flow

1. Build workload labels for the active pure workload.
2. Fit `FeatureScaler` on training points and query features.
3. Build padded windows that never cross trajectory boundaries.
4. Train the query-conditioned model and checkpoint by validation selection
   score.
5. Restore the best checkpoint for final evaluation/inference.

## Range Labels

`range_label_mode="usefulness"` is the current default. It provides local proxy
signal for point hits, ship presence, per-ship coverage, sampled entry/exit
points, crossing brackets, in-query temporal span, gap coverage, turns, and
shape. Cross-trajectory proximity is intentionally not used.

`range_label_mode="point_f1"` keeps the old point-hit proxy for ablations.
`range_boundary_prior_weight` remains available as an explicit optional prior,
but the testing baseline keeps it at `0.0`.

These labels are additive approximations for training, not an exact optimizer
for retained-set `RangeUseful`. The current objective reasoning lives in
[`../../../Aleks-Sprint/range-objective-redesign.md`](../../../Aleks-Sprint/range-objective-redesign.md).

## Loss And Selection

- `loss_objective="budget_topk"` is the default range loss. It trains the score
  stream to concentrate label mass inside soft top-k budgets across
  `budget_loss_ratios` (`0.01,0.02,0.05,0.10` by default).
- `residual_label_mode="temporal"` computes a per-budget temporal-base mask and
  trains only the learned fill that remains controlled by MLQDS. This keeps the
  loss aligned with `mlqds_temporal_fraction`.
- `loss_objective="ranking_bce"` is the legacy ranking/BCE ablation.
- `pointwise_loss_weight` remains an auxiliary BCE term for both objectives.
- `checkpoint_f1_variant="range_usefulness"` is the default selection target
  for range training. `answer` and `combined` are legacy diagnostics/ablations.
- `checkpoint_full_f1_every` and `checkpoint_candidate_pool_size` let cheap
  diagnostics produce candidates between exact validation epochs.

Training artifacts include `training_target_diagnostics`, epoch timing fields,
selected validation scores, and range-label component mass fractions. Use those
before changing the objective; they show whether the temporal base, residual
fill, or label mix is dominating a run.

## Runtime Knobs

Benchmark profile defaults are defined in
[`../experiments/benchmark_profiles.py`](../experiments/benchmark_profiles.py).
The current `range_testing_baseline` profile uses:

- `train_batch_size=64`
- `inference_batch_size=64`
- `query_chunk_size=2048`
- `allow_tf32=True`
- `amp_mode="bf16"`

Library-level defaults are more conservative so direct CLI calls remain
portable. Benchmark quality should be compared by `RangeUseful`,
`RangePointF1`, epoch time, and memory use together.

## Persistence

- `ModelArtifacts` stores the trained model, scaler, and experiment config.
- `save_checkpoint` writes model state, scaler stats, and config.
- `load_checkpoint` reconstructs the model, reloads the scaler, and returns it
  in eval mode.
