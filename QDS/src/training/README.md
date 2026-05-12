# Training Module

This module builds typed F1-contribution labels, batches trajectory-local windows, fits the shared scaler, trains the model, and persists the checkpoint artifacts used at inference time.

## Files

| File | Purpose |
| --- | --- |
| `importance_labels.py` | Build per-point, per-type F1 contribution labels and the labelled mask from typed queries. |
| `trajectory_batching.py` | Slice flattened trajectories into fixed-length windows with padding metadata. |
| `scaler.py` | Persisted min-max scaler for points and queries. |
| `train_model.py` | Ranking-loss training loop, diagnostics, and forward helpers. |
| `training_pipeline.py` | Checkpoint save/load helpers and deterministic prediction wrappers. |

## Training Flow

1. `compute_typed_importance_labels` turns the typed workload into per-point
   label targets `labels[N,4]` and `labelled_mask[N,4]`.
2. `FeatureScaler.fit` learns min-max stats from training points and query features, then normalizes both.
3. `build_trajectory_windows` produces padded windows that never cross trajectory boundaries.
4. `train_model` rescales sparse labels per type for optimization, then trains a
   query-conditioned transformer with the active loss objective plus balanced
   pointwise BCE supervision.
5. `windowed_predict` and `forward_predict` reuse the same windowing logic for deterministic inference.

## Label Construction

- Range labels now support two explicit modes. `point_f1` is the old point-hit
  proxy: every point inside a range query box receives the same singleton
  retained-hit gain for that query. `usefulness` is the default training mode:
  it adds local proxy signal for ship presence, sampled entry/exit points,
  in-query temporal span endpoints, and local shape/turn points. Cross-trajectory
  proximity is not used.
- `range_boundary_prior_weight` is still available as an explicit point-hit
  boundary prior, but the default real-usecase path keeps it at `0.0` because
  `usefulness` already reports entry/exit signal as a separate component.
- These labels remain local/additive approximations. They are closer to
  range-local retained-set usefulness than pure point hits, but they are not an
  exact optimizer for multi-budget retained-set utility; see
  `../../../Aleks-Sprint/range-objective-redesign.md`.
- kNN and similarity labels execute the query on the original data, identify the original trajectory-ID answer set, and assign points the F1 gain of recovering one true-positive trajectory ID.
- Similarity and clustering labels can use the optional `turn_score` feature as a small shape prior.
- Clustering labels execute the original clustering query, convert cluster labels to same-cluster trajectory pairs, and assign points in clustered trajectories the F1 gain of recovering their original co-membership pairs. Within a clustered query box, point mass is weighted by distance from the trajectory's in-box centroid.
- Labels are averaged per query type and clamped to `[0, 1]`, so higher labels directly mean higher expected query F1 contribution.
- Training keeps those raw labels for reporting and oracle diagnostics, but rescales each active type internally so tiny F1 gains still produce useful gradients.
- The old speed-mass, distance-decay, interpolation, and speed-baseline heuristics are no longer used.

## Training Notes

- Current experiment entrypoints train one pure workload per model. The model
  emits one score stream for that workload; per-type output heads are not part
  of training or evaluation.
- Windows never cross trajectory boundaries. Training prefilters windows with
  no positive labels for any active workload type before the model forward; any
  remaining per-type zero-positive lanes are still skipped for that type. The
  pointwise BCE term samples zeros to avoid all-zero collapse.
- `loss_objective="budget_topk"` is the default range objective. It trains the
  score stream to capture high label mass inside soft top-k retained budgets
  across `budget_loss_ratios` (`0.01,0.02,0.05,0.10` by default). This is closer
  to the final simplification decision than the old local pair sampler. The
  budget-top-k loss is computed across padded batch rows in one batched tensor
  path; scalar helpers remain as correctness references in tests.
- With `residual_label_mode="temporal"`, budget-top-k does not globally delete
  temporal-spine labels once. It computes per-budget temporal-base masks and
  trains the learned fill only on the points still controlled by the model at
  each retained-point ratio. This keeps the multi-budget objective aligned with
  `mlqds_temporal_fraction`.
- Training artifacts include `training_target_diagnostics`, which records the
  configured budget ratios, effective learned-fill ratios, temporal-base point
  counts, remaining candidate counts, residual positive-label counts, and how
  much positive label mass is consumed by the temporal spine versus left for
  learned residual fill.
- `loss_objective="ranking_bce"` keeps the legacy margin-ranking objective for
  ablation. `pointwise_loss_weight` remains an auxiliary BCE term for both
  objectives.
- The configured epoch count is used literally, with a minimum of 1 epoch. The
  returned model is restored to the best diagnostic epoch. `best_selection_score`
  is the canonical selected validation score; `best_f1` remains as a legacy
  compatibility alias.
- `checkpoint_f1_variant="range_usefulness"` is the default selection target
  for range training because it matches the range-local usefulness objective.
  `"answer"` and `"combined"` remain legacy diagnostics for explicit ablations.
  `uniform_gap`, `checkpoint_smoothing_window`, `checkpoint_full_f1_every`,
  `checkpoint_candidate_pool_size`, and `early_stopping_patience` are explicit
  selection stabilizers. `checkpoint_full_f1_every=1` keeps exact validation
  every eligible epoch. Higher values keep cheap-diagnostic candidate snapshots
  and run exact validation only on the best candidates in each validation round.
- Validation selection uses the same canonical MLQDS score conversion as final
  evaluation: one explicit workload score stream, `mlqds_score_mode`, and the
  temporal/diversity retained-mask simplifier. History rows report
  `val_selection_score` plus explicit `val_range_point_f1` and
  `val_range_usefulness` fields so range-usefulness checkpointing is not
  mislabeled as generic query F1. Range usefulness is versioned in audit JSON
  because component weights may change during objective redesign.
- AIS-scale stability knobs: `lr`, `pointwise_loss_weight`,
  `gradient_clip_norm`, `train_batch_size`, `inference_batch_size`,
  `query_chunk_size`, `float32_matmul_precision`, `allow_tf32`, and `amp_mode`.
  Benchmark changes by retained-set F1, epoch time, and peak memory together.
- Epoch diagnostics record `epoch_forward_seconds`, `epoch_loss_seconds`,
  `epoch_backward_seconds`, `epoch_diagnostic_seconds`,
  `epoch_f1_seconds`, and filtered-window counts for bottleneck analysis.
- Real-usecase profiling showed the old ranking loss could dominate epoch time
  while optimizing a local proxy. The budget-top-k objective is the first loss
  redesign step; judge it by retained-set `RangeUseful`/`RangePointF1` across
  compression ratios before further tuning.
- Defaults: `window_length=512`, `window_stride=256`,
  `query_chunk_size=2048`, `float32_matmul_precision="highest"`,
  `allow_tf32=False`, and `amp_mode="off"`. The real-usecase benchmark profile
  overrides runtime precision through its selected matrix variant.

## Persistence

- `ModelArtifacts` stores the trained model, scaler, and experiment config.
- `save_checkpoint` writes the model state, scaler stats, and config.
- `load_checkpoint` reconstructs the correct model class, reloads the scaler, and puts the model into eval mode.
- `save_training_summary` stores the diagnostics history as JSON.
