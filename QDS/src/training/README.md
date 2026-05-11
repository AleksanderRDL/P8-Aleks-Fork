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

1. `compute_typed_importance_labels` turns the typed workload into F1-contribution `labels[N,4]` and `labelled_mask[N,4]`.
2. `FeatureScaler.fit` learns min-max stats from training points and query features, then normalizes both.
3. `build_trajectory_windows` produces padded windows that never cross trajectory boundaries.
4. `train_model` rescales sparse F1 labels per type for optimization, then trains a query-conditioned transformer with per-type margin ranking losses plus balanced pointwise BCE supervision.
5. `windowed_predict` and `forward_predict` reuse the same windowing logic for deterministic inference.

## Label Construction

- Range labels default to pure point-F1 contribution: every point inside a range query box receives the same singleton retained-hit gain for that query. `range_boundary_prior_weight` can optionally boost in-box boundary-crossing points before normalization, but it defaults to `0.0` for rigorous pure-F1 benchmarks. Cross-trajectory proximity is not used for range labels.
- kNN and similarity labels execute the query on the original data, identify the original trajectory-ID answer set, and assign points the F1 gain of recovering one true-positive trajectory ID.
- Similarity and clustering labels can use the optional `turn_score` feature as a small shape prior.
- Clustering labels execute the original clustering query, convert cluster labels to same-cluster trajectory pairs, and assign points in clustered trajectories the F1 gain of recovering their original co-membership pairs. Within a clustered query box, point mass is weighted by distance from the trajectory's in-box centroid.
- Labels are averaged per query type and clamped to `[0, 1]`, so higher labels directly mean higher expected query F1 contribution.
- Training keeps those raw labels for reporting and oracle diagnostics, but rescales each active type internally so tiny F1 gains still produce useful gradients.
- The old speed-mass, distance-decay, interpolation, and speed-baseline heuristics are no longer used.

## Training Notes

- Current experiment entrypoints train one pure workload per model. Mixed
  workload helpers remain for low-level diagnostics only.
- Windows never cross trajectory boundaries. Windows with no positive label for
  a type are skipped for that type; the pointwise BCE term samples zeros to
  avoid all-zero collapse.
- `ranking_pair_sampling="vectorized"` is the default. Use `"legacy"` only when
  comparing against older runs.
- The effective epoch count is clamped to at least 8. The returned model is
  restored to the best diagnostic epoch; by default that means held-out final
  query F1.
- `checkpoint_f1_variant="answer"` is the default selection target. `"combined"`
  remains a legacy answer/support diagnostic. `uniform_gap`,
  `checkpoint_smoothing_window`, and `early_stopping_patience` are explicit
  selection stabilizers.
- Validation query-F1 uses the same canonical MLQDS score conversion as final
  evaluation: one explicit workload head, `mlqds_score_mode`, and the
  temporal/diversity retained-mask simplifier.
- AIS-scale stability knobs: `lr`, `pointwise_loss_weight`,
  `gradient_clip_norm`, `train_batch_size`, `inference_batch_size`,
  `query_chunk_size`, `float32_matmul_precision`, `allow_tf32`, and `amp_mode`.
  Benchmark changes by retained-set F1, epoch time, and peak memory together.
- Defaults: `window_length=512`, `window_stride=256`,
  `query_chunk_size=2048`, `float32_matmul_precision="highest"`,
  `allow_tf32=False`, and `amp_mode="off"`. The real-usecase benchmark profile
  overrides runtime precision through its selected matrix variant.

## Persistence

- `ModelArtifacts` stores the trained model, scaler, and experiment config.
- `save_checkpoint` writes the model state, scaler stats, and config.
- `load_checkpoint` reconstructs the correct model class, reloads the scaler, and puts the model into eval mode.
- `save_training_summary` stores the diagnostics history as JSON.
