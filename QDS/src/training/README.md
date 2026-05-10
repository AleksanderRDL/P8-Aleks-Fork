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

- Range labels score points inside the spatiotemporal query box as point-level F1 hits, with extra weight for boundary crossings and cross-trajectory proximity. Range labels still normalize by query count like every other query type.
- kNN and similarity labels execute the query on the original data, identify the original trajectory-ID answer set, and assign points the F1 gain of recovering one true-positive trajectory ID.
- Similarity and clustering labels can use the optional `turn_score` feature as a small shape prior.
- Clustering labels execute the original clustering query, convert cluster labels to same-cluster trajectory pairs, and assign points in clustered trajectories the F1 gain of recovering their original co-membership pairs. Within a clustered query box, point mass is weighted by distance from the trajectory's in-box centroid.
- Labels are averaged per query type and clamped to `[0, 1]`, so higher labels directly mean higher expected query F1 contribution.
- Training keeps those raw labels for reporting and oracle diagnostics, but rescales each active type internally so tiny F1 gains still produce useful gradients.
- The old speed-mass, distance-decay, interpolation, and speed-baseline heuristics are no longer used.

## Training Notes

- Epoch-level workload weights are sampled from a lightweight Dirichlet approximation so every type continues to receive signal.
- Windows with no positive label for a type are skipped for that type; the pointwise BCE term uses all positives and a bounded random sample of zero labels to avoid all-zero collapse.
- `ModelConfig.lr`, `pointwise_loss_weight`, and `gradient_clip_norm` are the main stability knobs for AIS-scale runs.
- The current loop clamps the effective epoch count to at least 8.
- The returned model is restored to the best diagnostic epoch by selection score. By default this is training loss with a small Kendall-tau tie signal and a collapse penalty for near-constant predictions.
- Set `checkpoint_selection_metric="f1"` with a held-out validation workload to select checkpoints by query-F1 semantics used in final evaluation. `checkpoint_f1_variant="answer"` selects on pure answer-set F1; `checkpoint_f1_variant="combined"` selects on the legacy answer/support product. Use `checkpoint_selection_metric="uniform_gap"` when the restored checkpoint should also be judged against the fair `uniform` validation score; this subtracts weighted per-type deficits so clustering cannot mask weak range/kNN/similarity performance. `checkpoint_smoothing_window` can average the last K diagnostic selection scores before deciding which checkpoint is best, which reduces one-epoch F1 noise during Phase 3 runs. These metrics are useful for model selection, but they are intentionally not used as the training loss because final query F1 is discrete after per-trajectory top-k simplification and query execution.
- Validation query-F1 diagnostics reuse the evaluation query cache across epochs, so the fixed full-data query answers and support masks are computed once for the held-out workload while each checkpoint still gets its own retained-mask evaluation.
- `f1_diagnostic_every` can record held-out query-F1 diagnostics while still selecting by loss. `TrainingOutputs.best_epoch`, `best_loss`, and `best_f1` record the selected checkpoint metadata.
- The loop tracks loss, prediction spread, quantiles, Kendall tau-style diagnostics, and optional validation query F1 in `TrainingOutputs.history`. Diagnostics run every `diagnostic_every` epochs, which defaults to every epoch so each epoch can be considered for checkpoint restoration.
- The current implementation uses trajectory-local windows with `window_length=512` and `window_stride=256` by default.
- `train_batch_size` controls how many trajectory windows are grouped per optimizer step; benchmark sweeps should compare epoch time, peak CUDA memory, and final retained-set F1 before changing the default.
- `windowed_predict` and `forward_predict` accept an optional inference device. When CUDA is requested, model windows and query tensors run on CUDA and predictions are moved back to the original point tensor device for downstream metrics.
- `ModelConfig.float32_matmul_precision` and `allow_tf32` control process-local torch matmul precision before training starts. The default is `highest` with TF32 disabled; use `high` plus TF32 for RTX throughput benchmarks and compare final retained-set F1, not only runtime.

## Persistence

- `ModelArtifacts` stores the trained model, scaler, and experiment config.
- `save_checkpoint` writes the model state, scaler stats, and config.
- `load_checkpoint` reconstructs the correct model class, reloads the scaler, and puts the model into eval mode.
- `save_training_summary` stores the diagnostics history as JSON.
