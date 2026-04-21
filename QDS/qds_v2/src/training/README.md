# Training Module

This module builds typed importance labels, batches trajectory-local windows, fits the shared scaler, trains the model, and persists the checkpoint artifacts used at inference time.

## Files

| File | Purpose |
| --- | --- |
| `importance_labels.py` | Build per-point, per-type labels and the labelled mask from typed queries. |
| `trajectory_batching.py` | Slice flattened trajectories into fixed-length windows with padding metadata. |
| `scaler.py` | Persisted min-max scaler for points and queries. |
| `train_model.py` | Ranking-loss training loop, diagnostics, and forward helpers. |
| `training_pipeline.py` | Checkpoint save/load helpers and deterministic prediction wrappers. |

## Training Flow

1. `compute_typed_importance_labels` turns the typed workload into `labels[N,4]` and `labelled_mask[N,4]`.
2. `FeatureScaler.fit` learns min-max stats from training points and query features, then normalizes both.
3. `build_trajectory_windows` produces padded windows that never cross trajectory boundaries.
4. `train_model` optimizes a query-conditioned transformer with per-type margin ranking losses plus a small MSE anchor.
5. `windowed_predict` and `forward_predict` reuse the same windowing logic for deterministic inference.

## Label Construction

- Range labels come from the speed mass inside the range box.
- kNN labels mark the selected point indices.
- Similarity labels sample points inside the query region and score them against the reference snippet.
- Clustering labels sample points in the query box and score them by normalized speed.
- The raw labels are interpolated within each trajectory and then lightly blended with a normalized speed baseline so every head has a stable signal.

## Training Notes

- `train_model` supports a `query_blind` ablation that feeds random query features during training.
- Epoch-level workload weights are sampled from a lightweight Dirichlet approximation so every type continues to receive signal.
- The current loop clamps the effective epoch count to at least 8.
- The loop tracks loss, prediction spread, quantiles, and Kendall tau-style diagnostics in `TrainingOutputs.history`.
- The current implementation uses trajectory-local windows with `window_length=512` and `window_stride=256` by default.

## Persistence

- `ModelArtifacts` stores the trained model, scaler, and experiment config.
- `save_checkpoint` writes the model state, scaler stats, and config.
- `load_checkpoint` reconstructs the correct model class, reloads the scaler, and puts the model into eval mode.
- `save_training_summary` stores the diagnostics history as JSON.
