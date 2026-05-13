# Models Module

This module contains the currently implemented query-conditioned trajectory
encoder used by AIS-QDS.

Important redesign context: these models are workload-aware because points
attend to query embeddings during prediction. They are useful diagnostics and
teacher candidates, but a final workload-blind range model must add a separate
path that scores points without future query inputs.

## Files

| File | Purpose |
| --- | --- |
| `attention_utils.py` | Chunked cross-attention helper that lets points attend to typed query embeddings. |
| `trajectory_qds_model.py` | Base transformer model and min-max normalization helper. |
| `turn_aware_qds_model.py` | Same model architecture with the extra `turn_score` feature. |
| `../training/model_features.py` | Builds baseline, turn-aware, and range-aware point feature matrices before model/scaler use. |

## Model Flow

```text
points[L,Fp] -> point encoder -> + sinusoidal positional encoding -> local Transformer -> H[L,D]
queries[M,Fq] + query_type_ids[M] -> type embedding -> query encoder -> Q[M,D]
H and Q -> chunked cross-attention -> context C[L,D]
H + C -> single pure-workload score head -> logits[L]
```

## Key Behavior

- `TrajectoryQDSModel` expects `query_type_ids` for every forward call. Current
  experiment paths train one pure workload per model and return one score per
  point, not four per-type output heads.
- `chunked_cross_attention_context` accumulates query-chunk outputs and divides by the chunk count, so context scale stays stable as workloads grow. The result is exact when `query_chunk_size >= n_queries`; otherwise it is a bounded approximation because each chunk has its own attention softmax.
- Cross-attention disables attention-weight materialization because the weights are not consumed by the pipeline.
- Sinusoidal positional encodings are cached in a non-persistent buffer keyed by effective length/device/dtype behavior, so repeated fixed-window training and inference forwards avoid rebuilding them.
- The attention direction is point-to-query, which keeps the per-point representation query-conditioned without leaking across trajectories.
- `normalize_points_and_queries` is the shared min-max transform used by the persisted `FeatureScaler`.

## Shapes And Defaults

- Point features: 7 columns for the baseline model, 8 for the turn-aware model,
  and 16 for the range-aware model.
- Query features: 12 padded features from `src.queries.query_types.pad_query_features`.
- Output: one pure-workload logit for each point.
- Default query chunk size: 2048. Current workload-aware diagnostic range
  benchmarks keep the workload below that so cross-attention runs as one exact
  query chunk.
