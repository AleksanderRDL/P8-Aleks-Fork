# Models Module

Contains the implemented query-conditioned trajectory encoder.

Important caveat: current models are workload-aware because points attend to
query embeddings during prediction. They are useful diagnostics and teacher
candidates, but final workload-blind range compression needs a separate
query-free scoring path.

## Files

| File | Purpose |
| --- | --- |
| `attention_utils.py` | Chunked point-to-query cross-attention. |
| `trajectory_qds_model.py` | Base transformer and normalization helper. |
| `turn_aware_qds_model.py` | Same architecture with `turn_score`. |
| `../training/model_features.py` | Baseline, turn-aware, and range-aware point features. |

## Flow

```text
point features -> point encoder -> positional encoding -> local transformer
query features + query type IDs -> query encoder
point states + query states -> chunked cross-attention -> score head
```

## Key Rules

- Every forward call needs `query_type_ids`.
- Current experiment paths train one pure workload per model and output one
  score per point.
- `query_chunk_size >= n_queries` gives exact one-chunk cross-attention;
  smaller chunks are an approximation.
- Attention is point-to-query and does not mix points across trajectories.
- Point features are 7 columns for baseline, 8 for turn-aware, and 16 for
  range-aware.
- Query features are 12 padded columns from `pad_query_features`.
