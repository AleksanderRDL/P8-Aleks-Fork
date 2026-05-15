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
| `workload_blind_qds_model.py` | Query-free range scorer used by workload-blind students. |
| `historical_prior_qds_model.py` | Query-free KNN historical-prior diagnostic model and prior-assisted student. |
| `workload_blind_range_v2.py` | Trainable factorized query-driven workload-blind model with prior-field and local/segment context encoders. |
| `../training/model_features.py` | Baseline, turn-aware, range-aware, and workload-blind point features. |

## Flow

```text
point features -> point encoder -> positional encoding -> local transformer
query features + query type IDs -> query encoder
point states + query states -> chunked cross-attention -> score head
```

Workload-blind neural range models skip the query branch. With `num_layers=0`,
they also skip positional encoding and the local transformer, reducing the path
to point features -> point encoder -> score head.

## Key Rules

- Query-conditioned models need `query_type_ids`; workload-blind models ignore
  query tensors and query type IDs.
- Current experiment paths train one pure workload per model and output one
  score per point.
- `query_chunk_size >= n_queries` gives exact one-chunk cross-attention;
  smaller chunks are an approximation.
- Query-conditioned attention is point-to-query and does not mix points across
  trajectories. The workload-blind neural model can use a local trajectory
  transformer, or skip it with `num_layers=0`.
- Point features are 7 columns for baseline, 8 for turn-aware, 16 for
  range-aware, 17 for the compact workload-blind path, 24 for `range_prior`,
  28 for `range_prior_clock_density` and `segment_context_range`, 35 for
  `workload_blind_range_v2`, and 23 for the historical-prior feature slice used
  by `historical_prior` and `historical_prior_student`.
  `historical_prior_mmsi` uses 27 columns by adding a deterministic query-free
  MMSI hash to the historical-prior slice.
- Query features are 12 padded columns from `pad_query_features`.

## Rework Classification

Legacy query-aware diagnostics:
- `baseline`
- `turn_aware`
- `range_aware`

Legacy workload-blind scalar scorers:
- `workload_blind_range`
- `range_prior`
- `range_prior_clock_density`
- `segment_context_range`

Historical-prior diagnostics:
- `historical_prior`
- `historical_prior_mmsi`
- `historical_prior_student`

`historical_prior` and `historical_prior_mmsi` are KNN diagnostics/teachers and
are not final learned-model success. `historical_prior_student` is trainable,
but it requires an ablation against the standalone KNN prior before it can claim
learned value. The current final-candidate model path is
`workload_blind_range_v2`, but it is not accepted unless the workload stability,
predictability, causality, global sanity, and full-grid gates pass.

Current density/sparsity feature columns are current-split point-cloud context
features. Do not call them train-derived query-prior fields.
