# Models Module

Defines neural network architectures for predicting per-point importance scores
given AIS trajectory data and a spatiotemporal query workload.

---

## Architecture Overview

Both models share the same four-stage architecture:

```
Points [N, F] ──► Point Encoder (F→64→64) ──────────────────────────────────► (+) ──► Importance Predictor (64→32→1→σ) ──► scores [N]
                                                                                ▲
Queries [M, 6] ─► Query Encoder (6→64→64) ──► Cross-Attention (Q attends K,V) ┘
                                               (weighted mean over M queries)
```

| Component            | Architecture                               |
|----------------------|--------------------------------------------|
| Point Encoder        | Linear(F→64) → ReLU → Linear(64→64)        |
| Query Encoder        | Linear(6→64) → ReLU → Linear(64→64)        |
| Cross-Attention      | MultiheadAttention(embed=64, heads=4)      |
| Importance Predictor | Linear(64→32) → ReLU → Linear(32→1) → σ   |

**Cross-Attention mechanism**: Queries (Q) attend to point embeddings (K, V).
The resulting attention weights `[M, N]` are transposed to `[N, M]` and
matrix-multiplied with the attended query embeddings `[M, D]`, producing a
per-point query-conditioned context vector `[N, D]`. This is added to the
original point embeddings before predicting importance scores.

---

## Components

### `attention_qds_model_base.py`

**`AttentionQDSModelBase`** — shared architecture backbone used by both model
variants.

- Defines the point encoder, query encoder, cross-attention block, and
  importance predictor once.
- Provides the common forward pass that computes per-point query-conditioned
  context vectors and outputs `[N]` importance scores.

---

### `trajectory_qds_model.py`

**`TrajectoryQDSModel`** — Baseline QDS model.

- Point encoder input: **7 features** — `[time, lat, lon, speed, heading, is_start, is_end]`
- Query encoder input: **6 features** — `[lat_min, lat_max, lon_min, lon_max, time_start, time_end]`
- Output: `[N]` importance scores in `[0, 1]`

**`normalize_points_and_queries(points, queries)`**  
Min-max normalises points and queries using point-cloud feature ranges so that
training and inference inputs are on consistent scales. The same scaling is
applied to semantically matching fields (time, lat, lon). Binary features
`is_start` and `is_end` (columns 5–6) and `turn_score` (column 7, if present)
are already in `[0, 1]` and are preserved unchanged.

---

### `turn_aware_qds_model.py`

**`TurnAwareQDSModel`** — Turn-aware QDS model variant.

Identical architecture to `TrajectoryQDSModel` but accepts an extended
**8-feature** point vector that includes `turn_score` as the final column.

- Point encoder input: **8 features** — `[time, lat, lon, speed, heading, is_start, is_end, turn_score]`
- Query encoder input: **6 features** (same as baseline)
- Output: `[N]` importance scores in `[0, 1]`

The `turn_score` feature allows the model to learn that points at trajectory
bends may carry structural importance independent of the query workload,
producing simplified trajectories that better preserve trajectory shapes.
