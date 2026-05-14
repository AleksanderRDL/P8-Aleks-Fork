# Simplification Module

Turns per-point scores into retained-point masks.

## Files

| File | Purpose |
| --- | --- |
| `mlqds_scoring.py` | Canonical score conversion used by validation and final evaluation. |
| `simplify_trajectories.py` | Per-trajectory top-k retention with endpoint preservation. |

## Rules

- Selection is trajectory-local; there is no global threshold across ships.
- Endpoints are preserved when a trajectory has retained points.
- Equal scores get deterministic pseudo-random jitter to avoid positional bias.
- Supported MLQDS score modes include `rank`, `rank_tie`, `raw`, `sigmoid`,
  `temperature_sigmoid`, `zscore_sigmoid`, and `rank_confidence`.
