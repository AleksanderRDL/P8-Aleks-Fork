# QDS Code Layout

| Path | Purpose |
| --- | --- |
| `data/` | AIS loading, segmentation, caching, and trajectory boundaries. |
| `queries/` | Typed workloads, query generation, and query execution. |
| `models/` | Query-conditioned diagnostics, historical priors, and workload-blind scorers. |
| `training/` | Labels, batching, scaler, losses, checkpointing, inference. |
| `simplification/` | Score-to-mask conversion. |
| `evaluation/` | Baselines, metrics, audits, and tables. |
| `experiments/` | CLI, config, benchmark profiles, and run orchestration. |

Main entry point: `experiments.run_ais_experiment`.

Operational tooling reference: `docs/dev-tooling-guide.md`.
