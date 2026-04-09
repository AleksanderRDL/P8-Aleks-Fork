# AIS Trajectory Query-Driven Simplification

A machine learning research project that compresses AIS vessel trajectory
datasets while preserving the accuracy of a spatiotemporal query workload,
implemented in Python + PyTorch.

See [`qds_project/README.md`](qds_project/README.md) for full documentation,
including architecture details, configuration parameters, baseline comparisons,
and the Python API.

---

## Project Goal

Learn to identify which AIS trajectory points can be safely removed without
significantly changing the answers to a set of spatiotemporal range queries.

---

## Pipeline Overview

1. **Data** — load real AIS CSV files or generate synthetic data
2. **Queries** — generate a spatiotemporal query workload (uniform / density-biased / mixed)
3. **Labels** — compute ground-truth importance via leave-one-out query error
4. **Training** — train a cross-attention QDS model to predict importance scores
5. **Simplification** — retain top-k points per trajectory using model predictions
6. **Evaluation** — compare query error, compression ratio, and latency against baselines

---

## Available Model Variants

| Variant               | Input Features | Description                                               |
|-----------------------|----------------|-----------------------------------------------------------|
| Baseline              | 7              | `TrajectoryQDSModel` — standard QDS model                 |
| Turn-aware            | 8              | `TurnAwareQDSModel` — adds `turn_score` feature           |
| Boundary-aware turn   | 9              | `BoundaryAwareTurnModel` — adds `boundary_proximity` too  |

---

## Quick Start

```bash
pip install -r qds_project/requirements.txt
cd qds_project
python -m src.experiments.run_ais_experiment
```

---

## Repository Structure

```
QDS/
├── README.md
└── qds_project/
    ├── README.md               ← full documentation
    ├── requirements.txt
    ├── src/
    │   ├── data/               ← AIS loading; README.md
    │   ├── queries/            ← query generation/execution; README.md
    │   │   ├── query_generator.py   ← range + typed query generators
    │   │   ├── query_executor.py    ← vectorised range query execution
    │   │   └── query_types.py       ← typed query dispatcher & executors
    │   ├── models/             ← model architectures; README.md
    │   ├── training/           ← importance labels, training loop; README.md
    │   ├── simplification/     ← simplification logic; README.md
    │   ├── evaluation/         ← metrics, baselines; README.md
    │   ├── visualization/      ← plotting utilities; README.md
    │   └── experiments/        ← end-to-end experiment; README.md
    └── tests/
```
