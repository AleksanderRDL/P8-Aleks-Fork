# AIS Trajectory Query-Driven Simplification (QDS)

A machine learning research project that learns to compress AIS (Automatic
Identification System) vessel trajectory datasets while preserving the accuracy
of a spatiotemporal query workload.  The model identifies which trajectory
points are important based on how much they influence query results, and
discards the rest.

---

## Research Problem

AIS data streams report vessel positions (lat, lon, speed, heading) every few
minutes, producing large trajectory datasets.  Many downstream analytics tasks
can be answered from much smaller subsets of the data.

**Query-Driven Simplification** answers:

> *Which trajectory points can be removed without significantly changing the
> answers to a set of spatiotemporal queries?*

For a trajectory dataset **D** and a query workload **Q**, each point receives
an importance score:

```
importance_i = mean_q | result(D, q) - result(D \ {p_i}, q) |
```

The model learns to approximate this score from point features and the query
workload, enabling fast inference-time compression without re-running the
expensive leave-one-out computation.

---

## Model Architecture

```
Points [N, 5] ──► Point Encoder (5→64→64) ─────────────────────────────────► (+) ──► Importance Predictor (64→32→1→σ) ──► scores [N]
                                                                               ▲
Queries [M, 6] ─► Query Encoder (6→64→64) ──► Cross-Attention (Q attends K,V) ┘
                                              (mean over M queries)
```

| Component            | Architecture                               |
|----------------------|--------------------------------------------|
| Point Encoder        | Linear(5→64) → ReLU → Linear(64→64)        |
| Query Encoder        | Linear(6→64) → ReLU → Linear(64→64)        |
| Cross-Attention      | MultiheadAttention(embed=64, heads=4)      |
| Importance Predictor | Linear(64→32) → ReLU → Linear(32→1) → σ   |

Point columns: `[time, lat, lon, speed, heading]`  
Query columns: `[lat_min, lat_max, lon_min, lon_max, time_start, time_end]`  
Query result: SUM of speed for all points inside the query rectangle.

---

## Repository Structure

```
qds_project/
├── requirements.txt
├── src/
│   ├── data/
│   │   ├── ais_loader.py          # load_ais_csv / generate_synthetic_ais_data
│   │   └── trajectory_dataset.py  # TrajectoryDataset (PyTorch Dataset)
│   ├── queries/
│   │   ├── query_generator.py     # uniform / density-biased / mixed workloads
│   │   └── query_executor.py      # run_query / run_queries
│   ├── models/
│   │   └── trajectory_qds_model.py  # TrajectoryQDSModel (nn.Module)
│   ├── training/
│   │   ├── importance_labels.py   # compute_importance (leave-one-out)
│   │   └── train_model.py         # training loop + CLI
│   ├── simplification/
│   │   └── simplify_trajectories.py  # simplify_trajectories(...)
│   ├── evaluation/
│   │   ├── metrics.py             # query_error / compression_ratio / query_latency
│   │   └── baselines.py           # random / temporal / Douglas-Peucker baselines
│   ├── visualization/
│   │   ├── trajectory_visualizer.py   # plot_trajectories / plot_queries_on_trajectories
│   │   └── importance_visualizer.py   # plot_importance / plot_simplification_results
│   └── experiments/
│       └── run_ais_experiment.py  # end-to-end experiment pipeline
└── tests/
    ├── test_data.py
    ├── test_query_executor.py
    ├── test_query_generator.py
    ├── test_model.py
    ├── test_metrics.py
    ├── test_baselines.py
    └── test_simplification.py
```

---

## Installation

```bash
pip install -r qds_project/requirements.txt
```

---

## Quick Start

### Run the full end-to-end experiment (synthetic AIS data)

```bash
cd qds_project
python -m src.experiments.run_ais_experiment \
    --n_ships 10 \
    --n_points 100 \
    --n_queries 50 \
    --epochs 30 \
    --threshold 0.5
```

Use automatic threshold selection by target retained ratio:

```bash
cd qds_project
python -m src.experiments.run_ais_experiment \
    --n_ships 50 \
    --n_points 150 \
    --n_queries 150 \
    --target_ratio 0.10
```

Choose query workload type (`uniform`, `density`, `mixed`, or `all`):

```bash
cd qds_project
python -m src.experiments.run_ais_experiment --workload density --n_queries 100
```

### Use real AIS data (CSV)

Supported column aliases:
- `mmsi`
- `lat` or `latitude`
- `lon` or `longitude`
- `speed` or `sog`
- `heading` or `cog` (optional)
- `timestamp` / `time` / `datetime` (optional)

```bash
cd qds_project
python -m src.experiments.run_ais_experiment \
    --csv_path /path/to/ais_data.csv \
    --n_queries 100 \
    --epochs 50
```

Retained points are exported to `MLClean-<original_filename>.csv` next to the
input file.

### Train only

```bash
cd qds_project
python -m src.training.train_model \
    --n_ships 20 \
    --n_points 200 \
    --n_queries 100 \
    --epochs 50 \
    --save_path results/model.pt
```

---

## Running Tests

```bash
cd qds_project
python -m pytest tests/ -v
```

---

## Configuration

All scripts accept command-line arguments.  Key parameters:

| Parameter           | Default | Description                                      |
|---------------------|---------|--------------------------------------------------|
| `--n_ships`         | 10      | Number of synthetic vessels                      |
| `--n_points`        | 100     | Points per vessel trajectory                     |
| `--n_queries`       | 100     | Number of spatiotemporal queries                 |
| `--epochs`          | 50      | Training epochs                                  |
| `--lr`              | 1e-3    | Learning rate                                    |
| `--threshold`       | 0.5     | Importance threshold for simplification          |
| `--target_ratio`    | None    | Auto-select threshold to retain this fraction    |
| `--workload`        | density | Query workload: `uniform`, `density`, `mixed`, `all` |
| `--density_ratio`   | 0.7     | Fraction of density-biased queries (mixed mode)  |
| `--csv_path`        | None    | Path to real AIS CSV file                        |
| `--max_train_points`| None    | Cap training points (for large datasets)         |

---

## Evaluation Metrics

| Metric            | Formula                                                |
|-------------------|--------------------------------------------------------|
| Query Error       | mean_q \|orig(q) - simp(q)\| / (\|orig(q)\| + 1e-8)  |
| Compression Ratio | \|simplified\| / \|original\|                         |
| Query Latency     | Average wall-clock time per query (seconds)            |

---

## Baseline Methods

| Baseline               | Description                                                |
|------------------------|------------------------------------------------------------|
| Random Sampling        | Uniformly random subset                                    |
| Uniform Temporal       | Every k-th point sorted by time                            |
| Douglas-Peucker        | Recursive line simplification on lat/lon coordinates       |
| ML QDS                 | Learned importance scores (this project)                   |

---

## Output Visualizations

When running the experiment, visualizations are saved to the system temporary
directory and `results/`:

- `ais_trajectories.png` — vessel paths in lat/lon space
- `ais_queries.png` — trajectories + semi-transparent query rectangles
- `ais_importance.png` — scatter plot coloured by importance score
- `ais_combined.png` — combined: lines + importance colours + queries
- `results/simplification_visualization.png` — simplification and query overlay
- `results/simplification_time_slices.png` — 4 time-window panels

---

## Python API

```python
import sys
sys.path.insert(0, 'qds_project')

from src.data.ais_loader import generate_synthetic_ais_data, load_ais_csv
from src.data.trajectory_dataset import TrajectoryDataset
from src.queries.query_generator import generate_spatiotemporal_queries
from src.queries.query_executor import run_queries
from src.models.trajectory_qds_model import TrajectoryQDSModel
from src.training.importance_labels import compute_importance
from src.training.train_model import train_model
from src.simplification.simplify_trajectories import simplify_trajectories
from src.evaluation.metrics import query_error, compression_ratio

# 1. Generate or load AIS trajectory data
trajectories = generate_synthetic_ais_data(n_ships=10, n_points_per_ship=100)

# 2. Get flat point cloud
ds = TrajectoryDataset(trajectories)
points = ds.get_all_points()          # [N, 5]

# 3. Generate spatiotemporal query workload
queries = generate_spatiotemporal_queries(trajectories, n_queries=100)  # [M, 6]

# 4. Train QDS model
model = train_model(trajectories, queries, epochs=50)

# 5. Simplify trajectories
simplified, mask, scores = simplify_trajectories(points, model, queries, threshold=0.5)

# 6. Evaluate
print(f"Query error:       {query_error(points, simplified, queries):.4f}")
print(f"Compression ratio: {compression_ratio(points, simplified):.4f}")
```
