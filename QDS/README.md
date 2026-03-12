# AIS Trajectory Query-Driven Simplification

A machine learning research project that compresses AIS vessel trajectory
datasets while preserving the accuracy of a spatiotemporal query workload,
implemented in Python + PyTorch.

See [`qds_project/README.md`](qds_project/README.md) for full documentation.

## Quick Start

```bash
pip install -r qds_project/requirements.txt
cd qds_project
python -m src.experiments.run_ais_experiment
```
