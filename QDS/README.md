# AIS-QDS

AIS-QDS trains query-conditioned simplification models for AIS trajectories and
evaluates retained-point sets against pure query workloads. The current sprint
focus is the range workload on cleaned AIS CSV days.

## Where To Look

| Need | File |
| --- | --- |
| Experiment CLI, benchmark profiles, tmux launchers | [`src/experiments/README.md`](src/experiments/README.md) |
| Generated artifacts, benchmark families, cleanup policy | [`artifacts/README.md`](artifacts/README.md) |
| Training labels, checkpoint selection, runtime knobs | [`src/training/README.md`](src/training/README.md) |
| CSV loading, segmentation, Parquet cache | [`src/data/README.md`](src/data/README.md) |
| Query generation and range footprint controls | [`src/queries/README.md`](src/queries/README.md) |
| Evaluation metrics and baseline methods | [`src/evaluation/README.md`](src/evaluation/README.md) |
| Model architecture | [`src/models/README.md`](src/models/README.md) |

The current research direction for range-query usefulness is summarized in
[`../Aleks-Sprint/range-objective-redesign.md`](../Aleks-Sprint/range-objective-redesign.md).

## Quick Start

Run commands from `QDS/`. The expected local interpreter is the repository-level
virtual environment at `../.venv`.

```bash
cd QDS
../.venv/bin/python -m pip install -r requirements.txt
make check-env
make test
make typecheck
```

Run a tiny synthetic smoke experiment:

```bash
make smoke
```

Run a cleaned-CSV smoke experiment:

```bash
make smoke-csv CLEANED_CSV=../AISDATA/cleaned/<cleaned-ais-file-or-dir>
```

Run the main experiment CLI directly:

```bash
../.venv/bin/python -m src.experiments.run_ais_experiment \
  --csv_path ../AISDATA/cleaned/<cleaned-ais-file.csv> \
  --cache_dir artifacts/cache/manual_csv \
  --workload range \
  --n_queries 128 \
  --epochs 6 \
  --compression_ratio 0.10 \
  --results_dir artifacts/results/manual_range
```

## Benchmark Commands

Before expensive runs:

```bash
make benchmark-preflight
```

Launch one testing-baseline range benchmark in tmux:

```bash
ATTACH=0 BENCHMARK_RUN_ID=range_testing_baseline_a make range-benchmark-tmux
```

Launch a sequential multi-seed queue:

```bash
make benchmark-queue-preflight
ATTACH=0 BENCHMARK_SEEDS=42,43,44 make range-benchmark-queue-tmux
```

Inspect and clean generated artifacts:

```bash
make list-runs
make clean-smoke-artifacts
make clean-smoke-artifacts CONFIRM=1
```

The active benchmark profile is `range_testing_baseline`: a range-only, three-day
cleaned-CSV profile with train, checkpoint-validation, and final-eval days and
no loader caps by default. Keep exact benchmark
settings in one place: [`src/experiments/README.md`](src/experiments/README.md).

## Dependencies

Dependency profiles are split by Torch target:

- [`requirements-common.txt`](requirements-common.txt) - shared non-Torch dependencies.
- [`requirements-cpu.txt`](requirements-cpu.txt) - CPU/generic Torch profile.
- [`requirements-cuda-cu130.txt`](requirements-cuda-cu130.txt) - CUDA reference profile.
- [`requirements.txt`](requirements.txt) - compatibility alias for the current CUDA sprint profile.

The local CUDA reference stack observed for this machine:

```text
torch 2.11.0+cu130
CUDA runtime 13.0
triton 3.6.0
```

Install a specific profile when changing environments:

```bash
../.venv/bin/python -m pip install -r requirements-cuda-cu130.txt
```

## Architecture

The pipeline is `data -> queries -> training/models -> simplification ->
evaluation -> experiments`. File-level behavior lives in the module READMEs
linked above.

## Outputs

Experiment and benchmark outputs should stay under `artifacts/` unless there is
a specific reason to use another local path. Start with
[`artifacts/README.md`](artifacts/README.md), then the run-local `README.md`,
`artifact_index.json`, or `benchmark_matrix.md`.
