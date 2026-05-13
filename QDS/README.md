# AIS-QDS

AIS-QDS trains and evaluates trajectory simplification models for AIS data.
The active long-term target is workload-blind range compression: compress once,
before future range queries are known, then score the frozen retained set.

The implemented strong range profile is currently
`range_workload_aware_diagnostic`. It is useful as a diagnostic/teacher path,
but it is not final workload-blind evidence.

## Setup

Run commands from `QDS/`. Use a canonical venv path; Python 3.14 warns if the
interpreter path contains `..`.

```bash
cd QDS
PYTHON="$(cd .. && pwd -P)/.venv/bin/python"
"$PYTHON" -m pip install -r requirements.txt
make check-env
make test
```

## Common Commands

```bash
make typecheck
make smoke
make smoke-csv CLEANED_CSV=../AISDATA/cleaned/<file-or-directory>
make benchmark-preflight
ATTACH=0 BENCHMARK_RUN_ID=range_workload_aware_diagnostic_a make range-benchmark-tmux
ATTACH=0 BENCHMARK_SEEDS=42,43,44 make range-benchmark-queue-tmux
make list-runs
make clean-smoke-artifacts CONFIRM=1
```

Direct CLI example:

```bash
"$PYTHON" -m src.experiments.run_ais_experiment \
  --csv_path ../AISDATA/cleaned/<cleaned-ais-file.csv> \
  --cache_dir artifacts/cache/manual_csv \
  --workload range \
  --n_queries 128 \
  --epochs 6 \
  --compression_ratio 0.10 \
  --results_dir artifacts/results/manual_range
```

## Where To Look

| Need | File |
| --- | --- |
| Redesign objective and acceptance criteria | [`../Aleks-Sprint/range-training-redesign.md`](../Aleks-Sprint/range-training-redesign.md) |
| Benchmark profile, CLI modes, artifact names | [`src/experiments/README.md`](src/experiments/README.md) |
| Generated artifact layout and cleanup | [`artifacts/README.md`](artifacts/README.md) |
| Training labels, loss, checkpoint selection | [`src/training/README.md`](src/training/README.md) |
| Query generation and execution | [`src/queries/README.md`](src/queries/README.md) |
| Evaluation metrics and baselines | [`src/evaluation/README.md`](src/evaluation/README.md) |
| Data loading and segmented cache | [`src/data/README.md`](src/data/README.md) |
| Model architecture | [`src/models/README.md`](src/models/README.md) |

## Requirements

- `requirements-common.txt`: shared non-Torch dependencies.
- `requirements-cpu.txt`: CPU/generic Torch profile.
- `requirements-cuda-cu130.txt`: CUDA reference profile.
- `requirements.txt`: compatibility alias for the current CUDA sprint profile.

Install a specific profile when changing environments:

```bash
"$PYTHON" -m pip install -r requirements-cuda-cu130.txt
```

## Output Policy

Experiment and benchmark output should stay under `artifacts/` unless a run
explicitly needs another local path. Source data belongs under `../AISDATA/`;
model outputs do not.
