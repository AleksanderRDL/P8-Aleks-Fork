# AIS-QDS

AIS-QDS trains and evaluates trajectory simplification models for AIS data.
The active long-term target is workload-blind range compression: compress once,
before future range queries are known, then score the frozen retained set.

The implemented strong range profile is currently
`range_workload_aware_diagnostic`. It is useful as a diagnostic/teacher path,
but it is not final workload-blind evidence.
The canonical redesign source of truth is
[`docs/query-driven-rework-guide.md`](docs/query-driven-rework-guide.md), with
checkpoint state in
[`docs/query-driven-rework-progress.md`](docs/query-driven-rework-progress.md).

## Setup

Run commands from `Range_QDS/`. The project uses `uv` from the repository root
with the `dev` dependency group.

```bash
cd Range_QDS
make sync
make lock-check
make check-env
make test
```

## Common Commands

```bash
make typecheck
make lint
make lint-yaml
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
cd ..
uv run --group dev -- python -m experiments.run_ais_experiment \
  --csv_path AISDATA/cleaned/<cleaned-ais-file.csv> \
  --cache_dir Range_QDS/artifacts/cache/manual_csv \
  --workload range \
  --n_queries 128 \
  --epochs 6 \
  --compression_ratio 0.10 \
  --results_dir Range_QDS/artifacts/results/manual_range
```

## Where To Look

| Need | File |
| --- | --- |
| Redesign objective and acceptance criteria | [`docs/query-driven-rework-guide.md`](docs/query-driven-rework-guide.md) |
| Code layout | [`CODE_LAYOUT.md`](CODE_LAYOUT.md) |
| Benchmark profile, CLI modes, artifact names | [`experiments/README.md`](experiments/README.md) |
| Generated artifact layout and cleanup | [`artifacts/README.md`](artifacts/README.md) |
| Training labels, loss, checkpoint selection | [`training/README.md`](training/README.md) |
| Query generation and execution | [`queries/README.md`](queries/README.md) |
| Evaluation metrics and baselines | [`evaluation/README.md`](evaluation/README.md) |
| Data loading and segmented cache | [`data/README.md`](data/README.md) |
| Model architecture | [`models/README.md`](models/README.md) |

## Requirements

Dependency source of truth lives in root [`../pyproject.toml`](../pyproject.toml).
Use `uv sync --group dev` from the repo root.

## Output Policy

Experiment and benchmark output should stay under `artifacts/` unless a run
explicitly needs another local path. Source data belongs under `../AISDATA/`;
model outputs do not.
