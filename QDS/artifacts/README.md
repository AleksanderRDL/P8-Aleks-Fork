# QDS Artifacts

This directory is for local generated outputs. Keep caches, checkpoints,
benchmark logs, and smoke leftovers out of git; only this README should be
tracked.

## Layout

```text
artifacts/
  benchmarks/
    range_testing_baseline/
      latest_run.txt
      latest_queue.txt
      runs_index.csv
      runs_index_events.jsonl
      runs/<run_id>/
      queues/<queue_id>/
  cache/
  results/
```

- `benchmarks/` contains comparable benchmark families.
- `cache/` contains reusable segmented trajectory and range diagnostic caches.
- `results/` is for ad hoc smoke/manual experiment output.

Within a benchmark run, start with `README.md`, `artifact_index.json`,
`run_status.json`, `benchmark_report.md`, and `benchmark_report.csv`.
Child experiment details live under the run-label subdirectory, usually
`range_testing_baseline/`.

## Useful Commands

```bash
make benchmark-preflight
make benchmark-queue-preflight
make list-runs
make clean-smoke-artifacts
make clean-smoke-artifacts CONFIRM=1
```

Preflight checks tmux, Python/Torch, cleaned CSV availability, artifact/cache
writes, disk, RAM, swap, GPU visibility, and dirty git state. RAM/swap and
dirty-git findings are warnings so intentional runs can still proceed.

## Run IDs

Use descriptive run IDs for runs that may be compared later:

```bash
ATTACH=0 BENCHMARK_RUN_ID=range_testing_baseline_seed42_a make range-benchmark-tmux
```

Timestamped IDs are fine for exploratory runs. Avoid reusing a run ID unless
overwriting that run directory and index row is intentional.

## Cleanup

Safe cleanup targets:

- `artifacts/results/smoke_*`
- `artifacts/results/post_training_runtime_smoke`
- `artifacts/benchmarks/*smoke*`
- `artifacts/benchmarks/*layout_smoke*`
- old task smoke directories such as `artifacts/benchmarks/task*_smoke`
  and `artifacts/benchmarks/task*_small`
- caches created only for smoke runs
- stale workload-aware diagnostic caches, especially
  `artifacts/cache/range_testing_baseline/range_diagnostics/`, after their
  report numbers are captured in notes

Keep `artifacts/benchmarks/range_testing_baseline/` runs until their report
rows have been reviewed or intentionally archived elsewhere.
