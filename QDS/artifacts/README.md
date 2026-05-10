# QDS Artifacts

This directory is for local generated artifacts. Keep run output here, but do
not commit benchmark payloads, caches, checkpoints, logs, or smoke leftovers.
Only this README is tracked.

## Layout

Use these top-level folders:

- `benchmarks/` - benchmark families and one directory per benchmark attempt.
- `cache/` - reusable loader/cache material such as segmented Parquet data.
- `results/` - ad hoc experiment and smoke-run result directories.

The recommended range benchmark family is:

```text
artifacts/benchmarks/range_workload_matrix_min_realistic/
  latest_run.txt
  runs_index.csv
  runs_index_events.jsonl
  runs/<run_id>/
    README.md
    run_config.json
    run_status.json
    artifact_index.json
    benchmark_matrix.json
    benchmark_matrix.csv
    benchmark_matrix.md
    logs/
    variants/
```

Read `runs_index.csv` first when comparing attempts. Read a run-local
`artifact_index.json` or `README.md` first when looking for a specific child
output, log, or diagnostic file.

## Run IDs

Use stable, descriptive run IDs for benchmark attempts that may be compared
later. Include the workload, profile, data span, point cap, and an iteration
suffix when useful:

```bash
BENCHMARK_RUN_ID=range_medium_2day_cap3000_a make range-benchmark-tmux
```

Timestamped default IDs are fine for exploratory runs. Avoid reusing a run ID
unless you intend to overwrite the previous run directory and family index row.

## Routine Commands

Run preflight before starting an expensive tmux benchmark:

```bash
make benchmark-preflight
```

List benchmark attempts in the active family:

```bash
make list-runs
```

Dry-run smoke artifact cleanup:

```bash
make clean-smoke-artifacts
```

Delete the known smoke/test artifact directories:

```bash
make clean-smoke-artifacts CONFIRM=1
```

## Cleanup Rules

It is safe to delete:

- `artifacts/results/smoke_*`
- `artifacts/benchmarks/*smoke*`
- `artifacts/benchmarks/*layout_smoke*`
- cache directories created only for smoke runs

Keep benchmark family roots that contain serious runs until their comparison
tables and run notes have been reviewed.
