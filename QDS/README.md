# QDS Repository

This repository contains two versions of the AIS query-driven simplification project:

- `qds_project/` - the legacy v1 implementation and documentation.
- `qds_v2/` - the current shift-aware rebuild with typed query workloads and distribution-shift evaluation.

Start with `qds_v2/README.md` if you want the active system. Use `qds_project/README.md` when you need the original architecture for comparison.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `qds_project/` | Original pipeline, tests, and results. |
| `qds_v2/` | Rebuilt pipeline, tests, and results. |
| `qds_project/requirements.txt` | Dependencies for the legacy system. |
| `qds_v2/requirements.txt` | Dependencies for the v2 stack. |

## How The Two Trees Differ

- `qds_project` documents the earlier query-driven simplification pipeline.
- `qds_v2` adds typed query workloads, query-conditioned training, shift-aware evaluation, and the turn-aware model variant.
- Both trees keep their own `src/`, `tests/`, and `results/` folders so experiments stay isolated.

## Outputs

In `qds_v2`, the main experiment artifacts are `results/example_run.json`, `results/matched_table.txt`, and `results/shift_table.txt`.

## Validation

Each tree ships with its own regression suite. In `qds_v2`, the tests cover attention leakage, query type ID requirements, scaler persistence, top-k behavior, in-distribution gains, and training stability.
