# QDS Tests (`tests/`)

Test suite for the QDS project.

## Categories

Pytest markers:

- `unit`: fast isolated tests (default for uncategorized tests).
- `integration`: cross-module orchestration and interaction tests.
- `slow`: higher-cost tests (for example training paths).

Default execution order is enforced as:

1. `unit`
2. `integration`
3. `slow`

## Common Commands

From `QDS/qds_project`:

```bash
python -m pytest tests/ -q
python -m pytest tests/ -m "not slow" -q
python -m pytest tests/ -m unit -q
python -m pytest tests/ -m integration -q
python -m pytest tests/ -m slow -q
```
