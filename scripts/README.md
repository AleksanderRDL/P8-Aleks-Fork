# Scripts (`scripts/`)

Operational command-line tools for database validation, CSV import, and query checks.

## Prerequisites

- Database running (`db/compose.yaml`).
- `DATABASE_URL` set (for example via `.env`).
- Python dependencies installed from repository root.

## Quick DB Smoke Test

Validates DB connectivity, PostGIS, and table presence:

```bash
python scripts/smoke_test_db.py
```

## CSV Import (`import_ais_csv.py`)

Imports cleaned AIS CSV into `ais_points_cleaned` with resumable progress.

Basic import:

```bash
python scripts/import_ais_csv.py AISDATA/aisdk-2026-02-05.cleaned.csv
```

Tune worker/chunk settings:

```bash
python scripts/import_ais_csv.py AISDATA/aisdk-2026-02-05.cleaned.csv \
  --workers 4 \
  --chunk-rows 200000 \
  --copy-buffer-rows 50000
```

Use filename only (auto-resolves from `AISDATA/`):

```bash
python scripts/import_ais_csv.py aisdk-2026-02-05.cleaned.csv
```

Resume behavior:

- Default resumes from `ais_import_progress`.
- `--reset-progress` restarts for the same file path.
- `--no-resume` ignores stored progress for current run.

Examples:

```bash
python scripts/import_ais_csv.py AISDATA/aisdk-2026-02-05.cleaned.csv --reset-progress
python scripts/import_ais_csv.py AISDATA/aisdk-2026-02-05.cleaned.csv --no-resume
python scripts/import_ais_csv.py AISDATA/aisdk-2026-02-05.cleaned.csv --limit 100000
```

## Range Query Validation (`run_range_query.py`)

Runs temporal + bounding-box summary queries against `ais_points_cleaned`.

Summary query:

```bash
python scripts/run_range_query.py \
  --t0 2026-02-05T10:00:00Z \
  --t1 2026-02-05T11:00:00Z \
  --min-lon 10.0 --min-lat 56.5 \
  --max-lon 11.0 --max-lat 57.5
```

Include MMSI sample list:

```bash
python scripts/run_range_query.py \
  --t0 2026-02-05T10:00:00Z \
  --t1 2026-02-05T11:00:00Z \
  --min-lon 10.0 --min-lat 56.5 \
  --max-lon 11.0 --max-lat 57.5 \
  --list-mmsi --list-limit 50
```

Explain plan:

```bash
python scripts/run_range_query.py \
  --t0 2026-02-05T10:00:00Z \
  --t1 2026-02-05T11:00:00Z \
  --min-lon 10.0 --min-lat 56.5 \
  --max-lon 11.0 --max-lat 57.5 \
  --explain
```

## Related Docs

- [`../db/README.md`](../db/README.md) for database lifecycle and schema checks.
- [`../AISDATA/README.md`](../AISDATA/README.md) for dataset file conventions.
