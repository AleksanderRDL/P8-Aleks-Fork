# P8

Mobility project centered on AIS data, with two main workstreams:

- Data cleaning and database ingestion/query tooling (repository root)
- Machine-learning query-driven simplification research in [`QDS/`](QDS/)

## Documentation Convention

This repository uses folder-local `README.md` files as the source of
documentation.

- Each folder with project context relevant for documentation should contain its own `README.md`.

## Documentation Map

- [`QDS/README.md`](QDS/README.md): ML simplification project overview.
- [`QDS/qds_project/README.md`](QDS/qds_project/README.md): full QDS technical docs.
- [`scripts/README.md`](scripts/README.md): import and query CLI tools.
- [`db/README.md`](db/README.md): PostGIS setup, lifecycle, and SQL checks.
- [`environment_setup/README.md`](environment_setup/README.md): Java/Hadoop/Spark bootstrap helpers.
- [`AISDATA/README.md`](AISDATA/README.md): data folder conventions and expected files.

## Quick Start (Root Pipeline)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```
