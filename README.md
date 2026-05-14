# P8

AIS data tooling and query-driven trajectory simplification research.

## Workstreams

| Area | Path | Purpose |
| --- | --- | --- |
| Cleaning pipeline | [`ais_pipeline/`](ais_pipeline/) | Spark-based AIS CSV cleaning. |
| Database tools | [`db/`](db/) | Local PostGIS setup, CSV import, and range-query checks. |
| QDS research | [`QDS/`](QDS/) | ML trajectory simplification, benchmarks, and redesign work. |
| Data folders | [`AISDATA/`](AISDATA/) | Raw and cleaned AIS source data. |
| Sprint notes | [`Sprint/`](Sprint/) | Active QDS redesign reference and short progress log. |

## Quick Start

Root cleaning pipeline:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python main.py
```

QDS work:

```bash
cd QDS
PYTHON="$(cd .. && pwd -P)/.venv/bin/python"
make check-env
make test
```

Database helpers:

```bash
make db-up
make db-smoke
make db-import CSV=AISDATA/cleaned/<file-or-directory>
make db-query QUERY_ARGS="--help"
```

## Documentation

- [`QDS/README.md`](QDS/README.md): QDS usage and where to look next.
- [`Sprint/range-training-redesign.md`](Sprint/range-training-redesign.md): current range-training redesign.
- [`AISDATA/README.md`](AISDATA/README.md): data folder conventions.
- [`ais_pipeline/README.md`](ais_pipeline/README.md): cleaning pipeline layout.
- [`db/README.md`](db/README.md): database lifecycle and scripts.
