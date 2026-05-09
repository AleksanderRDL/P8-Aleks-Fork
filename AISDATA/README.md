# AIS Data (`AISDATA/`)

Dataset folder for AIS data used by the root pipeline and related tools.

## Expected Conventions

- Raw AIS files go under [`raw/`](raw/).
- Cleaned AIS files go under [`cleaned/`](cleaned/).
- Set `AIS_INPUT_FILE` to a raw AIS CSV under `raw/`.
- Set `AIS_OUTPUT_PATH` to a cleaned Spark CSV output directory under `cleaned/`.

## Notes

- This folder can contain very large files.
- The `raw/` and `cleaned/` folders are kept in git, but their data contents are ignored.
- Root pipeline (`main.py`, backed by `ais_pipeline/pipeline.py`) reads and writes here unless overridden with:
  - `AIS_INPUT_FILE`
  - `AIS_OUTPUT_PATH`

## Related Docs

- [`../README.md`](../README.md) for root pipeline quick start.
- [`../db/README.md`](../db/README.md) for CSV import/query and database scripts.
