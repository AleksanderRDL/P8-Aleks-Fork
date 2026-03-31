# AIS Data (`AISDATA/`)

Dataset folder for raw and cleaned AIS CSV files used by the root pipeline.

## Expected Conventions

- Raw input file default: `aisdk-2026-02-05.csv`
- Cleaned output default: `aisdk-2026-02-05.cleaned.csv/` (Spark CSV output directory)

## Notes

- This folder can contain very large files.
- Root pipeline (`main.py`) reads and writes here by default unless overridden with:
  - `AIS_INPUT_FILE`
  - `AIS_OUTPUT_PATH`

## Related Docs

- [`../README.md`](../README.md) for root pipeline quick start.
- [`../scripts/README.md`](../scripts/README.md) for CSV import/query scripts.
