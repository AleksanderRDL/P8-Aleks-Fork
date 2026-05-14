"""Shared CLI argument normalization helpers."""

from __future__ import annotations


def normalized_gap_arg(value: float | None) -> float | None:
    """Normalize CLI gap controls so <=0 disables time-gap segmentation."""
    if value is None:
        return None
    value = float(value)
    return None if value <= 0.0 else value


def split_csv_path_list(raw: str | None) -> tuple[str, ...]:
    """Parse a comma-separated CSV path list while preserving order."""
    if raw is None:
        return ()
    return tuple(part.strip() for part in str(raw).split(",") if part.strip())
