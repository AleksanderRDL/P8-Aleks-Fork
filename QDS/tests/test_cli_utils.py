"""Tests for shared CLI argument normalization."""

from __future__ import annotations

from experiments.cli_utils import normalized_gap_arg, split_csv_path_list


def test_normalized_gap_arg_keeps_positive_values() -> None:
    assert normalized_gap_arg(3600) == 3600.0
    assert normalized_gap_arg(0.5) == 0.5


def test_normalized_gap_arg_disables_non_positive_and_none_values() -> None:
    assert normalized_gap_arg(None) is None
    assert normalized_gap_arg(0) is None
    assert normalized_gap_arg(-1) is None


def test_split_csv_path_list_preserves_order_and_drops_blank_items() -> None:
    assert split_csv_path_list(None) == ()
    assert split_csv_path_list(" train.csv, ,validation.csv,eval.csv ") == (
        "train.csv",
        "validation.csv",
        "eval.csv",
    )
