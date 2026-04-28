"""Tests F1-based query metrics. See src/evaluation/README.md for details."""

from __future__ import annotations

import pytest
import torch

from src.evaluation.evaluate_methods import evaluate_method, print_method_comparison_table
from src.evaluation.metrics import MethodEvaluation, clustering_f1, f1_score


class KeepAllMethod:
    name = "KeepAll"

    def simplify(
        self,
        points: torch.Tensor,
        boundaries: list[tuple[int, int]],
        compression_ratio: float,
    ) -> torch.Tensor:
        return torch.ones((points.shape[0],), dtype=torch.bool, device=points.device)


class DropAllMethod:
    name = "DropAll"

    def simplify(
        self,
        points: torch.Tensor,
        boundaries: list[tuple[int, int]],
        compression_ratio: float,
    ) -> torch.Tensor:
        return torch.zeros((points.shape[0],), dtype=torch.bool, device=points.device)


def test_f1_score_identical_sets() -> None:
    assert f1_score({1, 2, 3}, {1, 2, 3}) == pytest.approx(1.0)


def test_f1_score_disjoint_sets() -> None:
    assert f1_score({1, 2}, {3, 4}) == pytest.approx(0.0)


def test_f1_score_both_empty() -> None:
    assert f1_score(set(), set()) == pytest.approx(1.0)


def test_f1_score_one_empty() -> None:
    assert f1_score({1}, set()) == pytest.approx(0.0)
    assert f1_score(set(), {1}) == pytest.approx(0.0)


def test_f1_score_partial_overlap() -> None:
    # precision = 2/3, recall = 2/4, F1 = 4/7.
    assert f1_score({1, 2, 3, 4}, {2, 3, 5}) == pytest.approx(4.0 / 7.0)


def test_clustering_f1_uses_co_membership_pairs() -> None:
    labels_o = [0, 0, 1, 1, -1]
    labels_s = [0, 0, 0, 1, -1]

    # original pairs: {(0, 1), (2, 3)}; simplified pairs: {(0, 1), (0, 2), (1, 2)}
    # precision = 1/3, recall = 1/2, F1 = 0.4.
    assert clustering_f1(labels_o, labels_s) == pytest.approx(0.4)


def test_clustering_f1_empty_pair_sets_agree() -> None:
    assert clustering_f1([-1, -1], [-1, -1]) == pytest.approx(1.0)


def test_evaluate_method_scores_noop_above_degenerate_baseline() -> None:
    trajectories = [
        torch.tensor([[0.0, 0.0, 0.0, 1.0], [1.0, 0.2, 0.2, 1.0]], dtype=torch.float32),
        torch.tensor([[0.0, 5.0, 5.0, 1.0], [1.0, 5.2, 5.2, 1.0]], dtype=torch.float32),
    ]
    points = torch.cat(trajectories, dim=0)
    boundaries = [(0, 2), (2, 4)]
    typed_queries = [
        {
            "type": "range",
            "params": {
                "lat_min": -1.0,
                "lat_max": 1.0,
                "lon_min": -1.0,
                "lon_max": 1.0,
                "t_start": -1.0,
                "t_end": 2.0,
            },
        }
    ]

    keep_all = evaluate_method(
        method=KeepAllMethod(),
        points=points,
        boundaries=boundaries,
        typed_queries=typed_queries,
        workload_mix={"range": 1.0},
        compression_ratio=1.0,
    )
    drop_all = evaluate_method(
        method=DropAllMethod(),
        points=points,
        boundaries=boundaries,
        typed_queries=typed_queries,
        workload_mix={"range": 1.0},
        compression_ratio=0.0,
    )

    assert keep_all.aggregate_f1 == pytest.approx(1.0)
    assert drop_all.aggregate_f1 == pytest.approx(0.0)
    assert keep_all.aggregate_f1 > drop_all.aggregate_f1


def test_method_comparison_table_labels_f1() -> None:
    table = print_method_comparison_table(
        {
            "A": MethodEvaluation(
                aggregate_f1=1.0,
                per_type_f1={"range": 1.0},
                compression_ratio=1.0,
                latency_ms=0.0,
            )
        }
    )

    assert "AggregateF1" in table
    assert "AggregateErr" not in table
