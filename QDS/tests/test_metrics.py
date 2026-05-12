"""Tests F1-based query metrics. See src/evaluation/README.md for details."""

from __future__ import annotations

import pytest
import torch

from src.evaluation.baselines import OracleMethod, UniformTemporalMethod
from src.evaluation.evaluate_methods import (
    EvaluationQueryCache,
    _retained_point_gap_stats,
    evaluate_method,
    print_method_comparison_table,
    print_range_usefulness_table,
    score_range_boundary_preservation,
    score_range_usefulness,
    score_retained_mask,
)
from src.evaluation.metrics import MethodEvaluation, clustering_f1, compute_length_preservation, f1_score
from src.simplification.mlqds_scoring import pure_workload_scores
from src.simplification.simplify_trajectories import simplify_with_temporal_score_hybrid


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


class FixedMaskMethod:
    def __init__(self, retained_mask: torch.Tensor) -> None:
        self.retained_mask = retained_mask
        self.name = "FixedMask"

    def simplify(
        self,
        points: torch.Tensor,
        boundaries: list[tuple[int, int]],
        compression_ratio: float,
    ) -> torch.Tensor:
        return self.retained_mask.clone()


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
        workload_map={"range": 1.0},
        compression_ratio=1.0,
    )
    drop_all = evaluate_method(
        method=DropAllMethod(),
        points=points,
        boundaries=boundaries,
        typed_queries=typed_queries,
        workload_map={"range": 1.0},
        compression_ratio=0.0,
    )

    assert keep_all.aggregate_f1 == pytest.approx(1.0)
    assert drop_all.aggregate_f1 == pytest.approx(0.0)
    assert keep_all.aggregate_f1 > drop_all.aggregate_f1


def test_score_retained_mask_matches_evaluate_method() -> None:
    trajectories = [
        torch.tensor([[0.0, 0.0, 0.0, 1.0], [1.0, 0.2, 0.2, 1.0]], dtype=torch.float32),
        torch.tensor([[0.0, 5.0, 5.0, 1.0], [1.0, 5.2, 5.2, 1.0]], dtype=torch.float32),
    ]
    points = torch.cat(trajectories, dim=0)
    boundaries = [(0, 2), (2, 4)]
    queries = [
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
    retained = torch.tensor([True, False, True, True])

    aggregate, per_type, _, _ = score_retained_mask(
        points=points,
        boundaries=boundaries,
        retained_mask=retained,
        typed_queries=queries,
        workload_map={"range": 1.0},
    )
    evaluated = evaluate_method(
        method=FixedMaskMethod(retained),
        points=points,
        boundaries=boundaries,
        typed_queries=queries,
        workload_map={"range": 1.0},
        compression_ratio=0.75,
    )

    assert aggregate == pytest.approx(evaluated.aggregate_f1)
    assert per_type == pytest.approx(evaluated.per_type_f1)


def test_score_retained_mask_cache_reuses_full_query_results(monkeypatch) -> None:
    import src.evaluation.evaluate_methods as eval_methods

    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.2, 1.0],
            [0.0, 5.0, 5.0, 1.0],
            [1.0, 5.0, 5.2, 1.0],
        ],
        dtype=torch.float32,
    )
    boundaries = [(0, 2), (2, 4)]
    queries = [
        {
            "type": "knn",
            "params": {
                "lat": 0.0,
                "lon": 0.0,
                "t_center": 0.5,
                "t_half_window": 2.0,
                "k": 1,
            },
        }
    ]
    original_execute = eval_methods.execute_typed_query
    calls = {"full": 0, "simplified": 0}

    def counting_execute(
        query_points: torch.Tensor,
        trajectories: list[torch.Tensor],
        query: dict,
        query_boundaries: list[tuple[int, int]] | None = None,
    ):
        if query_points.data_ptr() == points.data_ptr() and tuple(query_points.shape) == tuple(points.shape):
            calls["full"] += 1
        else:
            calls["simplified"] += 1
        return original_execute(query_points, trajectories, query, query_boundaries)

    monkeypatch.setattr(eval_methods, "execute_typed_query", counting_execute)
    query_cache = EvaluationQueryCache.for_workload(points, boundaries, queries)

    for retained in (
        torch.tensor([True, False, True, True]),
        torch.tensor([True, True, True, False]),
    ):
        score_retained_mask(
            points=points,
            boundaries=boundaries,
            retained_mask=retained,
            typed_queries=queries,
            workload_map={"knn": 1.0},
            query_cache=query_cache,
        )

    assert calls == {"full": 1, "simplified": 2}


def test_score_retained_mask_cache_rejects_different_workload() -> None:
    points = torch.tensor([[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
    boundaries = [(0, 2)]
    queries = [
        {
            "type": "range",
            "params": {
                "lat_min": -1.0,
                "lat_max": 0.5,
                "lon_min": -1.0,
                "lon_max": 0.5,
                "t_start": -1.0,
                "t_end": 0.5,
            },
        }
    ]
    other_queries = [
        {
            "type": "range",
            "params": {
                "lat_min": -1.0,
                "lat_max": 2.0,
                "lon_min": -1.0,
                "lon_max": 2.0,
                "t_start": -1.0,
                "t_end": 2.0,
            },
        }
    ]
    query_cache = EvaluationQueryCache.for_workload(points, boundaries, queries)

    with pytest.raises(ValueError, match="EvaluationQueryCache"):
        score_retained_mask(
            points=points,
            boundaries=boundaries,
            retained_mask=torch.tensor([True, False]),
            typed_queries=other_queries,
            workload_map={"range": 1.0},
            query_cache=query_cache,
        )


def test_knn_f1_penalizes_missing_representative_points() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.1, 1.0],
            [2.0, 0.0, 0.2, 1.0],
            [3.0, 0.0, 0.3, 1.0],
            [0.0, 10.0, 10.0, 1.0],
        ],
        dtype=torch.float32,
    )
    boundaries = [(0, 4), (4, 5)]
    queries = [
        {
            "type": "knn",
            "params": {
                "lat": 0.0,
                "lon": 0.0,
                "t_center": 1.5,
                "t_half_window": 2.0,
                "k": 1,
            },
        }
    ]
    retained = torch.tensor([True, False, False, False, True])

    aggregate, per_type, agg_combined, per_type_combined = score_retained_mask(
        points=points,
        boundaries=boundaries,
        retained_mask=retained,
        typed_queries=queries,
        workload_map={"knn": 1.0},
    )

    # Pure answer F1: trajectory 0 still found via point 0 (closest to anchor).
    assert per_type["knn"] == pytest.approx(1.0)
    assert aggregate == pytest.approx(1.0)
    # Combined F1 still penalizes losing the kNN representative points.
    assert per_type_combined["knn"] == pytest.approx(0.4)
    assert agg_combined == pytest.approx(0.4)


def test_uniform_temporal_is_evenly_spaced() -> None:
    points = torch.stack(
        [torch.tensor([float(i), 0.0, float(i), 1.0], dtype=torch.float32) for i in range(10)]
    )
    boundaries = [(0, 10)]

    retained = UniformTemporalMethod().simplify(points, boundaries, compression_ratio=0.3)

    assert torch.where(retained)[0].tolist() == [0, 4, 9]


def test_temporal_score_hybrid_keeps_base_and_score_fill() -> None:
    scores = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0])
    retained = simplify_with_temporal_score_hybrid(
        scores=scores,
        boundaries=[(0, 10)],
        compression_ratio=0.3,
        temporal_fraction=0.5,
        diversity_bonus=0.0,
    )

    assert torch.where(retained)[0].tolist() == [0, 5, 9]


def test_temporal_score_hybrid_zero_temporal_fraction_is_pure_score() -> None:
    scores = torch.tensor([0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 4.0, 5.0, 6.0])
    retained = simplify_with_temporal_score_hybrid(
        scores=scores,
        boundaries=[(0, 10)],
        compression_ratio=0.3,
        temporal_fraction=0.0,
        diversity_bonus=0.0,
    )

    assert torch.where(retained)[0].tolist() == [4, 5, 6]


def test_pure_workload_scores_rank_mode_is_canonical_per_trajectory() -> None:
    predictions = torch.tensor([0.1, 0.9, 0.5, 0.2], dtype=torch.float32)

    scores = pure_workload_scores(predictions, [(0, 4)], "range", score_mode="rank")

    assert scores.tolist() == pytest.approx([0.0, 1.0, 2.0 / 3.0, 1.0 / 3.0])


def test_pure_workload_scores_support_raw_and_sigmoid_modes() -> None:
    predictions = torch.tensor([0.1, 0.9, 0.5, 0.2], dtype=torch.float32)

    raw = pure_workload_scores(predictions, [(0, 4)], "range", score_mode="raw")
    sigmoid = pure_workload_scores(predictions, [(0, 4)], "range", score_mode="sigmoid")

    assert raw.tolist() == pytest.approx([0.1, 0.9, 0.5, 0.2])
    assert sigmoid.tolist() == pytest.approx(torch.sigmoid(predictions).tolist())


def test_pure_workload_scores_support_tie_aware_rank() -> None:
    predictions = torch.tensor([0.1, 0.9, 0.9, 0.2], dtype=torch.float32)

    scores = pure_workload_scores(predictions, [(0, 4)], "range", score_mode="rank_tie")

    assert scores.tolist() == pytest.approx([0.0, 5.0 / 6.0, 5.0 / 6.0, 1.0 / 3.0])


def test_pure_workload_scores_support_calibrated_modes() -> None:
    predictions = torch.tensor([0.1, 0.9, 0.5, 0.2], dtype=torch.float32)

    zscore = pure_workload_scores(predictions, [(0, 4)], "range", score_mode="zscore_sigmoid")
    blend = pure_workload_scores(
        predictions,
        [(0, 4)],
        "range",
        score_mode="rank_confidence",
        rank_confidence_weight=0.50,
    )
    temp_sigmoid = pure_workload_scores(
        predictions,
        [(0, 4)],
        "range",
        score_mode="temperature_sigmoid",
        score_temperature=2.0,
    )

    assert torch.all((zscore >= 0.0) & (zscore <= 1.0))
    assert torch.all((blend >= 0.0) & (blend <= 1.0))
    assert temp_sigmoid.tolist() == pytest.approx(torch.sigmoid(predictions / 2.0).tolist())


def test_pure_workload_scores_zscore_mode_handles_flat_logits() -> None:
    predictions = torch.ones((4,), dtype=torch.float32)

    scores = pure_workload_scores(predictions, [(0, 4)], "range", score_mode="zscore_sigmoid")

    assert scores.tolist() == pytest.approx([0.5, 0.5, 0.5, 0.5])


def test_pure_workload_scores_reject_unknown_mode() -> None:
    predictions = torch.zeros((4,), dtype=torch.float32)

    with pytest.raises(ValueError, match="score_mode"):
        pure_workload_scores(predictions, [(0, 4)], "range", score_mode="not-a-mode")


def test_oracle_method_uses_explicit_workload_head() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.1, 1.0],
            [2.0, 0.0, 0.2, 1.0],
            [3.0, 0.0, 0.3, 1.0],
            [4.0, 0.0, 0.4, 1.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.zeros((5, 4), dtype=torch.float32)
    labels[1, 0] = 1.0
    labels[2, 0] = -1.0
    labels[1, 1] = -1.0
    labels[2, 1] = 1.0

    retained = OracleMethod(labels=labels, workload_type="range").simplify(
        points,
        boundaries=[(0, 5)],
        compression_ratio=0.4,
    )

    assert bool(retained[1].item()) is True
    assert bool(retained[2].item()) is False
    assert OracleMethod(labels=labels, workload_type="range").oracle_kind == "additive_label_greedy"


def test_range_boundary_preservation_is_separate_from_range_f1() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.1, 1.0],
            [2.0, 0.0, 0.2, 1.0],
            [3.0, 0.0, 0.3, 1.0],
            [4.0, 9.0, 9.0, 1.0],
        ],
        dtype=torch.float32,
    )
    boundaries = [(0, 5)]
    queries = [
        {
            "type": "range",
            "params": {
                "lat_min": -1.0,
                "lat_max": 1.0,
                "lon_min": -1.0,
                "lon_max": 1.0,
                "t_start": -1.0,
                "t_end": 3.5,
            },
        }
    ]
    retained = torch.tensor([True, False, False, True, True])

    aggregate, per_type, _, _ = score_retained_mask(points, boundaries, retained, queries, {"range": 1.0})
    boundary_f1 = score_range_boundary_preservation(points, boundaries, retained, queries)

    assert aggregate == pytest.approx(2.0 / 3.0)
    assert per_type["range"] == pytest.approx(2.0 / 3.0)
    assert boundary_f1 == pytest.approx(1.0)


def test_range_usefulness_audit_separates_point_hits_from_local_shape() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.1, 1.0],
            [2.0, 0.0, 0.2, 1.0],
            [3.0, 0.0, 0.3, 1.0],
            [4.0, 0.0, 0.4, 1.0],
        ],
        dtype=torch.float32,
    )
    boundaries = [(0, 5)]
    queries = [
        {
            "type": "range",
            "params": {
                "lat_min": -1.0,
                "lat_max": 1.0,
                "lon_min": -1.0,
                "lon_max": 1.0,
                "t_start": -1.0,
                "t_end": 5.0,
            },
        }
    ]
    endpoint_retained = torch.tensor([True, False, False, False, True])
    middle_retained = torch.tensor([False, True, True, False, False])

    endpoint_audit = score_range_usefulness(points, boundaries, endpoint_retained, queries)
    middle_audit = score_range_usefulness(points, boundaries, middle_retained, queries)

    assert endpoint_audit["range_point_f1"] == pytest.approx(middle_audit["range_point_f1"])
    assert endpoint_audit["range_entry_exit_f1"] > middle_audit["range_entry_exit_f1"]
    assert endpoint_audit["range_temporal_coverage"] > middle_audit["range_temporal_coverage"]
    assert endpoint_audit["range_usefulness_score"] > middle_audit["range_usefulness_score"]


def test_range_usefulness_ship_f1_requires_each_hit_ship_present() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.1, 1.0],
            [0.0, 0.2, 0.0, 1.0],
            [1.0, 0.2, 0.1, 1.0],
        ],
        dtype=torch.float32,
    )
    boundaries = [(0, 2), (2, 4)]
    queries = [
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
    retained = torch.tensor([True, False, False, False])

    audit = score_range_usefulness(points, boundaries, retained, queries)

    assert audit["range_point_f1"] == pytest.approx(0.4)
    assert audit["range_ship_f1"] == pytest.approx(2.0 / 3.0)


def test_range_usefulness_temporal_span_does_not_penalize_interior_gap() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.1, 1.0],
            [2.0, 0.0, 0.2, 1.0],
            [3.0, 0.0, 0.3, 1.0],
            [4.0, 0.0, 0.4, 1.0],
        ],
        dtype=torch.float32,
    )
    queries = [
        {
            "type": "range",
            "params": {
                "lat_min": -1.0,
                "lat_max": 1.0,
                "lon_min": -1.0,
                "lon_max": 1.0,
                "t_start": -1.0,
                "t_end": 5.0,
            },
        }
    ]
    retained = torch.tensor([True, False, False, False, True])

    audit = score_range_usefulness(points, [(0, 5)], retained, queries)

    assert audit["range_temporal_coverage"] == pytest.approx(1.0)
    assert audit["range_shape_score"] == pytest.approx(1.0)


def test_range_usefulness_shape_score_penalizes_curved_endpoint_shortcut() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 0.0, 2.0, 1.0],
        ],
        dtype=torch.float32,
    )
    queries = [
        {
            "type": "range",
            "params": {
                "lat_min": -1.0,
                "lat_max": 2.0,
                "lon_min": -1.0,
                "lon_max": 3.0,
                "t_start": -1.0,
                "t_end": 3.0,
            },
        }
    ]
    retained = torch.tensor([True, False, True])

    audit = score_range_usefulness(points, [(0, 3)], retained, queries)

    assert audit["range_temporal_coverage"] == pytest.approx(1.0)
    assert 0.0 < audit["range_shape_score"] < 0.8


def test_range_usefulness_cache_reuses_retained_independent_support() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.1, 1.0],
            [2.0, 0.0, 0.2, 1.0],
            [3.0, 0.0, 0.3, 1.0],
        ],
        dtype=torch.float32,
    )
    boundaries = [(0, 4)]
    queries = [
        {
            "type": "range",
            "params": {
                "lat_min": -1.0,
                "lat_max": 1.0,
                "lon_min": -1.0,
                "lon_max": 1.0,
                "t_start": -1.0,
                "t_end": 4.0,
            },
        }
    ]
    query_cache = EvaluationQueryCache.for_workload(points, boundaries, queries)

    score_range_usefulness(points, boundaries, torch.tensor([True, False, False, True]), queries, query_cache)
    support = query_cache.range_audit_supports[0]
    score_range_usefulness(points, boundaries, torch.tensor([False, True, True, False]), queries, query_cache)

    assert len(query_cache.range_audit_supports) == 1
    assert query_cache.range_audit_supports[0] is support
    assert support.boundary_indices_cpu.tolist() == [0, 3]


def test_retained_point_gap_stats_measure_original_spacing() -> None:
    retained = torch.tensor([True, False, False, True, False, True, True, False, True])

    avg_gap, avg_norm_gap, max_gap = _retained_point_gap_stats(retained, boundaries=[(0, 6), (6, 9)])

    assert avg_gap == pytest.approx((3.0 + 2.0 + 2.0) / 3.0)
    assert avg_norm_gap == pytest.approx(((3.0 / 5.0) + (2.0 / 5.0) + (2.0 / 2.0)) / 3.0)
    assert max_gap == pytest.approx(3.0)


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

    assert "RangePointF1" in table
    assert "RangeUseful" in table
    assert "AvgPtGap" in table
    assert "EntryExitF1" in table
    assert "AnswerF1" not in table
    assert "AggregateErr" not in table


def test_range_usefulness_table_reports_audit_components() -> None:
    table = print_range_usefulness_table(
        {
            "A": MethodEvaluation(
                aggregate_f1=0.5,
                per_type_f1={"range": 0.5},
                range_point_f1=0.5,
                range_ship_f1=0.7,
                range_entry_exit_f1=0.25,
                range_temporal_coverage=0.8,
                range_shape_score=0.6,
                range_usefulness_score=0.575,
            )
        }
    )

    assert "RangePointF1" in table
    assert "ShipF1" in table
    assert "RangeUseful" in table


def test_method_comparison_table_shows_close_f1_values() -> None:
    table = print_method_comparison_table(
        {
            "MLQDS": MethodEvaluation(
                aggregate_f1=0.1823688112,
                per_type_f1={"range": 0.1823688112},
                compression_ratio=0.1008,
                latency_ms=0.0,
            ),
            "Random": MethodEvaluation(
                aggregate_f1=0.1824232682,
                per_type_f1={"range": 0.1824232682},
                compression_ratio=0.1008,
                latency_ms=0.0,
            ),
        }
    )

    assert "0.182369" in table
    assert "0.182423" in table


def test_method_comparison_table_reports_canonical_baseline_diffs() -> None:
    table = print_method_comparison_table(
        {
            "MLQDS": MethodEvaluation(
                aggregate_f1=0.6,
                per_type_f1={"range": 0.6},
                compression_ratio=0.2,
                latency_ms=0.0,
            ),
            "uniform": MethodEvaluation(
                aggregate_f1=0.5,
                per_type_f1={"range": 0.5},
                compression_ratio=0.2,
                latency_ms=0.0,
            ),
            "DouglasPeucker": MethodEvaluation(
                aggregate_f1=0.55,
                per_type_f1={"range": 0.55},
                compression_ratio=0.2,
                latency_ms=0.0,
            ),
        }
    )

    assert "Diff vs MLQDS" in table
    assert "vs uniform" in table
    assert "vs DouglasPeucker" in table
    assert "vs Random" not in table


def test_length_preservation_and_legacy_loss_are_complements() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [2.0, 0.0, 2.0, 1.0],
        ],
        dtype=torch.float32,
    )
    retained = torch.tensor([True, False, True])

    preserved = compute_length_preservation(points, [(0, 3)], retained)
    metrics = MethodEvaluation(
        aggregate_f1=0.8,
        per_type_f1={"range": 0.8},
        compression_ratio=2.0 / 3.0,
        latency_ms=0.0,
        avg_length_preserved=preserved,
    )

    assert preserved == pytest.approx(1.0)
    assert metrics.avg_length_loss == pytest.approx(0.0)
