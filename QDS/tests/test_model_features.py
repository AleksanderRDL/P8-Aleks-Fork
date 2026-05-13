import pytest
import torch

from src.queries.query_types import QUERY_TYPE_ID_RANGE, pad_query_features
from src.queries.workload import TypedQueryWorkload
from src.training.model_features import RANGE_AWARE_POINT_DIM, build_model_point_features


def _range_workload() -> TypedQueryWorkload:
    queries = [
        {
            "type": "range",
            "params": {
                "lat_min": 0.0,
                "lat_max": 1.0,
                "lon_min": 0.0,
                "lon_max": 1.0,
                "t_start": 0.0,
                "t_end": 10.0,
            },
        }
    ]
    features, type_ids = pad_query_features(queries)
    return TypedQueryWorkload(query_features=features, typed_queries=queries, type_ids=type_ids)


def test_range_aware_features_expose_point_query_relation() -> None:
    points = torch.tensor(
        [
            [5.0, 0.5, 0.5, 1.0, 45.0, 0.0, 0.0, 0.2],
            [0.0, 0.0, 0.5, 1.0, 45.0, 1.0, 0.0, 0.9],
            [20.0, 3.0, 3.0, 1.0, 45.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    features = build_model_point_features(points, _range_workload(), "range_aware")
    relation = features[:, 8:]

    assert features.shape == (3, RANGE_AWARE_POINT_DIM)
    assert relation[0, 0].item() == pytest.approx(1.0)
    assert relation[2, 0].item() == pytest.approx(0.0)
    assert relation[0, 3].item() > relation[2, 3].item()
    assert relation[1, 5].item() > relation[0, 5].item()


def test_model_feature_modes_keep_expected_base_dimensions() -> None:
    workload = _range_workload()
    points = torch.zeros((2, 8), dtype=torch.float32)

    assert build_model_point_features(points, workload, "baseline").shape == (2, 7)
    assert build_model_point_features(points, workload, "turn_aware").shape == (2, 8)


def test_range_aware_rejects_non_range_workloads() -> None:
    queries = [{"type": "knn", "params": {"lat": 0.0, "lon": 0.0, "t_center": 0.0, "t_half_window": 1.0, "k": 1}}]
    features, type_ids = pad_query_features(queries)
    workload = TypedQueryWorkload(query_features=features, typed_queries=queries, type_ids=type_ids)
    points = torch.zeros((2, 8), dtype=torch.float32)

    assert int(type_ids[0].item()) != QUERY_TYPE_ID_RANGE
    with pytest.raises(ValueError, match="pure range"):
        build_model_point_features(points, workload, "range_aware")
