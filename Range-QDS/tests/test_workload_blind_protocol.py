import pytest
import torch

from evaluation.baselines import MLQDSMethod
from experiments.experiment_config import build_experiment_config
from models.workload_blind_qds_model import WorkloadBlindRangeQDSModel
from queries.query_types import NUM_QUERY_TYPES, pad_query_features
from queries.workload import TypedQueryWorkload
from training.scaler import FeatureScaler
from training.training_outputs import TrainingOutputs
from training.training_validation import _validation_query_score


def _points() -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.2, 0.1, 1.0, 5.0, 0.0, 0.0, 0.1],
            [2.0, 0.4, 0.2, 1.0, 10.0, 0.0, 0.0, 0.3],
            [3.0, 0.6, 0.3, 1.0, 15.0, 0.0, 0.0, 0.2],
            [4.0, 0.8, 0.4, 1.0, 20.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )


def _workload_with_query_features(query_features: torch.Tensor) -> TypedQueryWorkload:
    queries = [
        {
            "type": "range",
            "params": {
                "lat_min": -0.1,
                "lat_max": 1.0,
                "lon_min": -0.1,
                "lon_max": 0.5,
                "t_start": -1.0,
                "t_end": 5.0,
            },
        }
    ]
    _features, type_ids = pad_query_features(queries)
    return TypedQueryWorkload(query_features=query_features, typed_queries=queries, type_ids=type_ids)


def _trained_blind(points: torch.Tensor) -> TrainingOutputs:
    model = WorkloadBlindRangeQDSModel(
        point_dim=8,
        query_dim=12,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        query_chunk_size=8,
    )
    scaler = FeatureScaler.fit(points[:, :8], torch.zeros((1, 12), dtype=torch.float32))
    return TrainingOutputs(
        model=model,
        scaler=scaler,
        labels=torch.zeros((points.shape[0], NUM_QUERY_TYPES), dtype=torch.float32),
        labelled_mask=torch.ones((points.shape[0], NUM_QUERY_TYPES), dtype=torch.bool),
        history=[],
    )


def test_workload_blind_model_supports_mlp_only_mode() -> None:
    model = WorkloadBlindRangeQDSModel(
        point_dim=8,
        query_dim=12,
        embed_dim=16,
        num_heads=2,
        num_layers=0,
        query_chunk_size=8,
    )
    points = torch.randn((2, 5, 8), dtype=torch.float32)
    bad_queries = torch.full((3, 12), float("nan"), dtype=torch.float32)

    pred = model(points, queries=bad_queries, padding_mask=torch.zeros((2, 5), dtype=torch.bool))

    assert pred.shape == (2, 5)
    assert torch.isfinite(pred).all()
    assert model.local_transformer is None


def test_workload_blind_simplify_does_not_read_eval_query_features() -> None:
    points = _points()
    boundaries = [(0, points.shape[0])]
    trained = _trained_blind(points)
    bad_eval_features = torch.full((1, 12), float("nan"), dtype=torch.float32)

    retained = MLQDSMethod(
        name="MLQDS",
        trained=trained,
        workload=_workload_with_query_features(bad_eval_features),
        workload_type="range",
        temporal_fraction=0.0,
        inference_device="cpu",
    ).simplify(points, boundaries, compression_ratio=0.4)

    assert retained.shape == (points.shape[0],)
    assert retained.dtype == torch.bool


def test_mlqds_method_reuses_scores_across_compression_ratios(monkeypatch: pytest.MonkeyPatch) -> None:
    points = _points()
    boundaries = [(0, points.shape[0])]
    trained = _trained_blind(points)
    calls = {"count": 0}

    def fake_predict(**_kwargs: object) -> torch.Tensor:
        calls["count"] += 1
        return torch.linspace(-1.0, 1.0, steps=points.shape[0])

    monkeypatch.setattr("evaluation.baselines.windowed_predict", fake_predict)
    method = MLQDSMethod(
        name="MLQDS",
        trained=trained,
        workload=_workload_with_query_features(torch.full((1, 12), float("nan"), dtype=torch.float32)),
        workload_type="range",
        temporal_fraction=0.0,
        inference_device="cpu",
    )

    retained_low = method.simplify(points, boundaries, compression_ratio=0.4)
    retained_high = method.simplify(points, boundaries, compression_ratio=0.8)

    assert calls["count"] == 1
    assert int(retained_low.sum().item()) < int(retained_high.sum().item())

    method.simplify(points.clone(), boundaries, compression_ratio=0.4)
    assert calls["count"] == 2


def test_workload_blind_validation_scoring_does_not_read_validation_query_features() -> None:
    points = _points()
    boundaries = [(0, points.shape[0])]
    trajectories = [points]
    trained = _trained_blind(points)
    bad_validation_features = torch.full((1, 12), float("nan"), dtype=torch.float32)
    cfg = build_experiment_config(
        model_type="workload_blind_range",
        compression_ratio=0.4,
        checkpoint_score_variant="range_usefulness",
    )

    score, per_type = _validation_query_score(
        model=trained.model,
        scaler=trained.scaler,
        trajectories=trajectories,
        boundaries=boundaries,
        workload=_workload_with_query_features(bad_validation_features),
        workload_map={"range": 1.0},
        model_config=cfg.model,
        device=torch.device("cpu"),
        validation_points=points,
    )

    assert 0.0 <= score <= 1.0
    assert 0.0 <= per_type["range"] <= 1.0


def test_workload_blind_rejects_eval_geometry_blend() -> None:
    points = _points()
    trained = _trained_blind(points)

    with pytest.raises(ValueError, match="eval labels would affect the retained mask"):
        MLQDSMethod(
            name="MLQDS",
            trained=trained,
            workload=_workload_with_query_features(torch.zeros((1, 12), dtype=torch.float32)),
            workload_type="range",
            range_geometry_blend=0.1,
            range_geometry_scores=torch.ones((points.shape[0],), dtype=torch.float32),
        ).simplify(points, [(0, points.shape[0])], compression_ratio=0.4)
