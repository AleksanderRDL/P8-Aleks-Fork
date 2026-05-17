import pytest
import torch

from experiments.experiment_config import build_experiment_config
from models.historical_prior_qds_model import HistoricalPriorRangeQDSModel, HistoricalPriorStudentRangeQDSModel
from models.workload_blind_range_v2 import WorkloadBlindRangeV2Model
from models.workload_blind_qds_model import SegmentContextRangeQDSModel
from queries.query_types import pad_query_features
from queries.workload import TypedQueryWorkload
from training.checkpoints import ModelArtifacts, load_checkpoint, save_checkpoint
from training.inference import forward_predict, windowed_predict
from training.model_features import (
    CONTEXT_WORKLOAD_BLIND_POINT_DIM,
    HISTORICAL_PRIOR_DENSITY_POINT_DIM,
    HISTORICAL_PRIOR_MMSI_POINT_DIM,
    HISTORICAL_PRIOR_POINT_DIM,
    HISTORICAL_PRIOR_ROUTE_CONTEXT_POINT_DIM,
    RANGE_AWARE_POINT_DIM,
    RANGE_PRIOR_CLOCK_DENSITY_POINT_DIM,
    SUPPORTED_MODEL_TYPES,
    WORKLOAD_BLIND_POINT_DIM,
    WORKLOAD_BLIND_RANGE_V2_POINT_DIM,
    WORKLOAD_BLIND_RANGE_V2_ABSOLUTE_DIM,
    WORKLOAD_BLIND_RANGE_V2_PRIOR_DIM,
    build_model_point_features,
    build_model_point_features_for_dim,
    build_query_free_point_features_for_dim,
)
from training.scaler import FeatureScaler


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
    covered_model_types = {
        "baseline",
        "turn_aware",
        "range_aware",
        "workload_blind_range",
        "workload_blind_range_v2",
        "range_prior",
        "range_prior_clock_density",
        "segment_context_range",
        "historical_prior",
        "historical_prior_mmsi",
        "historical_prior_student",
    }

    assert build_model_point_features(points, workload, "baseline").shape == (2, 7)
    assert build_model_point_features(points, workload, "turn_aware").shape == (2, 8)
    assert build_model_point_features(points, workload, "workload_blind_range").shape == (2, WORKLOAD_BLIND_POINT_DIM)
    assert build_model_point_features(points, workload, "workload_blind_range_v2").shape == (
        2,
        WORKLOAD_BLIND_RANGE_V2_POINT_DIM,
    )
    assert build_model_point_features(points, workload, "range_prior").shape == (2, CONTEXT_WORKLOAD_BLIND_POINT_DIM)
    assert build_model_point_features(points, workload, "range_prior_clock_density").shape == (
        2,
        RANGE_PRIOR_CLOCK_DENSITY_POINT_DIM,
    )
    assert build_model_point_features(points, workload, "segment_context_range").shape == (
        2,
        RANGE_PRIOR_CLOCK_DENSITY_POINT_DIM,
    )
    assert build_model_point_features(points, workload, "historical_prior").shape == (2, HISTORICAL_PRIOR_POINT_DIM)
    assert build_model_point_features(
        points,
        workload,
        "historical_prior_mmsi",
        boundaries=[(0, 2)],
        trajectory_mmsis=[205000000],
    ).shape == (2, HISTORICAL_PRIOR_MMSI_POINT_DIM)
    assert build_model_point_features(points, workload, "historical_prior_student").shape == (
        2,
        HISTORICAL_PRIOR_POINT_DIM,
    )
    assert covered_model_types == set(SUPPORTED_MODEL_TYPES)


def test_workload_blind_features_include_query_free_continuity_context() -> None:
    workload = _range_workload()
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 5.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 7.0, 45.0, 0.0, 0.0, 0.1],
            [3.0, 1.0, 1.0, 9.0, 90.0, 0.0, 1.0, 0.2],
        ],
        dtype=torch.float32,
    )

    features = build_model_point_features(points, workload, "workload_blind_range")
    extras = features[:, 8:]

    assert features.shape == (3, WORKLOAD_BLIND_POINT_DIM)
    assert torch.isfinite(features).all()
    assert extras[0, 0].item() == pytest.approx(0.0)
    assert extras[-1, 0].item() == pytest.approx(1.0)
    assert extras[0, 1].item() == pytest.approx(1.0)
    assert extras[1, 3].item() > 0.0
    assert extras[1, 4].item() > 0.0
    assert extras[1, 7].item() > 0.0
    assert extras[1, 8].item() > 0.0


def test_workload_blind_features_do_not_encode_absolute_day_time() -> None:
    workload = _range_workload()
    points = torch.tensor(
        [
            [86_400.0, 0.0, 0.0, 5.0, 0.0, 1.0, 0.0, 0.0],
            [86_460.0, 1.0, 0.0, 7.0, 45.0, 0.0, 0.0, 0.1],
            [86_520.0, 1.0, 1.0, 9.0, 90.0, 0.0, 1.0, 0.2],
        ],
        dtype=torch.float32,
    )
    shifted = points.clone()
    shifted[:, 0] += 2 * 86_400.0

    features = build_model_point_features(points, workload, "workload_blind_range")
    shifted_features = build_model_point_features(shifted, workload, "workload_blind_range")

    assert torch.allclose(features, shifted_features)
    assert features[0, 0].item() == pytest.approx(0.0)
    assert features[-1, 0].item() == pytest.approx(1.0)


def test_workload_blind_range_v2_absolute_features_without_prior_are_day_invariant() -> None:
    workload = _range_workload()
    points = torch.tensor(
        [
            [10.0, 55.0, 12.0, 5.0, 0.0, 1.0, 0.0, 0.0],
            [20.0, 55.5, 12.5, 7.0, 45.0, 0.0, 0.0, 0.1],
        ],
        dtype=torch.float32,
    )
    shifted = points.clone()
    shifted[:, 0] += 172_800.0

    absolute_start = CONTEXT_WORKLOAD_BLIND_POINT_DIM
    absolute_end = absolute_start + WORKLOAD_BLIND_RANGE_V2_ABSOLUTE_DIM

    features = build_model_point_features(points, workload, "workload_blind_range_v2")
    shifted_features = build_model_point_features(shifted, workload, "workload_blind_range_v2")

    assert torch.allclose(features[:, absolute_start:absolute_end], shifted_features[:, absolute_start:absolute_end])
    prior_start = absolute_end
    prior_end = prior_start + WORKLOAD_BLIND_RANGE_V2_PRIOR_DIM
    assert torch.allclose(features[:, prior_start:prior_end], shifted_features[:, prior_start:prior_end])
    assert torch.allclose(features[:, prior_start:prior_end], torch.zeros_like(features[:, prior_start:prior_end]))


def test_workload_blind_base_feature_dim_stays_supported() -> None:
    points = torch.zeros((4, 8), dtype=torch.float32)
    points[0, 5] = 1.0
    points[-1, 6] = 1.0
    workload = _range_workload()

    base_features = build_model_point_features_for_dim(points, workload, WORKLOAD_BLIND_POINT_DIM)
    current_features = build_model_point_features_for_dim(points, workload, WORKLOAD_BLIND_POINT_DIM)
    context_features = build_model_point_features_for_dim(points, workload, CONTEXT_WORKLOAD_BLIND_POINT_DIM)
    prior_features = build_model_point_features_for_dim(points, workload, RANGE_PRIOR_CLOCK_DENSITY_POINT_DIM)

    assert base_features.shape == (4, WORKLOAD_BLIND_POINT_DIM)
    assert current_features.shape == (4, WORKLOAD_BLIND_POINT_DIM)
    assert context_features.shape == (4, CONTEXT_WORKLOAD_BLIND_POINT_DIM)
    assert prior_features.shape == (4, RANGE_PRIOR_CLOCK_DENSITY_POINT_DIM)
    assert torch.equal(base_features, current_features)
    assert torch.equal(current_features, context_features[:, :WORKLOAD_BLIND_POINT_DIM])
    assert torch.equal(context_features, prior_features[:, :CONTEXT_WORKLOAD_BLIND_POINT_DIM])


def test_range_prior_uses_full_query_free_context_feature_set() -> None:
    workload = _range_workload()
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 5.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 7.0, 45.0, 0.0, 0.0, 0.1],
            [3.0, 1.0, 1.0, 9.0, 90.0, 0.0, 0.0, 0.2],
            [6.0, 2.0, 1.0, 3.0, 60.0, 0.0, 1.0, 0.3],
        ],
        dtype=torch.float32,
    )

    features = build_model_point_features(points, workload, "range_prior")

    assert features.shape == (4, CONTEXT_WORKLOAD_BLIND_POINT_DIM)
    assert torch.isfinite(features).all()
    assert features[:, WORKLOAD_BLIND_POINT_DIM:].abs().sum().item() > 0.0


def test_range_prior_clock_density_adds_query_free_clock_and_density_features() -> None:
    workload = _range_workload()
    points = torch.tensor(
        [
            [0.0, 0.00, 0.00, 5.0, 0.0, 1.0, 0.0, 0.0],
            [21_600.0, 0.01, 0.01, 5.0, 0.0, 0.0, 0.0, 0.0],
            [43_200.0, 0.02, 0.01, 5.0, 0.0, 0.0, 0.0, 0.0],
            [64_800.0, 1.00, 1.00, 5.0, 0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    features = build_model_point_features(points, workload, "range_prior_clock_density")
    context = build_model_point_features(points, workload, "range_prior")

    assert features.shape == (4, RANGE_PRIOR_CLOCK_DENSITY_POINT_DIM)
    assert torch.isfinite(features).all()
    assert torch.allclose(features[:, :CONTEXT_WORKLOAD_BLIND_POINT_DIM], context)
    assert torch.allclose(build_query_free_point_features_for_dim(points, RANGE_PRIOR_CLOCK_DENSITY_POINT_DIM), features)
    assert features[0, CONTEXT_WORKLOAD_BLIND_POINT_DIM].item() == pytest.approx(0.0)
    assert features[1, CONTEXT_WORKLOAD_BLIND_POINT_DIM].item() == pytest.approx(1.0)
    assert features[2, CONTEXT_WORKLOAD_BLIND_POINT_DIM + 1].item() == pytest.approx(-1.0)
    assert features[0, -2].item() > features[-1, -2].item()
    assert features[0, -1].item() < features[-1, -1].item()


def test_segment_context_model_ignores_queries_and_round_trips_checkpoint(tmp_path) -> None:
    model = SegmentContextRangeQDSModel(
        point_dim=RANGE_PRIOR_CLOCK_DENSITY_POINT_DIM,
        query_dim=12,
        embed_dim=16,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
    )
    points = torch.randn((2, 6, RANGE_PRIOR_CLOCK_DENSITY_POINT_DIM), dtype=torch.float32)
    padding_mask = torch.zeros((2, 6), dtype=torch.bool)
    padding_mask[1, 4:] = True
    query_a = torch.randn((3, 12), dtype=torch.float32)
    query_b = torch.randn((5, 12), dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        scores_a = model(points, queries=query_a, query_type_ids=torch.zeros((3,), dtype=torch.long), padding_mask=padding_mask)
        scores_b = model(points, queries=query_b, query_type_ids=torch.ones((5,), dtype=torch.long), padding_mask=padding_mask)

    assert scores_a.shape == (2, 6)
    assert torch.isfinite(scores_a).all()
    assert torch.allclose(scores_a, scores_b)

    cfg = build_experiment_config(
        model_type="segment_context_range",
        embed_dim=16,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
    )
    scaler = FeatureScaler(
        point_min=torch.zeros((RANGE_PRIOR_CLOCK_DENSITY_POINT_DIM,), dtype=torch.float32),
        point_max=torch.ones((RANGE_PRIOR_CLOCK_DENSITY_POINT_DIM,), dtype=torch.float32),
        query_min=torch.zeros((12,), dtype=torch.float32),
        query_max=torch.ones((12,), dtype=torch.float32),
    )
    checkpoint = tmp_path / "segment_context.pt"
    save_checkpoint(str(checkpoint), ModelArtifacts(model=model, scaler=scaler, config=cfg))

    loaded = load_checkpoint(str(checkpoint))
    with torch.no_grad():
        loaded_scores = loaded.model(points, queries=query_a, query_type_ids=None, padding_mask=padding_mask)

    assert loaded.config.model.model_type == "segment_context_range"
    assert torch.allclose(scores_a, loaded_scores)


def test_workload_blind_range_v2_checkpoint_accepts_legacy_prior_branch_absence(tmp_path) -> None:
    model = WorkloadBlindRangeV2Model(
        point_dim=WORKLOAD_BLIND_RANGE_V2_POINT_DIM,
        query_dim=12,
        embed_dim=16,
        num_heads=2,
        num_layers=0,
        dropout=0.0,
    )
    cfg = build_experiment_config(
        model_type="workload_blind_range_v2",
        embed_dim=16,
        num_heads=2,
        num_layers=0,
        dropout=0.0,
    )
    scaler = FeatureScaler(
        point_min=torch.zeros((WORKLOAD_BLIND_RANGE_V2_POINT_DIM,), dtype=torch.float32),
        point_max=torch.ones((WORKLOAD_BLIND_RANGE_V2_POINT_DIM,), dtype=torch.float32),
        query_min=torch.zeros((12,), dtype=torch.float32),
        query_max=torch.ones((12,), dtype=torch.float32),
    )
    checkpoint = tmp_path / "range_v2_legacy.pt"
    save_checkpoint(str(checkpoint), ModelArtifacts(model=model, scaler=scaler, config=cfg))
    payload = torch.load(checkpoint, map_location="cpu")
    payload["model_state"] = {
        key: value
        for key, value in payload["model_state"].items()
        if not str(key).startswith("prior_feature_encoder.")
    }
    torch.save(payload, checkpoint)

    loaded = load_checkpoint(str(checkpoint))

    assert isinstance(loaded.model, WorkloadBlindRangeV2Model)
    assert loaded.model.prior_feature_dim == 6
    assert loaded.config.model.model_type == "workload_blind_range_v2"


def test_load_checkpoint_rejects_unknown_model_type(tmp_path) -> None:
    model = SegmentContextRangeQDSModel(
        point_dim=RANGE_PRIOR_CLOCK_DENSITY_POINT_DIM,
        query_dim=12,
        embed_dim=16,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
    )
    cfg = build_experiment_config(
        model_type="segment_context_range",
        embed_dim=16,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
    )
    scaler = FeatureScaler(
        point_min=torch.zeros((RANGE_PRIOR_CLOCK_DENSITY_POINT_DIM,), dtype=torch.float32),
        point_max=torch.ones((RANGE_PRIOR_CLOCK_DENSITY_POINT_DIM,), dtype=torch.float32),
        query_min=torch.zeros((12,), dtype=torch.float32),
        query_max=torch.ones((12,), dtype=torch.float32),
    )
    checkpoint = tmp_path / "unknown_model_type.pt"
    save_checkpoint(str(checkpoint), ModelArtifacts(model=model, scaler=scaler, config=cfg))
    payload = torch.load(checkpoint, map_location="cpu")
    payload["model_type"] = "removed_experimental_model"
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="Unsupported checkpoint model_type"):
        load_checkpoint(str(checkpoint))


def test_historical_prior_uses_query_free_route_context_subset() -> None:
    workload = _range_workload()
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 5.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 7.0, 45.0, 0.0, 0.0, 0.1],
            [3.0, 1.0, 1.0, 9.0, 90.0, 0.0, 0.0, 0.2],
            [6.0, 2.0, 1.0, 3.0, 60.0, 0.0, 1.0, 0.3],
        ],
        dtype=torch.float32,
    )

    features = build_model_point_features(points, workload, "historical_prior")

    assert features.shape == (4, HISTORICAL_PRIOR_POINT_DIM)
    assert torch.isfinite(features).all()
    assert features[:, :3].abs().sum().item() > 0.0


def test_historical_prior_features_include_spatial_density_and_route_context_shape() -> None:
    workload = _range_workload()
    points = torch.tensor(
        [
            [0.0, 0.00, 0.00, 5.0, 0.0, 1.0, 0.0, 0.0],
            [21_600.0, 0.01, 0.01, 5.0, 0.0, 0.0, 0.0, 0.0],
            [43_200.0, 0.02, 0.01, 5.0, 0.0, 0.0, 0.0, 0.0],
            [64_800.0, 1.00, 1.00, 5.0, 0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    features = build_model_point_features(points, workload, "historical_prior")
    route_context = build_model_point_features_for_dim(points, workload, HISTORICAL_PRIOR_ROUTE_CONTEXT_POINT_DIM)
    density_features = build_model_point_features_for_dim(points, workload, HISTORICAL_PRIOR_DENSITY_POINT_DIM)

    assert route_context.shape == (4, HISTORICAL_PRIOR_ROUTE_CONTEXT_POINT_DIM)
    assert density_features.shape == (4, HISTORICAL_PRIOR_DENSITY_POINT_DIM)
    assert features.shape == (4, HISTORICAL_PRIOR_POINT_DIM)
    assert torch.allclose(features[:, :HISTORICAL_PRIOR_ROUTE_CONTEXT_POINT_DIM], route_context)
    assert torch.allclose(density_features[:, :HISTORICAL_PRIOR_ROUTE_CONTEXT_POINT_DIM], route_context)
    assert torch.allclose(density_features[:, -2:], features[:, -2:])
    assert torch.allclose(build_query_free_point_features_for_dim(points, HISTORICAL_PRIOR_POINT_DIM), features)
    assert features[0, HISTORICAL_PRIOR_ROUTE_CONTEXT_POINT_DIM].item() == pytest.approx(0.0)
    assert features[1, HISTORICAL_PRIOR_ROUTE_CONTEXT_POINT_DIM].item() == pytest.approx(1.0)
    assert features[2, HISTORICAL_PRIOR_ROUTE_CONTEXT_POINT_DIM + 1].item() == pytest.approx(-1.0)
    assert features[0, -2].item() > features[-1, -2].item()
    assert features[0, -1].item() < features[-1, -1].item()


def test_historical_prior_mmsi_features_are_query_free_and_identity_stable() -> None:
    workload = _range_workload()
    points = torch.tensor(
        [
            [0.0, 0.00, 0.00, 5.0, 0.0, 1.0, 0.0, 0.0],
            [10.0, 0.01, 0.01, 5.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 1.00, 1.00, 5.0, 0.0, 1.0, 0.0, 0.0],
            [10.0, 1.01, 1.01, 5.0, 0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    boundaries = [(0, 2), (2, 4)]
    features = build_model_point_features(
        points,
        workload,
        "historical_prior_mmsi",
        boundaries=boundaries,
        trajectory_mmsis=[205000000, 205000001],
    )
    inferred = build_model_point_features_for_dim(
        points,
        workload,
        HISTORICAL_PRIOR_MMSI_POINT_DIM,
        boundaries=boundaries,
        trajectory_mmsis=[205000000, 205000001],
    )

    mmsi_start = HISTORICAL_PRIOR_ROUTE_CONTEXT_POINT_DIM
    mmsi_end = mmsi_start + 4
    assert features.shape == (4, HISTORICAL_PRIOR_MMSI_POINT_DIM)
    assert torch.allclose(features, inferred)
    assert torch.allclose(features[0, mmsi_start:mmsi_end], features[1, mmsi_start:mmsi_end])
    assert not torch.allclose(features[0, mmsi_start:mmsi_end], features[2, mmsi_start:mmsi_end])
    assert torch.allclose(
        build_query_free_point_features_for_dim(
            points,
            HISTORICAL_PRIOR_MMSI_POINT_DIM,
            boundaries=boundaries,
            trajectory_mmsis=[205000000, 205000001],
        ),
        features,
    )


def test_historical_prior_model_scores_and_round_trips_checkpoint(tmp_path) -> None:
    model = HistoricalPriorRangeQDSModel(
        point_dim=2,
        query_dim=1,
        historical_prior_k=1,
        historical_prior_density_weight=2.0,
        historical_prior_source_aggregation="mean",
        prior_feature_count=3,
    )
    prior_features = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]], dtype=torch.float32)
    prior_targets = torch.tensor([0.1, 0.9, 0.6], dtype=torch.float32)
    prior_source_ids = torch.tensor([0, 1, 0], dtype=torch.long)
    model.set_prior(prior_features, prior_targets, source_ids=prior_source_ids)

    points = torch.tensor([[[0.02, 0.01], [0.95, 1.0]]], dtype=torch.float32)
    scores = model(points, queries=None, query_type_ids=None)

    assert scores.shape == (1, 2)
    assert scores[0, 0].item() < scores[0, 1].item()

    cfg = build_experiment_config(
        model_type="historical_prior",
        historical_prior_k=1,
        historical_prior_clock_weight=0.25,
        historical_prior_mmsi_weight=2.5,
        historical_prior_density_weight=2.0,
        historical_prior_min_target=0.0,
        historical_prior_source_aggregation="mean",
    )
    scaler = FeatureScaler(
        point_min=torch.zeros((2,), dtype=torch.float32),
        point_max=torch.ones((2,), dtype=torch.float32),
        query_min=torch.zeros((1,), dtype=torch.float32),
        query_max=torch.ones((1,), dtype=torch.float32),
    )
    checkpoint = tmp_path / "historical.pt"
    save_checkpoint(str(checkpoint), ModelArtifacts(model=model, scaler=scaler, config=cfg))

    loaded = load_checkpoint(str(checkpoint))
    assert isinstance(loaded.model, HistoricalPriorRangeQDSModel)
    loaded_scores = loaded.model(points, queries=None, query_type_ids=None)

    assert loaded.config.model.model_type == "historical_prior"
    assert loaded.config.model.historical_prior_clock_weight == 0.25
    assert loaded.config.model.historical_prior_mmsi_weight == 2.5
    assert loaded.model.historical_prior_clock_weight == 0.25
    assert loaded.model.historical_prior_mmsi_weight == 2.5
    assert loaded.model.historical_prior_density_weight == 2.0
    assert loaded.model.historical_prior_min_target == 0.0
    assert loaded.model.historical_prior_source_aggregation == "mean"
    assert torch.equal(loaded.model.historical_source_ids, prior_source_ids)
    assert torch.allclose(scores, loaded_scores)


def test_historical_prior_source_mean_downweights_one_source_matches() -> None:
    prior_features = torch.tensor([[0.0, 0.0], [0.2, 0.0]], dtype=torch.float32)
    prior_targets = torch.tensor([1.0, 0.0], dtype=torch.float32)
    source_ids = torch.tensor([0, 1], dtype=torch.long)
    points = torch.tensor([[[0.0, 0.0]]], dtype=torch.float32)

    pooled = HistoricalPriorRangeQDSModel(point_dim=2, query_dim=1, historical_prior_k=1)
    pooled.set_prior(prior_features, prior_targets, source_ids=source_ids)
    source_mean = HistoricalPriorRangeQDSModel(
        point_dim=2,
        query_dim=1,
        historical_prior_k=1,
        historical_prior_source_aggregation="mean",
    )
    source_mean.set_prior(prior_features, prior_targets, source_ids=source_ids)

    pooled_score = pooled(points, queries=None, query_type_ids=None)[0, 0].item()
    source_mean_score = source_mean(points, queries=None, query_type_ids=None)[0, 0].item()

    assert pooled_score > 0.95
    assert source_mean_score == pytest.approx(0.5)


def test_historical_prior_windowed_predict_uses_pointwise_fast_path(monkeypatch: pytest.MonkeyPatch) -> None:
    model = HistoricalPriorRangeQDSModel(point_dim=2, query_dim=1, historical_prior_k=1)
    prior_features = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]], dtype=torch.float32)
    prior_targets = torch.tensor([0.1, 0.9, 0.6], dtype=torch.float32)
    model.set_prior(prior_features, prior_targets)
    points = torch.tensor(
        [[0.0, 0.0], [0.1, 0.0], [0.5, 0.5], [0.9, 1.0], [1.0, 1.0], [0.4, 0.5]],
        dtype=torch.float32,
    )
    expected = model._score_flat(points)
    calls: list[tuple[int, int]] = []
    original_score_flat = model._score_flat

    def wrapped_score_flat(flat_points: torch.Tensor) -> torch.Tensor:
        calls.append((int(flat_points.shape[0]), int(flat_points.shape[1])))
        return original_score_flat(flat_points)

    monkeypatch.setattr(model, "_score_flat", wrapped_score_flat)

    scores = windowed_predict(
        model=model,
        norm_points=points,
        boundaries=[(0, int(points.shape[0]))],
        queries=None,
        query_type_ids=None,
        window_length=4,
        window_stride=2,
        batch_size=1,
        device=torch.device("cpu"),
    )

    assert torch.allclose(scores, expected)
    assert calls == [(6, 2)]


def test_historical_prior_student_uses_prior_feature_and_round_trips_checkpoint(tmp_path) -> None:
    model = HistoricalPriorStudentRangeQDSModel(
        point_dim=2,
        query_dim=1,
        embed_dim=8,
        num_heads=2,
        num_layers=0,
        dropout=0.0,
        historical_prior_k=1,
        prior_feature_count=3,
    )
    prior_features = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]], dtype=torch.float32)
    prior_targets = torch.tensor([0.1, 0.9, 0.6], dtype=torch.float32)
    model.set_prior(prior_features, prior_targets)

    points = torch.tensor([[[0.02, 0.01], [0.95, 1.0]]], dtype=torch.float32)
    scores = model(points, queries=None, query_type_ids=None)

    assert scores.shape == (1, 2)
    assert torch.isfinite(scores).all()
    assert model.historical_targets.shape[0] == 3

    cfg = build_experiment_config(
        model_type="historical_prior_student",
        historical_prior_k=1,
        embed_dim=8,
        num_heads=2,
        num_layers=0,
        dropout=0.0,
    )
    scaler = FeatureScaler(
        point_min=torch.zeros((2,), dtype=torch.float32),
        point_max=torch.ones((2,), dtype=torch.float32),
        query_min=torch.zeros((1,), dtype=torch.float32),
        query_max=torch.ones((1,), dtype=torch.float32),
    )
    checkpoint = tmp_path / "historical_student.pt"
    save_checkpoint(str(checkpoint), ModelArtifacts(model=model, scaler=scaler, config=cfg))

    loaded = load_checkpoint(str(checkpoint))
    loaded_scores = loaded.model(points, queries=None, query_type_ids=None)

    assert loaded.config.model.model_type == "historical_prior_student"
    assert loaded.model.historical_prior_k == 1
    assert torch.allclose(scores, loaded_scores)


def test_historical_prior_min_target_filters_weak_support() -> None:
    model = HistoricalPriorRangeQDSModel(
        point_dim=2,
        query_dim=1,
        historical_prior_k=1,
        historical_prior_min_target=0.5,
        prior_feature_count=3,
    )
    prior_features = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]], dtype=torch.float32)
    prior_targets = torch.tensor([0.1, 0.9, 0.6], dtype=torch.float32)

    model.set_prior(prior_features, prior_targets)

    assert model.historical_features.shape[0] == 2
    assert model.historical_targets.tolist() == pytest.approx([0.9, 0.6])


def test_historical_prior_density_weight_changes_density_dimension_distance() -> None:
    model = HistoricalPriorRangeQDSModel(
        point_dim=21,
        query_dim=1,
        historical_prior_k=1,
        historical_prior_density_weight=4.0,
        prior_feature_count=2,
    )
    prior = torch.zeros((2, 21), dtype=torch.float32)
    prior[1, -2:] = 1.0
    model.set_prior(prior, torch.tensor([0.1, 0.9], dtype=torch.float32))

    point = torch.zeros((1, 1, 21), dtype=torch.float32)
    point[0, 0, -2:] = 0.9
    score = model(point)[0, 0]

    assert score.item() > 0.5


def test_historical_prior_mmsi_weight_changes_identity_dimension_distance() -> None:
    weak_identity = HistoricalPriorRangeQDSModel(
        point_dim=HISTORICAL_PRIOR_MMSI_POINT_DIM,
        query_dim=1,
        historical_prior_k=1,
        historical_prior_mmsi_weight=1.0,
    )
    strong_identity = HistoricalPriorRangeQDSModel(
        point_dim=HISTORICAL_PRIOR_MMSI_POINT_DIM,
        query_dim=1,
        historical_prior_k=1,
        historical_prior_mmsi_weight=8.0,
    )
    mmsi_start = HISTORICAL_PRIOR_ROUTE_CONTEXT_POINT_DIM
    mmsi_end = mmsi_start + 4
    prior = torch.zeros((2, HISTORICAL_PRIOR_MMSI_POINT_DIM), dtype=torch.float32)
    prior[1, :HISTORICAL_PRIOR_ROUTE_CONTEXT_POINT_DIM] = 0.4
    prior[1, mmsi_start:mmsi_end] = 0.2
    targets = torch.tensor([0.1, 0.9], dtype=torch.float32)
    weak_identity.set_prior(prior, targets)
    strong_identity.set_prior(prior, targets)

    point = torch.zeros((1, 1, HISTORICAL_PRIOR_MMSI_POINT_DIM), dtype=torch.float32)
    point[0, 0, mmsi_start:mmsi_end] = 0.2

    assert weak_identity(point)[0, 0].item() < 0.5
    assert strong_identity(point)[0, 0].item() > 0.5


def test_forward_predict_supports_historical_query_free_feature_dims() -> None:
    model = HistoricalPriorRangeQDSModel(point_dim=HISTORICAL_PRIOR_POINT_DIM, query_dim=1, historical_prior_k=1)
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 5.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 7.0, 45.0, 0.0, 0.0, 0.1],
            [3.0, 1.0, 1.0, 9.0, 90.0, 0.0, 1.0, 0.2],
        ],
        dtype=torch.float32,
    )
    features = build_query_free_point_features_for_dim(points, HISTORICAL_PRIOR_POINT_DIM)
    scaler = FeatureScaler.fit(features, torch.zeros((1, 1), dtype=torch.float32))
    norm_points = scaler.transform_points(features)
    model.set_prior(norm_points, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32))
    cfg = build_experiment_config(model_type="historical_prior")
    artifacts = ModelArtifacts(model=model, scaler=scaler, config=cfg)

    scores = forward_predict(
        artifacts,
        points=points,
        queries=None,
        query_type_ids=None,
        boundaries=[(0, 3)],
        window_length=3,
        window_stride=3,
    )

    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()


def test_range_aware_rejects_non_range_workloads() -> None:
    queries = [{"type": "legacy", "params": {}}]

    with pytest.raises(ValueError, match="Only range"):
        pad_query_features(queries)
