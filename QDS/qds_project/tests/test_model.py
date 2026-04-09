"""Tests for the TrajectoryQDSModel, TurnAwareQDSModel, and BoundaryAwareTurnModel."""
import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models.trajectory_qds_model import (
    TrajectoryQDSModel,
    normalize_points_and_queries,
    NUM_QUERY_TYPES,
    QUERY_TYPE_ID_RANGE,
    QUERY_TYPE_ID_INTERSECTION,
    QUERY_TYPE_ID_AGGREGATION,
    QUERY_TYPE_ID_NEAREST,
)
from src.models.turn_aware_qds_model import TurnAwareQDSModel
from src.models.boundary_aware_turn_model import (
    BoundaryAwareTurnModel,
    compute_boundary_proximity,
    extract_boundary_features,
)

class TestTrajectoryQDSModel:
    def test_output_shape(self):
        model = TrajectoryQDSModel()
        points = torch.randn(20, 7)
        queries = torch.randn(10, 6)
        scores = model(points, queries)
        assert scores.shape == (20,), f"Expected (20,), got {scores.shape}"

    def test_output_range(self):
        model = TrajectoryQDSModel()
        points = torch.randn(30, 7)
        queries = torch.randn(15, 6)
        scores = model(points, queries)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_single_point(self):
        model = TrajectoryQDSModel()
        points = torch.randn(1, 7)
        queries = torch.randn(5, 6)
        scores = model(points, queries)
        assert scores.shape == (1,)

    def test_single_query(self):
        model = TrajectoryQDSModel()
        points = torch.randn(10, 7)
        queries = torch.randn(1, 6)
        scores = model(points, queries)
        assert scores.shape == (10,)

    def test_forward_no_grad(self):
        model = TrajectoryQDSModel()
        points = torch.randn(10, 7)
        queries = torch.randn(5, 6)
        with torch.no_grad():
            scores = model(points, queries)
        assert scores.shape == (10,)

    def test_gradients_flow(self):
        model = TrajectoryQDSModel()
        points = torch.randn(10, 7)
        queries = torch.randn(5, 6)
        scores = model(points, queries)
        loss = scores.mean()
        loss.backward()
        for param in model.parameters():
            assert param.grad is not None

    def test_query_type_ids_explicit(self):
        """Passing explicit query_type_ids must produce the same output shape."""
        model = TrajectoryQDSModel()
        points = torch.randn(20, 7)
        queries = torch.randn(10, 6)
        type_ids = torch.randint(0, NUM_QUERY_TYPES, (10,))
        scores = model(points, queries, query_type_ids=type_ids)
        assert scores.shape == (20,)

    def test_query_type_ids_default_equals_zero_ids(self):
        """Omitting query_type_ids must give the same result as all-zero IDs."""
        model = TrajectoryQDSModel()
        model.eval()
        points = torch.randn(10, 7)
        queries = torch.randn(5, 6)
        zero_ids = torch.zeros(5, dtype=torch.long)
        with torch.no_grad():
            scores_default = model(points, queries)
            scores_zero    = model(points, queries, query_type_ids=zero_ids)
        assert torch.allclose(scores_default, scores_zero)

    def test_query_type_ids_affect_output(self):
        """Different query type IDs must produce different scores (type embedding is active)."""
        model = TrajectoryQDSModel()
        model.eval()
        torch.manual_seed(0)
        points = torch.randn(10, 7)
        queries = torch.randn(5, 6)
        range_ids     = torch.full((5,), QUERY_TYPE_ID_RANGE,       dtype=torch.long)
        nearest_ids   = torch.full((5,), QUERY_TYPE_ID_NEAREST,     dtype=torch.long)
        with torch.no_grad():
            scores_range  = model(points, queries, query_type_ids=range_ids)
            scores_nearest = model(points, queries, query_type_ids=nearest_ids)
        assert not torch.allclose(scores_range, scores_nearest), (
            "Scores must differ when query type IDs differ"
        )

    def test_all_query_types_valid(self):
        """Each of the four integer query type constants must run without error."""
        model = TrajectoryQDSModel()
        points = torch.randn(8, 7)
        queries = torch.randn(4, 6)
        for type_id in (
            QUERY_TYPE_ID_RANGE,
            QUERY_TYPE_ID_INTERSECTION,
            QUERY_TYPE_ID_AGGREGATION,
            QUERY_TYPE_ID_NEAREST,
        ):
            ids = torch.full((4,), type_id, dtype=torch.long)
            scores = model(points, queries, query_type_ids=ids)
            assert scores.shape == (8,)

    def test_single_query_with_type_id(self):
        """Single-query path must handle a scalar type_id tensor correctly."""
        model = TrajectoryQDSModel()
        points = torch.randn(10, 7)
        queries = torch.randn(6)  # 1-D single query
        type_id = torch.tensor(QUERY_TYPE_ID_AGGREGATION)
        scores = model(points, queries, query_type_ids=type_id)
        assert scores.shape == (10,)

    def test_query_type_constants(self):
        """Integer query type constants must cover the four expected types."""
        assert NUM_QUERY_TYPES == 4
        assert len({
            QUERY_TYPE_ID_RANGE,
            QUERY_TYPE_ID_INTERSECTION,
            QUERY_TYPE_ID_AGGREGATION,
            QUERY_TYPE_ID_NEAREST,
        }) == 4

    def test_has_point_self_attention(self):
        """TrajectoryQDSModel must expose a point_self_attn attribute."""
        model = TrajectoryQDSModel()
        assert hasattr(model, "point_self_attn")

    def test_has_layer_norms(self):
        """TrajectoryQDSModel must expose point_norm and cross_norm LayerNorm layers."""
        model = TrajectoryQDSModel()
        assert hasattr(model, "point_norm")
        assert hasattr(model, "cross_norm")

class TestNormalizePointsAndQueries:
    def test_output_shapes(self):
        points = torch.randn(20, 7)
        queries = torch.randn(10, 6)
        norm_pts, norm_q = normalize_points_and_queries(points, queries)
        assert norm_pts.shape == points.shape
        assert norm_q.shape == queries.shape

    def test_output_shapes_8_features(self):
        """normalize_points_and_queries must preserve 8-feature shape."""
        points = torch.randn(20, 8)
        queries = torch.randn(10, 6)
        norm_pts, norm_q = normalize_points_and_queries(points, queries)
        assert norm_pts.shape == points.shape
        assert norm_q.shape == queries.shape

    def test_binary_flags_preserved(self):
        """is_start/is_end columns (5, 6) must remain binary after normalisation."""
        points = torch.zeros(10, 7)
        points[0, 5] = 1.0   # is_start for first point
        points[-1, 6] = 1.0  # is_end for last point
        queries = torch.randn(5, 6)
        norm_pts, _ = normalize_points_and_queries(points, queries)
        assert norm_pts[0, 5].item() == pytest.approx(1.0, abs=1e-5)
        assert norm_pts[-1, 6].item() == pytest.approx(1.0, abs=1e-5)
        assert norm_pts[1:-1, 5].sum().item() == pytest.approx(0.0, abs=1e-5)
        assert norm_pts[1:-1, 6].sum().item() == pytest.approx(0.0, abs=1e-5)

    def test_turn_score_preserved(self):
        """turn_score column (7) must remain unchanged after normalisation."""
        points = torch.zeros(10, 8)
        points[:, 7] = torch.linspace(0.0, 1.0, 10)
        queries = torch.randn(5, 6)
        norm_pts, _ = normalize_points_and_queries(points, queries)
        assert torch.allclose(norm_pts[:, 7], points[:, 7], atol=1e-5)


class TestTurnAwareQDSModel:
    def test_output_shape(self):
        model = TurnAwareQDSModel()
        points = torch.randn(20, 8)
        queries = torch.randn(10, 6)
        scores = model(points, queries)
        assert scores.shape == (20,), f"Expected (20,), got {scores.shape}"

    def test_output_range(self):
        model = TurnAwareQDSModel()
        points = torch.randn(30, 8)
        queries = torch.randn(15, 6)
        scores = model(points, queries)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_single_point(self):
        model = TurnAwareQDSModel()
        points = torch.randn(1, 8)
        queries = torch.randn(5, 6)
        scores = model(points, queries)
        assert scores.shape == (1,)

    def test_single_query(self):
        model = TurnAwareQDSModel()
        points = torch.randn(10, 8)
        queries = torch.randn(1, 6)
        scores = model(points, queries)
        assert scores.shape == (10,)

    def test_forward_no_grad(self):
        model = TurnAwareQDSModel()
        points = torch.randn(10, 8)
        queries = torch.randn(5, 6)
        with torch.no_grad():
            scores = model(points, queries)
        assert scores.shape == (10,)

    def test_gradients_flow(self):
        model = TurnAwareQDSModel()
        points = torch.randn(10, 8)
        queries = torch.randn(5, 6)
        scores = model(points, queries)
        loss = scores.mean()
        loss.backward()
        for param in model.parameters():
            assert param.grad is not None

    def test_point_features_constant(self):
        """TurnAwareQDSModel must expect exactly 8 input point features."""
        assert TurnAwareQDSModel.POINT_FEATURES == 8


class TestComputeBoundaryProximity:
    def test_output_shape(self):
        points = torch.zeros(10, 7)
        points[:, 1] = torch.linspace(0.0, 1.0, 10)  # lat
        points[:, 2] = torch.linspace(0.0, 1.0, 10)  # lon
        queries = torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]])
        bp = compute_boundary_proximity(points, queries, sigma=1.0)
        assert bp.shape == (10,)

    def test_output_range(self):
        points = torch.randn(20, 7)
        queries = torch.randn(5, 6)
        bp = compute_boundary_proximity(points, queries, sigma=1.0)
        assert (bp > 0).all() and (bp <= 1).all()

    def test_boundary_point_is_max(self):
        """A point exactly on a boundary edge should have proximity 1.0."""
        # Point at lat=0.0 is exactly on the bottom boundary of the query
        points = torch.tensor([[0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]])  # lat=0.0
        queries = torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]])       # lat_min=0.0
        bp = compute_boundary_proximity(points, queries, sigma=1.0)
        assert bp[0].item() == pytest.approx(1.0, abs=1e-5)

    def test_sigma_affects_decay(self):
        """Smaller sigma should produce lower proximity for non-boundary points."""
        points = torch.tensor([[0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]])  # interior point
        queries = torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]])
        bp_large = compute_boundary_proximity(points, queries, sigma=10.0)
        bp_small = compute_boundary_proximity(points, queries, sigma=0.1)
        assert bp_large[0].item() > bp_small[0].item()

    def test_chunked_matches_reference_formula(self):
        """Chunked proximity computation must match the reference formulation."""
        torch.manual_seed(123)
        points = torch.rand(37, 8)
        queries = torch.rand(11, 6)

        lat = points[:, 1]
        lon = points[:, 2]
        lat_min = queries[:, 0]
        lat_max = queries[:, 1]
        lon_min = queries[:, 2]
        lon_max = queries[:, 3]

        d_bottom = (lat[:, None] - lat_min[None, :]).abs()
        d_top = (lat[:, None] - lat_max[None, :]).abs()
        d_left = (lon[:, None] - lon_min[None, :]).abs()
        d_right = (lon[:, None] - lon_max[None, :]).abs()

        ref = torch.exp(
            -torch.stack([d_bottom, d_top, d_left, d_right], dim=2).min(dim=2).values / 0.7
        ).max(dim=1).values

        chunked = compute_boundary_proximity(
            points,
            queries,
            sigma=0.7,
            point_chunk_size=5,
            query_chunk_size=3,
        )
        assert torch.allclose(chunked, ref, atol=1e-6)


class TestExtractBoundaryFeatures:
    def test_output_shape(self):
        points = torch.randn(15, 7)
        queries = torch.randn(5, 6)
        feat = extract_boundary_features(points, queries, sigma=1.0)
        assert feat.shape == (15, 1)

    def test_values_match_compute(self):
        points = torch.randn(10, 7)
        queries = torch.randn(4, 6)
        bp = compute_boundary_proximity(points, queries, sigma=1.0)
        feat = extract_boundary_features(points, queries, sigma=1.0)
        assert torch.allclose(feat.squeeze(-1), bp, atol=1e-6)


class TestBoundaryAwareTurnModel:
    def test_output_shape(self):
        model = BoundaryAwareTurnModel()
        points = torch.randn(20, 9)
        queries = torch.randn(10, 6)
        scores = model(points, queries)
        assert scores.shape == (20,), f"Expected (20,), got {scores.shape}"

    def test_output_range(self):
        model = BoundaryAwareTurnModel()
        points = torch.zeros(30, 9)
        queries = torch.zeros(15, 6)
        scores = model(points, queries)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_single_point(self):
        model = BoundaryAwareTurnModel()
        points = torch.randn(1, 9)
        queries = torch.randn(5, 6)
        scores = model(points, queries)
        assert scores.shape == (1,)

    def test_single_query(self):
        model = BoundaryAwareTurnModel()
        points = torch.randn(10, 9)
        queries = torch.randn(1, 6)
        scores = model(points, queries)
        assert scores.shape == (10,)

    def test_forward_no_grad(self):
        model = BoundaryAwareTurnModel()
        points = torch.randn(10, 9)
        queries = torch.randn(5, 6)
        with torch.no_grad():
            scores = model(points, queries)
        assert scores.shape == (10,)

    def test_gradients_flow(self):
        model = BoundaryAwareTurnModel()
        points = torch.randn(10, 9)
        queries = torch.randn(5, 6)
        scores = model(points, queries)
        loss = scores.mean()
        loss.backward()
        for param in model.parameters():
            assert param.grad is not None

    def test_point_features_constant(self):
        """BoundaryAwareTurnModel must expect exactly 9 input point features."""
        assert BoundaryAwareTurnModel.POINT_FEATURES == 9

    def test_hyperparameters_stored(self):
        model = BoundaryAwareTurnModel(sigma=0.5, alpha=0.2, beta=0.3)
        assert model.sigma == pytest.approx(0.5)
        assert model.alpha == pytest.approx(0.2)
        assert model.beta == pytest.approx(0.3)

    def test_boundary_proximity_bias(self):
        """Higher boundary_proximity (col 8) should increase scores when alpha > 0."""
        model = BoundaryAwareTurnModel(alpha=1.0, beta=0.0)
        model.eval()
        with torch.no_grad():
            points_low = torch.zeros(1, 9)
            points_high = torch.zeros(1, 9)
            points_high[0, 8] = 1.0  # max boundary proximity
            queries = torch.zeros(3, 6)
            score_low = model(points_low, queries)
            score_high = model(points_high, queries)
        assert score_high[0].item() >= score_low[0].item()

    def test_normalize_preserves_boundary_proximity(self):
        """boundary_proximity column (8) must be preserved by normalize_points_and_queries."""
        points = torch.zeros(10, 9)
        points[:, 8] = torch.linspace(0.0, 1.0, 10)
        queries = torch.randn(5, 6)
        norm_pts, _ = normalize_points_and_queries(points, queries)
        assert torch.allclose(norm_pts[:, 8], points[:, 8], atol=1e-5)
