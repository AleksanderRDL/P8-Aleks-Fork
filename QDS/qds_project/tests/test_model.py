"""Tests for the TrajectoryQDSModel and TurnAwareQDSModel."""
import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models.trajectory_qds_model import TrajectoryQDSModel, normalize_points_and_queries
from src.models.turn_aware_qds_model import TurnAwareQDSModel

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
