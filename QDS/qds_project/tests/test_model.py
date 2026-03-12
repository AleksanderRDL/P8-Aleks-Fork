"""Tests for the TrajectoryQDSModel."""
import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models.trajectory_qds_model import TrajectoryQDSModel, normalize_points_and_queries

class TestTrajectoryQDSModel:
    def test_output_shape(self):
        model = TrajectoryQDSModel()
        points = torch.randn(20, 5)
        queries = torch.randn(10, 6)
        scores = model(points, queries)
        assert scores.shape == (20,), f"Expected (20,), got {scores.shape}"

    def test_output_range(self):
        model = TrajectoryQDSModel()
        points = torch.randn(30, 5)
        queries = torch.randn(15, 6)
        scores = model(points, queries)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_single_point(self):
        model = TrajectoryQDSModel()
        points = torch.randn(1, 5)
        queries = torch.randn(5, 6)
        scores = model(points, queries)
        assert scores.shape == (1,)

    def test_single_query(self):
        model = TrajectoryQDSModel()
        points = torch.randn(10, 5)
        queries = torch.randn(1, 6)
        scores = model(points, queries)
        assert scores.shape == (10,)

    def test_forward_no_grad(self):
        model = TrajectoryQDSModel()
        points = torch.randn(10, 5)
        queries = torch.randn(5, 6)
        with torch.no_grad():
            scores = model(points, queries)
        assert scores.shape == (10,)

    def test_gradients_flow(self):
        model = TrajectoryQDSModel()
        points = torch.randn(10, 5)
        queries = torch.randn(5, 6)
        scores = model(points, queries)
        loss = scores.mean()
        loss.backward()
        for param in model.parameters():
            assert param.grad is not None

class TestNormalizePointsAndQueries:
    def test_output_shapes(self):
        points = torch.randn(20, 5)
        queries = torch.randn(10, 6)
        norm_pts, norm_q = normalize_points_and_queries(points, queries)
        assert norm_pts.shape == points.shape
        assert norm_q.shape == queries.shape
