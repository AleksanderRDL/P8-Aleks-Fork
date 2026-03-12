"""Tests for AIS trajectory simplification."""
import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.simplification.simplify_trajectories import simplify_trajectories
from src.models.trajectory_qds_model import TrajectoryQDSModel

def _make_points_and_queries(n=50):
    torch.manual_seed(0)
    pts = torch.zeros(n, 5)
    pts[:, 0] = torch.arange(n).float()
    pts[:, 1] = 50.0 + torch.rand(n)
    pts[:, 2] = 0.0 + torch.rand(n)
    pts[:, 3] = torch.rand(n) * 20.0
    pts[:, 4] = torch.rand(n) * 360.0
    queries = torch.tensor([
        [50.0, 51.0, 0.0, 1.0, 0.0, 25.0],
        [50.2, 50.8, 0.2, 0.8, 10.0, 40.0],
    ])
    return pts, queries

class TestSimplifyTrajectories:
    def test_returns_three_values(self):
        pts, queries = _make_points_and_queries()
        model = TrajectoryQDSModel()
        result = simplify_trajectories(pts, model, queries, threshold=0.5)
        assert len(result) == 3

    def test_output_columns(self):
        pts, queries = _make_points_and_queries()
        model = TrajectoryQDSModel()
        simplified, mask, scores = simplify_trajectories(pts, model, queries, threshold=0.5)
        assert simplified.shape[1] == 5

    def test_threshold_zero_keeps_all(self):
        pts, queries = _make_points_and_queries()
        model = TrajectoryQDSModel()
        simplified, mask, scores = simplify_trajectories(pts, model, queries, threshold=0.0)
        assert simplified.shape[0] == pts.shape[0]

    def test_threshold_one_falls_back(self):
        pts, queries = _make_points_and_queries()
        model = TrajectoryQDSModel()
        simplified, mask, scores = simplify_trajectories(pts, model, queries, threshold=1.0)
        assert simplified.shape[0] >= 1

    def test_non_empty_output(self):
        pts, queries = _make_points_and_queries()
        model = TrajectoryQDSModel()
        simplified, mask, scores = simplify_trajectories(pts, model, queries, threshold=0.5)
        assert simplified.shape[0] >= 1

    def test_mask_shape(self):
        pts, queries = _make_points_and_queries()
        model = TrajectoryQDSModel()
        simplified, mask, scores = simplify_trajectories(pts, model, queries, threshold=0.5)
        assert mask.shape == (pts.shape[0],)
        assert mask.dtype == torch.bool

    def test_scores_shape(self):
        pts, queries = _make_points_and_queries()
        model = TrajectoryQDSModel()
        simplified, mask, scores = simplify_trajectories(pts, model, queries, threshold=0.5)
        assert scores.shape == (pts.shape[0],)
