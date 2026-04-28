"""Baseline simplification methods for AIS-QDS v2 evaluation. See src/evaluation/README.md for details."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import math

import torch

from src.experiments.experiment_config import TypedQueryWorkload
from src.simplification.simplify_trajectories import simplify_with_scores
from src.training.train_model import TrainingOutputs
from src.training.training_pipeline import windowed_predict


def _query_first_select(
    scores: torch.Tensor,
    near_mask: torch.Tensor,
    boundaries: list[tuple[int, int]],
    compression_ratio: float,
) -> torch.Tensor:
    """Per-trajectory query-first selection (no outside-buffer fallback).

    For each trajectory of length ``n``, ``k = ceil(compression_ratio * n)`` is
    the *maximum* budget. We then:

    1. If the trajectory never enters the query buffer, fall back to plain
       top-k by score (the trajectory is irrelevant to every query, so we
       still represent it at the requested compression).
    2. If the in-buffer count ``<= k``, keep **exactly** those in-buffer
       points. The trajectory ends up below the global compression ratio — we
       deliberately stop rather than waste slots on open ocean.
    3. If the in-buffer count ``> k``, keep the top-``k`` in-buffer points.

    First and last points are pinned so trajectories remain bounded.
    """
    retained = torch.zeros(scores.shape[0], dtype=torch.bool, device=scores.device)
    for start, end in boundaries:
        n = end - start
        if n <= 2:
            retained[start:end] = True
            continue
        k = max(2, int(math.ceil(compression_ratio * n)))
        k = min(k, n)

        local_scores = scores[start:end]
        local_near = near_mask[start:end]
        in_buf = int(local_near.sum().item())

        if in_buf == 0:
            idx = torch.topk(local_scores, k=k, largest=True).indices
        elif in_buf <= k:
            idx = torch.where(local_near)[0]
        else:
            buf_scores = local_scores.clone()
            buf_scores[~local_near] = float("-inf")
            idx = torch.topk(buf_scores, k=k, largest=True).indices

        retained[start:end][idx] = True
        retained[start] = True
        retained[end - 1] = True
    return retained


class Method(Protocol):
    """Simplification method protocol. See src/evaluation/README.md for details."""

    name: str

    def simplify(
        self,
        points: torch.Tensor,
        boundaries: list[tuple[int, int]],
        compression_ratio: float,
    ) -> torch.Tensor:
        """Return retained point mask. See src/evaluation/README.md for details."""


@dataclass
class MLQDSMethod:
    """Query-aware model-based simplification method. See src/evaluation/README.md for details."""

    name: str
    trained: TrainingOutputs
    workload: TypedQueryWorkload
    workload_mix: dict[str, float]
    query_area_boost: float = 1.0
    query_buffer_deg: float = 0.20

    def _query_proximity_mask(self, points: torch.Tensor) -> torch.Tensor:
        """True for points within `query_buffer_deg` of any query geometry.

        Uses axis-aligned lat/lon boxes (with a buffer) drawn around each query.
        For knn / similarity queries that lack an explicit bbox, a square of
        radius `query_buffer_deg` around the anchor/centroid is used.
        """
        buf = float(self.query_buffer_deg)
        if buf <= 0.0:
            return torch.zeros((points.shape[0],), dtype=torch.bool)
        lat = points[:, 1]
        lon = points[:, 2]
        mask = torch.zeros_like(lat, dtype=torch.bool)
        for q in self.workload.typed_queries:
            p = q["params"]
            qtype = q["type"]
            if qtype in ("range", "clustering"):
                lo_lat, hi_lat = float(p["lat_min"]) - buf, float(p["lat_max"]) + buf
                lo_lon, hi_lon = float(p["lon_min"]) - buf, float(p["lon_max"]) + buf
            elif qtype == "knn":
                lo_lat, hi_lat = float(p["lat"]) - buf, float(p["lat"]) + buf
                lo_lon, hi_lon = float(p["lon"]) - buf, float(p["lon"]) + buf
            elif qtype == "similarity":
                r = float(p.get("radius", 0.05)) + buf
                lo_lat, hi_lat = float(p["lat_query_centroid"]) - r, float(p["lat_query_centroid"]) + r
                lo_lon, hi_lon = float(p["lon_query_centroid"]) - r, float(p["lon_query_centroid"]) + r
            else:
                continue
            mask |= (lat >= lo_lat) & (lat <= hi_lat) & (lon >= lo_lon) & (lon <= hi_lon)
        return mask

    def simplify(self, points: torch.Tensor, boundaries: list[tuple[int, int]], compression_ratio: float) -> torch.Tensor:
        """Simplify using workload-weighted typed scores.

        Budget strategy (query-first):

        * If the workload contains **0 queries**, keep every point — nothing to
          optimise for. This honours the "with 0 queries, remove 0 points" rule.
        * Otherwise, compute per-trajectory budget
          ``k = ceil(compression_ratio * n)``. Points inside the query buffer
          (see :meth:`_query_proximity_mask`) are prioritised: every in-buffer
          point is kept up to the budget, then the remaining slots are filled
          with the highest-scoring points outside the buffer. If the in-buffer
          set alone exceeds the budget, pick the top-scoring in-buffer points
          (and the first/last stay pinned).

        See ``src/evaluation/README.md`` for details.
        """
        if not self.workload.typed_queries:
            return torch.ones((points.shape[0],), dtype=torch.bool, device=points.device)

        point_dim = self.trained.model.point_dim
        p, q = self.trained.scaler.transform(points[:, :point_dim], self.workload.query_features)
        pred = windowed_predict(
            model=self.trained.model,
            norm_points=p,
            boundaries=boundaries,
            queries=q,
            query_type_ids=self.workload.type_ids,
        )

        type_order = ["range", "knn", "similarity", "clustering"]
        mix = torch.tensor([float(self.workload_mix.get(t, 0.0)) for t in type_order], dtype=torch.float32)
        if float(mix.sum().item()) <= 0.0:
            mix = torch.ones_like(mix) / mix.numel()
        else:
            mix = mix / mix.sum()
        score = (pred * mix.unsqueeze(0)).sum(dim=1)

        if self.query_buffer_deg > 0.0:
            near_mask = self._query_proximity_mask(points)
            if self.query_area_boost != 1.0 and bool(near_mask.any().item()):
                score = score.clone()
                score[near_mask] = score[near_mask] * float(self.query_area_boost)
            return _query_first_select(score, near_mask, boundaries, compression_ratio)

        return simplify_with_scores(score, boundaries, compression_ratio)

@dataclass
class RandomMethod:
    """Random baseline method. See src/evaluation/README.md for details."""

    name: str = "Random"
    seed: int = 0

    def simplify(self, points: torch.Tensor, boundaries: list[tuple[int, int]], compression_ratio: float) -> torch.Tensor:
        """Retain random points per trajectory at matched ratio. See src/evaluation/README.md for details."""
        g = torch.Generator().manual_seed(int(self.seed))
        scores = torch.rand((points.shape[0],), generator=g)
        return simplify_with_scores(scores, boundaries, compression_ratio)


@dataclass
class UniformTemporalMethod:
    """Uniform temporal sampling baseline. See src/evaluation/README.md for details."""

    name: str = "UniformTemporal"

    def simplify(self, points: torch.Tensor, boundaries: list[tuple[int, int]], compression_ratio: float) -> torch.Tensor:
        """Retain approximately every k-th point per trajectory. See src/evaluation/README.md for details."""
        scores = torch.zeros((points.shape[0],), dtype=torch.float32)
        for start, end in boundaries:
            n = end - start
            idx = torch.arange(n, dtype=torch.float32)
            scores[start:end] = -torch.abs(idx - (n - 1) / 2.0)
        return simplify_with_scores(scores, boundaries, compression_ratio)


@dataclass
class DouglasPeuckerMethod:
    """Douglas-Peucker style geometric baseline. See src/evaluation/README.md for details."""

    name: str = "DouglasPeucker"

    def _local_scores(self, traj_xy: torch.Tensor) -> torch.Tensor:
        """Approximate DP importance by perpendicular-distance proxy. See src/evaluation/README.md for details."""
        n = traj_xy.shape[0]
        if n <= 2:
            return torch.ones((n,), dtype=torch.float32)
        a = traj_xy[0]
        b = traj_xy[-1]
        v = b - a
        v_norm = torch.norm(v) + 1e-8
        rel = traj_xy - a
        proj = (rel @ v) / v_norm
        perp = torch.norm(rel - torch.outer(proj / v_norm, v), dim=1)
        perp[0] = perp[-1] = perp.max() + 1.0
        return perp

    def simplify(self, points: torch.Tensor, boundaries: list[tuple[int, int]], compression_ratio: float) -> torch.Tensor:
        """Simplify with DP-like per-trajectory geometric scores. See src/evaluation/README.md for details."""
        scores = torch.zeros((points.shape[0],), dtype=torch.float32)
        for start, end in boundaries:
            scores[start:end] = self._local_scores(points[start:end, 1:3])
        return simplify_with_scores(scores, boundaries, compression_ratio)


@dataclass
class OracleMethod:
    """Diagnostic upper-bound method using oracle labels directly. See src/evaluation/README.md for details."""

    labels: torch.Tensor
    workload_mix: dict[str, float]
    name: str = "Oracle"

    def simplify(self, points: torch.Tensor, boundaries: list[tuple[int, int]], compression_ratio: float) -> torch.Tensor:
        """Simplify using weighted oracle labels. See src/evaluation/README.md for details."""
        mix = torch.tensor(
            [
                self.workload_mix.get("range", 0.0),
                self.workload_mix.get("knn", 0.0),
                self.workload_mix.get("similarity", 0.0),
                self.workload_mix.get("clustering", 0.0),
            ],
            dtype=torch.float32,
        )
        mix = mix / max(float(mix.sum().item()), 1e-6)
        score = (self.labels * mix.unsqueeze(0)).sum(dim=1)
        return simplify_with_scores(score, boundaries, compression_ratio)
