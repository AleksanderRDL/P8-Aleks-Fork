"""Baseline simplification methods for AIS-QDS evaluation. See src/evaluation/README.md for details."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from src.queries.workload import TypedQueryWorkload
from src.simplification.mlqds_scoring import simplify_mlqds_predictions, workload_type_head
from src.simplification.simplify_trajectories import (
    evenly_spaced_indices,
    simplify_with_temporal_score_hybrid,
    simplify_with_scores,
)
from src.training.train_model import TrainingOutputs
from src.training.inference import default_inference_device, windowed_predict


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
        ...


@dataclass
class MLQDSMethod:
    """Query-aware model-based simplification method. See src/evaluation/README.md for details."""

    name: str
    trained: TrainingOutputs
    workload: TypedQueryWorkload
    workload_type: str
    score_mode: str = "rank"
    score_temperature: float = 1.0
    rank_confidence_weight: float = 0.15
    temporal_fraction: float = 0.50
    diversity_bonus: float = 0.0
    inference_device: str | torch.device | None = None
    amp_mode: str = "off"
    inference_batch_size: int = 16

    def simplify(self, points: torch.Tensor, boundaries: list[tuple[int, int]], compression_ratio: float) -> torch.Tensor:
        """Simplify using one explicit typed workload score.

        If the workload contains no queries, keep every point because there is
        no query F1 objective to optimize. Otherwise, use the learned pure
        workload score with a temporal-coverage base, then fill the remaining
        budget with learned query-aware scores.

        See ``src/evaluation/README.md`` for details.
        """
        if not self.workload.typed_queries:
            return torch.ones((points.shape[0],), dtype=torch.bool, device=points.device)

        point_dim = self.trained.model.point_dim
        p, q = self.trained.scaler.transform(points[:, :point_dim], self.workload.query_features)
        device = torch.device(self.inference_device) if self.inference_device is not None else default_inference_device()
        pred = windowed_predict(
            model=self.trained.model,
            norm_points=p,
            boundaries=boundaries,
            queries=q,
            query_type_ids=self.workload.type_ids,
            batch_size=self.inference_batch_size,
            device=device,
            amp_mode=self.amp_mode,
        )

        return simplify_mlqds_predictions(
            pred,
            boundaries,
            self.workload_type,
            compression_ratio,
            temporal_fraction=self.temporal_fraction,
            diversity_bonus=self.diversity_bonus,
            score_mode=self.score_mode,
            score_temperature=self.score_temperature,
            rank_confidence_weight=self.rank_confidence_weight,
        )


@dataclass
class UniformTemporalMethod:
    """True evenly spaced temporal sampling baseline."""

    name: str = "uniform"

    def simplify(self, points: torch.Tensor, boundaries: list[tuple[int, int]], compression_ratio: float) -> torch.Tensor:
        """Retain evenly spaced points per trajectory, including endpoints."""
        retained = torch.zeros((points.shape[0],), dtype=torch.bool, device=points.device)
        for start, end in boundaries:
            n = end - start
            if n <= 0:
                continue
            k = max(2, int(torch.ceil(torch.tensor(float(compression_ratio) * n)).item()))
            k = min(k, n)
            local_idx = evenly_spaced_indices(n, k, points.device)
            retained[start + local_idx] = True
        return retained


@dataclass
class ScoreHybridMethod:
    """Temporal-base plus caller-supplied score-fill diagnostic method."""

    name: str
    scores: torch.Tensor
    temporal_fraction: float = 0.50
    diversity_bonus: float = 0.0

    def simplify(self, points: torch.Tensor, boundaries: list[tuple[int, int]], compression_ratio: float) -> torch.Tensor:
        """Retain temporal base points, then fill with supplied per-point scores."""
        if int(self.scores.numel()) != int(points.shape[0]):
            raise ValueError(
                "ScoreHybridMethod scores must match flattened points: "
                f"got {int(self.scores.numel())}, expected {int(points.shape[0])}."
            )
        scores = self.scores.to(device=points.device, dtype=torch.float32)
        return simplify_with_temporal_score_hybrid(
            scores,
            boundaries,
            compression_ratio,
            temporal_fraction=self.temporal_fraction,
            diversity_bonus=self.diversity_bonus,
        )


@dataclass
class DouglasPeuckerMethod:
    """Recursive Douglas-Peucker geometric baseline (heap-based).

    Standard DP recursion: keep first/last point, split at the inner point with
    max perpendicular distance to the chord. Implemented as a heap so we only
    perform the K splits needed to hit the target compression — not the full
    log(N) recursion depth — which is critical for long AIS trajectories.

    Returns a per-trajectory point retained mask directly; no surrogate scores.
    """

    name: str = "DouglasPeucker"

    @staticmethod
    def _farthest_in_segment(xy: np.ndarray, s: int, e: int) -> tuple[int, float]:
        """Return (split_idx, max_perp) for points strictly between s and e."""
        if e - s < 2:
            return -1, 0.0
        a = xy[s]
        b = xy[e]
        v = b - a
        v_norm_sq = float(v[0] * v[0] + v[1] * v[1])
        seg = xy[s + 1 : e]
        if v_norm_sq < 1e-12:
            d = np.linalg.norm(seg - a, axis=1)
        else:
            rel = seg - a
            proj = (rel @ v) / v_norm_sq
            closest = a + proj[:, None] * v
            d = np.linalg.norm(seg - closest, axis=1)
        local = int(np.argmax(d))
        return s + 1 + local, float(d[local])

    def _dp_retained_mask(self, traj_xy_np: np.ndarray, k_keep: int) -> np.ndarray:
        """Retain the k_keep points produced by DP recursion (largest perp first)."""
        n = int(traj_xy_np.shape[0])
        mask = np.zeros((n,), dtype=bool)
        if n == 0 or k_keep <= 0:
            return mask
        mask[0] = True
        if n == 1 or k_keep == 1:
            return mask
        mask[n - 1] = True
        if k_keep <= 2:
            return mask

        # Negative-distance heap so largest perp pops first.
        heap: list[tuple[float, int, int, int]] = []
        split_idx, perp = self._farthest_in_segment(traj_xy_np, 0, n - 1)
        if split_idx >= 0:
            heapq.heappush(heap, (-perp, split_idx, 0, n - 1))

        kept = 2
        tie = 0  # heap-stability tiebreak counter (keeps push order deterministic)
        while heap and kept < k_keep:
            _, split_idx, s, e = heapq.heappop(heap)
            if mask[split_idx]:
                continue
            mask[split_idx] = True
            kept += 1
            left_split, left_perp = self._farthest_in_segment(traj_xy_np, s, split_idx)
            if left_split >= 0:
                tie += 1
                heapq.heappush(heap, (-left_perp, left_split, s, split_idx))
            right_split, right_perp = self._farthest_in_segment(traj_xy_np, split_idx, e)
            if right_split >= 0:
                tie += 1
                heapq.heappush(heap, (-right_perp, right_split, split_idx, e))
        return mask

    def simplify(self, points: torch.Tensor, boundaries: list[tuple[int, int]], compression_ratio: float) -> torch.Tensor:
        """Retain DP-selected points per trajectory at the requested ratio."""
        retained = torch.zeros((points.shape[0],), dtype=torch.bool, device=points.device)
        ratio = max(0.0, min(1.0, float(compression_ratio)))
        xy_np = points[:, 1:3].detach().cpu().numpy().astype(np.float64)
        for start, end in boundaries:
            n = int(end - start)
            if n <= 0:
                continue
            k = max(2, int(math.ceil(ratio * n)))
            k = min(k, n)
            traj_mask = self._dp_retained_mask(xy_np[start:end], k)
            retained[start:end] = torch.from_numpy(traj_mask).to(retained.device)
        return retained


@dataclass
class OracleMethod:
    """Diagnostic additive-label Oracle, not an exact retained-set F1 optimizer."""

    labels: torch.Tensor
    workload_type: str
    name: str = "Oracle"
    oracle_kind: str = "additive_label_greedy"

    def simplify(self, points: torch.Tensor, boundaries: list[tuple[int, int]], compression_ratio: float) -> torch.Tensor:
        """Simplify using oracle label gains for one explicit workload."""
        _name, type_idx = workload_type_head(self.workload_type)
        if self.labels.ndim != 2 or type_idx >= self.labels.shape[1]:
            raise ValueError("Oracle labels must have shape [n_points, NUM_QUERY_TYPES].")
        score = self.labels[:, type_idx].float()
        return simplify_with_scores(score, boundaries, compression_ratio)
