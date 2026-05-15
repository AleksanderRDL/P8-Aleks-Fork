"""Baseline simplification methods for AIS-QDS evaluation. See evaluation/README.md for details."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch

from queries.workload import TypedQueryWorkload
from simplification.mlqds_scoring import mlqds_simplification_scores, workload_type_head
from simplification.learned_segment_budget import simplify_with_learned_segment_budget_v1
from simplification.simplify_trajectories import (
    evenly_spaced_indices,
    simplify_with_global_score_budget,
    simplify_with_temporal_score_hybrid,
    simplify_with_scores,
)
from training.training_outputs import TrainingOutputs
from training.inference import (
    _is_workload_blind_model,
    _model_point_dim,
    default_inference_device,
    windowed_predict,
    windowed_predict_with_heads,
)
from training.model_features import build_model_point_features_for_dim
from training.query_useful_targets import QUERY_USEFUL_V1_HEAD_NAMES


class Method(Protocol):
    """Simplification method protocol. See evaluation/README.md for details."""

    name: str

    def simplify(
        self,
        points: torch.Tensor,
        boundaries: list[tuple[int, int]],
        compression_ratio: float,
    ) -> torch.Tensor:
        """Return retained point mask. See evaluation/README.md for details."""
        ...


@dataclass
class FrozenMaskMethod:
    """Evaluation wrapper for a retained mask frozen before query scoring."""

    name: str
    retained_mask: torch.Tensor
    latency_ms: float = 0.0

    def simplify(
        self,
        points: torch.Tensor,
        boundaries: list[tuple[int, int]],
        compression_ratio: float,
    ) -> torch.Tensor:
        """Return the precomputed retained mask."""
        del boundaries, compression_ratio
        if int(self.retained_mask.numel()) != int(points.shape[0]):
            raise ValueError(
                "Frozen retained mask must match point count: "
                f"got {int(self.retained_mask.numel())}, expected {int(points.shape[0])}."
            )
        return self.retained_mask.to(device=points.device, dtype=torch.bool)


@dataclass
class MLQDSMethod:
    """Query-aware model-based simplification method. See evaluation/README.md for details."""

    name: str
    trained: TrainingOutputs
    workload: TypedQueryWorkload
    workload_type: str
    score_mode: str = "rank"
    score_temperature: float = 1.0
    rank_confidence_weight: float = 0.15
    temporal_fraction: float = 0.50
    diversity_bonus: float = 0.0
    hybrid_mode: str = "fill"
    stratified_center_weight: float = 0.0
    min_learned_swaps: int = 0
    selector_type: str = "temporal_hybrid"
    range_geometry_blend: float = 0.0
    range_geometry_scores: torch.Tensor | None = None
    trajectory_mmsis: list[int] | None = None
    inference_device: str | torch.device | None = None
    amp_mode: str = "off"
    inference_batch_size: int = 16
    _score_cache_key: tuple[object, ...] | None = field(default=None, init=False, repr=False)
    _score_cache: torch.Tensor | None = field(default=None, init=False, repr=False)
    _raw_pred_cache: torch.Tensor | None = field(default=None, init=False, repr=False)
    _head_logit_cache: torch.Tensor | None = field(default=None, init=False, repr=False)
    _segment_score_cache: torch.Tensor | None = field(default=None, init=False, repr=False)

    def _current_score_cache_key(
        self,
        points: torch.Tensor,
        boundaries: list[tuple[int, int]],
    ) -> tuple[object, ...]:
        """Return a conservative key for scores that are reusable across budgets."""
        geometry_key: tuple[object, ...] | None = None
        if self.range_geometry_scores is not None:
            geometry_key = (
                int(self.range_geometry_scores.data_ptr()),
                tuple(int(dim) for dim in self.range_geometry_scores.shape),
                str(self.range_geometry_scores.device),
                str(self.range_geometry_scores.dtype),
            )
        query_key: tuple[object, ...] | None = None
        if self.workload.query_features is not None:
            query_key = (
                int(self.workload.query_features.data_ptr()),
                tuple(int(dim) for dim in self.workload.query_features.shape),
                str(self.workload.query_features.device),
                str(self.workload.query_features.dtype),
            )
        type_id_key: tuple[object, ...] | None = None
        if self.workload.type_ids is not None:
            type_id_key = (
                int(self.workload.type_ids.data_ptr()),
                tuple(int(dim) for dim in self.workload.type_ids.shape),
                str(self.workload.type_ids.device),
                str(self.workload.type_ids.dtype),
            )
        return (
            id(self.trained.model),
            id(self.trained.scaler),
            int(points.data_ptr()),
            tuple(int(dim) for dim in points.shape),
            str(points.device),
            str(points.dtype),
            tuple((int(start), int(end)) for start, end in boundaries),
            str(self.workload_type),
            query_key,
            type_id_key,
            str(self.score_mode),
            float(self.score_temperature),
            float(self.rank_confidence_weight),
            float(self.range_geometry_blend),
            geometry_key,
            tuple(int(mmsi) for mmsi in self.trajectory_mmsis) if self.trajectory_mmsis is not None else None,
            int(self.min_learned_swaps),
            str(self.selector_type),
            str(self.inference_device),
            str(self.amp_mode),
            int(self.inference_batch_size),
        )

    def _simplification_scores(
        self,
        points: torch.Tensor,
        boundaries: list[tuple[int, int]],
        *,
        workload_blind: bool,
    ) -> torch.Tensor:
        """Return cached query/workload scores for budget-independent retention."""
        cache_key = self._current_score_cache_key(points, boundaries)
        if self._score_cache_key == cache_key and self._score_cache is not None:
            return self._score_cache

        if self.range_geometry_blend >= 1.0 and self.range_geometry_scores is not None:
            pred = torch.zeros((points.shape[0],), dtype=torch.float32, device=points.device)
            self._raw_pred_cache = pred.detach().cpu().float()
            self._head_logit_cache = None
            self._segment_score_cache = None
        else:
            point_dim = _model_point_dim(self.trained.model)
            model_points = build_model_point_features_for_dim(
                points,
                self.workload,
                point_dim,
                boundaries=boundaries,
                trajectory_mmsis=self.trajectory_mmsis,
                query_prior_field=self.trained.feature_context.get("query_prior_field"),
            )
            if workload_blind:
                norm_points = self.trained.scaler.transform_points(model_points)
                norm_queries = None
                query_type_ids = None
            else:
                norm_points, norm_queries = self.trained.scaler.transform(model_points, self.workload.query_features)
                query_type_ids = self.workload.type_ids
            device = (
                torch.device(self.inference_device)
                if self.inference_device is not None
                else default_inference_device()
            )
            if workload_blind and callable(getattr(self.trained.model, "forward_with_heads", None)):
                pred, head_logits = windowed_predict_with_heads(
                    model=self.trained.model,
                    norm_points=norm_points,
                    boundaries=boundaries,
                    queries=norm_queries,
                    query_type_ids=query_type_ids,
                    batch_size=self.inference_batch_size,
                    device=device,
                    amp_mode=self.amp_mode,
                )
            else:
                pred = windowed_predict(
                    model=self.trained.model,
                    norm_points=norm_points,
                    boundaries=boundaries,
                    queries=norm_queries,
                    query_type_ids=query_type_ids,
                    batch_size=self.inference_batch_size,
                    device=device,
                    amp_mode=self.amp_mode,
                )
                head_logits = None
            if head_logits is not None:
                self._head_logit_cache = head_logits.detach().cpu().float()
                try:
                    segment_head_idx = tuple(QUERY_USEFUL_V1_HEAD_NAMES).index("segment_budget_target")
                except ValueError:
                    segment_head_idx = -1
                if segment_head_idx >= 0 and int(head_logits.shape[-1]) > segment_head_idx:
                    self._segment_score_cache = head_logits[:, segment_head_idx].detach().cpu().float()
            else:
                self._head_logit_cache = None
                self._segment_score_cache = None
            self._raw_pred_cache = pred.detach().cpu().float()

        scores = mlqds_simplification_scores(
            pred,
            boundaries,
            self.workload_type,
            score_mode=self.score_mode,
            score_temperature=self.score_temperature,
            rank_confidence_weight=self.rank_confidence_weight,
            range_geometry_scores=self.range_geometry_scores,
            range_geometry_blend=self.range_geometry_blend,
        )
        self._score_cache_key = cache_key
        self._score_cache = scores
        return scores

    def simplify(
        self,
        points: torch.Tensor,
        boundaries: list[tuple[int, int]],
        compression_ratio: float,
    ) -> torch.Tensor:
        """Simplify using one explicit typed workload score.

        If the workload contains no queries, keep every point because there is
        no query F1 objective to optimize. Otherwise, use the learned pure
        workload score with a temporal-coverage base, then fill the remaining
        budget with learned query-aware scores.

        See ``evaluation/README.md`` for details.
        """
        if not self.workload.typed_queries:
            return torch.ones((points.shape[0],), dtype=torch.bool, device=points.device)
        workload_blind = _is_workload_blind_model(self.trained.model)
        if workload_blind and self.range_geometry_blend > 0.0:
            raise ValueError(
                "workload_blind_range evaluation cannot use mlqds_range_geometry_blend; "
                "eval labels would affect the retained mask."
            )

        scores = self._simplification_scores(points, boundaries, workload_blind=workload_blind)
        if str(self.selector_type).lower() == "learned_segment_budget_v1":
            return simplify_with_learned_segment_budget_v1(
                scores,
                boundaries,
                compression_ratio,
                segment_scores=self._segment_score_cache,
                points=points,
            )
        return simplify_with_temporal_score_hybrid(
            scores,
            boundaries,
            compression_ratio,
            temporal_fraction=self.temporal_fraction,
            diversity_bonus=self.diversity_bonus,
            hybrid_mode=self.hybrid_mode,
            stratified_center_weight=self.stratified_center_weight,
            min_learned_swaps=self.min_learned_swaps,
        )


@dataclass
class UniformTemporalMethod:
    """True evenly spaced temporal sampling baseline."""

    name: str = "uniform"

    def simplify(
        self,
        points: torch.Tensor,
        boundaries: list[tuple[int, int]],
        compression_ratio: float,
    ) -> torch.Tensor:
        """Retain evenly spaced points per trajectory, including endpoints."""
        retained = torch.zeros((points.shape[0],), dtype=torch.bool, device=points.device)
        for start, end in boundaries:
            point_count = end - start
            if point_count <= 0:
                continue
            keep_count = max(2, int(torch.ceil(torch.tensor(float(compression_ratio) * point_count)).item()))
            keep_count = min(keep_count, point_count)
            local_indices = evenly_spaced_indices(point_count, keep_count, points.device)
            retained[start + local_indices] = True
        return retained


@dataclass
class ScoreHybridMethod:
    """Temporal-base plus caller-supplied score-fill diagnostic method."""

    name: str
    scores: torch.Tensor
    temporal_fraction: float = 0.50
    diversity_bonus: float = 0.0
    hybrid_mode: str = "fill"
    stratified_center_weight: float = 0.0
    min_learned_swaps: int = 0

    def simplify(
        self,
        points: torch.Tensor,
        boundaries: list[tuple[int, int]],
        compression_ratio: float,
    ) -> torch.Tensor:
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
            hybrid_mode=self.hybrid_mode,
            stratified_center_weight=self.stratified_center_weight,
            min_learned_swaps=self.min_learned_swaps,
        )


@dataclass
class ScoreGlobalBudgetMethod:
    """Diagnostic global score allocation with per-trajectory skeleton safeguards."""

    name: str
    scores: torch.Tensor
    min_points_per_trajectory: int = 2

    def simplify(
        self,
        points: torch.Tensor,
        boundaries: list[tuple[int, int]],
        compression_ratio: float,
    ) -> torch.Tensor:
        """Retain endpoint skeletons, then spend remaining budget by global score."""
        if int(self.scores.numel()) != int(points.shape[0]):
            raise ValueError(
                "ScoreGlobalBudgetMethod scores must match flattened points: "
                f"got {int(self.scores.numel())}, expected {int(points.shape[0])}."
            )
        scores = self.scores.to(device=points.device, dtype=torch.float32)
        return simplify_with_global_score_budget(
            scores,
            boundaries,
            compression_ratio,
            min_points_per_trajectory=self.min_points_per_trajectory,
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
    def _farthest_in_segment(xy: np.ndarray, start: int, end: int) -> tuple[int, float]:
        """Return (split_idx, max_perp) for points strictly between start and end."""
        if end - start < 2:
            return -1, 0.0
        start_point = xy[start]
        end_point = xy[end]
        segment_vector = end_point - start_point
        segment_norm_sq = float(segment_vector[0] * segment_vector[0] + segment_vector[1] * segment_vector[1])
        interior_points = xy[start + 1 : end]
        if segment_norm_sq < 1e-12:
            distances = np.linalg.norm(interior_points - start_point, axis=1)
        else:
            relative_points = interior_points - start_point
            projection = (relative_points @ segment_vector) / segment_norm_sq
            closest_points = start_point + projection[:, None] * segment_vector
            distances = np.linalg.norm(interior_points - closest_points, axis=1)
        local_split_idx = int(np.argmax(distances))
        return start + 1 + local_split_idx, float(distances[local_split_idx])

    def _dp_retained_mask(self, traj_xy_np: np.ndarray, k_keep: int) -> np.ndarray:
        """Retain the k_keep points produced by DP recursion (largest perp first)."""
        point_count = int(traj_xy_np.shape[0])
        mask = np.zeros((point_count,), dtype=bool)
        if point_count == 0 or k_keep <= 0:
            return mask
        mask[0] = True
        if point_count == 1 or k_keep == 1:
            return mask
        mask[point_count - 1] = True
        if k_keep <= 2:
            return mask

        # Negative-distance heap so largest perp pops first.
        heap: list[tuple[float, int, int, int]] = []
        split_idx, perpendicular_distance = self._farthest_in_segment(traj_xy_np, 0, point_count - 1)
        if split_idx >= 0:
            heapq.heappush(heap, (-perpendicular_distance, split_idx, 0, point_count - 1))

        kept = 2
        while heap and kept < k_keep:
            _, split_idx, start, end = heapq.heappop(heap)
            if mask[split_idx]:
                continue
            mask[split_idx] = True
            kept += 1
            left_split, left_perpendicular = self._farthest_in_segment(traj_xy_np, start, split_idx)
            if left_split >= 0:
                heapq.heappush(heap, (-left_perpendicular, left_split, start, split_idx))
            right_split, right_perpendicular = self._farthest_in_segment(traj_xy_np, split_idx, end)
            if right_split >= 0:
                heapq.heappush(heap, (-right_perpendicular, right_split, split_idx, end))
        return mask

    def simplify(
        self,
        points: torch.Tensor,
        boundaries: list[tuple[int, int]],
        compression_ratio: float,
    ) -> torch.Tensor:
        """Retain DP-selected points per trajectory at the requested ratio."""
        retained = torch.zeros((points.shape[0],), dtype=torch.bool, device=points.device)
        ratio = max(0.0, min(1.0, float(compression_ratio)))
        xy_np = points[:, 1:3].detach().cpu().numpy().astype(np.float64)
        for start, end in boundaries:
            point_count = int(end - start)
            if point_count <= 0:
                continue
            keep_count = max(2, int(math.ceil(ratio * point_count)))
            keep_count = min(keep_count, point_count)
            trajectory_mask = self._dp_retained_mask(xy_np[start:end], keep_count)
            retained[start:end] = torch.from_numpy(trajectory_mask).to(retained.device)
        return retained


@dataclass
class OracleMethod:
    """Diagnostic additive-label Oracle, not an exact retained-set F1 optimizer."""

    labels: torch.Tensor
    workload_type: str
    name: str = "Oracle"
    oracle_kind: str = "additive_label_greedy"

    def simplify(
        self,
        points: torch.Tensor,
        boundaries: list[tuple[int, int]],
        compression_ratio: float,
    ) -> torch.Tensor:
        """Simplify using oracle label gains for one explicit workload."""
        _name, type_idx = workload_type_head(self.workload_type)
        if self.labels.ndim != 2 or type_idx >= self.labels.shape[1]:
            raise ValueError("Oracle labels must have shape [n_points, NUM_QUERY_TYPES].")
        score = self.labels[:, type_idx].float()
        return simplify_with_scores(score, boundaries, compression_ratio)
