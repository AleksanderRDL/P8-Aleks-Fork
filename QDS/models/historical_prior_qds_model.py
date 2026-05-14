"""Query-free historical-prior scorer for workload-blind range compression."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.workload_blind_qds_model import WorkloadBlindRangeQDSModel
from training.model_features import (
    HISTORICAL_PRIOR_DENSITY_POINT_DIM,
    HISTORICAL_PRIOR_MMSI_POINT_DIM,
    HISTORICAL_PRIOR_POINT_DIM,
)


_SOURCE_AGGREGATIONS = {"none", "mean", "min", "median"}


class HistoricalPriorRangeQDSModel(nn.Module):
    """Nonparametric scorer fitted from train-day range supervision.

    The model stores normalized query-free train point features and their
    retained-frequency targets. At inference it scores future points by
    inverse-distance weighted nearest-neighbor target value. It never consumes
    validation or eval query tensors while building retained masks.
    """

    workload_blind = True
    window_independent = True

    def __init__(
        self,
        point_dim: int,
        query_dim: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        type_embed_dim: int = 16,
        query_chunk_size: int = 512,
        dropout: float = 0.1,
        historical_prior_k: int = 32,
        historical_prior_clock_weight: float = 0.0,
        historical_prior_mmsi_weight: float = 1.0,
        historical_prior_density_weight: float = 1.0,
        historical_prior_min_target: float = 0.0,
        historical_prior_source_aggregation: str = "none",
        prior_feature_count: int = 0,
    ) -> None:
        super().__init__()
        del num_heads, num_layers, type_embed_dim, dropout
        self.point_dim = int(point_dim)
        self.query_dim = int(query_dim)
        self.embed_dim = int(embed_dim)
        self.query_chunk_size = int(query_chunk_size)
        self.historical_prior_k = max(1, int(historical_prior_k))
        self.historical_prior_clock_weight = max(0.0, float(historical_prior_clock_weight))
        self.historical_prior_mmsi_weight = max(0.0, float(historical_prior_mmsi_weight))
        self.historical_prior_density_weight = max(0.0, float(historical_prior_density_weight))
        self.historical_prior_min_target = max(0.0, float(historical_prior_min_target))
        self.historical_prior_source_aggregation = str(historical_prior_source_aggregation).lower()
        if self.historical_prior_source_aggregation not in _SOURCE_AGGREGATIONS:
            raise ValueError(
                "historical_prior_source_aggregation must be one of "
                f"{sorted(_SOURCE_AGGREGATIONS)}; got {historical_prior_source_aggregation!r}."
            )
        self.historical_prior_clock_dim = 2 if self.point_dim in {
            HISTORICAL_PRIOR_POINT_DIM,
            HISTORICAL_PRIOR_MMSI_POINT_DIM,
        } else 0
        self.historical_prior_density_dim = 2 if self.point_dim in {
            HISTORICAL_PRIOR_DENSITY_POINT_DIM,
            HISTORICAL_PRIOR_POINT_DIM,
            HISTORICAL_PRIOR_MMSI_POINT_DIM,
        } else 0
        self.historical_prior_mmsi_dim = 4 if self.point_dim == HISTORICAL_PRIOR_MMSI_POINT_DIM else 0
        self.register_buffer(
            "historical_features",
            torch.zeros((max(0, int(prior_feature_count)), self.point_dim), dtype=torch.float32),
        )
        self.register_buffer(
            "historical_targets",
            torch.zeros((max(0, int(prior_feature_count)),), dtype=torch.float32),
        )
        self.register_buffer(
            "historical_source_ids",
            torch.zeros((max(0, int(prior_feature_count)),), dtype=torch.long),
        )

    def set_prior(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        source_ids: torch.Tensor | None = None,
    ) -> None:
        """Store normalized query-free train features and soft retained targets."""
        if features.ndim != 2 or int(features.shape[1]) != self.point_dim:
            raise ValueError(
                "historical prior features must have shape [n_points, point_dim]; "
                f"got {tuple(features.shape)} for point_dim={self.point_dim}."
            )
        if targets.ndim != 1 or int(targets.shape[0]) != int(features.shape[0]):
            raise ValueError("historical prior targets must be a vector matching feature rows.")
        if int(features.shape[0]) == 0:
            raise ValueError("historical prior requires at least one training point.")
        if source_ids is not None:
            if source_ids.ndim != 1 or int(source_ids.shape[0]) != int(features.shape[0]):
                raise ValueError("historical prior source ids must be a vector matching feature rows.")
            stored_source_ids = source_ids.detach().to(device=features.device, dtype=torch.long)
            if bool((stored_source_ids < 0).any().item()):
                raise ValueError("historical prior source ids must be non-negative.")
        else:
            stored_source_ids = torch.zeros((int(features.shape[0]),), dtype=torch.long, device=features.device)
        stored_features = features.detach().to(dtype=torch.float32)
        stored_targets = targets.detach().to(dtype=torch.float32).clamp(0.0, 1.0)
        min_target = float(self.historical_prior_min_target)
        if min_target > 0.0:
            keep = stored_targets >= min_target
            if not bool(keep.any().item()):
                raise ValueError(
                    "historical_prior_min_target removed every training point; "
                    f"threshold={min_target:.6f}, max_target={float(stored_targets.max().item()):.6f}."
                )
            stored_features = stored_features[keep]
            stored_targets = stored_targets[keep]
            stored_source_ids = stored_source_ids[keep]
        self.historical_features = stored_features.clone()
        self.historical_targets = stored_targets.clone()
        self.historical_source_ids = stored_source_ids.detach().to(dtype=torch.long).clone()

    def _score_against(
        self,
        flat: torch.Tensor,
        features: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Score flattened points against one historical support matrix."""
        k = min(self.historical_prior_k, int(features.shape[0]))
        chunk_size = max(1, min(max(1, int(self.query_chunk_size)), 2048))
        scores = torch.empty((flat.shape[0],), dtype=torch.float32, device=flat.device)
        for start in range(0, int(flat.shape[0]), chunk_size):
            end = min(int(flat.shape[0]), start + chunk_size)
            distances = torch.cdist(flat[start:end], features, p=2).square()
            nearest_distances, nearest_idx = torch.topk(distances, k=k, largest=False, dim=1)
            weights = 1.0 / (nearest_distances + 1e-4)
            scores[start:end] = (weights * targets[nearest_idx]).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-9)
        return scores

    def _score_flat(self, flat_points: torch.Tensor) -> torch.Tensor:
        """Score flattened normalized points against stored train priors."""
        if int(self.historical_features.shape[0]) == 0:
            raise RuntimeError("HistoricalPriorRangeQDSModel has no stored prior features.")
        features = self.historical_features.to(device=flat_points.device, dtype=torch.float32)
        targets = self.historical_targets.to(device=flat_points.device, dtype=torch.float32)
        source_ids = self.historical_source_ids.to(device=flat_points.device, dtype=torch.long)
        flat = flat_points.float()
        needs_weighting = (
            (self.historical_prior_clock_dim > 0 and self.historical_prior_clock_weight != 1.0)
            or (self.historical_prior_mmsi_dim > 0 and self.historical_prior_mmsi_weight != 1.0)
            or (self.historical_prior_density_dim > 0 and self.historical_prior_density_weight != 1.0)
        )
        if needs_weighting:
            features = features.clone()
            flat = flat.clone()
        if self.historical_prior_clock_dim > 0 and self.historical_prior_clock_weight != 1.0:
            feature_weight = float(self.historical_prior_clock_weight)
            clock_start = self.point_dim - self.historical_prior_density_dim - self.historical_prior_clock_dim
            clock_end = self.point_dim - self.historical_prior_density_dim
            features[:, clock_start:clock_end] *= feature_weight
            flat[:, clock_start:clock_end] *= feature_weight
        if self.historical_prior_mmsi_dim > 0 and self.historical_prior_mmsi_weight != 1.0:
            feature_weight = float(self.historical_prior_mmsi_weight)
            clock_start = self.point_dim - self.historical_prior_density_dim - self.historical_prior_clock_dim
            mmsi_start = clock_start - self.historical_prior_mmsi_dim
            features[:, mmsi_start:clock_start] *= feature_weight
            flat[:, mmsi_start:clock_start] *= feature_weight
        if self.historical_prior_density_dim > 0 and self.historical_prior_density_weight != 1.0:
            feature_weight = float(self.historical_prior_density_weight)
            features[:, -self.historical_prior_density_dim :] *= feature_weight
            flat[:, -self.historical_prior_density_dim :] *= feature_weight
        source_aggregation = str(self.historical_prior_source_aggregation)
        unique_sources = torch.unique(source_ids, sorted=True)
        if source_aggregation == "none" or int(unique_sources.numel()) <= 1:
            return self._score_against(flat, features, targets)

        source_scores = []
        for source_id in unique_sources:
            source_mask = source_ids == source_id
            if not bool(source_mask.any().item()):
                continue
            source_scores.append(self._score_against(flat, features[source_mask], targets[source_mask]))
        if not source_scores:
            return self._score_against(flat, features, targets)
        stacked = torch.stack(source_scores, dim=0)
        if source_aggregation == "mean":
            return stacked.mean(dim=0)
        if source_aggregation == "min":
            return stacked.min(dim=0).values
        if source_aggregation == "median":
            return stacked.median(dim=0).values
        raise RuntimeError(f"Unsupported historical prior source aggregation: {source_aggregation}")

    def forward(
        self,
        points: torch.Tensor,
        queries: torch.Tensor | None = None,
        query_type_ids: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict one query-independent per-point prior score stream."""
        del queries, query_type_ids, padding_mask
        if points.ndim != 3 or int(points.shape[-1]) != self.point_dim:
            raise ValueError(
                "HistoricalPriorRangeQDSModel expects points with shape [batch, length, point_dim]."
            )
        batch, length, _ = points.shape
        flat_scores = self._score_flat(points.reshape(batch * length, self.point_dim))
        return flat_scores.reshape(batch, length).to(dtype=points.dtype)


class HistoricalPriorStudentRangeQDSModel(nn.Module):
    """Trainable workload-blind scorer with an explicit historical-prior feature.

    The stored prior is fitted only from train-day query-derived supervision.
    During validation/eval compression the model computes a query-free KNN prior
    score for each point, appends it to the blind point features, and lets the
    neural scorer learn how to use or ignore that prior.
    """

    workload_blind = True

    def __init__(
        self,
        point_dim: int,
        query_dim: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        type_embed_dim: int = 16,
        query_chunk_size: int = 512,
        dropout: float = 0.1,
        historical_prior_k: int = 32,
        historical_prior_clock_weight: float = 0.0,
        historical_prior_mmsi_weight: float = 1.0,
        historical_prior_density_weight: float = 1.0,
        historical_prior_min_target: float = 0.0,
        historical_prior_source_aggregation: str = "none",
        prior_feature_count: int = 0,
    ) -> None:
        super().__init__()
        self.point_dim = int(point_dim)
        self.query_dim = int(query_dim)
        self.embed_dim = int(embed_dim)
        self.query_chunk_size = int(query_chunk_size)
        self.prior = HistoricalPriorRangeQDSModel(
            point_dim=self.point_dim,
            query_dim=self.query_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            type_embed_dim=type_embed_dim,
            query_chunk_size=query_chunk_size,
            dropout=dropout,
            historical_prior_k=historical_prior_k,
            historical_prior_clock_weight=historical_prior_clock_weight,
            historical_prior_mmsi_weight=historical_prior_mmsi_weight,
            historical_prior_density_weight=historical_prior_density_weight,
            historical_prior_min_target=historical_prior_min_target,
            historical_prior_source_aggregation=historical_prior_source_aggregation,
            prior_feature_count=prior_feature_count,
        )
        self.student = WorkloadBlindRangeQDSModel(
            point_dim=self.point_dim + 1,
            query_dim=self.query_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            type_embed_dim=type_embed_dim,
            query_chunk_size=query_chunk_size,
            dropout=dropout,
        )

    @property
    def historical_prior_k(self) -> int:
        return self.prior.historical_prior_k

    @property
    def historical_prior_clock_weight(self) -> float:
        return self.prior.historical_prior_clock_weight

    @property
    def historical_prior_mmsi_weight(self) -> float:
        return self.prior.historical_prior_mmsi_weight

    @property
    def historical_prior_density_weight(self) -> float:
        return self.prior.historical_prior_density_weight

    @property
    def historical_prior_min_target(self) -> float:
        return self.prior.historical_prior_min_target

    @property
    def historical_prior_source_aggregation(self) -> str:
        return self.prior.historical_prior_source_aggregation

    @property
    def historical_features(self) -> torch.Tensor:
        return self.prior.historical_features

    @property
    def historical_targets(self) -> torch.Tensor:
        return self.prior.historical_targets

    @property
    def historical_source_ids(self) -> torch.Tensor:
        return self.prior.historical_source_ids

    def set_prior(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        source_ids: torch.Tensor | None = None,
    ) -> None:
        """Store normalized train features for the query-free prior input."""
        self.prior.set_prior(features, targets, source_ids=source_ids)

    def forward(
        self,
        points: torch.Tensor,
        queries: torch.Tensor | None = None,
        query_type_ids: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict one query-independent per-point score stream."""
        if points.ndim != 3 or int(points.shape[-1]) != self.point_dim:
            raise ValueError(
                "HistoricalPriorStudentRangeQDSModel expects points with shape "
                "[batch, length, point_dim]."
            )
        batch, length, _ = points.shape
        with torch.no_grad():
            prior_scores = self.prior._score_flat(points.reshape(batch * length, self.point_dim))
        prior_feature = prior_scores.reshape(batch, length, 1).to(dtype=points.dtype)
        augmented_points = torch.cat([points, prior_feature], dim=-1)
        return self.student(
            points=augmented_points,
            queries=queries,
            query_type_ids=query_type_ids,
            padding_mask=padding_mask,
        )
