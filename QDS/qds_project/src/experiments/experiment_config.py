"""Shared configuration and result types for the AIS experiment."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Mapping

import torch


@dataclass(frozen=True)
class DataConfig:
    """Configuration for loading or generating AIS trajectories."""

    n_ships: int = 10
    n_points_per_ship: int = 100
    csv_path: str | None = None
    save_csv: bool = False


@dataclass(frozen=True)
class QueryConfig:
    """Configuration for generating query workloads."""

    workload: str = "density"
    n_queries: int = 100
    density_ratio: float = 0.7
    spatial_fraction: float = 0.03
    temporal_fraction: float = 0.10
    spatial_lower_quantile: float = 0.01
    spatial_upper_quantile: float = 0.99


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for training and simplification."""

    epochs: int = 50
    threshold: float = 0.5
    target_ratio: float | None = None
    compression_ratio: float | None = 0.2
    min_points_per_trajectory: int = 5
    max_train_points: int | None = None
    model_max_points: int | None = 300_000
    point_batch_size: int = 50_000
    importance_chunk_size: int = 200_000
    model_type: str = "baseline"
    turn_bias_weight: float = 0.1
    turn_score_method: str = "heading"


@dataclass(frozen=True)
class BaselineConfig:
    """Configuration for baseline execution."""

    dp_max_points: int = 200_000
    skip_baselines: bool = False


@dataclass(frozen=True)
class VisualizationConfig:
    """Configuration for visualization generation."""

    skip_visualizations: bool = False
    max_visualization_points: int = 200_000
    max_visualization_ships: int = 200
    max_points_per_ship_plot: int = 2_000


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment configuration object."""

    data: DataConfig = field(default_factory=DataConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    baselines: BaselineConfig = field(default_factory=BaselineConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


@dataclass(frozen=True)
class MethodMetrics:
    """Evaluation metrics for a simplified method."""

    query_error: float
    compression_ratio: float
    latency_ms: float


@dataclass(frozen=True)
class ModelSimplificationResult:
    """Simplification outputs for one trained model variant."""

    simplified_points: torch.Tensor
    retained_mask: torch.Tensor
    scores: torch.Tensor
    label: str


_DATA_ALIASES = {"n_points_per_ship": "n_points"}
_QUERY_ALIASES = {
    "spatial_fraction": "query_spatial_fraction",
    "temporal_fraction": "query_temporal_fraction",
    "spatial_lower_quantile": "query_spatial_lower_quantile",
    "spatial_upper_quantile": "query_spatial_upper_quantile",
}


def _section_kwargs(
    flat_args: Mapping[str, object],
    section_type: type[object],
    *,
    aliases: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Extract constructor kwargs for a config dataclass from flat arguments."""
    alias_map = aliases or {}
    section: dict[str, object] = {}
    for f in fields(section_type):
        source_key = alias_map.get(f.name, f.name)
        if source_key in flat_args:
            section[f.name] = flat_args[source_key]
    return section


def build_experiment_config(**flat_args: object) -> ExperimentConfig:
    """Build a structured config object from flat argument names."""
    return ExperimentConfig(
        data=DataConfig(**_section_kwargs(flat_args, DataConfig, aliases=_DATA_ALIASES)),
        query=QueryConfig(**_section_kwargs(flat_args, QueryConfig, aliases=_QUERY_ALIASES)),
        model=ModelConfig(**_section_kwargs(flat_args, ModelConfig)),
        baselines=BaselineConfig(**_section_kwargs(flat_args, BaselineConfig)),
        visualization=VisualizationConfig(**_section_kwargs(flat_args, VisualizationConfig)),
    )
