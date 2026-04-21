"""Experiment configuration dataclasses for AIS-QDS v2. See src/experiments/README.md for details."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch

LCG_MULTIPLIER = 6364136223846793005


@dataclass
class DataConfig:
    """Data loading and splitting configuration. See src/data/README.md for details."""

    n_ships: int = 24
    n_points_per_ship: int = 200
    csv_path: str | None = None
    seed: int = 42
    train_fraction: float = 0.70
    val_fraction: float = 0.15

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a dictionary. See src/experiments/README.md for details."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataConfig":
        """Deserialize config from a dictionary. See src/experiments/README.md for details."""
        return cls(**data)


@dataclass
class QueryConfig:
    """Query generation and workload-mix configuration. See src/queries/README.md for details."""

    n_queries: int = 128
    workload: str = "mixed"
    train_workload_mix: dict[str, float] = field(
        default_factory=lambda: {"range": 0.5, "knn": 0.5}
    )
    eval_workload_mix: dict[str, float] = field(
        default_factory=lambda: {"similarity": 0.5, "clustering": 0.5}
    )
    similarity_top_k: int = 5

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a dictionary. See src/experiments/README.md for details."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QueryConfig":
        """Deserialize config from a dictionary. See src/experiments/README.md for details."""
        return cls(**data)


@dataclass
class ModelConfig:
    """Model architecture and training behavior config. See src/models/README.md for details."""

    embed_dim: int = 64
    num_heads: int = 4
    num_layers: int = 3
    type_embed_dim: int = 16
    query_chunk_size: int = 128
    dropout: float = 0.1
    window_length: int = 512
    window_stride: int = 256
    epochs: int = 6
    lr: float = 2e-3
    compression_ratio: float = 0.2
    model_type: str = "baseline"
    rank_margin: float = 0.05
    ranking_pairs_per_type: int = 96
    ranking_top_quantile: float = 0.80
    l2_score_weight: float = 1e-4
    dirichlet_alpha: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a dictionary. See src/experiments/README.md for details."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        """Deserialize config from a dictionary. See src/experiments/README.md for details."""
        return cls(**data)


@dataclass
class BaselineConfig:
    """Baseline methods configuration. See src/evaluation/README.md for details."""

    include_oracle: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a dictionary. See src/experiments/README.md for details."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaselineConfig":
        """Deserialize config from a dictionary. See src/experiments/README.md for details."""
        return cls(**data)


@dataclass
class VisualizationConfig:
    """Visualization config. See src/visualization/README.md for details."""

    enabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a dictionary. See src/experiments/README.md for details."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VisualizationConfig":
        """Deserialize config from a dictionary. See src/experiments/README.md for details."""
        return cls(**data)


@dataclass
class TypedQueryWorkload:
    """Typed query workload container. See src/queries/README.md for details."""

    query_features: torch.Tensor
    typed_queries: list[dict[str, Any]]
    type_ids: torch.Tensor

    def to_dict(self) -> dict[str, Any]:
        """Serialize workload to a dictionary. See src/queries/README.md for details."""
        return {
            "query_features": self.query_features.tolist(),
            "typed_queries": self.typed_queries,
            "type_ids": self.type_ids.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TypedQueryWorkload":
        """Deserialize workload from a dictionary. See src/queries/README.md for details."""
        return cls(
            query_features=torch.tensor(data["query_features"], dtype=torch.float32),
            typed_queries=list(data["typed_queries"]),
            type_ids=torch.tensor(data["type_ids"], dtype=torch.long),
        )


@dataclass
class ExperimentConfig:
    """Top-level experiment config container. See src/experiments/README.md for details."""

    data: DataConfig = field(default_factory=DataConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    baselines: BaselineConfig = field(default_factory=BaselineConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a dictionary. See src/experiments/README.md for details."""
        return {
            "data": self.data.to_dict(),
            "query": self.query.to_dict(),
            "model": self.model.to_dict(),
            "baselines": self.baselines.to_dict(),
            "visualization": self.visualization.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        """Deserialize config from a dictionary. See src/experiments/README.md for details."""
        return cls(
            data=DataConfig.from_dict(data["data"]),
            query=QueryConfig.from_dict(data["query"]),
            model=ModelConfig.from_dict(data["model"]),
            baselines=BaselineConfig.from_dict(data["baselines"]),
            visualization=VisualizationConfig.from_dict(data["visualization"]),
        )


@dataclass
class SeedBundle:
    """Derived deterministic sub-seeds for experiment stages. See src/experiments/README.md for details."""

    split_seed: int
    train_query_seed: int
    eval_query_seed: int
    torch_seed: int


def build_experiment_config(
    n_ships: int = 24,
    n_points: int = 200,
    n_queries: int = 128,
    epochs: int = 6,
    compression_ratio: float = 0.2,
    csv_path: str | None = None,
    model_type: str = "baseline",
    workload: str = "mixed",
    train_workload_mix: dict[str, float] | None = None,
    eval_workload_mix: dict[str, float] | None = None,
    seed: int = 42,
    **_ignored_kwargs: Any,
) -> ExperimentConfig:
    """Build a structured experiment config from flat arguments. See src/experiments/README.md for details."""
    return ExperimentConfig(
        data=DataConfig(
            n_ships=n_ships,
            n_points_per_ship=n_points,
            csv_path=csv_path,
            seed=seed,
        ),
        query=QueryConfig(
            n_queries=n_queries,
            workload=workload,
            train_workload_mix=train_workload_mix or {"range": 0.5, "knn": 0.5},
            eval_workload_mix=eval_workload_mix or {"similarity": 0.5, "clustering": 0.5},
        ),
        model=ModelConfig(
            epochs=epochs,
            compression_ratio=compression_ratio,
            model_type=model_type,
        ),
    )


def derive_seed_bundle(master_seed: int) -> SeedBundle:
    """Derive deterministic sub-seeds from a master seed. See src/experiments/README.md for details."""
    return SeedBundle(
        split_seed=(master_seed * LCG_MULTIPLIER + 1) & 0xFFFF_FFFF,
        train_query_seed=(master_seed * LCG_MULTIPLIER + 3) & 0xFFFF_FFFF,
        eval_query_seed=(master_seed * LCG_MULTIPLIER + 5) & 0xFFFF_FFFF,
        torch_seed=(master_seed * LCG_MULTIPLIER + 7) & 0xFFFF_FFFF,
    )
