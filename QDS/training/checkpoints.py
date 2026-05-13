"""Model checkpoint persistence for trained AIS-QDS models."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch

from experiments.experiment_config import ExperimentConfig, QueryConfig
from models.trajectory_qds_model import TrajectoryQDSModel
from models.turn_aware_qds_model import TurnAwareQDSModel
from training.scaler import FeatureScaler


@dataclass
class ModelArtifacts:
    """Model + scaler checkpoint payload."""

    model: TrajectoryQDSModel
    scaler: FeatureScaler
    config: ExperimentConfig
    epochs_trained: int = 0
    workload_type: str | None = None


def _checkpoint_config_payload(raw_config: dict[str, Any]) -> dict[str, Any]:
    """Return a loadable config payload from a persisted checkpoint."""
    config = dict(raw_config)
    allowed_query_keys = {field.name for field in fields(QueryConfig)}
    query_config = dict(config.get("query") or {})
    config["query"] = {
        key: value
        for key, value in query_config.items()
        if key in allowed_query_keys
    }
    return config


def save_checkpoint(path: str, artifacts: ModelArtifacts) -> None:
    """Save model weights, scaler stats, and config to a checkpoint."""
    payload = {
        "model_state": artifacts.model.state_dict(),
        "point_dim": artifacts.model.point_dim,
        "query_dim": artifacts.model.query_dim,
        "embed_dim": artifacts.model.embed_dim,
        "query_chunk_size": artifacts.model.query_chunk_size,
        "model_type": artifacts.config.model.model_type,
        "scaler": artifacts.scaler.to_dict(),
        "config": artifacts.config.to_dict(),
        "epochs_trained": int(artifacts.epochs_trained),
        "workload_type": artifacts.workload_type or artifacts.config.query.workload,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str) -> ModelArtifacts:
    """Load model weights, scaler stats, and config from checkpoint."""
    payload = torch.load(path, map_location="cpu")
    cfg = ExperimentConfig.from_dict(_checkpoint_config_payload(payload["config"]))
    model_cls = TurnAwareQDSModel if payload["model_type"] == "turn_aware" else TrajectoryQDSModel
    model = model_cls(
        point_dim=int(payload["point_dim"]),
        query_dim=int(payload["query_dim"]),
        embed_dim=int(payload["embed_dim"]),
        query_chunk_size=int(payload["query_chunk_size"]),
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        type_embed_dim=cfg.model.type_embed_dim,
        dropout=cfg.model.dropout,
    )
    model.load_state_dict(payload["model_state"])
    model.eval()
    scaler = FeatureScaler.from_dict(payload["scaler"])
    return ModelArtifacts(
        model=model,
        scaler=scaler,
        config=cfg,
        epochs_trained=int(payload.get("epochs_trained", 0)),
        workload_type=str(payload.get("workload_type") or cfg.query.workload),
    )
