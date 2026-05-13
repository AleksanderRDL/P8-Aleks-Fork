"""Model checkpoint persistence for trained AIS-QDS models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from experiments.experiment_config import ExperimentConfig
from models.trajectory_qds_model import TrajectoryQDSModel
from models.turn_aware_qds_model import TurnAwareQDSModel
from training.scaler import FeatureScaler
from training.training_outputs import TrainingOutputs


@dataclass
class ModelArtifacts:
    """Model + scaler checkpoint payload."""

    model: TrajectoryQDSModel
    scaler: FeatureScaler
    config: ExperimentConfig
    epochs_trained: int = 0
    workload_type: str | None = None


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
    cfg = ExperimentConfig.from_dict(payload["config"])
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


def save_training_summary(path: str, outputs: TrainingOutputs) -> None:
    """Save training diagnostics history to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(outputs.history, f, indent=2)
