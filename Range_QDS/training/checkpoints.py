"""Model checkpoint persistence for trained AIS-QDS models."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch

from experiments.experiment_config import BaselineConfig, DataConfig, ExperimentConfig, ModelConfig, QueryConfig
from models.historical_prior_qds_model import HistoricalPriorRangeQDSModel, HistoricalPriorStudentRangeQDSModel
from models.trajectory_qds_model import TrajectoryQDSModel
from models.turn_aware_qds_model import TurnAwareQDSModel
from models.workload_blind_range_v2 import WorkloadBlindRangeV2Model
from models.workload_blind_qds_model import SegmentContextRangeQDSModel, WorkloadBlindRangeQDSModel
from training.scaler import FeatureScaler
from training.model_features import (
    HISTORICAL_PRIOR_MODEL_TYPES,
    NONPARAMETRIC_HISTORICAL_PRIOR_MODEL_TYPES,
    SUPPORTED_MODEL_TYPES,
    is_workload_blind_model_type,
)


@dataclass
class ModelArtifacts:
    """Model + scaler checkpoint payload."""

    model: torch.nn.Module
    scaler: FeatureScaler
    config: ExperimentConfig
    epochs_trained: int = 0
    workload_type: str | None = None
    query_prior_field: dict[str, Any] | None = None


def _filter_config_section(raw_section: Any, config_cls: type) -> dict[str, Any]:
    """Drop stale checkpoint keys for one dataclass-backed config section."""
    allowed_keys = {field.name for field in fields(config_cls)}
    section = dict(raw_section or {})
    return {
        key: value
        for key, value in section.items()
        if key in allowed_keys
    }


def _checkpoint_config_payload(raw_config: dict[str, Any]) -> dict[str, Any]:
    """Return a loadable config payload from a persisted checkpoint."""
    config = dict(raw_config)
    return {
        "data": _filter_config_section(config.get("data"), DataConfig),
        "query": _filter_config_section(config.get("query"), QueryConfig),
        "model": _filter_config_section(config.get("model"), ModelConfig),
        "baselines": _filter_config_section(config.get("baselines"), BaselineConfig),
    }


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
        "query_prior_field": artifacts.query_prior_field,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str) -> ModelArtifacts:
    """Load model weights, scaler stats, and config from checkpoint."""
    payload = torch.load(path, map_location="cpu")
    cfg = ExperimentConfig.from_dict(_checkpoint_config_payload(payload["config"]))
    model_type = str(payload["model_type"])
    if model_type not in SUPPORTED_MODEL_TYPES:
        choices = ", ".join(SUPPORTED_MODEL_TYPES)
        raise ValueError(f"Unsupported checkpoint model_type={model_type!r}; choices: {choices}.")
    model_state = payload["model_state"]
    if model_type in NONPARAMETRIC_HISTORICAL_PRIOR_MODEL_TYPES:
        model_cls = HistoricalPriorRangeQDSModel
        prior = model_state.get("historical_targets")
        prior_feature_count = int(prior.shape[0]) if isinstance(prior, torch.Tensor) else 0
        if "historical_source_ids" not in model_state:
            model_state["historical_source_ids"] = torch.zeros((prior_feature_count,), dtype=torch.long)
    elif model_type == "historical_prior_student":
        model_cls = HistoricalPriorStudentRangeQDSModel
        prior = model_state.get("prior.historical_targets")
        prior_feature_count = int(prior.shape[0]) if isinstance(prior, torch.Tensor) else 0
        if "prior.historical_source_ids" not in model_state:
            model_state["prior.historical_source_ids"] = torch.zeros((prior_feature_count,), dtype=torch.long)
    elif model_type == "segment_context_range":
        model_cls = SegmentContextRangeQDSModel
        prior_feature_count = 0
    elif model_type == "workload_blind_range_v2":
        model_cls = WorkloadBlindRangeV2Model
        prior_feature_count = 0
    elif is_workload_blind_model_type(model_type):
        model_cls = WorkloadBlindRangeQDSModel
        prior_feature_count = 0
    elif model_type == "turn_aware":
        model_cls = TurnAwareQDSModel
        prior_feature_count = 0
    else:
        model_cls = TrajectoryQDSModel
        prior_feature_count = 0
    model_kwargs = {
        "point_dim": int(payload["point_dim"]),
        "query_dim": int(payload["query_dim"]),
        "embed_dim": int(payload["embed_dim"]),
        "query_chunk_size": int(payload["query_chunk_size"]),
        "num_heads": cfg.model.num_heads,
        "num_layers": cfg.model.num_layers,
        "type_embed_dim": cfg.model.type_embed_dim,
        "dropout": cfg.model.dropout,
    }
    if model_type in HISTORICAL_PRIOR_MODEL_TYPES:
        model_kwargs["historical_prior_k"] = int(getattr(cfg.model, "historical_prior_k", 32))
        model_kwargs["historical_prior_clock_weight"] = float(
            getattr(cfg.model, "historical_prior_clock_weight", 0.0)
        )
        model_kwargs["historical_prior_mmsi_weight"] = float(
            getattr(cfg.model, "historical_prior_mmsi_weight", 1.0)
        )
        model_kwargs["historical_prior_density_weight"] = float(
            getattr(cfg.model, "historical_prior_density_weight", 1.0)
        )
        model_kwargs["historical_prior_min_target"] = float(
            getattr(cfg.model, "historical_prior_min_target", 0.0)
        )
        model_kwargs["historical_prior_source_aggregation"] = str(
            getattr(cfg.model, "historical_prior_source_aggregation", "none")
        )
        model_kwargs["prior_feature_count"] = prior_feature_count
    model = model_cls(**model_kwargs)
    query_prior_field = payload.get("query_prior_field")
    if query_prior_field is not None:
        setattr(model, "query_prior_field", query_prior_field)
    if model_type == "workload_blind_range_v2":
        load_result = model.load_state_dict(model_state, strict=False)
        allowed_missing = {
            name
            for name in model.state_dict()
            if name == "prior_feature_scale" or name.startswith("prior_feature_encoder.")
        }
        missing = set(load_result.missing_keys)
        unexpected = set(load_result.unexpected_keys)
        if missing - allowed_missing or unexpected:
            raise RuntimeError(
                "Incompatible workload_blind_range_v2 checkpoint state: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
    else:
        model.load_state_dict(model_state)
    model.eval()
    scaler = FeatureScaler.from_dict(payload["scaler"])
    return ModelArtifacts(
        model=model,
        scaler=scaler,
        config=cfg,
        epochs_trained=int(payload.get("epochs_trained", 0)),
        workload_type=str(payload.get("workload_type") or cfg.query.workload),
        query_prior_field=query_prior_field,
    )
