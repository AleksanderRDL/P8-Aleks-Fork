"""Training setup helpers shared by the model loop and validation scoring."""

from __future__ import annotations

import torch

from queries.query_types import NUM_QUERY_TYPES, normalize_pure_workload_map
from queries.workload import TypedQueryWorkload


def _model_state_on_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Copy a model state dict to CPU tensors for best-epoch restoration."""
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _workload_map_tensor(workload_map: dict[str, float], device: torch.device) -> torch.Tensor:
    """Return normalized pure-workload weights in query-type ID order."""
    normalized = normalize_pure_workload_map(workload_map)
    return torch.tensor(
        [float(normalized.get("range", 0.0))],
        dtype=torch.float32,
        device=device,
    )


def _query_frequency_workload_map(workload: TypedQueryWorkload) -> dict[str, float]:
    """Infer type weights from a workload when no explicit training workload map is provided."""
    counts = torch.bincount(workload.type_ids.detach().cpu(), minlength=NUM_QUERY_TYPES).float()
    return {"range": float(counts[0].item())}


def _single_active_type_id(type_weights: torch.Tensor) -> int:
    """Return the one active query type for pure-workload training."""
    active = torch.where(type_weights.detach().cpu() > 0.0)[0]
    if int(active.numel()) != 1:
        raise ValueError("Pure-workload training requires exactly one active query type.")
    return int(active[0].item())


def _pure_query_type_id(type_ids: torch.Tensor) -> int:
    """Return the only query type id in a pure workload."""
    unique_ids = torch.unique(type_ids.detach().cpu())
    if int(unique_ids.numel()) != 1:
        raise ValueError("Pure-workload training/evaluation requires exactly one query type id.")
    return int(unique_ids[0].item())
