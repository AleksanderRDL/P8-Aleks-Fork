"""Typed query schemas and feature padding helpers. See queries/README.md for details."""

from __future__ import annotations

from typing import Any

import torch

QUERY_TYPE_ID_RANGE = 0
NUM_QUERY_TYPES = 1

QUERY_NAME_TO_ID = {
    "range": QUERY_TYPE_ID_RANGE,
}
ID_TO_QUERY_NAME = {v: k for k, v in QUERY_NAME_TO_ID.items()}


def normalize_pure_workload_map(workload_map: dict[str, float]) -> dict[str, float]:
    """Normalize one pure workload map to ``{type: 1.0}``."""
    filtered = {k.lower(): float(v) for k, v in workload_map.items() if float(v) > 0.0}
    total = sum(filtered.values())
    if total <= 0.0:
        raise ValueError("Workload map must contain at least one positive weight.")
    for name in filtered:
        if name not in QUERY_NAME_TO_ID:
            raise ValueError(f"Only range workloads are supported; got query type: {name}")
    if len(filtered) != 1:
        raise ValueError(f"Expected exactly one active workload type; got {workload_map}.")
    return {k: v / total for k, v in filtered.items()}


def single_workload_type(workload_map: dict[str, float]) -> str:
    """Return the one active workload type, rejecting mixed workloads."""
    normalized = normalize_pure_workload_map(workload_map)
    return next(iter(normalized))


def pad_query_features(typed_queries: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert heterogeneous typed queries to padded numeric features + type IDs. See queries/README.md for details."""
    feature_dim = 12
    feats = torch.zeros((len(typed_queries), feature_dim), dtype=torch.float32)
    type_ids = torch.zeros((len(typed_queries),), dtype=torch.long)

    for i, query in enumerate(typed_queries):
        qtype = str(query["type"]).lower()
        params = query["params"]
        if qtype == "range":
            type_ids[i] = QUERY_TYPE_ID_RANGE
            feats[i, :6] = torch.tensor(
                [
                    params["lat_min"],
                    params["lat_max"],
                    params["lon_min"],
                    params["lon_max"],
                    params["t_start"],
                    params["t_end"],
                ],
                dtype=torch.float32,
            )
        else:
            raise ValueError(f"Only range queries are supported; got query type: {qtype}")
    return feats, type_ids
