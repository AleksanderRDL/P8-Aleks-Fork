"""Typed query workload containers used across generation, training, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class TypedQueryWorkload:
    """Typed query workload container."""

    query_features: torch.Tensor
    typed_queries: list[dict[str, Any]]
    type_ids: torch.Tensor
    coverage_fraction: float | None = None
    covered_points: int | None = None
    total_points: int | None = None
    generation_diagnostics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize workload to a JSON-compatible dictionary."""
        return {
            "query_features": self.query_features.tolist(),
            "typed_queries": self.typed_queries,
            "type_ids": self.type_ids.tolist(),
            "coverage_fraction": self.coverage_fraction,
            "covered_points": self.covered_points,
            "total_points": self.total_points,
            "generation_diagnostics": self.generation_diagnostics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TypedQueryWorkload":
        """Deserialize workload from a JSON-compatible dictionary."""
        return cls(
            query_features=torch.tensor(data["query_features"], dtype=torch.float32),
            typed_queries=list(data["typed_queries"]),
            type_ids=torch.tensor(data["type_ids"], dtype=torch.long),
            coverage_fraction=data.get("coverage_fraction"),
            covered_points=data.get("covered_points"),
            total_points=data.get("total_points"),
            generation_diagnostics=data.get("generation_diagnostics"),
        )
