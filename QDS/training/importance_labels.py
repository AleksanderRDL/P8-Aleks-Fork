"""Typed F1-contribution label construction. See training/README.md for details."""

from __future__ import annotations

from typing import Any

import torch

from data.trajectory_index import trajectory_ids_for_points
from evaluation.range_usefulness import RANGE_USEFULNESS_WEIGHTS
from queries.range_geometry import (
    points_in_range_box,
    segment_box_bracket_mask,
)
from queries.query_types import QUERY_NAME_TO_ID, QUERY_TYPE_ID_RANGE, NUM_QUERY_TYPES

RANGE_LABEL_MODES = ("point_f1", "usefulness", "usefulness_balanced", "usefulness_ship_balanced")
RANGE_USEFULNESS_LABEL_WEIGHTS = dict(RANGE_USEFULNESS_WEIGHTS)
RANGE_USEFULNESS_LABEL_COMPONENTS = tuple(RANGE_USEFULNESS_LABEL_WEIGHTS.keys())


def _label_targets(
    labels: torch.Tensor,
    component_labels: dict[str, torch.Tensor] | None,
    component_name: str,
) -> tuple[torch.Tensor, ...]:
    """Return main labels plus the optional component-specific label tensor."""
    if component_labels is None:
        return (labels,)
    return (labels, component_labels[component_name])


def _set_query_singleton_gain(original_ids: set[int]) -> float:
    """F1 gained when one true-positive trajectory ID is recovered from an empty answer."""
    if not original_ids:
        return 0.0
    return float(2.0 / (len(original_ids) + 1.0))


def _add_distributed_hit_label(labels: torch.Tensor, support: torch.Tensor, type_idx: int, gain: float) -> None:
    """Distribute one trajectory-hit gain over its interchangeable support points."""
    support_count = int(support.sum().item())
    if support_count <= 0:
        return
    labels[support, type_idx] += float(gain) / float(support_count)


def _add_weighted_index_label(
    labels: torch.Tensor,
    indices: torch.Tensor,
    type_idx: int,
    gain: float,
    weights: torch.Tensor,
) -> None:
    """Distribute label gain over explicit indices using compact weights."""
    if indices.numel() == 0:
        return
    w = weights.to(device=labels.device, dtype=torch.float32).clamp(min=0.0)
    total = float(w.sum().item())
    if total <= 0.0:
        labels[indices, type_idx] += float(gain) / float(indices.numel())
        return
    labels[indices, type_idx] += float(gain) * (w / total)


def _range_boundary_weights(
    points: torch.Tensor,
    box_mask: torch.Tensor,
    boundaries: list[tuple[int, int]],
    boundary_prior_weight: float,
) -> torch.Tensor:
    """Optional boundary-crossing prior for range labels.

    ``boundary_prior_weight=0`` keeps pure point-F1 labels. Positive values
    boost in-box points whose previous or next trajectory neighbour is outside
    the query box, then mean-normalize over the in-box support so total query
    label mass remains the same.
    """
    weights = torch.zeros(points.shape[0], dtype=torch.float32, device=points.device)
    in_box_idx = torch.where(box_mask)[0]
    if in_box_idx.numel() == 0:
        return weights
    boost = max(0.0, float(boundary_prior_weight))
    if boost <= 0.0:
        weights[in_box_idx] = 1.0
        return weights

    boundary_full = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
    for start, end in boundaries:
        if end <= start:
            continue
        traj_in = box_mask[start:end]
        if not bool(traj_in.any().item()):
            continue
        prev_out = torch.zeros_like(traj_in)
        prev_out[1:] = traj_in[1:] & ~traj_in[:-1]
        prev_out[0] = traj_in[0]
        next_out = torch.zeros_like(traj_in)
        next_out[:-1] = traj_in[:-1] & ~traj_in[1:]
        next_out[-1] = traj_in[-1]
        boundary_full[start:end] = prev_out | next_out

    boundary_in_box = boundary_full[in_box_idx].float()
    raw = 1.0 + boost * boundary_in_box
    mean_raw = float(raw.mean().item())
    weights[in_box_idx] = raw / mean_raw if mean_raw > 1e-12 else 1.0
    return weights


def _range_entry_exit_mask(
    box_mask: torch.Tensor,
    boundaries: list[tuple[int, int]],
) -> torch.Tensor:
    """Return sampled in-box range entry/exit points."""
    boundary_full = torch.zeros_like(box_mask, dtype=torch.bool)
    for start, end in boundaries:
        if end <= start:
            continue
        traj_in = box_mask[start:end]
        if not bool(traj_in.any().item()):
            continue
        enters = torch.zeros_like(traj_in)
        enters[1:] = traj_in[1:] & ~traj_in[:-1]
        enters[0] = traj_in[0]
        exits = torch.zeros_like(traj_in)
        exits[:-1] = traj_in[:-1] & ~traj_in[1:]
        exits[-1] = traj_in[-1]
        boundary_full[start:end] = enters | exits
    return boundary_full


def _local_shape_weights(points: torch.Tensor, global_indices: torch.Tensor) -> torch.Tensor:
    """Return range-local shape weights for one trajectory slice."""
    count = int(global_indices.numel())
    weights = torch.ones((count,), dtype=torch.float32, device=points.device)
    if count >= 3:
        coords = points[global_indices, 1:3].float()
        before = torch.linalg.vector_norm(coords[1:-1] - coords[:-2], dim=1)
        after = torch.linalg.vector_norm(coords[2:] - coords[1:-1], dim=1)
        shortcut = torch.linalg.vector_norm(coords[2:] - coords[:-2], dim=1)
        curvature = torch.clamp(before + after - shortcut, min=0.0)
        mean_curvature = float(curvature.mean().item())
        if mean_curvature > 1e-12:
            weights[1:-1] = weights[1:-1] + curvature / mean_curvature
    if points.shape[1] >= 8:
        weights = weights + points[global_indices, 7].float().clamp(min=0.0)
    return weights


def _local_turn_weights(points: torch.Tensor, global_indices: torch.Tensor) -> torch.Tensor:
    """Return range-local route-change weights for one trajectory slice."""
    count = int(global_indices.numel())
    weights = torch.zeros((count,), dtype=torch.float32, device=points.device)
    if count >= 3:
        coords = points[global_indices, 1:3].float()
        before = torch.linalg.vector_norm(coords[1:-1] - coords[:-2], dim=1)
        after = torch.linalg.vector_norm(coords[2:] - coords[1:-1], dim=1)
        shortcut = torch.linalg.vector_norm(coords[2:] - coords[:-2], dim=1)
        weights[1:-1] = torch.clamp(before + after - shortcut, min=0.0)
    if points.shape[1] >= 8:
        weights = torch.maximum(weights, points[global_indices, 7].float().clamp(min=0.0))
    return weights


def _local_gap_weights(count: int, device: torch.device) -> torch.Tensor:
    """Return interior-biased weights for range gap-coverage labels."""
    weights = torch.ones((int(count),), dtype=torch.float32, device=device)
    if count >= 3:
        positions = torch.linspace(0.0, 1.0, int(count), dtype=torch.float32, device=device)
        interior = torch.minimum(positions, 1.0 - positions)
        max_interior = float(interior.max().item())
        if max_interior > 1e-12:
            weights = weights + interior / max_interior
    return weights


def _add_range_point_f1_labels(
    labels: torch.Tensor,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    box_support: torch.Tensor,
    type_idx: int,
    range_boundary_prior_weight: float,
    component_weight: float = 1.0,
) -> None:
    """Add point-F1 singleton gain for every in-box point."""
    hit_count = int(box_support.sum().item())
    if hit_count <= 0:
        return
    base_gain = float(2.0 / (hit_count + 1.0))
    boundary_weights = _range_boundary_weights(
        points,
        box_support,
        boundaries,
        range_boundary_prior_weight,
    )
    labels[box_support, type_idx] += float(component_weight) * base_gain * boundary_weights[box_support]


def _support_trajectory_ids(
    point_trajectory_ids: torch.Tensor,
    support: torch.Tensor,
) -> set[int]:
    """Return nonnegative trajectory IDs represented by a boolean point support."""
    return {
        int(value)
        for value in torch.unique(point_trajectory_ids[support]).detach().cpu().tolist()
        if int(value) >= 0
    }


def _add_ship_balanced_support_labels(
    labels: torch.Tensor,
    support: torch.Tensor,
    point_trajectory_ids: torch.Tensor,
    trajectory_ids: set[int],
    type_idx: int,
    component_mass: float,
    *,
    points: torch.Tensor | None = None,
    boundaries: list[tuple[int, int]] | None = None,
    range_boundary_prior_weight: float = 0.0,
) -> None:
    """Distribute one component's support labels with query-hit ships normalized.

    Dense trajectory hits otherwise dominate point, entry/exit, and crossing
    labels. This helper keeps the component training-only and query-derived,
    but makes each represented ship compete on comparable mass.
    """
    if not trajectory_ids:
        return
    mass_per_ship = float(component_mass) / float(len(trajectory_ids))
    for trajectory_id in sorted(trajectory_ids):
        trajectory_support = support & (point_trajectory_ids == int(trajectory_id))
        support_count = int(trajectory_support.sum().item())
        if support_count <= 0:
            continue
        gain = float(2.0 / (support_count + 1.0))
        if points is not None and boundaries is not None and range_boundary_prior_weight > 0.0:
            support_weights = _range_boundary_weights(
                points=points,
                box_mask=trajectory_support,
                boundaries=boundaries,
                boundary_prior_weight=range_boundary_prior_weight,
            )
            labels[trajectory_support, type_idx] += (
                mass_per_ship * gain * support_weights[trajectory_support]
            )
        else:
            labels[trajectory_support, type_idx] += mass_per_ship * gain


def _add_range_usefulness_labels(
    labels: torch.Tensor,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    box_support: torch.Tensor,
    params: dict[str, float],
    point_trajectory_ids: torch.Tensor,
    type_idx: int,
    range_boundary_prior_weight: float,
    component_labels: dict[str, torch.Tensor] | None = None,
    ship_balanced_support: bool = False,
) -> None:
    """Add a local proxy for the range usefulness audit components."""
    hit_count = int(box_support.sum().item())
    crossing_support = segment_box_bracket_mask(points, boundaries, params)
    crossing_count = int(crossing_support.sum().item())
    if hit_count <= 0 and crossing_count <= 0:
        return

    weights = RANGE_USEFULNESS_LABEL_WEIGHTS
    hit_ids = _support_trajectory_ids(point_trajectory_ids, box_support)
    if hit_count > 0:
        if ship_balanced_support and hit_ids:
            for target in _label_targets(labels, component_labels, "range_point_f1"):
                _add_ship_balanced_support_labels(
                    labels=target,
                    support=box_support,
                    point_trajectory_ids=point_trajectory_ids,
                    trajectory_ids=hit_ids,
                    type_idx=type_idx,
                    component_mass=float(weights["range_point_f1"]),
                    points=points,
                    boundaries=boundaries,
                    range_boundary_prior_weight=range_boundary_prior_weight,
                )
        else:
            for target in _label_targets(labels, component_labels, "range_point_f1"):
                _add_range_point_f1_labels(
                    labels=target,
                    points=points,
                    boundaries=boundaries,
                    box_support=box_support,
                    type_idx=type_idx,
                    range_boundary_prior_weight=range_boundary_prior_weight,
                    component_weight=float(weights["range_point_f1"]),
                )

    if crossing_count > 0:
        if ship_balanced_support:
            crossing_ids = _support_trajectory_ids(point_trajectory_ids, crossing_support)
            for target in _label_targets(labels, component_labels, "range_crossing_f1"):
                _add_ship_balanced_support_labels(
                    labels=target,
                    support=crossing_support,
                    point_trajectory_ids=point_trajectory_ids,
                    trajectory_ids=crossing_ids,
                    type_idx=type_idx,
                    component_mass=float(weights["range_crossing_f1"]),
                )
        else:
            crossing_gain = float(2.0 / (crossing_count + 1.0))
            for target in _label_targets(labels, component_labels, "range_crossing_f1"):
                target[crossing_support, type_idx] += float(weights["range_crossing_f1"]) * crossing_gain

    if not hit_ids:
        return
    ship_count = len(hit_ids)
    ship_gain = _set_query_singleton_gain(hit_ids)
    ship_coverage_mass_per_ship = float(weights["range_ship_coverage"]) / float(ship_count)
    temporal_mass_per_ship = float(weights["range_temporal_coverage"]) / float(ship_count)
    gap_mass_per_ship = float(weights["range_gap_coverage"]) / float(ship_count)
    turn_mass_per_ship = float(weights["range_turn_coverage"]) / float(ship_count)
    shape_mass = float(weights["range_shape_score"]) * float(ship_gain)

    boundary_support = _range_entry_exit_mask(box_support, boundaries)
    boundary_count = int(boundary_support.sum().item())
    if boundary_count > 0:
        if ship_balanced_support:
            boundary_ids = _support_trajectory_ids(point_trajectory_ids, boundary_support)
            for target in _label_targets(labels, component_labels, "range_entry_exit_f1"):
                _add_ship_balanced_support_labels(
                    labels=target,
                    support=boundary_support,
                    point_trajectory_ids=point_trajectory_ids,
                    trajectory_ids=boundary_ids,
                    type_idx=type_idx,
                    component_mass=float(weights["range_entry_exit_f1"]),
                )
        else:
            boundary_gain = float(2.0 / (boundary_count + 1.0))
            for target in _label_targets(labels, component_labels, "range_entry_exit_f1"):
                target[boundary_support, type_idx] += float(weights["range_entry_exit_f1"]) * boundary_gain

    for trajectory_id in sorted(hit_ids):
        if trajectory_id >= len(boundaries):
            continue
        start, end = boundaries[trajectory_id]
        if end <= start:
            continue
        trajectory_support = box_support & (point_trajectory_ids == int(trajectory_id))
        if not bool(trajectory_support.any().item()):
            continue

        for target in _label_targets(labels, component_labels, "range_ship_f1"):
            _add_distributed_hit_label(
                target,
                trajectory_support,
                type_idx,
                float(weights["range_ship_f1"]) * float(ship_gain),
            )

        support_count = int(trajectory_support.sum().item())
        if support_count > 0:
            ship_coverage_gain = float(2.0 / (support_count + 1.0))
            for target in _label_targets(labels, component_labels, "range_ship_coverage"):
                target[trajectory_support, type_idx] += ship_coverage_mass_per_ship * ship_coverage_gain

        in_offsets = torch.where(box_support[start:end])[0]
        if in_offsets.numel() == 0:
            continue
        if in_offsets.numel() == 1:
            for target in _label_targets(labels, component_labels, "range_temporal_coverage"):
                target[start + in_offsets[0], type_idx] += temporal_mass_per_ship
        else:
            for target in _label_targets(labels, component_labels, "range_temporal_coverage"):
                target[start + in_offsets[0], type_idx] += 0.5 * temporal_mass_per_ship
                target[start + in_offsets[-1], type_idx] += 0.5 * temporal_mass_per_ship

        global_indices = start + in_offsets
        turn_weights = _local_turn_weights(points, global_indices)
        if float(turn_weights.sum().item()) > 1e-12:
            for target in _label_targets(labels, component_labels, "range_turn_coverage"):
                _add_weighted_index_label(
                    target,
                    global_indices,
                    type_idx,
                    turn_mass_per_ship,
                    turn_weights,
                )
        for target in _label_targets(labels, component_labels, "range_gap_coverage"):
            _add_weighted_index_label(
                target,
                global_indices,
                type_idx,
                gap_mass_per_ship,
                _local_gap_weights(int(global_indices.numel()), labels.device),
            )
        for target in _label_targets(labels, component_labels, "range_shape_score"):
            _add_weighted_index_label(
                target,
                global_indices,
                type_idx,
                shape_mass,
                _local_shape_weights(points, global_indices),
            )


def _balance_range_component_label_mass(
    labels: torch.Tensor,
    component_labels: dict[str, torch.Tensor],
    type_idx: int,
) -> None:
    """Redistribute usefulness component mass to match the RangeUseful audit weights."""
    masses: dict[str, float] = {}
    total_mass = 0.0
    available_weight = 0.0
    for component_name in RANGE_USEFULNESS_LABEL_COMPONENTS:
        values = component_labels[component_name][:, type_idx].clamp(min=0.0)
        mass = float(values.sum().item())
        masses[component_name] = mass
        if mass > 1e-12:
            total_mass += mass
            available_weight += float(RANGE_USEFULNESS_LABEL_WEIGHTS[component_name])

    if total_mass <= 1e-12 or available_weight <= 1e-12:
        labels[:, type_idx] = torch.clamp(labels[:, type_idx], 0.0, 1.0)
        return

    balanced = torch.zeros_like(labels[:, type_idx])
    for component_name in RANGE_USEFULNESS_LABEL_COMPONENTS:
        values = component_labels[component_name][:, type_idx].clamp(min=0.0)
        mass = masses[component_name]
        if mass <= 1e-12:
            component_labels[component_name][:, type_idx] = 0.0
            continue
        target_mass = total_mass * float(RANGE_USEFULNESS_LABEL_WEIGHTS[component_name]) / available_weight
        scaled = values * float(target_mass / mass)
        component_labels[component_name][:, type_idx] = scaled
        balanced += scaled
    labels[:, type_idx] = torch.clamp(balanced, 0.0, 1.0)


def _compute_typed_importance_labels(
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, Any]],
    range_boundary_prior_weight: float = 0.0,
    range_label_mode: str = "point_f1",
    return_range_components: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
    """Compute per-point per-type labels as expected query-F1 contribution."""
    range_label_mode = str(range_label_mode).lower()
    if range_label_mode not in RANGE_LABEL_MODES:
        raise ValueError(f"range_label_mode must be one of {RANGE_LABEL_MODES}.")

    n = points.shape[0]
    labels = torch.zeros((n, NUM_QUERY_TYPES), dtype=torch.float32, device=points.device)
    needs_component_labels = return_range_components or range_label_mode == "usefulness_balanced"
    component_labels = (
        {
            component_name: torch.zeros_like(labels)
            for component_name in RANGE_USEFULNESS_LABEL_COMPONENTS
        }
        if needs_component_labels
        and range_label_mode in {"usefulness", "usefulness_balanced", "usefulness_ship_balanced"}
        else None
    )
    labelled_mask = torch.zeros((n, NUM_QUERY_TYPES), dtype=torch.bool, device=points.device)
    query_counts = torch.zeros((NUM_QUERY_TYPES,), dtype=torch.float32, device=points.device)
    point_trajectory_ids = trajectory_ids_for_points(n, boundaries, points.device)

    for q in typed_queries:
        qtype = str(q["type"]).lower()
        if qtype != "range":
            raise ValueError(f"Only range queries are supported for labels; got query type: {qtype}")
        t_idx = QUERY_NAME_TO_ID[qtype]
        params = q["params"]
        query_counts[t_idx] += 1.0

        box_support = points_in_range_box(points, params)
        if range_label_mode in {"usefulness", "usefulness_balanced", "usefulness_ship_balanced"}:
            _add_range_usefulness_labels(
                labels=labels,
                points=points,
                boundaries=boundaries,
                box_support=box_support,
                params=params,
                point_trajectory_ids=point_trajectory_ids,
                type_idx=t_idx,
                range_boundary_prior_weight=range_boundary_prior_weight,
                component_labels=component_labels,
                ship_balanced_support=range_label_mode == "usefulness_ship_balanced",
            )
        else:
            _add_range_point_f1_labels(
                labels=labels,
                points=points,
                boundaries=boundaries,
                box_support=box_support,
                type_idx=t_idx,
                range_boundary_prior_weight=range_boundary_prior_weight,
            )

    for type_idx in range(NUM_QUERY_TYPES):
        count = float(query_counts[type_idx].item())
        if count > 0.0:
            labels[:, type_idx] = labels[:, type_idx] / count
            if component_labels is not None:
                for component_values in component_labels.values():
                    component_values[:, type_idx] = component_values[:, type_idx] / count
            if (
                range_label_mode == "usefulness_balanced"
                and component_labels is not None
                and type_idx == QUERY_TYPE_ID_RANGE
            ):
                _balance_range_component_label_mass(labels, component_labels, type_idx)
            else:
                labels[:, type_idx] = torch.clamp(labels[:, type_idx], 0.0, 1.0)
            labelled_mask[:, type_idx] = True

    return labels, labelled_mask, component_labels


def compute_typed_importance_labels(
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, Any]],
    range_boundary_prior_weight: float = 0.0,
    range_label_mode: str = "point_f1",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-point per-type labels as expected query-F1 contribution."""
    labels, labelled_mask, _component_labels = _compute_typed_importance_labels(
        points=points,
        boundaries=boundaries,
        typed_queries=typed_queries,
        range_boundary_prior_weight=range_boundary_prior_weight,
        range_label_mode=range_label_mode,
        return_range_components=False,
    )
    return labels, labelled_mask


def compute_typed_importance_labels_with_range_components(
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, Any]],
    range_boundary_prior_weight: float = 0.0,
    range_label_mode: str = "usefulness",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Compute usefulness labels plus per-component range label contributions."""
    range_label_mode = str(range_label_mode).lower()
    if range_label_mode not in {"usefulness", "usefulness_balanced", "usefulness_ship_balanced"}:
        raise ValueError(
            "range_label_mode must be 'usefulness', 'usefulness_balanced', or "
            "'usefulness_ship_balanced'."
        )
    labels, labelled_mask, component_labels = _compute_typed_importance_labels(
        points=points,
        boundaries=boundaries,
        typed_queries=typed_queries,
        range_boundary_prior_weight=range_boundary_prior_weight,
        range_label_mode=range_label_mode,
        return_range_components=True,
    )
    if component_labels is None:
        component_labels = {
            component_name: torch.zeros_like(labels)
            for component_name in RANGE_USEFULNESS_LABEL_COMPONENTS
        }
    return labels, labelled_mask, component_labels
