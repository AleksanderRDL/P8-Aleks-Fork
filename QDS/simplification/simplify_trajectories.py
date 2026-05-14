"""Per-trajectory top-k simplification utilities. See simplification/README.md for details."""

from __future__ import annotations

import math
from typing import Any

import torch


def deterministic_topk_with_jitter(
    scores: torch.Tensor,
    keep_count: int,
    trajectory_id: int,
) -> torch.Tensor:
    """Select top-k indices with deterministic pseudo-random tie jitter."""
    score_count = scores.numel()
    if keep_count >= score_count:
        return torch.arange(score_count, dtype=torch.long, device=scores.device)

    positions = torch.arange(score_count, device=scores.device, dtype=torch.float32)
    # Deterministic hash-like noise in [-0.5, 0.5].
    noise = torch.frac(torch.sin(positions * 12.9898 + float(trajectory_id) * 78.233) * 43758.5453) - 0.5
    jittered = scores + 1e-6 * noise
    top = torch.topk(jittered, k=keep_count, largest=True).indices
    return torch.sort(top).values


def diverse_topk_with_jitter(
    scores: torch.Tensor,
    keep_count: int,
    trajectory_id: int,
    existing_indices: torch.Tensor | None = None,
    diversity_bonus: float = 0.0,
) -> torch.Tensor:
    """Select top-k scores while penalizing clusters around already retained points."""
    score_count = int(scores.numel())
    keep_count = max(0, min(int(keep_count), score_count))
    if keep_count <= 0 or score_count <= 0:
        return torch.empty((0,), dtype=torch.long, device=scores.device)
    bonus = max(0.0, float(diversity_bonus))
    if bonus <= 0.0:
        return deterministic_topk_with_jitter(scores, keep_count, trajectory_id)

    candidate_scores = scores.clone()
    selected: list[torch.Tensor] = []
    if existing_indices is not None and existing_indices.numel() > 0:
        retained = existing_indices.to(device=scores.device, dtype=torch.long).flatten()
    else:
        retained = torch.empty((0,), dtype=torch.long, device=scores.device)
    positions = torch.arange(score_count, dtype=torch.float32, device=scores.device)
    denom = float(max(1, score_count - 1))

    for step in range(keep_count):
        if retained.numel() > 0:
            distance_to_retained = torch.abs(
                positions.unsqueeze(1) - retained.float().unsqueeze(0)
            ).min(dim=1).values / denom
        else:
            distance_to_retained = torch.zeros_like(positions)
        adjusted_scores = candidate_scores + bonus * distance_to_retained
        next_idx = deterministic_topk_with_jitter(
            adjusted_scores,
            keep_count=1,
            trajectory_id=trajectory_id * 1009 + step,
        )
        if next_idx.numel() == 0 or not torch.isfinite(candidate_scores[next_idx[0]]):
            break
        selected.append(next_idx)
        retained = torch.cat([retained, next_idx])
        candidate_scores[next_idx] = -float("inf")

    if not selected:
        return torch.empty((0,), dtype=torch.long, device=scores.device)
    return torch.sort(torch.cat(selected).to(dtype=torch.long)).values


def evenly_spaced_indices(point_count: int, keep_count: int, device: torch.device) -> torch.Tensor:
    """Return deterministic evenly spaced local indices, including endpoints when possible."""
    point_count = int(point_count)
    keep_count = max(0, min(int(keep_count), point_count))
    if keep_count <= 0 or point_count <= 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    if keep_count >= point_count:
        return torch.arange(point_count, dtype=torch.long, device=device)
    local_indices = torch.linspace(0, point_count - 1, steps=keep_count, device=device).round().long().unique()
    if local_indices.numel() < keep_count:
        filler_indices = torch.arange(point_count, dtype=torch.long, device=device)
        missing_indices = filler_indices[~torch.isin(filler_indices, local_indices)][
            : keep_count - local_indices.numel()
        ]
        local_indices = torch.cat([local_indices, missing_indices])
    return torch.sort(local_indices).values


def stratified_topk_with_jitter(
    scores: torch.Tensor,
    keep_count: int,
    trajectory_id: int,
    center_weight: float = 0.0,
) -> torch.Tensor:
    """Select one learned top-score point from each temporal/index stratum.

    This keeps the retained set spread across the trajectory while still making
    every non-endpoint choice score-dependent. It is query-blind and deterministic.
    """
    score_count = int(scores.numel())
    keep_count = max(0, min(int(keep_count), score_count))
    if keep_count <= 0 or score_count <= 0:
        return torch.empty((0,), dtype=torch.long, device=scores.device)
    if keep_count >= score_count:
        return torch.arange(score_count, dtype=torch.long, device=scores.device)
    if score_count == 1 or keep_count == 1:
        return deterministic_topk_with_jitter(scores, keep_count=keep_count, trajectory_id=trajectory_id)
    if keep_count == 2:
        return torch.tensor([0, score_count - 1], dtype=torch.long, device=scores.device)
    center_penalty = max(0.0, float(center_weight))

    selected = torch.zeros((score_count,), dtype=torch.bool, device=scores.device)
    selected[0] = True
    selected[score_count - 1] = True

    interior_count = score_count - 2
    interior_slots = keep_count - 2
    if interior_slots >= interior_count:
        return torch.arange(score_count, dtype=torch.long, device=scores.device)

    selected_parts = [
        torch.tensor([0, score_count - 1], dtype=torch.long, device=scores.device),
    ]
    for slot in range(interior_slots):
        left = 1 + int(math.floor(slot * interior_count / interior_slots))
        right = 1 + int(math.floor((slot + 1) * interior_count / interior_slots))
        if right <= left:
            continue
        candidates = torch.arange(left, right, dtype=torch.long, device=scores.device)
        local_scores = scores[candidates]
        if center_penalty > 0.0 and int(candidates.numel()) > 1:
            center = 0.5 * float(left + right - 1)
            denom = max(1.0, 0.5 * float(right - left))
            center_distance = torch.abs(candidates.float() - center) / denom
            local_scores = local_scores - center_penalty * center_distance
        local_choice = deterministic_topk_with_jitter(
            local_scores,
            keep_count=1,
            trajectory_id=trajectory_id * 1009 + slot,
        )
        if local_choice.numel() > 0:
            choice = candidates[local_choice]
            selected[choice] = True
            selected_parts.append(choice)

    selected_count = int(selected.sum().item())
    if selected_count < keep_count:
        remaining_scores = scores.clone()
        remaining_scores[selected] = -float("inf")
        fallback = deterministic_topk_with_jitter(
            remaining_scores,
            keep_count=keep_count - selected_count,
            trajectory_id=trajectory_id * 9176 + keep_count,
        )
        selected_parts.append(fallback)

    return torch.sort(torch.cat(selected_parts).unique()).values[:keep_count]


def _trajectory_budget(point_count: int, compression_ratio: float) -> int:
    """Return the existing per-trajectory retained-point budget."""
    point_count = int(point_count)
    if point_count <= 0:
        return 0
    keep_count = max(2, int(math.ceil(float(compression_ratio) * point_count)))
    return min(keep_count, point_count)


def temporal_hybrid_selector_budget_diagnostics(
    boundaries: list[tuple[int, int]],
    compression_ratios: list[float] | tuple[float, ...],
    *,
    temporal_fraction: float,
    hybrid_mode: str,
    min_learned_swaps: int = 0,
) -> dict[str, Any]:
    """Describe how much retained budget can be decided by learned scores."""
    mode = str(hybrid_mode).lower()
    base_fraction = min(1.0, max(0.0, float(temporal_fraction)))
    minimum_swaps = max(0, int(min_learned_swaps))
    point_total = int(sum(max(0, end - start) for start, end in boundaries))
    rows: list[dict[str, Any]] = []
    for raw_ratio in compression_ratios:
        ratio = min(1.0, max(0.0, float(raw_ratio)))
        total_budget = 0
        learned_slots = 0
        temporal_or_skeleton_slots = 0
        zero_learned_trajectory_count = 0
        endpoint_only_trajectory_count = 0
        active_trajectory_count = 0
        for start, end in boundaries:
            point_count = int(end - start)
            if point_count <= 0:
                continue
            active_trajectory_count += 1
            total_keep = _trajectory_budget(point_count, ratio)
            total_budget += total_keep
            if total_keep <= min(point_count, 2):
                endpoint_only_trajectory_count += 1

            if mode == "stratified":
                learned = max(0, total_keep - min(total_keep, 2))
                base_count = total_keep - learned
            elif mode in {"swap", "local_swap", "local_delta_swap"}:
                base_indices = evenly_spaced_indices(point_count, total_keep, torch.device("cpu"))
                removable_count = int(((base_indices != 0) & (base_indices != point_count - 1)).sum().item())
                protected_count = min(total_keep, max(2, int(math.ceil(total_keep * base_fraction))))
                swap_count = min(total_keep - protected_count, point_count - total_keep)
                if minimum_swaps > 0:
                    swap_count = max(swap_count, minimum_swaps)
                learned = min(max(0, swap_count), removable_count, max(0, point_count - total_keep))
                base_count = total_keep - learned
            elif mode == "global_budget":
                base_count = min(point_count, 2)
                learned = max(0, total_keep - base_count)
            else:
                base_keep = 0
                if base_fraction > 0.0:
                    base_keep = min(total_keep, max(2, int(math.ceil(total_keep * base_fraction))))
                base_count = base_keep
                learned = max(0, total_keep - base_keep)

            learned_slots += learned
            temporal_or_skeleton_slots += base_count
            if learned <= 0:
                zero_learned_trajectory_count += 1

        rows.append(
            {
                "compression_ratio": ratio,
                "trajectory_count": active_trajectory_count,
                "point_count": point_total,
                "total_budget_count": total_budget,
                "temporal_or_skeleton_slot_count": temporal_or_skeleton_slots,
                "learned_slot_count": learned_slots,
                "learned_slot_fraction_of_budget": float(learned_slots / max(1, total_budget)),
                "zero_learned_slot_trajectory_count": zero_learned_trajectory_count,
                "zero_learned_slot_trajectory_fraction": float(
                    zero_learned_trajectory_count / max(1, active_trajectory_count)
                ),
                "endpoint_only_trajectory_count": endpoint_only_trajectory_count,
                "endpoint_only_trajectory_fraction": float(
                    endpoint_only_trajectory_count / max(1, active_trajectory_count)
                ),
            }
        )
    return {
        "hybrid_mode": mode,
        "mlqds_temporal_fraction": base_fraction,
        "mlqds_min_learned_swaps": minimum_swaps,
        "budget_rows": rows,
    }


def simplify_with_scores(
    scores: torch.Tensor,
    boundaries: list[tuple[int, int]],
    compression_ratio: float,
) -> torch.Tensor:
    """Build retained mask by per-trajectory score top-k. See simplification/README.md for details."""
    retained = torch.zeros(scores.shape[0], dtype=torch.bool, device=scores.device)
    for trajectory_id, (start, end) in enumerate(boundaries):
        local_scores = scores[start:end]
        point_count = local_scores.numel()
        if point_count <= 0:
            continue
        keep_count = _trajectory_budget(point_count, compression_ratio)
        local_indices = deterministic_topk_with_jitter(local_scores, keep_count=keep_count, trajectory_id=trajectory_id)
        retained[start:end][local_indices] = True
        retained[start] = True
        retained[end - 1] = True
    return retained


def simplify_with_global_score_budget(
    scores: torch.Tensor,
    boundaries: list[tuple[int, int]],
    compression_ratio: float,
    min_points_per_trajectory: int = 2,
) -> torch.Tensor:
    """Retain endpoint safeguards, then allocate remaining budget globally by score.

    This is intended for diagnostics. It preserves a minimal per-trajectory
    skeleton, but it does not enforce the normal per-trajectory budget.
    """
    point_total = int(scores.numel())
    retained = torch.zeros((point_total,), dtype=torch.bool, device=scores.device)
    if point_total <= 0:
        return retained

    total_budget = 0
    minimum = max(0, int(min_points_per_trajectory))
    for trajectory_id, (start, end) in enumerate(boundaries):
        point_count = int(end - start)
        if point_count <= 0:
            continue
        total_budget += _trajectory_budget(point_count, compression_ratio)
        if minimum <= 0:
            continue
        keep_count = min(point_count, minimum)
        local_indices = evenly_spaced_indices(point_count, keep_count, scores.device)
        retained[start + local_indices] = True

    total_budget = min(point_total, max(int(retained.sum().item()), int(total_budget)))
    remaining = total_budget - int(retained.sum().item())
    if remaining <= 0:
        return retained

    candidate_scores = scores.float().clone()
    candidate_scores[retained] = -float("inf")
    finite = torch.isfinite(candidate_scores)
    if not bool(finite.any().item()):
        return retained
    keep_count = min(int(remaining), int(finite.sum().item()))
    global_indices = deterministic_topk_with_jitter(
        candidate_scores,
        keep_count=keep_count,
        trajectory_id=point_total + 7919,
    )
    retained[global_indices] = True
    return retained


def simplify_with_temporal_global_score_fill(
    scores: torch.Tensor,
    boundaries: list[tuple[int, int]],
    compression_ratio: float,
    temporal_fraction: float = 0.50,
    diversity_bonus: float = 0.0,
) -> torch.Tensor:
    """Retain a per-trajectory temporal base, then fill residual slots globally.

    This keeps trajectory continuity safeguards from the temporal hybrid while
    letting learned score decide which trajectories receive the remaining
    residual budget.
    """
    retained = torch.zeros(scores.shape[0], dtype=torch.bool, device=scores.device)
    base_fraction = min(1.0, max(0.0, float(temporal_fraction)))
    bonus = max(0.0, float(diversity_bonus))
    total_budget = 0
    candidate_scores = scores.float().clone()

    for trajectory_id, (start, end) in enumerate(boundaries):
        local_scores = scores[start:end]
        point_count = int(local_scores.numel())
        if point_count <= 0:
            continue
        total_keep_count = _trajectory_budget(point_count, compression_ratio)
        total_budget += total_keep_count
        if base_fraction <= 0.0:
            continue
        base_keep_count = min(
            total_keep_count,
            max(2, int(math.ceil(total_keep_count * base_fraction))),
        )
        base_indices = evenly_spaced_indices(point_count, base_keep_count, scores.device)
        retained[start + base_indices] = True
        if bonus > 0.0 and base_indices.numel() > 0 and point_count > 1:
            positions = torch.arange(point_count, dtype=torch.float32, device=scores.device)
            distance_to_base = torch.abs(
                positions.unsqueeze(1) - base_indices.float().unsqueeze(0)
            ).min(dim=1).values
            local_adjusted = candidate_scores[start:end]
            local_adjusted = local_adjusted + bonus * (distance_to_base / float(max(1, point_count - 1)))
            candidate_scores[start:end] = local_adjusted

    total_budget = min(int(scores.numel()), max(int(retained.sum().item()), int(total_budget)))
    remaining = total_budget - int(retained.sum().item())
    if remaining <= 0:
        return retained

    candidate_scores[retained] = -float("inf")
    finite = torch.isfinite(candidate_scores)
    if not bool(finite.any().item()):
        return retained
    keep_count = min(int(remaining), int(finite.sum().item()))
    global_indices = deterministic_topk_with_jitter(
        candidate_scores,
        keep_count=keep_count,
        trajectory_id=int(scores.numel()) + 15401,
    )
    retained[global_indices] = True
    return retained


def simplify_with_temporal_score_hybrid(
    scores: torch.Tensor,
    boundaries: list[tuple[int, int]],
    compression_ratio: float,
    temporal_fraction: float = 0.50,
    diversity_bonus: float = 0.0,
    hybrid_mode: str = "fill",
    stratified_center_weight: float = 0.0,
    min_learned_swaps: int = 0,
) -> torch.Tensor:
    """Retain a temporal coverage base, then fill remaining slots by learned score.

    Pure top-k scoring tends to over-select neighbouring points with similar
    logits.  The fill/swap modes keep a temporal base and spend part of the
    budget by learned score.  Global fill keeps a temporal base per trajectory
    and spends only the residual budget globally.  Global budget keeps only a
    minimal trajectory skeleton before spending the remaining budget across all
    trajectories by learned score.  Local swap pairs each learned addition with the
    nearest unprotected temporal-base removal to reduce continuity damage.
    Local delta swap additionally requires the candidate's learned score to
    exceed that paired base point before replacing it.  The stratified mode
    instead selects learned top-score points inside disjoint trajectory-order
    bins.
    """
    retained = torch.zeros(scores.shape[0], dtype=torch.bool, device=scores.device)
    base_fraction = min(1.0, max(0.0, float(temporal_fraction)))
    bonus = max(0.0, float(diversity_bonus))
    center_weight = max(0.0, float(stratified_center_weight))
    minimum_swaps = max(0, int(min_learned_swaps))
    mode = str(hybrid_mode).lower()
    if mode == "global_fill":
        return simplify_with_temporal_global_score_fill(
            scores,
            boundaries,
            compression_ratio,
            temporal_fraction=temporal_fraction,
            diversity_bonus=diversity_bonus,
        )
    if mode == "global_budget":
        return simplify_with_global_score_budget(scores, boundaries, compression_ratio)
    if mode not in {"fill", "swap", "local_swap", "local_delta_swap", "stratified"}:
        raise ValueError(
            "hybrid_mode must be 'fill', 'swap', 'local_swap', 'local_delta_swap', "
            "'stratified', 'global_fill', or 'global_budget'."
        )

    for trajectory_id, (start, end) in enumerate(boundaries):
        local_scores = scores[start:end]
        point_count = local_scores.numel()
        if point_count <= 0:
            continue
        total_keep_count = _trajectory_budget(point_count, compression_ratio)
        if mode == "stratified":
            local_indices = stratified_topk_with_jitter(
                local_scores,
                keep_count=total_keep_count,
                trajectory_id=trajectory_id,
                center_weight=center_weight,
            )
            retained[start + local_indices] = True
            continue
        if mode in {"swap", "local_swap", "local_delta_swap"}:
            base_indices = evenly_spaced_indices(point_count, total_keep_count, scores.device)
            retained[start + base_indices] = True
            protected_count = min(total_keep_count, max(2, int(math.ceil(total_keep_count * base_fraction))))
            swap_count = min(total_keep_count - protected_count, point_count - total_keep_count)
            removable_indices = base_indices[(base_indices != 0) & (base_indices != point_count - 1)]
            if minimum_swaps > 0:
                swap_count = max(swap_count, minimum_swaps)
            swap_count = min(swap_count, int(removable_indices.numel()), point_count - total_keep_count)
            if swap_count <= 0:
                continue

            remove_positions = deterministic_topk_with_jitter(
                -local_scores[removable_indices],
                keep_count=swap_count,
                trajectory_id=trajectory_id,
            )
            remove_indices = removable_indices[remove_positions]
            candidate_scores = local_scores.clone()
            candidate_scores[base_indices] = -float("inf")
            add_indices = diverse_topk_with_jitter(
                candidate_scores,
                keep_count=swap_count,
                trajectory_id=trajectory_id,
                existing_indices=base_indices,
                diversity_bonus=bonus,
            )
            if mode == "local_swap" and add_indices.numel() > 0:
                remove_values: list[int] = []
                available = removable_indices.detach().cpu().tolist()
                ordered_add = add_indices[
                    torch.argsort(local_scores[add_indices], descending=True)
                ].detach().cpu().tolist()
                for add_value in ordered_add:
                    if not available:
                        break
                    add_int = int(add_value)
                    remove_int = min(available, key=lambda value: (abs(int(value) - add_int), int(value)))
                    available.remove(remove_int)
                    remove_values.append(remove_int)
                if remove_values:
                    remove_indices = torch.tensor(remove_values, dtype=torch.long, device=scores.device)
                    add_indices = torch.tensor(
                        ordered_add[: len(remove_values)],
                        dtype=torch.long,
                        device=scores.device,
                    )
                else:
                    add_indices = add_indices[:0]
                    remove_indices = remove_indices[:0]
            elif mode == "local_delta_swap":
                add_values: list[int] = []
                remove_values: list[int] = []
                available = removable_indices.detach().cpu().tolist()
                candidate_values = torch.where(torch.isfinite(candidate_scores))[0].detach().cpu().tolist()
                for step in range(swap_count):
                    if not available or not candidate_values:
                        break
                    best_candidate: int | None = None
                    best_remove: int | None = None
                    best_value: float | None = None
                    for candidate_value in candidate_values:
                        candidate_int = int(candidate_value)
                        remove_int = min(available, key=lambda value: (abs(int(value) - candidate_int), int(value)))
                        delta = float(local_scores[candidate_int].item()) - float(local_scores[remove_int].item())
                        # Deterministic jitter only breaks exact ties; it must not turn a
                        # non-positive learned replacement into a positive one.
                        jitter = 1e-6 * math.sin(
                            float(candidate_int * 12.9898 + trajectory_id * 78.233 + step * 37.719)
                        )
                        value = delta + jitter
                        if best_value is None or value > best_value:
                            best_value = value
                            best_candidate = candidate_int
                            best_remove = int(remove_int)
                    if best_candidate is None or best_remove is None:
                        break
                    raw_delta = float(local_scores[best_candidate].item()) - float(local_scores[best_remove].item())
                    if raw_delta <= 0.0:
                        break
                    add_values.append(best_candidate)
                    remove_values.append(best_remove)
                    candidate_values.remove(best_candidate)
                    available.remove(best_remove)
                if add_values:
                    add_indices = torch.tensor(add_values, dtype=torch.long, device=scores.device)
                    remove_indices = torch.tensor(remove_values, dtype=torch.long, device=scores.device)
                else:
                    add_indices = add_indices[:0]
                    remove_indices = remove_indices[:0]
            retained[start + remove_indices] = False
            retained[start + add_indices] = True
            continue

        base_keep_count = 0
        if base_fraction > 0.0:
            base_keep_count = min(total_keep_count, max(2, int(math.ceil(total_keep_count * base_fraction))))
        base_indices = evenly_spaced_indices(point_count, base_keep_count, scores.device)
        retained[start + base_indices] = True

        remaining_count = total_keep_count - int(base_indices.numel())
        if remaining_count <= 0:
            continue

        candidate_scores = local_scores.clone()
        candidate_scores[base_indices] = -float("inf")
        fill_indices = diverse_topk_with_jitter(
            candidate_scores,
            keep_count=remaining_count,
            trajectory_id=trajectory_id,
            existing_indices=base_indices,
            diversity_bonus=bonus,
        )
        retained[start + fill_indices] = True

    return retained
