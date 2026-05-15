"""Selector diagnostic field names for learned-contribution auditing.

The current implementation lives in ``learned_segment_budget`` and in the
experiment pipeline. Keep this module as a small shared vocabulary so older
imports do not point at a stale pre-rework placeholder.
"""

LEARNED_SELECTOR_DIAGNOSTIC_FIELDS = (
    "learned_controlled_retained_slots",
    "learned_controlled_retained_slot_fraction",
    "trajectories_with_at_least_one_learned_decision",
    "trajectories_with_zero_learned_decisions",
    "segment_budget_entropy",
    "segment_budget_entropy_normalized",
    "shuffled_score_ablation_delta",
    "untrained_score_ablation_delta",
    "shuffled_prior_field_ablation_delta",
    "no_behavior_head_ablation_delta",
    "no_segment_budget_head_ablation_delta",
)
