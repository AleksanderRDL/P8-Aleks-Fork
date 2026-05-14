"""Factorized QueryUsefulV1 target scaffolding.

Planned heads:
- query_hit_probability
- conditional_behavior_utility
- boundary_event_utility
- marginal_replacement_gain
- segment_budget_target

This module is intentionally not implemented in the pre-rework cleanup checkpoint.
"""

QUERY_USEFUL_V1_TARGET_MODES = frozenset({"query_useful_v1_factorized"})


def build(*_args: object, **_kwargs: object) -> None:
    """Fail clearly until factorized QueryUsefulV1 targets are implemented."""
    raise NotImplementedError(
        "QueryUsefulV1 factorized targets are not implemented yet. "
        "See Range_QDS/docs/query-driven-rework-guide.md."
    )
