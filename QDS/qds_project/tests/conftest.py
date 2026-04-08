"""Shared pytest configuration for QDS test suite."""

from __future__ import annotations

import pytest


_CATEGORY_MARKERS = {"unit", "integration", "slow"}
_CATEGORY_ORDER = {"unit": 0, "integration": 1, "slow": 2}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-label tests and enforce run order: unit -> integration -> slow."""
    for item in items:
        marker_names = {mark.name for mark in item.iter_markers()}
        if marker_names.isdisjoint(_CATEGORY_MARKERS):
            item.add_marker(pytest.mark.unit)

    def _item_priority(item: pytest.Item) -> int:
        marker_names = {mark.name for mark in item.iter_markers()}
        priorities = [_CATEGORY_ORDER[name] for name in marker_names if name in _CATEGORY_ORDER]
        return max(priorities) if priorities else _CATEGORY_ORDER["unit"]

    items.sort(key=lambda item: (_item_priority(item), str(item.fspath), item.nodeid))
