# Visualization Module

This package is intentionally minimal.

## Current State

- Only `__init__.py` is present right now.
- `VisualizationConfig.enabled` exists in the experiment config, but the current experiment runner writes JSON, CSV, GeoJSON, and fixed-width tables instead of plots.
- New plotting or notebook-style reporting code should live here if it is added later.
