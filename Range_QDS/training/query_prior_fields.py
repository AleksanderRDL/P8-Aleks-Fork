"""Train-derived query prior fields for the QueryUsefulV1 rework.

These fields may use train workloads and train labels.

They must not use validation/eval queries before retained masks are frozen.

Planned fields:
- spatial_query_hit_probability
- spatiotemporal_query_hit_probability
- boundary_entry_exit_likelihood
- crossing_likelihood
- behavior_utility_prior
- route_density_prior

Current density/sparsity features are split-local point-cloud context features.
They are not train-derived query-prior fields.

This module is intentionally not implemented in the pre-rework cleanup checkpoint.
"""
