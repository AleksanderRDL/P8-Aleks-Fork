from __future__ import annotations

# pyright: reportPrivateImportUsage=false
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from training.query_prior_fields import QUERY_PRIOR_FIELD_NAMES, zero_query_prior_field_like

pytestmark = pytest.mark.property


FAST_PROPERTY_SETTINGS = settings(max_examples=50, deadline=None)


@given(
    grid_bins=st.integers(min_value=2, max_value=32),
    time_bins=st.integers(min_value=1, max_value=16),
)
@FAST_PROPERTY_SETTINGS
def test_zero_prior_field_preserves_metadata_shape_and_extent(
    grid_bins: int,
    time_bins: int,
) -> None:
    extent = {
        "t_min": 0.0,
        "t_max": 10.0,
        "lat_min": 1.0,
        "lat_max": 2.0,
        "lon_min": 3.0,
        "lon_max": 4.0,
    }
    prior = {
        "field_names": list(QUERY_PRIOR_FIELD_NAMES),
        "extent": extent,
        "grid_bins": grid_bins,
        "time_bins": time_bins,
        "contains_eval_queries": True,
        "contains_validation_queries": True,
        "diagnostics": {"source": "property_fixture"},
    }
    for field_name in QUERY_PRIOR_FIELD_NAMES:
        prior[field_name] = torch.ones((time_bins, grid_bins, grid_bins), dtype=torch.float32)

    zeroed = zero_query_prior_field_like(prior)

    assert zeroed["extent"] == extent
    assert zeroed["grid_bins"] == grid_bins
    assert zeroed["time_bins"] == time_bins
    assert zeroed["field_names"] == list(QUERY_PRIOR_FIELD_NAMES)
    assert zeroed["built_from_split"] == "train_only"
    assert zeroed["contains_eval_queries"] is False
    assert zeroed["contains_validation_queries"] is False
    assert zeroed["diagnostics"]["zeroed_prior_features_preserve_train_extent"] is True
    for field_name in QUERY_PRIOR_FIELD_NAMES:
        assert zeroed[field_name].shape == prior[field_name].shape
        assert torch.count_nonzero(zeroed[field_name]).item() == 0
