"""Tests for training loss helpers."""

from __future__ import annotations

import torch

from training.training_losses import (
    _budget_stratified_recall_loss_rows,
    _budget_temporal_cdf_loss_rows,
    _pointwise_bce_loss_rows,
)


def test_pointwise_bce_loss_uses_all_valid_soft_labels() -> None:
    valid_mask = torch.tensor([[True, True, False], [True, True, True]])
    target = torch.tensor([[1.0, 0.0, 1.0], [0.75, 0.25, 0.0]])
    good_pred = torch.tensor([[5.0, -5.0, 0.0], [2.0, -2.0, -5.0]])
    bad_pred = -good_pred

    good_loss, active_rows = _pointwise_bce_loss_rows(
        pred=good_pred,
        target=target,
        valid_mask=valid_mask,
    )
    bad_loss, _ = _pointwise_bce_loss_rows(
        pred=bad_pred,
        target=target,
        valid_mask=valid_mask,
    )

    assert active_rows.tolist() == [True, True]
    assert torch.all(good_loss < bad_loss)


def test_budget_temporal_cdf_loss_penalizes_clustered_budget_mass() -> None:
    valid_mask = torch.ones((2, 6), dtype=torch.bool)
    pred = torch.tensor(
        [
            [5.0, 5.0, 5.0, -5.0, -5.0, -5.0],
            [5.0, -5.0, 5.0, -5.0, 5.0, -5.0],
        ]
    )

    loss_rows, active_rows = _budget_temporal_cdf_loss_rows(
        pred=pred,
        valid_mask=valid_mask,
        budget_ratios=(0.50,),
        temperature=0.05,
    )

    assert active_rows.tolist() == [True, True]
    assert loss_rows[0] > loss_rows[1]


def test_budget_stratified_loss_prefers_one_hit_per_stratum() -> None:
    valid_mask = torch.ones((2, 10), dtype=torch.bool)
    target = torch.zeros((2, 10), dtype=torch.float32)
    target[:, 2] = 1.0
    target[:, 7] = 1.0
    good_pred = torch.full((10,), -4.0)
    good_pred[2] = 4.0
    good_pred[7] = 4.0
    clustered_pred = torch.full((10,), -4.0)
    clustered_pred[2] = 4.0
    clustered_pred[3] = 4.0
    pred = torch.stack([good_pred, clustered_pred])

    loss_rows, active_rows = _budget_stratified_recall_loss_rows(
        pred=pred,
        target=target,
        valid_mask=valid_mask,
        budget_ratios=(0.40,),
        temperature=0.10,
    )

    assert active_rows.tolist() == [True, True]
    assert loss_rows[0] < loss_rows[1]


def test_budget_stratified_loss_skips_endpoint_only_budgets() -> None:
    pred = torch.randn((1, 5), dtype=torch.float32)
    target = torch.ones((1, 5), dtype=torch.float32)
    valid_mask = torch.ones((1, 5), dtype=torch.bool)

    loss_rows, active_rows = _budget_stratified_recall_loss_rows(
        pred=pred,
        target=target,
        valid_mask=valid_mask,
        budget_ratios=(0.01,),
        temperature=0.10,
    )

    assert active_rows.tolist() == [False]
    assert loss_rows.tolist() == [0.0]
