"""The masked loss: weights from the mask, per-picture normalisation, and the prior outside it."""

from __future__ import annotations

import torch

from maltrainer.training.masked_loss import mask_weight, prior_loss, weighted_mse


def test_the_mask_never_weighs_below_the_floor_and_white_is_one() -> None:
    weight = mask_weight(torch.tensor([0.0, 0.05, 0.5, 1.0, 1.5]), 0.1)

    assert torch.allclose(weight, torch.tensor([0.1, 0.1, 0.5, 1.0, 1.0]))


def test_without_a_mask_the_loss_is_plain_mse() -> None:
    pred, target = torch.tensor([[[[1.0, 2.0]]]]), torch.zeros(1, 1, 1, 2)

    assert weighted_mse(pred, target, None).item() == 2.5


def test_a_picture_counts_the_same_however_little_of_it_is_white() -> None:
    pred, target = torch.ones(1, 1, 2, 2), torch.zeros(1, 1, 2, 2)
    small = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    big = torch.ones(1, 1, 2, 2)

    assert weighted_mse(pred, target, small).item() == weighted_mse(pred, target, big).item() == 1.0


def test_the_error_is_weighted_where_the_mask_says() -> None:
    pred = torch.tensor([[[[2.0, 1.0]]]])
    weight = torch.tensor([[[[1.0, 0.0]]]])  # only the first cell counts

    assert weighted_mse(pred, torch.zeros(1, 1, 1, 2), weight).item() == 4.0


def test_the_prior_pulls_the_outside_to_the_base_prediction_only() -> None:
    pred = torch.tensor([[[[3.0, 5.0]]]])
    base = torch.tensor([[[[0.0, 4.0]]]])
    weight = torch.tensor([[[[1.0, 0.1]]]])  # the second cell is outside the mask, at the floor

    loss = prior_loss(pred, base, weight, 2.0)

    # outside = [0, 0.9] -> weights [0, 2] over errors [9, 1]: mean 1.0, times the prior 2
    assert torch.isclose(loss, torch.tensor(2.0))
    assert prior_loss(pred, base, torch.ones(1, 1, 1, 2), 2.0).item() == 0.0  # nothing outside, nothing preserved
