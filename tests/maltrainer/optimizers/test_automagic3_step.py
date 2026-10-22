from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from maltrainer.optimizers import Automagic3


def _param(*values: float) -> nn.Parameter:
    return nn.Parameter(torch.tensor(values))


def _step(optimizer: Automagic3, param: nn.Parameter, grad: list[float]) -> None:
    param.grad = torch.tensor(grad)
    optimizer.step()


def test_a_first_step_moves_each_element_by_the_normalised_gradient() -> None:
    p = _param(1.0, 2.0, 3.0)
    opt = Automagic3([p], lr=0.5, beta2=0.5, clip_threshold=10.0)

    _step(opt, p, [4.0, -0.25, 0.0])

    torch.testing.assert_close(p.detach(), torch.tensor([1.0 - 0.5 * math.sqrt(2.0), 2.0 + 0.5 * math.sqrt(2.0), 3.0]))
    torch.testing.assert_close(opt.state[p]["exp_avg_sq"], torch.tensor([8.0, 0.03125, 0.0]))


def test_the_second_moment_is_an_exponential_average_of_squared_gradients() -> None:
    p = _param(0.0, 0.0)
    opt = Automagic3([p], lr=0.5, beta2=0.5, clip_threshold=10.0)
    _step(opt, p, [2.0, 1.0])

    _step(opt, p, [1.0, 1.0])

    v = torch.tensor([0.5 * 2.0 + 0.5 * 1.0, 0.5 * 0.5 + 0.5 * 1.0])
    first = torch.tensor([math.sqrt(2.0), math.sqrt(2.0)])
    torch.testing.assert_close(opt.state[p]["exp_avg_sq"], v)
    torch.testing.assert_close(p.detach(), -0.5 * (first + 1.0 / v.sqrt()))


def test_an_update_is_scaled_to_the_clip_rms_then_clamped_per_element() -> None:
    p = _param(0.0, 0.0, 0.0)
    opt = Automagic3([p], lr=0.5, beta2=0.5, clip_threshold=1.0)

    _step(opt, p, [3.0, 0.0, -3.0])

    torch.testing.assert_close(p.detach(), torch.tensor([-0.5, 0.0, 0.5]))


def test_a_matrix_uses_the_factored_row_and_column_second_moment() -> None:
    torch.manual_seed(0)
    grad = torch.randn(2, 3)
    p = nn.Parameter(torch.zeros(2, 3))
    opt = Automagic3([p], lr=0.1, beta2=0.5, clip_threshold=1e3)

    p.grad = grad.clone()
    opt.step()

    row = 0.5 * grad.pow(2).mean(dim=1)
    col = 0.5 * grad.pow(2).mean(dim=0)
    expected_update = grad / torch.sqrt(torch.outer(row, col) / row.mean())
    torch.testing.assert_close(opt.state[p]["exp_avg_sq_row"], row)
    torch.testing.assert_close(opt.state[p]["exp_avg_sq_col"], col)
    torch.testing.assert_close(p.detach(), -0.1 * expected_update)
    assert "exp_avg_sq" not in opt.state[p]


def test_weight_decay_pulls_the_weights_towards_zero() -> None:
    p = _param(1.0, -2.0)
    opt = Automagic3([p], lr=0.5, weight_decay=0.5)

    _step(opt, p, [0.0, 0.0])

    torch.testing.assert_close(p.detach(), torch.tensor([0.75, -1.5]))


@pytest.mark.parametrize("scale", [0.0, 0.25, 1.0])
def test_lr_scale_multiplies_the_step(scale: float) -> None:
    p = _param(1.0, 1.0)
    opt = Automagic3([p], lr=0.5, beta2=0.0, clip_threshold=10.0)
    opt.lr_scale = scale

    _step(opt, p, [1.0, -1.0])

    torch.testing.assert_close(p.detach(), torch.tensor([1.0 - 0.5 * scale, 1.0 + 0.5 * scale]))


def test_nan_and_inf_gradients_are_zeroed_and_counted() -> None:
    p = _param(1.0, 1.0, 1.0)
    opt = Automagic3([p], lr=0.5, beta2=0.0)

    _step(opt, p, [math.nan, math.inf, 1.0])
    _step(opt, p, [-math.inf, 0.0, 0.0])

    torch.testing.assert_close(p.detach(), torch.tensor([1.0, 1.0, 0.5]))
    assert opt.pop_stats()["nonfinite"] == 3
    assert opt.pop_stats()["nonfinite"] == 0


def test_a_step_clears_the_gradient_and_counts_the_steps() -> None:
    p = _param(1.0)
    opt = Automagic3([p], lr=0.5)

    _step(opt, p, [1.0])
    _step(opt, p, [1.0])

    assert p.grad is None
    assert opt.state[p]["step"] == 2


def test_parameters_without_a_gradient_or_frozen_are_left_alone() -> None:
    trained, idle = _param(1.0), _param(2.0)
    frozen = torch.tensor([3.0])
    frozen.grad = torch.tensor([1.0])
    opt = Automagic3([trained, idle, frozen], lr=0.5)

    _step(opt, trained, [1.0])

    assert trained.item() == 0.5
    assert idle.item() == 2.0
    assert frozen.item() == 3.0
    assert idle not in opt.state
    assert frozen not in opt.state


def test_step_runs_the_closure_with_gradients_enabled_and_returns_its_loss() -> None:
    p = _param(1.0, 2.0)
    opt = Automagic3([p], lr=0.5, beta2=0.0, clip_threshold=10.0)

    def closure() -> torch.Tensor:
        loss = (p * torch.tensor([1.0, -1.0])).sum()
        loss.backward()
        return loss

    loss = opt.step(closure)

    assert loss.item() == -1.0
    torch.testing.assert_close(p.detach(), torch.tensor([0.5, 2.5]))


def test_sparse_gradients_are_rejected() -> None:
    embedding = nn.Embedding(3, 2, sparse=True)
    opt = Automagic3(embedding.parameters(), lr=1e-3)
    embedding(torch.tensor([0, 2])).sum().backward()

    with pytest.raises(RuntimeError, match="sparse"):
        opt.step()
