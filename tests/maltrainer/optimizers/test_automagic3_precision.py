from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from maltrainer.optimizers import Automagic3


def _backward(param: nn.Parameter, weight: list[float]) -> None:
    (param.float() * torch.tensor(weight)).sum().backward()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_low_precision_gradients_accumulate_across_backward_passes(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    p = nn.Parameter(torch.ones(3, dtype=dtype))
    opt = Automagic3([p], lr=0.5, beta2=0.0, clip_threshold=10.0)

    _backward(p, [1.0, -0.25, 0.5])
    _backward(p, [0.5, -0.25, -1.5])

    assert p.grad is None
    assert len(opt.state) == 0
    torch.testing.assert_close(p._accum_grad, torch.tensor([1.5, -0.5, -1.0], dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_a_step_consumes_the_accumulated_low_precision_gradient(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    p = nn.Parameter(torch.ones(3, dtype=dtype))
    opt = Automagic3([p], lr=0.5, beta2=0.0, clip_threshold=10.0)
    _backward(p, [1.0, -0.25, 0.0])
    _backward(p, [0.5, -0.25, 0.0])

    opt.step()

    assert not hasattr(p, "_accum_grad")
    torch.testing.assert_close(p.detach(), torch.tensor([0.5, 1.5, 1.0], dtype=dtype))
    assert opt.state[p]["exp_avg_sq"].dtype == dtype
    assert opt.state[p]["lr"].dtype == torch.float32


def test_a_bf16_matrix_keeps_its_factored_moments_in_bf16() -> None:
    torch.manual_seed(0)
    p = nn.Parameter(torch.zeros(2, 2, dtype=torch.bfloat16))
    opt = Automagic3([p], lr=0.5, beta2=0.0, clip_threshold=10.0)
    p.grad = torch.tensor([[2.0, 2.0], [2.0, 2.0]], dtype=torch.bfloat16)

    opt.step()

    torch.testing.assert_close(opt.state[p]["exp_avg_sq_row"], torch.full((2,), 4.0, dtype=torch.bfloat16))
    torch.testing.assert_close(opt.state[p]["exp_avg_sq_col"], torch.full((2,), 4.0, dtype=torch.bfloat16))
    torch.testing.assert_close(p.detach(), torch.full((2, 2), -0.5, dtype=torch.bfloat16))


def test_bf16_weights_are_stochastically_rounded_without_bias() -> None:
    torch.manual_seed(0)
    p = nn.Parameter(torch.ones(4096, dtype=torch.bfloat16))
    opt = Automagic3([p], lr=2**-10, beta2=0.0)
    p.grad = torch.ones(4096, dtype=torch.bfloat16)

    opt.step()

    values = p.detach().float()
    assert set(values.unique().tolist()) == {1.0 - 2**-8, 1.0}
    assert values.mean().item() == pytest.approx(1.0 - 2**-10, abs=2e-4)


def test_fp8_stochastic_copies_round_to_either_neighbour_without_bias() -> None:
    torch.manual_seed(0)
    dst = torch.empty(4096, dtype=torch.float8_e4m3fn)

    Automagic3._stochastic_copy_(dst, torch.full((4096,), 1.0625))

    values = dst.float()
    assert set(values.unique().tolist()) == {1.0, 1.125}
    assert values.mean().item() == pytest.approx(1.0625, abs=0.005)


def test_a_float64_parameter_is_updated_at_fp32_precision() -> None:
    p = nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))
    opt = Automagic3([p], lr=0.5, beta2=0.0, clip_threshold=10.0, weight_decay=0.5)
    p.grad = torch.tensor([1.0, -1.0], dtype=torch.float64)

    opt.step()

    torch.testing.assert_close(p.detach(), torch.tensor([1.0 - 0.5 * 1.5, 2.0 - 0.5 * 0.0], dtype=torch.float64))


def test_a_fused_optimizer_updates_each_parameter_during_backward() -> None:
    p = nn.Parameter(torch.ones(2))
    opt = Automagic3([p], lr=0.5, beta2=0.0, clip_threshold=10.0, fused=True)

    _backward(p, [1.0, -1.0])
    after_backward = p.detach().clone()
    opt.step()

    torch.testing.assert_close(after_backward, torch.tensor([0.5, 1.5]))
    torch.testing.assert_close(p.detach(), after_backward)
    assert p.grad is None


def test_a_fused_optimizer_applies_the_pooled_vote_at_step() -> None:
    p = nn.Parameter(torch.ones(2))
    opt = Automagic3([p], lr=1e-4, beta2=0.0, polarity_history=2, fused=True)

    for _ in range(2):
        _backward(p, [1.0, 1.0])
    before_step = opt.state[p]["lr"].item()
    opt.step()

    assert before_step == pytest.approx(1e-4)
    assert opt.state[p]["lr"].item() == pytest.approx(1e-4 * math.e, rel=1e-5)


def test_hooks_are_registered_only_where_they_are_needed() -> None:
    fp32, bf16 = nn.Parameter(torch.ones(1)), nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
    frozen = nn.Parameter(torch.ones(1, dtype=torch.bfloat16), requires_grad=False)

    plain = Automagic3([fp32, bf16, frozen])
    fused = Automagic3([fp32, bf16, frozen], fused=True)

    assert len(plain._hook_handles) == 1
    assert len(fused._hook_handles) == 2
    for handle in plain._hook_handles + fused._hook_handles:
        handle.remove()


def test_a_parameter_whose_gradient_was_cleared_is_left_alone() -> None:
    p = nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    opt = Automagic3([p], fused=True)

    opt._update_param(p, opt.param_groups[0])
    opt._make_accum_hook()(p)

    assert len(opt.state[p]) == 0
    assert not hasattr(p, "_accum_grad")
    torch.testing.assert_close(p.detach(), torch.ones(2, dtype=torch.bfloat16))
    for handle in opt._hook_handles:
        handle.remove()
