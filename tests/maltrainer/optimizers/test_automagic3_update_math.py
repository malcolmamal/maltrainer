"""The pieces of an Automagic3 update (state layout, second moments, the applied step, the vote signal) one by one."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from maltrainer.optimizers import Automagic3
from maltrainer.optimizers.automagic3_update import ParamUpdate


def an_optimizer(*shape: int, dtype: torch.dtype = torch.float32, **options: float) -> tuple[Automagic3, nn.Parameter]:
    p = nn.Parameter(torch.zeros(*shape, dtype=dtype))
    return Automagic3([p], **options), p


def test_the_rms_is_the_two_norm_over_the_square_root_of_the_element_count() -> None:
    assert ParamUpdate._rms(torch.tensor([3.0, 4.0, 0.0, 0.0])).item() == pytest.approx(2.5)


def test_the_factored_second_moment_is_rsqrt_of_the_row_over_its_mean_times_rsqrt_of_the_column() -> None:
    row = torch.tensor([[1.0, 3.0], [2.0, 2.0]])
    col = torch.tensor([[1.0, 4.0, 9.0], [4.0, 4.0, 16.0]])

    out = ParamUpdate._approx_sq_grad(row.clone(), col)

    expected = torch.empty(2, 2, 3)
    for a in range(2):
        for i in range(2):
            for j in range(3):
                expected[a, i, j] = (row[a, i] / row[a].mean()) ** -0.5 * col[a, j] ** -0.5
    torch.testing.assert_close(out, expected)


def test_a_fresh_state_has_empty_history_a_clamped_rate_and_moments_shaped_by_the_parameter() -> None:
    opt, cube = an_optimizer(2, 3, 4, lr=5.0, max_lr=2.0, polarity_history=4)

    opt._init_state(cube, opt.param_groups[0])

    state = opt.state[cube]
    assert state["step"] == 0 and state["hist_idx"] == 0 and state["hist_fill"] == 0
    assert state["lr"].item() == 2.0
    assert state["sign_history"].shape == (4, 3) and state["sign_history"].dtype == torch.uint8
    assert state["exp_avg_sq_row"].shape == (2, 3) and state["exp_avg_sq_col"].shape == (2, 4)
    assert "exp_avg_sq" not in state


def test_a_vector_state_has_one_second_moment_per_element_and_a_rate_raised_to_the_floor() -> None:
    opt, vector = an_optimizer(10, lr=1e-9, min_lr=1e-4, polarity_history=2)

    opt._init_state(vector, opt.param_groups[0])

    state = opt.state[vector]
    assert state["exp_avg_sq"].shape == (10,)
    assert state["lr"].item() == pytest.approx(1e-4)
    assert state["sign_history"].shape == (2, 2)


def test_a_half_precision_gradient_is_cleaned_in_fp32_and_its_bad_values_are_counted() -> None:
    opt, p = an_optimizer(4, dtype=torch.bfloat16)
    p.grad = torch.tensor([1.0, float("nan"), float("inf"), -2.0], dtype=torch.bfloat16)

    grad = opt._clean_grad(p)

    assert grad.dtype == torch.float32
    assert grad.tolist() == [1.0, 0.0, 0.0, -2.0]
    assert opt._nonfinite is not None and int(opt._nonfinite) == 2


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_a_factored_moment_is_averaged_in_fp32_and_rounded_once_into_the_state_dtype(dtype: torch.dtype) -> None:
    row0 = torch.tensor([[1.0, 3.0, 5.0], [7.0, 9.0, 11.0]], dtype=dtype)
    col0 = torch.tensor([[1.0, 3.0, 5.0, 7.0], [2.0, 4.0, 6.0, 8.0]], dtype=dtype)
    state = {"exp_avg_sq_row": row0.clone(), "exp_avg_sq_col": col0.clone()}
    sq = torch.arange(1.0, 25.0).reshape(2, 3, 4) / 7
    grad = torch.linspace(-1.0, 1.0, 24).reshape(2, 3, 4)
    beta2, eps = 0.9, 1e-3

    out = ParamUpdate._factored_update(state, sq, grad, beta2, eps)

    row = row0.float() * beta2 + (sq.mean(dim=-1) + eps) * (1.0 - beta2)
    col = col0.float() * beta2 + (sq.mean(dim=-2) + eps) * (1.0 - beta2)
    assert torch.equal(state["exp_avg_sq_row"], row.to(dtype))
    assert torch.equal(state["exp_avg_sq_col"], col.to(dtype))
    torch.testing.assert_close(out, ParamUpdate._approx_sq_grad(row, col) * grad)


def test_a_half_precision_vector_moment_is_averaged_in_fp32_and_rounded_once() -> None:
    v0 = torch.tensor([1.0, 3.0, 5.0, 7.0], dtype=torch.bfloat16)
    state = {"exp_avg_sq": v0.clone()}
    sq = torch.tensor([0.3, 0.7, 1.1, 1.3])
    grad = torch.tensor([1.0, -1.0, 2.0, -2.0])

    out = ParamUpdate._vector_update(state, sq, grad, 0.9, 1e-3)

    v = v0.float() * 0.9 + sq * (1.0 - 0.9)
    assert torch.equal(state["exp_avg_sq"], v.to(torch.bfloat16))
    torch.testing.assert_close(out, (v + 1e-3).rsqrt() * grad)


def test_a_float32_vector_moment_is_updated_in_place() -> None:
    v = torch.tensor([2.0, 4.0])
    state = {"exp_avg_sq": v}

    ParamUpdate._vector_update(state, torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0]), 0.5, 0.0)

    assert v.tolist() == [1.5, 2.5]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(("decay", "expected"), [(0.0, 1.5), (1.0, 0.5), (0.5, 1.0)])
def test_the_step_is_minus_rate_times_the_update_plus_weight_decay_times_the_weight(
    dtype: torch.dtype, decay: float, expected: float
) -> None:
    opt, p = an_optimizer(3, dtype=dtype)
    p.data.fill_(2.0)
    update = torch.ones(3)

    with torch.no_grad():
        opt._apply_update(p, update, torch.tensor(0.5), decay)

    assert p.detach().float().tolist() == [expected] * 3


def test_the_rate_scale_multiplies_the_step() -> None:
    opt, p = an_optimizer(2)
    opt.lr_scale = 0.25

    with torch.no_grad():
        opt._apply_update(p, torch.ones(2), torch.tensor(0.5), 0.0)

    assert p.detach().tolist() == [-0.125, -0.125]


def test_a_vote_adds_up_minus_down_weighted_by_the_update_size_to_its_group() -> None:
    opt, p = an_optimizer(4, polarity_history=2)
    hist = torch.stack([ParamUpdate._pack_bits(torch.tensor(step)) for step in ([1, 1, 0, 0], [1, 0, 1, 0])])
    update = torch.tensor([1.0, 2.0, 4.0, 8.0])

    opt._record_votes(p, hist, 0, update)
    opt._record_votes(p, hist, 0, update)

    # element 0 voted up (+1), element 3 voted up (equal, both 0: +8); elements 1 and 2 flipped (-2, -4)
    assert opt._group_num[0].item() == pytest.approx(2 * (1.0 - 2.0 - 4.0 + 8.0))
    assert opt._group_den[0].item() == pytest.approx(2 * 15.0)


def test_the_vote_signal_is_num_over_den_clamped_to_unit_range_and_moves_the_rates() -> None:
    opt, p = an_optimizer(2, lr=1e-3, min_lr=1e-9, max_lr=1.0)
    opt._init_state(p, opt.param_groups[0])

    for num, den, signal in ((0.2, 0.4, 0.5), (3.0, 1.0, 1.0), (-3.0, 1.0, -1.0), (5e-4, 1e-3, 0.5)):
        opt._group_num[0], opt._group_den[0] = torch.tensor(num), torch.tensor(den)
        before = opt.state[p]["lr"].item()

        opt._apply_group_votes()

        assert opt._signal_sum["0"].item() == pytest.approx(signal)
        assert opt.state[p]["lr"].item() == pytest.approx(before * torch.exp(torch.tensor(signal)).item(), rel=1e-5)
        assert opt._group_num[0] is None and opt._group_den[0] is None
        opt._signal_sum.clear()


def test_the_average_rate_of_an_optimizer_without_groups_is_its_default() -> None:
    opt, _p = an_optimizer(2, lr=3e-4)
    opt.param_groups.clear()

    assert opt.get_avg_learning_rate() == pytest.approx(3e-4)
