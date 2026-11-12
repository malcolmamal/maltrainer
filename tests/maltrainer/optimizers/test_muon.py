"""BatchedMuon: PyTorch's Muon update, with the matrices of one shape orthogonalised together."""

from __future__ import annotations

import pytest
import torch

from maltrainer.optimizers.muon import BatchedMuon

SHAPES = [(8, 3), (3, 8), (4, 4), (8, 3), (6, 6)]  # tall, wide, square, a repeat, another square


def _params(seed: int = 0) -> list[torch.nn.Parameter]:
    generator = torch.Generator().manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(shape, generator=generator)) for shape in SHAPES]


def _give_grads(params: list[torch.nn.Parameter], seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    for parameter in params:
        parameter.grad = torch.randn(parameter.shape, generator=generator)


@pytest.mark.parametrize("adjust", [None, "match_rms_adamw", "spectral_unclamped"])
@pytest.mark.parametrize("nesterov", [True, False])
def test_it_takes_the_same_steps_as_torch_optim_muon(adjust: str | None, nesterov: bool) -> None:
    ours, theirs = _params(), _params()
    options = {"lr": 1e-2, "weight_decay": 0.05, "momentum": 0.9, "nesterov": nesterov, "adjust_lr_fn": adjust}
    ours_opt, theirs_opt = BatchedMuon(ours, **options), torch.optim.Muon(theirs, **options)
    start = [p.detach().clone() for p in ours]

    for step in range(3):
        _give_grads(ours, step), _give_grads(theirs, step)
        ours_opt.step(), theirs_opt.step()

    for mine, reference, before in zip(ours, theirs, start, strict=True):
        moved = (reference - before).norm()
        assert moved > 0 and (mine - reference).norm() / moved < 0.02  # Newton-Schulz runs in bf16 either way


def test_a_matrix_without_a_gradient_stays_put_and_gets_no_state() -> None:
    params = _params()
    optimizer = BatchedMuon(params, lr=1e-2)
    _give_grads(params, 0)
    params[1].grad = None
    before = params[1].detach().clone()

    optimizer.step()

    assert torch.equal(params[1], before) and params[1] not in optimizer.state
    assert all(p in optimizer.state for i, p in enumerate(params) if i != 1)


def test_a_group_with_no_gradients_is_skipped() -> None:
    optimizer = BatchedMuon(_params(), lr=1e-2)

    optimizer.step()

    assert not optimizer.state


def test_the_closure_is_run_and_its_loss_returned() -> None:
    params = _params()
    optimizer = BatchedMuon(params, lr=1e-2)

    def closure() -> torch.Tensor:
        _give_grads(params, 0)
        return torch.tensor(1.5)

    assert optimizer.step(closure) == 1.5


def test_only_matrices_are_accepted() -> None:
    with pytest.raises(ValueError, match=r"2D parameters only, got \(3,\)"):
        BatchedMuon([torch.nn.Parameter(torch.zeros(3))])


def test_an_unknown_learning_rate_adjustment_is_refused() -> None:
    with pytest.raises(ValueError, match="adjust_lr_fn 'sideways'"):
        BatchedMuon(_params(), adjust_lr_fn="sideways")
