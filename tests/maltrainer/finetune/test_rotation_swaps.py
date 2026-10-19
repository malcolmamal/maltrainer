from __future__ import annotations

import torch
from torch import nn

from maltrainer.models.comfy_fp8 import Fp8ScaledLinear, dequantize
from tests.maltrainer.finetune.rotation_model import ATTN, MLP, Optimizers, rotator


def test_activating_swaps_in_trainable_linears_from_the_master() -> None:
    optimizers = Optimizers()
    windows = rotator(make_optimizer=optimizers)
    mlp = windows.model.blocks[0].mlp

    windows.activate(0)

    wq = windows.model.blocks[0].attn.wq
    assert type(wq) is nn.Linear
    assert (wq.weight.requires_grad, wq.weight.dtype, wq.bias) == (True, torch.float32, None)
    torch.testing.assert_close(wq.weight.detach(), ATTN)
    torch.testing.assert_close(windows.model.blocks[1].attn.wk.weight.detach(), ATTN * 2)
    assert windows.model.blocks[0].mlp is mlp
    name, params, optimizer = optimizers.made[0]
    assert (name, windows.optimizer, windows.current) == ("attn", optimizer, 0)
    assert params[0] is wq.weight
    assert len(params) == 4


def test_activating_the_active_window_changes_nothing() -> None:
    optimizers = Optimizers()
    windows = rotator(make_optimizer=optimizers)
    windows.activate(0)
    wq = windows.model.blocks[0].attn.wq

    windows.activate(0)

    assert len(optimizers.made) == 1
    assert windows.model.blocks[0].attn.wq is wq


def test_a_biased_linear_keeps_its_bias_frozen() -> None:
    windows = rotator()
    bias = windows.model.blocks[0].mlp.bias.detach().clone()

    windows.activate(1)

    mlp = windows.model.blocks[0].mlp
    assert mlp.bias.requires_grad is False
    assert torch.equal(mlp.bias, bias)
    assert [p.shape for p in windows.optimizer.param_groups[0]["params"]] == [(3, 2), (3, 2)]


def test_deactivating_writes_the_trained_weights_back_and_freezes_them_to_fp8() -> None:
    windows = rotator()
    windows.activate(0)
    with torch.no_grad():
        windows.model.blocks[0].attn.wq.weight.mul_(2)

    windows.deactivate()

    wq = windows.model.blocks[0].attn.wq
    assert isinstance(wq, Fp8ScaledLinear)
    assert wq.weight.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(dequantize(wq.weight, wq.weight_scale, torch.float32), ATTN * 2)
    torch.testing.assert_close(wq(torch.eye(2)), (ATTN * 2).T)
    assert torch.equal(windows.master["blocks.0.attn.wq.weight"], (ATTN * 2).to(torch.bfloat16))
    assert torch.equal(windows.master["blocks.1.attn.wq.weight"], ATTN.to(torch.bfloat16))
    assert (windows.current, windows.optimizer, list(windows.optimizer_states)) == (None, None, [0])


def test_a_frozen_biased_linear_keeps_its_bias() -> None:
    windows = rotator()
    bias = windows.model.blocks[1].mlp.bias.detach().clone()
    windows.activate(1)

    windows.deactivate()

    mlp = windows.model.blocks[1].mlp
    assert isinstance(mlp, Fp8ScaledLinear)
    assert torch.equal(mlp.bias, bias)
    torch.testing.assert_close(mlp(torch.zeros(1, 2)), bias[None, :])
    torch.testing.assert_close(dequantize(mlp.weight, mlp.weight_scale, torch.float32), MLP)


def test_deactivating_with_no_active_window_does_nothing() -> None:
    windows = rotator()
    wq = windows.model.blocks[0].attn.wq

    windows.deactivate()

    assert (windows.model.blocks[0].attn.wq, windows.optimizer_states) == (wq, {})


def test_activating_another_window_freezes_the_active_one() -> None:
    windows = rotator()
    windows.activate(0)

    windows.activate(1)

    assert isinstance(windows.model.blocks[0].attn.wq, Fp8ScaledLinear)
    assert type(windows.model.blocks[0].mlp) is nn.Linear
    assert (windows.current, list(windows.optimizer_states)) == (1, [0])


def test_a_weight_reads_live_from_the_active_window_and_from_the_master_otherwise() -> None:
    windows = rotator()
    windows.activate(0)
    with torch.no_grad():
        windows.model.blocks[0].attn.wq.weight.add_(1)

    live, frozen = windows.weight("blocks.0.attn.wq.weight"), windows.weight("blocks.0.mlp.weight")

    assert torch.equal(live, (ATTN + 1).to(torch.bfloat16))
    assert frozen is windows.master["blocks.0.mlp.weight"]
