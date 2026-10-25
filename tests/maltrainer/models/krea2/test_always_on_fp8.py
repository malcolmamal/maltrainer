"""Always-on modules must be trainable: an fp8 layer outside the windows is dequantized to bf16, one the windows
rotate is refused, never silently skipped."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from maltrainer.models.comfy_fp8 import Fp8ScaledLinear
from maltrainer.models.krea2.finetune import always_on_linears, dequantize_always_on


class _Model(nn.Module):
    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.txt = nn.ModuleList([layer])


def test_an_fp8_scaled_linear_among_the_always_on_modules_is_refused() -> None:
    model = _Model(Fp8ScaledLinear(4, 4, bias=False))

    with pytest.raises(ValueError, match=r"txt\.0 is fp8; always-on modules must be bf16"):
        always_on_linears(model, ["txt"])


def test_a_linear_holding_fp8_weights_is_refused_too() -> None:
    layer = nn.Linear(4, 4, bias=False)
    layer.weight = nn.Parameter(torch.zeros(4, 4).to(torch.float8_e4m3fn), requires_grad=False)

    with pytest.raises(ValueError, match=r"txt\.0 is fp8"):
        always_on_linears(_Model(layer), ["txt"])


def test_plain_linears_are_made_trainable_and_listed() -> None:
    model = _Model(nn.Linear(4, 4, bias=False))
    model.txt[0].weight.requires_grad_(False)

    listed = always_on_linears(model, ["txt"])

    assert [name for name, _ in listed] == ["txt.0.weight"]
    assert listed[0][1].requires_grad


def _fp8(values: torch.Tensor, scale: float, bias: bool = False) -> Fp8ScaledLinear:
    layer = Fp8ScaledLinear(values.shape[1], values.shape[0], bias=bias, compute_dtype=torch.bfloat16)
    layer.weight = (values / scale).to(torch.float8_e4m3fn)
    layer.weight_scale = torch.tensor(scale)
    if bias:
        layer.bias = torch.tensor([0.5, -0.5], dtype=torch.bfloat16)
    return layer


def test_an_fp8_linear_outside_the_windows_is_dequantized_into_a_trainable_bf16_linear() -> None:
    values = torch.tensor([[1.0, 0.5], [-2.0, 0.25]])
    model = _Model(_fp8(values, 0.5, bias=True))

    swapped = dequantize_always_on(model, ["txt"], rotating=set())
    listed = always_on_linears(model, ["txt"])

    assert swapped == 1 and type(model.txt[0]) is nn.Linear
    assert torch.equal(model.txt[0].weight.detach(), values.to(torch.bfloat16))  # fp8 holds these exactly
    assert model.txt[0].bias is not None and not model.txt[0].bias.requires_grad
    assert [name for name, _ in listed] == ["txt.0.weight"] and listed[0][1].requires_grad


def test_an_fp8_linear_the_windows_rotate_stays_fp8_and_is_refused() -> None:
    model = _Model(_fp8(torch.eye(2), 1.0))

    assert dequantize_always_on(model, ["txt"], rotating={"txt.0.weight"}) == 0
    with pytest.raises(ValueError, match=r"txt\.0 is fp8"):
        always_on_linears(model, ["txt"])
