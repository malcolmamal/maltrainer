"""Always-on modules must be trainable: an fp8 layer among them is refused, never silently skipped."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from maltrainer.models.comfy_fp8 import Fp8ScaledLinear
from maltrainer.models.krea2.finetune import always_on_linears


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
