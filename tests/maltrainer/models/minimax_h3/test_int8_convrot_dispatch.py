from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from maltrainer.models.minimax_h3 import int8_convrot
from maltrainer.models.minimax_h3.int8_convrot import _int8_linear_forward, _quantize_rows, rotate_activation

GROUP = 16


def _linear(group: int = GROUP, precision: str | None = None) -> nn.Linear:
    torch.manual_seed(0)
    quantized, scale = _quantize_rows(rotate_activation(torch.randn(8, 32), GROUP))
    module = nn.Linear(32, 8, bias=True)
    module.weight = nn.Parameter(quantized, requires_grad=False)
    module.register_buffer("scale_weight", scale)
    module.register_buffer("int8_convrot_groupsize", torch.tensor(group, dtype=torch.int32))
    if precision is not None:
        module.int8_convrot_grad_precision = precision
    return module


def test_the_patched_linear_runs_the_eager_fallback_on_cpu() -> None:
    module = _linear()
    x = torch.randn(3, 32)

    out = _int8_linear_forward(module, x)

    expected = F.linear(rotate_activation(x, GROUP), module.weight.float() * module.scale_weight, module.bias)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


class _Recorder:
    calls: list[tuple]

    @classmethod
    def apply(cls, *args):
        cls.calls.append(args)
        return "recorded"


@pytest.mark.parametrize(("has_int_mm", "uses_int8"), [(True, True), (False, False)])
def test_cuda_inputs_use_the_int8_kernel_only_when_torch_provides_it(
    monkeypatch: pytest.MonkeyPatch, has_int_mm: bool, uses_int8: bool
) -> None:
    int8_path, eager_path = type("Int8", (_Recorder,), {"calls": []}), type("Eager", (_Recorder,), {"calls": []})
    monkeypatch.setattr(int8_convrot, "_Int8ConvRotFunction", int8_path)
    monkeypatch.setattr(int8_convrot, "_TransientInt8ConvRotFunction", eager_path)
    if not has_int_mm:
        monkeypatch.delattr(torch, "_int_mm")
    module = _linear(precision="fp32")

    result = _int8_linear_forward(module, SimpleNamespace(is_cuda=True))

    chosen = int8_path if uses_int8 else eager_path
    assert result == "recorded" and len(chosen.calls) == 1
    assert chosen.calls[0][1:] == (module.weight, module.scale_weight, module.bias, GROUP, "fp32")


def test_a_linear_without_a_gradient_precision_attribute_uses_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = type("Eager", (_Recorder,), {"calls": []})
    monkeypatch.setattr(int8_convrot, "_TransientInt8ConvRotFunction", recorder)

    _int8_linear_forward(_linear(), torch.zeros(1, 32))

    assert recorder.calls[0][-1] == "bf16"
