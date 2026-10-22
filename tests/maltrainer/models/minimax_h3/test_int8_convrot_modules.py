from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from torch.nn import functional as F

from maltrainer.models.minimax_h3.int8_convrot import (
    _quantize_rows,
    enable_int8_convrot,
    load_comfy_int8_convrot_state_dict,
    prepare_int8_convrot_modules,
    rotate_activation,
)

GROUP = 16


class _Net(nn.Module):
    def __init__(self, bias: bool = True) -> None:
        super().__init__()
        self.fc = nn.Linear(32, 8, bias=bias)
        self.other = nn.Linear(32, 4)
        self.norm = nn.LayerNorm(8)


def _state(prefix: str = "fc", shape: tuple[int, ...] = (8, 1), group: int = GROUP) -> dict[str, torch.Tensor]:
    return {
        f"{prefix}.scale_weight": torch.ones(shape),
        f"{prefix}.int8_convrot_groupsize": torch.tensor(group, dtype=torch.int32),
    }


def _prepared(net: _Net) -> _Net:
    prepare_int8_convrot_modules(net, _state())
    net.fc.weight.data = torch.zeros(8, 32, dtype=torch.int8)
    net.fc.int8_convrot_groupsize.fill_(GROUP)
    return net


def test_preparing_freezes_the_named_linears_and_registers_their_scale_and_group_buffers() -> None:
    net = _Net()

    registered = prepare_int8_convrot_modules(net, _state())

    assert registered == 1
    assert not net.fc.weight.requires_grad and not net.fc.bias.requires_grad
    assert net.fc.scale_weight.shape == (8, 1) and net.fc.scale_weight.dtype == torch.float32
    assert net.fc.int8_convrot_groupsize.dtype == torch.int32 and net.fc.int8_convrot_groupsize.shape == ()
    assert net.other.weight.requires_grad and not hasattr(net.other, "scale_weight")


def test_preparing_a_linear_without_a_bias_works() -> None:
    net = _Net(bias=False)

    assert prepare_int8_convrot_modules(net, _state()) == 1
    assert net.fc.bias is None


def test_a_layer_without_its_group_size_is_an_error() -> None:
    state = {"fc.scale_weight": torch.ones(8, 1)}

    with pytest.raises(ValueError, match=re.escape("missing fc.int8_convrot_groupsize")):
        prepare_int8_convrot_modules(_Net(), state)


@pytest.mark.parametrize("prefix", ["missing", "norm"])
def test_scales_for_layers_the_model_does_not_have_as_linears_are_an_error(prefix: str) -> None:
    state = {**_state(), **_state(prefix)}

    with pytest.raises(ValueError, match="targets 2 linears, but the model exposes 1"):
        prepare_int8_convrot_modules(_Net(), state)


def test_enabling_patches_only_the_quantized_linears_and_records_the_precision() -> None:
    net = _prepared(_Net())

    patched = enable_int8_convrot(net, "float32")

    assert patched == 1
    assert net.fc.int8_convrot_grad_precision == "fp32"
    assert "forward" in vars(net.fc) and "forward" not in vars(net.other)


def test_enabling_with_the_default_precision_uses_bf16_without_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    net = _prepared(_Net())

    with caplog.at_level(logging.INFO, logger="maltrainer.models.minimax_h3.int8_convrot"):
        enable_int8_convrot(net)

    assert net.fc.int8_convrot_grad_precision == "bf16"
    assert [record.levelname for record in caplog.records] == ["INFO"]
    assert "(backward precision: bf16)" in caplog.text


def test_the_int8_gradient_precision_warns_that_it_fits_quantization_noise(caplog: pytest.LogCaptureFixture) -> None:
    net = _prepared(_Net())

    with caplog.at_level(logging.INFO, logger="maltrainer.models.minimax_h3.int8_convrot"):
        enable_int8_convrot(net, "int8")

    assert [record.levelname for record in caplog.records] == ["INFO", "WARNING"]
    assert "every one of the 1 layers" in caplog.text


def test_an_unknown_precision_is_rejected_before_any_layer_is_patched() -> None:
    net = _Net()
    prepare_int8_convrot_modules(net, _state())

    with pytest.raises(ValueError, match="Unknown INT8 ConvRot gradient precision"):
        enable_int8_convrot(net, "fp8")
    assert "forward" not in vars(net.fc)


def test_a_scale_on_a_float_weight_is_an_error() -> None:
    net = _Net()
    prepare_int8_convrot_modules(net, _state())

    with pytest.raises(TypeError, match="non-INT8 Linear weight"):
        enable_int8_convrot(net)


def test_a_group_size_that_does_not_divide_the_input_features_is_an_error() -> None:
    net = _prepared(_Net())
    net.fc.int8_convrot_groupsize.fill_(64)

    with pytest.raises(ValueError, match="group size 64 does not divide 32 input features"):
        enable_int8_convrot(net)


def _convrot_file(tmp_path: Path, weight: torch.Tensor, bias: torch.Tensor) -> str:
    quantized, scale = _quantize_rows(rotate_activation(weight, GROUP))
    marker = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GROUP})
    tensors = {
        "fc.weight": quantized,
        "fc.weight_scale": scale,
        "fc.comfy_quant": torch.tensor(list(marker.encode()), dtype=torch.uint8),
        "fc.bias": bias,
    }
    path = tmp_path / "net.safetensors"
    save_file(tensors, str(path))
    return str(path)


def test_a_checkpoint_loaded_through_the_loader_chain_runs_like_the_float_layer(tmp_path: Path) -> None:
    torch.manual_seed(0)
    weight, bias, x = torch.randn(8, 32), torch.randn(8), torch.randn(5, 32, requires_grad=True)
    net = _Net()
    state, count = load_comfy_int8_convrot_state_dict(_convrot_file(tmp_path, weight, bias), device=torch.device("cpu"))
    state = {key: value for key, value in state.items() if key.startswith("fc.")}

    prepare_int8_convrot_modules(net, state)
    net.load_state_dict(state, strict=False, assign=True)
    enable_int8_convrot(net, "fp32")
    out = net.fc(x)
    out.sum().backward()

    assert count == 1 and net.fc.weight.dtype == torch.int8
    torch.testing.assert_close(out, F.linear(x, weight, bias), rtol=0, atol=0.25)
    torch.testing.assert_close(x.grad, weight.sum(dim=0).expand(5, 32), rtol=0, atol=0.4)
