from __future__ import annotations

import pytest
import torch
from torch import nn

from maltrainer.models.comfy_fp8 import (
    FP8_MAX,
    Fp8ScaledLinear,
    dequantize,
    dequantize_checkpoint,
    quantize_rows,
    swap_in_fp8_linears,
)

W8 = torch.tensor([[1.0, -2.0, 0.5], [4.0, 0.25, -1.0]]).to(torch.float8_e4m3fn)


def _linear(compute_dtype: torch.dtype = torch.float32, scale: torch.Tensor | None = None) -> Fp8ScaledLinear:
    layer = Fp8ScaledLinear(3, 2, bias=True, compute_dtype=compute_dtype)
    layer.weight.copy_(W8)
    layer.weight_scale = torch.tensor(0.5) if scale is None else scale
    layer.bias.copy_(torch.tensor([1.0, -1.0]))
    return layer


def test_rows_are_scaled_so_their_largest_magnitude_maps_to_fp8_max() -> None:
    weight = torch.tensor([[1.0, -2.0], [0.5, 0.25], [0.0, 0.0]], dtype=torch.bfloat16)

    fp8, scale = quantize_rows(weight)

    assert FP8_MAX == 448.0
    assert fp8.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(scale, torch.tensor([[2.0 / 448], [0.5 / 448], [1e-12]]))
    torch.testing.assert_close(fp8.float(), torch.tensor([[224.0, -448.0], [448.0, 224.0], [0.0, 0.0]]))


def test_a_quantize_dequantize_round_trip_stays_within_fp8_precision() -> None:
    torch.manual_seed(0)
    weight = torch.randn(8, 16) * torch.logspace(-3, 2, 8).unsqueeze(1)

    fp8, scale = quantize_rows(weight)
    restored = dequantize(fp8, scale, torch.float32)

    assert bool(((restored - weight).abs() <= weight.abs() / 16 + scale * 2**-9).all())


def test_dequantize_multiplies_by_a_per_tensor_scale_into_the_requested_dtype() -> None:
    restored = dequantize(W8, torch.tensor(0.5), torch.bfloat16)

    assert restored.dtype == torch.bfloat16
    torch.testing.assert_close(restored.float(), torch.tensor([[0.5, -1.0, 0.25], [2.0, 0.125, -0.5]]))


def test_a_checkpoint_is_dequantized_and_cast_while_integer_tensors_are_kept() -> None:
    tensors = {
        "blk.weight": W8,
        "blk.weight_scale": torch.tensor(2.0),
        "unscaled.weight": W8,
        "ids": torch.arange(3),
        "norm.weight": torch.ones(2),
    }
    layers = {"blk": {"format": "float8_e4m3fn"}, "unscaled": {}, "absent": {}}

    out = dequantize_checkpoint(tensors, layers, torch.bfloat16)

    assert sorted(out) == ["blk.weight", "ids", "norm.weight", "unscaled.weight"]
    torch.testing.assert_close(out["blk.weight"], (W8.float() * 2).to(torch.bfloat16))
    torch.testing.assert_close(out["unscaled.weight"], W8.to(torch.bfloat16))
    assert out["ids"].dtype == torch.int64
    assert out["norm.weight"].dtype == torch.bfloat16
    assert "blk.weight_scale" in tensors


def test_the_fp8_linear_computes_x_times_the_dequantized_weight_plus_bias() -> None:
    layer = _linear()

    y = layer(torch.tensor([[1.0, 2.0, 3.0]]))

    torch.testing.assert_close(y, torch.tensor([[0.25, -0.25]]))


def test_a_per_row_scale_scales_each_output_row() -> None:
    layer = _linear(scale=torch.tensor([[1.0], [0.25]]))

    y = layer(torch.tensor([[1.0, 0.0, 0.0]]))

    torch.testing.assert_close(y, torch.tensor([[2.0, 0.0]]))


def test_the_input_gradient_uses_the_dequantized_weight_in_the_input_dtype() -> None:
    layer = _linear(torch.bfloat16)
    x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)

    y = layer(x)
    y.sum().backward()

    assert y.dtype == torch.bfloat16
    assert x.grad.dtype == torch.float32
    torch.testing.assert_close(x.grad, torch.tensor([[2.5, -0.875, -0.25]]))


def test_the_fp8_linear_holds_frozen_buffers_and_no_parameters() -> None:
    layer = Fp8ScaledLinear(3, 2, bias=False)

    assert list(layer.parameters()) == []
    assert sorted(name for name, _ in layer.named_buffers()) == ["weight", "weight_scale"]
    assert layer.bias is None
    assert layer.extra_repr() == "in_features=3, out_features=2, fp8 scaled"


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(3, 2)])
        self.head = nn.Linear(2, 4, bias=False)
        self.norm = nn.LayerNorm(2)


def test_named_linears_are_swapped_for_fp8_linears_of_the_same_shape() -> None:
    model = _Model()

    swapped = swap_in_fp8_linears(model, {"blocks.0": {}, "head": {}}, torch.bfloat16)

    block, head = model.blocks[0], model.head
    assert swapped == 2
    assert isinstance(block, Fp8ScaledLinear)
    assert isinstance(head, Fp8ScaledLinear)
    assert (block.in_features, block.out_features, block.bias.dtype) == (3, 2, torch.bfloat16)
    assert (head.in_features, head.out_features, head.bias) == (2, 4, None)
    assert block.weight_scale.shape == ()


def test_a_key_map_and_per_row_scales_let_a_strict_load_fill_the_swapped_layer() -> None:
    model = _Model()
    tensors = {"dm.blocks.0.weight": W8, "dm.blocks.0.weight_scale": torch.tensor([[1.0], [0.25]])}

    swap_in_fp8_linears(model, {"dm.blocks.0": {}}, torch.float32, lambda key: key[3:], tensors)
    model.blocks[0].load_state_dict(
        {"weight": W8, "weight_scale": tensors["dm.blocks.0.weight_scale"], "bias": torch.zeros(2)}, assign=True
    )

    torch.testing.assert_close(model.blocks[0](torch.tensor([[1.0, 0.0, 0.0]])), torch.tensor([[1.0, 1.0]]))


@pytest.mark.parametrize(
    ("prefix", "error", "message"),
    [("missing", ValueError, "has no module missing"), ("norm", TypeError, "maps to LayerNorm, expected Linear")],
)
def test_a_quantized_layer_that_is_not_a_linear_in_the_model_is_an_error(
    prefix: str, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        swap_in_fp8_linears(_Model(), {prefix: {}}, torch.bfloat16)
