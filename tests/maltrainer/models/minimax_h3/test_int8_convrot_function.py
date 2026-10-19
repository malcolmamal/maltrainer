from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from maltrainer.models.minimax_h3.int8_convrot import (
    _Int8ConvRotFunction,
    _quantize_rows,
    _resolve_grad_precision,
    _TransientInt8ConvRotFunction,
    rotate_activation,
)

GROUP = 16


def _layer(out: int = 8, features: int = 32, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    weight = torch.randn(out, features)
    quantized, scale = _quantize_rows(rotate_activation(weight, GROUP))
    return weight, quantized, scale


def _dense(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return quantized.float() * scale


def _reference(x: torch.Tensor, quantized: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None):
    return F.linear(rotate_activation(x, GROUP), _dense(quantized, scale), bias)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, "bf16"), ("", "bf16"), ("FP32", "fp32"), ("float32", "fp32"), ("bfloat16", "bf16"), ("int8", "int8")],
)
def test_gradient_precision_names_and_aliases_resolve_to_canonical_names(value: str | None, expected: str) -> None:
    assert _resolve_grad_precision(value) == expected


def test_an_unknown_gradient_precision_lists_the_valid_ones() -> None:
    with pytest.raises(ValueError, match=r"Unknown INT8 ConvRot gradient precision 'fp8'.*'bf16', 'fp32', 'int8'"):
        _resolve_grad_precision("fp8")


def test_rotating_the_weight_offline_and_the_input_at_run_time_cancels_in_the_product() -> None:
    weight, quantized, scale = _layer()
    x = torch.randn(5, 32)

    out = _TransientInt8ConvRotFunction.apply(x, quantized, scale, None, GROUP, None)

    torch.testing.assert_close(out, _reference(x, quantized, scale, None), rtol=1e-5, atol=1e-5)
    assert float((out - F.linear(x, weight)).abs().max()) < 0.1


def test_the_eager_fallback_keeps_the_input_dtype_and_adds_the_bias() -> None:
    _, quantized, scale = _layer()
    x, bias = torch.randn(2, 3, 32).bfloat16(), torch.randn(8)

    out = _TransientInt8ConvRotFunction.apply(x, quantized, scale, bias, GROUP, None)

    assert out.dtype == torch.bfloat16 and out.shape == (2, 3, 8)
    torch.testing.assert_close(out.float(), _reference(x.float(), quantized, scale, bias), rtol=3e-2, atol=0.1)


def test_the_eager_fallbacks_input_gradient_is_that_of_the_dequantized_linear() -> None:
    _, quantized, scale = _layer()
    x = torch.randn(4, 32, requires_grad=True)
    grad_out = torch.randn(4, 8)

    _TransientInt8ConvRotFunction.apply(x, quantized, scale, None, GROUP, None).backward(grad_out)

    expected = torch.autograd.grad(_reference(x, quantized, scale, None), x, grad_out)[0]
    torch.testing.assert_close(x.grad, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("rows", [3, 40])
def test_the_int8_forward_is_within_the_input_quantization_error_of_the_dequantized_linear(rows: int) -> None:
    _, quantized, scale = _layer()
    x, bias = torch.randn(rows, 32), torch.randn(8)

    out = _Int8ConvRotFunction.apply(x, quantized, scale, bias, GROUP, "bf16")

    exact = _reference(x, quantized, scale, bias)
    _, input_scale = _quantize_rows(rotate_activation(x, GROUP))
    bound = input_scale / 2 * _dense(quantized, scale).abs().sum(dim=1) + 1e-4
    assert out.shape == (rows, 8)
    assert bool(((out - exact).abs() <= bound).all())


def test_the_int8_forward_accepts_batched_inputs_without_a_bias() -> None:
    _, quantized, scale = _layer()
    x = torch.randn(2, 5, 32)

    out = _Int8ConvRotFunction.apply(x, quantized, scale, None, GROUP, "fp32")

    assert out.shape == (2, 5, 8)
    torch.testing.assert_close(out, _reference(x, quantized, scale, None), rtol=0.05, atol=0.2)


@pytest.mark.parametrize(("precision", "tolerance"), [("fp32", 1e-5), ("bf16", 2e-2), ("int8", 5e-2)])
def test_the_input_gradient_matches_autograd_through_the_dequantized_weights(precision: str, tolerance: float) -> None:
    _, quantized, scale = _layer()
    x = torch.randn(6, 32, requires_grad=True)
    grad_out = torch.randn(6, 8)

    _Int8ConvRotFunction.apply(x, quantized, scale, None, GROUP, precision).backward(grad_out)

    expected = torch.autograd.grad(_reference(x, quantized, scale, None), x, grad_out)[0]
    torch.testing.assert_close(x.grad, expected, rtol=0, atol=tolerance * float(expected.abs().max()))


def test_the_input_gradient_has_the_input_dtype_and_shape() -> None:
    _, quantized, scale = _layer()
    x = torch.randn(2, 3, 32).bfloat16().requires_grad_()

    _Int8ConvRotFunction.apply(x, quantized, scale, None, GROUP, "bf16").sum().backward()

    assert x.grad.dtype == torch.bfloat16 and x.grad.shape == x.shape


def test_a_backward_over_more_rows_than_one_dequantization_chunk_accumulates_every_chunk() -> None:
    generator = torch.Generator().manual_seed(4)
    out_features = 2**18 + 8
    quantized = torch.randint(-127, 128, (out_features, 16), dtype=torch.int8, generator=generator)
    scale = torch.rand(out_features, 1, generator=generator) / 100
    x = torch.randn(2, 16, requires_grad=True)
    grad_out = torch.randn(2, out_features)

    _Int8ConvRotFunction.apply(x, quantized, scale, None, GROUP, "fp32").backward(grad_out)

    expected = rotate_activation((grad_out * scale.t()) @ quantized.float(), GROUP)
    torch.testing.assert_close(x.grad, expected, rtol=1e-3, atol=1e-3)
