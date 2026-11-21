"""Decoding an INT8 ConvRot weight back to a dense one: codes times the row scale, with the input rotation undone."""

from __future__ import annotations

import pytest
import torch

from maltrainer.models.minimax_h3.int8_rotation import _quantize_rows, dequantize_weight, rotate_activation

GROUP = 16


def _weight(rows: int = 8, columns: int = 32, seed: int = 0) -> torch.Tensor:
    return torch.randn(rows, columns, generator=torch.Generator().manual_seed(seed))


def test_a_decoded_weight_is_the_rotated_codes_times_the_row_scale() -> None:
    codes = torch.randint(-127, 128, (4, 32), dtype=torch.int8, generator=torch.Generator().manual_seed(1))
    scale = torch.tensor([[0.5], [1.0], [2.0], [0.25]])

    decoded = dequantize_weight(codes, scale, GROUP, torch.float32)

    torch.testing.assert_close(decoded, rotate_activation(codes.float() * scale, GROUP))


def test_encoding_then_decoding_gives_the_weight_back_to_int8_precision() -> None:
    weight = _weight()
    codes, scale = _quantize_rows(rotate_activation(weight, GROUP))

    decoded = dequantize_weight(codes, scale, GROUP, torch.float32)

    assert (decoded - weight).norm() / weight.norm() < 0.01


def test_the_decoded_weight_computes_what_the_int8_runtime_computes() -> None:
    """The runtime rotates the activation and multiplies by the codes; the decoded weight must agree with that."""
    weight, x = _weight(), _weight(5, 32, seed=3)
    codes, scale = _quantize_rows(rotate_activation(weight, GROUP))

    runtime = (rotate_activation(x, GROUP) @ codes.float().T) * scale.T
    decoded = x @ dequantize_weight(codes, scale, GROUP, torch.float32).T

    torch.testing.assert_close(decoded, runtime, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(8,), (8, 1)])
def test_the_scale_may_be_a_vector_or_a_column(shape: tuple[int, ...]) -> None:
    codes, scale = _quantize_rows(rotate_activation(_weight(), GROUP))

    decoded = dequantize_weight(codes, scale.reshape(shape), GROUP, torch.bfloat16)

    assert decoded.dtype == torch.bfloat16 and decoded.shape == codes.shape
