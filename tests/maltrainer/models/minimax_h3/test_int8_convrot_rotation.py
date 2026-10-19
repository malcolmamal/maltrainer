from __future__ import annotations

import json

import pytest
import torch

from maltrainer.models.minimax_h3 import int8_convrot
from maltrainer.models.minimax_h3.int8_convrot import (
    _hadamard,
    _int_mm,
    _is_power_of_four,
    _quantize_rows,
    parse_comfy_quant_marker,
    rotate_activation,
)

CPU = torch.device("cpu")
H4 = torch.tensor([[1.0, 1.0, 1.0, -1.0], [1.0, 1.0, -1.0, 1.0], [1.0, -1.0, 1.0, 1.0], [-1.0, 1.0, 1.0, 1.0]]) / 2


def _blob(payload: bytes) -> torch.Tensor:
    return torch.tensor(list(payload), dtype=torch.uint8)


def test_a_marker_tensor_holds_the_layers_json_config() -> None:
    config = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}

    parsed = parse_comfy_quant_marker(_blob(json.dumps(config).encode()).reshape(1, -1))

    assert parsed == config


@pytest.mark.parametrize(
    ("blob", "message"),
    [
        (_blob(b"\xff\xfe"), "Invalid Comfy quantization marker"),
        (_blob(b"{not json"), "Invalid Comfy quantization marker"),
        (torch.tensor([-1, 5], dtype=torch.int8), "Invalid Comfy quantization marker"),
        (_blob(b"[1, 2]"), "must contain a JSON object"),
    ],
)
def test_a_marker_that_is_not_a_json_object_is_rejected(blob: torch.Tensor, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_comfy_quant_marker(blob)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (4, True),
        (16, True),
        (64, True),
        (256, True),
        (-4, False),
        (0, False),
        (1, False),
        (2, False),
        (8, False),
        (32, False),
    ],
)
def test_group_sizes_must_be_powers_of_four_of_at_least_four(value: int, expected: bool) -> None:
    assert _is_power_of_four(value) is expected


def test_the_four_point_hadamard_is_the_hand_written_matrix_over_two() -> None:
    torch.testing.assert_close(_hadamard(4, CPU, torch.float32), H4)


@pytest.mark.parametrize("size", [4, 16, 64])
def test_the_rotation_matrix_is_symmetric_and_orthogonal(size: int) -> None:
    matrix = _hadamard(size, CPU, torch.float64)

    torch.testing.assert_close(matrix, matrix.t())
    torch.testing.assert_close(matrix @ matrix.t(), torch.eye(size, dtype=torch.float64))


def test_larger_rotations_are_kronecker_products_of_the_four_point_matrix() -> None:
    torch.testing.assert_close(_hadamard(16, CPU, torch.float32), torch.kron(H4, H4))


def test_a_rotation_matrix_is_built_once_per_size_device_and_dtype() -> None:
    first = _hadamard(16, CPU, torch.float32)

    assert _hadamard(16, CPU, torch.float32) is first
    assert _hadamard(16, CPU, torch.bfloat16) is not first
    assert _hadamard(16, CPU, torch.bfloat16).dtype == torch.bfloat16


@pytest.mark.parametrize("size", [3, 8, 12, 0])
def test_a_group_size_that_is_not_a_power_of_four_is_an_error(size: int) -> None:
    with pytest.raises(ValueError, match=f"power of four, got {size}"):
        _hadamard(size, CPU, torch.float32)


def test_each_group_of_features_is_rotated_independently() -> None:
    torch.manual_seed(0)
    value = torch.randn(3, 8)

    rotated = rotate_activation(value, 4)

    torch.testing.assert_close(rotated[:, :4], value[:, :4] @ H4)
    torch.testing.assert_close(rotated[:, 4:], value[:, 4:] @ H4)


@pytest.mark.parametrize(("shape", "group"), [((5, 16), 16), ((2, 3, 64), 16), ((2, 3, 64), 64), ((7, 8), 4)])
def test_rotating_twice_restores_the_input_and_keeps_group_norms(shape: tuple[int, ...], group: int) -> None:
    torch.manual_seed(1)
    value = torch.randn(shape, dtype=torch.float64)

    once = rotate_activation(value, group)

    assert once.shape == value.shape
    torch.testing.assert_close(rotate_activation(once, group), value)
    torch.testing.assert_close(once.reshape(-1, group).norm(dim=-1), value.reshape(-1, group).norm(dim=-1))


def test_a_group_size_that_does_not_divide_the_features_is_an_error() -> None:
    with pytest.raises(ValueError, match="group size 16 does not divide 24 input features"):
        rotate_activation(torch.ones(2, 24), 16)


def test_rows_are_quantized_to_int8_against_their_largest_magnitude() -> None:
    value = torch.tensor([[127.0, -50.0, 3.0], [0.0, 0.0, 0.0], [-2.0, 0.5, 0.0]])

    quantized, scale = _quantize_rows(value)

    assert quantized.dtype == torch.int8
    torch.testing.assert_close(scale, torch.tensor([[1.0], [1e-30], [2.0 / 127]]))
    assert quantized.tolist() == [[127, -50, 3], [0, 0, 0], [-127, 32, 0]]


def test_dequantizing_a_row_is_within_half_a_step_of_the_original() -> None:
    torch.manual_seed(2)
    value = torch.randn(6, 32) * torch.logspace(-2, 2, 6).unsqueeze(1)

    quantized, scale = _quantize_rows(value.bfloat16())

    assert bool(((quantized.float() * scale - value.bfloat16().float()).abs() <= scale / 2 + 1e-6).all())


@pytest.mark.parametrize("rows", [1, 5, 16, 17, 40])
def test_int_mm_matches_an_integer_reference_for_small_and_large_row_counts(rows: int) -> None:
    generator = torch.Generator().manual_seed(3)
    left = torch.randint(-127, 128, (rows, 16), dtype=torch.int8, generator=generator)
    right = torch.randint(-127, 128, (16, 8), dtype=torch.int8, generator=generator)

    out = _int_mm(left, right)

    assert out.dtype == torch.int32
    assert torch.equal(out, left.int() @ right.int())


def test_int_mm_without_torch_support_asks_for_a_cuda_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(torch, "_int_mm")

    with pytest.raises(RuntimeError, match="requires torch._int_mm"):
        int8_convrot._int_mm(torch.zeros(2, 4, dtype=torch.int8), torch.zeros(4, 2, dtype=torch.int8))
