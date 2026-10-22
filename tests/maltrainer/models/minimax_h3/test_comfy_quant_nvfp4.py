from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from maltrainer.models.minimax_h3.comfy_quant import ComfyNvfp4Linear, get_nvfp4_lut, unswizzle_nvfp4_scales

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
OUT, FEATURES = 4, 32


def swizzle(matrix: torch.Tensor) -> torch.Tensor:
    """The cuBLAS 128x4 block layout: element (r, c) of a block sits at (r % 32) * 16 + (r // 32) * 4 + c."""
    rows, columns = matrix.shape
    blocked = torch.empty(matrix.numel(), dtype=matrix.dtype)
    r, c = torch.arange(128)[:, None], torch.arange(4)[None, :]
    inside = (r % 32) * 16 + (r // 32) * 4 + c
    for row_block in range(rows // 128):
        for column_block in range(columns // 4):
            base = (row_block * (columns // 4) + column_block) * 512
            tile = matrix[row_block * 128 : (row_block + 1) * 128, column_block * 4 : (column_block + 1) * 4]
            blocked[base + inside] = tile
    return blocked


def fp4_value(code: int) -> float:
    return -E2M1[code & 7] if code & 8 else E2M1[code & 7]


@pytest.mark.parametrize(("rows", "columns"), [(128, 4), (256, 8), (130, 5), (3, 2)])
def test_blocked_scales_are_restored_to_row_major_order(rows: int, columns: int) -> None:
    original = torch.arange(rows * columns, dtype=torch.float32).reshape(rows, columns) + 1
    padded = torch.zeros(-(-rows // 128) * 128, -(-columns // 4) * 4)
    padded[:rows, :columns] = original

    restored = unswizzle_nvfp4_scales(swizzle(padded), rows, columns)

    assert torch.equal(restored, original)


def test_a_scale_tensor_of_the_wrong_size_is_an_error() -> None:
    with pytest.raises(ValueError, match="expected 512 elements for 100x3, got 500"):
        unswizzle_nvfp4_scales(torch.zeros(500), 100, 3)


def test_every_byte_decodes_to_its_high_nibble_then_its_low_nibble() -> None:
    lut = get_nvfp4_lut(torch.device("cpu"), torch.float32)

    expected = torch.tensor([[fp4_value(byte >> 4), fp4_value(byte & 15)] for byte in range(256)])
    assert lut.shape == (256, 2)
    assert torch.equal(lut, expected)


def test_a_decoding_table_is_built_once_per_device_and_dtype() -> None:
    first = get_nvfp4_lut(torch.device("cpu"), torch.bfloat16)

    assert get_nvfp4_lut(torch.device("cpu"), torch.bfloat16) is first
    assert first.dtype == torch.bfloat16
    assert get_nvfp4_lut(torch.device("cpu"), torch.float32) is not first


def _parts(seed: int = 0, features: int = FEATURES) -> dict:
    generator = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (OUT, features), dtype=torch.uint8, generator=generator)
    values = torch.tensor([[fp4_value(int(code)) for code in row] for row in codes])
    blocks = (torch.rand(OUT, features // 16, generator=generator) + 0.5).to(torch.float8_e4m3fn)
    padded = torch.zeros(128, 4)
    padded[:OUT, : features // 16] = blocks.float()
    return {
        "packed": (codes[:, 0::2] << 4) | codes[:, 1::2],
        "blocked": swizzle(padded).to(torch.float8_e4m3fn),
        "per_tensor": torch.tensor(0.25),
        "weight": values * blocks.float().repeat_interleave(16, dim=1) * 0.25,
    }


def _layer(parts: dict, pre_quant: torch.Tensor | None = None, bias: bool = True) -> ComfyNvfp4Linear:
    source = nn.Linear(FEATURES, OUT, bias=bias)
    return ComfyNvfp4Linear(source, parts["packed"], parts["blocked"], parts["per_tensor"], pre_quant, torch.float32)


def test_the_dequantized_weight_is_codes_times_block_scale_times_the_tensor_scale() -> None:
    parts = _parts()

    weight = _layer(parts).dequantize_weight(torch.float32)

    torch.testing.assert_close(weight, parts["weight"])


def test_the_forward_is_a_linear_over_the_dequantized_weight() -> None:
    parts = _parts()
    layer = _layer(parts)
    x = torch.randn(3, FEATURES)

    out = layer(x)

    torch.testing.assert_close(out, F.linear(x, parts["weight"], layer.bias))


def test_a_pre_quant_scale_multiplies_the_input_columns_first() -> None:
    parts = _parts()
    pre_quant = (torch.rand(FEATURES) + 0.5).bfloat16()
    layer = _layer(parts, pre_quant, bias=False)
    x = torch.randn(2, FEATURES)

    out = layer(x)

    torch.testing.assert_close(out, F.linear(x * pre_quant.float(), parts["weight"]))


def test_the_layer_shares_the_sources_bias_and_keeps_everything_else_out_of_the_state_dict() -> None:
    source = nn.Linear(FEATURES, OUT)
    parts = _parts()

    layer = ComfyNvfp4Linear(source, parts["packed"], parts["blocked"], parts["per_tensor"], None, torch.float32)

    assert layer.bias is source.bias
    assert list(layer.state_dict()) == ["bias"]
    assert layer._pre_quant_scale() is None and not hasattr(layer, "pre_quant_scale_u8")
    assert (layer.in_features, layer.out_features) == (FEATURES, OUT)


def test_the_layer_follows_the_device_of_its_input() -> None:
    layer = _layer(_parts())

    out = layer(torch.randn(2, FEATURES, device="meta"))

    assert layer.packed_weight.device.type == "meta" and out.device.type == "meta" and out.shape == (2, OUT)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"packed": torch.zeros(OUT, FEATURES // 2, dtype=torch.int8)}, "Invalid NVFP4 packed weight shape"),
        ({"packed": torch.zeros(OUT, FEATURES // 4, dtype=torch.uint8)}, "Invalid NVFP4 packed weight shape"),
        ({"per_tensor": torch.ones(2)}, "weight_scale_2 must be a scalar"),
    ],
)
def test_malformed_packed_weights_or_tensor_scales_are_rejected(change: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _layer({**_parts(), **change})


def test_input_sizes_that_are_not_a_multiple_of_the_block_size_are_rejected() -> None:
    with pytest.raises(ValueError, match="divisible by 16"):
        ComfyNvfp4Linear(
            nn.Linear(8, OUT),
            torch.zeros(OUT, 4, dtype=torch.uint8),
            torch.zeros(1),
            torch.ones(1),
            None,
            torch.float32,
        )


@pytest.mark.parametrize(
    ("pre_quant", "message"),
    [(torch.ones(FEATURES - 2).bfloat16(), "one value per input column"), (torch.ones(FEATURES), "must be bfloat16")],
)
def test_a_pre_quant_scale_must_be_bfloat16_with_one_value_per_input(pre_quant: torch.Tensor, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _layer(_parts(), pre_quant)
