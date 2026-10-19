from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from maltrainer.models.minimax_h3.comfy_quant import ComfyInt8Embedding, _parse_quant_config, has_comfy_quantized_layers


def _blob(payload: bytes) -> torch.Tensor:
    return torch.tensor(list(payload), dtype=torch.uint8)


def test_a_marker_tensor_is_decoded_into_its_config() -> None:
    config = {"format": "nvfp4", "group": 16}

    assert _parse_quant_config(_blob(json.dumps(config).encode())) == config


@pytest.mark.parametrize(
    ("payload", "error", "message"),
    [
        (b"\xff\xfe", ValueError, "Invalid Comfy quantization marker"),
        (b"{nope", ValueError, "Invalid Comfy quantization marker"),
        (b'"text"', TypeError, "must contain a JSON object"),
    ],
)
def test_a_marker_that_is_not_a_json_object_is_rejected(payload: bytes, error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        _parse_quant_config(_blob(payload))


def test_checkpoints_are_detected_by_the_comfy_quant_marker_suffix(tmp_path: Path) -> None:
    quantized, plain = tmp_path / "quantized.safetensors", tmp_path / "plain.safetensors"
    save_file({"fc.weight": torch.ones(2), "fc.comfy_quant": _blob(b"{}")}, str(quantized))
    save_file({"fc.weight": torch.ones(2), "comfy_quant_notes": _blob(b"{}")}, str(plain))

    assert has_comfy_quantized_layers(quantized) is True
    assert has_comfy_quantized_layers(str(quantized)) is True
    assert has_comfy_quantized_layers(plain) is False


def test_a_path_that_is_not_a_file_has_no_quantized_layers(tmp_path: Path) -> None:
    assert has_comfy_quantized_layers(tmp_path / "absent.safetensors") is False
    assert has_comfy_quantized_layers(tmp_path) is False


def _table(rows: int = 6, dim: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return (
        torch.randint(-127, 128, (rows, dim), dtype=torch.int8, generator=generator),
        torch.rand(rows, 1, generator=generator) + 0.5,
    )


def test_a_lookup_returns_each_row_times_its_own_scale() -> None:
    qweight, scales = _table()
    embedding = ComfyInt8Embedding(qweight, scales, torch.float32)
    ids = torch.tensor([[5, 0, 2], [2, 2, 1]])

    out = embedding(ids)

    assert (embedding.num_embeddings, embedding.embedding_dim) == (6, 4)
    assert out.shape == (2, 3, 4)
    torch.testing.assert_close(out, qweight.float()[ids] * scales[ids])


def test_the_output_has_the_requested_dtype_and_scales_may_be_half_precision_or_flat() -> None:
    qweight, scales = _table()
    embedding = ComfyInt8Embedding(qweight, scales.reshape(-1).half(), torch.bfloat16)

    out = embedding(torch.tensor([3]))

    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), (qweight[3:4].float() * scales.half().float()[3]).bfloat16().float())


def test_the_table_is_held_in_non_persistent_buffers_that_survive_a_dtype_cast() -> None:
    qweight, scales = _table()
    embedding = ComfyInt8Embedding(qweight, scales, torch.float32)

    embedding.to(torch.bfloat16)

    assert embedding.state_dict() == {}
    assert sorted(name for name, _ in embedding.named_buffers()) == ["qweight", "scales_u8"]
    assert embedding.scales_u8.dtype == torch.uint8 and embedding.scales_u8.numel() == 6 * 4
    torch.testing.assert_close(embedding(torch.tensor([1])), (qweight[1:2].float() * scales[1]))


@pytest.mark.parametrize(
    ("qweight", "scales", "message"),
    [
        (torch.zeros(3, 4), torch.ones(3), "rank-2 int8 tensor"),
        (torch.zeros(3, 4, 2, dtype=torch.int8), torch.ones(3), "rank-2 int8 tensor"),
        (torch.zeros(3, 4, dtype=torch.int8), torch.ones(2), "one scale per row"),
    ],
)
def test_a_table_that_is_not_int8_with_one_scale_per_row_is_rejected(
    qweight: torch.Tensor, scales: torch.Tensor, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ComfyInt8Embedding(qweight, scales, torch.float32)
