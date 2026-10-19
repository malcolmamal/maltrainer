from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import nn

from maltrainer.models.minimax_h3.comfy_quant import (
    ComfyInt8Embedding,
    ComfyNvfp4Linear,
    load_comfy_quantized_state_dict,
)

OUT, FEATURES = 4, 16


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(FEATURES, OUT)


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(6, 4)
        self.layers = nn.ModuleList([_Block()])
        self.norm = nn.LayerNorm(4)


def _key_map(key: str) -> str:
    return key.removeprefix("model.")


def _marker(config: dict | str) -> torch.Tensor:
    return torch.tensor(list((config if isinstance(config, str) else json.dumps(config)).encode()), dtype=torch.uint8)


def _tensors() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    blocked = torch.zeros(512)
    blocked[::16][:OUT] = torch.rand(OUT, generator=generator) + 0.5
    return {
        "model.embed.weight": torch.randint(-127, 128, (6, 4), dtype=torch.int8, generator=generator),
        "model.embed.weight_scale": torch.rand(6, 1, generator=generator) + 0.5,
        "model.embed.comfy_quant": _marker({"format": "int8_tensorwise"}),
        "model.layers.0.fc.weight": torch.randint(0, 256, (OUT, FEATURES // 2), dtype=torch.uint8, generator=generator),
        "model.layers.0.fc.weight_scale": blocked.to(torch.float8_e4m3fn),
        "model.layers.0.fc.weight_scale_2": torch.tensor(0.5),
        "model.layers.0.fc.comfy_quant": _marker({"format": "nvfp4"}),
        "model.layers.0.fc.bias": torch.randn(OUT, generator=generator),
        "model.norm.weight": torch.randn(4, generator=generator),
        "model.norm.bias": torch.randn(4, generator=generator),
    }


def _load(tmp_path: Path, tensors: dict[str, torch.Tensor], model: nn.Module | None = None) -> int:
    path = tmp_path / "model.safetensors"
    save_file(tensors, str(path))
    return load_comfy_quantized_state_dict(model or _Tiny(), path, key_map=_key_map, output_dtype=torch.float32)


@pytest.mark.parametrize("with_pre_quant_scale", [False, True])
def test_quantized_layers_are_swapped_in_and_the_remaining_tensors_loaded(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, with_pre_quant_scale: bool
) -> None:
    tensors = _tensors()
    if with_pre_quant_scale:
        tensors["model.layers.0.fc.pre_quant_scale"] = torch.full((FEATURES,), 2.0).bfloat16()
    model = _Tiny()

    with caplog.at_level(logging.INFO, logger="maltrainer.models.minimax_h3.comfy_quant"):
        count = _load(tmp_path, tensors, model)

    fc, x = model.layers[0].fc, torch.randn(3, FEATURES)
    assert count == 2
    assert "Attached 2 Comfy-quantized H3 layers: {'int8_tensorwise': 1, 'nvfp4': 1}" in caplog.text
    assert isinstance(model.embed, ComfyInt8Embedding) and isinstance(fc, ComfyNvfp4Linear)
    torch.testing.assert_close(fc.bias.detach(), tensors["model.layers.0.fc.bias"])
    torch.testing.assert_close(model.norm.weight.detach(), tensors["model.norm.weight"])
    scale = 2.0 if with_pre_quant_scale else 1.0
    torch.testing.assert_close(fc(x), F.linear(x * scale, fc.dequantize_weight(torch.float32), fc.bias))


def test_the_embedding_replacement_looks_rows_up_with_their_scales(tmp_path: Path) -> None:
    tensors, model = _tensors(), _Tiny()

    _load(tmp_path, tensors, model)

    ids = torch.tensor([[0, 5]])
    expected = tensors["model.embed.weight"].float() * tensors["model.embed.weight_scale"]
    torch.testing.assert_close(model.embed(ids), expected[ids])


def _drop(*keys: str):
    return lambda tensors: [tensors.pop(key) for key in keys]


def _set(key: str, value: torch.Tensor | None):
    return lambda tensors: tensors.update({key: value})


@pytest.mark.parametrize(
    ("edit", "error", "message"),
    [
        (_drop(*[k for k in _tensors() if k.endswith(".comfy_quant")]), ValueError, "does not contain Comfy quantized"),
        (_set("model.embed.comfy_quant", _marker("{}")), TypeError, "model.embed has no quantization format"),
        (_set("model.embed.comfy_quant", _marker('{"format": 4}')), TypeError, "has no quantization format"),
        (_set("model.embed.comfy_quant", _marker("[1]")), TypeError, "must contain a JSON object"),
        (_set("model.embed.comfy_quant", _marker("{bad")), ValueError, "Invalid Comfy quantization marker"),
        (_drop("model.embed.weight_scale"), ValueError, "model.embed is missing its quantized weight or scale"),
        (_drop("model.layers.0.fc.weight"), ValueError, "fc is missing its quantized weight or scale"),
        (_drop("model.layers.0.fc.weight_scale_2"), ValueError, "fc is missing weight_scale_2"),
        (_set("model.embed.comfy_quant", _marker({"format": "nvfp4"})), ValueError, "embedding quantization 'nvfp4'"),
        (_set("model.layers.0.fc.comfy_quant", _marker({"format": "int8_tensorwise"})), ValueError, "Linear quant"),
    ],
)
def test_a_marker_that_cannot_be_applied_to_its_module_is_an_error(
    tmp_path: Path, edit, error: type[Exception], message: str
) -> None:
    tensors = _tensors()
    edit(tensors)

    with pytest.raises(error, match=message):
        _load(tmp_path, tensors)


def test_a_marker_on_a_module_that_is_neither_a_linear_nor_an_embedding_is_an_error(tmp_path: Path) -> None:
    tensors = {**_tensors(), "model.norm.comfy_quant": _marker({"format": "nvfp4"})}
    tensors["model.norm.weight_scale"] = torch.ones(1)

    with pytest.raises(TypeError, match="model.norm maps to LayerNorm; expected Linear or Embedding"):
        _load(tmp_path, tensors)


def test_a_marker_for_a_module_the_model_lacks_is_an_error(tmp_path: Path) -> None:
    tensors = {**_tensors(), "model.ghost.comfy_quant": _marker({"format": "nvfp4"})}

    with pytest.raises(AttributeError, match="ghost"):
        _load(tmp_path, tensors)


@pytest.mark.parametrize("key", ["stray", "model.norm.extra", "model.other.weight"])
def test_a_tensor_the_model_does_not_expect_is_an_error(tmp_path: Path, key: str) -> None:
    with pytest.raises(ValueError, match=f"unexpected key '{key}'"):
        _load(tmp_path, {**_tensors(), key: torch.zeros(1)})


def test_a_model_tensor_the_file_lacks_is_an_error(tmp_path: Path) -> None:
    tensors = _tensors()
    del tensors["model.norm.bias"]

    with pytest.raises(ValueError, match=r"is missing 1 tensor\(s\), examples: \['norm.bias'\]"):
        _load(tmp_path, tensors)


def test_a_tensor_of_the_wrong_shape_fails_the_load(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="size mismatch"):
        _load(tmp_path, {**_tensors(), "model.norm.weight": torch.ones(3)})


def test_a_load_the_model_reports_as_incomplete_is_an_error(tmp_path: Path) -> None:
    model = _Tiny()
    model.register_load_state_dict_post_hook(lambda _module, keys: keys.missing_keys.append("norm.extra"))

    with pytest.raises(RuntimeError, match="Strict NVFP4/AWQ load failed"):
        _load(tmp_path, _tensors(), model)
