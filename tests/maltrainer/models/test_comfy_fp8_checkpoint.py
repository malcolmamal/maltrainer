from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from maltrainer.models.comfy_fp8 import container_prefix, read_comfy_checkpoint, strip_prefix

FP8 = torch.tensor([[1.0, -2.0], [0.5, 448.0]]).to(torch.float8_e4m3fn)
E4M3 = {"format": "float8_e4m3fn"}


def _save(path: Path, tensors: dict, metadata: dict | None = None) -> Path:
    save_file(tensors, str(path), metadata=metadata)
    return path


def _quant_blob(config: dict) -> torch.Tensor:
    return torch.tensor(list(json.dumps(config).encode("utf-8")), dtype=torch.uint8)


def test_layers_listed_in_the_header_metadata_are_reported(tmp_path: Path) -> None:
    header = {"_quantization_metadata": json.dumps({"layers": {"blk": E4M3}})}
    path = _save(tmp_path / "m.safetensors", {"blk.weight": FP8, "blk.weight_scale": torch.tensor(0.5)}, header)

    tensors, layers = read_comfy_checkpoint(path)

    assert layers == {"blk": E4M3}
    assert tensors["blk.weight"].dtype == torch.float8_e4m3fn
    torch.testing.assert_close(tensors["blk.weight"].float(), FP8.float())
    assert tensors["blk.weight_scale"].item() == 0.5


def test_comfy_quant_tensors_mark_layers_and_raw_bytes_are_viewed_as_fp8(tmp_path: Path) -> None:
    e5m2 = torch.tensor([1.0, -0.5]).to(torch.float8_e5m2)
    tensors = {
        "a.weight": FP8.view(torch.uint8),
        "a.comfy_quant": _quant_blob(E4M3),
        "b.weight": e5m2.view(torch.uint8),
        "b.comfy_quant": _quant_blob({"format": "float8_e5m2"}),
        "norm.weight": torch.ones(2),
    }

    loaded, layers = read_comfy_checkpoint(_save(tmp_path / "q.safetensors", tensors))

    assert layers == {"a": E4M3, "b": {"format": "float8_e5m2"}}
    assert sorted(loaded) == ["a.weight", "b.weight", "norm.weight"]
    torch.testing.assert_close(loaded["a.weight"].float(), FP8.float())
    torch.testing.assert_close(loaded["b.weight"].float(), torch.tensor([1.0, -0.5]))


@pytest.mark.parametrize("marker", ["scaled_fp8", "model.scaled_fp8"])
def test_legacy_scaled_fp8_files_rename_the_scale_and_drop_activation_scales(tmp_path: Path, marker: str) -> None:
    tensors = {
        marker: torch.zeros(0, dtype=torch.float8_e4m3fn),
        "blk.weight": FP8,
        "blk.scale_weight": torch.tensor(2.0),
        "blk.scale_input": torch.tensor(1.0),
        "other.input_scale": torch.tensor(1.0),
        "other.bias": torch.zeros(2),
    }

    loaded, layers = read_comfy_checkpoint(_save(tmp_path / "legacy.safetensors", tensors))

    assert layers == {"blk": {"format": "float8_e4m3fn", "legacy": True}}
    assert sorted(loaded) == ["blk.weight", "blk.weight_scale", "other.bias"]
    assert loaded["blk.weight_scale"].item() == 2.0


def test_a_scale_weight_without_the_legacy_marker_is_not_a_quantized_layer(tmp_path: Path) -> None:
    tensors = {"blk.weight": torch.ones(2, 2), "blk.scale_weight": torch.tensor(2.0)}

    loaded, layers = read_comfy_checkpoint(_save(tmp_path / "plain.safetensors", tensors, {"format": "pt"}))

    assert layers == {}
    assert sorted(loaded) == ["blk.weight", "blk.weight_scale"]


def test_a_layer_listed_without_its_weight_is_tolerated(tmp_path: Path) -> None:
    header = {"_quantization_metadata": json.dumps({"layers": {"gone": E4M3}})}

    loaded, layers = read_comfy_checkpoint(_save(tmp_path / "m.safetensors", {"x": torch.ones(1)}, header))

    assert layers == {"gone": E4M3}
    assert list(loaded) == ["x"]


def test_metadata_without_a_layers_table_means_no_quantized_layers(tmp_path: Path) -> None:
    header = {"_quantization_metadata": json.dumps({"version": 1})}

    _, layers = read_comfy_checkpoint(_save(tmp_path / "m.safetensors", {"x": torch.ones(1)}, header))

    assert layers == {}


def test_a_non_fp8_quantization_format_is_rejected_by_name(tmp_path: Path) -> None:
    tensors = {"blk.weight": torch.zeros(2, 2, dtype=torch.int8), "blk.comfy_quant": _quant_blob({"format": "int8"})}
    path = _save(tmp_path / "int8.safetensors", tensors)

    with pytest.raises(ValueError, match=r"int8\.safetensors: layer blk uses quantization 'int8'"):
        read_comfy_checkpoint(path)


@pytest.mark.parametrize(
    ("keys", "expected"),
    [
        (["model.diffusion_model.a", "model.diffusion_model.b"], "model.diffusion_model."),
        (["diffusion_model.a", "diffusion_model.b"], "diffusion_model."),
        (["diffusion_model.a", "vae.b"], ""),
        (["a.weight"], ""),
        ([], ""),
    ],
)
def test_the_container_prefix_is_the_one_every_key_carries(keys: list[str], expected: str) -> None:
    assert container_prefix(iter(keys)) == expected


def test_custom_candidates_are_tried_in_order() -> None:
    assert container_prefix(["te.model.x", "te.model.y"], candidates=("te.", "te.model.")) == "te."


def test_strip_prefix_removes_the_container_from_tensors_and_layers() -> None:
    tensors = {"diffusion_model.blk.weight": FP8, "diffusion_model.norm": torch.ones(1)}
    layers = {"diffusion_model.blk": E4M3, "bare": E4M3}

    stripped, stripped_layers = strip_prefix(tensors, layers)

    assert sorted(stripped) == ["blk.weight", "norm"]
    assert stripped_layers == {"blk": E4M3, "bare": E4M3}


def test_strip_prefix_leaves_mixed_keys_alone() -> None:
    tensors = {"diffusion_model.blk.weight": FP8, "vae.weight": torch.ones(1)}
    layers = {"diffusion_model.blk": E4M3}

    stripped, stripped_layers = strip_prefix(tensors, layers)

    assert stripped is tensors
    assert stripped_layers is layers
