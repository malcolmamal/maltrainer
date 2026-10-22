from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from maltrainer.models.moss_image.loading import load_moss_image_transformer
from maltrainer.models.moss_image.model import MOSSImageTransformer

from .tiny import CPU, json_tensor

Maker = Callable[..., MOSSImageTransformer]
Writer = Callable[..., Path]


def _same_weights(loaded: MOSSImageTransformer, source: MOSSImageTransformer, dtype: torch.dtype) -> bool:
    expected = source.state_dict()
    return all(torch.equal(v, expected[k].to(dtype)) for k, v in loaded.state_dict().items())


@pytest.mark.parametrize("prefix", ["", "diffusion_model.", "model.diffusion_model."])
def test_a_plain_checkpoint_loads_with_or_without_its_container_prefix(
    make_model: Maker, write_checkpoint: Writer, prefix: str
) -> None:
    source = make_model()

    model, info = load_moss_image_transformer(write_checkpoint(source, prefix=prefix), CPU, torch.float32)

    assert _same_weights(model, source, torch.float32)
    assert info == {
        "file": "moss_image_base_bf16.safetensors",
        "format": "float32",
        "int8_layers": 0,
        "dim": 128,
        "num_layers": 2,
        "fused_qkv": False,
        "inference_attention": {},
    }


def test_the_loaded_model_is_a_frozen_eval_copy_cast_to_the_requested_dtype(
    make_model: Maker, write_checkpoint: Writer
) -> None:
    source = make_model()

    model, _ = load_moss_image_transformer(write_checkpoint(source), CPU, torch.bfloat16)

    assert {p.dtype for p in model.parameters()} == {torch.bfloat16}
    assert not any(p.requires_grad for p in model.parameters())
    assert not model.training
    assert model.compute_dtype == torch.bfloat16
    assert _same_weights(model, source, torch.bfloat16)


def test_the_info_names_the_stored_dtype_and_accepts_a_string_path(make_model: Maker, write_checkpoint: Writer) -> None:
    path = write_checkpoint(make_model(), dtype=torch.bfloat16)

    _, info = load_moss_image_transformer(str(path), CPU, torch.float32)

    assert info["format"] == "bfloat16"


def test_a_fused_qkv_checkpoint_builds_a_fused_model(make_model: Maker, write_checkpoint: Writer) -> None:
    source = make_model(fused_qkv=True)

    model, info = load_moss_image_transformer(write_checkpoint(source), CPU, torch.float32)

    assert info["fused_qkv"] is True
    assert "qkv" in model.layers[0].proj
    assert _same_weights(model, source, torch.float32)


@pytest.mark.parametrize("prefix", ["", "model.diffusion_model."])
def test_rope_bands_are_dropped_and_attention_hints_reported_not_loaded(
    make_model: Maker, write_checkpoint: Writer, prefix: str
) -> None:
    extra = {
        "rope.bands": torch.zeros(4),
        "layers.0.comfy_attention.config": json_tensor({"attention": "sage"}),
        "layers.1.comfy_attention.config": json_tensor([{"attention": "sage"}, {"attention": "flash"}]),
        "text_embedder.blocks.0.comfy_attention.config": torch.tensor([200, 1], dtype=torch.uint8),
        "text_embedder.blocks.1.comfy_attention.config": json_tensor({"attention": "sage"}),
    }
    path = write_checkpoint(make_model(), prefix=prefix, extra=extra)

    model, info = load_moss_image_transformer(path, CPU, torch.float32)

    assert info["inference_attention"] == {"sage": "2 blocks", "sage,flash": "1 blocks", "unreadable": "1 blocks"}
    assert not any("rope" in key or "comfy_attention" in key for key in model.state_dict())


def test_a_missing_file_is_reported_by_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="MOSS-Image transformer not found"):
        load_moss_image_transformer(tmp_path / "absent.safetensors", CPU)


def test_a_file_that_is_not_a_moss_image_dit_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "other.safetensors"
    save_file({"layers.0.proj.q.weight": torch.zeros(2, 2)}, str(path))

    with pytest.raises(ValueError, match="Not a MOSS-Image DiT checkpoint"):
        load_moss_image_transformer(path, CPU)


@pytest.mark.parametrize("broken", ["layers.1.proj.o.weight", "final_norm_video.weight"])
def test_a_checkpoint_missing_a_weight_fails_strictly(make_model: Maker, tmp_path: Path, broken: str) -> None:
    tensors = {k: v.detach().clone() for k, v in make_model().state_dict().items() if k != broken}
    path = tmp_path / "moss_image_partial.safetensors"
    save_file(tensors, str(path))

    with pytest.raises(RuntimeError, match=broken):
        load_moss_image_transformer(path, CPU)


def test_an_unexpected_tensor_fails_strictly(make_model: Maker, write_checkpoint: Writer) -> None:
    path = write_checkpoint(make_model(), extra={"layers.0.proj.extra.weight": torch.zeros(1)})

    with pytest.raises(RuntimeError, match=re.escape("layers.0.proj.extra.weight")):
        load_moss_image_transformer(path, CPU)


def test_the_sft_checkpoint_loads_with_a_warning(
    make_model: Maker, write_checkpoint: Writer, caplog: pytest.LogCaptureFixture
) -> None:
    path = write_checkpoint(make_model(), name="moss_image_sft_bf16.safetensors")

    with caplog.at_level(logging.WARNING):
        load_moss_image_transformer(path, CPU)

    assert "SFT (supervised fine-tune) checkpoint" in caplog.text


def test_the_base_checkpoint_loads_without_a_warning(
    make_model: Maker, write_checkpoint: Writer, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        load_moss_image_transformer(write_checkpoint(make_model()), CPU)

    assert caplog.text == ""
