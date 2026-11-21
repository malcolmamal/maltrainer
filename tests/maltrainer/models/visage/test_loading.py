from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from maltrainer.models.visage.loading import load_visage_transformer, strip_container_prefix

from .tiny_model import TINY, tiny_inputs, tiny_transformer

CPU = torch.device("cpu")


def _write(path: Path, prefix: str = "") -> dict[str, torch.Tensor]:
    state = {key: value.detach().clone().contiguous() for key, value in tiny_transformer().state_dict().items()}
    save_file({prefix + key: value for key, value in state.items()}, str(path))
    return state


def test_a_missing_checkpoint_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=re.escape("gone.safetensors")):
        load_visage_transformer(tmp_path / "gone.safetensors", CPU)


@pytest.mark.parametrize("prefix", ["", "model.diffusion_model.", "diffusion_model.", "transformer."])
def test_the_file_loads_under_any_container_prefix_with_the_weights_it_holds(tmp_path: Path, prefix: str) -> None:
    state = _write(tmp_path / "v.safetensors", prefix)

    model, info = load_visage_transformer(tmp_path / "v.safetensors", CPU, torch.float32)

    assert model.config.width == TINY.width and model.config.depth == TINY.depth
    assert all(torch.equal(model.state_dict()[key], value) for key, value in state.items())
    assert info == {"file": "v.safetensors", "format": "float32", "width": 128, "depth": 2, "latent_channels": 8}


def test_the_loaded_model_is_the_one_that_was_saved(tmp_path: Path) -> None:
    _write(tmp_path / "v.safetensors")
    inputs = tiny_inputs()

    model, _ = load_visage_transformer(tmp_path / "v.safetensors", CPU, torch.float32)

    assert torch.equal(model(*inputs), tiny_transformer()(*inputs))


def test_the_weights_are_cast_frozen_and_put_in_eval_mode(tmp_path: Path) -> None:
    _write(tmp_path / "v.safetensors")

    model, _ = load_visage_transformer(tmp_path / "v.safetensors", CPU, torch.bfloat16)

    assert {p.dtype for p in model.parameters()} == {torch.bfloat16}
    assert not model.training and not any(p.requires_grad for p in model.parameters())


def test_a_checkpoint_that_is_not_a_visage_dit_is_refused(tmp_path: Path) -> None:
    save_file({"layers.0.weight": torch.zeros(2, 2)}, str(tmp_path / "other.safetensors"))

    with pytest.raises(ValueError, match="not a Visage-Image DiT checkpoint"):
        load_visage_transformer(tmp_path / "other.safetensors", CPU)


def test_a_dit_with_a_missing_tensor_fails_the_strict_load(tmp_path: Path) -> None:
    state = _write(tmp_path / "v.safetensors")
    del state["blocks.1.attn.qkv.weight"]
    save_file(state, str(tmp_path / "short.safetensors"))

    with pytest.raises(RuntimeError, match=r"blocks\.1\.attn\.qkv\.weight"):
        load_visage_transformer(tmp_path / "short.safetensors", CPU)


def test_a_prefix_is_only_stripped_when_every_key_has_it() -> None:
    mixed = {"transformer.a": torch.zeros(1), "b": torch.zeros(1)}

    assert strip_container_prefix(mixed) is mixed
    assert strip_container_prefix({}) == {}
