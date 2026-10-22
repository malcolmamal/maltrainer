from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import maltrainer.models.minimax_h3.video_vae as vv
from maltrainer.models.minimax_h3.video_vae import _map_encoder_key, load_minimax_h3_video_vae

from .test_video_vae_encoder import small_encoder

CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def small_default_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    """The loader builds the default (1024-wide) encoder; give it the 4-channel one instead."""
    real = vv.MiniMaxH3VideoEncoderModel
    monkeypatch.setattr(vv, "MiniMaxH3VideoEncoderModel", lambda: real(block_out_channels=(4,) * 6, norm_num_groups=2))


@pytest.mark.parametrize(
    "key, expected",
    [
        ("encoder.conv_in.weight", "conv_in.weight"),
        ("encoder.norm_out.bias", "norm_out.bias"),
        ("encoder.conv_out.weight", "conv_out.weight"),
        ("quant_conv.weight", "quant_conv.weight"),
        ("encoder.down.2.block.1.conv1.weight", "down_blocks.2.resnets.1.conv1.weight"),
        ("encoder.down.0.block.0.nin_shortcut.weight", "down_blocks.0.resnets.0.conv_shortcut.weight"),
        ("encoder.down.3.downsample.conv.weight", "down_blocks.3.downsamplers.0.conv.weight"),
        ("encoder.mid.block.1.norm2.weight", "mid_block.resnets.1.norm2.weight"),
        ("encoder.mid.block.0.nin_shortcut.bias", "mid_block.resnets.0.conv_shortcut.bias"),
        ("decoder.conv_in.weight", None),
        ("post_quant_conv.weight", None),
        ("latents_mean", None),
    ],
)
def test_original_release_key_names_map_to_module_names_and_non_encoder_keys_are_dropped(
    key: str, expected: str | None
) -> None:
    assert _map_encoder_key(key) == expected


def _release_name(key: str) -> str:
    """The inverse of the loader's mapping: this module's names back to the release's."""
    if key.startswith("quant_conv."):
        return key
    match = re.match(r"down_blocks\.(\d+)\.(.*)", key)
    if not match:
        return f"encoder.{key}"
    level, rest = match.groups()
    rest = rest.replace("resnets.", "block.", 1).replace("conv_shortcut.", "nin_shortcut.", 1)
    rest = rest.replace("downsamplers.0.", "downsample.", 1)
    return f"encoder.down.{level}.{rest}"


def _write(state: dict, path: Path) -> Path:
    save_file({k: v.contiguous() for k, v in state.items()}, str(path))
    return path


def _release_state(source: torch.nn.Module) -> dict:
    return {_release_name(k): v.detach().clone() for k, v in source.state_dict().items()}


def test_every_encoder_tensor_survives_the_round_trip_through_the_release_names() -> None:
    source = small_encoder()

    mapped = {_map_encoder_key(k) for k in _release_state(source)}

    assert mapped == set(source.state_dict())


def test_a_release_checkpoint_loads_into_a_frozen_fp16_eval_encoder_with_the_same_weights(tmp_path: Path) -> None:
    source = small_encoder()
    state = _release_state(source)
    state["decoder.conv_in.weight"] = torch.zeros(1)  # the decoder half is ignored
    state["post_quant_conv.weight"] = torch.zeros(1)

    loaded = load_minimax_h3_video_vae(_write(state, tmp_path / "vae.safetensors"), CPU)

    assert not loaded.training
    assert not any(p.requires_grad for p in loaded.parameters())
    assert {p.dtype for p in loaded.parameters()} == {torch.float16}
    for key, value in source.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value.half()), key


def test_a_checkpoint_lacking_encoder_tensors_is_refused_naming_how_many_and_which(tmp_path: Path) -> None:
    state = _release_state(small_encoder())
    del state["encoder.conv_in.weight"], state["quant_conv.bias"]

    with pytest.raises(ValueError, match=r"vae.safetensors did not supply 2 video VAE encoder tensor.*conv_in.weight"):
        load_minimax_h3_video_vae(_write(state, tmp_path / "vae.safetensors"), CPU)


def test_a_checkpoint_with_an_encoder_tensor_the_module_lacks_is_refused_naming_it(tmp_path: Path) -> None:
    state = _release_state(small_encoder())
    state["encoder.down.9.block.0.conv1.weight"] = torch.zeros(1)

    with pytest.raises(ValueError, match=r"mapped 1 tensor.*does not expect.*down_blocks.9.resnets.0.conv1.weight"):
        load_minimax_h3_video_vae(_write(state, tmp_path / "vae.safetensors"), CPU)


def test_a_checkpoint_with_a_mid_block_is_refused_because_the_encoder_has_none(tmp_path: Path) -> None:
    state = _release_state(small_encoder())
    state["encoder.mid.block.0.norm1.weight"] = torch.zeros(4)

    with pytest.raises(ValueError, match=r"mid_block.resnets.0.norm1.weight"):
        load_minimax_h3_video_vae(_write(state, tmp_path / "vae.safetensors"), CPU)
