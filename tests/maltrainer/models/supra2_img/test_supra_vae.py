from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from diffusers import AutoencoderKL
from safetensors.torch import save_file

from maltrainer.models.supra2_img.model import VAE_SCALE
from maltrainer.models.supra2_img.vae import encode_latents, load_sd_vae

CPU = torch.device("cpu")
CONFIG = {
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 4,
    "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D"],
    "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D"],
    "block_out_channels": [32, 32],
    "layers_per_block": 1,
    "norm_num_groups": 32,
    "sample_size": 16,
}


def _vae(seed: int = 0) -> AutoencoderKL:
    torch.manual_seed(seed)
    return AutoencoderKL.from_config(CONFIG).eval()


def _files(tmp_path: Path, vae: AutoencoderKL | None = None, drop: str | None = None) -> tuple[Path, Path]:
    vae = vae or _vae()
    config = tmp_path / "config.json"
    config.write_text(json.dumps(CONFIG), encoding="utf-8")
    state = {k: v.clone() for k, v in vae.state_dict().items() if drop is None or not k.startswith(drop)}
    save_file(state, str(tmp_path / "vae.safetensors"))
    return tmp_path / "vae.safetensors", config


def test_the_loader_reproduces_the_saved_weights_frozen_and_in_eval_mode(tmp_path: Path) -> None:
    source = _vae()
    weights, config = _files(tmp_path, source)

    loaded = load_sd_vae(weights, config, CPU, torch.float32)

    for (name, value), reference in zip(loaded.state_dict().items(), source.state_dict().values()):
        assert torch.equal(value, reference), name
    assert not loaded.training and not any(p.requires_grad for p in loaded.parameters())


def test_the_loader_casts_to_the_requested_dtype(tmp_path: Path) -> None:
    weights, config = _files(tmp_path)

    loaded = load_sd_vae(weights, config, CPU, torch.bfloat16)

    assert {p.dtype for p in loaded.parameters()} == {torch.bfloat16}


def test_the_loader_is_strict_so_a_missing_tensor_fails(tmp_path: Path) -> None:
    weights, config = _files(tmp_path, drop="quant_conv")

    with pytest.raises(RuntimeError, match="quant_conv"):
        load_sd_vae(weights, config, CPU, torch.float32)


def test_legacy_attention_keys_are_renamed_on_load(tmp_path: Path) -> None:
    source = _vae()
    weights, config = _files(tmp_path, source)
    state = {k: v.clone() for k, v in source.state_dict().items()}
    for key in [k for k in state if ".to_q." in k]:
        state[key.replace(".to_q.", ".query.")] = state.pop(key)
    save_file(state, str(weights))

    loaded = load_sd_vae(weights, config, CPU, torch.float32)

    assert torch.equal(
        loaded.state_dict()["encoder.mid_block.attentions.0.to_q.weight"],
        source.state_dict()["encoder.mid_block.attentions.0.to_q.weight"],
    )


def test_encoding_maps_unit_pixels_to_minus_one_one_and_scales_the_posterior_mode() -> None:
    vae = _vae()
    pixels = torch.rand(2, 3, 16, 16)

    latents = encode_latents(vae, pixels)

    expected = vae.encode(pixels * 2.0 - 1.0).latent_dist.mode() * VAE_SCALE
    assert latents.shape == (2, 4, 8, 8)
    assert torch.allclose(latents, expected)
    assert VAE_SCALE == 0.18215


def test_encoding_is_deterministic_and_carries_no_gradient() -> None:
    vae = _vae()
    vae.requires_grad_(True)
    pixels = torch.rand(1, 3, 16, 16, requires_grad=True)

    first, second = encode_latents(vae, pixels), encode_latents(vae, pixels)

    assert torch.equal(first, second) and not first.requires_grad
