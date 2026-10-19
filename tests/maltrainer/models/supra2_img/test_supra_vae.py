from __future__ import annotations

import json
from pathlib import Path

import torch
from diffusers import AutoencoderKL
from safetensors.torch import save_file

import maltrainer.models.supra2_img as package
from maltrainer.models.supra2_img.vae import VAE_SCALE, encode_latents, load_sd_vae

CPU = torch.device("cpu")
CONFIG = {
    "in_channels": 3,
    "out_channels": 3,
    "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D"],
    "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D"],
    "block_out_channels": [8, 16],
    "layers_per_block": 1,
    "latent_channels": 4,
    "norm_num_groups": 4,
    "sample_size": 16,
}


def _files(tmp_path: Path) -> tuple[Path, Path, AutoencoderKL]:
    torch.manual_seed(0)
    vae = AutoencoderKL.from_config(CONFIG)
    weights, config = tmp_path / "vae.safetensors", tmp_path / "config.json"
    save_file({key: value.clone() for key, value in vae.state_dict().items()}, str(weights))
    config.write_text(json.dumps(CONFIG), encoding="utf-8")
    return weights, config, vae


def test_the_vae_loads_strictly_frozen_and_in_the_requested_dtype(tmp_path: Path) -> None:
    weights, config, source = _files(tmp_path)

    vae = load_sd_vae(weights, config, CPU, torch.bfloat16)

    assert not vae.training and not any(p.requires_grad for p in vae.parameters())
    assert {p.dtype for p in vae.parameters()} == {torch.bfloat16}
    assert torch.equal(vae.encoder.conv_in.weight, source.encoder.conv_in.weight.bfloat16())


def test_latents_are_the_scaled_posterior_mode_of_pixels_mapped_to_minus_one_to_one(tmp_path: Path) -> None:
    weights, config, _ = _files(tmp_path)
    vae = load_sd_vae(weights, config, CPU, torch.float32)
    pixels = torch.rand(2, 3, 16, 16, generator=torch.Generator().manual_seed(1))

    latents = encode_latents(vae, pixels)

    expected = vae.encode(pixels * 2 - 1).latent_dist.mode() * 0.18215
    assert VAE_SCALE == 0.18215
    assert latents.shape == (2, 4, 8, 8) and not latents.requires_grad
    torch.testing.assert_close(latents, expected)


def test_the_package_exports_its_public_entry_points() -> None:
    assert sorted(package.__all__) == [
        "SupraDiT",
        "SupraLoRANetwork",
        "encode_latents",
        "load_flan_t5",
        "load_sd_vae",
        "load_supra2_img",
    ]
    assert all(callable(getattr(package, name)) for name in package.__all__)
