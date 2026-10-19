from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from maltrainer.models.flux3.vae import (
    Flux2VAEDecoder,
    Flux2VAEEncoder,
    load_flux2_vae_decoder,
    load_flux2_vae_encoder,
)

CPU = torch.device("cpu")
CH, Z, EMBED = 32, 4, 2


def _stats(model: torch.nn.Module) -> None:
    generator = torch.Generator().manual_seed(9)
    model.bn_running_mean.copy_(torch.randn(4 * EMBED, generator=generator))
    model.bn_running_var.copy_(torch.rand(4 * EMBED, generator=generator) + 0.5)


def _encoder(z: int = 32) -> Flux2VAEEncoder:
    """z defaults to 32: that is what the loader assumes when the file carries no decoder to read it from."""
    torch.manual_seed(0)
    model = Flux2VAEEncoder(ch=CH, z_channels=z, embed_dim=EMBED).eval()
    _stats(model)
    return model


def _decoder() -> Flux2VAEDecoder:
    torch.manual_seed(1)
    model = Flux2VAEDecoder(ch=CH, z_channels=Z, embed_dim=EMBED).eval()
    _stats(model)
    return model


def _write(path: Path, *models: torch.nn.Module, nested: bool = False, prefix: str = "") -> Path:
    """The ComfyUI layout: encoder.*, decoder.*, root quant_conv/post_quant_conv and bn.running_*."""
    tensors: dict[str, torch.Tensor] = {}
    for model in models:
        for key, value in model.state_dict().items():
            if key.startswith("bn_running_"):
                key = "bn.running_" + key[len("bn_running_") :]
            elif nested and key.startswith(("quant_conv.", "post_quant_conv.")):
                key = ("encoder." if key.startswith("quant") else "decoder.") + key
            tensors[prefix + key] = value.clone()
    save_file(tensors, str(path))
    return path


def test_the_encoder_loader_reproduces_the_source_encoder_exactly(tmp_path: Path) -> None:
    source = _encoder()
    path = _write(tmp_path / "vae.safetensors", source)
    images = torch.rand(1, 3, 32, 32)

    loaded = load_flux2_vae_encoder(path, CPU)

    assert torch.equal(loaded.encode(images), source.encode(images))
    assert not loaded.training and not any(p.requires_grad for p in loaded.parameters())


def test_the_encoder_loader_infers_channel_widths_from_the_weights(tmp_path: Path) -> None:
    loaded = load_flux2_vae_encoder(_write(tmp_path / "v.safetensors", _encoder(Z), _decoder()), CPU)

    assert loaded.encoder.conv_in.out_channels == CH
    assert loaded.quant_conv.in_channels == 2 * Z and loaded.quant_conv.out_channels == 2 * EMBED


def test_the_encoder_loader_defaults_to_32_latent_channels_when_the_file_has_no_decoder(tmp_path: Path) -> None:
    loaded = load_flux2_vae_encoder(_write(tmp_path / "e.safetensors", _encoder()), CPU)

    assert loaded.quant_conv.in_channels == 64


@pytest.mark.parametrize("prefix", ["first_stage_model.", "vae."])
def test_the_encoder_loader_strips_a_container_prefix(tmp_path: Path, prefix: str) -> None:
    source = _encoder()
    path = _write(tmp_path / "vae.safetensors", source, prefix=prefix)

    loaded = load_flux2_vae_encoder(path, CPU)

    assert torch.equal(loaded.bn_running_mean, source.bn_running_mean)


def test_the_encoder_loader_moves_a_nested_encoder_quant_conv_to_the_root(tmp_path: Path) -> None:
    source = _encoder()
    path = _write(tmp_path / "vae.safetensors", source, nested=True)

    loaded = load_flux2_vae_encoder(path, CPU)

    assert torch.equal(loaded.quant_conv.weight, source.quant_conv.weight)


def test_the_encoder_loader_casts_to_the_requested_dtype(tmp_path: Path) -> None:
    loaded = load_flux2_vae_encoder(_write(tmp_path / "v.safetensors", _encoder()), CPU, dtype=torch.bfloat16)

    assert loaded.quant_conv.weight.dtype == torch.bfloat16 and loaded.bn_running_var.dtype == torch.bfloat16


def test_the_encoder_loader_rejects_a_vae_without_batch_norm_statistics(tmp_path: Path) -> None:
    save_file({"encoder.conv_in.weight": torch.zeros(1)}, str(tmp_path / "sd.safetensors"))

    with pytest.raises(ValueError, match=re.escape("FLUX.3 needs the Flux2 VAE")):
        load_flux2_vae_encoder(tmp_path / "sd.safetensors", CPU)


def test_the_decoder_loader_reproduces_the_source_decoder_exactly(tmp_path: Path) -> None:
    source = _decoder()
    path = _write(tmp_path / "vae.safetensors", source)
    latents = torch.randn(1, 4 * EMBED, 2, 2)

    loaded = load_flux2_vae_decoder(path, CPU)

    assert torch.equal(loaded.decode(latents), source.decode(latents))
    assert loaded.decoder.conv_in.in_channels == Z and loaded.post_quant_conv.in_channels == EMBED
    assert not loaded.training and not any(p.requires_grad for p in loaded.parameters())


def test_the_decoder_loader_moves_a_nested_decoder_post_quant_conv_to_the_root(tmp_path: Path) -> None:
    source = _decoder()
    path = _write(tmp_path / "vae.safetensors", source, nested=True)

    loaded = load_flux2_vae_decoder(path, CPU)

    assert torch.equal(loaded.post_quant_conv.weight, source.post_quant_conv.weight)


def test_the_decoder_loader_ignores_the_encoder_half_of_a_full_checkpoint(tmp_path: Path) -> None:
    source = _decoder()

    loaded = load_flux2_vae_decoder(_write(tmp_path / "v.safetensors", _encoder(Z), source), CPU, dtype=torch.float64)

    assert loaded.post_quant_conv.weight.dtype == torch.float64
    assert not any(key.startswith("encoder.") for key in loaded.state_dict())


@pytest.mark.parametrize("prefix", ["first_stage_model.", "vae."])
def test_the_decoder_loader_strips_a_container_prefix(tmp_path: Path, prefix: str) -> None:
    source = _decoder()

    loaded = load_flux2_vae_decoder(_write(tmp_path / "v.safetensors", source, prefix=prefix), CPU)

    assert torch.equal(loaded.bn_running_var, source.bn_running_var)


def test_the_decoder_loader_rejects_an_encoder_only_file(tmp_path: Path) -> None:
    path = _write(tmp_path / "enc.safetensors", _encoder())

    with pytest.raises(ValueError, match=re.escape("enc.safetensors has no decoder")):
        load_flux2_vae_decoder(path, CPU)


def test_the_decoder_loader_rejects_a_vae_without_batch_norm_statistics(tmp_path: Path) -> None:
    save_file({"decoder.conv_out.weight": torch.zeros(1)}, str(tmp_path / "sd.safetensors"))

    with pytest.raises(ValueError, match="no batch-norm latent statistics"):
        load_flux2_vae_decoder(tmp_path / "sd.safetensors", CPU)


@pytest.mark.parametrize("loader", [load_flux2_vae_decoder, load_flux2_vae_encoder])
def test_both_loaders_name_a_missing_file(tmp_path: Path, loader) -> None:
    with pytest.raises(FileNotFoundError, match=re.escape("absent.safetensors")):
        loader(tmp_path / "absent.safetensors", CPU)
