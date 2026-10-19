from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from maltrainer.models.flux3.vae import Flux2VAEDecoder, Flux2VAEEncoder, load_flux2_vae_decoder, load_flux2_vae_encoder

CPU = torch.device("cpu")
Z, EMBED = 4, 2


def _models() -> tuple:
    torch.manual_seed(0)
    encoder = Flux2VAEEncoder(ch=32, z_channels=Z, embed_dim=EMBED).eval()
    decoder = Flux2VAEDecoder(ch=32, z_channels=Z, embed_dim=EMBED).eval()
    for model in (encoder, decoder):
        model.bn_running_mean.copy_(torch.linspace(-1, 1, 4 * EMBED))
        model.bn_running_var.copy_(torch.linspace(0.5, 2, 4 * EMBED))
    return encoder, decoder


def _file(path: Path, prefix: str = "", nested: bool = False, decoder_keys: bool = True) -> Path:
    encoder, decoder = _models()
    state = {k: v for k, v in encoder.state_dict().items() if not k.startswith("bn_")}
    if decoder_keys:
        state.update({k: v for k, v in decoder.state_dict().items() if not k.startswith("bn_")})
    if nested:
        state = {
            k.replace("quant_conv.", "encoder.quant_conv.") if k.startswith("quant_conv") else k: v
            for k, v in state.items()
        }
        state = {
            k.replace("post_quant_conv.", "decoder.post_quant_conv.", 1) if k.startswith("post_") else k: v
            for k, v in state.items()
        }
    state["bn.running_mean"] = encoder.bn_running_mean
    state["bn.running_var"] = encoder.bn_running_var
    save_file({prefix + k: v.contiguous() for k, v in state.items()}, str(path))
    return path


@pytest.mark.parametrize(("prefix", "nested"), [("", False), ("first_stage_model.", False), ("vae.", True)])
def test_the_encoder_loads_from_any_key_layout_and_encodes_like_its_source(
    tmp_path: Path, prefix: str, nested: bool
) -> None:
    source, _ = _models()
    images = torch.rand(1, 3, 32, 32, generator=torch.Generator().manual_seed(1))

    loaded = load_flux2_vae_encoder(_file(tmp_path / "vae.safetensors", prefix, nested), CPU)

    torch.testing.assert_close(loaded.encode(images), source.encode(images))
    assert not loaded.training and not any(p.requires_grad for p in loaded.parameters())


def test_without_decoder_weights_the_encoder_expects_32_latent_channels(tmp_path: Path) -> None:
    path = _file(tmp_path / "enc.safetensors", decoder_keys=False)

    with pytest.raises(RuntimeError, match="size mismatch"):
        load_flux2_vae_encoder(path, CPU)


@pytest.mark.parametrize(("prefix", "nested"), [("", False), ("vae.", True)])
def test_the_decoder_loads_and_decodes_like_its_source(tmp_path: Path, prefix: str, nested: bool) -> None:
    _, source = _models()
    latents = torch.randn(1, 4 * EMBED, 2, 2, generator=torch.Generator().manual_seed(1))

    loaded = load_flux2_vae_decoder(_file(tmp_path / "vae.safetensors", prefix, nested), CPU, dtype=torch.float32)

    torch.testing.assert_close(loaded.decode(latents), source.decode(latents))


def test_a_decoder_is_refused_from_a_file_without_decoder_weights(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="has no decoder"):
        load_flux2_vae_decoder(_file(tmp_path / "enc.safetensors", decoder_keys=False), CPU)


@pytest.mark.parametrize("loader", [load_flux2_vae_encoder, load_flux2_vae_decoder])
def test_a_missing_file_is_reported(tmp_path: Path, loader) -> None:
    with pytest.raises(FileNotFoundError, match="Flux2 VAE not found"):
        loader(tmp_path / "absent.safetensors", CPU)


@pytest.mark.parametrize("loader", [load_flux2_vae_encoder, load_flux2_vae_decoder])
def test_a_vae_without_batch_norm_statistics_is_refused(tmp_path: Path, loader) -> None:
    path = tmp_path / "sd.safetensors"
    save_file({"encoder.conv_in.weight": torch.zeros(1)}, str(path))

    with pytest.raises(ValueError, match="no batch-norm latent statistics"):
        loader(path, CPU)
