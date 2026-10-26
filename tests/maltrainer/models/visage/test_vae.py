from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from maltrainer.models.visage.vae import load_visage_vae
from maltrainer.models.visage.vae_decoder import ViTDecoder, axial_rope_tables, rotate
from maltrainer.models.visage.vae_layers import HaarDWT, fold_mean, groups

from .tiny_model import LATENT, TINY_VAE, tiny_vae

CPU = torch.device("cpu")


def _pixels(side: int = 32, batch: int = 1) -> torch.Tensor:
    return torch.rand(batch, 3, side, side, generator=torch.Generator().manual_seed(4))


def _write(path: Path, prefix: str = "") -> dict[str, torch.Tensor]:
    state = {key: value.detach().clone() for key, value in tiny_vae().state_dict().items()}
    save_file({prefix + key: value for key, value in state.items()}, str(path))
    return state


def test_a_16_px_cell_becomes_one_latent_cell_and_back() -> None:
    vae = tiny_vae()

    latents = vae.encode(_pixels(48, batch=2))
    image = vae.decode(latents)

    assert latents.shape == (2, LATENT, 3, 3) and image.shape == (2, 3, 48, 48)
    assert image.min() >= 0 and image.max() <= 1


def test_encode_returns_zero_mean_unit_std_latents_by_the_stored_statistics() -> None:
    vae = tiny_vae()
    with torch.no_grad():
        vae.latent_mean.copy_(torch.arange(LATENT, dtype=torch.float32))
        vae.latent_std.fill_(2.0)
    pixels = _pixels()
    raw = vae.wavelet_encoder(pixels * 2 - 1)

    latents = vae.encode(pixels)

    assert torch.allclose(latents, (raw - torch.arange(LATENT).view(1, -1, 1, 1)) / 2.0)


def test_decode_undoes_the_normalisation_before_the_decoder() -> None:
    vae = tiny_vae()
    with torch.no_grad():
        vae.latent_mean.fill_(0.5)
        vae.latent_std.fill_(3.0)
    latents = torch.randn(1, LATENT, 2, 2)

    image = vae.decode(latents)

    assert torch.allclose(image, ((vae.vit_decoder(latents * 3.0 + 0.5) + 1) / 2).clamp(0, 1))


def test_encoding_and_decoding_leave_no_autograd_graph() -> None:
    vae = tiny_vae()

    assert not vae.encode(_pixels()).requires_grad and not vae.decode(torch.randn(1, LATENT, 2, 2)).requires_grad


def test_the_vae_file_loads_under_a_container_prefix_with_its_weights_frozen(tmp_path: Path) -> None:
    state = _write(tmp_path / "vae.safetensors", "transformer.")

    vae = load_visage_vae(tmp_path / "vae.safetensors", CPU, config=TINY_VAE)

    assert all(torch.equal(vae.state_dict()[key], value) for key, value in state.items())
    assert not vae.training and not any(p.requires_grad for p in vae.parameters())
    assert torch.equal(vae.encode(_pixels()), tiny_vae().encode(_pixels()))


def test_a_bf16_file_is_loaded_exactly_into_fp32_and_can_be_kept_in_bf16(tmp_path: Path) -> None:
    state = {key: value.bfloat16() for key, value in tiny_vae().state_dict().items()}
    save_file(state, str(tmp_path / "vae.safetensors"))

    exact = load_visage_vae(tmp_path / "vae.safetensors", CPU, config=TINY_VAE)
    small = load_visage_vae(tmp_path / "vae.safetensors", CPU, torch.bfloat16, TINY_VAE)

    assert {p.dtype for p in exact.parameters()} == {torch.float32}
    assert torch.equal(exact.vit_decoder.head.weight, state["vit_decoder.head.weight"].float())
    assert {p.dtype for p in small.parameters()} == {torch.bfloat16}
    assert small.encode(_pixels()).dtype == torch.bfloat16


def test_a_missing_file_is_named(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"gone\.safetensors"):
        load_visage_vae(tmp_path / "gone.safetensors", CPU)


def test_a_file_that_is_not_the_visage_vae_is_refused_naming_it(tmp_path: Path) -> None:
    save_file({"decoder.weight": torch.zeros(2, 2)}, str(tmp_path / "flux2.safetensors"))

    with pytest.raises(ValueError, match=r"flux2\.safetensors is not the Visage-Image VAE: Error"):
        load_visage_vae(tmp_path / "flux2.safetensors", CPU, config=TINY_VAE)


def test_the_haar_level_keeps_the_energy_and_puts_the_ll_band_first() -> None:
    x = torch.randn(2, 3, 8, 8)

    bands = HaarDWT(3)(x)

    assert bands.shape == (2, 12, 4, 4)
    assert torch.allclose(bands.square().sum(), x.square().sum(), rtol=1e-5)
    assert torch.allclose(bands[:, :3], torch.nn.functional.avg_pool2d(x, 2) * 2, atol=1e-5)


def test_the_haar_level_runs_in_fp32_inside_a_bf16_model() -> None:
    dwt = HaarDWT(3).bfloat16()
    x = torch.randn(1, 3, 4, 4, dtype=torch.bfloat16)

    assert dwt(x).dtype == torch.bfloat16


def test_fold_mean_averages_the_unshuffled_channel_groups() -> None:
    x = torch.arange(3 * 4 * 4, dtype=torch.float32).view(1, 3, 4, 4)

    out = fold_mean(x, 2, 6)

    assert out.shape == (1, 6, 2, 2)
    assert torch.allclose(out, torch.nn.functional.pixel_unshuffle(x, 2).view(1, 6, 2, 2, 2).mean(2))


@pytest.mark.parametrize(("channels", "expected"), [(496, 16), (376, 8), (120, 8), (32, 4), (16, 2), (7, 1)])
def test_group_norm_takes_the_most_groups_that_leave_eight_channels_each(channels: int, expected: int) -> None:
    assert groups(channels) == expected


def test_the_decoder_only_takes_a_square_latent_grid() -> None:
    with pytest.raises(ValueError, match="latent grid must be square"):
        ViTDecoder(TINY_VAE)(torch.randn(1, LATENT, 2, 3))


def test_the_axial_rope_has_a_row_half_and_a_column_half() -> None:
    cos, sin = axial_rope_tables(32, 2, CPU)

    assert cos.shape == sin.shape == (4, 16)
    assert torch.equal(cos[0], torch.ones(16))  # cell (0, 0)
    # cell (0, 1) is a column step, cell (1, 0) a row step
    assert torch.equal(cos[1, :8], torch.ones(8)) and not torch.equal(cos[1, 8:], torch.ones(8))
    assert torch.equal(cos[2, 8:], torch.ones(8)) and not torch.equal(cos[2, :8], torch.ones(8))
    x = torch.randn(1, 1, 4, 32)
    assert torch.allclose(rotate(x, cos, sin).norm(dim=-1), x.norm(dim=-1), atol=1e-5)
