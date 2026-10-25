"""The Wan 2.1 still-image decoder behind Krea 2's previews: its layers against their definitions, and its file."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from maltrainer.models.krea2.vae import LATENTS_MEAN, LATENTS_STD, ResidualBlock
from maltrainer.models.krea2.vae_decoder import Krea2VAEDecoder, Upsample, WanImageDecoder, load_krea2_vae_decoder

CPU = torch.device("cpu")


def seeded(module: torch.nn.Module, seed: int = 0) -> torch.nn.Module:
    """CausalConv2d allocates its weight with ``torch.empty`` (checkpoints always overwrite it): fill it."""
    generator = torch.Generator().manual_seed(seed)
    for param in module.parameters():
        param.data = torch.randn(param.shape, generator=generator) * 0.2
    return module


def test_upsample_doubles_the_size_nearest_exact_then_halves_the_channels_with_a_3x3_conv() -> None:
    layer = seeded(Upsample(4))
    x = torch.randn(1, 4, 3, 5)
    conv = layer.resample[1]
    expected = F.conv2d(F.interpolate(x, scale_factor=2.0, mode="nearest-exact"), conv.weight, conv.bias, padding=1)
    assert torch.allclose(layer(x), expected, atol=1e-6)
    assert layer(x).shape == (1, 2, 6, 10)


def test_the_decoder_grows_the_latent_8x_into_three_channels() -> None:
    out = seeded(WanImageDecoder(dim=4))(torch.randn(1, 16, 2, 3))
    assert out.shape == (1, 3, 16, 24)


def test_decoder_widths_follow_wan_dim_mult_reversed_and_halve_after_each_upsample() -> None:
    decoder = WanImageDecoder(dim=4)
    assert decoder.conv1.weight.shape[:2] == (16, 16)  # z 16 -> 4 * the last multiplier
    stages = [m for m in decoder.upsamples if isinstance(m, ResidualBlock)]
    widths = [(m.residual[2].weight.shape[1], m.residual[2].weight.shape[0]) for m in stages]
    assert widths == [(16, 16)] * 3 + [(8, 16)] + [(16, 16)] * 2 + [(8, 8)] * 3 + [(4, 4)] * 3
    assert [m.resample[1].out_channels for m in decoder.upsamples if isinstance(m, Upsample)] == [8, 8, 4]
    assert decoder.head[-1].weight.shape[:2] == (3, 4)


class _Recorder(torch.nn.Module):
    """Stands in for the decoder: remembers its input, returns it widened to three channels of 0.5."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen = x
        return torch.full((x.shape[0], 3, 2, 2), 0.5)


def test_decode_unnormalizes_with_the_wan21_stats_and_maps_minus_one_to_one_onto_zero_to_one() -> None:
    model = Krea2VAEDecoder(dim=4)
    model.conv2.weight.data = torch.zeros(16, 16, 1, 1, 1)
    model.conv2.weight.data[range(16), range(16)] = 1.0  # the identity, so the decoder sees the un-normalized z
    model.conv2.bias.data.zero_()
    recorder = _Recorder()
    setattr(model, "decoder", recorder)  # noqa: B010 - a stand-in of another type
    z = torch.ones(1, 16, 1, 1)

    out = model.decode(z)

    expected = torch.tensor(LATENTS_MEAN) + torch.tensor(LATENTS_STD)
    assert torch.allclose(recorder.seen.flatten(), expected)
    assert torch.allclose(out, torch.full((1, 3, 2, 2), 0.75))  # (0.5 + 1) / 2


def test_decode_clamps_to_zero_to_one() -> None:
    model = Krea2VAEDecoder(dim=4)
    setattr(model, "decoder", torch.nn.Identity())  # noqa: B010 - a stand-in of another type
    seeded(model.conv2)
    out = model.decode(torch.randn(1, 16, 2, 2) * 10)
    assert out.min() >= 0.0 and out.max() <= 1.0 and out.dtype == torch.float32


def _comfy_file(path: Path, drop: str | None = None) -> Path:
    """A Wan 2.1 VAE file as ComfyUI names it: the decoder, conv2, an encoder tensor and a time_conv to ignore."""
    state = {k: v.contiguous() for k, v in seeded(Krea2VAEDecoder(dim=4)).state_dict().items() if k != drop}
    state["encoder.conv1.weight"] = torch.zeros(4, 3, 3, 3, 3)
    state["decoder.upsamples.3.time_conv.weight"] = torch.zeros(32, 16, 3, 1, 1)
    save_file(state, str(path))
    return path


def test_the_loader_reads_the_decoder_and_conv2_by_comfyui_names_and_ignores_the_rest(tmp_path: Path) -> None:
    path = _comfy_file(tmp_path / "wan.safetensors")
    reference = seeded(Krea2VAEDecoder(dim=4))

    loaded = load_krea2_vae_decoder(path, CPU, dtype=torch.bfloat16)

    assert loaded.decoder.head[-1].weight.shape[1] == 4  # dim read from the file
    assert all(not p.requires_grad and p.dtype == torch.bfloat16 for p in loaded.parameters())
    assert not loaded.training
    z = torch.randn(1, 16, 2, 2)
    assert torch.allclose(loaded.float().decode(z), reference.decode(z), atol=2e-2)


def test_a_file_missing_decoder_tensors_is_refused_with_its_name(tmp_path: Path) -> None:
    path = _comfy_file(tmp_path / "broken.safetensors", drop="conv2.bias")
    with pytest.raises(ValueError, match=r"broken\.safetensors is not a Wan 2\.1 VAE .*conv2\.bias"):
        load_krea2_vae_decoder(path, CPU)
