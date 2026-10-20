"""The Wan 2.1 still-image encoder behind Krea 2's latents, checked against its 3D original."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from maltrainer.models.krea2.vae import (
    LATENTS_MEAN,
    LATENTS_STD,
    AttentionBlock,
    CausalConv2d,
    Krea2VAE,
    Resample,
    ResidualBlock,
    RMSNorm,
    WanImageEncoder,
    load_krea2_vae,
)

CPU = torch.device("cpu")


def seeded(module: torch.nn.Module, seed: int = 0) -> torch.nn.Module:
    """CausalConv2d allocates its weight with ``torch.empty`` (checkpoints always overwrite it): fill it."""
    generator = torch.Generator().manual_seed(seed)
    for param in module.parameters():
        param.data = torch.randn(param.shape, generator=generator) * 0.2
    return module


def test_latent_statistics_cover_all_16_channels():
    assert len(LATENTS_MEAN) == len(LATENTS_STD) == 16 and min(LATENTS_STD) > 0


@pytest.mark.parametrize("kernel, padding", [(3, 1), (1, 0)])
def test_causal_conv_on_one_frame_equals_the_causal_3d_conv(kernel, padding):
    torch.manual_seed(0)
    conv = seeded(CausalConv2d(3, 5, kernel, padding))
    x = torch.randn(2, 3, 6, 7)
    video = x.unsqueeze(2)  # one frame
    video = F.pad(video, (0, 0, 0, 0, kernel - 1, 0))  # causal: zero frames in front
    expected = F.conv3d(video, conv.weight, conv.bias, padding=(0, padding, padding)).squeeze(2)
    assert torch.allclose(conv(x), expected, atol=1e-5)


def test_causal_conv_casts_its_weights_to_the_input_dtype():
    out = seeded(CausalConv2d(2, 2, 1))(torch.randn(1, 2, 3, 3, dtype=torch.bfloat16))
    assert out.dtype == torch.bfloat16


def test_wan_rms_norm_is_unit_norm_over_channels_times_sqrt_c_times_gamma():
    norm = RMSNorm(4, images=False)
    norm.gamma.data = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(4, 1, 1, 1)
    x = torch.randn(2, 4, 3, 3)
    out = norm(x)
    unit = x / x.norm(dim=1, keepdim=True)
    assert torch.allclose(out, unit * 2.0 * torch.tensor([1.0, 2.0, 3.0, 4.0]).view(1, 4, 1, 1), atol=1e-5)
    assert RMSNorm(4, images=True).gamma.shape == (4, 1, 1)


def test_residual_block_adds_a_1x1_shortcut_only_when_channels_change():
    assert isinstance(ResidualBlock(4, 4).shortcut, torch.nn.Identity)
    block = seeded(ResidualBlock(4, 8))
    assert isinstance(block.shortcut, CausalConv2d)
    x = torch.randn(1, 4, 5, 5)
    assert block(x).shape == (1, 8, 5, 5)
    same = seeded(ResidualBlock(4, 4))
    same.residual[-1].weight.data.zero_()
    same.residual[-1].bias.data.zero_()
    assert torch.allclose(same(x), x)  # zeroed residual branch: identity


def test_attention_block_is_single_head_attention_over_pixels_with_a_residual():
    torch.manual_seed(0)
    block = AttentionBlock(6)
    x = torch.randn(1, 6, 3, 2)
    q, k, v = block.to_qkv(block.norm(x)).chunk(3, dim=1)
    tokens = lambda t: t.reshape(6, 6).T  # noqa: E731  (pixels, channels)
    weights = torch.softmax(tokens(q) @ tokens(k).T / math.sqrt(6), dim=-1)
    attended = (weights @ tokens(v)).T.reshape(1, 6, 3, 2)
    assert torch.allclose(block(x), block.proj(attended) + x, atol=1e-5)


def test_resample_halves_the_resolution_with_bottom_right_padding():
    layer = Resample(2, temporal=False)
    x = torch.randn(1, 2, 8, 8)
    out = layer(x)
    assert out.shape == (1, 2, 4, 4)
    conv = layer.resample[1]
    padded = F.pad(x, (0, 1, 0, 1))
    assert torch.allclose(out, F.conv2d(padded, conv.weight, conv.bias, stride=2), atol=1e-6)


def test_encoder_reduces_resolution_8x_and_emits_mean_and_logvar_channels():
    out = seeded(WanImageEncoder(dim=4))(torch.randn(1, 3, 16, 24))
    assert out.shape == (1, 32, 2, 3)


def test_encoder_layer_widths_follow_dim_mult():
    encoder = WanImageEncoder(dim=4)
    assert encoder.conv1.weight.shape[0] == 4
    assert encoder.head[-1].weight.shape[:2] == (32, 16)  # 16 = 4 * the last multiplier
    assert sum(isinstance(m, Resample) for m in encoder.downsamples) == 3


class _Recorder(torch.nn.Module):
    """Stands in for the encoder: remembers its input, returns mean = 5 and a junk half."""

    def forward(self, x):
        self.seen = x
        return torch.cat([torch.full((x.shape[0], 16, 1, 1), 5.0), torch.full((x.shape[0], 16, 1, 1), 99.0)], dim=1)


def test_encode_maps_pixels_to_minus_one_one_and_normalizes_the_posterior_mean():
    vae = Krea2VAE(dim=4)
    vae.encoder = _Recorder()
    vae.conv1.weight.data = torch.eye(32).view(32, 32, 1, 1, 1)
    vae.conv1.bias.data.zero_()
    out = vae.encode(torch.tensor([0.0, 0.5, 1.0]).view(1, 3, 1, 1).expand(1, 3, 2, 2))
    assert vae.encoder.seen[0, :, 0, 0].tolist() == [-1.0, 0.0, 1.0]
    assert out.shape == (1, 16, 1, 1) and out.dtype == torch.float32  # logvar half dropped
    expected = [(5.0 - m) / s for m, s in zip(LATENTS_MEAN, LATENTS_STD, strict=False)]
    assert out[0, :, 0, 0].tolist() == pytest.approx(expected, rel=1e-5)


def test_encode_returns_latents_8x_smaller_with_16_channels():
    vae = seeded(Krea2VAE(dim=4))
    assert vae.encode(torch.rand(2, 3, 16, 16)).shape == (2, 16, 2, 2)


def test_stat_buffers_are_not_saved():
    assert not [k for k in Krea2VAE(dim=4).state_dict() if k.startswith("latents_")]


@pytest.fixture
def vae_file(tmp_path):
    vae = seeded(Krea2VAE(dim=4))
    state = {k: v.clone() for k, v in vae.state_dict().items()}
    state["decoder.conv1.weight"] = torch.zeros(3)  # decoder tensors are ignored
    state["encoder.downsamples.3.time_conv.weight"] = torch.zeros(3)  # as are the temporal convs
    path = tmp_path / "wan_2.1_vae.safetensors"
    save_file(state, str(path))
    return vae, path


def test_loader_infers_width_ignores_decoder_and_reproduces_the_encoder(vae_file):
    original, path = vae_file
    loaded = load_krea2_vae(path, CPU)
    image = torch.rand(1, 3, 16, 16)
    assert loaded.encoder.conv1.weight.shape[0] == 4
    assert torch.allclose(loaded.encode(image), original.encode(image), atol=1e-6)
    assert not loaded.training and not any(p.requires_grad for p in loaded.parameters())


def test_loader_casts_to_the_requested_dtype(vae_file):
    loaded = load_krea2_vae(vae_file[1], CPU, dtype=torch.bfloat16)
    assert loaded.conv1.weight.dtype == torch.bfloat16
    assert loaded.encode(torch.rand(1, 3, 16, 16)).dtype == torch.float32


def test_loader_refuses_a_file_missing_encoder_tensors(tmp_path):
    state = {k: v for k, v in seeded(Krea2VAE(dim=4)).state_dict().items() if k != "conv1.bias"}
    path = tmp_path / "bad.safetensors"
    save_file(state, str(path))
    with pytest.raises(ValueError, match=r"is not a Wan 2\.1 VAE .*missing \['conv1\.bias'\]"):
        load_krea2_vae(path, CPU)


def test_a_file_without_the_encoder_is_read_at_the_default_width_and_refused(tmp_path):
    path = tmp_path / "other.safetensors"
    save_file({"unet.weight": torch.zeros(1)}, str(path))
    with pytest.raises(ValueError, match=r"is not a Wan 2\.1 VAE"):
        load_krea2_vae(path, CPU)
