from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import maltrainer.models.minimax_h3.audio_vae as av
from maltrainer.models.minimax_h3.audio_vae import (
    DEFAULT_LATENTS_MEAN,
    DEFAULT_LATENTS_STD,
    HOP_LENGTH,
    LATENT_CHANNELS,
    LATENTS_PER_SECOND,
    MiniMaxH3AudioEncoder,
    load_minimax_h3_audio_encoder,
)

CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def small_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the 2048-wide DAC encoder and projection; strides, hop and 32 latents stay real."""
    real_dac, real_proj = av.DacEncoder, av.AttnProjection
    monkeypatch.setattr(av, "DacEncoder", lambda *_: real_dac(2, (2, 4, 4, 5, 5), 64))
    monkeypatch.setattr(av, "AttnProjection", lambda *_args, **_kw: real_proj(64, LATENT_CHANNELS, num_heads=2))


def _model(seed: int = 0) -> MiniMaxH3AudioEncoder:
    torch.manual_seed(seed)
    model = MiniMaxH3AudioEncoder()
    for p in model.parameters():
        p.data = torch.randn_like(p) * 0.3
    return model.eval()


def _write(state: dict, path: Path) -> Path:
    save_file({k: v.contiguous() for k, v in state.items()}, str(path))
    return path


def test_the_hop_is_the_product_of_the_strides_giving_forty_latent_frames_per_second() -> None:
    assert HOP_LENGTH == 2 * 4 * 4 * 5 * 5 and LATENTS_PER_SECOND == 40
    assert len(DEFAULT_LATENTS_MEAN) == len(DEFAULT_LATENTS_STD) == LATENT_CHANNELS


@pytest.mark.parametrize("length, frames", [(1600, 2), (1601, 3), (1000, 2), (1, 1), (800, 1)])
def test_a_stereo_waveform_encodes_to_thirty_two_channels_by_two_by_ceil_length_over_hop(
    length: int, frames: int
) -> None:
    out = _model().encode(torch.randn(3, 2, length).clamp(-1, 1))

    assert out.shape == (3, LATENT_CHANNELS, 2, frames) and out.dtype == torch.float32


def test_the_latents_are_the_posterior_mean_standardised_with_the_per_channel_statistics() -> None:
    model = _model()
    wave = torch.randn(1, 2, 1600)

    x = model.encoder(wave.reshape(2, 1, 1600))
    x = model.pre_block(x.transpose(1, 2)).transpose(1, 2)
    raw = model.mean_proj(x)
    expected = (raw - model.latents_mean.view(1, -1, 1)) / model.latents_std.view(1, -1, 1)

    got = model.encode(wave)

    torch.testing.assert_close(got[:, :, 1], expected[1:2])
    torch.testing.assert_close(got[:, :, 0], expected[0:1])


def test_the_two_stereo_channels_go_through_the_mono_encoder_independently() -> None:
    model = _model()
    left, right = torch.randn(1, 1, 1600), torch.randn(1, 1, 1600)

    both = model.encode(torch.cat([left, right], dim=1))
    swapped = model.encode(torch.cat([right, left], dim=1))

    torch.testing.assert_close(both[:, :, 0], swapped[:, :, 1])
    torch.testing.assert_close(both[:, :, 1], swapped[:, :, 0])
    assert not torch.allclose(both[:, :, 0], both[:, :, 1])


def test_a_short_tail_is_zero_padded_so_it_matches_an_explicitly_padded_waveform() -> None:
    model = _model()
    wave = torch.randn(1, 2, 1000)
    padded = torch.nn.functional.pad(wave, (0, 600))

    torch.testing.assert_close(model.encode(wave), model.encode(padded))


def test_half_precision_input_is_encoded_in_fp32() -> None:
    model = _model()
    wave = torch.randn(1, 2, 800).clamp(-1, 1).to(torch.bfloat16)

    out = model.encode(wave)

    assert out.dtype == torch.float32
    torch.testing.assert_close(out, model.encode(wave.float()))


def _state(model: MiniMaxH3AudioEncoder, with_stats: bool = True) -> dict:
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if not with_stats:
        del state["latents_mean"], state["latents_std"]
    return state


def test_a_comfy_single_file_loads_into_a_frozen_fp32_eval_encoder_that_reproduces_the_source(tmp_path: Path) -> None:
    source = _model()
    state = _state(source)
    state["decoder.conv.weight"] = torch.zeros(1)  # the BigVGAN decoder is skipped
    state["latents_mean"] = torch.full((LATENT_CHANNELS,), 0.5)

    loaded = load_minimax_h3_audio_encoder(_write(state, tmp_path / "vae.safetensors"), CPU)

    assert not loaded.training and not any(p.requires_grad for p in loaded.parameters())
    assert {p.dtype for p in loaded.parameters()} == {torch.float32}
    assert torch.equal(loaded.latents_mean, torch.full((LATENT_CHANNELS,), 0.5))
    wave = torch.randn(1, 2, 800)
    source.latents_mean.fill_(0.5)
    torch.testing.assert_close(loaded.encode(wave), source.encode(wave))


def test_half_precision_weights_are_upcast_to_fp32_on_load(tmp_path: Path) -> None:
    state = {k: v.to(torch.bfloat16) for k, v in _state(_model()).items()}

    loaded = load_minimax_h3_audio_encoder(_write(state, tmp_path / "vae.safetensors"), CPU)

    assert {p.dtype for p in loaded.parameters()} == {torch.float32}
    assert loaded.latents_std.dtype == torch.float32


def test_the_released_weight_norm_file_is_folded_and_falls_back_to_the_config_statistics(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    source = _model()
    state = _state(source, with_stats=False)
    for key in [k for k in state if k.startswith("encoder.") and k.endswith(".weight") and state[k].ndim == 3]:
        weight = state.pop(key)
        norm = weight.flatten(1).norm(dim=1).view(-1, 1, 1)
        state[key[: -len("weight")] + "weight_g"] = norm
        state[key[: -len("weight")] + "weight_v"] = weight * 3.0  # any scale: g alone sets the magnitude

    with caplog.at_level(logging.INFO, logger=av.logger.name):
        loaded = load_minimax_h3_audio_encoder(_write(state, tmp_path / "vae.safetensors"), CPU)

    assert "carries no latent statistics" in caplog.text
    assert loaded.latents_mean.tolist() == pytest.approx(list(DEFAULT_LATENTS_MEAN))
    wave = torch.randn(1, 2, 800)
    torch.testing.assert_close(loaded.encode(wave), source.encode(wave), atol=1e-4, rtol=1e-4)


def test_a_file_missing_encoder_tensors_is_refused_with_examples(tmp_path: Path) -> None:
    state = _state(_model())
    del state["mean_proj.weight"]
    del state["pre_block.proj.bias"]

    with pytest.raises(ValueError, match=r"did not supply 2 audio VAE encoder tensor.*mean_proj.weight"):
        load_minimax_h3_audio_encoder(_write(state, tmp_path / "vae.safetensors"), CPU)


def test_a_file_with_an_extra_encoder_tensor_is_refused_with_examples(tmp_path: Path) -> None:
    state = _state(_model())
    state["encoder.block.99.weight"] = torch.zeros(1)

    with pytest.raises(ValueError, match=r"1 encoder tensor.*not expect.*encoder.block.99.weight"):
        load_minimax_h3_audio_encoder(_write(state, tmp_path / "vae.safetensors"), CPU)
