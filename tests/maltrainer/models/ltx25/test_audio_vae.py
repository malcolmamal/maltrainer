from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from maltrainer.models.ltx25.audio_features import (
    HOP,
    MEL_BINS,
    SAMPLE_RATE,
    hz_to_mel,
    log_mel,
    mel_filterbank,
    resample,
)
from maltrainer.models.ltx25.audio_vae import AudioVAEEncoder, Encoder, load_audio_vae_encoder

from .tiny_model import randomise

CPU = torch.device("cpu")
DDCONFIG: dict[str, Any] = {
    "double_z": True,
    "mel_bins": 64,
    "z_channels": 8,
    "in_channels": 2,
    "out_ch": 2,
    "ch": 4,
    "ch_mult": [1, 2, 4],
    "num_res_blocks": 1,
    "attn_resolutions": [],
    "mid_block_add_attention": False,
    "norm_type": "pixel",
    "causality_axis": "height",
}


def _vae() -> AudioVAEEncoder:
    model = AudioVAEEncoder(DDCONFIG)
    randomise(model)
    return model.eval()


def _wave(seconds: float, channels: int = 2, rate: int = SAMPLE_RATE) -> torch.Tensor:
    return torch.randn(1, channels, round(seconds * rate), generator=torch.Generator().manual_seed(4)) * 0.2


# ------------------------------------------------------------------------------------------ features


def test_a_waveform_gives_one_mel_frame_per_hop_plus_one_with_64_bins() -> None:
    mel = log_mel(_wave(1.0))

    assert mel.shape == (1, 2, SAMPLE_RATE // HOP + 1, MEL_BINS)


def test_silence_sits_at_the_log_floor() -> None:
    assert torch.unique(log_mel(torch.zeros(1, 1, 3200))).tolist() == pytest.approx([math.log(1e-5)])


def test_a_louder_signal_has_a_higher_log_mel() -> None:
    quiet, loud = _wave(0.5), _wave(0.5) * 10

    assert log_mel(loud).mean() > log_mel(quiet).mean()


def test_the_mel_filterbank_covers_the_band_with_area_normalised_triangles() -> None:
    bank = mel_filterbank(MEL_BINS, SAMPLE_RATE, 513)

    assert bank.shape == (513, MEL_BINS) and bank.min().item() >= 0 and (bank.sum(dim=0) > 0).all()
    assert hz_to_mel(500.0) == pytest.approx(500.0 / (200.0 / 3)) and hz_to_mel(2000.0) > hz_to_mel(1000.0) == 15.0


def test_resampling_scales_the_length_and_keeps_a_low_tone() -> None:
    t = torch.arange(44100) / 44100
    tone = torch.sin(2 * math.pi * 440 * t)[None, None]

    out = resample(tone, 44100, 16000)

    assert out.shape[-1] == 16000 and resample(tone, 44100, 44100) is tone
    reference = torch.sin(2 * math.pi * 440 * torch.arange(16000) / 16000)
    assert (out[0, 0, 200:-200] - reference[200:-200]).abs().max().item() < 0.01


# ------------------------------------------------------------------------------------------ encoder


def test_the_latents_have_eight_channels_sixteen_bins_and_a_quarter_of_the_frames() -> None:
    latents = _vae().encode(_wave(2.0), SAMPLE_RATE)

    assert latents.shape == (1, 8, 51, 16)  # 201 mel frames -> (201 - 1) // 2 + 1 = 101 -> (101 - 1) // 2 + 1 = 51


def test_the_number_of_latents_follows_the_window_the_data_module_cuts() -> None:
    latents_per_clip = 50  # two seconds at 25 latent frames per second
    samples = (4 * latents_per_clip - 3) * HOP

    latents = _vae().encode(torch.randn(1, 2, samples) * 0.1, SAMPLE_RATE)

    assert latents.shape[2] == latents_per_clip


def test_mono_audio_is_duplicated_into_both_channels_and_more_than_two_channels_are_refused() -> None:
    vae = _vae()
    mono = _wave(0.5, channels=1)

    assert torch.allclose(vae.encode(mono, SAMPLE_RATE), vae.encode(mono.expand(-1, 2, -1), SAMPLE_RATE))
    with pytest.raises(ValueError, match="audio must have 2 channels, got 3"):
        vae.encode(_wave(0.5, channels=3), SAMPLE_RATE)


def test_audio_at_another_rate_is_resampled_first() -> None:
    vae = _vae()

    assert vae.encode(_wave(1.0, rate=44100), 44100).shape == vae.encode(_wave(1.0), SAMPLE_RATE).shape


def test_the_latent_at_a_time_depends_only_on_the_sound_before_it() -> None:
    vae = _vae()
    wave = _wave(2.0)
    changed = wave.clone()
    changed[:, :, -SAMPLE_RATE // 2 :] += 0.5

    first, second = vae.encode(wave, SAMPLE_RATE), vae.encode(changed, SAMPLE_RATE)

    assert torch.allclose(first[:, :, :20], second[:, :, :20], atol=1e-5) and not torch.allclose(
        first[:, :, -5:], second[:, :, -5:]
    )


def test_the_latents_are_normalised_per_channel_and_bin_in_patch_space() -> None:
    vae = _vae()
    wave = _wave(1.0)
    stats = vae.per_channel_statistics
    with torch.no_grad():
        stats.get_buffer("mean-of-means").zero_()
        stats.get_buffer("std-of-means").fill_(1.0)
    raw = vae.encode(wave, SAMPLE_RATE)
    with torch.no_grad():
        stats.get_buffer("mean-of-means").fill_(1.0)
        stats.get_buffer("std-of-means").fill_(4.0)

    assert torch.allclose(vae.encode(wave, SAMPLE_RATE), (raw - 1.0) / 4.0, atol=1e-5)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"attn_resolutions": [16]}, "attention-free"),
        ({"mid_block_add_attention": True}, "attention-free"),
        ({"norm_type": "group"}, "attention-free"),
        ({"causality_axis": "width"}, "time-causal"),
    ],
)
def test_the_variants_this_trainer_does_not_run_are_refused(change: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Encoder({**DDCONFIG, **change})


# ------------------------------------------------------------------------------------------ loading


def _write(path: Path, model: AudioVAEEncoder, config: bool = True, drop: str = "") -> Path:
    state = {f"audio_vae.{k}": v.contiguous() for k, v in model.state_dict().items() if not drop or drop not in k}
    state["vocoder.conv_pre.weight"] = torch.zeros(1)  # the file also holds the vocoder
    metadata = {"config": json.dumps({"audio_vae": {"model": {"params": {"ddconfig": DDCONFIG}}}})} if config else {}
    save_file(state, str(path), metadata=metadata)
    return path


def test_the_loader_reads_the_encoder_and_its_statistics_from_the_file(tmp_path: Path) -> None:
    source = _vae()
    path = _write(tmp_path / "audio.safetensors", source)

    loaded = load_audio_vae_encoder(path, CPU)

    assert torch.allclose(loaded.encode(_wave(0.5), SAMPLE_RATE), source.encode(_wave(0.5), SAMPLE_RATE), atol=1e-6)
    assert not loaded.training and not any(p.requires_grad for p in loaded.parameters())


def test_the_loader_refuses_a_missing_file_a_file_without_config_and_missing_tensors(tmp_path: Path) -> None:
    source = _vae()

    with pytest.raises(FileNotFoundError, match=r"LTX audio VAE not found at .*gone"):
        load_audio_vae_encoder(tmp_path / "gone.safetensors", CPU)
    with pytest.raises(ValueError, match="no config metadata"):
        load_audio_vae_encoder(_write(tmp_path / "bare.safetensors", source, config=False), CPU)
    with pytest.raises(ValueError, match=r"lacks audio encoder tensors.*conv_out"):
        load_audio_vae_encoder(_write(tmp_path / "short.safetensors", source, drop="conv_out"), CPU)
