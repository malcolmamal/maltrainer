"""
Waveform -> log-mel spectrogram, the input of the LTX-2 audio VAE.

Adapted from TorchAudio (BSD-2-Clause) as ComfyUI does in comfy/audio.py, which this reproduces: a Hann-windowed STFT
magnitude, a Slaney mel filterbank with area normalisation, ``log(clamp(mel, 1e-5))``, laid out as
[B, channels, time, mel].
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

SAMPLE_RATE, N_FFT, HOP, MEL_BINS = 16000, 1024, 160, 64
MEL_FLOOR = 1e-5


def resample(
    waveform: torch.Tensor, orig_freq: int, new_freq: int, width_factor: int = 6, rolloff: float = 0.99
) -> torch.Tensor:
    """Band-limited sinc resampling (Hann window) along the last dimension."""
    if orig_freq == new_freq:
        return waveform
    divisor = math.gcd(orig_freq, new_freq)
    orig_freq, new_freq = orig_freq // divisor, new_freq // divisor
    base = min(orig_freq, new_freq) * rolloff
    width = math.ceil(width_factor * orig_freq / base)
    idx = torch.arange(-width, width + orig_freq, dtype=waveform.dtype, device=waveform.device)[None, None] / orig_freq
    t = torch.arange(0, -new_freq, -1, dtype=waveform.dtype, device=waveform.device)[:, None, None] / new_freq + idx
    t = (t * base).clamp_(-width_factor, width_factor)
    window = torch.cos(t * math.pi / width_factor / 2) ** 2
    t *= math.pi
    kernel = torch.where(t == 0, 1.0, t.sin() / t) * window * (base / orig_freq)
    length = waveform.shape[-1]
    flat = F.pad(waveform.reshape(-1, length), (width, width + orig_freq))
    out = F.conv1d(flat[:, None], kernel, stride=orig_freq).transpose(1, 2).reshape(flat.shape[0], -1)
    out = out[..., : math.ceil(np.float32(new_freq * length / orig_freq))]
    return out.reshape(*waveform.shape[:-1], out.shape[-1])


def hz_to_mel(freq: float) -> float:
    if freq >= 1000.0:
        return 15.0 + math.log(freq / 1000.0) / (math.log(6.4) / 27.0)
    return freq / (200.0 / 3)


def mel_filterbank(n_mels: int, sample_rate: int, n_stft: int) -> torch.Tensor:
    """Slaney mel filterbank [n_stft, n_mels] from 0 Hz to Nyquist, area normalised."""
    all_freqs = torch.linspace(0, sample_rate // 2, n_stft)
    mels = torch.linspace(hz_to_mel(0.0), hz_to_mel(sample_rate // 2), n_mels + 2)
    freqs = (200.0 / 3) * mels
    log_region = mels >= 15.0
    freqs[log_region] = 1000.0 * torch.exp((math.log(6.4) / 27.0) * (mels[log_region] - 15.0))
    diff = freqs[1:] - freqs[:-1]
    slopes = freqs.unsqueeze(0) - all_freqs.unsqueeze(1)
    bank = torch.minimum(-slopes[:, :-2] / diff[:-1], slopes[:, 2:] / diff[1:]).clamp_min(0)
    return bank * (2.0 / (freqs[2:] - freqs[:-2])).unsqueeze(0)


def log_mel(waveform: torch.Tensor) -> torch.Tensor:
    """[B, channels, samples] at 16 kHz -> log-mel [B, channels, frames, 64]."""
    device = waveform.device
    shape = waveform.shape
    window = torch.hann_window(N_FFT, device=device)
    spec = torch.stft(
        waveform.reshape(-1, shape[-1]),
        N_FFT,
        HOP,
        N_FFT,
        window,
        center=True,
        pad_mode="reflect",
        onesided=True,
        return_complex=True,
    )
    magnitude = spec.reshape(*shape[:-1], *spec.shape[-2:]).abs()
    bank = mel_filterbank(MEL_BINS, SAMPLE_RATE, N_FFT // 2 + 1).to(device)
    mel = (magnitude.transpose(-1, -2) @ bank).transpose(-1, -2)
    return torch.log(torch.clamp(mel, min=MEL_FLOOR)).permute(0, 1, 3, 2).contiguous()
