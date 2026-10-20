"""Stand-ins for what a Qwen-Image-2.1 run loads from giant checkpoints: the VAE, the text encoder and the
transformer. The transformer is the repo's own tiny model; everything between the loaders is the trainer's real code."""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional

from maltrainer.models.qwen_image_2_1.model import QwenImage21Transformer

LATENT_CHANNELS = 4
TEXT_DIM = 6


class FakeVAE:
    """Pixels [B, 3, H, W] in [0, 1] -> latents [B, 4, H/16, W/16]: a fixed linear map of the 16 px cell means."""

    def __init__(self) -> None:
        self.projection = torch.randn(LATENT_CHANNELS, 3, generator=torch.Generator().manual_seed(11))
        self.encoded: list[tuple[tuple[int, ...], torch.dtype, bool]] = []

    def encode(self, pixels: torch.Tensor, deterministic: bool) -> torch.Tensor:
        self.encoded.append((tuple(pixels.shape), pixels.dtype, deterministic))
        cells = functional.avg_pool2d(pixels.float(), 16) - 0.5
        return torch.einsum("oc,bchw->bohw", self.projection, cells)


class FakeTextEncoder:
    """A caption -> [1, L, 6] embedding from a generator seeded by the caption; L is its word count (>= 1)."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def encode_prompts(self, prompts: list[str], device: torch.device) -> torch.Tensor:
        caption = prompts[0]
        self.prompts.append(caption)
        generator = torch.Generator().manual_seed(zlib.crc32(caption.encode("utf-8")))
        return torch.randn(1, max(1, len(caption.split())), TEXT_DIM, generator=generator).to(device)


def _tiny() -> QwenImage21Transformer:
    """The tiny transformer with fixed weights, built without touching the global RNG the trainer seeded."""
    with torch.random.fork_rng():
        torch.manual_seed(0)
        return QwenImage21Transformer(
            in_channels=LATENT_CHANNELS,
            out_channels=LATENT_CHANNELS,
            num_layers=2,
            dim=16,
            heads=2,
            dim_head=8,
            mlp_ratio=2,
            context_in_dim=TEXT_DIM,
            axes_dims_rope=(2, 2, 4),
        )


@dataclass
class Fakes:
    vae: FakeVAE = field(default_factory=FakeVAE)
    text: FakeTextEncoder = field(default_factory=FakeTextEncoder)
    loads: dict[str, int] = field(default_factory=lambda: {"vae": 0, "text": 0, "transformer": 0})
    vae_options: list[dict[str, object]] = field(default_factory=list)
    text_options: list[tuple[str | None, torch.device]] = field(default_factory=list)
    transformer_paths: list[str] = field(default_factory=list)
    grad_precisions: list[str] = field(default_factory=list)
    dtypes: list[torch.dtype] = field(default_factory=list)

    def reference_state(self, dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
        """The base checkpoint's tensors as the trainer's loader leaves them (a fresh copy each call)."""
        return {k: v.detach().clone().to(dtype) for k, v in _tiny().state_dict().items()}


def install_fakes(monkeypatch: pytest.MonkeyPatch, module: object) -> Fakes:
    """Replace the three loaders ``module`` imported with ones returning the fakes."""
    fakes = Fakes()

    def vae(_path: str, device: torch.device, dtype: torch.dtype, include_decoder: bool) -> FakeVAE:
        fakes.loads["vae"] += 1
        fakes.vae_options.append({"device": device, "dtype": dtype, "include_decoder": include_decoder})
        return fakes.vae

    def text(_path: str, asset_dir: str | None, device: torch.device) -> FakeTextEncoder:
        fakes.loads["text"] += 1
        fakes.text_options.append((asset_dir, device))
        return fakes.text

    def transformer(path: str, device: torch.device, dtype: torch.dtype, grad_precision: str) -> QwenImage21Transformer:
        fakes.loads["transformer"] += 1
        fakes.transformer_paths.append(Path(path).name)
        fakes.grad_precisions.append(grad_precision)
        fakes.dtypes.append(dtype)
        return _tiny().requires_grad_(False).to(device=device, dtype=dtype).eval()

    monkeypatch.setattr(module, "load_qwen_image_2_1_vae", vae)
    monkeypatch.setattr(module, "load_qwen_image_2_1_text_encoder", text)
    monkeypatch.setattr(module, "load_qwen_image_2_1_transformer", transformer)
    return fakes


def skip_housekeeping(monkeypatch: pytest.MonkeyPatch, module: object, commit: str = "abc1234") -> None:
    """The trainer's two slow environment calls, a full ``gc.collect()`` and ``git``, answered at once."""
    monkeypatch.setattr(module, "gc", SimpleNamespace(collect=lambda: 0))
    monkeypatch.setattr(module, "git_commit", lambda: commit)
