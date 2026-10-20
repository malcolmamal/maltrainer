"""Stand-ins for the two heavy components the FLUX.3 trainer loads: the VAE encoder and the text encoder.
Everything else in a trainer run (the DiT, the LoRA network, the optimizer, the loop) is the real code."""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional

from maltrainer.models.flux3.model import CONTEXT_DIM, LATENT_CHANNELS
from maltrainer.training import session


class FakeVAEEncoder:
    """Pixels [B, 3, H, W] in [0, 1] -> latents [B, 128, H/16, W/16]: a fixed linear map of the 16 px cell means."""

    def __init__(self) -> None:
        self.projection = torch.randn(LATENT_CHANNELS, 3, generator=torch.Generator().manual_seed(11))
        self.shapes: list[tuple[int, ...]] = []

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        self.shapes.append(tuple(pixels.shape))
        cells = functional.avg_pool2d(pixels.float(), 16) - 0.5
        return torch.einsum("oc,bchw->bohw", self.projection, cells)


class FakeTextEncoder:
    """A caption -> [L, 20480] embedding drawn from a generator seeded by the caption, L in 2..4."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def encode(self, caption: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        self.prompts.append(caption)
        seed = zlib.crc32(caption.encode("utf-8"))
        embedding = torch.randn(2 + seed % 3, CONTEXT_DIM, generator=torch.Generator().manual_seed(seed))
        return embedding.to(device, dtype)


@dataclass
class Fakes:
    encoder: FakeVAEEncoder = field(default_factory=FakeVAEEncoder)
    text: FakeTextEncoder = field(default_factory=FakeTextEncoder)
    vae_loads: int = 0
    text_loads: list[tuple[str, str | None]] = field(default_factory=list)


def install_fakes(monkeypatch: pytest.MonkeyPatch, module: object) -> Fakes:
    """Replace the two loaders ``module`` imported with ones returning the fakes."""
    fakes = Fakes()

    def vae_encoder(_path: str, _device: torch.device) -> FakeVAEEncoder:
        fakes.vae_loads += 1
        return fakes.encoder

    def text_encoder(path: str, _device: torch.device, _dtype: torch.dtype, tokenizer: str | None) -> FakeTextEncoder:
        fakes.text_loads.append((path, tokenizer))
        return fakes.text

    monkeypatch.setattr(module, "load_flux2_vae_encoder", vae_encoder)
    monkeypatch.setattr(module, "load_flux3_text_encoder", text_encoder)
    return fakes


def skip_housekeeping(monkeypatch: pytest.MonkeyPatch, commit: str = "abc1234") -> None:
    """The trainer's two slow environment calls, a full ``gc.collect()`` and ``git`` (twice), answered at once."""
    monkeypatch.setattr(session, "gc", SimpleNamespace(collect=lambda: 0))
    monkeypatch.setattr(session, "git_commit", lambda: commit)
