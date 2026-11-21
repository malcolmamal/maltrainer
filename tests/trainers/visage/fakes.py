"""Stand-ins for the two heavy components the Visage-Image trainer loads: the VAE and the text encoder.
Everything else in a trainer run (the DiT, the LoRA network, the optimizer, the loop) is the real code."""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional

from maltrainer.training import session
from tests.maltrainer.models.visage.tiny_model import CAP, LATENT


class FakeVAE:
    """Pixels [B, 3, H, W] in [0, 1] -> latents [B, 8, H/16, W/16]: a fixed linear map of the 16 px cell means."""

    def __init__(self) -> None:
        self.projection = torch.randn(LATENT, 3, generator=torch.Generator().manual_seed(11))
        self.shapes: list[tuple[int, ...]] = []

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        self.shapes.append(tuple(pixels.shape))
        cells = functional.avg_pool2d(pixels.float(), 16) - 0.5
        return torch.einsum("oc,bchw->bohw", self.projection, cells)


class FakeTextEncoder:
    """A caption -> [L, 16] states drawn from a generator seeded by the caption, L in 2..4."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def encode(self, caption: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        self.prompts.append(caption)
        seed = zlib.crc32(caption.encode("utf-8"))
        states = torch.randn(2 + seed % 3, CAP, generator=torch.Generator().manual_seed(seed))
        return states.to(device, dtype)


@dataclass
class Fakes:
    vae: FakeVAE = field(default_factory=FakeVAE)
    text: FakeTextEncoder = field(default_factory=FakeTextEncoder)
    vae_loads: int = 0
    text_loads: list[tuple[str, str | None]] = field(default_factory=list)


def install_fakes(monkeypatch: pytest.MonkeyPatch, module: object) -> Fakes:
    """Replace the two loaders ``module`` imported with ones returning the fakes."""
    fakes = Fakes()

    def vae(_path: str, _device: torch.device) -> FakeVAE:
        fakes.vae_loads += 1
        return fakes.vae

    def text_encoder(path: str, _device: torch.device, _dtype: torch.dtype, tokenizer: str | None) -> FakeTextEncoder:
        fakes.text_loads.append((path, tokenizer))
        return fakes.text

    monkeypatch.setattr(module, "load_visage_vae", vae)
    monkeypatch.setattr(module, "load_visage_text_encoder", text_encoder)
    return fakes


def skip_housekeeping(monkeypatch: pytest.MonkeyPatch, commit: str = "abc1234") -> None:
    """The trainer's two slow environment calls, a full ``gc.collect()`` and ``git`` (twice), answered at once."""
    monkeypatch.setattr(session, "gc", SimpleNamespace(collect=lambda: 0))
    monkeypatch.setattr(session, "git_commit", lambda: commit)
