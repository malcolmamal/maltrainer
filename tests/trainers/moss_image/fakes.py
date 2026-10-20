"""Stand-ins for the three heavy components the MOSS-Image trainers load: the VAE (encoder and
decoder) and the 9B text encoder. Everything else in a trainer run is the real code."""

from __future__ import annotations

import io
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch.nn import functional

from maltrainer.trainers.moss_image import cache, previews
from maltrainer.training import session
from maltrainer.utils.keyboard import KeyCommands

LATENT_CHANNELS = 128
TEXT_DIM = 8
CPU = torch.device("cpu")


class FakeVAEEncoder:
    """Pixels [B, 3, H, W] in [0, 1] -> latents [B, 128, H/16, W/16]: a fixed linear map of the 16 px cell means."""

    def __init__(self) -> None:
        self.projection = torch.randn(LATENT_CHANNELS, 3, generator=torch.Generator().manual_seed(11))
        self.encoded = 0

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        self.encoded += pixels.shape[0]
        cells = functional.avg_pool2d(pixels.float(), 16) - 0.5
        return torch.einsum("oc,bchw->bohw", self.projection, cells)


class FakeVAEDecoder:
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        gray = latents.float().mean(dim=1, keepdim=True).sigmoid().repeat(1, 3, 1, 1)
        return gray.repeat_interleave(16, dim=2).repeat_interleave(16, dim=3)


class FakeTextEncoder:
    """A caption -> [L, 8] embedding drawn from a generator seeded by the caption, L in 2..4."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def encode(self, caption: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        self.prompts.append(caption)
        seed = zlib.crc32(caption.encode("utf-8"))
        return torch.randn(2 + seed % 3, TEXT_DIM, generator=torch.Generator().manual_seed(seed)).to(device, dtype)


@dataclass
class Fakes:
    encoder: FakeVAEEncoder = field(default_factory=FakeVAEEncoder)
    decoder: FakeVAEDecoder = field(default_factory=FakeVAEDecoder)
    text: FakeTextEncoder = field(default_factory=FakeTextEncoder)
    vae_loads: int = 0
    decoder_loads: int = 0
    text_loads: list[torch.device] = field(default_factory=list)


def install_fakes(monkeypatch: pytest.MonkeyPatch) -> Fakes:
    """Replace the loaders the trainers' caching and preview parts import with ones returning the fakes."""
    fakes = Fakes()

    def vae_encoder(_path: str, _device: torch.device) -> FakeVAEEncoder:
        fakes.vae_loads += 1
        return fakes.encoder

    def vae_decoder(_path: str, _device: torch.device) -> FakeVAEDecoder:
        fakes.decoder_loads += 1
        return fakes.decoder

    def text_encoder(_path: str, device: torch.device, _dtype: torch.dtype, _tokenizer: str | None) -> FakeTextEncoder:
        fakes.text_loads.append(device)
        return fakes.text

    monkeypatch.setattr(cache, "load_flux2_vae_encoder", vae_encoder)
    monkeypatch.setattr(previews, "load_flux2_vae_decoder", vae_decoder)
    monkeypatch.setattr(cache, "load_moss_image_text_encoder", text_encoder)  # the caption cache
    monkeypatch.setattr(previews, "load_moss_image_text_encoder", text_encoder)  # prompts added mid-run
    return fakes


def skip_housekeeping(monkeypatch: pytest.MonkeyPatch, *modules: object, commit: str = "abc1234") -> None:
    """The trainers' two slow environment calls, a full ``gc.collect()`` and ``git`` (twice), answered at once.

    Only the trainers' own references are replaced, so the rest of the process keeps the real ``gc``.
    """
    for module in (session, *modules):
        monkeypatch.setattr(module, "gc", SimpleNamespace(collect=lambda: 0))
    monkeypatch.setattr(session, "git_commit", lambda: commit)


def capture_menu(monkeypatch: pytest.MonkeyPatch, module: Any) -> io.StringIO:
    """Draw the preview menu into a string instead of the stdout that was bound when the menu module loaded."""
    screen = io.StringIO()
    real = module.run_menu

    def run_menu(*args: Any, **kwargs: Any) -> list[str]:
        return real(*args, out=screen, **kwargs)

    monkeypatch.setattr(module, "run_menu", run_menu)
    return screen


class ScriptedKeys(KeyCommands):
    """The real key handling, fed from a script: the n-th poll returns the n-th list of keys."""

    def __init__(self, script: list[list[str]], menu: Callable[[], None] | None, gallery: Callable[[], None] | None):
        super().__init__(menu=menu, gallery=gallery)
        self.script = list(script)

    def poll(self) -> list[str]:
        return self.script.pop(0) if self.script else []


def script_keys(monkeypatch: pytest.MonkeyPatch, modules: list[object], script: list[list[str]]) -> list[ScriptedKeys]:
    """Make the trainers' ``KeyCommands`` a scripted one; the list collects the instances made."""
    made: list[ScriptedKeys] = []

    def factory(menu: Callable[[], None] | None = None, gallery: Callable[[], None] | None = None) -> ScriptedKeys:
        keys = ScriptedKeys(script, menu, gallery)
        made.append(keys)
        return keys

    for module in modules:
        monkeypatch.setattr(module, "KeyCommands", factory)
    return made
