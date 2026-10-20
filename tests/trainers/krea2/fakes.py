"""Stand-ins for the two heavy components the Krea 2 trainers load: the VAE encoder and the text encoder.
Everything else in a trainer run (the DiT, the LoRA network or window rotation, the loop) is the real code."""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional

from maltrainer.utils.keyboard import KeyCommands

from .tiny import CONFIG


class FakeVAEEncoder:
    """Pixels [B, 3, H, W] in [0, 1] -> latents [B, 4, H/8, W/8]: a fixed linear map of the 8 px cell means."""

    def __init__(self) -> None:
        self.projection = torch.randn(CONFIG.channels, 3, generator=torch.Generator().manual_seed(11))
        self.shapes: list[tuple[int, ...]] = []

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        self.shapes.append(tuple(pixels.shape))
        cells = functional.avg_pool2d(pixels.float(), 8) - 0.5
        return torch.einsum("oc,bchw->bohw", self.projection, cells)


class FakeTextEncoder:
    """A caption -> [L, 3, 32] embedding drawn from a generator seeded by the caption, L in 2..4."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def encode(self, caption: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        self.prompts.append(caption)
        seed = zlib.crc32(caption.encode("utf-8"))
        shape = (2 + seed % 3, CONFIG.txtlayers, CONFIG.txtdim)
        return torch.randn(shape, generator=torch.Generator().manual_seed(seed)).to(device, dtype)


@dataclass
class Fakes:
    encoder: FakeVAEEncoder = field(default_factory=FakeVAEEncoder)
    text: FakeTextEncoder = field(default_factory=FakeTextEncoder)
    vae_loads: int = 0
    text_loads: list[tuple[str, str | None]] = field(default_factory=list)


def install_fakes(monkeypatch: pytest.MonkeyPatch, module: object) -> Fakes:
    """Replace the two loaders ``module`` (the LoRA trainer's) imported with ones returning the fakes."""
    fakes = Fakes()

    def vae(_path: str, _device: torch.device) -> FakeVAEEncoder:
        fakes.vae_loads += 1
        return fakes.encoder

    def text_encoder(path: str, _device: torch.device, _dtype: torch.dtype, tokenizer: str | None) -> FakeTextEncoder:
        fakes.text_loads.append((path, tokenizer))
        return fakes.text

    monkeypatch.setattr(module, "load_krea2_vae", vae)
    monkeypatch.setattr(module, "load_krea2_text_encoder", text_encoder)
    return fakes


def skip_housekeeping(monkeypatch: pytest.MonkeyPatch, module: object, commit: str = "abc1234") -> None:
    """The trainers' two slow environment calls, a full ``gc.collect()`` and ``git`` (twice), answered at once."""
    monkeypatch.setattr(module, "gc", SimpleNamespace(collect=lambda: 0))
    monkeypatch.setattr(module, "git_commit", lambda: commit)


class ScriptedKeys(KeyCommands):
    """The real key handling, fed from a script: the n-th poll returns the n-th list of keys."""

    def __init__(self, script: list[list[str]]) -> None:
        super().__init__()
        self.script = list(script)

    def poll(self) -> list[str]:
        return self.script.pop(0) if self.script else []


def script_keys(monkeypatch: pytest.MonkeyPatch, modules: list[object], script: list[list[str]]) -> list[ScriptedKeys]:
    """Make the trainers' ``KeyCommands`` a scripted one; the list collects the instances made."""
    made: list[ScriptedKeys] = []

    def factory() -> ScriptedKeys:
        keys = ScriptedKeys(script)
        made.append(keys)
        return keys

    for module in modules:
        monkeypatch.setattr(module, "KeyCommands", factory)
    return made
