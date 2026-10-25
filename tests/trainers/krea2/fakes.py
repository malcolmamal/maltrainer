"""Stand-ins for the heavy components the Krea 2 trainers load: the VAE encoder and decoder and the text encoder.
Everything else in a trainer run (the DiT, the LoRA network or window rotation, the loop, the sampler) is the real
code."""

from __future__ import annotations

import zlib
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional

from maltrainer.training import session
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


class FakeVAEDecoder:
    """Latents [B, 4, h, w] -> pixels [B, 3, 8h, 8w] in [0, 1]: the channel mean through a sigmoid, in gray."""

    def __init__(self) -> None:
        self.decoded: list[tuple[int, ...]] = []

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        self.decoded.append(tuple(latents.shape))
        gray = latents.float().mean(dim=1, keepdim=True).sigmoid().repeat(1, 3, 1, 1)
        return gray.repeat_interleave(8, dim=2).repeat_interleave(8, dim=3)


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
    decoder: FakeVAEDecoder = field(default_factory=FakeVAEDecoder)
    text: FakeTextEncoder = field(default_factory=FakeTextEncoder)
    vae_loads: int = 0
    decoder_loads: list[torch.dtype] = field(default_factory=list)
    text_loads: list[tuple[str, str | None]] = field(default_factory=list)
    cpu_text_loads: list[torch.device] = field(default_factory=list)


def install_fakes(monkeypatch: pytest.MonkeyPatch, module: object, previews: object) -> Fakes:
    """Replace the loaders ``module`` (the LoRA trainer's) and ``previews`` (the finetune's previews) imported with
    ones returning the fakes."""
    fakes = Fakes()

    def vae(_path: str, _device: torch.device) -> FakeVAEEncoder:
        fakes.vae_loads += 1
        return fakes.encoder

    def text_encoder(path: str, _device: torch.device, _dtype: torch.dtype, tokenizer: str | None) -> FakeTextEncoder:
        fakes.text_loads.append((path, tokenizer))
        return fakes.text

    def decoder(_path: str, _device: torch.device, dtype: torch.dtype) -> FakeVAEDecoder:
        fakes.decoder_loads.append(dtype)
        return fakes.decoder

    def cpu_text_encoder(
        _path: str, device: torch.device, _dtype: torch.dtype, _tokenizer: str | None
    ) -> FakeTextEncoder:
        fakes.cpu_text_loads.append(device)
        return fakes.text

    monkeypatch.setattr(module, "load_krea2_vae", vae)
    monkeypatch.setattr(module, "load_krea2_text_encoder", text_encoder)
    monkeypatch.setattr(previews, "load_krea2_vae_decoder", decoder)
    monkeypatch.setattr(previews, "load_krea2_text_encoder", cpu_text_encoder)  # prompts added mid-run
    return fakes


def skip_housekeeping(monkeypatch: pytest.MonkeyPatch, commit: str = "abc1234") -> None:
    """The trainers' two slow environment calls, a full ``gc.collect()`` and ``git`` (twice), answered at once."""
    monkeypatch.setattr(session, "gc", SimpleNamespace(collect=lambda: 0))
    monkeypatch.setattr(session, "git_commit", lambda: commit)


class ScriptedKeys(KeyCommands):
    """The real key handling, fed from a script: the n-th poll returns the n-th list of keys."""

    def __init__(
        self, script: list[list[str]], menu: Callable[[], None] | None, gallery: Callable[[], None] | None
    ) -> None:
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
