"""Stand-ins for what the Qwen-Image-2.1 sampling scripts load (text encoder, transformer, VAE) and for CUDA.

The scripts build ``torch.device("cuda")``; ``CpuTorch`` is a ``torch`` whose "cuda" device is the CPU, put in the
script module's namespace so nothing else in the process is affected.
"""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from safetensors.torch import save_file
from torch import nn

CHANNELS = 64


class CpuTorch:
    """``torch`` with ``torch.device(...)`` always answering the CPU."""

    def device(self, _name: str) -> torch.device:
        return torch.device("cpu")

    def __getattr__(self, name: str) -> Any:
        return getattr(torch, name)


class FakeTextEncoder:
    def encode_prompts(self, prompts: list[str], device: torch.device) -> torch.Tensor:
        seed = zlib.crc32(prompts[0].encode("utf-8"))
        return torch.randn(1, 3, 4, generator=torch.Generator().manual_seed(seed)).to(device)


class FakeTransformer(nn.Module):
    """Velocity = 0.1 * a linear map of the latents; ``to_q`` is where a LoRA attaches."""

    def __init__(self) -> None:
        super().__init__()
        self.to_q = nn.Linear(CHANNELS, CHANNELS, bias=False)
        with torch.no_grad():
            self.to_q.weight.copy_(torch.randn(CHANNELS, CHANNELS, generator=torch.Generator().manual_seed(5)) * 0.1)

    def forward(self, hidden_states: torch.Tensor, **_inputs: Any) -> torch.Tensor:
        return self.to_q(hidden_states)


class FakeVAE:
    def encode(self, pixels: torch.Tensor, **_options: Any) -> torch.Tensor:
        cells = nn.functional.avg_pool2d(pixels.float(), 16) - 0.5
        projection = torch.randn(CHANNELS, 3, generator=torch.Generator().manual_seed(6))
        return torch.einsum("oc,bchw->bohw", projection, cells)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        gray = latents.float().mean(dim=1, keepdim=True).sigmoid().repeat(1, 3, 1, 1)
        return gray.repeat_interleave(16, dim=2).repeat_interleave(16, dim=3)


def install(monkeypatch: pytest.MonkeyPatch, module: Any) -> None:
    """Replace the script's CUDA and its three model loaders with CPU fakes."""

    def no_encoder_args(_path: str, **_options: Any) -> FakeTextEncoder:
        return FakeTextEncoder()

    monkeypatch.setattr(module, "torch", CpuTorch())
    monkeypatch.setattr(module, "load_qwen_image_2_1_text_encoder", no_encoder_args)
    monkeypatch.setattr(module, "load_qwen_image_2_1_transformer", lambda _path, **_options: FakeTransformer())
    monkeypatch.setattr(module, "load_qwen_image_2_1_vae", lambda _path, **_options: FakeVAE())


def write_checkpoint(path: Path, seed: int, rank: int = 2) -> Path:
    gen = torch.Generator().manual_seed(seed)
    save_file(
        {
            "diffusion_model.to_q.lora_A.weight": torch.randn(rank, CHANNELS, generator=gen),
            "diffusion_model.to_q.lora_B.weight": torch.randn(CHANNELS, rank, generator=gen) * 0.5,
            "diffusion_model.to_q.alpha": torch.tensor(float(rank)),
        },
        str(path),
    )
    return path


def write_config(path: Path, out: Path, dataset: Path | None = None, **model: str) -> Path:
    """A training config naming dummy model files, the run's output folder and (optionally) a dataset."""
    paths = {"transformer_path": "t.safetensors", "vae_path": "v.safetensors", "text_encoder_path": "e.safetensors"}
    config: dict[str, Any] = {"model": {**paths, **model}, "dtype": "float32", "output": {"path": str(out)}}
    config["dataset"] = {"path": str(dataset), "resolution": 64, "trigger": "zed"} if dataset else {"path": "none"}
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path
