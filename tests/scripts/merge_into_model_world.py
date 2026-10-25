"""Small model files and a runner for the scripts/merge_lora_into_model.py tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

import merge_lora_into_model

P = "model.diffusion_model."


def model_file(path: Path, seed: int, scale: float = 1.0, prefix: str = P, **extra: torch.Tensor) -> Path:
    gen = torch.Generator().manual_seed(seed)
    tensors = {
        f"{prefix}blocks.0.lin.weight": torch.randn(4, 6, generator=gen) * scale,
        f"{prefix}blocks.0.norm.scale": torch.randn(6, generator=gen) * scale,
        f"{prefix}step": torch.tensor([seed]),
        **extra,
    }
    save_file(tensors, str(path), metadata={"origin": path.stem})
    return path


def near(path: Path, of: Path, amount: float, seed: int) -> Path:
    """A model 'trained from' ``of``: its tensors moved by about ``amount`` of their size."""
    gen = torch.Generator().manual_seed(seed)
    tensors = {
        k: v + amount * v.abs().mean() * torch.randn(v.shape, generator=gen) if v.is_floating_point() else v
        for k, v in load_file(str(of)).items()
    }
    save_file(tensors, str(path))
    return path


def run(monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    monkeypatch.setattr("sys.argv", ["merge", *args, "--device", "cpu"])
    return merge_lora_into_model.main()
