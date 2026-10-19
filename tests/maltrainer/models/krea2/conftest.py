"""Fixtures writing the tiny Krea 2 DiT as the ComfyUI checkpoint files the loaders read."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Optional

import pytest
import torch
from safetensors.torch import save_file

from maltrainer.models.comfy_fp8 import quantize_rows
from maltrainer.models.krea2 import loading
from maltrainer.models.krea2.loading import is_block_linear

from .tiny import CONFIG, tiny_model


@pytest.fixture(autouse=True)
def tiny_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """The loader builds the full 12.9B config by default; build the tiny one instead."""
    monkeypatch.setattr(loading, "Krea2Config", lambda: CONFIG)


@pytest.fixture
def write_checkpoint(tmp_path: Path) -> Callable[..., Path]:
    """``write_checkpoint(prefix="", fp8=False, drop=(), extra={})``: the tiny model as a ComfyUI file."""

    def write(
        prefix: str = "",
        fp8: bool = False,
        drop=(),
        extra: Optional[Dict[str, torch.Tensor]] = None,
        name: str = "krea2.safetensors",
    ) -> Path:
        tensors: Dict[str, torch.Tensor] = {}
        layers: Dict[str, dict] = {}
        for key, value in tiny_model().state_dict().items():
            if key in drop:
                continue
            if fp8 and is_block_linear(key):
                tensors[prefix + key], scale = quantize_rows(value)
                tensors[prefix + key[: -len(".weight")] + ".weight_scale"] = scale
                layers[prefix + key[: -len(".weight")]] = {"format": "float8_e4m3fn"}
            else:
                tensors[prefix + key] = value.to(torch.bfloat16).contiguous()
        tensors.update(extra or {})
        metadata = {"_quantization_metadata": json.dumps({"layers": layers})} if layers else None
        path = tmp_path / name
        save_file(tensors, str(path), metadata=metadata)
        return path

    return write
