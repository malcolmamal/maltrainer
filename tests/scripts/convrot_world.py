"""A stand-in for comfy-kitchen's INT8 quantiser and small source / reference files for convert_int8_convrot."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

import convert_int8_convrot

QUANT = "comfy_kitchen.backends.eager.quantization"


def blob(value: dict[str, Any] | str) -> torch.Tensor:
    text = value if isinstance(value, str) else json.dumps(value)
    return torch.tensor(list(text.encode("utf-8")), dtype=torch.uint8)


CONVROT = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 4}


def install_quantizer(monkeypatch: pytest.MonkeyPatch) -> list[tuple[torch.Tensor, int]]:
    """comfy-kitchen's module with a row-wise int8 quantiser that records its calls."""
    seen: list[tuple[torch.Tensor, int]] = []

    def quantize(weight: torch.Tensor, group: int) -> tuple[torch.Tensor, torch.Tensor]:
        seen.append((weight, group))
        scale = weight.abs().amax(dim=1) / 127
        return (weight / scale[:, None]).round().to(torch.int8), scale

    for name in ("comfy_kitchen", "comfy_kitchen.backends", "comfy_kitchen.backends.eager", QUANT):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    vars(sys.modules[QUANT])["quantize_int8_convrot_weight"] = quantize
    return seen


def make_files(tmp_path: Path, config: dict[str, Any] | None = None) -> tuple[Path, Path, Path]:
    """A source with one Linear (to quantize), a norm and a bias; a reference quantizing the Linear."""
    gen = torch.Generator().manual_seed(2)
    source = tmp_path / "src.safetensors"
    save_file(
        {
            "blocks.0.lin.weight": torch.randn(3, 8, generator=gen).to(torch.bfloat16),
            "blocks.0.lin.bias": torch.randn(3, generator=gen),
            "blocks.0.norm.weight": torch.randn(8, generator=gen),
        },
        str(source),
        metadata={"origin": "test"},
    )
    reference = tmp_path / "ref.safetensors"
    save_file(
        {
            "blocks.0.lin.weight": torch.zeros(3, 8, dtype=torch.int8),
            "blocks.0.lin.weight_scale": torch.ones(3, 1),
            "blocks.0.lin.comfy_quant": blob(config or CONVROT),
            "blocks.0.comfy_attention.config": blob({"attention": "int8"}),
            "blocks.0.other_scale": torch.ones(1),
        },
        str(reference),
    )
    return source, reference, tmp_path / "out.safetensors"


def run(monkeypatch: pytest.MonkeyPatch, source: Path, output: Path, reference: Path, *flags: str) -> int:
    argv = ["convert", str(source), str(output), "--reference", str(reference), "--device", "cpu", *flags]
    monkeypatch.setattr("sys.argv", argv)
    return convert_int8_convrot.main()
