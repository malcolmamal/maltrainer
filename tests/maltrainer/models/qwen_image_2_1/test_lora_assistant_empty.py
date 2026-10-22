"""An assistant adapter that attaches nothing is reported, not a crash."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from maltrainer.models.qwen_image_2_1.lora import QwenImage21LoRANetwork

from ._tiny import tiny_transformer


def _network() -> QwenImage21LoRANetwork:
    return QwenImage21LoRANetwork(tiny_transformer(), rank=2, alpha=2.0, target_modules=["attn.to_q"])


def _file(tmp_path: Path, tensors: dict[str, torch.Tensor]) -> Path:
    path = tmp_path / "assistant.safetensors"
    save_file(tensors, str(path))
    return path


def test_an_adapter_for_modules_the_transformer_lacks_attaches_none_and_names_what_it_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    tensors = {
        "diffusion_model.nope.lora_A.weight": torch.zeros(2, 16),
        "diffusion_model.nope.lora_B.weight": torch.zeros(16, 2),
    }

    with caplog.at_level(logging.INFO):
        attached = _network().load_assistant_adapter(_file(tmp_path, tensors))

    assert attached == 0
    assert "skipped 1 module(s) not found in transformer: ['nope']" in caplog.text
    assert "Attached frozen assistant adapter: 0 modules" in caplog.text


def test_a_file_with_no_lora_tensors_attaches_none(tmp_path: Path) -> None:
    network = _network()

    attached = network.load_assistant_adapter(_file(tmp_path, {"stray": torch.zeros(1)}))

    assert attached == 0
    assert len(network.assistant_layers) == 0
