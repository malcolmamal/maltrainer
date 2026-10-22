"""Real-device rotary tables for the meta-built Qwen3-VL text encoder, across transformers versions."""

from __future__ import annotations

import inspect
from typing import Any

import torch
from torch import nn


def rebuild_vision_rotary(old: nn.Module, vision_config: Any, device: torch.device) -> nn.Module:
    """A real-device copy of the vision rotary table (the meta-built one holds no values).

    transformers 4.57/5.0 build it from the rotary dimension, 5.20 from the vision config.
    """
    rotary_class = type(old)
    if "config" in inspect.signature(rotary_class).parameters:
        return rotary_class(vision_config).to(device)
    return rotary_class(vision_config.hidden_size // vision_config.num_heads // 2).to(device)


def rebuild_text_rotary(old: nn.Module, text_config: Any, device: torch.device) -> nn.Module:
    """A real-device copy of the text rotary table; transformers 5.20 dropped the ``device`` argument."""
    rotary_class = type(old)
    if "device" in inspect.signature(rotary_class).parameters:
        return rotary_class(text_config, device=device)
    return rotary_class(text_config).to(device)
