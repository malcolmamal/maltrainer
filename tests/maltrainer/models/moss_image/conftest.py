"""Fixtures that write the tiny model's weights as the checkpoint files the loaders read."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import pytest
import torch
from safetensors.torch import save_file

from maltrainer.models.moss_image.model import MOSSImageTransformer

from .tiny import int8_tensors, tiny_model


@pytest.fixture
def make_model() -> Callable[..., MOSSImageTransformer]:
    return tiny_model


@pytest.fixture
def write_checkpoint(tmp_path: Path) -> Callable[..., Path]:
    """``write_checkpoint(model, name, prefix="", extra={}, dtype)``: the weights, behind ``prefix``."""

    def write(
        model: MOSSImageTransformer,
        name: str = "moss_image_base_bf16.safetensors",
        prefix: str = "",
        extra: Optional[Dict[str, torch.Tensor]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> Path:
        tensors = {k: v.detach().to(dtype).contiguous() for k, v in model.state_dict().items()}
        tensors.update(extra or {})
        path = tmp_path / name
        save_file({prefix + key: value for key, value in tensors.items()}, str(path))
        return path

    return write


@pytest.fixture
def write_int8_checkpoint(tmp_path: Path) -> Callable[..., Path]:
    def write(model: MOSSImageTransformer, name: str = "moss_image_base_int8_convrot.safetensors") -> Path:
        path = tmp_path / name
        save_file({k: v.detach().contiguous() for k, v in int8_tensors(model).items()}, str(path))
        return path

    return write
