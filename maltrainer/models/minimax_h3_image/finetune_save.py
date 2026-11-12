"""Saving a finetuned H3 Image model as an INT8 ConvRot file, and the weights a resume needs.

The file is written from the base file's own tensor list, so ComfyUI reads it as it reads the original: the trained
Linears as INT8 codes with their row scale, an always-trained weight in its stored dtype, and every other tensor (the
quantisation markers, the attention-kernel hints, the norms, the AdaLN table) copied unchanged. The base file's
metadata (the noise-level grid among it) is kept, with the run's added.

A resume needs more than the INT8 file: the bf16 value of every weight a window has trained (quantizing it to INT8
loses what the next steps would have added), which ``write_resume_weights`` keeps.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

from maltrainer.finetune.rotation_int8 import Int8WindowRotator

ALWAYS_PREFIX = "always."


def write_int8_model(
    source: Path,
    rotator: Int8WindowRotator,
    always_on: Sequence[tuple[str, nn.Parameter]],
    path: Path,
    metadata: dict[str, str] | None = None,
) -> None:
    """The base file ``source`` with the finetuned weights in it, written aside and then renamed to ``path``."""
    rotating = set(rotator.keys())
    trained = dict(always_on)
    out: dict[str, torch.Tensor] = {}
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        stored = dict(handle.metadata() or {})
        for key in handle.keys():
            base, _, leaf = key.rpartition(".")
            if leaf in ("weight", "weight_scale") and f"{base}.weight" in rotating:
                codes, scale = rotator.int8(f"{base}.weight")
                out[key] = (
                    codes if leaf == "weight" else scale.reshape(handle.get_slice(key).get_shape())
                ).contiguous()
            elif key in trained:
                original = handle.get_tensor(key)
                out[key] = trained[key].detach().to("cpu", original.dtype).contiguous()
            else:
                out[key] = handle.get_tensor(key)
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    save_file(out, str(tmp), metadata={**(metadata or {}), **stored})
    tmp.replace(path)


def write_resume_weights(rotator: Int8WindowRotator, always_on: Sequence[tuple[str, nn.Parameter]], path: Path) -> None:
    """The bf16 value of every trained block weight and every always-trained weight, for ``read_resume_weights``."""
    tensors = {key: value.contiguous() for key, value in rotator.touched().items()}
    for name, param in always_on:
        tensors[ALWAYS_PREFIX + name] = param.detach().to("cpu").contiguous()
    save_file(tensors, str(path))


def read_resume_weights(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """(the trained block weights, the always-trained weights by name) of a ``write_resume_weights`` file."""
    tensors = load_file(str(path))
    always = {key[len(ALWAYS_PREFIX) :]: value for key, value in tensors.items() if key.startswith(ALWAYS_PREFIX)}
    blocks = {key: value for key, value in tensors.items() if not key.startswith(ALWAYS_PREFIX)}
    return blocks, always
