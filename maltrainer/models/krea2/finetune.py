"""
What a Krea 2 full finetune trains, and how its checkpoint is saved
(train_krea2_finetune.py; the rotation itself is maltrainer/finetune/rotation.py).

Krea 2 is 12.9B parameters: 28 single-stream blocks of ~434M each (attention 132M, MLP 302M) and
a ~345M text-fusion stack. Its bf16 weights alone are 25.8 GB, so a 32 GB card trains them a
window at a time, as Fizgig does:

    attention   blocks.N.attn.{wq,wk,wv,gate,wo}   3.7B
    mlp.gate    blocks.N.mlp.gate                  2.8B
    mlp.up      blocks.N.mlp.up                    2.8B
    mlp.down    blocks.N.mlp.down                  2.8B

Each window spans all 28 blocks, so a concept is learned at every depth at once. The text-fusion
stack trains all run: it is where prompt-to-concept binding happens, and a rotation over the
blocks would never reach it (Fizgig: FTSpec.always_on). Norms, modulation, biases and the
input/output layers stay frozen.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from safetensors.torch import save_file
from torch import nn

from maltrainer.finetune.rotation import Window, WindowRotator

logger = logging.getLogger(__name__)

DEFAULT_WINDOWS: Tuple[Window, ...] = (
    ("attention", ("attn.",)),
    ("mlp.gate", ("mlp.gate",)),
    ("mlp.up", ("mlp.up",)),
    ("mlp.down", ("mlp.down",)),
)
DEFAULT_ALWAYS_ON = ("txtfusion",)


def parse_windows(config: Optional[Sequence]) -> Tuple[Window, ...]:
    """finetune.windows: [{name: attention, prefixes: ["attn."]}, ...] or [[name, [prefixes]], ...]."""
    if not config:
        return DEFAULT_WINDOWS
    out = []
    for entry in config:
        if isinstance(entry, dict):
            name, prefixes = entry["name"], entry["prefixes"]
        else:
            name, prefixes = entry
        out.append((str(name), tuple(str(p) for p in ([prefixes] if isinstance(prefixes, str) else prefixes))))
    return tuple(out)


def always_on_linears(model: nn.Module, modules: Sequence[str]) -> List[Tuple[str, nn.Parameter]]:
    """(name, weight) of every Linear under ``modules``, made trainable (weights only)."""
    out = []
    for module_name in modules:
        root = model.get_submodule(module_name)
        for name, module in root.named_modules():
            if isinstance(module, nn.Linear):
                if module.weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    raise ValueError(f"{module_name}.{name} is fp8; always-on modules must be bf16")
                module.weight.requires_grad_(True)
                out.append((f"{module_name}.{name}.weight", module.weight))
    return out


def save_krea2_model(model: nn.Module, rotator: WindowRotator, path: Path, key_prefix: str = "",
                     metadata: Optional[Dict[str, str]] = None) -> None:
    """A full bf16 Krea 2 file with the source's key prefix, which ComfyUI and load_krea2_transformer
    read back: the rotating weights from the master (the active window live), everything else
    (text fusion, norms, modulation, I/O) as the model holds it. Written aside, then renamed."""
    rotating = set(rotator.master)  # every block Linear, in a window or not
    out: Dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        module, _, leaf = name.rpartition(".")
        if f"{module}.weight" in rotating and leaf in ("weight", "weight_scale"):
            continue  # an fp8 copy or the active window: written from the master below
        out[key_prefix + name] = tensor.detach().to("cpu").contiguous()
    for key in rotator.master:
        out[key_prefix + key] = rotator.weight(key).contiguous()
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    save_file(out, str(tmp), metadata=metadata)
    os.replace(tmp, path)
