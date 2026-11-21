"""A tiny H3 Image finetune world: a real INT8 ConvRot file of the tiny model, a config, and runs over them."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from maltrainer.models.minimax_h3_image.finetune_int8 import encode
from maltrainer.models.minimax_h3_image.grid import PDD_SIGMAS
from tests.trainers.minimax_h3.world import Workspace
from tests.trainers.minimax_h3_image.conftest import image_state
from train_minimax_h3_image_finetune import MiniMaxH3ImageFinetuneTrainer

GROUP = 4
BLOCK_LINEAR = re.compile(r"blocks\.\d+\.(attn\.(qkv|out)_proj|mlp\.fc[12])")
MARKER = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GROUP})
NAME = "ft"


def write_int8_checkpoint(path: Path) -> Path:
    """The tiny image model as Comfy-Org stores H3 Image: block Linears as INT8 codes with a scale and a marker, a
    kernel hint per attention, everything else dense, and the noise-level grid in the metadata."""
    tensors: dict[str, torch.Tensor] = {}
    for key, value in image_state().items():
        base = key.removesuffix(".weight")
        if key.endswith(".weight") and BLOCK_LINEAR.fullmatch(base):
            tensors[key], tensors[f"{base}.weight_scale"] = encode(value, GROUP)
            tensors[f"{base}.comfy_quant"] = torch.tensor(list(MARKER.encode()), dtype=torch.uint8)
        else:
            tensors[key] = value
    tensors["blocks.0.attn.comfy_attention.config"] = torch.tensor(list(b'{"attention": "int8"}'), dtype=torch.uint8)
    config = json.dumps({"transformer": {"pdd_sigmas": list(PDD_SIGMAS)}})
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path), metadata={"config": config})
    return path


def finetune_config(ws: Workspace, **sections: Any) -> dict[str, Any]:
    """Windows of one block rotating every step, Automagic3 at a rate that moves the tiny weights."""
    base = {
        "train": {
            "optimizer": "automagic3",
            "learning_rate": 1e-3,
            "automagic_min_lr": 1e-3,
            "automagic_max_lr": 1e-3,
            "num_epochs": 1,
            "save_every": 0,
        },
        "finetune": {"blocks_per_window": 1, "rotate_every": 1, "training_adapter": False},
    }
    return ws.config(base, **sections)


def run_finetune(config: dict[str, Any]) -> MiniMaxH3ImageFinetuneTrainer:
    trainer = MiniMaxH3ImageFinetuneTrainer(config)
    trainer.run()
    return trainer
