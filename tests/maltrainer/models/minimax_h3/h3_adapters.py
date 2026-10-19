from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

PEFT = ("lora_A", "lora_B")
KOHYA = ("lora_down", "lora_up")


def write_adapter(
    path: Path,
    modules: dict[str, tuple[torch.Tensor, torch.Tensor, float | None]],
    *,
    names: tuple[str, str] = PEFT,
    prefix: str = "diffusion_model.",
    kohya_names: bool = False,
) -> Path:
    """Write ``{module path: (down, up, alpha)}`` as a LoRA safetensors file in PEFT or Kohya naming."""
    tensors: dict[str, torch.Tensor] = {}
    for module_path, (down, up, alpha) in modules.items():
        stem = "lora_unet_" + module_path.replace(".", "_") if kohya_names else f"{prefix}{module_path}"
        tensors[f"{stem}.{names[0]}.weight"] = down.clone()
        tensors[f"{stem}.{names[1]}.weight"] = up.clone()
        if alpha is not None:
            tensors[f"{stem}.alpha"] = torch.tensor(alpha)
    save_file(tensors, str(path))
    return path


def hand_pair(in_features: int, out_features: int, rank: int = 2, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rank, in_features, generator=generator), torch.randn(out_features, rank, generator=generator)
