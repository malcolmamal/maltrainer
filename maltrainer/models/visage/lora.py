"""
LoRA for the Visage-Image DiT.

Saved keys follow the other trainers: ``diffusion_model.<module path>.lora_A.weight`` / ``lora_B.weight`` /
``alpha``, on the DiT's own Linears (``attn.qkv``, ``attn.out_proj``, ``mlp.fc1``, ``mlp.fc2``).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any, cast

import torch
from torch import nn

from maltrainer.models.flux3.lora import LoRALayer

logger = logging.getLogger(__name__)

ATTN = ("attn.qkv", "attn.out_proj")
MLP = ("mlp.fc1", "mlp.fc2")
IO_LAYERS = ("text_proj", "time_embed.1", "time_embed.3", "out_proj", "final_mod.1")

# Named target sets.
#   blocks  attention and SwiGLU Linears of the 18 blocks (72)
#   attn    the blocks' attention only (36)
#   all     blocks, every block's adaLN modulation, and the text, timestep and output Linears
TARGET_PRESETS = ("blocks", "attn", "all")


def selected(name: str, preset: str) -> bool:
    """Whether the Linear at ``name`` gets an adapter."""
    if preset not in TARGET_PRESETS:
        raise ValueError(f"lora.targets must be one of {TARGET_PRESETS}, got {preset!r}")
    if not name.startswith("blocks."):
        return preset == "all" and name in IO_LAYERS
    path = name.split(".", 2)[2]
    if path in ATTN:
        return True
    if path in MLP:
        return preset != "attn"
    return preset == "all" and path == "mod.1"


def param_group_of(name: str) -> str:
    """Automagic grouping: each module family adapts its own learning rate."""
    if name.endswith(ATTN):
        return "attn"
    return "mlp" if name.endswith(MLP) else "other"


class VisageLoRANetwork(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        rank: int = 16,
        alpha: float = 16.0,
        targets: str = "blocks",
        prefix: str = "diffusion_model.",
    ):
        super().__init__()
        if targets not in TARGET_PRESETS:
            raise ValueError(f"lora.targets must be one of {TARGET_PRESETS}, got {targets!r}")
        self.transformer = transformer
        self.prefix = prefix
        self.lora_layers = nn.ModuleDict()
        self.name_map: dict[str, str] = {}
        for name, module in list(transformer.named_modules()):
            if isinstance(module, nn.Linear) and selected(name, targets):
                layer = LoRALayer(module.in_features, module.out_features, rank, alpha)
                safe = name.replace(".", "_")
                self.lora_layers[safe] = layer
                self.name_map[safe] = name
                self._wrap(module, layer)
        if not self.lora_layers:
            raise ValueError("No Visage-Image modules matched the LoRA targets")
        counts: dict[str, int] = {}
        for name in self.name_map.values():
            counts[param_group_of(name)] = counts.get(param_group_of(name), 0) + 1
        logger.info(
            "Visage-Image LoRA: %d adapters (%s), rank %d, alpha %s, targets %s",
            len(self.lora_layers),
            counts,
            rank,
            alpha,
            targets,
        )

    @staticmethod
    def _wrap(module: nn.Module, lora: nn.Module) -> None:
        original = module.forward

        def forward(x: torch.Tensor) -> torch.Tensor:
            base, delta = original(x), lora(x)
            return cast(torch.Tensor, (base + delta).to(base.dtype) if delta.dtype != base.dtype else base + delta)

        module.forward = forward

    def set_lora_dtype(self, device: torch.device | str, dtype: torch.dtype) -> None:
        self.lora_layers.to(device=device, dtype=dtype)

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return self.lora_layers.parameters()

    def param_groups(self, split: bool) -> list[dict[str, Any]]:
        if not split:
            return [{"params": list(self.lora_layers.parameters()), "name": "lora"}]
        groups: dict[str, list[nn.Parameter]] = {}
        for safe, layer in self.lora_layers.items():
            groups.setdefault(param_group_of(self.name_map[safe]), []).extend(layer.parameters())
        return [{"params": params, "name": name} for name, params in groups.items()]

    def get_state_dict(self, save_dtype: torch.dtype | None = torch.bfloat16) -> dict[str, torch.Tensor]:
        out = {}
        for safe, layer in cast("dict[str, LoRALayer]", self.lora_layers).items():
            key = f"{self.prefix}{self.name_map[safe]}"
            down = layer.lora_down.weight.detach().cpu()
            up = layer.lora_up.weight.detach().cpu()
            out[f"{key}.lora_A.weight"] = down if save_dtype is None else down.to(save_dtype)
            out[f"{key}.lora_B.weight"] = up if save_dtype is None else up.to(save_dtype)
            out[f"{key}.alpha"] = torch.tensor(float(layer.alpha), dtype=torch.float32)
        return out

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.transformer(*args, **kwargs)
