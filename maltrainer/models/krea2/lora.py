"""
LoRA for the Krea 2 DiT.

Saved keys are ComfyUI's own module paths, ``diffusion_model.<path>.lora_A.weight`` /
``lora_B.weight`` / ``alpha``, which ComfyUI applies to krea2_raw / krea2_turbo without
conversion.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any, cast

import torch
from torch import nn

logger = logging.getLogger(__name__)

BLOCK_ATTN = ("attn.wq", "attn.wk", "attn.wv", "attn.gate", "attn.wo")
BLOCK_MLP = ("mlp.gate", "mlp.up", "mlp.down")
IO_LAYERS = ("first", "tmlp.0", "tmlp.2", "txtmlp.1", "txtmlp.3", "tproj.1", "last.linear")

# Named target sets.
#   attn-mlp   the 28 DiT blocks' attention + MLP Linears (224)
#   blocks+txt those plus the 4 text-fusion blocks' (32 more = 256): OneTrainer's "attn,ff"
#              filter and ai-toolkit's "blocks" filter both select exactly these
#   all        every Linear (264): also txtfusion.projector and the 7 I/O layers -- Fizgig, musubi
TARGET_PRESETS = ("attn-mlp", "blocks+txt", "all")


def _selected(name: str, preset: str, block_range: tuple[int, ...] | None) -> bool:
    if name.startswith("blocks."):
        index = int(name.split(".", 2)[1])
        if block_range is not None and not block_range[0] <= index <= block_range[1]:
            return False
        return name.split(".", 2)[2] in BLOCK_ATTN + BLOCK_MLP
    if name.startswith("txtfusion."):
        if name == "txtfusion.projector":
            return preset == "all"
        return preset in ("blocks+txt", "all")
    return preset == "all" and name in IO_LAYERS


def param_group_of(name: str) -> str:
    """Fizgig's Automagic grouping: each family adapts its own learning rate."""
    if name.startswith("txtfusion."):
        return "txtfusion"
    if ".attn." in name:
        return "attn"
    if ".mlp." in name:
        return "mlp"
    return "other"


class LoRALayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float):
        super().__init__()
        self.rank, self.alpha = rank, alpha
        self.scale = alpha / rank
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = self.lora_down.weight.dtype
        if x.dtype != dtype and not torch.is_autocast_enabled():
            x = x.to(dtype)
        return cast(torch.Tensor, self.lora_up(self.lora_down(x)) * self.scale)


class Krea2LoRANetwork(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        rank: int = 16,
        alpha: float = 16.0,
        targets: str = "attn-mlp",
        block_range: Iterable[int] | None = None,
        prefix: str = "diffusion_model.",
    ):
        super().__init__()
        if targets not in TARGET_PRESETS:
            raise ValueError(f"lora.targets must be one of {TARGET_PRESETS}, got {targets!r}")
        self.transformer = transformer
        self.prefix = prefix
        self.targets = targets
        self.lora_layers = nn.ModuleDict()
        self.name_map: dict[str, str] = {}
        self._active = [True]  # shared with every wrapped forward; see disabled()
        block_range = tuple(block_range) if block_range is not None else None
        for name, module in transformer.named_modules():
            if not hasattr(module, "in_features") or not _selected(name, targets, block_range):
                continue
            layer = LoRALayer(module.in_features, module.out_features, rank, alpha)
            safe = name.replace(".", "_")
            self.lora_layers[safe] = layer
            self.name_map[safe] = name
            self._wrap(module, layer, self._active)
        if not self.lora_layers:
            raise ValueError("No Krea 2 modules matched the LoRA targets")
        counts: dict[str, int] = {}
        for name in self.name_map.values():
            counts[param_group_of(name)] = counts.get(param_group_of(name), 0) + 1
        logger.info(
            "Krea 2 LoRA: %d layers (%s), rank %d, alpha %s, targets %s",
            len(self.lora_layers),
            counts,
            rank,
            alpha,
            targets,
        )

    @staticmethod
    def _wrap(module: nn.Module, lora: nn.Module, active: list[bool]) -> None:
        original = module.forward

        def forward(x: torch.Tensor, _orig: Any = original, _lora: Any = lora, _active: Any = active) -> torch.Tensor:
            base = _orig(x)
            if not _active[0]:
                return cast(torch.Tensor, base)
            delta = _lora(x)
            return cast(torch.Tensor, (base + delta).to(base.dtype) if delta.dtype != base.dtype else base + delta)

        module.forward = forward

    @contextmanager
    def disabled(self) -> Iterator[None]:
        """Run the base model alone: every adapter is skipped until the block exits."""
        self._active[0] = False
        try:
            yield
        finally:
            self._active[0] = True

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
