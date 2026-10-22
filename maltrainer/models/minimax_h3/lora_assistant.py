"""Reading a frozen assistant ("training") adapter file and building its layers for the H3 transformer."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn

from maltrainer.models.minimax_h3.lora_layers import GridAdaLNLoRA, MiniMaxH3LoRALayer

_SLOT_SUFFIXES = (
    (".lora_A.weight", "down"),
    (".lora_down.weight", "down"),
    (".lora_B.weight", "up"),
    (".lora_up.weight", "up"),
    (".alpha", "alpha"),
)


def read_adapter_pairs(path: str) -> dict[str, dict[str, torch.Tensor]]:
    """The adapter's tensors grouped per module name into ``down`` / ``up`` / ``alpha`` slots."""
    state = load_file(str(path))
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in state.items():
        for suffix, slot in _SLOT_SUFFIXES:
            if key.endswith(suffix):
                pairs.setdefault(key[: -len(suffix)], {})[slot] = value
                break
    return pairs


class AssistantAttacher:
    """One ``load_assistant_adapter`` pass: builds the frozen layers and keeps what it attached or left out."""

    def __init__(
        self,
        transformer: nn.Module,
        prefix: str,
        multiplier: float,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
        adaln_grid_path: str | None,
    ):
        self.transformer = transformer
        self.prefix = prefix
        self.multiplier = multiplier
        self.device = device
        self.dtype = dtype
        self.adaln_grid_path = adaln_grid_path
        self.kohya_names = {
            "lora_unet_" + name.replace(".", "_"): name
            for name, module in transformer.named_modules()
            if isinstance(module, nn.Linear)
        }
        self.table = getattr(transformer, "adaln_t_table", None)
        self.grid: torch.Tensor | None = None
        self.attached = self.grid_rows = self.rank = 0
        self.skipped: list[str] = []
        self.unbridged: list[str] = []

    def resolve(self, name: str) -> tuple[str, nn.Linear | None]:
        """The module path an adapter entry names and its Linear, or None (recorded as skipped)."""
        module_path = name[len(self.prefix) :] if name.startswith(self.prefix) else name
        module_path = self.kohya_names.get(module_path, module_path)
        try:
            module = self.transformer.get_submodule(module_path)
        except AttributeError:
            self.skipped.append(module_path)
            return module_path, None
        if not isinstance(module, nn.Linear):
            self.skipped.append(module_path)
            return module_path, None
        return module_path, module

    def build(
        self, module_path: str, module: nn.Linear, parts: dict[str, torch.Tensor]
    ) -> GridAdaLNLoRA | MiniMaxH3LoRALayer | None:
        """A frozen layer for one adapter entry, or None when it cannot be attached (recorded)."""
        down, up = parts["down"], parts["up"]
        self.rank = rank = down.shape[0]
        alpha = float(parts["alpha"].item()) if "alpha" in parts else float(rank)
        scale = (alpha / rank) * self.multiplier

        layer: GridAdaLNLoRA | MiniMaxH3LoRALayer | None
        if down.shape[1] != module.in_features:
            # A full-model AdaLN row on the pruned base (see ``load_assistant_adapter``).
            layer = self._grid_layer(module_path, module, down, up, scale)
            if layer is None:
                return None
        else:
            layer = MiniMaxH3LoRALayer(module, rank, alpha)
            layer.lora_down.weight.data.copy_(down.to(layer.lora_down.weight.dtype))
            layer.lora_up.weight.data.copy_(up.to(layer.lora_up.weight.dtype))
            layer.scale = scale
        layer.requires_grad_(False)
        layer.eval()
        if isinstance(layer, GridAdaLNLoRA):
            layer.to(device=self.device)  # fp32: the nearest-row match needs the table exact
        elif self.dtype is not None or self.device is not None:
            layer.to(device=self.device, dtype=self.dtype)
        return layer

    def _grid_layer(
        self, module_path: str, module: nn.Linear, down: torch.Tensor, up: torch.Tensor, scale: float
    ) -> GridAdaLNLoRA | None:
        if "adaln_proj" not in module_path or self.table is None:
            self.skipped.append(f"{module_path} (in {down.shape[1]} != {module.in_features})")
            return None
        if self.adaln_grid_path is None:
            self.unbridged.append(module_path)
            return None
        if self.grid is None:
            self.grid = load_file(str(self.adaln_grid_path))["silu_t_emb_grid"].float()
            if self.grid.shape[0] != self.table.shape[0]:
                raise ValueError(
                    f"AdaLN grid has {self.grid.shape[0]} rows but this checkpoint's curve table has "
                    f"{self.table.shape[0]}; they must come from the same H3 build."
                )
        layer = GridAdaLNLoRA(down, up, scale, self.table, self.grid)
        self.grid_rows += 1
        return layer

    def report(self, path: str, logger: logging.Logger) -> None:
        """Raise on modules the transformer lacks; log the unbridged AdaLN rows and what was attached."""
        if self.skipped:
            raise ValueError(
                f"Training adapter targets {len(self.skipped)} module(s) this transformer does not "
                f"expose; examples: {self.skipped[:5]}. Is it built for a different H3 checkpoint?"
            )
        if self.unbridged:
            logger.warning(
                "Training adapter has %d full-model AdaLN row(s) the pruned base cannot run without "
                "model.training_adapter_adaln_grid (Fizgig's assets/h3_silu_temb_grid.safetensors); "
                "they were skipped, so the adapter acts only through its other %d modules.",
                len(self.unbridged),
                self.attached,
            )
        logger.info(
            "Attached frozen training adapter: %d modules (%d AdaLN via the t_emb grid), rank %d, "
            "multiplier %.2f (from %s)",
            self.attached,
            self.grid_rows,
            self.rank,
            self.multiplier,
            Path(path).name,
        )
