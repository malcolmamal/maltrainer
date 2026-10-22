"""
LoRA network injection for Qwen-Image-2.1 Transformer.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from maltrainer.models.qwen_image_2_1.lora_assistant import build_assistant_layer, find_linear, read_adapter_pairs
from maltrainer.models.qwen_image_2_1.lora_layer import QwenImage21LoRALayer

logger = logging.getLogger(__name__)

# Default target linear modules in Qwen-Image-2.1 Transformer blocks
DEFAULT_QWEN_IMAGE_2_1_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0", "gate_up", "img_mlp.out"]

__all__ = ["DEFAULT_QWEN_IMAGE_2_1_TARGET_MODULES", "QwenImage21LoRALayer", "QwenImage21LoRANetwork"]


def _log_assistant(attached: int, skipped: list[str], rank: int, multiplier: float, path: str | Path) -> None:
    if skipped:
        logger.warning(
            "Assistant adapter skipped %d module(s) not found in transformer: %s",
            len(skipped),
            skipped[:5],
        )
    logger.info(
        "Attached frozen assistant adapter: %d modules, rank %d, multiplier %.2f (from %s)",
        attached,
        rank,
        multiplier,
        Path(path).name,
    )


class QwenImage21LoRANetwork(nn.Module):
    """LoRA network adapter for Qwen-Image-2.1 Transformer."""

    def __init__(
        self,
        transformer: nn.Module,
        rank: int = 16,
        alpha: float = 16.0,
        target_modules: list[str] | None = None,
        prefix: str = "diffusion_model.",
    ):
        super().__init__()
        self.transformer = transformer
        self.rank = rank
        self.alpha = alpha
        self.target_modules = target_modules or DEFAULT_QWEN_IMAGE_2_1_TARGET_MODULES
        self.prefix = prefix

        self.lora_layers = nn.ModuleDict()
        self.name_map: dict[str, str] = {}
        self._inject_lora()

    def _inject_lora(self) -> None:
        """Inject LoRA layers into target linear modules of Qwen-Image-2.1 Transformer blocks."""
        injected = 0
        for name, module in self.transformer.named_modules():
            if not name.startswith("transformer_blocks."):
                continue

            # Check if module matches any target pattern
            matched = any(name.endswith(f".{target}") or name == target for target in self.target_modules)

            if matched and isinstance(module, nn.Linear):
                self._add_lora_layer(name, module)
                injected += 1

        logger.info(
            "Injected %d LoRA layers into Qwen-Image-2.1 transformer (rank=%d, alpha=%s)",
            injected,
            self.rank,
            self.alpha,
        )

    def _add_lora_layer(self, name: str, module: nn.Linear) -> None:
        lora_layer = QwenImage21LoRALayer(module, self.rank, self.alpha)
        safe_name = name.replace(".", "_")
        self.lora_layers[safe_name] = lora_layer
        self.name_map[safe_name] = name
        self._wrap_forward(module, lora_layer)

    @staticmethod
    def _wrap_forward(module: nn.Linear, lora: nn.Module) -> None:
        original_forward = module.forward

        def forward(x: torch.Tensor, _orig: Any = original_forward, _lora: Any = lora) -> torch.Tensor:
            base = _orig(x)
            delta = _lora(x)
            if delta.dtype == base.dtype:
                return cast(torch.Tensor, base + delta)
            return cast(torch.Tensor, (base + delta).to(base.dtype))

        module.forward = forward  # type: ignore[method-assign, assignment]  # monkeypatching the instance forward is the point; the stub types forward as a method

    def load_assistant_adapter(
        self,
        path: str | Path,
        multiplier: float = 1.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> int:
        """Attach a frozen assistant adapter live during training."""
        pairs = read_adapter_pairs(path)

        if not hasattr(self, "assistant_layers"):
            self.assistant_layers = nn.ModuleDict()

        attached, skipped, rank = 0, [], 0
        for name, parts in sorted(pairs.items()):
            if "down" not in parts or "up" not in parts:
                continue
            module_path = name[len(self.prefix) :] if name.startswith(self.prefix) else name
            module = find_linear(self.transformer, module_path)
            if module is None:
                skipped.append(module_path)
                continue

            layer, rank = build_assistant_layer(module, parts, multiplier, device, dtype)
            self.assistant_layers[module_path.replace(".", "_")] = layer
            self._wrap_forward(module, layer)
            attached += 1

        _log_assistant(attached, skipped, rank, multiplier, path)
        return attached

    def set_lora_dtype(self, device: torch.device, dtype: torch.dtype) -> None:
        """Move and cast only the LoRA adapters."""
        self.lora_layers.to(device=device, dtype=dtype)

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Yield only the adapter parameters for optimization."""
        return self.lora_layers.parameters()

    def get_state_dict(
        self,
        save_dtype: torch.dtype | None = torch.bfloat16,
        format_type: str = "diffusers",
    ) -> dict[str, torch.Tensor]:
        """Get LoRA state dict for saving in safetensors format.

        Supports both standard diffusers/ai-toolkit format (.lora_A / .lora_B)
        and kohya format (.lora_down / .lora_up).
        """
        state_dict = {}
        for safe_name, lora_layer in cast("dict[str, QwenImage21LoRALayer]", self.lora_layers).items():
            name = self.name_map[safe_name]
            full_name = f"{self.prefix}{name}"

            def _store(tensor: torch.Tensor) -> torch.Tensor:
                # copy=True: .cpu() and .to(same dtype) return the live weight itself, which training changes
                return tensor.detach().to("cpu", save_dtype or tensor.dtype, copy=True)

            if format_type == "kohya":
                state_dict[f"{full_name}.lora_down.weight"] = _store(lora_layer.lora_down.weight)
                state_dict[f"{full_name}.lora_up.weight"] = _store(lora_layer.lora_up.weight)
            else:
                state_dict[f"{full_name}.lora_A.weight"] = _store(lora_layer.lora_down.weight)
                state_dict[f"{full_name}.lora_B.weight"] = _store(lora_layer.lora_up.weight)

            state_dict[f"{full_name}.alpha"] = torch.tensor(lora_layer.alpha, dtype=torch.float32)
        return state_dict

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.transformer(*args, **kwargs)
