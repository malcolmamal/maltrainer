"""
LoRA network injection for Qwen-Image-2.1 Transformer.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger(__name__)

# Default target linear modules in Qwen-Image-2.1 Transformer blocks
DEFAULT_QWEN_IMAGE_2_1_TARGET_MODULES = [
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "gate_up",
    "img_mlp.out",
]


class QwenImage21LoRALayer(nn.Module):
    """Single LoRA layer for Linear projection in Qwen-Image-2.1."""

    def __init__(self, original_module: nn.Linear, rank: int, alpha: float):
        super().__init__()
        in_features = original_module.in_features
        out_features = original_module.out_features

        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_dtype = self.lora_down.weight.dtype
        if x.is_floating_point() and x.dtype != target_dtype and not torch.is_autocast_enabled():
            x = x.to(target_dtype)
        return self.lora_up(self.lora_down(x)) * self.scale


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

    def _inject_lora(self):
        """Inject LoRA layers into target linear modules of Qwen-Image-2.1 Transformer blocks."""
        injected = 0
        for name, module in self.transformer.named_modules():
            if not name.startswith("transformer_blocks."):
                continue

            # Check if module matches any target pattern
            matched = False
            for target in self.target_modules:
                if name.endswith(f".{target}") or name == target:
                    matched = True
                    break

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

        def forward(x, _orig=original_forward, _lora=lora):
            base = _orig(x)
            delta = _lora(x)
            if delta.dtype == base.dtype:
                return base + delta
            return (base + delta).to(base.dtype)

        module.forward = forward

    def load_assistant_adapter(
        self,
        path: str | Path,
        multiplier: float = 1.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> int:
        """Attach a frozen assistant adapter live during training."""
        from safetensors.torch import load_file

        state = load_file(str(path))
        pairs: dict[str, dict[str, torch.Tensor]] = {}
        for key, value in state.items():
            for suffix, slot in (
                (".lora_A.weight", "down"),
                (".lora_down.weight", "down"),
                (".lora_B.weight", "up"),
                (".lora_up.weight", "up"),
                (".alpha", "alpha"),
            ):
                if key.endswith(suffix):
                    pairs.setdefault(key[: -len(suffix)], {})[slot] = value
                    break

        if not hasattr(self, "assistant_layers"):
            self.assistant_layers = nn.ModuleDict()

        attached, skipped = 0, []
        for name, parts in sorted(pairs.items()):
            if "down" not in parts or "up" not in parts:
                continue
            module_path = name[len(self.prefix) :] if name.startswith(self.prefix) else name
            try:
                module = self.transformer.get_submodule(module_path)
            except AttributeError:
                skipped.append(module_path)
                continue
            if not isinstance(module, nn.Linear):
                skipped.append(module_path)
                continue

            down, up = parts["down"], parts["up"]
            rank = down.shape[0]
            alpha = float(parts["alpha"].item()) if "alpha" in parts else float(rank)
            layer = QwenImage21LoRALayer(module, rank, alpha)
            layer.lora_down.weight.data.copy_(down.to(layer.lora_down.weight.dtype))
            layer.lora_up.weight.data.copy_(up.to(layer.lora_up.weight.dtype))
            layer.scale = (alpha / rank) * multiplier
            layer.requires_grad_(False)
            layer.eval()
            if dtype is not None or device is not None:
                layer.to(device=device, dtype=dtype)

            self.assistant_layers[module_path.replace(".", "_")] = layer
            self._wrap_forward(module, layer)
            attached += 1

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
        return attached

    def set_lora_dtype(self, device: torch.device, dtype: torch.dtype) -> None:
        """Move and cast only the LoRA adapters."""
        self.lora_layers.to(device=device, dtype=dtype)

    def trainable_parameters(self):
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
        for safe_name, lora_layer in self.lora_layers.items():
            name = self.name_map[safe_name]
            full_name = f"{self.prefix}{name}"

            def _store(tensor: torch.Tensor) -> torch.Tensor:
                tensor = tensor.detach().cpu()
                return tensor if save_dtype is None else tensor.to(save_dtype)

            if format_type == "kohya":
                state_dict[f"{full_name}.lora_down.weight"] = _store(lora_layer.lora_down.weight)
                state_dict[f"{full_name}.lora_up.weight"] = _store(lora_layer.lora_up.weight)
            else:
                state_dict[f"{full_name}.lora_A.weight"] = _store(lora_layer.lora_down.weight)
                state_dict[f"{full_name}.lora_B.weight"] = _store(lora_layer.lora_up.weight)

            state_dict[f"{full_name}.alpha"] = torch.tensor(lora_layer.alpha, dtype=torch.float32)
        return state_dict

    def forward(self, *args, **kwargs):
        return self.transformer(*args, **kwargs)
