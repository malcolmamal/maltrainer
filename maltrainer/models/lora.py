"""
LoRA (Low-Rank Adaptation) implementation for Z-Image transformers.
"""

import logging
from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import nn

logger = logging.getLogger(__name__)


class LoRALayer(nn.Module):
    """Single LoRA layer with down/up projection."""

    def __init__(self, original_module: nn.Module, rank: int, alpha: float):
        super().__init__()

        if isinstance(original_module, nn.Linear):
            in_features = original_module.in_features
            out_features = original_module.out_features
        else:
            raise ValueError(f"Unsupported module type: {type(original_module)}")

        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        # LoRA decomposition: W = W_0 + (B @ A) * scale
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)

        # Initialize
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LoRA layer."""
        return self.lora_up(self.lora_down(x)) * self.scale

    def set_alpha(self, alpha: float):
        """Update alpha and scaling factor."""
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank


class LoRANetwork(nn.Module):
    """LoRA network wrapper for transformer models."""

    def __init__(
        self,
        transformer: nn.Module,
        rank: int = 4,
        alpha: float = 1.0,
        target_modules: list[str] | None = None,
    ):
        super().__init__()

        self.transformer = transformer
        self.rank = rank
        self.alpha = alpha
        self.target_modules = target_modules or ["to_q", "to_k", "to_v", "to_out"]
        self.prefix = ""  # Can be set to "diffusion_model." for some loaders

        self.lora_layers = nn.ModuleDict()
        self.name_map: dict[str, str] = {}  # Map safe_name to original dot_name
        self._inject_lora()

    def _inject_lora(self):
        """Inject LoRA layers into the transformer."""
        for name, module in self.transformer.named_modules():
            # Check if this module should have LoRA
            if any(target in name for target in self.target_modules) and isinstance(module, nn.Linear):
                # Create LoRA layer
                lora_layer = LoRALayer(module, self.rank, self.alpha)

                # Store it
                safe_name = name.replace(".", "_")
                self.lora_layers[safe_name] = lora_layer
                self.name_map[safe_name] = name

                # Monkey-patch the forward method
                original_forward = module.forward

                def make_forward(orig_fwd, lora):
                    def forward(x):
                        return orig_fwd(x) + lora(x)

                    return forward

                module.forward = make_forward(original_forward, lora_layer)

    def get_state_dict(self) -> dict:
        """Get LoRA state dict for saving in Kohya/AI-Toolkit format."""
        state_dict = {}
        for safe_name, lora_layer in self.lora_layers.items():
            lora_layer = cast(LoRALayer, lora_layer)
            name = self.name_map[safe_name]
            full_name = f"{self.prefix}{name}"
            # Map lora_down -> lora_A, lora_up -> lora_B
            state_dict[f"{full_name}.lora_A.weight"] = lora_layer.lora_down.weight
            state_dict[f"{full_name}.lora_B.weight"] = lora_layer.lora_up.weight
            state_dict[f"{full_name}.alpha"] = torch.tensor(lora_layer.alpha)
        return state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):  # noqa: ARG002 - nn.Module's signature
        """Load LoRA weights from a state dict: kohya's lora_down / lora_up names or the lora_A /
        lora_B ones get_state_dict writes, by dot or safe name, with or without a prefix."""
        missing = []
        for safe_name, lora_layer in self.lora_layers.items():
            lora_layer = cast(LoRALayer, lora_layer)
            keys = _layer_keys(state_dict, (self.name_map[safe_name], safe_name))
            if keys is None:
                missing.append(self.name_map[safe_name])
                continue
            down, up, alpha = keys
            lora_layer.lora_down.weight.data = state_dict[down]
            lora_layer.lora_up.weight.data = state_dict[up]
            if alpha in state_dict:
                lora_layer.set_alpha(float(state_dict[alpha]))
        if missing:
            logger.warning(
                "LoRA: %d of %d layers not in the state dict, e.g. %s", len(missing), len(self.lora_layers), missing[:3]
            )

    def forward(self, *args, **kwargs):
        """Forward pass through transformer with LoRA."""
        return self.transformer(*args, **kwargs)


# (down, up) key suffixes: kohya's names, then the PEFT / AI-Toolkit ones get_state_dict writes
LORA_SUFFIXES = ((".lora_down.weight", ".lora_up.weight"), (".lora_A.weight", ".lora_B.weight"))


def _layer_keys(state_dict: Mapping[str, Any], names: tuple) -> tuple | None:
    """(down, up, alpha) keys of the first of ``names`` in ``state_dict``: an exact key first, then
    one behind a prefix ending in "." or "_" ("diffusion_model.", "lora_unet_")."""
    for down, up in LORA_SUFFIXES:
        for name in names:
            if f"{name}{down}" in state_dict:
                return f"{name}{down}", f"{name}{up}", f"{name}.alpha"
    for down, up in LORA_SUFFIXES:
        for key in state_dict:
            stem = key[: -len(down)] if key.endswith(down) else None
            if stem and any(stem.endswith((f".{name}", f"_{name}")) for name in names):
                return key, f"{stem}{up}", f"{stem}.alpha"
    return None
