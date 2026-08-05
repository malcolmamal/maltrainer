"""
LoRA network injection for MiniMax H3 Transformer blocks.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MiniMaxH3LoRALayer(nn.Module):
    """Single LoRA layer for Linear projection in MiniMax H3."""

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
        return self.lora_up(self.lora_down(x)) * self.scale


class MiniMaxH3LoRANetwork(nn.Module):
    """LoRA network adapter for MiniMax H3 Transformer."""

    def __init__(
        self,
        transformer: nn.Module,
        rank: int = 16,
        alpha: float = 16.0,
        target_modules: Optional[List[str]] = None,
        prefix: str = "diffusion_model.",
    ):
        super().__init__()
        self.transformer = transformer
        self.rank = rank
        self.alpha = alpha
        self.target_modules = target_modules or ["qkv_proj", "out_proj", "fc1", "fc2"]
        self.prefix = prefix

        self.lora_layers = nn.ModuleDict()
        self.name_map: Dict[str, str] = {}
        self._inject_lora()

    def _inject_lora(self):
        """Inject LoRA layers into target linear modules of MiniMaxH3TransformerBlock."""
        injected = 0
        for name, module in self.transformer.named_modules():
            # Skip AdaLN, modulation, or norm projections
            if "adaln_proj" in name or "modulation" in name or "norm" in name:
                continue

            if any(target in name for target in self.target_modules):
                if isinstance(module, nn.Linear):
                    lora_layer = MiniMaxH3LoRALayer(module, self.rank, self.alpha)
                    safe_name = name.replace(".", "_")
                    self.lora_layers[safe_name] = lora_layer
                    self.name_map[safe_name] = name

                    original_forward = module.forward

                    def make_forward(orig_fwd, lora):
                        def forward(x):
                            return orig_fwd(x) + lora(x)
                        return forward

                    module.forward = make_forward(original_forward, lora_layer)
                    injected += 1

        logger.info("Injected %d LoRA layers into MiniMax H3 transformer (rank=%d, alpha=%s)", injected, self.rank, self.alpha)

    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get LoRA state dict for saving in Kohya/AI-Toolkit safetensors format."""
        state_dict = {}
        for safe_name, lora_layer in self.lora_layers.items():
            name = self.name_map[safe_name]
            full_name = f"{self.prefix}{name}"
            state_dict[f"{full_name}.lora_A.weight"] = lora_layer.lora_down.weight
            state_dict[f"{full_name}.lora_B.weight"] = lora_layer.lora_up.weight
            state_dict[f"{full_name}.alpha"] = torch.tensor(lora_layer.alpha)
        return state_dict

    def forward(self, *args, **kwargs):
        return self.transformer(*args, **kwargs)
