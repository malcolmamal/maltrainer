"""LoRA adapters for SupraDiT linear layers."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger(__name__)

DEFAULT_SUPRA_TARGET_MODULES = [
    "self_attn.qkv",
    "self_attn.proj",
    "cross_attn.q",
    "cross_attn.kv",
    "cross_attn.proj",
    "mlp.0",
    "mlp.2",
]


class SupraLoRALayer(nn.Module):
    def __init__(self, original_module: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.lora_down = nn.Linear(original_module.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, original_module.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_up(self.lora_down(x)) * self.scale


class SupraLoRANetwork(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        rank: int = 16,
        alpha: float = 16.0,
        target_modules: list[str] | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.transformer = transformer
        self.rank = rank
        self.alpha = alpha
        self.target_modules = target_modules or list(DEFAULT_SUPRA_TARGET_MODULES)
        self.prefix = prefix
        self.lora_layers = nn.ModuleDict()
        self.name_map: dict[str, str] = {}
        self._inject_lora()

    def _matches(self, name: str) -> bool:
        if not name.startswith("blocks."):
            return False
        return any(name.endswith(target) for target in self.target_modules)

    def _inject_lora(self) -> None:
        injected = 0
        for name, module in self.transformer.named_modules():
            if isinstance(module, nn.Linear) and self._matches(name):
                layer = SupraLoRALayer(module, self.rank, self.alpha)
                safe_name = name.replace(".", "_")
                self.lora_layers[safe_name] = layer
                self.name_map[safe_name] = name
                self._wrap_forward(module, layer)
                injected += 1
        if injected == 0:
            raise RuntimeError(f"No SupraDiT linears matched LoRA targets {self.target_modules}")
        logger.info("Injected %d Supra2-IMG LoRA layers (rank=%d, alpha=%s)", injected, self.rank, self.alpha)

    @staticmethod
    def _wrap_forward(module: nn.Linear, lora: nn.Module) -> None:
        original_forward = module.forward

        def forward(x, _orig=original_forward, _lora=lora):
            return _orig(x) + _lora(x)

        module.forward = forward

    def set_lora_dtype(self, device: torch.device, dtype: torch.dtype) -> None:
        self.lora_layers.to(device=device, dtype=dtype)

    def trainable_parameters(self):
        return self.lora_layers.parameters()

    def get_state_dict(self, save_dtype: torch.dtype | None = torch.bfloat16) -> dict[str, torch.Tensor]:
        state = {}
        for safe_name, layer in self.lora_layers.items():
            name = f"{self.prefix}{self.name_map[safe_name]}"

            def _store(tensor: torch.Tensor) -> torch.Tensor:
                tensor = tensor.detach().cpu()
                return tensor if save_dtype is None else tensor.to(save_dtype)

            state[f"{name}.lora_A.weight"] = _store(layer.lora_down.weight)
            state[f"{name}.lora_B.weight"] = _store(layer.lora_up.weight)
            state[f"{name}.alpha"] = torch.tensor(layer.alpha, dtype=torch.float32)
        return state

    def load_weights(self, path: str | Path) -> int:
        from safetensors.torch import load_file

        tensors = load_file(str(path), device="cpu")
        missing = []
        loaded = 0
        for safe_name, layer in self.lora_layers.items():
            name = f"{self.prefix}{self.name_map[safe_name]}"
            down_key = f"{name}.lora_A.weight"
            up_key = f"{name}.lora_B.weight"
            if down_key not in tensors or up_key not in tensors:
                missing.append(name)
                continue
            layer.lora_down.weight.data.copy_(tensors[down_key].to(dtype=layer.lora_down.weight.dtype))
            layer.lora_up.weight.data.copy_(tensors[up_key].to(dtype=layer.lora_up.weight.dtype))
            alpha_key = f"{name}.alpha"
            if alpha_key in tensors:
                layer.alpha = float(tensors[alpha_key])
                layer.scale = layer.alpha / layer.rank
            loaded += 1
        if missing:
            preview = ", ".join(missing[:3])
            raise RuntimeError(f"Resume LoRA is missing {len(missing)} layers, including {preview}")
        logger.info("Loaded %d Supra2-IMG LoRA layers from %s", loaded, path)
        return loaded

    def forward(self, *args, **kwargs):
        return self.transformer(*args, **kwargs)
