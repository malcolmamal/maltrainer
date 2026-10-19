"""
LoRA for the FLUX.3 [klein] DiT.

Saved keys are ComfyUI's own module paths, ``diffusion_model.<path>.lora_A.weight`` /
``lora_B.weight`` / ``alpha``. ComfyUI applies them to the bf16 and int8 Klein files
without conversion. Checkpoints with a fused ``qkv`` still get separate ``q_proj`` /
``k_proj`` / ``v_proj`` adapters, which is the layout ComfyUI's FLUX.3 LoRA loader slices into
the fused weight (comfy/lora.py).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)

ATTN = ("q_proj", "k_proj", "v_proj", "attn_out")
MLP = ("mlp_in", "mlp_out")
# The text-to-image path. content_mode_blocks.image_cond (reference images) never runs here.
BLOCK_SECTIONS = ("txt_mode_blocks.", "content_mode_blocks.image.", "single_blocks.")
IO_LAYERS = (
    "emb_in.image", "txt_in", "time_in.in_layer", "time_in.out_layer",
    "early_stream_modulations.image.lin", "early_stream_modulations.txt.lin",
    "single_stream_modulations.image.lin", "single_stream_modulations.txt.lin",
    "final_layer.image.linear", "final_layer.image.adaLN_modulation.1",
)

# Named target sets.
#   blocks  attention + MLP of every block on the text-to-image path: the 5 text, 5 image and
#           28 single blocks (228 adapters). The diffusers FLUX.3 trainer's default set.
#   single  only the 28 joint single blocks (168)
#   all     blocks plus the input, timestep, modulation and output Linears (238)
TARGET_PRESETS = ("blocks", "single", "all")


def selected(name: str, preset: str) -> bool:
    """Whether the (possibly virtual q/k/v) Linear at ``name`` gets an adapter."""
    if preset not in TARGET_PRESETS:
        raise ValueError(f"lora.targets must be one of {TARGET_PRESETS}, got {preset!r}")
    sections = ("single_blocks.",) if preset == "single" else BLOCK_SECTIONS
    if name.startswith(sections):
        return name.rsplit(".", 1)[-1] in ATTN + MLP
    return preset == "all" and name in IO_LAYERS


def param_group_of(name: str) -> str:
    """Automagic grouping: each module family adapts its own learning rate."""
    leaf = name.rsplit(".", 1)[-1]
    if leaf in ATTN:
        return "attn"
    if leaf in MLP:
        return "mlp"
    return "other"


class LoRALayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float):
        super().__init__()
        self.rank, self.alpha = rank, alpha
        self.scale = alpha / rank
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = self.lora_down.weight.dtype
        if x.dtype != dtype and not torch.is_autocast_enabled():
            x = x.to(dtype)
        return self.lora_up(self.lora_down(x)) * self.scale


class Flux3LoRANetwork(nn.Module):
    def __init__(self, transformer: nn.Module, rank: int = 16, alpha: float = 16.0, targets: str = "blocks",
                 prefix: str = "diffusion_model."):
        super().__init__()
        if targets not in TARGET_PRESETS:
            raise ValueError(f"lora.targets must be one of {TARGET_PRESETS}, got {targets!r}")
        self.transformer = transformer
        self.prefix = prefix
        self.targets = targets
        self.lora_layers = nn.ModuleDict()
        self.name_map: Dict[str, str] = {}

        for name, module in list(transformer.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if name.endswith(".qkv"):
                base = name[: -len(".qkv")]
                parts = [f"{base}.{proj}" for proj in ("q_proj", "k_proj", "v_proj")]
                if not selected(parts[0], targets):
                    continue
                third = module.out_features // 3
                layers = [self._add(part, module.in_features, third, rank, alpha) for part in parts]
                self._wrap(module, layers, concat=True)
            elif selected(name, targets):
                self._wrap(module, [self._add(name, module.in_features, module.out_features, rank, alpha)])
        if not self.lora_layers:
            raise ValueError("No FLUX.3 modules matched the LoRA targets")

        counts: Dict[str, int] = {}
        for name in self.name_map.values():
            counts[param_group_of(name)] = counts.get(param_group_of(name), 0) + 1
        logger.info("FLUX.3 LoRA: %d adapters (%s), rank %d, alpha %s, targets %s", len(self.lora_layers), counts, rank, alpha, targets)

    def _add(self, name: str, in_features: int, out_features: int, rank: int, alpha: float) -> LoRALayer:
        layer = LoRALayer(in_features, out_features, rank, alpha)
        safe = name.replace(".", "_")
        self.lora_layers[safe] = layer
        self.name_map[safe] = name
        return layer

    @staticmethod
    def _wrap(module: nn.Module, loras: List[nn.Module], concat: bool = False) -> None:
        original = module.forward

        def forward(x, _orig=original, _loras=loras):
            base = _orig(x)
            delta = torch.cat([lora(x) for lora in _loras], dim=-1) if concat else _loras[0](x)
            return (base + delta).to(base.dtype) if delta.dtype != base.dtype else base + delta

        module.forward = forward

    def set_lora_dtype(self, device, dtype: torch.dtype) -> None:
        self.lora_layers.to(device=device, dtype=dtype)

    def trainable_parameters(self):
        return self.lora_layers.parameters()

    def param_groups(self, split: bool) -> List[dict]:
        if not split:
            return [{"params": list(self.lora_layers.parameters()), "name": "lora"}]
        groups: Dict[str, List[nn.Parameter]] = {}
        for safe, layer in self.lora_layers.items():
            groups.setdefault(param_group_of(self.name_map[safe]), []).extend(layer.parameters())
        return [{"params": params, "name": name} for name, params in groups.items()]

    def get_state_dict(self, save_dtype: Optional[torch.dtype] = torch.bfloat16) -> Dict[str, torch.Tensor]:
        out = {}
        for safe, layer in self.lora_layers.items():
            key = f"{self.prefix}{self.name_map[safe]}"
            down = layer.lora_down.weight.detach().cpu()
            up = layer.lora_up.weight.detach().cpu()
            out[f"{key}.lora_A.weight"] = down if save_dtype is None else down.to(save_dtype)
            out[f"{key}.lora_B.weight"] = up if save_dtype is None else up.to(save_dtype)
            out[f"{key}.alpha"] = torch.tensor(float(layer.alpha), dtype=torch.float32)
        return out

    def forward(self, *args, **kwargs):
        return self.transformer(*args, **kwargs)
