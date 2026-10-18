"""
LoRA for the MOSS-Image DiT.

Saved keys are ComfyUI's own module paths, ``diffusion_model.<path>.lora_A.weight`` /
``lora_B.weight`` / ``alpha``. ComfyUI applies them to the bf16 and int8 files without
conversion. Checkpoints with a fused ``proj.qkv`` still get separate ``proj.q`` / ``proj.k`` /
``proj.v`` adapters, which is the layout ComfyUI's MOSS-Image LoRA loader slices into the fused
weight (comfy/lora.py).
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional

import torch
from torch import nn

from maltrainer.models.flux3.lora import LoRALayer

logger = logging.getLogger(__name__)

# Leaf names inside layers.N.proj (the 42 joint blocks) and text_embedder.blocks.N (the 2 text blocks).
BLOCK_ATTN = ("q", "k", "v", "o")
BLOCK_FFN = ("ffn_up_gate", "ffn_down")
TEXT_ATTN = ("qkv", "proj")
TEXT_FFN = ("ffn_gate", "ffn_up", "ffn_down")
IO_LAYERS = (
    "video_embedder", "text_embedder.proj", "time_embedding.0", "time_embedding.2",
    "shared_block_modulation", "video_head_modulation", "final_linear_video",
)

# Named target sets.
#   blocks  attention + feed-forward of the 42 joint blocks (252 adapters)
#   attn    attention only of the 42 joint blocks (168)
#   all     blocks + the 2 text-embedder blocks + the input, timestep, modulation and output
#           Linears (269)
TARGET_PRESETS = ("blocks", "attn", "all")

# The blocks the reference sampler skips in CFG's negative branch (the pipeline's
# uncond_skip_layers=[22]; ComfyUI's SkipLayerGuidanceDiTSimple on layer 22). A LoRA there acts on
# the positive branch only, so CFG multiplies its effect instead of partly cancelling it. AI
# Toolkit and Fizgig leave these blocks out; so does lora.skip_blocks by default.
UNCOND_SKIP_BLOCKS = (22,)


def _block_leaf(name: str) -> Optional[str]:
    parts = name.split(".")
    if len(parts) == 4 and parts[0] == "layers" and parts[2] == "proj":
        return parts[3]
    return None


def _text_leaf(name: str) -> Optional[str]:
    parts = name.split(".")
    if len(parts) == 4 and parts[:2] == ["text_embedder", "blocks"]:
        return parts[3]
    return None


def selected(name: str, preset: str) -> bool:
    """Whether the (possibly virtual q/k/v) Linear at ``name`` gets an adapter."""
    if preset not in TARGET_PRESETS:
        raise ValueError(f"lora.targets must be one of {TARGET_PRESETS}, got {preset!r}")
    leaf = _block_leaf(name)
    if leaf is not None:
        return leaf in (BLOCK_ATTN if preset == "attn" else BLOCK_ATTN + BLOCK_FFN)
    if preset != "all":
        return False
    leaf = _text_leaf(name)
    if leaf is not None:
        return leaf in TEXT_ATTN + TEXT_FFN
    return name in IO_LAYERS


def param_group_of(name: str) -> str:
    """Automagic grouping: each module family adapts its own learning rate."""
    leaf = _block_leaf(name) or _text_leaf(name)
    if leaf in BLOCK_ATTN + TEXT_ATTN:
        return "attn"
    if leaf in BLOCK_FFN + TEXT_FFN:
        return "mlp"
    return "other"


class MOSSImageLoRANetwork(nn.Module):
    def __init__(self, transformer: nn.Module, rank: int = 16, alpha: float = 16.0, targets: str = "blocks",
                 prefix: str = "diffusion_model.", skip_blocks: Iterable[int] = UNCOND_SKIP_BLOCKS):
        super().__init__()
        if targets not in TARGET_PRESETS:
            raise ValueError(f"lora.targets must be one of {TARGET_PRESETS}, got {targets!r}")
        self.transformer = transformer
        self.prefix = prefix
        self.targets = targets
        self.skip_blocks = tuple(sorted({int(i) for i in skip_blocks}))
        skipped = tuple(f"layers.{i}." for i in self.skip_blocks)
        self.lora_layers = nn.ModuleDict()
        self.name_map: Dict[str, str] = {}

        for name, module in list(transformer.named_modules()):
            if not isinstance(module, nn.Linear) or name.startswith(skipped):
                continue
            if _block_leaf(name) == "qkv":
                base = name[: -len(".qkv")]
                parts = [f"{base}.{proj}" for proj in ("q", "k", "v")]
                if not selected(parts[0], targets):
                    continue
                third = module.out_features // 3
                layers = [self._add(part, module.in_features, third, rank, alpha) for part in parts]
                self._wrap(module, layers, concat=True)
            elif selected(name, targets):
                self._wrap(module, [self._add(name, module.in_features, module.out_features, rank, alpha)])
        if not self.lora_layers:
            raise ValueError("No MOSS-Image modules matched the LoRA targets")

        counts: Dict[str, int] = {}
        for name in self.name_map.values():
            counts[param_group_of(name)] = counts.get(param_group_of(name), 0) + 1
        logger.info(
            "MOSS-Image LoRA: %d adapters (%s), rank %d, alpha %s, targets %s, skipped blocks %s",
            len(self.lora_layers), counts, rank, alpha, targets, list(self.skip_blocks) or "none",
        )

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
