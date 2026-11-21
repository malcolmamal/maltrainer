"""
LoRA for the LTX-2.5 DiT.

Saved keys are ComfyUI's own module paths, ``diffusion_model.<path>.lora_A.weight`` / ``lora_B.weight`` / ``alpha``,
which ComfyUI applies to the bf16 and int8 files without conversion.

Target sets (the ``to_*`` projections of every attention, the two ``ff`` Linears of every feed-forward):
- ``attn``   all attention: video, audio and the audio <-> video cross-attention (the official t2v LoRA)
- ``video``  the video stream's self- and text cross-attention only: leaves audio and cross-modal layers untouched
- ``attn_ff`` ``attn`` plus every feed-forward
- ``video_ff`` ``video`` plus the video feed-forward
- ``full``   every Linear of every block
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from typing import Any, cast

import torch
from torch import nn

from maltrainer.models.flux3.lora import LoRALayer

logger = logging.getLogger(__name__)

ATTENTION = r"transformer_blocks\.\d+\.(?P<module>{names})\.(to_q|to_k|to_v|to_out\.0)"
FEED_FORWARD = r"transformer_blocks\.\d+\.(?P<module>{names})\.net\.(0\.proj|2)"
VIDEO_ATTN, ALL_ATTN = "attn1|attn2", "attn1|attn2|audio_attn1|audio_attn2|audio_to_video_attn|video_to_audio_attn"
VIDEO_FF, ALL_FF = "ff", "ff|audio_ff"
PRESETS: dict[str, tuple[str, ...]] = {
    "attn": (ATTENTION.format(names=ALL_ATTN),),
    "video": (ATTENTION.format(names=VIDEO_ATTN),),
    "attn_ff": (ATTENTION.format(names=ALL_ATTN), FEED_FORWARD.format(names=ALL_FF)),
    "video_ff": (ATTENTION.format(names=VIDEO_ATTN), FEED_FORWARD.format(names=VIDEO_FF)),
    "full": (r"transformer_blocks\.\d+\..*",),
}
TARGET_PRESETS = tuple(PRESETS)


def selected(name: str, preset: str) -> bool:
    """Whether the Linear at ``name`` gets an adapter."""
    if preset not in PRESETS:
        raise ValueError(f"lora.targets must be one of {TARGET_PRESETS}, got {preset!r}")
    return any(re.fullmatch(pattern, name) for pattern in PRESETS[preset])


def param_group_of(name: str) -> str:
    """Automagic grouping: each module family adapts its own learning rate."""
    if ".ff." in name or ".audio_ff." in name:
        return "mlp"
    return "attn" if re.search(r"\.(to_q|to_k|to_v|to_out\.0)$", name) else "other"


class LTX25LoRANetwork(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        rank: int = 16,
        alpha: float = 16.0,
        targets: str = "attn",
        prefix: str = "diffusion_model.",
    ):
        super().__init__()
        if targets not in PRESETS:
            raise ValueError(f"lora.targets must be one of {TARGET_PRESETS}, got {targets!r}")
        self.transformer, self.prefix = transformer, prefix
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
            raise ValueError("No LTX-2.5 modules matched the LoRA targets")
        counts: dict[str, int] = {}
        for name in self.name_map.values():
            counts[param_group_of(name)] = counts.get(param_group_of(name), 0) + 1
        logger.info(
            "LTX-2.5 LoRA: %d adapters (%s), rank %d, alpha %s, targets %s",
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
