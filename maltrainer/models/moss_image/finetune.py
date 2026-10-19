"""
Full finetuning of the MOSS-Image DiT: what is MOSS-Image's own. Which parameters train by
default, their Automagic groups, writing the result as a ComfyUI single file, and how a LoRA is
extracted from it. The model-agnostic parts (parameter selection, prior preservation,
regularisation images, the SVD extraction) are in maltrainer/finetune/.

A saved model keeps the source checkpoint's exact layout: the same keys in the same order, the
same container prefix, and the entries training does not use (``rope.bands``, the
``comfy_attention.config`` hints) copied through unchanged. So ComfyUI loads it exactly as it
loads moss_image_base_bf16, and the trainer can load it back as a base or to resume.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from maltrainer.finetune.extract import ExtractProfile
from maltrainer.models.moss_image.loading import CONTAINER_PREFIXES

logger = logging.getLogger(__name__)

ATTN_LEAVES = ("q", "k", "v", "qkv", "o", "proj")

# The default trainable set: the Linear weights of the 42 joint blocks (finetune.linears_only).
# The text side stays frozen, as OpenMOSS's guide asks ("keep the text encoder frozen") and
# Fizgig does: the text embedder (projection + 2 blocks) reads every prompt, so retraining it on a
# few captions changes how all prompts are read, not only the trigger words. The embedders, the
# shared modulation, the output head and the norms stay frozen too.
DEFAULT_TRAINABLE_PREFIXES = ("layers.",)


def param_group_of(param_name: str) -> str:
    """Automagic grouping by module family: attention, feed-forward, block norms, the rest."""
    module = param_name.rsplit(".", 1)[0]
    leaf = module.rsplit(".", 1)[-1]
    if module.startswith(("layers.", "text_embedder.blocks.")):
        if leaf in ATTN_LEAVES:
            return "attn"
        if leaf.startswith("ffn"):
            return "mlp"
        return "norm"
    return "other"


def param_groups(named: List[Tuple[str, nn.Parameter]], split: bool) -> List[dict]:
    if not split:
        return [{"params": [p for _, p in named], "name": "all"}]
    groups: Dict[str, List[nn.Parameter]] = {}
    for name, param in named:
        groups.setdefault(param_group_of(name), []).append(param)
    return [{"params": params, "name": name} for name, params in groups.items()]


def _container_prefix(keys: List[str]) -> str:
    for prefix in CONTAINER_PREFIXES:
        if keys and all(key.startswith(prefix) for key in keys):
            return prefix
    return ""


def save_full_model(
    model: nn.Module,
    layout_path: Union[str, Path],
    out_path: Union[str, Path],
    overrides: Optional[Dict[str, torch.Tensor]] = None,
    metadata: Optional[Dict[str, str]] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Write ``model`` in the key layout of the plain (bf16) checkpoint at ``layout_path``.

    ``overrides`` maps module-name keys to tensors that replace the live weights, e.g. the EMA
    average. Keys the model does not have are copied from ``layout_path`` as stored.
    """
    overrides = overrides or {}
    live = model.state_dict()
    out: Dict[str, torch.Tensor] = {}
    with safe_open(str(layout_path), framework="pt", device="cpu") as source:
        keys = list(source.keys())
        prefix = _container_prefix(keys)
        for key in keys:
            name = key[len(prefix) :]
            tensor = overrides.get(name, live.get(name))
            if tensor is None:
                out[key] = source.get_tensor(key)
            else:
                out[key] = tensor.detach().to("cpu", dtype).contiguous()
    missing = set(live) - {key[len(prefix) :] for key in keys}
    if missing:
        raise ValueError(
            f"{Path(layout_path).name} has no entry for {sorted(missing)[:5]}; "
            "save against the file the model was loaded from"
        )
    tmp = Path(str(out_path) + ".tmp")
    save_file(out, str(tmp), metadata=metadata)
    tmp.replace(out_path)


# ---------------------------------------------------------------- training monitor

# Blocks whose weights the monitor's drift line tracks: first, last and three between.
DRIFT_BLOCKS = (0, 10, 21, 31, 41)


def drift_targets(named: List[Tuple[str, nn.Parameter]]) -> List[Tuple[str, nn.Parameter]]:
    """(label, parameter) for maltrainer.utils.train_monitor.WeightDrift: the attention output and
    the feed-forward down projection of DRIFT_BLOCKS, where they train. ~350 MB of bf16 copies."""
    leaves = {"proj.o.weight": "attn", "proj.ffn_down.weight": "mlp"}
    out = []
    for name, param in named:
        parts = name.split(".", 2)
        if len(parts) == 3 and parts[0] == "layers" and int(parts[1]) in DRIFT_BLOCKS and parts[2] in leaves:
            out.append((leaves[parts[2]], param))
    return out


# ---------------------------------------------------------------- LoRA extraction


def _split_fused_qkv(name: str, diff: torch.Tensor) -> List[Tuple[str, torch.Tensor]]:
    """ComfyUI's files fuse each block's q / k / v into ``proj.qkv`` = [q; k; v]. The LoRA names
    them apart, as trained MOSS-Image LoRAs do: ComfyUI's MOSS-Image loader slices them into the
    fused weight, and an unfused model (the diffusers shards) takes them as they are."""
    if name.startswith("layers.") and name.endswith(".proj.qkv"):
        base = name[: -len(".qkv")]
        return [(f"{base}.{part}", chunk) for part, chunk in zip(("q", "k", "v"), diff.chunk(3, dim=0), strict=False)]
    return [(name, diff)]


EXTRACT_PROFILE = ExtractProfile(
    name="moss_image",
    split=_split_fused_qkv,
    detect=lambda names: (
        any(n == "shared_block_modulation.weight" for n in names)
        and any(n.startswith("layers.0.proj.ffn_up_gate") for n in names)
    ),
)
