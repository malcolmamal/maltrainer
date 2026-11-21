"""
Loading ComfyUI's LTX-2.5 single-file DiT (``ltx-2.5-22b-dev-transformer-comfy-int8-convrot`` or ``-bf16``).

The int8 file stores every block Linear as int8 with a per-row scale and a Hadamard-rotated input (INT8 ConvRot). It
stays int8 on the GPU and runs through the kernels shared with FLUX.3 and MiniMax H3, so the LoRA learns against the
weights ComfyUI runs. The distilled files are refused: LTX says to train the dev model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from safetensors.torch import load_file

from maltrainer.models.fibo2.loading import strip_container_prefix
from maltrainer.models.flux3.loading import is_int8_convrot
from maltrainer.models.ltx25.config import config_from_state
from maltrainer.models.ltx25.model import LTX25Transformer
from maltrainer.models.minimax_h3.int8_convrot import (
    enable_int8_convrot,
    load_comfy_int8_convrot_state_dict,
    prepare_int8_convrot_modules,
)

logger = logging.getLogger(__name__)

AUDIO_MARKERS = ("audio", "av_ca_", "a2v_ca")  # names of the audio stream's and the cross-modal modules
TEXT_PROJECTION = "text_embedding_projection."


def read_state(path: Path) -> tuple[dict[str, torch.Tensor], int, dict[str, str]]:
    """The file's tensors (int8 weights keep their scales), the number of int8 layers and the metadata."""
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if is_int8_convrot(path):
        state, quantized = load_comfy_int8_convrot_state_dict(str(path), device=torch.device("cpu"))
    else:
        state, quantized = load_file(str(path)), 0
    return state, quantized, metadata


def drop_audio(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Without the audio stream the model has no audio modules, connector, adaLN or cross-modal layers."""
    return {k: v for k, v in state.items() if not any(marker in k for marker in AUDIO_MARKERS)}


def load_ltx25_transformer(
    path: str | Path,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    with_audio: bool = True,
    grad_precision: str = "bf16",
) -> tuple[LTX25Transformer, dict[str, Any]]:
    """The frozen DiT on ``device``, plus a small summary for the checkpoint metadata."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LTX-2.5 transformer not found at {path}")
    if "distilled" in path.name:
        raise ValueError(
            f"{path.name} is the distilled LTX-2.5 model; train the dev model (ltx-2.5-22b-dev-transformer-*)"
        )
    logger.info("Loading LTX-2.5 DiT from %s (%s)...", path, "with audio" if with_audio else "video only")
    raw, quantized, metadata = read_state(path)
    state = {k: v for k, v in strip_container_prefix(raw).items() if not k.startswith(TEXT_PROJECTION)}
    config = config_from_state(state, metadata)
    if config.caption_projection:
        raise ValueError(
            f"{path.name} carries a caption projection, which only LTX-2.0 checkpoints do; use LTX-2.3 or newer"
        )
    if "audio_patchify_proj.weight" not in state and with_audio:
        raise ValueError(f"{path.name} has no audio stream; set the model to video only")
    if not with_audio:
        state = drop_audio(state)
    with init_empty_weights(include_buffers=False):
        model = LTX25Transformer(config, with_audio)
    if quantized:
        quantized = prepare_int8_convrot_modules(model, state)
    model.load_state_dict(state, strict=True, assign=True)
    if quantized:
        enable_int8_convrot(model, grad_precision=grad_precision)
    for parameter in model.parameters():
        if parameter.is_floating_point():
            parameter.data = parameter.data.to(dtype)
    model.requires_grad_(False)
    model.to(device=device)
    model.compute_dtype = dtype
    model.eval()
    info = {
        "file": path.name,
        "format": "int8_convrot"
        if quantized
        else str(state["transformer_blocks.0.attn1.to_q.weight"].dtype).replace("torch.", ""),
        "int8_layers": quantized,
        "layers": config.num_layers,
        "audio": with_audio,
    }
    logger.info("LTX-2.5 DiT ready: %s", info)
    return model, info
