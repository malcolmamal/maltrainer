"""
MOSS-Image text conditioning: Qwen3.5-9B's last hidden state.

Mirrors comfy/text_encoders/moss_image.py (MOSSImageTokenizer / MOSSImageTEModel) and
comfy/text_encoders/qwen35.py:
- the prompt is stripped and wrapped in the chat template with thinking enabled, which leaves an
  open think block: ``<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n``
- no padding and no truncation of the window: one prompt is encoded at its true length, and the
  DiT's RoPE follows that length. Capped at 4096 tokens, the final pretraining stage's limit.
- the output is every token's final hidden state after the model's last RMSNorm, (L, 4096)

Qwen3.5 is a hybrid: of its 32 decoder layers, every 4th is gated full attention (QK-norm,
partial RoPE on 64 of 256 head dims, a sigmoid output gate); the rest are Gated DeltaNet linear
attention (a causal depthwise conv, then the chunked gated delta rule). The port below is the
text path of ComfyUI's, without the KV cache, the vision tower, the MTP head and lm_head.

Weights come from ComfyUI's qwen3.5_9b_int8_convrot.safetensors (its Linears run through the
INT8 ConvRot kernels) or a plain bf16 file. They stay in their stored dtype, but activations run
in fp32 and each weight is upcast per op: ComfyUI's SDClipModel embeds tokens with out_dtype
float32 and runs the encoder with dtype=float32, which keeps its bf16 drift ~3x below an
all-bf16 run.

The Qwen3.5 text model itself lives in ``qwen35_config``, ``qwen35_layers``, ``qwen35_delta`` and
``qwen35_model``; every name is re-exported here, next to the encoder and the checkpoint loader.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import MethodType
from typing import Any, cast

import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from safetensors.torch import load_file
from torch import nn

from maltrainer.models.flux3.loading import is_int8_convrot
from maltrainer.models.minimax_h3.int8_convrot import (
    enable_int8_convrot,
    load_comfy_int8_convrot_state_dict,
    prepare_int8_convrot_modules,
)
from maltrainer.models.moss_image.qwen35_config import Qwen35TextConfig
from maltrainer.models.moss_image.qwen35_delta import GatedDeltaNet, chunk_gated_delta_rule
from maltrainer.models.moss_image.qwen35_layers import MLP, GatedAttention, RMSNorm, RMSNormGated, rotate_half
from maltrainer.models.moss_image.qwen35_model import DecoderLayer, Qwen35TextModel
from maltrainer.models.moss_image.text_encoder_keys import KEY_PREFIXES, SKIPPED, _text_key
from maltrainer.models.qwen_image_2_1.text_encoder import Int8Embedding

logger = logging.getLogger(__name__)

__all__ = [
    "KEY_PREFIXES",
    "MAX_SEQUENCE_LENGTH",
    "MLP",
    "SKIPPED",
    "TEMPLATE",
    "TOKENIZER_ASSETS",
    "DecoderLayer",
    "GatedAttention",
    "GatedDeltaNet",
    "MOSSImageTextEncoder",
    "Qwen35TextConfig",
    "Qwen35TextModel",
    "RMSNorm",
    "RMSNormGated",
    "_text_key",
    "chunk_gated_delta_rule",
    "load_moss_image_text_encoder",
    "rotate_half",
]

TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n"
MAX_SEQUENCE_LENGTH = 4096  # OpenMOSS's ComfyUI node and the report's final-stage cap
TOKENIZER_ASSETS = Path(__file__).resolve().parent / "assets" / "tokenizer"


class MOSSImageTextEncoder(nn.Module):
    def __init__(self, model: Qwen35TextModel, tokenizer: Any):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def token_ids(self, prompt: str) -> list[Any]:
        ids = self.tokenizer(TEMPLATE.format(prompt.strip()), add_special_tokens=False).input_ids
        if len(ids) > MAX_SEQUENCE_LENGTH:
            logger.warning("Prompt is %d tokens; truncated to %d.", len(ids), MAX_SEQUENCE_LENGTH)
        return cast(list[Any], ids[:MAX_SEQUENCE_LENGTH])

    @torch.no_grad()
    def encode(self, prompt: str, device: torch.device, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """One prompt -> (L, 4096) conditioning at its true length."""
        input_ids = torch.tensor([self.token_ids(prompt)], device=device)
        return cast(torch.Tensor, self.model(input_ids)[0].to(dtype))


def _upcast_linear_forward(module: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """A plain Linear in the input's dtype, as comfy.ops casts a bf16 weight up to fp32 activations."""
    bias = module.bias.to(x.dtype) if module.bias is not None else None
    return F.linear(x, module.weight.to(x.dtype), bias)


def _read_state(path: Path, quantized: bool) -> dict[str, torch.Tensor]:
    if quantized:
        raw, _ = load_comfy_int8_convrot_state_dict(str(path), device=torch.device("cpu"))
    else:
        raw = load_file(str(path))
    return {target: value for key, value in raw.items() if (target := _text_key(key)) is not None}


def _int8_embedding(config: Qwen35TextConfig, state: dict[str, torch.Tensor]) -> Int8Embedding:
    # The table is stored Hadamard-rotated; Int8Embedding un-rotates it on lookup.
    embed = Int8Embedding(
        config.vocab_size, config.hidden_size, group_size=int(state["embed_tokens.int8_convrot_groupsize"].item())
    )
    embed.weight.data.copy_(state.pop("embed_tokens.weight"))
    cast(Any, embed).scale_weight.data.copy_(state.pop("embed_tokens.scale_weight"))
    state.pop("embed_tokens.int8_convrot_groupsize")
    return embed


def _check_loaded(path: Path, model: Qwen35TextModel, info: Any) -> None:
    replaced_embed = isinstance(model.embed_tokens, Int8Embedding)
    missing = [key for key in info.missing_keys if not (replaced_embed and key.startswith("embed_tokens."))]
    if missing or info.unexpected_keys:
        raise ValueError(
            f"{path.name} does not look like a Qwen3.5 text encoder "
            f"(missing {missing[:5]}, unexpected {info.unexpected_keys[:5]})"
        )


def _finish_model(model: Qwen35TextModel, device: torch.device, dtype: torch.dtype, compute_dtype: torch.dtype) -> None:
    for module in model.modules():
        if isinstance(module, nn.Linear) and not hasattr(module, "scale_weight"):
            cast(Any, module).forward = MethodType(_upcast_linear_forward, module)
    for parameter in model.parameters():
        if parameter.is_floating_point():
            parameter.data = parameter.data.to(dtype)
    model.compute_dtype = compute_dtype
    model.to(device=device).eval().requires_grad_(False)


def load_moss_image_text_encoder(
    path: str | Path,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    tokenizer_path: str | Path | None = None,
    grad_precision: str = "bf16",
    compute_dtype: torch.dtype = torch.float32,
) -> MOSSImageTextEncoder:
    """``dtype`` is the storage dtype of the float weights; ``compute_dtype`` that of the activations."""
    from transformers import Qwen2Tokenizer

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Qwen3.5-9B text encoder not found at {path}")
    tokenizer = Qwen2Tokenizer.from_pretrained(str(Path(tokenizer_path) if tokenizer_path else TOKENIZER_ASSETS))

    quantized = is_int8_convrot(path)
    state = _read_state(path, quantized)
    config = Qwen35TextConfig.from_state_dict(state)

    with init_empty_weights(include_buffers=False):
        model = Qwen35TextModel(config)

    if quantized and "embed_tokens.int8_convrot_groupsize" in state:
        model.embed_tokens = _int8_embedding(config, state)
    if quantized:
        prepare_int8_convrot_modules(model, state)

    _check_loaded(path, model, model.load_state_dict(state, strict=False, assign=True))
    if quantized:
        enable_int8_convrot(model, grad_precision=grad_precision)

    _finish_model(model, device, dtype, compute_dtype)
    logger.info(
        "MOSS-Image text encoder: Qwen3.5, %d layers (%d full attention), hidden %d, %s",
        config.num_hidden_layers,
        len(config.full_attention_layers),
        config.hidden_size,
        "INT8 ConvRot" if quantized else "plain",
    )
    return MOSSImageTextEncoder(model, tokenizer)
