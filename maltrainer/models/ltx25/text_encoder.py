"""
LTX-2.5 text conditioning: Gemma 4 12B hidden states, projected for the video and audio streams.

Mirrors comfy/text_encoders/lt.py (``LTXAVTEModel``, dual linear projection) and gemma4.py:
- the prompt is tokenised as it is, behind a BOS token (id 2), with no chat template and no padding
- every hidden state of the decoder (the embeddings, the input of each layer and the final normalised output: 49 for
  48 layers) is stacked, RMS-normalised per layer and projected: (tokens, video 4096 + audio 2048)
- the DiT's own connectors take it from there, so these features are what gets cached

The file (``gemma4-12b-with-proj-ltx-2.5-*``) holds the Gemma weights under ``model.`` and the projection under
``text_embedding_projection.``; it embeds the tokenizer and the Gemma config. The module tree uses the file's names.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn

from maltrainer.models.flux3.loading import is_int8_convrot
from maltrainer.models.ltx25.text_projection import DualProjection, stack_hidden_states
from maltrainer.models.minimax_h3.int8_convrot import (
    enable_int8_convrot,
    load_comfy_int8_convrot_state_dict,
    prepare_int8_convrot_modules,
)

logger = logging.getLogger(__name__)

BOS = 2
MAX_TOKENS = 1024
# Gemma 4 12B "unified" (comfy/text_encoders/gemma4.py Gemma4_12B_Config), used when the file carries no gemma_config
GEMMA_DEFAULT: dict[str, Any] = {
    "vocab_size": 262144,
    "hidden_size": 3840,
    "intermediate_size": 15360,
    "num_hidden_layers": 48,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 256,
    "global_head_dim": 512,
    "num_global_key_value_heads": 1,
    "attention_k_eq_v": True,
    "sliding_window": 1024,
    "hidden_size_per_layer_input": 0,
    "num_kv_shared_layers": 0,
}


class LTX25TextStack(nn.Module):
    """Gemma's text decoder and the projection, named as in the file."""

    def __init__(self, model: Any, projection: DualProjection):
        super().__init__()
        self.model = model
        self.text_embedding_projection = projection


class ScaledEmbedding(nn.Module):
    """An embedding lookup times Gemma's sqrt(hidden) scale, for the INT8 table (which has no scale of its own)."""

    def __init__(self, embedding: nn.Module, scale: float):
        super().__init__()
        self.embedding, self.scale = embedding, scale

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.embedding(input_ids) * self.scale)


class LTX25TextEncoder:
    def __init__(self, stack: LTX25TextStack, tokenizer: Any, max_tokens: int = MAX_TOKENS):
        self.stack, self.tokenizer, self.max_tokens = stack, tokenizer, max_tokens

    def token_ids(self, prompt: str) -> list[int]:
        return [BOS, *self.tokenizer.encode(prompt.strip(), add_special_tokens=False).ids][: self.max_tokens]

    @torch.no_grad()
    def encode(self, prompt: str, device: torch.device, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """One prompt -> (tokens, video_dim + audio_dim) features."""
        ids = torch.tensor([self.token_ids(prompt)], device=device)
        out = self.stack.model(input_ids=ids, output_hidden_states=True, use_cache=False)
        return cast(
            torch.Tensor, self.stack.text_embedding_projection(stack_hidden_states(out.hidden_states))[0].to(dtype)
        )


def gemma_config(path: Path) -> dict[str, Any]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if "gemma_config" not in metadata:
        return dict(GEMMA_DEFAULT)
    text = json.loads(metadata["gemma_config"]).get("text_config", {})
    return {key: value for key, value in text.items() if key != "dtype"}


def text_keys(raw: dict[str, Any]) -> dict[str, Any]:
    """The text decoder's tensors as ``model.<name>`` and the projection's, without the vision and audio towers."""
    out = {}
    for key, value in raw.items():
        if key.startswith("text_embedding_projection."):
            out[key] = value
        elif key.startswith("model.language_model."):
            out["model." + key[len("model.language_model.") :]] = value
        elif key.startswith("model.") and not key.startswith(("model.vision", "model.audio", "model.embed_vision")):
            out[key] = value
    return out


def build_stack(config: dict[str, Any], state: dict[str, Any]) -> LTX25TextStack:
    from transformers import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

    text_config = Gemma4TextConfig(**config)
    setattr(text_config, "_attn_implementation", "sdpa")  # noqa: B010 (transformers reads the private field)
    video, audio = (state[f"text_embedding_projection.{name}_aggregate_embed.weight"] for name in ("video", "audio"))
    with init_empty_weights(include_buffers=False):
        projection = DualProjection(
            video.shape[1],
            video.shape[0],
            audio.shape[0],
            "text_embedding_projection.video_aggregate_embed.bias" in state,
            "text_embedding_projection.audio_aggregate_embed.bias" in state,
        )
        return LTX25TextStack(Gemma4TextModel(text_config), projection)


def load_ltx25_text_encoder(
    path: str | Path, device: torch.device, dtype: torch.dtype = torch.bfloat16, max_tokens: int = MAX_TOKENS
) -> LTX25TextEncoder:
    from tokenizers import Tokenizer

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LTX-2.5 text encoder not found at {path}")
    quantized = is_int8_convrot(path)
    raw = (
        load_comfy_int8_convrot_state_dict(str(path), device=torch.device("cpu"))[0]
        if quantized
        else load_file(str(path))
    )
    if "tokenizer_json" not in raw:
        raise ValueError(f"{path.name} carries no tokenizer; LTX-2.5 needs gemma4-12b-with-proj-ltx-2.5-*.safetensors")
    tokenizer = Tokenizer.from_str(bytes(raw["tokenizer_json"].tolist()).decode("utf-8"))
    state = text_keys(raw)
    stack = build_stack(gemma_config(path), state)
    if quantized:
        install_int8_embedding(stack, state)
        prepare_int8_convrot_modules(stack, state)
    info = stack.load_state_dict(state, strict=False, assign=True)
    missing = [key for key in info.missing_keys if not (quantized and key.startswith("model.embed_tokens."))]
    if missing or info.unexpected_keys:
        raise ValueError(
            f"{path.name} does not match the Gemma 4 12B text encoder "
            f"(missing {missing[:5]}, unexpected {info.unexpected_keys[:5]})"
        )
    if quantized:
        enable_int8_convrot(stack)
    for parameter in stack.parameters():
        if parameter.is_floating_point():
            parameter.data = parameter.data.to(dtype)
    stack.to(device=device).eval().requires_grad_(False)
    logger.info(
        "LTX-2.5 text encoder: Gemma 4, %d layers, %s",
        stack.model.config.num_hidden_layers,
        "INT8 ConvRot" if quantized else "plain",
    )
    return LTX25TextEncoder(stack, tokenizer, max_tokens)


def install_int8_embedding(stack: LTX25TextStack, state: dict[str, Any]) -> None:
    """Swap the token table for the INT8 ConvRot embedding (times sqrt(hidden)) when the file stores it quantised."""
    from maltrainer.models.qwen_image_2_1.int8_embedding import Int8Embedding

    prefix = "model.embed_tokens."
    if f"{prefix}int8_convrot_groupsize" not in state:
        return
    config = stack.model.config
    table = Int8Embedding(
        config.vocab_size, config.hidden_size, group_size=int(state.pop(f"{prefix}int8_convrot_groupsize").item())
    )
    table.weight.data.copy_(state.pop(f"{prefix}weight"))
    table.get_buffer("scale_weight").data.copy_(state.pop(f"{prefix}scale_weight"))
    stack.model.embed_tokens = ScaledEmbedding(table, config.hidden_size**0.5)
