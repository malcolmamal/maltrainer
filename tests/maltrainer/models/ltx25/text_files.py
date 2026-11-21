"""A tiny Gemma 4 text encoder file in ComfyUI's layout: ``model.*`` weights, projection, tokenizer and config."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

from tests.maltrainer.models.flux3.checkpoints import GROUP_SIZE, marker

VIDEO_DIM, AUDIO_DIM = 24, 12
GEMMA: dict[str, Any] = {
    "vocab_size": 50,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 16,
    "global_head_dim": 32,
    "num_global_key_value_heads": 1,
    "attention_k_eq_v": True,
    "sliding_window": 8,
    "hidden_size_per_layer_input": 0,
    "num_kv_shared_layers": 0,
}
WORDS = ["a", "red", "fox", "runs", "the", "sea"]


def tokenizer_blob() -> torch.Tensor:
    """A word-level tokenizer (ids 3.. for WORDS, 0 for anything else) as the uint8 tensor the file embeds."""
    vocab = {"<unk>": 0, **{word: 3 + i for i, word in enumerate(WORDS)}}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return torch.tensor(list(tokenizer.to_str().encode("utf-8")), dtype=torch.uint8)


def gemma_state(seed: int = 0) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    model = Gemma4TextModel(Gemma4TextConfig(**GEMMA))
    generator = torch.Generator().manual_seed(seed)
    return {
        f"model.{k}": (torch.randn(v.shape, generator=generator) * 0.1).to(v.dtype)
        for k, v in model.state_dict().items()
    }


def projection_state(seed: int = 1) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    width = GEMMA["hidden_size"] * (GEMMA["num_hidden_layers"] + 1)
    shapes = {"video": VIDEO_DIM, "audio": AUDIO_DIM}
    out = {}
    for name, size in shapes.items():
        out[f"text_embedding_projection.{name}_aggregate_embed.weight"] = (
            torch.randn(size, width, generator=generator) * 0.05
        )
        out[f"text_embedding_projection.{name}_aggregate_embed.bias"] = torch.randn(size, generator=generator) * 0.05
    return out


def quantise(state: dict[str, torch.Tensor], names: list[str]) -> dict[str, torch.Tensor]:
    """The modules ``names`` (weight keys without ``.weight``) as int8 with a row scale and a ConvRot marker."""
    generator = torch.Generator().manual_seed(9)
    out = dict(state)
    for base in names:
        rows, cols = out[f"{base}.weight"].shape
        out[f"{base}.weight"] = torch.randint(-100, 100, (rows, cols), generator=generator, dtype=torch.int8)
        out[f"{base}.weight_scale"] = torch.full((rows,), 0.001)
        out[f"{base}.comfy_quant"] = marker(format="int8_tensorwise", convrot=True, convrot_groupsize=GROUP_SIZE)
    return out


def write_text_encoder(
    path: Path,
    with_config: bool = True,
    tokenizer: bool = True,
    int8: bool = False,
    extra: dict[str, torch.Tensor] | None = None,
) -> Path:
    state = {**gemma_state(), **projection_state(), **(extra or {})}
    if tokenizer:
        state["tokenizer_json"] = tokenizer_blob()
    if int8:
        linears = [
            f"model.layers.{i}.{part}"
            for i in range(GEMMA["num_hidden_layers"])
            for part in ("self_attn.q_proj", "mlp.down_proj")
        ]
        state = quantise(state, [*linears, "model.embed_tokens"])
    metadata = {"gemma_config": json.dumps({"text_config": {**GEMMA, "dtype": "bfloat16"}})} if with_config else {}
    save_file({k: v.contiguous() for k, v in state.items()}, str(path), metadata=metadata)
    return path
