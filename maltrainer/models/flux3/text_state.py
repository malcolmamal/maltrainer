"""Loading steps of the FLUX.3 text encoder: the INT8 embedding table and the strict-enough state load."""

from __future__ import annotations

from typing import Any, cast

import torch

from maltrainer.models.qwen_image_2_1.text_encoder import Int8Embedding


def install_int8_embedding(model: Any, state: dict[str, Any], config: Any) -> None:
    # The table is stored Hadamard-rotated; Int8Embedding un-rotates it on lookup.
    embed = Int8Embedding(
        config.vocab_size, config.hidden_size, group_size=int(state["embed_tokens.int8_convrot_groupsize"].item())
    )
    embed.weight.data.copy_(state.pop("embed_tokens.weight"))
    cast("torch.Tensor", embed.scale_weight).data.copy_(state.pop("embed_tokens.scale_weight"))
    state.pop("embed_tokens.int8_convrot_groupsize")
    model.embed_tokens = embed


def load_text_state(model: Any, state: dict[str, Any], name: str) -> None:
    info = model.load_state_dict(state, strict=False, assign=True)
    replaced_embed = isinstance(model.embed_tokens, Int8Embedding)
    missing = [
        key
        for key in info.missing_keys
        if "rotary_emb" not in key and not (replaced_embed and key.startswith("embed_tokens."))
    ]
    if missing or info.unexpected_keys:
        raise ValueError(
            f"{name} does not look like a Qwen3-VL-4B text encoder "
            f"(missing {missing[:5]}, unexpected {info.unexpected_keys[:5]})"
        )
