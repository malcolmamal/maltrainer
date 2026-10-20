"""Z-Image's prompt encoding: each caption in the tokenizer's chat template, through the text encoder's base
model, the second-to-last hidden state as the transformer's caption features."""

from __future__ import annotations

import inspect
from typing import Any

import torch


def chat_prompts(tokenizer: Any, captions: list[str], thinking: bool) -> list[str]:
    """Each caption as a user turn in the tokenizer's chat template (the raw caption without one). ``thinking``
    passes ``enable_thinking=True`` where the template takes it (the reference does for the conditional prompt)."""
    if not hasattr(tokenizer, "apply_chat_template"):
        return list(captions)
    parameters = inspect.signature(tokenizer.apply_chat_template).parameters
    extra = (
        {"enable_thinking": True} if thinking and ("enable_thinking" in parameters or "kwargs" in parameters) else {}
    )
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": c}], tokenize=False, add_generation_prompt=True, **extra
        )
        for c in captions
    ]


def tokens(tokenizer: Any, prompts: list[str], device: torch.device) -> Any:
    # "longest" padding: a batch of one has no padding tokens, as the model does not take an attention mask
    return tokenizer(
        prompts, padding="longest", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt"
    ).to(device)


def caption_features(text_encoder: Any, batch_tokens: Any) -> torch.Tensor:
    """The second-to-last hidden state of the encoder's base model (no grad)."""
    with torch.no_grad():
        base_text_encoder = getattr(text_encoder, "model", text_encoder)
        out = base_text_encoder(
            batch_tokens.input_ids, attention_mask=batch_tokens.attention_mask, output_hidden_states=True
        )
        features: torch.Tensor = out.hidden_states[-2]
        return features
