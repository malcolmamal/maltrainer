"""Putting several samples into one batch."""

from __future__ import annotations

import torch


def batch_text_embeddings(embeds: list[torch.Tensor]) -> torch.Tensor:
    """``[1, tokens, dim]`` embeddings joined into one batch. The transformers take no text mask, so
    the prompts must have the same number of tokens, and a batch of different ones is refused."""
    if len({e.shape[1] for e in embeds}) > 1:
        raise ValueError(
            "Captions of different token lengths cannot be batched: use batch_size 1 "
            "with train.gradient_accumulation_steps"
        )
    return torch.cat(embeds, dim=0)
