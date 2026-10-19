from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import List

import pytest
import torch
from torch import nn

from maltrainer.models.moss_image.text_encoder import (
    MAX_SEQUENCE_LENGTH,
    TEMPLATE,
    TOKENIZER_ASSETS,
    MOSSImageTextEncoder,
)

CPU = torch.device("cpu")


class FakeTokenizer:
    """One token per character, or ``count`` tokens when given; records what it was asked to encode."""

    def __init__(self, count: int | None = None) -> None:
        self.count = count
        self.seen: List[tuple] = []

    def __call__(self, text: str, add_special_tokens: bool = True):
        self.seen.append((text, add_special_tokens))
        return SimpleNamespace(input_ids=list(range(self.count if self.count is not None else len(text))))


class IdsToFeatures(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: List[torch.Tensor] = []

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.calls.append(input_ids)
        return input_ids.float()[..., None].expand(*input_ids.shape, 4) * torch.ones(4, requires_grad=True)


def test_the_template_is_the_chat_prompt_with_an_open_think_block() -> None:
    assert TEMPLATE.format("a cat") == "<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n<think>\n"
    assert MAX_SEQUENCE_LENGTH == 4096


def test_a_prompt_is_stripped_and_wrapped_without_the_tokenizers_own_special_tokens() -> None:
    tokenizer = FakeTokenizer()
    encoder = MOSSImageTextEncoder(IdsToFeatures(), tokenizer)

    ids = encoder.token_ids("  a cat \n")

    assert tokenizer.seen == [(TEMPLATE.format("a cat"), False)]
    assert len(ids) == len(TEMPLATE.format("a cat"))


def test_a_prompt_at_the_limit_is_kept_whole_without_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    encoder = MOSSImageTextEncoder(IdsToFeatures(), FakeTokenizer(MAX_SEQUENCE_LENGTH))

    with caplog.at_level(logging.WARNING):
        ids = encoder.token_ids("x")

    assert len(ids) == MAX_SEQUENCE_LENGTH
    assert caplog.text == ""


def test_a_longer_prompt_is_cut_to_its_first_4096_tokens_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    encoder = MOSSImageTextEncoder(IdsToFeatures(), FakeTokenizer(MAX_SEQUENCE_LENGTH + 10))

    with caplog.at_level(logging.WARNING):
        ids = encoder.token_ids("x")

    assert ids == list(range(MAX_SEQUENCE_LENGTH))
    assert f"Prompt is {MAX_SEQUENCE_LENGTH + 10} tokens; truncated to {MAX_SEQUENCE_LENGTH}" in caplog.text


def test_encode_runs_one_prompt_at_its_true_length_and_returns_it_in_the_requested_dtype() -> None:
    model = IdsToFeatures()
    encoder = MOSSImageTextEncoder(model, FakeTokenizer(5))

    out = encoder.encode("anything", CPU, torch.bfloat16)

    assert model.calls[0].tolist() == [[0, 1, 2, 3, 4]]
    assert out.shape == (5, 4)
    assert out.dtype == torch.bfloat16
    assert out[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert not out.requires_grad


def test_the_shipped_tokenizer_maps_the_template_to_its_special_tokens() -> None:
    from transformers import Qwen2Tokenizer

    encoder = MOSSImageTextEncoder(IdsToFeatures(), Qwen2Tokenizer.from_pretrained(str(TOKENIZER_ASSETS)))

    ids = encoder.token_ids(" a cat ")

    im_start, im_end, think = 248045, 248046, 248068
    assert ids == [im_start, 846, 198, 64, 7993, im_end, 198, im_start, 74455, 198, think, 198]
