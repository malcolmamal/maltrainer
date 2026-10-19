from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from maltrainer.models.minimax_h3.text_encoder import (
    QWEN3VL_BOS_TOKEN_ID,
    MiniMaxH3TextEncoder,
)

CPU = torch.device("cpu")
HIDDEN = 6


class FakeTokenizer:
    """Whitespace words become ids 10, 11, ...; shorter rows are right-padded with mask 0."""

    def __init__(self, ids: list[list[int]] | None = None) -> None:
        self.rows = ids
        self.calls: list[dict] = []
        self.padding_side = "left"

    def __call__(self, prompts: list[str], **kwargs) -> SimpleNamespace:
        self.calls.append({"prompts": prompts, "padding_side": self.padding_side, **kwargs})
        rows = self.rows if self.rows is not None else [[10 + i for i, _ in enumerate(p.split())] for p in prompts]
        width = max(len(r) for r in rows)
        ids = torch.tensor([r + [0] * (width - len(r)) for r in rows], dtype=torch.long).reshape(len(rows), width)
        mask = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows], dtype=torch.long).reshape(
            len(rows), width
        )
        return SimpleNamespace(input_ids=ids, attention_mask=mask)


class FakeLanguageModel(nn.Module):
    """Hidden state = the token id broadcast over HIDDEN features, so outputs are checkable by eye."""

    def __init__(self, tuple_output: bool = False) -> None:
        super().__init__()
        self.tuple_output = tuple_output
        self.seen: list[dict] = []

    def forward(self, input_ids, attention_mask=None, use_cache=None):
        self.seen.append({"ids": input_ids.clone(), "mask": attention_mask.clone(), "use_cache": use_cache})
        hidden = input_ids.float()[..., None].expand(-1, -1, HIDDEN).clone()
        return (hidden,) if self.tuple_output else SimpleNamespace(last_hidden_state=hidden)


def _encoder(tokenizer: FakeTokenizer | None = None, **kwargs) -> MiniMaxH3TextEncoder:
    holder = nn.Module()
    holder.language_model = FakeLanguageModel(**kwargs)
    return MiniMaxH3TextEncoder(holder, tokenizer or FakeTokenizer())


def test_the_wrapper_is_frozen_and_in_eval_mode() -> None:
    encoder = _encoder()

    assert not encoder.training and not encoder.model.training


def test_prompts_are_tokenized_raw_without_special_tokens_right_padded_and_truncated() -> None:
    tokenizer = FakeTokenizer()
    encoder = _encoder(tokenizer)

    encoder.encode_prompts(["a b c"], CPU, max_length=77)

    call = tokenizer.calls[0]
    assert call["add_special_tokens"] is False and call["padding"] is True and call["truncation"] is True
    assert call["max_length"] == 77 and call["padding_side"] == "right" and call["return_tensors"] == "pt"


def test_the_hidden_states_are_the_language_model_output_at_the_prompt_length_in_the_requested_dtype() -> None:
    encoder = _encoder()

    out = encoder.encode_prompts(["a b c", "d e f"], CPU, dtype=torch.float16)

    assert out.shape == (2, 3, HIDDEN) and out.dtype == torch.float16
    assert out[0, :, 0].tolist() == [10.0, 11.0, 12.0]
    seen = encoder.model.language_model.seen[0]
    assert seen["use_cache"] is False and seen["mask"].tolist() == [[1, 1, 1], [1, 1, 1]]


def test_a_model_without_a_language_model_attribute_is_called_directly_and_tuple_outputs_are_accepted() -> None:
    model = FakeLanguageModel(tuple_output=True)
    encoder = MiniMaxH3TextEncoder(model, FakeTokenizer())

    out = encoder.encode_prompts(["a b"], CPU, dtype=torch.float32)

    assert out[0, :, 0].tolist() == [10.0, 11.0]


def test_a_repeated_prompt_is_served_from_the_cache_without_running_the_model_again() -> None:
    tokenizer = FakeTokenizer()
    encoder = _encoder(tokenizer)
    first = encoder.encode_prompts(["a b c"], CPU)

    second = encoder.encode_prompts(["a b c"], CPU)

    assert len(tokenizer.calls) == 1 and len(encoder.model.language_model.seen) == 1
    assert torch.equal(first, second)


def test_the_cache_is_keyed_by_dtype_and_by_the_whole_prompt_list() -> None:
    encoder = _encoder()
    encoder.encode_prompts(["a b c"], CPU, dtype=torch.float32)

    other_dtype = encoder.encode_prompts(["a b c"], CPU, dtype=torch.float16)
    other_prompts = encoder.encode_prompts(["a b c", "d e f"], CPU, dtype=torch.float32)

    assert other_dtype.dtype == torch.float16 and other_prompts.shape[0] == 2
    assert len(encoder.model.language_model.seen) == 3


def test_an_empty_prompt_is_represented_by_the_qwen_bos_token() -> None:
    encoder = _encoder(FakeTokenizer(ids=[[]]))

    out = encoder.encode_prompts([""], CPU, dtype=torch.float32)

    seen = encoder.model.language_model.seen[0]
    assert seen["ids"].tolist() == [[QWEN3VL_BOS_TOKEN_ID]] and seen["mask"].tolist() == [[1]]
    assert out.shape == (1, 1, HIDDEN) and out[0, 0, 0].item() == QWEN3VL_BOS_TOKEN_ID


def test_a_batch_of_empty_prompts_becomes_one_bos_token_per_row() -> None:
    tokenizer = FakeTokenizer(ids=[[], []])
    encoder = _encoder(tokenizer)

    out = encoder.encode_prompts(["", ""], CPU, dtype=torch.float32)

    assert out.shape == (2, 1, HIDDEN)
    assert out[:, 0, 0].tolist() == [float(QWEN3VL_BOS_TOKEN_ID)] * 2


def test_an_all_padding_row_among_real_ones_is_repaired_then_rejected_as_unequal_length() -> None:
    class Ragged(FakeTokenizer):
        def __call__(self, *_args, **_kwargs):
            return SimpleNamespace(
                input_ids=torch.tensor([[5, 6], [0, 0]]), attention_mask=torch.tensor([[1, 1], [0, 0]])
            )

    encoder = _encoder(Ragged())

    with pytest.raises(ValueError, match="equal token length"):
        encoder.encode_prompts(["x y", ""], CPU)

    assert encoder.model.language_model.seen[0]["ids"].tolist() == [[5, 6], [QWEN3VL_BOS_TOKEN_ID, 0]]
    assert encoder._prompt_cache == {}


def test_prompts_of_different_token_lengths_are_refused_and_never_cached() -> None:
    encoder = _encoder()

    with pytest.raises(ValueError, match="equal token length in one call"):
        encoder.encode_prompts(["a b c", "d"], CPU)

    assert encoder._prompt_cache == {}
