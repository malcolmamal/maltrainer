from __future__ import annotations

import pytest
import torch
from torch import nn

from maltrainer.models.minimax_h3 import text_encoder as te
from tests.maltrainer.models.minimax_h3.h3_text import FakeLanguageModel, FakeTokenizer, encoder

CPU = torch.device("cpu")


def test_the_wrapper_freezes_and_evals_the_model() -> None:
    wrapper = nn.Module()
    wrapper.language_model = nn.Linear(1, 1)

    wrapped = te.MiniMaxH3TextEncoder(wrapper, FakeTokenizer())

    assert not wrapped.training
    assert not any(p.requires_grad for p in wrapped.parameters())


def test_prompts_are_tokenized_raw_right_padded_and_truncated_to_the_max_length() -> None:
    enc = encoder()

    enc.encode_prompts(["a bb"], CPU, max_length=7)

    call = enc.tokenizer.calls[0]
    assert call["add_special_tokens"] is False
    assert (call["padding"], call["truncation"], call["max_length"]) == (True, True, 7)
    assert call["padding_side"] == "right"


def test_the_hidden_states_are_the_language_models_last_hidden_state_cast_to_the_dtype() -> None:
    out = encoder().encode_prompts(["a bb"], CPU, dtype=torch.float64)

    assert out.shape == (1, 2, 4)
    assert out.dtype == torch.float64
    assert out[0, :, 0].tolist() == [11.0, 12.0]
    assert not out.requires_grad


def test_a_model_returning_a_tuple_is_read_by_its_first_element() -> None:
    out = encoder(FakeLanguageModel(as_tuple=True)).encode_prompts(["a bb"], CPU)

    assert out[0, :, 0].tolist() == [11.0, 12.0]


def test_a_model_without_a_language_model_attribute_is_called_directly() -> None:
    enc = te.MiniMaxH3TextEncoder(FakeLanguageModel(), FakeTokenizer())

    assert enc.encode_prompts(["a"], CPU).shape == (1, 1, 4)


def test_the_language_model_runs_without_a_cache_on_the_ids_and_mask() -> None:
    enc = encoder()

    enc.encode_prompts(["a bb ccc"], CPU)

    ids, mask, use_cache = enc.model.language_model.calls[0]
    assert ids.tolist() == [[11, 12, 13]]
    assert mask.tolist() == [[1, 1, 1]]
    assert use_cache is False


def test_hidden_states_beyond_the_prompt_length_are_dropped() -> None:
    out = encoder(FakeLanguageModel(extra=3)).encode_prompts(["a bb"], CPU)

    assert out.shape[1] == 2


def test_an_empty_prompt_is_represented_by_the_bos_token() -> None:
    enc = encoder()

    out = enc.encode_prompts([""], CPU, dtype=torch.float32)

    assert out.shape == (1, 1, 4)
    assert out[0, 0, 0].item() == te.QWEN3VL_BOS_TOKEN_ID


def test_a_row_with_no_tokens_gets_a_bos_when_the_batch_is_otherwise_one_token_wide() -> None:
    enc = encoder()

    out = enc.encode_prompts(["", "a"], CPU, dtype=torch.float32)

    assert out[:, 0, 0].tolist() == [float(te.QWEN3VL_BOS_TOKEN_ID), 11.0]
    assert enc.model.language_model.calls[0][1].tolist() == [[1], [1]]


def test_prompts_of_different_token_lengths_are_rejected_and_not_cached() -> None:
    enc = encoder()

    with pytest.raises(ValueError, match="equal token length"):
        enc.encode_prompts(["a", "a b"], CPU)

    assert enc._prompt_cache == {}


def test_a_repeated_prompt_is_served_from_the_cache_per_dtype() -> None:
    enc = encoder()

    first = enc.encode_prompts(["a bb"], CPU)
    again = enc.encode_prompts(["a bb"], CPU)
    other_dtype = enc.encode_prompts(["a bb"], CPU, dtype=torch.float32)

    assert len(enc.model.language_model.calls) == 2
    assert torch.equal(first, again)
    assert other_dtype.dtype == torch.float32
