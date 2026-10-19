from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from maltrainer.models.krea2 import text_encoder as te
from maltrainer.models.krea2.text_encoder import (
    IM_START,
    NEWLINE,
    TAP_LAYERS,
    TEMPLATE,
    USER,
    Krea2TextEncoder,
    _prefix_end,
    _text_key,
)

CPU = torch.device("cpu")
IM_END = 151645
SPECIAL = {"<|im_start|>": IM_START, "<|im_end|>": IM_END, "user": USER, "\n": NEWLINE}


class AddOne(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + 1.0


class StubModel(nn.Module):
    """35 layers that each add one to a two-wide copy of the token id, so every tap is known by hand."""

    def __init__(self, as_tuple: bool = False) -> None:
        super().__init__()
        self.layers = nn.ModuleList([AddOne() for _ in range(35)])
        self.as_tuple = as_tuple
        self.calls: list = []

    def forward(self, **kwargs: torch.Tensor) -> object:
        self.calls.append(kwargs)
        h = kwargs["input_ids"].float()[..., None].repeat(1, 1, 2)
        for layer in self.layers:
            h = layer(h)
        return (h,) if self.as_tuple else SimpleNamespace(last_hidden_state=h)


class FakeTokenizer:
    """Chat-template tokens get their real ids, every other character its code point."""

    def __init__(self) -> None:
        self.texts: list = []
        self.decoded: list = []

    def __call__(self, text: str, add_special_tokens: bool) -> SimpleNamespace:
        assert add_special_tokens is False
        self.texts.append(text)
        pieces = re.split(r"(<\|im_start\|>|<\|im_end\|>|\n|user)", text)
        ids = [SPECIAL[p] if p in SPECIAL else ord(c) for p in pieces for c in ([p] if p in SPECIAL else p)]
        return SimpleNamespace(input_ids=ids)

    def decode(self, ids: list) -> str:
        self.decoded.append(ids)
        return "".join(chr(i) for i in ids)


def _encoder(**kwargs: object) -> Krea2TextEncoder:
    return Krea2TextEncoder(StubModel(**kwargs), FakeTokenizer())


@pytest.mark.parametrize(
    ("ids", "expected"),
    [
        ([IM_START, 1, 2, IM_START, USER, NEWLINE, 7, 8], 6),
        ([IM_START, 1, IM_START, USER, NEWLINE, 7, IM_START, 9], 5),
        ([IM_START, USER, NEWLINE, 7], 3),
        ([IM_START, 1, IM_START, 5, 6, 7], 2),
        ([IM_START, 1, IM_START, USER, NEWLINE], 2),
        ([1, 2, 3], 0),
        ([], 0),
    ],
)
def test_the_prefix_ends_after_the_second_turn_header_when_it_is_followed_by_user_and_a_newline(
    ids: list, expected: int
) -> None:
    assert _prefix_end(ids) == expected


def test_the_prefix_up_to_the_users_text_is_dropped_but_the_assistant_suffix_stays() -> None:
    encoder = _encoder()

    out = encoder.encode("ab", CPU, dtype=torch.float32)

    suffix = 1 + 1 + 1 + len("assistant") + 1
    assert out.shape == (2 + suffix, 12, 2)
    assert out[0, 0, 0].item() == ord("a") + TAP_LAYERS[0]
    assert out[2, 0, 0].item() == IM_END + TAP_LAYERS[0]


def test_the_prompt_goes_through_the_image_description_system_template() -> None:
    encoder = _encoder()

    encoder.encode("a red fox", CPU)

    assert encoder.tokenizer.texts[0] == "a red fox"
    text = encoder.tokenizer.texts[-1]
    assert text == TEMPLATE.format("a red fox")
    assert text.startswith("<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture")
    assert text.endswith("a red fox<|im_end|>\n<|im_start|>assistant\n")


def test_twelve_hidden_states_are_stacked_per_token_with_the_last_from_the_model_output() -> None:
    out = _encoder().encode("a", CPU, dtype=torch.float32)

    base = float(ord("a"))
    assert out[0, :, 0].tolist() == [base + k for k in TAP_LAYERS[:-1]] + [base + 35]
    assert [2 + 3 * i for i in range(12)] == TAP_LAYERS


def test_a_model_that_returns_a_tuple_is_read_the_same_way() -> None:
    plain = _encoder().encode("ab", CPU)

    assert torch.equal(_encoder(as_tuple=True).encode("ab", CPU), plain)


def test_the_model_sees_every_token_unmasked_and_the_output_defaults_to_bf16() -> None:
    encoder = _encoder()

    out = encoder.encode("ab", CPU)

    call = encoder.model.calls[0]
    assert call["attention_mask"].all() and call["use_cache"] is False
    assert call["input_ids"].shape[0] == 1
    assert out.dtype == torch.bfloat16


def test_a_prompt_over_the_token_cap_is_cut_before_templating() -> None:
    encoder = _encoder()

    encoder.encode("abcdef", CPU, max_prompt_tokens=3)

    assert encoder.tokenizer.decoded == [[ord("a"), ord("b"), ord("c")]]
    assert encoder.tokenizer.texts[-1] == TEMPLATE.format("abc")


def test_a_prompt_within_the_cap_is_not_decoded_again() -> None:
    encoder = _encoder()

    encoder.encode("abc", CPU, max_prompt_tokens=3)

    assert encoder.tokenizer.decoded == []
    assert encoder.tokenizer.texts[-1] == TEMPLATE.format("abc")


def test_the_capture_hook_reads_the_hidden_states_from_kwargs_when_called_without_positionals() -> None:
    encoder = _encoder()
    hidden = torch.ones(1, 2, 2, requires_grad=True)

    encoder._capture(2)(None, (), {"hidden_states": hidden})

    assert not encoder._captured[2].requires_grad


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("text_encoders.qwen3vl_4b.transformer.layers.0.mlp.up_proj.weight", "layers.0.mlp.up_proj.weight"),
        ("model.language_model.layers.1.input_layernorm.weight", "layers.1.input_layernorm.weight"),
        ("language_model.model.embed_tokens.weight", "embed_tokens.weight"),
        ("language_model.layers.2.self_attn.q_norm.weight", "layers.2.self_attn.q_norm.weight"),
        ("model.norm.weight", "norm.weight"),
        ("layers.3.mlp.down_proj.weight", "layers.3.mlp.down_proj.weight"),
        ("model.visual.blocks.0.attn.weight", None),
        ("visual.blocks.0.attn.weight", None),
        ("lm_head.weight", None),
    ],
)
def test_checkpoint_key_spellings_map_to_the_text_model_or_are_dropped(key: str, expected: str | None) -> None:
    assert _text_key(key) == expected


def test_the_template_token_ids_match_the_vendored_qwen3_vl_tokenizer() -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(te.TOKENIZER_ASSETS))

    ids = tokenizer(TEMPLATE.format("hi"), add_special_tokens=False).input_ids

    start = _prefix_end(ids)
    assert ids[0] == IM_START == tokenizer.convert_tokens_to_ids("<|im_start|>")
    assert tokenizer.decode(ids[start : start + 1]) == "hi"
    assert tokenizer.decode(ids[start - 3 : start]) == "<|im_start|>user\n"
