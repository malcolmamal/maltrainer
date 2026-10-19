from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from maltrainer.models.flux3 import text_encoder as te
from maltrainer.models.flux3.text_encoder import Flux3TextEncoder, _text_key

HIDDEN = 2


class AddOne(nn.Module):
    def forward(self, hidden_states: torch.Tensor, **_: object) -> torch.Tensor:
        return hidden_states + 1.0


class FakeStack(nn.Module):
    """32 layers that each add one: the input of layer k is the embedding plus k."""

    def __init__(self, tuple_output: bool = False) -> None:
        super().__init__()
        self.layers = nn.ModuleList([AddOne() for _ in range(32)])
        self.tuple_output = tuple_output
        self.seen: dict = {}

    def forward(self, input_ids, attention_mask, position_ids, use_cache) -> object:
        self.seen = {"ids": input_ids, "mask": attention_mask, "pos": position_ids, "use_cache": use_cache}
        h = input_ids.float().unsqueeze(-1).expand(-1, -1, HIDDEN).clone()
        for layer in self.layers:
            h = layer(h)
        return (h,) if self.tuple_output else SimpleNamespace(last_hidden_state=h)


class FakeTokenizer:
    def __init__(self, length: int) -> None:
        self.length = length
        self.texts: list[str] = []

    def __call__(self, text: str, add_special_tokens: bool) -> SimpleNamespace:
        assert add_special_tokens is False
        self.texts.append(text)
        return SimpleNamespace(input_ids=list(range(100, 100 + self.length)))


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("text_encoders.qwen3vl_4b.transformer.layers.0.q.weight", "layers.0.q.weight"),
        ("model.language_model.layers.0.q.weight", "layers.0.q.weight"),
        ("language_model.model.embed_tokens.weight", "embed_tokens.weight"),
        ("language_model.norm.weight", "norm.weight"),
        ("model.layers.1.q.weight", "layers.1.q.weight"),
        ("layers.2.q.weight", "layers.2.q.weight"),
        ("model.visual.blocks.0.weight", None),
        ("visual.merger.weight", None),
        ("lm_head.weight", None),
    ],
)
def test_checkpoint_key_spellings_reduce_to_text_model_names_and_vision_or_head_keys_are_dropped(
    key: str, expected: str | None
) -> None:
    assert _text_key(key) == expected


def test_prompts_use_the_instruct_template_without_a_system_turn() -> None:
    tokenizer = FakeTokenizer(5)
    Flux3TextEncoder(FakeStack(), tokenizer).encode("a cat", torch.device("cpu"))

    assert tokenizer.texts == ["<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n"]


def test_the_output_concatenates_eight_taps_with_each_tap_being_that_layers_input() -> None:
    encoder = Flux3TextEncoder(FakeStack(), FakeTokenizer(5))

    out = encoder.encode("x", torch.device("cpu"), dtype=torch.float32)

    assert out.shape == (80, 8 * HIDDEN)
    first_token = out[0].view(8, HIDDEN)[:, 0]
    # token id 100 plus the layer index it entered; the last tap is the output after layer 31
    assert first_token.tolist() == [104.0, 108.0, 112.0, 116.0, 120.0, 124.0, 128.0, 132.0]


def test_the_sequence_is_right_padded_to_a_multiple_of_80_with_the_endoftext_token() -> None:
    model = FakeStack()
    Flux3TextEncoder(model, FakeTokenizer(81)).encode("x", torch.device("cpu"))

    ids = model.seen["ids"][0]
    assert ids.shape == (160,)
    assert ids[:81].tolist() == list(range(100, 181)) and set(ids[81:].tolist()) == {te.PAD_TOKEN}


def test_padding_is_masked_and_sits_at_rope_position_one() -> None:
    model = FakeStack()
    Flux3TextEncoder(model, FakeTokenizer(3)).encode("x", torch.device("cpu"))

    mask, pos = model.seen["mask"][0], model.seen["pos"][0]
    assert mask[:3].tolist() == [1, 1, 1] and int(mask[3:].sum()) == 0
    assert pos[:3].tolist() == [0, 1, 2] and set(pos[3:].tolist()) == {1}
    assert pos[3:].shape == (77,) and model.seen["use_cache"] is False


def test_an_exact_multiple_of_80_gets_no_padding() -> None:
    assert Flux3TextEncoder(FakeStack(), FakeTokenizer(80)).encode("x", torch.device("cpu")).shape[0] == 80


def test_prompts_over_the_reference_cap_are_truncated_to_4096_tokens() -> None:
    model = FakeStack()
    out = Flux3TextEncoder(model, FakeTokenizer(5000)).encode("x", torch.device("cpu"))

    assert out.shape[0] == te.MAX_SEQUENCE_LENGTH
    assert int(model.seen["mask"].sum()) == te.MAX_SEQUENCE_LENGTH


def test_a_model_returning_a_plain_tuple_is_read_by_index() -> None:
    encoder = Flux3TextEncoder(FakeStack(tuple_output=True), FakeTokenizer(5))

    out = encoder.encode("x", torch.device("cpu"), dtype=torch.float32)

    assert out[0].view(8, HIDDEN)[-1, 0].item() == 132.0


def test_the_default_output_dtype_is_bfloat16_and_a_second_call_does_not_reuse_stale_taps() -> None:
    encoder = Flux3TextEncoder(FakeStack(), FakeTokenizer(5))

    first = encoder.encode("x", torch.device("cpu"))
    second = encoder.encode("x", torch.device("cpu"))

    assert first.dtype == torch.bfloat16 and torch.equal(first, second)
