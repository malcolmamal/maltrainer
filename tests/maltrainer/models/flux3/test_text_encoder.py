from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from maltrainer.models.flux3 import text_encoder as te
from maltrainer.models.flux3.text_encoder import PAD_TOKEN, TAP_LAYERS, TEMPLATE, Flux3TextEncoder, _text_key

CPU = torch.device("cpu")
BASE = len(TEMPLATE.format(""))


class AddOne(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + 1.0


class StubModel(nn.Module):
    """32 layers that each add one to a two-wide copy of the token id, so every tap is known by hand."""

    def __init__(self, as_tuple: bool = False) -> None:
        super().__init__()
        self.layers = nn.ModuleList([AddOne() for _ in range(32)])
        self.as_tuple = as_tuple
        self.calls: list = []

    def forward(self, **kwargs: torch.Tensor) -> object:
        self.calls.append(kwargs)
        h = kwargs["input_ids"].float()[..., None].repeat(1, 1, 2)
        for layer in self.layers:
            h = layer(h)
        return (h,) if self.as_tuple else SimpleNamespace(last_hidden_state=h)


class CharTokenizer:
    def __init__(self) -> None:
        self.texts: list = []

    def __call__(self, text: str, add_special_tokens: bool) -> SimpleNamespace:
        assert add_special_tokens is False
        self.texts.append(text)
        return SimpleNamespace(input_ids=[ord(c) for c in text])


def _encoder(**kwargs: object) -> Flux3TextEncoder:
    return Flux3TextEncoder(StubModel(**kwargs), CharTokenizer())


def _prompt(real_tokens: int) -> str:
    return "a" * (real_tokens - BASE)


@pytest.mark.parametrize(("real", "window"), [(BASE, 80), (79, 80), (80, 80), (81, 160), (161, 240)])
def test_the_sequence_is_padded_to_a_multiple_of_eighty_tokens(real: int, window: int) -> None:
    encoder = _encoder()

    out = encoder.encode(_prompt(real), CPU)

    assert out.shape == (window, 8 * 2)


def test_the_prompt_goes_through_the_instruct_template_without_special_tokens() -> None:
    encoder = _encoder()

    encoder.encode("a red fox", CPU)

    assert encoder.tokenizer.texts == ["<|im_start|>user\na red fox<|im_end|>\n<|im_start|>assistant\n"]


def test_padding_is_masked_at_rope_position_one_and_filled_with_endoftext() -> None:
    encoder = _encoder()
    real = 100

    encoder.encode(_prompt(real), CPU)

    call = encoder.model.calls[0]
    assert call["input_ids"].shape == (1, 160)
    assert call["input_ids"][0, real:].unique().tolist() == [PAD_TOKEN]
    assert call["attention_mask"][0].tolist() == [1] * real + [0] * 60
    assert call["position_ids"][0].tolist() == list(range(real)) + [1] * 60
    assert call["use_cache"] is False


def test_padding_tokens_are_kept_in_the_output_and_carry_the_pad_id() -> None:
    out = _encoder().encode(_prompt(100), CPU, dtype=torch.float32)

    assert out[99, 0].item() == ord("\n") + TAP_LAYERS[0]
    assert out[100, 0].item() == PAD_TOKEN + TAP_LAYERS[0]


@pytest.mark.parametrize("as_tuple", [False, True])
def test_eight_hidden_states_are_concatenated_per_token_with_the_last_from_the_model_output(as_tuple: bool) -> None:
    out = _encoder(as_tuple=as_tuple).encode(_prompt(80), CPU, dtype=torch.float32)

    token = float(ord(TEMPLATE.format("a" * (80 - BASE))[0]))
    expected = [token + k for k in TAP_LAYERS[:-1]] + [token + 32]
    assert out[0].tolist() == [v for v in expected for _ in range(2)]
    assert out.dtype == torch.float32


def test_the_output_defaults_to_bf16() -> None:
    assert _encoder().encode("x", CPU).dtype == torch.bfloat16


def test_the_sequence_is_cut_at_the_maximum_length(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(te, "MAX_SEQUENCE_LENGTH", 100)
    encoder = _encoder()

    out = encoder.encode(_prompt(250), CPU)

    assert out.shape[0] == 100
    assert encoder.model.calls[0]["attention_mask"].sum().item() == 100


def test_each_call_starts_from_fresh_captures() -> None:
    encoder = _encoder()
    encoder.encode("one", CPU)

    encoder.encode("two", CPU)

    assert sorted(encoder._captured) == list(TAP_LAYERS[:-1])


def test_the_capture_hook_reads_the_hidden_states_from_kwargs_when_called_without_positionals() -> None:
    encoder = _encoder()
    hidden = torch.ones(1, 2, 2, requires_grad=True)

    encoder._capture(4)(None, (), {"hidden_states": hidden})

    assert not encoder._captured[4].requires_grad
    assert torch.equal(encoder._captured[4], hidden)


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("text_encoders.qwen3vl_4b.transformer.layers.0.mlp.up_proj.weight", "layers.0.mlp.up_proj.weight"),
        ("model.language_model.layers.1.input_layernorm.weight", "layers.1.input_layernorm.weight"),
        ("language_model.model.embed_tokens.weight", "embed_tokens.weight"),
        ("language_model.layers.2.self_attn.q_norm.weight", "layers.2.self_attn.q_norm.weight"),
        ("model.norm.weight", "norm.weight"),
        ("language_model.norm.weight", "norm.weight"),
        ("model.layers.1.q.weight", "layers.1.q.weight"),
        ("layers.3.mlp.down_proj.weight", "layers.3.mlp.down_proj.weight"),
        ("model.visual.blocks.0.attn.weight", None),
        ("visual.blocks.0.attn.weight", None),
        ("visual.merger.weight", None),
        ("lm_head.weight", None),
    ],
)
def test_checkpoint_key_spellings_map_to_the_text_model_or_are_dropped(key: str, expected: str | None) -> None:
    assert _text_key(key) == expected


def test_the_constants_match_the_vendored_qwen3_vl_tokenizer() -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(te.TOKENIZER_ASSETS))

    ids = tokenizer(TEMPLATE.format("hi"), add_special_tokens=False).input_ids

    assert tokenizer.convert_tokens_to_ids("<|endoftext|>") == PAD_TOKEN
    assert ids[0] == tokenizer.convert_tokens_to_ids("<|im_start|>")
    assert len(ids) < BASE
    assert te.MAX_SEQUENCE_LENGTH == 4096
