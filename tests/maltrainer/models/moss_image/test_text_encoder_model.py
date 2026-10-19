from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch

import maltrainer.models.moss_image.text_encoder as te
from maltrainer.models.moss_image.text_encoder import (
    MAX_SEQUENCE_LENGTH,
    TEMPLATE,
    MOSSImageTextEncoder,
    Qwen35TextConfig,
    _text_key,
)

from .qwen35_tiny import HIDDEN, VOCAB, tiny_text_config, tiny_text_model

CPU = torch.device("cpu")


class FakeTokenizer:
    """Maps each character to ``ord(c) % VOCAB`` and records what it was asked."""

    def __init__(self) -> None:
        self.calls: list = []

    def __call__(self, text: str, add_special_tokens: bool = True) -> SimpleNamespace:
        self.calls.append((text, add_special_tokens))
        return SimpleNamespace(input_ids=[ord(c) % VOCAB for c in text])


def test_the_config_is_read_back_off_the_weights_it_was_built_for() -> None:
    state = tiny_text_model().state_dict()

    config = Qwen35TextConfig.from_state_dict(state)

    expected = tiny_text_config()
    assert config == expected
    assert config.full_attention_layers == (3,)


def test_a_state_dict_without_linear_attention_markers_is_not_a_qwen35_text_encoder() -> None:
    state = {k: v for k, v in tiny_text_model().state_dict().items() if "A_log" not in k}

    with pytest.raises(ValueError, match="Not a Qwen3.5 text encoder"):
        Qwen35TextConfig.from_state_dict(state)


def test_the_default_config_places_full_attention_on_every_fourth_layer() -> None:
    config = Qwen35TextConfig()

    assert config.full_attention_layers == tuple(range(3, 32, 4))
    assert len(config.full_attention_layers) == 8


def test_only_the_marked_layers_carry_gated_attention_and_the_rest_linear_attention() -> None:
    model = tiny_text_model()

    kinds = [hasattr(layer, "self_attn") for layer in model.layers]

    assert kinds == [False, False, False, True]
    assert [hasattr(layer, "linear_attn") for layer in model.layers] == [True, True, True, False]


def test_the_model_returns_the_final_normed_state_of_every_token_in_the_compute_dtype() -> None:
    model = tiny_text_model()
    ids = torch.tensor([[1, 5, 9, 2, 7]])

    out = model(ids)

    cos, sin = model.rope(5, CPU)
    x = model.embed_tokens(ids)
    for layer in model.layers:
        x = layer(x, cos, sin)
    torch.testing.assert_close(out, model.norm(x))
    assert out.shape == (1, 5, HIDDEN) and out.dtype == torch.float32


def test_a_token_only_sees_the_tokens_before_it() -> None:
    model = tiny_text_model()
    a = model(torch.tensor([[1, 2, 3, 4, 5]]))
    b = model(torch.tensor([[1, 2, 3, 9, 9]]))

    torch.testing.assert_close(a[:, :3], b[:, :3])
    assert not torch.allclose(a[:, 3:], b[:, 3:])


def test_the_prompt_is_stripped_and_wrapped_in_the_open_think_chat_template() -> None:
    tokenizer = FakeTokenizer()
    encoder = MOSSImageTextEncoder(tiny_text_model(), tokenizer)

    ids = encoder.token_ids("  a cat \n")

    text = TEMPLATE.format("a cat")
    assert tokenizer.calls == [(text, False)]
    assert ids == [ord(c) % VOCAB for c in text]
    assert text.endswith("<|im_start|>assistant\n<think>\n")


def test_a_prompt_longer_than_the_window_is_truncated_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    encoder = MOSSImageTextEncoder(tiny_text_model(), FakeTokenizer())

    with caplog.at_level(logging.WARNING, logger=te.logger.name):
        ids = encoder.token_ids("x" * (MAX_SEQUENCE_LENGTH + 10))

    assert len(ids) == MAX_SEQUENCE_LENGTH
    assert f"truncated to {MAX_SEQUENCE_LENGTH}" in caplog.text


def test_a_prompt_at_exactly_the_window_is_kept_whole_without_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    encoder = MOSSImageTextEncoder(tiny_text_model(), FakeTokenizer())
    filler = MAX_SEQUENCE_LENGTH - len(TEMPLATE.format(""))

    with caplog.at_level(logging.WARNING, logger=te.logger.name):
        ids = encoder.token_ids("x" * filler)

    assert len(ids) == MAX_SEQUENCE_LENGTH and not caplog.records


def test_encode_returns_one_row_per_token_at_the_requested_dtype() -> None:
    model = tiny_text_model()
    tokenizer = FakeTokenizer()
    encoder = MOSSImageTextEncoder(model, tokenizer)

    out = encoder.encode("hi", CPU, torch.float16)

    ids = encoder.token_ids("hi")
    assert out.shape == (len(ids), HIDDEN) and out.dtype == torch.float16
    torch.testing.assert_close(out, model(torch.tensor([ids]))[0].to(torch.float16))


@pytest.mark.parametrize(
    "key, expected",
    [
        ("text_encoders.qwen35_9b.transformer.model.layers.0.mlp.up_proj.weight", "layers.0.mlp.up_proj.weight"),
        ("text_encoders.qwen35_9b.transformer.embed_tokens.weight", "embed_tokens.weight"),
        ("model.language_model.norm.weight", "norm.weight"),
        ("language_model.model.layers.1.x", "layers.1.x"),
        ("language_model.layers.1.x", "layers.1.x"),
        ("model.layers.2.x", "layers.2.x"),
        ("layers.3.x", "layers.3.x"),
        ("visual.blocks.0.attn.weight", None),
        ("lm_head.weight", None),
        ("mtp.fc.weight", None),
        ("model.visual.blocks.0.weight", None),
        ("model.language_model.visual.x", None),
    ],
)
def test_checkpoint_key_spellings_map_to_module_names_and_non_text_towers_are_skipped(
    key: str, expected: str | None
) -> None:
    assert _text_key(key) == expected
