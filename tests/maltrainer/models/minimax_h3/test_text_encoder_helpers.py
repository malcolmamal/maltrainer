from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import maltrainer.models.minimax_h3.text_encoder as te
from maltrainer.models.minimax_h3.text_encoder import (
    BUNDLED_TOKENIZER_ASSETS,
    _describe_text_config,
    _resolve_tokenizer_assets,
    _text_encoder_key,
)


def test_the_bundled_assets_directory_is_used_when_no_tokenizer_path_is_given() -> None:
    assert _resolve_tokenizer_assets(None) == BUNDLED_TOKENIZER_ASSETS


def test_an_explicit_tokenizer_directory_with_a_tokenizer_json_is_returned_as_a_path(tmp_path: Path) -> None:
    (tmp_path / "tokenizer.json").write_text("{}")

    assert _resolve_tokenizer_assets(str(tmp_path)) == tmp_path


def test_an_explicit_directory_without_tokenizer_json_is_refused_blaming_the_setting(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"No tokenizer.json in .*from model.tokenizer_path"):
        _resolve_tokenizer_assets(tmp_path)


def test_missing_bundled_assets_are_reported_as_such(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(te, "BUNDLED_TOKENIZER_ASSETS", tmp_path)

    with pytest.raises(FileNotFoundError, match="from the bundled assets directory"):
        _resolve_tokenizer_assets(None)


def test_the_text_tower_summary_lists_the_sizes_and_the_merged_rope_parameters() -> None:
    config = SimpleNamespace(
        num_hidden_layers=50,
        hidden_size=5120,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=128,
        rope_parameters={"rope_theta": 5e6},
    )

    text = _describe_text_config(config)

    assert text == (
        "num_hidden_layers=50 hidden_size=5120 num_attention_heads=64 num_key_value_heads=8 "
        "head_dim=128 rope={'rope_theta': 5000000.0}"
    )


def test_the_text_tower_summary_falls_back_to_the_older_rope_fields_and_never_raises() -> None:
    config = SimpleNamespace(hidden_size=8, rope_theta=10.0)

    text = _describe_text_config(config)

    assert "hidden_size=8" in text and "num_hidden_layers=?" in text
    assert "rope={'rope_theta': 10.0, 'rope_scaling': None}" in text


@pytest.mark.parametrize(
    "prefix, expected",
    [
        ("visual", "visual"),
        ("visual.blocks.3.attn", "visual.blocks.3.attn"),
        ("model", "language_model"),
        ("model.layers.4.mlp", "language_model.layers.4.mlp"),
        ("model.embed_tokens", "language_model.embed_tokens"),
        ("language_model.norm", "language_model.norm"),
        ("lm_head", "lm_head"),
    ],
)
def test_checkpoint_prefixes_map_onto_the_vision_and_language_towers(prefix: str, expected: str) -> None:
    assert _text_encoder_key(prefix) == expected
