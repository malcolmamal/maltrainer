"""Prompt-token bookkeeping and checkpoint key mapping of the Krea 2 text encoder."""

from __future__ import annotations

import pytest
from transformers import AutoTokenizer

from maltrainer.models.krea2 import text_encoder as te
from maltrainer.models.krea2.text_encoder import IM_START, NEWLINE, TAP_LAYERS, USER, _prefix_end, _text_key


def test_tap_layers_are_every_third_hidden_state_from_2_to_35():
    assert list(range(2, 36, 3)) == TAP_LAYERS and len(TAP_LAYERS) == 12


def test_template_wraps_the_prompt_in_the_qwen_image_chat_format_without_a_think_block():
    text = te.TEMPLATE.format("a cat")
    assert text.startswith("<|im_start|>system\nDescribe the image")
    assert text.endswith("<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n")
    assert "<think>" not in text


def test_prefix_end_skips_system_and_user_header_so_the_prompt_starts_the_output():
    ids = [IM_START, 1, 2, IM_START, USER, NEWLINE, 50, 51, 7, IM_START, 9]
    assert _prefix_end(ids) == 6  # second <|im_start|> at 3, plus "<|im_start|>user\n"
    assert ids[_prefix_end(ids)] == 50


def test_prefix_end_only_counts_the_first_two_im_starts():
    ids = [IM_START, 1, IM_START, USER, NEWLINE, 5, IM_START, 6, 7]
    assert _prefix_end(ids) == 5  # the assistant header's <|im_start|> is not a template end


def test_prefix_end_without_the_user_header_stays_on_the_second_im_start():
    assert _prefix_end([IM_START, 1, IM_START, 4, 5, 6]) == 2


def test_prefix_end_needs_room_after_the_header_and_never_goes_negative():
    assert _prefix_end([IM_START, 1, IM_START, USER, NEWLINE]) == 2  # nothing after the header
    assert _prefix_end([1, 2, 3]) == 0
    assert _prefix_end([]) == 0


def test_prefix_end_on_the_real_tokenizer_lands_on_the_first_prompt_token():
    tokenizer = AutoTokenizer.from_pretrained(str(te.TOKENIZER_ASSETS), trust_remote_code=True)
    ids = tokenizer(te.TEMPLATE.format("a cat"), add_special_tokens=False).input_ids
    start = _prefix_end(ids)
    assert tokenizer.decode(ids[start:]) == "a cat<|im_end|>\n<|im_start|>assistant\n"


@pytest.mark.parametrize(
    "key, expected",
    [
        ("model.layers.3.mlp.up_proj.weight", "layers.3.mlp.up_proj.weight"),
        ("model.language_model.layers.3.mlp.up_proj.weight", "layers.3.mlp.up_proj.weight"),
        ("language_model.model.layers.3.mlp.up_proj.weight", "layers.3.mlp.up_proj.weight"),
        ("language_model.layers.0.input_layernorm.weight", "layers.0.input_layernorm.weight"),
        ("embed_tokens.weight", "embed_tokens.weight"),
        ("model.visual.blocks.0.attn.qkv.weight", None),
        ("visual.blocks.0.attn.qkv.weight", None),
        ("lm_head.weight", None),
    ],
)
def test_text_key_strips_container_prefixes_and_drops_vision_and_head_tensors(key, expected):
    assert _text_key(key) == expected


def test_text_key_handles_the_combined_comfyui_checkpoint_prefix():
    assert _text_key("text_encoders.qwen3vl_4b.transformer.model.layers.0.mlp.up_proj.weight") == (
        "layers.0.mlp.up_proj.weight"
    )
