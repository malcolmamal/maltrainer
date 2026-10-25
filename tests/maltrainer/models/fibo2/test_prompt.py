from __future__ import annotations

import json

import pytest

from maltrainer.models.fibo2.prompt import NULL_CAPTION, clean, parse_caption, prompt_text


def test_a_plain_sentence_is_used_as_it_is() -> None:
    assert prompt_text("a photo of zed") == "a photo of zed"
    assert parse_caption("a photo of zed") == "a photo of zed"


def test_json_that_is_not_an_object_is_a_plain_prompt() -> None:
    assert prompt_text("[1, 2]") == "[1, 2]"
    assert prompt_text("42") == "42"


@pytest.mark.parametrize("empty", ["", "   ", "\n"])
def test_an_empty_prompt_is_the_null_caption(empty: str) -> None:
    assert prompt_text(empty) == NULL_CAPTION
    assert json.loads(NULL_CAPTION)["objects"] == []


def test_a_caption_is_compacted_and_its_empty_values_dropped_at_every_depth() -> None:
    caption = {
        "short_description": "A bike",
        "objects": [],
        "lighting": {"conditions": "sun", "direction": ""},
        "context": "",
    }

    assert prompt_text(json.dumps(caption)) == '{"short_description":"A bike","lighting":{"conditions":"sun"}}'


def test_the_scores_are_dropped_at_the_top_level_and_one_level_down() -> None:
    caption = {"short_description": "x", "aesthetic_score": 5, "aesthetics": {"preference_score": 3, "mood": "calm"}}

    assert parse_caption(json.dumps(caption)) == {"short_description": "x", "aesthetics": {"mood": "calm"}}


def test_non_ascii_text_is_kept_unescaped() -> None:
    assert prompt_text('{"short_description": "Zażółć"}') == '{"short_description":"Zażółć"}'


def test_a_caption_with_only_a_description_and_a_text_render_list_keeps_both_even_when_empty() -> None:
    assert parse_caption('{"short_description": "", "text_render": []}') == {"short_description": "", "text_render": []}
    assert parse_caption('{"short_description": "x", "text_render": [{"text": "hi"}]}') == {
        "short_description": "x",
        "text_render": [{"text": "hi"}],
    }


def test_an_edit_instruction_is_taken_out_and_a_structured_caption_unwrapped() -> None:
    wrapped = {"structured_caption": {"short_description": "x", "objects": []}, "edit_instruction": "remove it"}

    assert parse_caption(json.dumps(wrapped)) == {"short_description": "x"}
    assert parse_caption('{"structured_caption": "a sentence"}') == "a sentence"
    assert parse_caption('{"structured_caption": ""}') is None


def test_a_caption_that_cleans_to_nothing_is_an_empty_object() -> None:
    assert parse_caption('{"objects": []}') is None
    assert prompt_text('{"objects": []}') == "{}"


def test_clean_drops_nan_none_and_empty_containers_but_keeps_zero_and_false() -> None:
    assert clean({"a": float("nan"), "b": None, "c": [], "d": {}, "e": 0, "f": False, "g": [[], "x"]}) == {
        "e": 0,
        "f": False,
        "g": ["x"],
    }
