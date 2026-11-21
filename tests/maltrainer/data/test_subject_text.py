"""A trigger joined to what it is, and a caption without it."""

from __future__ import annotations

import pytest

from maltrainer.data.subject_text import SubjectWords, class_phrase, dop_caption, fuse, subject_words

CAPTIONS = ["zwxem, a young woman smiling in a park", "zwxem, a woman at a cafe", "zwxem, a man by mistake"]


def test_a_class_word_becomes_a_phrase_with_the_right_article() -> None:
    assert [class_phrase(w) for w in ("man", "old man", "a dog", "The cat", "  ", "")] == [
        "a man",
        "an old man",
        "a dog",
        "The cat",
        "",
        "",
    ]


def test_the_class_comes_from_the_captions_most_common_opening() -> None:
    words = subject_words("zwxem", CAPTIONS)

    assert (words.phrase, words.fused, words.kind) == ("a woman", "zwxemwoman", "person")
    assert fuse("zwxem, a woman smiling in a park", words.rules) == "zwxemwoman smiling in a park"
    assert fuse(CAPTIONS[0], words.rules) == "zwxemwoman, a young woman smiling in a park"
    assert fuse("a photo of zwxem outdoors", words.rules) == "a photo of zwxemwoman outdoors"
    assert fuse("zwxem a woman", words.rules) == "zwxemwoman"


def test_without_an_opening_the_class_word_its_captions_mention_most_is_used() -> None:
    words = subject_words("rexo", ["a photo of rexo, the dog is running", "rexo lying down near the dog"])

    assert (words.phrase, words.fused, words.kind) == ("a dog", "rexodog", "thing")


def test_what_it_is_overrides_the_captions_and_an_unknown_class_is_refused() -> None:
    assert subject_words("qrxt", ["qrxt standing"], what="old man").fused == "qrxtman"
    with pytest.raises(ValueError, match="never say what 'qrxt' is"):
        subject_words("qrxt", ["qrxt standing"])


def test_a_style_keeps_its_trigger_as_written() -> None:
    assert subject_words("papercut", ["papercut art"], kind="style") == SubjectWords(
        "papercut", "", "papercut", "style"
    )


def test_kind_overrides_what_the_class_word_says() -> None:
    assert subject_words("rexo", [], what="dog", kind="person").kind == "person"
    with pytest.raises(ValueError, match="kind must be one of"):
        subject_words("rexo", [], what="dog", kind="alien")
    with pytest.raises(ValueError, match="needs a trigger word"):
        subject_words(" ", [])


def test_captions_written_with_the_joined_word_are_left_alone() -> None:
    words = subject_words("zwxemwoman", [], as_written=True)

    assert (words.phrase, words.fused, words.rules) == ("a woman", "zwxemwoman", [])
    assert subject_words("zed", [], what="boy", as_written=True).phrase == "a boy"
    with pytest.raises(ValueError, match="say what 'zed' is"):
        subject_words("zed", [], as_written=True)
    assert subject_words("man", [], what="man", as_written=True).fused == "man"


def test_dop_puts_the_class_phrase_back_for_every_subject() -> None:
    pairs = [("zwxemwoman", "a woman"), ("qrxtman", "a man")]

    assert dop_caption("zwxemwoman and qrxtman laughing", pairs) == "a woman and a man laughing"
    assert dop_caption("a tree", pairs) is None


def test_dop_never_says_the_class_twice_and_a_style_is_dropped() -> None:
    assert dop_caption("zwxem, a woman walking", [("zwxem", "a woman")]) == "a woman walking"
    assert dop_caption("zwxemwoman, a young woman smiling", [("zwxemwoman", "a woman")]) == "a young woman smiling"
    assert dop_caption("papercut, a tree", [("papercut", "")]) == "a tree"
    assert dop_caption("a tree, papercut", [("papercut", "")]) == "a tree"
    assert dop_caption("a, papercut, b", [("papercut", "")]) == "a, b"
