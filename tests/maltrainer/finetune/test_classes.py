from __future__ import annotations

from typing import Dict, Optional

import pytest

from maltrainer.finetune.classes import class_caption, resolve_classes


@pytest.mark.parametrize(
    ("caption", "classes", "expected"),
    [
        ("photo of feliciaday, a woman", {"feliciaday": "woman"}, "photo of woman, a woman"),
        ("FeliciaDay smiling", {"feliciaday": "woman"}, "woman smiling"),
        ("ben and feliciaday", {"feliciaday": "woman", "ben": "man"}, "man and woman"),
        ("feliciaday again feliciaday", {"feliciaday": "woman"}, "woman again woman"),
        ("photo of anna smith", {"anna": "girl", "anna smith": "woman"}, "photo of woman"),
        ("photo of c.j", {"c.j": "man"}, "photo of man"),
    ],
)
def test_every_trigger_becomes_its_class_word(caption: str, classes: Dict[str, str], expected: str) -> None:
    assert class_caption(caption, classes) == expected


@pytest.mark.parametrize("caption", ["photo of a woman", "photo of feliciadays", "photo of xfeliciaday", ""])
def test_a_caption_naming_no_whole_trigger_has_no_class_caption(caption: str) -> None:
    assert class_caption(caption, {"feliciaday": "woman"}) is None


def test_classes_come_from_the_prior_config_stripped() -> None:
    config = {"classes": {" feliciaday ": " woman ", " ": "nobody", "ben": "man"}, "class_word": "person"}

    assert resolve_classes(config, "someone") == {"feliciaday": "woman", "ben": "man"}


def test_the_dataset_trigger_takes_the_class_word_when_no_classes_are_given() -> None:
    assert resolve_classes({"classes": None, "class_word": " woman "}, " feliciaday ") == {"feliciaday": "woman"}


@pytest.mark.parametrize(
    ("config", "trigger"),
    [
        ({}, None),
        ({"class_word": "woman"}, None),
        ({"class_word": "woman"}, ""),
        ({"class_word": "  "}, "feliciaday"),
        ({"classes": {" ": "woman"}}, None),
    ],
)
def test_a_prior_without_any_class_is_an_error(config: dict, trigger: Optional[str]) -> None:
    with pytest.raises(ValueError, match="prior.weight > 0 needs prior.classes"):
        resolve_classes(config, trigger)
