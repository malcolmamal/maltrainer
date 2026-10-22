"""class_caption makes one pass over the caption. For every config without a class word that is
itself a trigger, or one with a backslash, it must give what the earlier one-trigger-at-a-time
version gave: MOSS-Image finetune's prior and regularisation captions depend on it."""

from __future__ import annotations

import random
import re

from maltrainer.finetune.classes import class_caption

TRIGGERS = ["anna", "annasmith", "anna smith", "bob", "Bob", "ohwx", "sks", "feliciaday", "x7", "rhea"]
CLASS_WORDS = ["woman", "man", "person", "young woman", "dog", "", "Lady"]
FILLER = ["photo", "of", "a", "with", "and", "annabelle", "bobby", "smith", "sksx", "the", "x7b"]
SEPARATORS = [" ", " ", ", ", ". ", "-", "'s ", "\n", "(", ")"]


def one_trigger_at_a_time(caption: str, classes: dict[str, str]) -> str | None:
    out, hit = caption, False
    for trigger in sorted(classes, key=len, reverse=True):
        pattern = re.compile(rf"(?<![\w]){re.escape(trigger)}(?![\w])", re.IGNORECASE)
        out, count = pattern.subn(classes[trigger], out)
        hit = hit or count > 0
    return out if hit else None


def a_caption(rng: random.Random, triggers: list[str]) -> str:
    words = [rng.choice(triggers + FILLER) for _ in range(rng.randint(0, 8))]
    words = [word.upper() if rng.random() < 0.1 else word.title() if rng.random() < 0.1 else word for word in words]
    return "".join(word + rng.choice(SEPARATORS) for word in words).strip()


def test_one_pass_matches_one_trigger_at_a_time_on_configs_without_chains() -> None:
    rng = random.Random(0)
    for _ in range(5000):
        triggers = rng.sample(TRIGGERS, rng.randint(1, 4))
        classes = {trigger: rng.choice(CLASS_WORDS) for trigger in triggers}
        caption = a_caption(rng, triggers)

        assert class_caption(caption, classes) == one_trigger_at_a_time(caption, classes), (caption, classes)


def test_an_inserted_class_word_is_not_replaced_again() -> None:
    assert class_caption("photo of bob", {"bob": "man", "man": "person"}) == "photo of man"


def test_a_class_word_is_inserted_as_written() -> None:
    assert class_caption("photo of ann", {"ann": r"wo\man"}) == r"photo of wo\man"


def test_no_classes_means_no_class_caption() -> None:
    assert class_caption("photo of bob", {}) is None
