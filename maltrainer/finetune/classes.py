"""Triggers and their class words: which words in a caption name a subject, and the caption with
each replaced by its class word ("photo of feliciaday" -> "photo of woman"). Prior preservation and
regularisation both caption by class."""

from __future__ import annotations

import re
from typing import Any


def class_caption(caption: str, classes: dict[str, str]) -> str | None:
    """``caption`` with every trigger (a whole word, any case) replaced by its class word, or
    None when it names no trigger. Longer triggers go first, so "annasmith" wins over "anna".

    One pass over the caption: an inserted class word is never itself replaced (bob -> man,
    man -> person leaves "man"), and it is inserted as written, backslashes included."""
    triggers = sorted(classes, key=len, reverse=True)
    if not triggers:
        return None
    alternatives = "|".join(f"(?P<t{i}>{re.escape(trigger)})" for i, trigger in enumerate(triggers))
    pattern = re.compile(rf"(?<![\w])(?:{alternatives})(?![\w])", re.IGNORECASE)
    out, count = pattern.subn(lambda match: classes[triggers[int((match.lastgroup or "t0")[1:])]], caption)
    return out if count else None


def resolve_classes(prior_config: dict[str, Any], trigger: str | None) -> dict[str, str]:
    """``prior.classes`` (trigger -> class word), or ``dataset.trigger`` -> ``prior.class_word``."""
    classes = {str(k).strip(): str(v).strip() for k, v in (prior_config.get("classes") or {}).items() if str(k).strip()}
    word = str(prior_config.get("class_word") or "").strip()
    if not classes and trigger and word:
        classes = {trigger.strip(): word}
    if not classes:
        raise ValueError(
            "prior.weight > 0 needs prior.classes (trigger: class word, e.g. {feliciaday: woman}) "
            "or dataset.trigger with prior.class_word"
        )
    return classes
