"""The words of a multi-subject LoRA (Fizgig LoRA Isolation): a trigger joined to what it is, and the same caption
without it.

``zwxem, a woman smiling`` is read as ``zwxemwoman smiling`` so the subject never shares the plain " woman" token
every other woman uses; DOP's trigger-free caption then puts the class phrase back (``a woman smiling``). The caption
files on disk are never changed.
"""

from __future__ import annotations

import collections
import re
from collections.abc import Callable
from dataclasses import dataclass, field

# what a subject can be, read from the captions when ``what`` is not given
CLASS_WORDS = (
    "woman", "man", "girl", "boy", "lady", "guy", "person", "child", "kid", "baby", "teenager",
    "dog", "cat", "puppy", "kitten", "horse",
)  # fmt: skip
PEOPLE_WORDS = CLASS_WORDS[:11]  # only people get the side-by-side pictures
KINDS = ("auto", "person", "thing", "style")


@dataclass
class SubjectWords:
    trigger: str
    phrase: str  # "a woman"; "" for a style
    fused: str  # "zwxemwoman"; the trigger itself for a style
    kind: str  # person | thing | style
    rules: list[tuple[str, str]] = field(default_factory=list)  # caption rewrites, in order


def class_phrase(word: str) -> str:
    """ "man" -> "a man", "old man" -> "an old man"; a phrase that already starts with a / an / the as it is."""
    text = word.strip()
    if not text or re.match(r"^(a|an|the)\s", text, re.I):
        return text
    return ("an " if text[0].lower() in "aeiou" else "a ") + text


def _word_pattern(word: str) -> re.Pattern[str]:
    return re.compile(r"(?<!\w)" + re.escape(word) + r"(?!\w)", re.I)


def _replacement(value: str) -> Callable[[re.Match[str]], str]:
    """A replacement taken literally (no backslash or group escapes)."""
    return lambda _match: value


def fuse(text: str, rules: list[tuple[str, str]]) -> str:
    """A caption read with a subject's fused trigger (the rules in order)."""
    for old, new in rules:
        text = _word_pattern(old).sub(_replacement(new), text)
    return text


def _class_from(captions: list[str], trigger: str) -> str:
    """The class phrase the trigger's captions use most: "<trigger>, a young woman" gives "a woman"."""
    words = "|".join(CLASS_WORDS)
    opening = re.compile(
        r"^\s*" + re.escape(trigger) + r"\s*,?\s*(an?)\s+(?:[A-Za-z'-]+\s+){0,3}?(" + words + r")\b", re.I
    )
    found = collections.Counter(
        f"{m.group(1).lower()} {m.group(2).lower()}" for c in captions if (m := opening.match(c))
    )
    if not found:  # no "<trigger>, a woman" opening: the class word its captions mention most
        anywhere, own = re.compile(r"(?<!\w)(" + words + r")(?!\w)", re.I), _word_pattern(trigger)
        mentioned = collections.Counter(w.lower() for c in captions if own.search(c) for w in anywhere.findall(c))
        found = collections.Counter({class_phrase(w): n for w, n in mentioned.items()})
    if not found:
        raise ValueError(
            f"the captions never say what {trigger!r} is (woman, man, dog ...): set that subject's 'what' "
            "(for a style: kind: style)"
        )
    return found.most_common(1)[0][0]


def subject_words(
    trigger: str, captions: list[str], what: str = "", kind: str = "auto", as_written: bool = False
) -> SubjectWords:
    """One subject's words. ``what`` (else the captions) says what it is; a style keeps its trigger as written;
    ``as_written``: the captions already hold the joined word, nothing is rewritten."""
    trigger = trigger.strip()
    if not trigger:
        raise ValueError("a subject needs a trigger word")
    if kind not in KINDS:
        raise ValueError(f"subject kind must be one of {KINDS}, got {kind!r}")
    if kind == "style":
        return SubjectWords(trigger, "", trigger, "style")
    if as_written:
        ends = [
            w
            for w in sorted(CLASS_WORDS, key=len, reverse=True)
            if trigger.lower().endswith(w) and len(trigger) > len(w)
        ]
        phrase = class_phrase(what) or (class_phrase(ends[0]) if ends else "")
        if not phrase:
            raise ValueError(f"say what {trigger!r} is: set that subject's 'what' (woman, man, dog ...)")
        return SubjectWords(trigger, phrase, trigger, _kind_of(kind, phrase))
    phrase = class_phrase(what) or _class_from(captions, trigger)
    fused = re.sub(r"\s+", "", trigger) + phrase.rsplit(maxsplit=1)[-1].lower()
    rules = [(f"{trigger}, {phrase}", fused), (f"{trigger} {phrase}", fused), (trigger, fused)]
    return SubjectWords(trigger, phrase, fused, _kind_of(kind, phrase), rules)


def _kind_of(kind: str, phrase: str) -> str:
    if kind in ("person", "thing"):
        return kind
    return "person" if phrase.rsplit(maxsplit=1)[-1].lower() in PEOPLE_WORDS else "thing"


def _swap(text: str, pattern: re.Pattern[str], phrase: str) -> str:
    """``pattern`` (a trigger) in ``text`` swapped for ``phrase``, or dropped when the text says it right after."""
    noun = phrase.rsplit(maxsplit=1)[-1] if phrase.strip() else ""
    said = re.compile(r"\s*,?\s*(?:(?:a|an|the)\s+)?(?:[\w'-]+\s+){0,2}?" + re.escape(noun) + r"(?!\w)", re.I)
    return pattern.sub(lambda match: "" if noun and said.match(text, match.end()) else phrase, text)


def dop_caption(caption: str, pairs: list[tuple[str, str]]) -> str | None:
    """The caption with each trigger swapped for its class phrase (dropped when empty, or when the caption already
    says it right after the trigger), or None when it names none of the triggers: nothing in it to keep apart."""
    out, hit = caption, False
    for trigger, phrase in pairs:
        pattern = _word_pattern(trigger)
        if not pattern.search(out):
            continue
        hit = True
        out = _swap(out, pattern, phrase)
    if not hit:
        return None
    out = re.sub(r"\s*,\s*(?:,\s*)+", ", ", out)  # a dropped trigger leaves ", ," or a stray comma
    out = re.sub(r"^\s*,\s*|\s*,\s*$", "", out)
    return re.sub(r"\s{2,}", " ", out).strip()
