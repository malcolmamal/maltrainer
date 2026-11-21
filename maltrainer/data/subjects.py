"""The subjects of a multi-subject LoRA: up to four folders, each with its own trigger word (Fizgig LoRA Isolation).

Each subject is a folder of its own pictures. Its trigger is joined to what it is in every caption as they are read
(see subject_text), and DOP's trigger-free caption puts the class phrase back, for every subject at once.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from maltrainer.data.masks import is_mask_file
from maltrainer.data.minimax_dataset import MediaSource, scan_media_sources
from maltrainer.data.subject_text import SubjectWords, fuse, subject_words

logger = logging.getLogger(__name__)

MAX_SUBJECTS = 4


@dataclass
class Subject:
    path: Path
    words: SubjectWords
    sources: list[MediaSource]  # captions already read with the fused trigger
    word: str  # what the captions say for the subject: the joined word, or the trigger when nothing is joined


def _one(entry: dict[str, Any], defaults: dict[str, Any], as_written: bool, fuse_triggers: bool) -> Subject:
    path, trigger = entry.get("path"), str(entry.get("trigger", "")).strip()
    if not path or not trigger:
        raise ValueError("every isolation subject needs a path and a trigger")
    found = scan_media_sources(path, ["image"], trigger, str(entry.get("default_caption", defaults["default_caption"])))
    sources = [source for source in found if not is_mask_file(source.path)]
    if not sources:
        raise ValueError(f"No images found for subject {trigger!r} under {path}")
    words = subject_words(
        trigger, [s.caption for s in sources], str(entry.get("what", "")), str(entry.get("kind", "auto")), as_written
    )
    return Subject(Path(path), words, sources, words.fused if fuse_triggers else words.trigger)


def _check_apart(subjects: list[Subject]) -> None:
    triggers = [s.words.trigger.lower() for s in subjects]
    if len(set(triggers)) != len(triggers) or len({s.path.resolve() for s in subjects}) != len(subjects):
        raise ValueError("isolation subjects need their own trigger word and their own folder each")
    firsts = [t[0] for t in triggers]
    if len(set(firsts)) != len(firsts):
        logger.warning(
            "Isolation: triggers %s share a first letter; start each with a different one so the text encoder "
            "keeps them apart.",
            ", ".join(triggers),
        )


def load_subjects(isolation: dict[str, Any], default_caption: str) -> list[Subject]:
    """The subjects ``isolation.subjects`` names, their captions read and joined."""
    entries = isolation.get("subjects") or []
    if not 1 <= len(entries) <= MAX_SUBJECTS:
        raise ValueError(f"isolation.subjects needs 1 to {MAX_SUBJECTS} subjects, got {len(entries)}")
    as_written = bool(isolation.get("as_written", False))
    fuse_triggers = bool(isolation.get("fuse_triggers", True))
    defaults = {"default_caption": default_caption}
    subjects = [_one(entry, defaults, as_written, fuse_triggers) for entry in entries]
    _check_apart(subjects)
    if fuse_triggers and not as_written:
        # every subject's rules read every caption: a photo of two subjects sits in one folder, and names both
        rules = [rule for subject in subjects for rule in subject.words.rules]
        for subject in subjects:
            subject.sources = [MediaSource(s.path, s.kind, fuse(s.caption, rules)) for s in subject.sources]
    for subject in subjects:
        logger.info(
            "Isolation subject %s (%s, %d images): %s",
            subject.words.trigger,
            subject.words.kind,
            len(subject.sources),
            f"captions read as {subject.words.fused!r}"
            if subject.words.fused != subject.words.trigger
            else "as written",
        )
    return subjects


def balanced_repeats(counts: list[int]) -> list[int]:
    """How many times each subject's pictures go into an epoch so every subject trains about as often as the
    largest: [36, 53] -> [1, 1] (53/36 rounds to 1); [10, 50] -> [5, 1]."""
    top = max(counts)
    return [max(1, round(top / count)) for count in counts]
