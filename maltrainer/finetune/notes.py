"""Notes the step planner prints about a finetune's dataset and prior (``--suggest-steps``)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List

from maltrainer.data.minimax_dataset import scan_media_sources
from maltrainer.finetune.prior import resolve_classes
from maltrainer.finetune.repeats import unknown_keys
from maltrainer.utils.step_planner import folder_counts


def names_its_folder(root: str, folder: Path, captions: List[str]) -> bool:
    # the nearest prefix.txt decides, as it does for the captions: "[folder]" in it names
    # the person; fixed text ("extreme close-up photo of human eyes") is a concept
    stop = Path(root).resolve().parent
    current = folder.resolve()
    while True:
        prefix = current / "prefix.txt"
        if prefix.exists():
            return "[folder]" in prefix.read_text(encoding="utf-8")
        if current == stop or current.parent == current:
            break
        current = current.parent
    # no prefix.txt: a person only if the captions spell the folder's name out
    return any(re.search(rf"(?<!\w){re.escape(folder.name)}(?!\w)", c, re.IGNORECASE) for c in captions)


def step_notes(config: dict) -> List[str]:
    """Folders prior preservation would miss. People are named by their folder through
    prefix.txt's [folder] ("photo of feliciaday, ..."), so a folder whose captions name its
    own folder is a person, and needs a prior.classes entry. Folders that name nobody (e.g.
    body/eyes: "extreme close-up photo of human eyes") are concepts and get no prior term.
    Folders with no caption at all would train on the empty prompt: they are named too."""
    dataset_config = config["dataset"]
    root = dataset_config.get("path", "")
    stray = unknown_keys(dataset_config.get("folder_repeats"), [name for name, _ in folder_counts(root)])
    stray_note = (
        [f"dataset.folder_repeats names {', '.join(stray)}, which matches no folder (no effect)"] if stray else []
    )
    sources = scan_media_sources(
        root,
        kinds=["image"],
        trigger_token=dataset_config.get("trigger"),
        default_caption=dataset_config.get("default_caption", ""),
    )
    by_folder: dict = {}
    for source in sources:
        by_folder.setdefault(source.path.parent, []).append(source.caption)
    notes = list(stray_note)
    uncaptioned = sorted(
        Path(os.path.relpath(folder, root)).as_posix()
        for folder, captions in by_folder.items()
        if any(not c.strip() for c in captions)
    )
    if uncaptioned:
        notes.append(
            f"images with no caption at all in {', '.join(uncaptioned)}: they would train on the empty "
            f"prompt (CFG's negative); add .txt captions or a prefix.txt"
        )
    prior_config = config.get("prior", {}) or {}
    if float(prior_config.get("weight", 0.0) or 0.0) <= 0:
        return notes
    try:
        classes = {k.lower() for k in resolve_classes(prior_config, dataset_config.get("trigger"))}
    except ValueError as error:
        return notes + [str(error)]

    people = {
        folder.name.lower(): Path(os.path.relpath(folder, root)).as_posix()
        for folder, captions in by_folder.items()
        if names_its_folder(root, folder, captions)
    }
    missing = sorted(rel for name, rel in people.items() if name not in classes)
    if missing:
        notes.append(
            f"prior.classes has no entry for {', '.join(missing)}: add each as "
            f"<folder>: woman / man, or those people train without prior preservation"
        )
    unused = sorted(classes - set(people))
    if people and unused:
        notes.append(f"prior.classes names {', '.join(unused)}, which no folder here names (harmless)")
    return notes
