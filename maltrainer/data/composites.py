"""Side-by-side pictures for a multi-subject LoRA: one of a subject's photos next to a stranger in a single 2:1 frame.

They teach "this person is the subject, that one stays a stranger" inside one picture, which is where a normal LoRA
leaks. The captions describe the stranger first and the subject second: the text encoder is causal, so words after the
trigger carry it and the stranger's description is kept clean of it.

The strangers are a folder of pictures with a ``strangers.json`` ([{"file": ..., "description": ...}, ...]), for
example Fizgig's published set (ShootTheSound/fizgig-multi-person-strangers).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from maltrainer.data.minimax_dataset import MediaSource
from maltrainer.data.subjects import Subject

PREFIX = "maltrainer_sbs_"  # the composites' file names (the trainer's lr boost looks for it)
FACECROP = "FaceCrop_"  # Image Prep's face crops: framed on the face already, used first
TAIL = "two people side by side, two close-up portraits"


@dataclass
class Stranger:
    path: Path
    description: str


def load_strangers(folder: str | Path) -> list[Stranger]:
    """The strangers in ``folder``, from its strangers.json."""
    root = Path(folder)
    try:
        rows = json.loads((root / "strangers.json").read_text(encoding="utf-8"))
        strangers = [Stranger(root / row["file"], str(row["description"])) for row in rows]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(
            f"isolation.side_by_side.strangers_path {root} has no usable strangers.json: {error}"
        ) from error
    missing = [s.path.name for s in strangers if not s.path.is_file()]
    if not strangers or missing:
        raise ValueError(f"isolation.side_by_side.strangers_path {root}: no strangers, or files missing: {missing[:3]}")
    return strangers


def _square(image: Image.Image, size: int) -> Image.Image:
    """The largest centred square, no zoom, at ``size``."""
    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    return (
        image.convert("RGB").crop((left, top, left + side, top + side)).resize((size, size), Image.Resampling.LANCZOS)
    )


def _picks(photos: list[Path], count: int, rng: random.Random) -> list[Path]:
    """``count`` photos, face crops first, no photo twice until every one has been used."""
    crops = [p for p in photos if p.name.startswith(FACECROP)]
    rest = [p for p in photos if not p.name.startswith(FACECROP)]
    rng.shuffle(crops)
    rng.shuffle(rest)
    order = crops + rest
    return [order[k % len(order)] for k in range(count)]


def build_composites(
    subject: Subject, index: int, strangers: list[Stranger], out_dir: Path, count: int, seed: int, panel: int = 512
) -> list[MediaSource]:
    """``count`` side-by-sides of one subject in ``out_dir``, as dataset sources (captions name the fused trigger)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed * 1000 + index)
    order = list(range(len(strangers)))
    rng.shuffle(order)
    photos = _picks([s.path for s in subject.sources], count, rng)
    sources: list[MediaSource] = []
    for k, photo in enumerate(photos):
        stranger = strangers[order[k % len(order)]]
        mine, theirs = ("left", "right") if rng.random() < 0.5 else ("right", "left")
        sheet = Image.new("RGB", (2 * panel, panel))
        with Image.open(photo) as a, Image.open(stranger.path) as b:
            sheet.paste(_square(a, panel), (0 if mine == "left" else panel, 0))
            sheet.paste(_square(b, panel), (0 if theirs == "left" else panel, 0))
        path = out_dir / f"{PREFIX}{index}_{k:02d}.png"
        sheet.save(path)
        caption = f"{stranger.description} on the {theirs}, and {subject.words.fused} on the {mine}, {TAIL}"
        sources.append(MediaSource(path, "image", caption))
    return sources
