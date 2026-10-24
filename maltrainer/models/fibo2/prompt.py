"""
Fibo 2.0 prompts: a structured JSON caption cleaned and compacted the way the model was trained, or plain text as is.

Mirrors diffusers' ``BriaFibo2Pipeline._parse_prompt`` and ComfyUI's comfy/text_encoders/fibo2.py. The images of a
caption's ``text_render`` entries are not drawn: they need the Qwen3-VL vision tower, which this trainer does not run.
"""

from __future__ import annotations

import json
import math
from typing import Any

# The null caption training used for classifier-free guidance; an empty prompt (caption dropout) is this.
NULL_CAPTION = (
    '{"short_description":"","objects":[],"background_setting":"","lighting":{},'
    '"aesthetics":{},"photographic_characteristics":{},"style_medium":"",'
    '"context":"","artistic_style":""}'
)
SCORES = ("aesthetic_score", "preference_score")


def dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def is_empty(value: Any) -> bool:
    if isinstance(value, float):
        return math.isnan(value)
    return value is None or (isinstance(value, (str, dict, list)) and not value)


def clean(value: Any) -> Any:
    """Empty values dropped, at every depth."""
    if isinstance(value, dict):
        cleaned = {key: clean(item) for key, item in value.items()}
        return {key: item for key, item in cleaned.items() if not is_empty(item)}
    if isinstance(value, list):
        return [item for item in map(clean, value) if not is_empty(item)]
    return value


def parse_caption(prompt: str) -> dict[str, Any] | str | None:
    """The caption as training saw it. A prompt that is not a JSON object comes back as it is."""
    try:
        caption = json.loads(prompt)
    except json.JSONDecodeError:
        return prompt
    if not isinstance(caption, dict):
        return prompt
    caption.pop("edit_instruction", None)
    if "structured_caption" in caption:
        caption = caption.pop("structured_caption")
        if not isinstance(caption, dict):
            return caption or None
    keys = set(caption)
    caption = clean(
        {
            key: {k: v for k, v in value.items() if k not in SCORES} if isinstance(value, dict) else value
            for key, value in caption.items()
            if key not in SCORES
        }
    )
    # exactly a short_description and a text_render list keeps both, the list even when empty: the image has no text
    if keys == {"short_description", "text_render"}:
        caption = {
            "short_description": caption.get("short_description") or "",
            "text_render": caption.get("text_render") or [],
        }
    return caption or None


def prompt_text(prompt: str) -> str:
    """What goes into the chat turn: the compact caption, the plain sentence as is, the null caption for ''."""
    if prompt.strip() == "":
        return NULL_CAPTION
    caption = parse_caption(prompt)
    return caption if isinstance(caption, str) else dumps(caption or {})
