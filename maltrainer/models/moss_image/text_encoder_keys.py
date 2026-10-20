"""
Checkpoint key names of Qwen3.5 text encoders: prefixes to strip, tensors to skip (see ``text_encoder``).
"""

from __future__ import annotations

KEY_PREFIXES = (
    "text_encoders.qwen35_9b.transformer.model.",
    "text_encoders.qwen35_9b.transformer.",
    "model.language_model.",
    "language_model.model.",
    "language_model.",
    "model.",
)
SKIPPED = ("visual.", "lm_head.", "mtp.")


def _text_key(key: str) -> str | None:
    for prefix in KEY_PREFIXES:
        if key.startswith(prefix):
            key = key[len(prefix) :]
            break
    if key.startswith(SKIPPED):
        return None
    return key
