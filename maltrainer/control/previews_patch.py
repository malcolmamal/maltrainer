"""The preview settings the API shows and edits: the ``e`` menu's choices (every, rows, prompts on or off) as data."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from maltrainer.utils.previews import Previews

logger = logging.getLogger(__name__)


def preview_snapshot(previews: Previews, row_labels: Sequence[str]) -> dict[str, Any]:
    return {
        "every": previews.every,
        "rows": {label: previews.rows.get(label, True) for label in row_labels},
        "prompts": [
            {"slot": previews.slots[p], "prompt": p, "enabled": p in previews.enabled_prompts} for p in previews.prompts
        ],
    }


def apply_preview_patch(previews: Previews, patch: dict[str, Any]) -> list[str]:
    """Apply ``every`` / ``rows`` / ``prompts`` ([{slot, enabled}]) of an already validated patch; a line per change.
    For this run only: the config file is untouched, like the menu's changes."""
    by_slot = {slot: prompt for prompt, slot in previews.slots.items() if prompt in previews.prompts}
    toggles = [(item["slot"], item["enabled"]) for item in patch.get("prompts") or []]
    missing = [slot for slot, _ in toggles if slot not in by_slot]
    if missing:  # before anything changes: a patch is applied whole or not at all
        raise ValueError(f"no prompt with slot {', '.join(map(str, missing))}")
    changes: list[str] = []
    if patch.get("every") is not None and patch["every"] != previews.every:
        changes.append(f"every: {previews.every} -> {patch['every']}")
        previews.set_every(patch["every"])
    for label, on in (patch.get("rows") or {}).items():
        if previews.rows.get(label, True) != on:
            changes.append(f"rows: {label} {'on' if on else 'off'}")
            previews.set_row(label, on)
    for slot, on in toggles:
        prompt = by_slot[slot]
        if (prompt in previews.enabled_prompts) != on:
            changes.append(f"{'enabled ' if on else 'disabled'} [{slot:02d}] {prompt}")
            previews.set_enabled(prompt, on)
    if changes:
        previews._write_listing()
        logger.info("[sample] preview settings changed by the control API (this run only):\n  %s", "\n  ".join(changes))
    return changes
