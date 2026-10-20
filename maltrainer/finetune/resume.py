"""Whether a saved finetune state belongs to the run that wants to resume it."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def check_same_trainable_set(state: Mapping[str, Any], group_names: Sequence[str], param_names: Sequence[str]) -> None:
    """Refuse a resume whose optimizer state was kept for other parameters than this run trains.

    The optimizer state is stored by position, so a changed ``finetune.trainable_prefixes`` would
    otherwise load one block's state into another's (or fail on a size mismatch deep inside torch).
    The group names are always checked; the parameter names are checked when the state recorded
    them (a state written before they were recorded has none, and is taken as it is).
    """
    saved_groups = state.get("group_names")
    if saved_groups != list(group_names):
        raise ValueError(
            f"The resumed optimizer had groups {saved_groups}, this config builds {list(group_names)}; "
            "keep finetune.trainable_prefixes and train.automagic_param_groups as they were"
        )
    saved = state.get("param_names")
    if saved is None or list(saved) == list(param_names):
        return
    gone, new = sorted(set(saved) - set(param_names)), sorted(set(param_names) - set(saved))
    raise ValueError(
        f"The resumed run trained {len(saved)} parameters, this config trains {len(param_names)} "
        f"(only in the saved state: {gone[:3]}; only now: {new[:3]}); keep finetune.trainable_prefixes "
        "and finetune.linears_only as they were"
    )
