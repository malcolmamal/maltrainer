"""Which parameters a full finetune trains."""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

from torch import nn


def select_trainable(
    model: nn.Module,
    prefixes: Optional[Iterable[str]] = None,
    linears_only: bool = False,
) -> List[Tuple[str, nn.Parameter]]:
    """Unfreeze the parameters whose name starts with one of ``prefixes`` (none = all of them).

    ``linears_only``: only nn.Linear weights train. Biases, norm scales and other vectors stay
    frozen, as in Fizgig's finetune: they are few parameters, but each one moves a whole channel
    of every token, so they drift the model's look far faster than a matrix entry does.
    """
    prefixes = tuple(prefixes or ())
    model.requires_grad_(False)
    allowed = None
    if linears_only:
        allowed = {f"{name}.weight" for name, module in model.named_modules() if isinstance(module, nn.Linear)}
    selected = []
    for name, param in model.named_parameters():
        if prefixes and not name.startswith(prefixes):
            continue
        if allowed is not None and name not in allowed:
            continue
        param.requires_grad_(True)
        selected.append((name, param))
    if not selected:
        raise ValueError(
            f"finetune.trainable_prefixes {list(prefixes)}"
            f"{' with linears_only' if linears_only else ''} matched no parameter"
        )
    return selected
