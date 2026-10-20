"""Choosing the MiniMax H3 linears that get a trainable LoRA layer."""

from __future__ import annotations

import logging
from collections.abc import Callable

from torch import nn


def is_excluded(name: str, include_token_refiner: bool, block_range: tuple[int, ...] | None) -> bool:
    """Whether a module path is left out before the target names are even looked at."""
    if "modulation" in name or "norm" in name:
        return True
    if name.startswith("token_refiner.") and not include_token_refiner:
        return True
    if name.startswith("blocks.") and block_range is not None:
        index = int(name.split(".", 2)[1])
        if not block_range[0] <= index <= block_range[1]:
            return True
    return False


def inject_lora_layers(
    transformer: nn.Module,
    add_layer: Callable[[str, nn.Linear], None],
    logger: logging.Logger,
    *,
    rank: int,
    alpha: float,
    target_modules: list[str],
    include_adaln: bool,
    include_token_refiner: bool,
    block_range: tuple[int, ...] | None,
) -> None:
    """Add a LoRA layer (through ``add_layer``) to every target linear of MiniMaxH3TransformerBlock.

    Targets are matched against the final component of the module path, not as a
    substring of the whole path, so a name like ``video_patch_proj`` can never be
    caught by a ``proj`` style target. With the default target list this selects the
    208 layers of musubi-tuner's ``attn_mlp`` preset: 50 blocks x
    (attn.qkv_proj, attn.out_proj, mlp.fc1, mlp.fc2) plus the 2 token_refiner blocks.

    ``block_range`` and ``include_token_refiner`` narrow that. Fizgig's default trains
    only blocks 20-49 and never the refiner: identity lives in the back of the stack,
    and adapters on the early blocks mostly damage anatomy and colour. With no adapter
    before block 20 nothing upstream needs a gradient, so the backward pass also stops
    there.
    """
    injected = adaln = 0
    for name, module in transformer.named_modules():
        if is_excluded(name, include_token_refiner, block_range):
            continue

        # adaln_proj is checkpoint-dependent, which is why it is a switch rather than a
        # blanket skip. On the FULL bf16 model its up-matrices are [96768, 2688] and soak
        # up most of the LoRA's capacity for weights ComfyUI's pruned inference build then
        # discards. On the PRUNED build it is [96768, 8] -- small, deploy-consistent, and
        # the timestep-conditioned modulation, so excluding it presents from the outside as
        # "the mid/low-noise range never trains" (Fizgig's finding on this checkpoint).
        # Per-block AdaLN only: final_layer.adaln_proj stays out.
        if "adaln_proj" in name:
            if not (include_adaln and name.startswith("blocks.")):
                continue
            if isinstance(module, nn.Linear):
                add_layer(name, module)
                adaln += 1
                injected += 1
            continue

        if name.rsplit(".", 1)[-1] in target_modules and isinstance(module, nn.Linear):
            add_layer(name, module)
            injected += 1

    if injected == 0:
        raise ValueError("No MiniMax H3 modules matched the LoRA targets / block_range")
    logger.info(
        "Injected %d LoRA layers into MiniMax H3 transformer (rank=%d, alpha=%s%s, blocks %s%s)",
        injected,
        rank,
        alpha,
        f", including {adaln} per-block adaln_proj" if adaln else ", adaln_proj excluded",
        f"{block_range[0]}-{block_range[1]}" if block_range else "0-49",
        "" if include_token_refiner else ", token refiner frozen",
    )
