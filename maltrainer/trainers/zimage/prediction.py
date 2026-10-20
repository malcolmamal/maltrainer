"""Z-Image's transformer output as one [B, C, H, W] prediction, whatever form diffusers returned it in."""

from __future__ import annotations

from typing import Any

import torch


def prediction(output: Any, latents: torch.Tensor, stack: bool) -> torch.Tensor:
    """The output's tensor: ``.sample``, a tuple's first item, or the output itself. A per-sample list is
    stacked when ``stack`` (else its first item is taken); a frame axis is squeezed; a [C, B, H, W] mix-up
    is put back to [B, C, H, W]."""
    if hasattr(output, "sample"):
        pred = output.sample
    elif isinstance(output, (list, tuple)):
        pred = output[0]
    else:
        pred = output

    # If it's still a list (some models return [tensor]), take the first element
    if isinstance(pred, (list, tuple)):
        pred = torch.stack(list(pred)) if stack and pred[0].ndim == 4 else pred[0]

    # Squeeze the frame dimension if it exists
    if len(pred.shape) == 5:
        pred = pred.squeeze(2)

    # Fix potential channel/batch swap to avoid broadcasting
    if (
        pred.ndim == 4
        and pred.shape[0] == latents.shape[1]
        and pred.shape[1] == latents.shape[0]
        and pred.shape[2:] == latents.shape[2:]
    ):
        pred = pred.permute(1, 0, 2, 3).contiguous()
    tensor: torch.Tensor = pred
    return tensor
