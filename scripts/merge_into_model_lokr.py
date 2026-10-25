"""LoKr (LyCORIS Kronecker) modules for scripts/merge_lora_into_model.py, applied as ComfyUI applies them
(comfy/weight_adapter/lokr.py, ``LoKrAdapter.calculate_weight``):

    w1 = lokr_w1                      or  lokr_w1_a @ lokr_w1_b    (then rank = lokr_w1_b rows)
    w2 = lokr_w2                      or  lokr_w2_a @ lokr_w2_b    (then rank = lokr_w2_b rows)
    W' = W + strength * scale * kron(w1, w2).reshape(W.shape)

with scale = alpha / rank when either factor is stored as a low-rank pair (the w2 rank when both are), and 1.0 when
both are stored whole -- alpha is then ignored, as in ComfyUI. The convolutional form (``lokr_t2``) and DoRA scales are
refused: no Krea 2 weight is a convolution, and a DoRA rescale is not an additive delta.
"""

from __future__ import annotations

from typing import Any, cast

import torch

PARTS = (".lokr_w1", ".lokr_w2", ".lokr_w1_a", ".lokr_w1_b", ".lokr_w2_a", ".lokr_w2_b")


def _factor(tensors: dict[str, torch.Tensor], module: str, which: str) -> tuple[Any, int | None]:
    """(the whole factor, or its (a, b) pair) and the pair's rank (None when stored whole)."""
    whole = tensors.get(f"{module}.lokr_{which}")
    if whole is not None:
        return whole, None
    a, b = tensors.get(f"{module}.lokr_{which}_a"), tensors.get(f"{module}.lokr_{which}_b")
    if a is None or b is None:
        raise ValueError(f"{module}: has neither lokr_{which} nor both lokr_{which}_a and lokr_{which}_b")
    return (a, b), int(b.shape[0])


def read_lokr(
    handle: Any, keys: list[str], strength: float, name: str
) -> tuple[dict[str, tuple[float, Any, Any]], set[str]]:
    """module -> (strength * scale, w1 or (a, b), w2 or (a, b)), and the keys they used (alphas included)."""
    modules = sorted({k[: -len(s)] for k in keys for s in PARTS if k.endswith(s)})
    if not modules:
        return {}, set()
    out, used = {}, set()
    for module in modules:
        mine = [k for k in keys if k.startswith(module + ".") and k.count(".") == module.count(".") + 1]
        refused = [k for k in mine if k.endswith((".lokr_t2", ".dora_scale"))]
        if refused:
            raise ValueError(f"{name}: {refused[0]} is a convolutional or DoRA LoKr this merge does not apply")
        tensors = {k: handle.get_tensor(k) for k in mine}
        w1, rank1 = _factor(tensors, module, "w1")
        w2, rank2 = _factor(tensors, module, "w2")
        rank = rank2 if rank2 is not None else rank1
        alpha = tensors.get(f"{module}.alpha")
        scale = float(alpha.item()) / rank if alpha is not None and rank is not None else 1.0
        out[module] = (strength * scale, w1, w2)
        used.update(mine)
    return out, used


def _matrix(factor: Any, device: str) -> torch.Tensor:
    if isinstance(factor, tuple):
        return cast(torch.Tensor, factor[0].to(device).float() @ factor[1].to(device).float())
    return cast(torch.Tensor, factor.to(device).float())


def kron_delta(part: tuple[float, Any, Any], shape: torch.Size, key: str, device: str) -> torch.Tensor:
    """strength * scale * kron(w1, w2), in the weight's shape."""
    factor, w1, w2 = part
    product = torch.kron(_matrix(w1, device), _matrix(w2, device))
    if product.numel() != shape.numel():
        raise ValueError(f"{key}: the LoKr product is {tuple(product.shape)}, the base weight {tuple(shape)}")
    return factor * product.reshape(shape)
