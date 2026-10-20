"""
Automagic v3 helpers that need no optimizer state: the packed sign history and stochastic rounding.

Mixed into ``Automagic3`` (see ``automagic3``), which keeps every method name.
"""

from __future__ import annotations

import itertools
from typing import ClassVar

import torch


class SignBits:
    """1-bit packing of the per-element update signs, and the up / down votes read off them."""

    _PACK_CONSTS: ClassVar[dict[torch.device, tuple[torch.Tensor, torch.Tensor]]] = {}

    @classmethod
    def _pack_consts(cls, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        consts = cls._PACK_CONSTS.get(device)
        if consts is None:
            consts = (
                torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=device, dtype=torch.uint8),
                torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device=device, dtype=torch.uint8),
            )
            cls._PACK_CONSTS[device] = consts
        return consts

    @classmethod
    def _pack_bits(cls, bits: torch.Tensor) -> torch.Tensor:
        weights, _ = cls._pack_consts(bits.device)
        flat = bits.reshape(-1).to(torch.uint8)
        pad = (-flat.numel()) % 8
        if pad:
            flat = torch.cat([flat, flat.new_zeros(pad)])
        return (flat.view(-1, 8) * weights).sum(-1, dtype=torch.uint8)

    @classmethod
    def _unpack_bits(cls, packed: torch.Tensor, numel: int) -> torch.Tensor:
        _, shifts = cls._pack_consts(packed.device)
        vals = (packed.unsqueeze(-1) >> shifts).bitwise_and_(1)
        return vals.view(-1)[:numel]

    @classmethod
    def _vote_masks(cls, hist: torch.Tensor, start: int, numel: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(up, down) 0/1 masks from the packed sign history, oldest row at ``start``.

        up: the sign was the same in all H steps. down: it flipped at every step. Both are
        bitwise ANDs over the packed rows (1 bit per element), so only the two results get
        unpacked -- the same masks as unpacking all H rows to a byte per bit and counting,
        at a fraction of the memory traffic (that was ~0.6 s per step on 5B parameters).
        """
        H = hist.shape[0]
        rows = [hist[(start + i) % H] for i in range(H)]
        all_set, any_set = rows[0].clone(), rows[0].clone()
        alternating = rows[0] ^ rows[1]
        for prev, cur in itertools.pairwise(rows[1:]):
            alternating.bitwise_and_(prev ^ cur)
        for row in rows[1:]:
            all_set.bitwise_and_(row)
            any_set.bitwise_or_(row)
        up = all_set.bitwise_or_(any_set.bitwise_not_())
        return cls._unpack_bits(up, numel), cls._unpack_bits(alternating, numel)


class StochasticRounding:
    """Writing an fp32 value into a low-precision tensor without a systematic round-to-nearest bias."""

    @staticmethod
    def _sr_truncate(v_fp32: torch.Tensor, drop_bits: int) -> torch.Tensor:
        as_int = v_fp32.view(torch.int32)
        as_int.add_(torch.randint_like(as_int, 1 << drop_bits))
        as_int.bitwise_and_(-(1 << drop_bits))
        return v_fp32

    @staticmethod
    def _stochastic_round(v: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        finfo = torch.finfo(dtype)
        absv = v.abs().clamp_(min=finfo.tiny)
        ulp = torch.exp2(torch.floor(torch.log2(absv))).mul_(finfo.eps)
        noise = torch.rand_like(v).sub_(0.5).mul_(ulp)
        return v.add_(noise).to(dtype)

    @classmethod
    def _stochastic_copy_(cls, dst: torch.Tensor, src_fp32: torch.Tensor) -> None:
        if dst.dtype == torch.bfloat16:
            dst.copy_(cls._sr_truncate(src_fp32, 16))
        elif dst.dtype == torch.float16:
            dst.copy_(cls._sr_truncate(src_fp32, 13))
        else:
            dst.copy_(cls._stochastic_round(src_fp32, dst.dtype))
