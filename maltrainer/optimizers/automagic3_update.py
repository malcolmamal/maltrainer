"""
Automagic v3: the update of one parameter, split into the steps ``Automagic3._update_param`` runs in order.

Mixed into ``Automagic3`` (see ``automagic3``). The statements and their order are the original's.
"""

from __future__ import annotations

from typing import Any, cast

import torch

from maltrainer.optimizers.automagic3_base import AutomagicState
from maltrainer.optimizers.automagic3_ops import SignBits, StochasticRounding


class ParamUpdate(SignBits, StochasticRounding, AutomagicState):
    @staticmethod
    def _rms(t: torch.Tensor) -> torch.Tensor:
        # Root-mean-square of a tensor; used to size the trust-region clip.
        return cast(torch.Tensor, t.norm(2) / (t.numel() ** 0.5))

    @staticmethod
    def _approx_sq_grad(row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        # Adafactor's factored second moment.
        return torch.mul((row / row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1), col.unsqueeze(-2).rsqrt())

    def _init_state(self, p: torch.Tensor, group: dict[str, Any]) -> None:
        state = self.state[p]
        state["step"] = 0
        state["lr"] = torch.tensor(
            min(max(float(group["lr"]), group["min_lr"]), group["max_lr"]), dtype=torch.float32, device=p.device
        )
        H = group["polarity_history"]
        width = (p.numel() + 7) // 8
        state["sign_history"] = torch.zeros((H, width), dtype=torch.uint8, device=p.device)
        state["hist_idx"] = 0
        state["hist_fill"] = 0
        if p.dim() >= 2:
            state["exp_avg_sq_row"] = torch.zeros(p.shape[:-1], dtype=p.dtype, device=p.device)
            state["exp_avg_sq_col"] = torch.zeros(p.shape[:-2] + p.shape[-1:], dtype=p.dtype, device=p.device)
        else:
            state["exp_avg_sq"] = torch.zeros(p.shape, dtype=p.dtype, device=p.device)

    def _clean_grad(self, p: torch.Tensor) -> torch.Tensor:
        """The fp32 gradient with NaN / inf zeroed (they are counted in ``_nonfinite``)."""
        grad = cast(torch.Tensor, p.grad)
        if grad.is_sparse:
            raise RuntimeError("Automagic3 does not support sparse gradients.")
        if grad.dtype != torch.float32:
            grad = grad.to(torch.float32)

        bad = grad.isfinite().logical_not_().sum()
        self._nonfinite = bad if self._nonfinite is None else self._nonfinite.add_(bad.to(self._nonfinite.device))
        grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        return grad

    @classmethod
    def _factored_update(
        cls, state: dict[str, Any], sq: torch.Tensor, grad: torch.Tensor, beta2: float, eps: float
    ) -> torch.Tensor:
        row_state = state["exp_avg_sq_row"]
        col_state = state["exp_avg_sq_col"]
        if row_state.dtype == torch.float32:
            row, col = row_state, col_state
            row.mul_(beta2).add_(sq.mean(dim=-1).add_(eps), alpha=1.0 - beta2)
            col.mul_(beta2).add_(sq.mean(dim=-2).add_(eps), alpha=1.0 - beta2)
        else:
            row = row_state.to(torch.float32)
            col = col_state.to(torch.float32)
            row.mul_(beta2).add_(sq.mean(dim=-1).add_(eps), alpha=1.0 - beta2)
            col.mul_(beta2).add_(sq.mean(dim=-2).add_(eps), alpha=1.0 - beta2)
            row_state.copy_(row.to(row_state.dtype))
            col_state.copy_(col.to(col_state.dtype))
        return cls._approx_sq_grad(row, col).mul_(grad)

    @staticmethod
    def _vector_update(
        state: dict[str, Any], sq: torch.Tensor, grad: torch.Tensor, beta2: float, eps: float
    ) -> torch.Tensor:
        v_state = state["exp_avg_sq"]
        if v_state.dtype == torch.float32:
            v = v_state
            v.mul_(beta2).add_(sq, alpha=1.0 - beta2)
        else:
            v = v_state.to(torch.float32)
            v.mul_(beta2).add_(sq, alpha=1.0 - beta2)
            v_state.copy_(v.to(v_state.dtype))
        return cast(torch.Tensor, v.add(eps).rsqrt().mul_(grad))

    def _record_votes(self, p: torch.Tensor, hist: torch.Tensor, start: int, update: torch.Tensor) -> None:
        """Add this parameter's vote (up minus down, weighted by update size) to its group's running sums."""
        up, down = self._vote_masks(hist, start, update.numel())
        w = update.abs().view(-1)
        num = (w * up).sum().sub_((w * down).sum())
        den = w.sum()
        gi = self._param_group_index.get(p)
        if gi is not None:
            if self._group_num[gi] is None:
                self._group_num[gi] = num
                self._group_den[gi] = den
            else:
                acc = self._group_num[gi]
                if num.device != acc.device:
                    num = num.to(acc.device)
                    den = den.to(acc.device)
                acc.add_(num)
                self._group_den[gi].add_(den)

    def _apply_update(self, p: torch.Tensor, update: torch.Tensor, lr_t: torch.Tensor, wd: float) -> None:
        if p.dtype == torch.float32:
            if wd != 0.0:
                update.add_(p, alpha=wd)
            p.addcmul_(update, lr_t, value=-self.lr_scale)
        else:
            new_p_fp32 = p.to(torch.float32)
            if wd != 0.0:
                update.add_(new_p_fp32, alpha=wd)
            new_p_fp32.addcmul_(update, lr_t, value=-self.lr_scale)
            self._stochastic_copy_(p, new_p_fp32)

    @torch.no_grad()
    def _update_param(self, p: torch.Tensor, group: dict[str, Any]) -> None:
        if p.grad is None:
            return
        state = self.state[p]
        if len(state) == 0:
            self._init_state(p, group)

        grad = self._clean_grad(p)

        beta2, eps = group["beta2"], group["eps"]
        sq = grad * grad

        if p.dim() >= 2:
            update = self._factored_update(state, sq, grad, beta2, eps)
        else:
            update = self._vector_update(state, sq, grad, beta2, eps)

        update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
        update.clamp_(-group["clip_threshold"], group["clip_threshold"])

        cur_bits = update.gt(0.0)
        hist = state["sign_history"]
        idx = state["hist_idx"]
        H = hist.shape[0]
        lr_t = state["lr"]

        hist[idx].copy_(self._pack_bits(cur_bits))
        state["hist_idx"] = (idx + 1) % H
        fill = min(H, state["hist_fill"] + 1)
        state["hist_fill"] = fill

        if fill == H:
            self._record_votes(p, hist, state["hist_idx"], update)

        state["step"] += 1

        self._apply_update(p, update, lr_t, group["weight_decay"])

        p.grad = None
