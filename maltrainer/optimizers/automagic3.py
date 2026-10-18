"""
Automagic v3 — Ostris's self-adjusting optimizer, vendored from AI-Toolkit / Fizgig (MIT License).
"""

from typing import List
import torch


class Automagic3(torch.optim.Optimizer):
    """
    Automagic v3.

    A single learning rate is kept per param group (typically: one lr for
    the whole run). The control principle: the lr RISES while elements hold
    a decisively consistent update direction at the current step size, FALLS
    while their signs decisively alternate (the overshoot signature: weights
    hopping across a minimum flip sign step to step -- shrinking the step is
    what makes a trajectory reappear at a finer scale), and HOLDS on
    everything in between, which is treated as noise.

    Each element keeps a window of its last H (= ``polarity_history``,
    default 4 or 8) update sign bits ("is the update positive", 1-bit packed) --
    H/8 bytes per element, the only per-element optimizer state.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-6,
        min_lr: float = 1e-8,
        max_lr: float = 1e3,
        beta2: float = 0.999,
        eps: float = 1e-30,
        clip_threshold: float = 1.0,
        weight_decay: float = 0.0,
        polarity_history: int = 8,
        fused: bool = False,
    ):
        if min_lr > max_lr:
            raise ValueError(
                f"min_lr ({min_lr}) must be <= max_lr ({max_lr})"
            )
        if lr > 1e-3:
            print(
                f"Note: start lr {lr} is high; the controller will correct it "
                f"(the pooled vote will walk it down)."
            )
        defaults = dict(
            lr=lr,
            min_lr=min_lr,
            max_lr=max_lr,
            beta2=beta2,
            eps=eps,
            clip_threshold=clip_threshold,
            weight_decay=weight_decay,
            polarity_history=max(2, min(64, int(polarity_history))),
        )
        super().__init__(params, defaults)

        self.fused = fused
        # Multiplies every update's step size (not the learned rates): regularisation images train
        # at a fraction of the step (maltrainer/finetune/regularisation.py). 1 = off.
        self.lr_scale = 1.0
        # Training-health statistics, summed on the device (no sync per step) and read with
        # pop_stats(): each group's vote signal, and how many gradient values were NaN / inf
        # (they are zeroed before the update, so without this count they would pass unseen).
        self._signal_sum: dict = {}
        self._signal_steps: dict = {}
        self._nonfinite = None
        self._rebuild_group_index()
        self._hook_handles = []
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                if self.fused:
                    # Fused: update each param the moment its grad is ready.
                    handle = p.register_post_accumulate_grad_hook(
                        self._make_backward_hook(group)
                    )
                    self._hook_handles.append(handle)
                elif p.dtype != torch.float32:
                    # Non-fused: the actual update happens in .step(); here we
                    # only stochastically accumulate low-precision grads across
                    # micro-batches so repeated round-to-nearest doesn't drop
                    # small grads (fp32 grads accumulate losslessly on their own).
                    handle = p.register_post_accumulate_grad_hook(
                        self._make_accum_hook()
                    )
                    self._hook_handles.append(handle)

        total = sum(p.numel() for g in self.param_groups for p in g["params"])
        print(f"Total training parameters: {total:,}")

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _rms(t: torch.Tensor) -> torch.Tensor:
        # Root-mean-square of a tensor; used to size the trust-region clip.
        return t.norm(2) / (t.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        # Adafactor's factored second moment.
        r = (row / row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c = col.unsqueeze(-2).rsqrt()
        return torch.mul(r, c)

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

    _PACK_CONSTS: dict = {}

    @classmethod
    def _pack_consts(cls, device):
        consts = cls._PACK_CONSTS.get(device)
        if consts is None:
            consts = (
                torch.tensor(
                    [1, 2, 4, 8, 16, 32, 64, 128], device=device, dtype=torch.uint8
                ),
                torch.tensor(
                    [0, 1, 2, 3, 4, 5, 6, 7], device=device, dtype=torch.uint8
                ),
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
    def _vote_masks(cls, hist: torch.Tensor, start: int, numel: int):
        """(up, down) 0/1 masks from the packed sign history, oldest row at ``start``.

        up: the sign was the same in all H steps. down: it flipped at every step. Both are
        bitwise ANDs over the packed rows (1 bit per element), so only the two results get
        unpacked -- the same masks as unpacking all H rows to a byte per bit and counting,
        at a fraction of the memory traffic (that was ~0.6 s per step on 5B parameters).
        """
        H = hist.shape[0]
        rows = [hist[(start + i) % H] for i in range(H)]
        all_set = rows[0].clone()
        any_set = rows[0].clone()
        alternating = rows[0] ^ rows[1]
        for prev, cur in zip(rows[1:], rows[2:]):
            alternating.bitwise_and_(prev ^ cur)
        for row in rows[1:]:
            all_set.bitwise_and_(row)
            any_set.bitwise_or_(row)
        up = all_set.bitwise_or_(any_set.bitwise_not_())
        return cls._unpack_bits(up, numel), cls._unpack_bits(alternating, numel)

    def _rebuild_group_index(self) -> None:
        self._param_group_index = {
            p: gi for gi, group in enumerate(self.param_groups) for p in group["params"]
        }
        self._group_num: List = [None] * len(self.param_groups)
        self._group_den: List = [None] * len(self.param_groups)

    @classmethod
    def _stochastic_copy_(cls, dst: torch.Tensor, src_fp32: torch.Tensor) -> None:
        if dst.dtype == torch.bfloat16:
            dst.copy_(cls._sr_truncate(src_fp32, 16))
        elif dst.dtype == torch.float16:
            dst.copy_(cls._sr_truncate(src_fp32, 13))
        else:
            dst.copy_(cls._stochastic_round(src_fp32, dst.dtype))

    def _make_accum_hook(self):
        def _hook(p: torch.Tensor):
            if p.grad is None:
                return
            if hasattr(p, "_accum_grad"):
                acc = p._accum_grad.to(torch.float32).add_(p.grad.to(torch.float32))
                self._stochastic_copy_(p._accum_grad, acc)
            else:
                p._accum_grad = p.grad.clone()
            p.grad = None

        return _hook

    def _init_state(self, p: torch.Tensor, group: dict) -> None:
        state = self.state[p]
        state["step"] = 0
        state["lr"] = torch.tensor(
            min(max(float(group["lr"]), group["min_lr"]), group["max_lr"]),
            dtype=torch.float32,
            device=p.device,
        )
        H = group["polarity_history"]
        width = (p.numel() + 7) // 8
        state["sign_history"] = torch.zeros(
            (H, width), dtype=torch.uint8, device=p.device
        )
        state["hist_idx"] = 0
        state["hist_fill"] = 0
        if p.dim() >= 2:
            state["exp_avg_sq_row"] = torch.zeros(
                p.shape[:-1], dtype=p.dtype, device=p.device
            )
            state["exp_avg_sq_col"] = torch.zeros(
                p.shape[:-2] + p.shape[-1:], dtype=p.dtype, device=p.device
            )
        else:
            state["exp_avg_sq"] = torch.zeros(p.shape, dtype=p.dtype, device=p.device)

    def _make_backward_hook(self, group):
        def _hook(p: torch.Tensor):
            self._update_param(p, group)

        return _hook

    # -------------------------------------------------------------- per-param

    @torch.no_grad()
    def _update_param(self, p: torch.Tensor, group: dict) -> None:
        if p.grad is None:
            return
        state = self.state[p]
        if len(state) == 0:
            self._init_state(p, group)

        grad = p.grad
        if grad.is_sparse:
            raise RuntimeError("Automagic3 does not support sparse gradients.")
        if grad.dtype != torch.float32:
            grad = grad.to(torch.float32)

        bad = grad.isfinite().logical_not_().sum()
        self._nonfinite = bad if self._nonfinite is None else self._nonfinite.add_(bad.to(self._nonfinite.device))
        grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

        beta2 = group["beta2"]
        eps = group["eps"]
        sq = grad * grad

        if p.dim() >= 2:
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
            update = self._approx_sq_grad(row, col).mul_(grad)
        else:
            v_state = state["exp_avg_sq"]
            if v_state.dtype == torch.float32:
                v = v_state
                v.mul_(beta2).add_(sq, alpha=1.0 - beta2)
            else:
                v = v_state.to(torch.float32)
                v.mul_(beta2).add_(sq, alpha=1.0 - beta2)
                v_state.copy_(v.to(v_state.dtype))
            update = v.add(eps).rsqrt().mul_(grad)

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
            up, down = self._vote_masks(hist, state["hist_idx"], update.numel())
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

        state["step"] += 1

        wd = group["weight_decay"]

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

        p.grad = None

    # ----------------------------------------------------------- optimizer API

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if not self.fused:
            for group in self.param_groups:
                for p in group["params"]:
                    if not p.requires_grad:
                        continue
                    accum = getattr(p, "_accum_grad", None)
                    if accum is not None:
                        p.grad = accum
                        del p._accum_grad
                    if p.grad is None:
                        continue
                    self._update_param(p, group)
        self._apply_group_votes()
        return loss

    def _apply_group_votes(self) -> None:
        for gi, group in enumerate(self.param_groups):
            num = self._group_num[gi]
            if num is None:
                continue
            den = self._group_den[gi]
            signal = num.div_(den.clamp_(min=1e-30)).clamp_(-1.0, 1.0)
            name = group.get("name", str(gi))
            total = self._signal_sum.get(name)
            self._signal_sum[name] = signal.detach().clone() if total is None else total.add_(signal.to(total.device))
            self._signal_steps[name] = self._signal_steps.get(name, 0) + 1
            factor = torch.exp(signal)
            for p in group["params"]:
                st = self.state.get(p)
                if st is None or "lr" not in st:
                    continue
                lr_t = st["lr"]
                f = factor if factor.device == lr_t.device else factor.to(lr_t.device)
                lr_t.mul_(f).clamp_(min=group["min_lr"], max=group["max_lr"])
            self._group_num[gi] = None
            self._group_den[gi] = None

    def pop_stats(self) -> dict:
        """Statistics since the last call, then reset (one GPU sync).

        signal: per group, the mean of the vote signal in [-1, 1], weighted by update size: the
            share of elements whose update kept one sign over the last ``polarity_history`` steps
            minus the share whose sign flipped every step. Positive = updates keep a direction
            (learning); near 0 = sign noise; negative = overshooting. Computed even when the rate
            is pinned (min_lr = max_lr), where it no longer moves the rate.
        nonfinite: gradient values that were NaN or inf (zeroed before their update).
        """
        signal = {name: float(total) / self._signal_steps[name] for name, total in self._signal_sum.items()}
        nonfinite = int(self._nonfinite) if self._nonfinite is not None else 0
        self._signal_sum, self._signal_steps, self._nonfinite = {}, {}, None
        return {"signal": signal, "nonfinite": nonfinite}

    def get_learning_rates(self) -> List[float]:
        out = []
        for group in self.param_groups:
            lrs = [
                self.state[p]["lr"]
                for p in group["params"]
                if p in self.state and "lr" in self.state[p]
            ]
            out.append(float(torch.stack(lrs).mean()) if lrs else float(group["lr"]))
        return out

    def get_avg_learning_rate(self) -> float:
        lrs = self.get_learning_rates()
        return sum(lrs) / len(lrs) if lrs else float(self.defaults["lr"])

    def load_state_dict(self, state_dict):
        # torch casts floating state to each param's dtype, which would round the fp32 rates
        # of bf16 params (a full finetune) to bf16; put the saved values back afterwards.
        saved_lrs = {
            idx: st["lr"].detach().clone()
            for idx, st in state_dict["state"].items()
            if isinstance(st.get("lr"), torch.Tensor)
        }
        super().load_state_dict(state_dict)
        params = [p for group in self.param_groups for p in group["params"]]
        for idx, lr in saved_lrs.items():
            st = self.state.get(params[int(idx)])
            if st is not None:
                st["lr"] = lr.to(device=params[int(idx)].device, dtype=torch.float32)
        for group in self.param_groups:
            for k, v in self.defaults.items():
                group[k] = v
            lrs = [
                st["lr"]
                for p in group["params"]
                if (st := self.state.get(p)) is not None
                and isinstance(st.get("lr"), torch.Tensor)
            ]
            med = None
            if lrs:
                dev = lrs[0].device
                med = (
                    torch.stack([t.to(torch.float32).to(dev) for t in lrs])
                    .log_()
                    .median()
                    .exp_()
                )
            for p in group["params"]:
                st = self.state.get(p)
                if st is None:
                    continue
                if isinstance(st.get("lr"), torch.Tensor):
                    st["lr"] = st["lr"].to(torch.float32)
                    if med is not None:
                        st["lr"].copy_(med.to(st["lr"].device))
                numel = p.numel()
                H = group["polarity_history"]
                width = (numel + 7) // 8
                sh = st.get("sign_history")
                hist_ok = (
                    isinstance(sh, torch.Tensor)
                    and sh.shape == (H, width)
                    and isinstance(st.get("hist_idx"), int)
                    and 0 <= st["hist_idx"] < H
                    and isinstance(st.get("hist_fill"), int)
                    and 0 <= st["hist_fill"] <= H
                )
                if hist_ok:
                    st["sign_history"] = sh.to(torch.uint8)
                else:
                    st["sign_history"] = torch.zeros(
                        (H, width), dtype=torch.uint8, device=p.device
                    )
                    st["hist_idx"] = 0
                    st["hist_fill"] = 0
        self._rebuild_group_index()
