"""
Automagic v3: restoring the optimizer state from a checkpoint (see ``automagic3``).
"""

from __future__ import annotations

from typing import Any

import torch

from maltrainer.optimizers.automagic3_base import AutomagicState


class StateRestore(AutomagicState):
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
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
        for group, built in zip(self.param_groups, self._built, strict=False):
            group.update(built)
            lrs = [
                st["lr"]
                for p in group["params"]
                if (st := self.state.get(p)) is not None and isinstance(st.get("lr"), torch.Tensor)
            ]
            med = None
            if lrs:
                dev = lrs[0].device
                med = torch.stack([t.to(torch.float32).to(dev) for t in lrs]).median()
            for p in group["params"]:
                st = self.state.get(p)
                if st is None:
                    continue
                if isinstance(st.get("lr"), torch.Tensor):
                    st["lr"] = st["lr"].to(torch.float32)
                    if med is not None:
                        # clamped to this run's min_lr / max_lr, which the config may have changed
                        st["lr"].copy_(med.to(st["lr"].device)).clamp_(min=group["min_lr"], max=group["max_lr"])
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
                    st["sign_history"] = torch.zeros((H, width), dtype=torch.uint8, device=p.device)
                    st["hist_idx"] = 0
                    st["hist_fill"] = 0
        self._rebuild_group_index()
