"""
LoRA network injection for MiniMax H3 Transformer blocks.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MiniMaxH3LoRALayer(nn.Module):
    """Single LoRA layer for Linear projection in MiniMax H3."""

    def __init__(self, original_module: nn.Linear, rank: int, alpha: float):
        super().__init__()
        in_features = original_module.in_features
        out_features = original_module.out_features

        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Adapters are held in higher precision than the frozen base (fp32 on a bf16/INT8
        # base) so that optimizer updates accumulate; the matmuls themselves do not need it.
        # Under autocast the linear would recast to bf16 anyway, so upcasting the activation
        # there would only materialize a large fp32 tensor for nothing.
        target_dtype = self.lora_down.weight.dtype
        if x.is_floating_point() and x.dtype != target_dtype and not torch.is_autocast_enabled():
            x = x.to(target_dtype)
        return self.lora_up(self.lora_down(x)) * self.scale


class MiniMaxH3LoRANetwork(nn.Module):
    """LoRA network adapter for MiniMax H3 Transformer."""

    def __init__(
        self,
        transformer: nn.Module,
        rank: int = 16,
        alpha: float = 16.0,
        target_modules: Optional[List[str]] = None,
        prefix: str = "diffusion_model.",
    ):
        super().__init__()
        self.transformer = transformer
        self.rank = rank
        self.alpha = alpha
        self.target_modules = target_modules or ["qkv_proj", "out_proj", "fc1", "fc2"]
        self.prefix = prefix

        self.lora_layers = nn.ModuleDict()
        self.name_map: Dict[str, str] = {}
        self._inject_lora()

    def _inject_lora(self):
        """Inject LoRA layers into target linear modules of MiniMaxH3TransformerBlock.

        Targets are matched against the final component of the module path, not as a
        substring of the whole path, so a name like ``video_patch_proj`` can never be
        caught by a ``proj`` style target. With the default target list this selects the
        208 layers of musubi-tuner's ``attn_mlp`` preset: 50 blocks x
        (attn.qkv_proj, attn.out_proj, mlp.fc1, mlp.fc2) plus the 2 token_refiner blocks.
        """
        injected = 0
        for name, module in self.transformer.named_modules():
            # Skip AdaLN, modulation, or norm projections
            if "adaln_proj" in name or "modulation" in name or "norm" in name:
                continue

            if name.rsplit(".", 1)[-1] in self.target_modules:
                if isinstance(module, nn.Linear):
                    lora_layer = MiniMaxH3LoRALayer(module, self.rank, self.alpha)
                    safe_name = name.replace(".", "_")
                    self.lora_layers[safe_name] = lora_layer
                    self.name_map[safe_name] = name

                    original_forward = module.forward

                    def make_forward(orig_fwd, lora):
                        def forward(x):
                            base = orig_fwd(x)
                            delta = lora(x)
                            if delta.dtype == base.dtype:
                                return base + delta
                            # Sum in the delta's (higher) precision, then round once, so a
                            # small fp32 update is not truncated before it is even added.
                            return (base + delta).to(base.dtype)
                        return forward

                    module.forward = make_forward(original_forward, lora_layer)
                    injected += 1

        logger.info("Injected %d LoRA layers into MiniMax H3 transformer (rank=%d, alpha=%s)", injected, self.rank, self.alpha)

    def set_lora_dtype(self, device, dtype: torch.dtype) -> None:
        """Move/cast only the adapters, leaving the frozen base untouched.

        ``self.transformer`` is a submodule, so ``network.to(dtype=...)`` would also
        downcast the base -- including the tensors H3 deliberately keeps in fp32 (the AdaLN
        curve table and projections, the patch projection, the final video/audio heads).
        """
        self.lora_layers.to(device=device, dtype=dtype)

    def trainable_parameters(self):
        """Only the adapter parameters, for the optimizer and gradient clipping.

        ``parameters()`` also yields the frozen transformer's tensors. They carry no grads so
        a step skips them, but handing them to an adaptive optimizer means its bookkeeping
        walks tens of billions of frozen elements for nothing.
        """
        return self.lora_layers.parameters()

    def get_state_dict(self, save_dtype: Optional[torch.dtype] = torch.bfloat16) -> Dict[str, torch.Tensor]:
        """Get LoRA state dict for saving in Kohya/AI-Toolkit safetensors format.

        Adapters train in fp32 so small updates survive accumulation, but the finished delta
        is stored in ``save_dtype`` (bf16 by default). At rank 64 across H3's dimensions the
        adapters are ~310M parameters, so fp32 files would be ~1.2 GB each. Pass None to
        keep the training dtype.
        """
        state_dict = {}
        for safe_name, lora_layer in self.lora_layers.items():
            name = self.name_map[safe_name]
            full_name = f"{self.prefix}{name}"

            def _store(tensor: torch.Tensor) -> torch.Tensor:
                tensor = tensor.detach().cpu()
                return tensor if save_dtype is None else tensor.to(save_dtype)

            state_dict[f"{full_name}.lora_A.weight"] = _store(lora_layer.lora_down.weight)
            state_dict[f"{full_name}.lora_B.weight"] = _store(lora_layer.lora_up.weight)
            state_dict[f"{full_name}.alpha"] = torch.tensor(lora_layer.alpha, dtype=torch.float32)
        return state_dict

    def forward(self, *args, **kwargs):
        return self.transformer(*args, **kwargs)
