"""
LoRA network injection for MiniMax H3 Transformer blocks.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any, cast

import torch
from torch import nn

from maltrainer.models.minimax_h3.lora_assistant import AssistantAttacher, read_adapter_pairs
from maltrainer.models.minimax_h3.lora_inject import inject_lora_layers
from maltrainer.models.minimax_h3.lora_layers import GridAdaLNLoRA, MiniMaxH3LoRALayer

logger = logging.getLogger(__name__)

__all__ = ["GridAdaLNLoRA", "MiniMaxH3LoRALayer", "MiniMaxH3LoRANetwork"]


class MiniMaxH3LoRANetwork(nn.Module):
    """LoRA network adapter for MiniMax H3 Transformer."""

    def __init__(
        self,
        transformer: nn.Module,
        rank: int = 16,
        alpha: float = 16.0,
        target_modules: list[str] | None = None,
        prefix: str = "diffusion_model.",
        include_adaln: bool = False,
        include_token_refiner: bool = True,
        block_range: tuple[int, int] | None = None,
    ):
        super().__init__()
        self.transformer = transformer
        self.rank = rank
        self.alpha = alpha
        self.target_modules = target_modules or ["qkv_proj", "out_proj", "fc1", "fc2"]
        self.prefix = prefix
        self.include_adaln = include_adaln
        self.include_token_refiner = include_token_refiner
        # Inclusive [first, last] DiT block indices that get adapters; None = all 50.
        self.block_range = tuple(block_range) if block_range is not None else None

        self.lora_layers = nn.ModuleDict()
        self.name_map: dict[str, str] = {}
        self._inject_lora()

    def _inject_lora(self) -> None:
        """Inject LoRA layers into the target linears of MiniMaxH3TransformerBlock (see ``inject_lora_layers``)."""
        inject_lora_layers(
            self.transformer,
            self._add_lora_layer,
            logger,
            rank=self.rank,
            alpha=self.alpha,
            target_modules=self.target_modules,
            include_adaln=self.include_adaln,
            include_token_refiner=self.include_token_refiner,
            block_range=self.block_range,
        )

    def _add_lora_layer(self, name: str, module: nn.Linear) -> None:
        safe_name = name.replace(".", "_")
        self.lora_layers[safe_name] = lora_layer = MiniMaxH3LoRALayer(module, self.rank, self.alpha)
        self.name_map[safe_name] = name
        self._wrap_forward(module, lora_layer)

    @staticmethod
    def _wrap_forward(module: nn.Linear, lora: nn.Module) -> None:
        """Add ``lora(x)`` to a module's output, chaining over any existing wrapper.

        Wrapping composes, so a frozen assistant adapter and the trainable adapter can both
        sit on the same Linear: each adds its own delta to the same input, and the order
        they were attached in does not change the sum.
        """
        original_forward = module.forward

        def forward(x: torch.Tensor, _orig: Any = original_forward, _lora: nn.Module = lora) -> torch.Tensor:
            base = _orig(x)
            delta = _lora(x)
            if delta.dtype == base.dtype:
                return cast(torch.Tensor, base + delta)
            # Sum in the delta's (higher) precision, then round once, so a small fp32
            # update is not truncated before it is even added.
            return cast(torch.Tensor, (base + delta).to(base.dtype))

        cast(Any, module).forward = forward

    def load_assistant_adapter(
        self,
        path: str,
        *,
        multiplier: float = 1.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        adaln_grid_path: str | None = None,
    ) -> int:
        """Attach a frozen de-distillation ("training") adapter alongside the trainable LoRA.

        H3 is guidance-distilled, and training against it directly degrades that
        distillation. ai-toolkit's mitigation is to run a pre-trained adapter LIVE during
        training -- never merged (the base is quantized, so a merge would resample every
        INT8 scale) and never trained. It shapes the distribution the trainable LoRA sees,
        and is absent at inference, so the LoRA learns under deployment-like conditions.

        Kept in a separate ModuleDict from the trainable adapters, so it is excluded from
        ``trainable_parameters()`` and from the saved checkpoint automatically.

        Accepts PEFT (``lora_A`` / ``lora_B``) and Kohya (``lora_down`` / ``lora_up``,
        ``lora_unet_blocks_0_attn_qkv_proj``) naming. Adapters trained on the FULL model may
        carry AdaLN rows that read the full time embedding (2688 wide); the pruned base only
        has an 8-wide curve table, so those rows need ``adaln_grid_path`` -- the full model's
        silu(t_emb) precomputed on the table's grid (Fizgig ships it as
        ``assets/h3_silu_temb_grid.safetensors``). Without it they are skipped with a warning.
        """
        pairs = read_adapter_pairs(path)

        if not hasattr(self, "assistant_layers"):
            self.assistant_layers = nn.ModuleDict()

        attacher = AssistantAttacher(self.transformer, self.prefix, multiplier, device, dtype, adaln_grid_path)
        for name, parts in sorted(pairs.items()):
            if "down" not in parts or "up" not in parts:
                continue
            module_path, module = attacher.resolve(name)
            if module is None:
                continue
            layer = attacher.build(module_path, module, parts)
            if layer is None:
                continue

            self.assistant_layers[module_path.replace(".", "_")] = layer
            self._wrap_forward(module, layer)
            attacher.attached += 1

        attacher.report(path, logger)
        return attacher.attached

    def set_lora_dtype(self, device: torch.device | str, dtype: torch.dtype) -> None:
        """Move/cast only the adapters, leaving the frozen base untouched.

        ``self.transformer`` is a submodule, so ``network.to(dtype=...)`` would also
        downcast the base -- including the tensors H3 deliberately keeps in fp32 (the AdaLN
        curve table and projections, the patch projection, the final video/audio heads).
        """
        self.lora_layers.to(device=device, dtype=dtype)

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Only the adapter parameters, for the optimizer and gradient clipping.

        ``parameters()`` also yields the frozen transformer's tensors. They carry no grads so
        a step skips them, but handing them to an adaptive optimizer means its bookkeeping
        walks tens of billions of frozen elements for nothing.
        """
        return self.lora_layers.parameters()

    def get_state_dict(self, save_dtype: torch.dtype | None = torch.bfloat16) -> dict[str, torch.Tensor]:
        """Get LoRA state dict for saving in Kohya/AI-Toolkit safetensors format.

        Adapters train in fp32 so small updates survive accumulation, but the finished delta
        is stored in ``save_dtype`` (bf16 by default). At rank 64 across H3's dimensions the
        adapters are ~310M parameters, so fp32 files would be ~1.2 GB each. Pass None to
        keep the training dtype.
        """
        state_dict = {}
        for safe_name, module in self.lora_layers.items():
            lora_layer = cast(MiniMaxH3LoRALayer, module)
            name = self.name_map[safe_name]
            full_name = f"{self.prefix}{name}"

            def _store(tensor: torch.Tensor) -> torch.Tensor:
                tensor = tensor.detach().cpu()
                return tensor if save_dtype is None else tensor.to(save_dtype)

            state_dict[f"{full_name}.lora_A.weight"] = _store(lora_layer.lora_down.weight)
            state_dict[f"{full_name}.lora_B.weight"] = _store(lora_layer.lora_up.weight)
            state_dict[f"{full_name}.alpha"] = torch.tensor(lora_layer.alpha, dtype=torch.float32)
        return state_dict

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.transformer(*args, **kwargs)
