"""
LoRA network injection for MiniMax H3 Transformer blocks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


class GridAdaLNLoRA(nn.Module):
    """A full-model AdaLN LoRA row, run on the pruned base's curve-table AdaLN.

    The full model's adaln_proj reads silu(time_embedder(t)) [2688]; the pruned build replaced
    that with an 8-wide table lerped by t, so the full-width input does not exist at run time.
    Each incoming table row is matched to its nearest grid row and the full model's
    precomputed silu(t_emb) at that point stands in -- half a grid step (1/2048 in t) of error
    at worst, the approach of larryvrh's ComfyUI-MiniMax-H3-Turbo node and of Fizgig.
    """

    def __init__(self, down: torch.Tensor, up: torch.Tensor, scale: float, table: torch.Tensor, grid: torch.Tensor):
        super().__init__()
        self.register_buffer("down", down.detach().clone().float())
        self.register_buffer("up", up.detach().clone().float())
        self.register_buffer("table", table.detach().clone().float())
        self.register_buffer("grid", grid.detach().clone().float())
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        index = torch.cdist(x.detach().float(), self.table.float()).argmin(dim=1)
        full = self.grid[index].to(self.down.dtype)
        return (full @ self.down.T @ self.up.T) * self.scale


class MiniMaxH3LoRANetwork(nn.Module):
    """LoRA network adapter for MiniMax H3 Transformer."""

    def __init__(
        self,
        transformer: nn.Module,
        rank: int = 16,
        alpha: float = 16.0,
        target_modules: Optional[List[str]] = None,
        prefix: str = "diffusion_model.",
        include_adaln: bool = False,
        include_token_refiner: bool = True,
        block_range: Optional[Tuple[int, int]] = None,
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
        self.name_map: Dict[str, str] = {}
        self._inject_lora()

    def _inject_lora(self):
        """Inject LoRA layers into target linear modules of MiniMaxH3TransformerBlock.

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
        injected = 0
        adaln = 0
        for name, module in self.transformer.named_modules():
            if "modulation" in name or "norm" in name:
                continue
            if name.startswith("token_refiner.") and not self.include_token_refiner:
                continue
            if name.startswith("blocks.") and self.block_range is not None:
                index = int(name.split(".", 2)[1])
                if not self.block_range[0] <= index <= self.block_range[1]:
                    continue

            # adaln_proj is checkpoint-dependent, which is why it is a switch rather than a
            # blanket skip. On the FULL bf16 model its up-matrices are [96768, 2688] and soak
            # up most of the LoRA's capacity for weights ComfyUI's pruned inference build then
            # discards. On the PRUNED build it is [96768, 8] -- small, deploy-consistent, and
            # the timestep-conditioned modulation, so excluding it presents from the outside as
            # "the mid/low-noise range never trains" (Fizgig's finding on this checkpoint).
            # Per-block AdaLN only: final_layer.adaln_proj stays out.
            if "adaln_proj" in name:
                if not (self.include_adaln and name.startswith("blocks.")):
                    continue
                if isinstance(module, nn.Linear):
                    self._add_lora_layer(name, module)
                    adaln += 1
                    injected += 1
                continue

            if name.rsplit(".", 1)[-1] in self.target_modules:
                if isinstance(module, nn.Linear):
                    self._add_lora_layer(name, module)
                    injected += 1

        if injected == 0:
            raise ValueError("No MiniMax H3 modules matched the LoRA targets / block_range")
        logger.info(
            "Injected %d LoRA layers into MiniMax H3 transformer (rank=%d, alpha=%s%s, blocks %s%s)",
            injected,
            self.rank,
            self.alpha,
            f", including {adaln} per-block adaln_proj" if adaln else ", adaln_proj excluded",
            "%d-%d" % self.block_range if self.block_range else "0-49",
            "" if self.include_token_refiner else ", token refiner frozen",
        )

    def _add_lora_layer(self, name: str, module: nn.Linear) -> None:
        lora_layer = MiniMaxH3LoRALayer(module, self.rank, self.alpha)
        safe_name = name.replace(".", "_")
        self.lora_layers[safe_name] = lora_layer
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

        def forward(x, _orig=original_forward, _lora=lora):
            base = _orig(x)
            delta = _lora(x)
            if delta.dtype == base.dtype:
                return base + delta
            # Sum in the delta's (higher) precision, then round once, so a small fp32
            # update is not truncated before it is even added.
            return (base + delta).to(base.dtype)

        module.forward = forward

    def load_assistant_adapter(
        self,
        path: str,
        *,
        multiplier: float = 1.0,
        device=None,
        dtype: Optional[torch.dtype] = None,
        adaln_grid_path: Optional[str] = None,
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
        from safetensors.torch import load_file

        state = load_file(str(path))
        pairs: Dict[str, Dict[str, torch.Tensor]] = {}
        for key, value in state.items():
            for suffix, slot in (
                (".lora_A.weight", "down"),
                (".lora_down.weight", "down"),
                (".lora_B.weight", "up"),
                (".lora_up.weight", "up"),
                (".alpha", "alpha"),
            ):
                if key.endswith(suffix):
                    pairs.setdefault(key[: -len(suffix)], {})[slot] = value
                    break

        if not hasattr(self, "assistant_layers"):
            self.assistant_layers = nn.ModuleDict()

        kohya_names = {
            "lora_unet_" + name.replace(".", "_"): name
            for name, module in self.transformer.named_modules()
            if isinstance(module, nn.Linear)
        }
        table = getattr(self.transformer, "adaln_t_table", None)
        grid = None

        attached, grid_rows, skipped, unbridged = 0, 0, [], []
        rank = 0
        for name, parts in sorted(pairs.items()):
            if "down" not in parts or "up" not in parts:
                continue
            module_path = name[len(self.prefix) :] if name.startswith(self.prefix) else name
            module_path = kohya_names.get(module_path, module_path)
            try:
                module = self.transformer.get_submodule(module_path)
            except AttributeError:
                skipped.append(module_path)
                continue
            if not isinstance(module, nn.Linear):
                skipped.append(module_path)
                continue

            down, up = parts["down"], parts["up"]
            rank = down.shape[0]
            alpha = float(parts["alpha"].item()) if "alpha" in parts else float(rank)
            scale = (alpha / rank) * multiplier

            if down.shape[1] != module.in_features:
                # A full-model AdaLN row on the pruned base (see the docstring).
                if "adaln_proj" not in module_path or table is None:
                    skipped.append(f"{module_path} (in {down.shape[1]} != {module.in_features})")
                    continue
                if adaln_grid_path is None:
                    unbridged.append(module_path)
                    continue
                if grid is None:
                    grid = load_file(str(adaln_grid_path))["silu_t_emb_grid"].float()
                    if grid.shape[0] != table.shape[0]:
                        raise ValueError(
                            f"AdaLN grid has {grid.shape[0]} rows but this checkpoint's curve table has "
                            f"{table.shape[0]}; they must come from the same H3 build."
                        )
                layer = GridAdaLNLoRA(down, up, scale, table, grid)
                grid_rows += 1
            else:
                layer = MiniMaxH3LoRALayer(module, rank, alpha)
                layer.lora_down.weight.data.copy_(down.to(layer.lora_down.weight.dtype))
                layer.lora_up.weight.data.copy_(up.to(layer.lora_up.weight.dtype))
                layer.scale = scale
            layer.requires_grad_(False)
            layer.eval()
            if isinstance(layer, GridAdaLNLoRA):
                layer.to(device=device)  # fp32: the nearest-row match needs the table exact
            elif dtype is not None or device is not None:
                layer.to(device=device, dtype=dtype)

            self.assistant_layers[module_path.replace(".", "_")] = layer
            self._wrap_forward(module, layer)
            attached += 1

        if skipped:
            raise ValueError(
                f"Training adapter targets {len(skipped)} module(s) this transformer does not "
                f"expose; examples: {skipped[:5]}. Is it built for a different H3 checkpoint?"
            )
        if unbridged:
            logger.warning(
                "Training adapter has %d full-model AdaLN row(s) the pruned base cannot run without "
                "model.training_adapter_adaln_grid (Fizgig's assets/h3_silu_temb_grid.safetensors); "
                "they were skipped, so the adapter acts only through its other %d modules.",
                len(unbridged),
                attached,
            )
        logger.info(
            "Attached frozen training adapter: %d modules (%d AdaLN via the t_emb grid), rank %d, "
            "multiplier %.2f (from %s)",
            attached,
            grid_rows,
            rank,
            multiplier,
            Path(path).name,
        )
        return attached

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
