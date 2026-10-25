"""
Fibo 2.0 DiT, written for LoRA training (text-to-image; the edit mode's reference latents are not trained).

Module names match ComfyUI's ``fibo2_dev_bf16.safetensors`` exactly, so it loads with ``strict=True`` and LoRA keys
come out as ``diffusion_model.layers.N.attention.qkv`` etc., which ComfyUI applies without conversion.

Port of diffusers' ``BriaFibo2Transformer2DModel`` (via ComfyUI's comfy/ldm/fibo2 and the ai-toolkit port):
- the image is 128-channel Flux2 latents, one token per latent pixel (no further patching)
- a Perceiver turns the first of six text bundles into 256..512 gist tokens; the sequence is ``[gist | image]``
- 2 noise-refiner blocks run on the image alone, then 44 blocks on the whole sequence. Five of them (6, 14, 22, 30,
  38) also cross-attend to their own text bundle.
- the gist tokens are clean (t = 0 modulation); the image has the real t
- RoPE ids are (plane, row, column): gist token i on plane i + 1, the image on plane ``max_gist_tokens + 1``

Conventions: the DiT gets sigma in [0, 1] (multiplied by 1000 inside) and predicts v = noise - x0.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from maltrainer.models.fibo2.blocks import Block, Perceiver, Span
from maltrainer.models.fibo2.config import Fibo2Config, gist_count
from maltrainer.models.fibo2.layers import FinalLayer, TimestepEmbedder
from maltrainer.models.fibo2.rope import gist_ids, grid_ids, rope_angles

TIME_SCALE = 1000.0


class Fibo2Transformer(nn.Module):
    def __init__(self, config: Fibo2Config | None = None):
        super().__init__()
        self.config = config = config or Fibo2Config()
        self.x_embedder = nn.Linear(config.in_channels, config.dim)
        self.t_embedder = TimestepEmbedder()
        self.noise_refiner = nn.ModuleList([Block(config, False) for _ in range(config.n_refiner_layers)])
        self.perceiver = Perceiver(config)
        self.layers = nn.ModuleList([Block(config, i in config.injection_layers) for i in range(config.n_layers)])
        self.final_layer = FinalLayer(config.dim, config.in_channels)
        self.gradient_checkpointing = False
        self.compute_dtype = torch.bfloat16

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def _run(self, module: nn.Module, *args: object) -> torch.Tensor:
        if self.gradient_checkpointing and torch.is_grad_enabled():
            return cast(torch.Tensor, checkpoint(module, *args, use_reentrant=False))
        return cast(torch.Tensor, module(*args))

    def _gist(self, bundle: torch.Tensor, count: int) -> torch.Tensor:
        gist = self.perceiver.latent_queries[:count].to(bundle.dtype).unsqueeze(0).expand(bundle.shape[0], -1, -1)
        for layer in self.perceiver.layers:
            gist = self._run(layer, gist, bundle)
        return cast(torch.Tensor, self.perceiver.norm_out(gist))

    def forward(self, x: torch.Tensor, sigma: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """x [1, 128, H, W] noisy latent, sigma [1] in [0, 1], context [1, T, 6 x 7680] -> v [1, 128, H, W]."""
        B, C, H, W = x.shape
        if B != 1:
            raise ValueError("Fibo 2.0 trains at batch size 1: every caption has its own gist-token count")
        config, dtype, device = self.config, self.compute_dtype, x.device
        t = sigma.to(device, torch.float32).reshape(-1) * TIME_SCALE
        adaln_noisy = self.t_embedder(t)
        adaln_both = torch.cat((adaln_noisy, self.t_embedder(torch.zeros_like(t))))

        image = self.x_embedder(x.flatten(2).transpose(1, 2).to(dtype))
        image_rope = rope_angles(grid_ids(config.max_gist_tokens + 1, H, W, device))
        for block in self.noise_refiner:
            image = self._run(block, image, image_rope, adaln_noisy, None)

        bundles = context.to(dtype).chunk(len(config.injection_layers) + 1, dim=-1)
        count = gist_count(context.shape[1], config)
        sequence = torch.cat((self._gist(bundles[0], count), image), dim=1)
        gist_rope = rope_angles(gist_ids(count, device))
        rope = (torch.cat((gist_rope[0], image_rope[0]), dim=1), torch.cat((gist_rope[1], image_rope[1]), dim=1))
        span: Span = (count, count + H * W)

        text = dict(zip(config.injection_layers, bundles[1:], strict=True))
        for index, block in enumerate(self.layers):
            sequence = self._run(block, sequence, rope, adaln_both, span, text.get(index))
        out = self.final_layer(sequence[:, span[0] : span[1]], adaln_noisy)
        return cast(torch.Tensor, out.transpose(1, 2).reshape(B, C, H, W))
