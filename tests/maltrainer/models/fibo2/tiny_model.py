"""A Fibo 2.0 DiT small enough for CPU tests: the head dimension stays 128 (the three RoPE axes), so 2 heads."""

from __future__ import annotations

import torch

from maltrainer.models.fibo2.config import BUNDLES, LATENT_CHANNELS, Fibo2Config
from maltrainer.models.fibo2.model import Fibo2Transformer

CAP = 16
TINY = Fibo2Config(
    dim=256,
    n_heads=2,
    n_layers=6,
    n_refiner_layers=1,
    injection_layers=(1, 2, 3, 4, 5),
    cap_feat_dim=CAP,
    perceiver_layers=1,
    max_gist_tokens=256,
    ffn_dim=32,
)


def tiny_transformer(config: Fibo2Config = TINY, seed: int = 0) -> Fibo2Transformer:
    """Random small weights (the real zero-initialised gates and queries would hide most of the network)."""
    torch.manual_seed(seed)
    model = Fibo2Transformer(config)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(
                torch.ones_like(parameter) if name.endswith("norm.weight") else torch.randn_like(parameter) * 0.05
            )
    model.compute_dtype = torch.float32
    return model.eval()


def tiny_inputs(h: int = 2, w: int = 3, length: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(1, LATENT_CHANNELS, h, w, generator=generator)
    context = torch.randn(1, length, BUNDLES * CAP, generator=generator)
    return x, torch.tensor([0.4]), context
