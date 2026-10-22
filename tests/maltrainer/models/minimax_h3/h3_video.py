from __future__ import annotations

import torch

from maltrainer.models.minimax_h3.video_vae import MiniMaxH3VideoEncoderModel

MEAN = (0.5, -1.0)
STD = (2.0, 4.0)


def tiny_video_encoder(seed: int = 0) -> MiniMaxH3VideoEncoderModel:
    torch.manual_seed(seed)
    model = MiniMaxH3VideoEncoderModel(
        latent_channels=2,
        block_out_channels=(4, 4, 8),
        layers_per_block=1,
        spatial_downsample_factors=(2, 2, 2),
        temporal_downsample_factors=(1, 2, 2),
        norm_num_groups=2,
        latents_mean=MEAN,
        latents_std=STD,
    )
    return model.eval().requires_grad_(False)


def clip(frames: int, size: int = 16, seed: int = 0) -> torch.Tensor:
    return torch.rand(1, 3, frames, size, size, generator=torch.Generator().manual_seed(seed))


def hf_encoder_key(key: str) -> str:
    """Inverse of the loader's key map: our module names back to the released checkpoint's."""
    if key.startswith("quant_conv."):
        return key
    head, _, rest = key.partition(".")
    if head == "down_blocks":
        level, _, sub = rest.partition(".")
        sub = sub.replace("resnets.", "block.", 1).replace("conv_shortcut.", "nin_shortcut.", 1)
        sub = sub.replace("downsamplers.0.", "downsample.", 1)
        return f"encoder.down.{level}.{sub}"
    return f"encoder.{key}"
