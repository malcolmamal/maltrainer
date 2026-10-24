"""Fibo 2.0 DiT hyper-parameters, read off ComfyUI's single file, and the gist-token rule."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

ADALN_DIM = 256
LATENT_CHANNELS = 128  # the Flux2 VAE's 2x2-folded latent
BUNDLES = 6  # Perceiver bundle + one per injection block
ROPE_AXES = (32, 48, 48)  # plane (gist index or image plane), row, column
ROPE_THETA = 256.0
MIN_GIST, GIST_STEP = 256, 64
GIST_MIN_TEXT, GIST_MAX_TEXT = 256, 2048


@dataclass(frozen=True)
class Fibo2Config:
    dim: int = 3584
    n_heads: int = 28
    n_layers: int = 44
    n_refiner_layers: int = 2
    injection_layers: tuple[int, ...] = (6, 14, 22, 30, 38)
    cap_feat_dim: int = 7680
    perceiver_layers: int = 2
    max_gist_tokens: int = 512
    ffn_dim: int = 9728
    in_channels: int = LATENT_CHANNELS
    norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.dim // self.n_heads != 2 * sum(axis // 2 for axis in ROPE_AXES):
            raise ValueError("the RoPE axes must cover the head dimension")
        if len(self.injection_layers) + 1 != BUNDLES:
            raise ValueError(
                f"Fibo 2.0 feeds {BUNDLES} text bundles, got {len(self.injection_layers)} injection layers"
            )

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> Fibo2Config:
        """The sizes in a ComfyUI Fibo 2.0 file (the Linear weights keep their (out, in) shape, int8 or not)."""

        def count(prefix: str, position: int) -> int:
            return 1 + max(int(key.split(".")[position]) for key in state if key.startswith(prefix))

        max_gist, dim = state["perceiver.latent_queries"].shape
        n_layers = count("layers.", 1)
        return cls(
            dim=dim,
            n_heads=dim // state["layers.0.attention.q_norm.weight"].shape[0],
            n_layers=n_layers,
            n_refiner_layers=count("noise_refiner.", 1),
            injection_layers=tuple(i for i in range(n_layers) if f"layers.{i}.cross_attn_gate" in state),
            cap_feat_dim=state["perceiver.layers.0.cross_attention.to_kv.weight"].shape[1],
            perceiver_layers=count("perceiver.layers.", 2),
            max_gist_tokens=max_gist,
            ffn_dim=state["layers.0.feed_forward.w13.weight"].shape[0] // 2,
            in_channels=state["x_embedder.weight"].shape[1],
        )


def gist_count(text_length: int, config: Fibo2Config) -> int:
    """Gist tokens for ``text_length`` text tokens: 256 up to 256 tokens, rising linearly to the maximum at 2048."""
    ratio = min(1.0, max(0.0, (text_length - GIST_MIN_TEXT) / (GIST_MAX_TEXT - GIST_MIN_TEXT)))
    quantized = math.ceil((MIN_GIST + ratio * (config.max_gist_tokens - MIN_GIST)) / GIST_STEP) * GIST_STEP
    return max(MIN_GIST, min(config.max_gist_tokens, quantized))
