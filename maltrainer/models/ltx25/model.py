"""
LTX-2.5 audio-video DiT, written for LoRA training.

Module names match ComfyUI's single file (``comfy/ldm/lightricks/av_model.py``) exactly, so it loads with
``strict=True`` and LoRA keys come out as ``diffusion_model.transformer_blocks.N.attn1.to_q`` etc.

- 48 blocks over a video stream (4096 wide, one token per 128-channel latent pixel) and an audio stream (2048 wide,
  one token per audio latent frame). Each block: self-attention, text cross-attention, audio <-> video
  cross-attention, feed-forward, all modulated by the stream's timestep (AdaLN-single).
- the text arrives as the Gemma features the text-encoder file projects (video and audio halves); the DiT's own
  1D connectors turn them into the contexts.
- without audio (``audio`` is None, or the model built with ``with_audio=False``) it is a video model.

Conventions (ComfyUI ``ModelType.FLUX``): the DiT gets sigma in [0, 1], multiplies it by ``timestep_scale``
(1000) and predicts the velocity noise - x0 for both streams.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from maltrainer.models.ltx25.blocks import BasicAVTransformerBlock, Stream
from maltrainer.models.ltx25.conditioning import build_streams, positions
from maltrainer.models.ltx25.config import AUDIO_BINS, AUDIO_CHANNELS, LTX25Config
from maltrainer.models.ltx25.connector import Embeddings1DConnector
from maltrainer.models.ltx25.layers import AdaLayerNormSingle


class LTX25Transformer(nn.Module):
    def __init__(self, config: LTX25Config | None = None, with_audio: bool = True):
        super().__init__()
        self.config = config = config or LTX25Config()
        self.with_audio = with_audio
        v, a, ada = config.inner_dim, config.audio_inner_dim, config.ada_params
        self.patchify_proj = nn.Linear(config.in_channels, v)
        self.adaln_single = AdaLayerNormSingle(v, ada)
        if config.cross_attention_adaln and config.prompt_adaln:
            self.prompt_adaln_single = AdaLayerNormSingle(v, 2)
        self.video_embeddings_connector = Embeddings1DConnector(
            config.video_connector, config.theta, config.split_rope, config.double_precision_rope
        )
        self.transformer_blocks = nn.ModuleList(
            [BasicAVTransformerBlock(config, with_audio) for _ in range(config.num_layers)]
        )
        self.scale_shift_table = nn.Parameter(torch.zeros(2, v))
        self.norm_out = nn.LayerNorm(v, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(v, config.in_channels)
        if with_audio:
            self._build_audio(config, a, ada)
        self.gradient_checkpointing = False
        self.compute_dtype = torch.bfloat16

    def _build_audio(self, config: LTX25Config, a: int, ada: int) -> None:
        v = config.inner_dim
        self.audio_patchify_proj = nn.Linear(config.audio_in_channels, a)
        self.audio_adaln_single = AdaLayerNormSingle(a, ada)
        if config.cross_attention_adaln and config.prompt_adaln:
            self.audio_prompt_adaln_single = AdaLayerNormSingle(a, 2)
        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(v, 4)
        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(v, 1)
        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(a, 4)
        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(a, 1)
        self.audio_embeddings_connector = Embeddings1DConnector(
            config.audio_connector, config.theta, config.split_rope, config.double_precision_rope
        )
        self.audio_scale_shift_table = nn.Parameter(torch.zeros(2, a))
        self.audio_norm_out = nn.LayerNorm(a, elementwise_affine=False, eps=1e-6)
        self.audio_proj_out = nn.Linear(a, config.audio_in_channels)

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    # ------------------------------------------------------------------ conditioning

    def prompt_contexts(self, features: torch.Tensor, with_audio: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
        """[B, N, video_dim + audio_dim] Gemma features -> the video and audio contexts (connectors applied)."""
        config = self.config
        video, audio = features[..., : config.cross_attention_dim], features[..., config.cross_attention_dim :]
        context = self.video_embeddings_connector(video)
        if not with_audio:
            return context, None
        return context, self.audio_embeddings_connector(audio)

    # ------------------------------------------------------------------ forward

    def _run(
        self, block: nn.Module, vx: torch.Tensor, ax: torch.Tensor | None, video: Stream, audio: Stream | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.gradient_checkpointing and torch.is_grad_enabled():
            return cast(
                tuple[torch.Tensor, torch.Tensor | None], checkpoint(block, vx, ax, video, audio, use_reentrant=False)
            )
        return cast(tuple[torch.Tensor, torch.Tensor | None], block(vx, ax, video, audio))

    def forward(
        self,
        video: torch.Tensor,
        sigma: torch.Tensor,
        features: torch.Tensor,
        frame_rate: float = 24.0,
        audio: torch.Tensor | None = None,
        audio_sigma: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """video [B, 128, F, H, W], sigma [B] (or [B, F * H * W]), features [B, N, video_dim + audio_dim],
        audio [B, 8, Ta, 16] and audio_sigma [B] -> velocity of the video and (if given) of the audio."""
        dtype = self.compute_dtype
        batch, channels, frames, height, width = video.shape
        use_audio = audio is not None and self.with_audio
        vx = self.patchify_proj(video.flatten(2).transpose(1, 2).to(dtype))
        ax = None
        if use_audio:
            tokens = cast(torch.Tensor, audio).permute(0, 2, 1, 3).flatten(2)
            ax = self.audio_patchify_proj(tokens.to(dtype))
        contexts = self.prompt_contexts(features.to(dtype), use_audio)
        pes = positions(
            self.config, frames, height, width, ax.shape[1] if ax is not None else None, frame_rate, video.device
        )
        a_sigma = (sigma if audio_sigma is None else audio_sigma) if use_audio else None
        video_stream, audio_stream, v_embedded, a_embedded = build_streams(self, sigma, a_sigma, contexts, pes)
        for block in self.transformer_blocks:
            vx, ax = self._run(block, vx, ax, video_stream, audio_stream)
        shift, scale = (self.scale_shift_table[None, None].to(v_embedded.dtype) + v_embedded[:, :, None]).unbind(dim=2)
        out = self.proj_out(self.norm_out(vx) * (1 + scale) + shift)
        out = out.transpose(1, 2).reshape(batch, channels, frames, height, width)
        if ax is None or a_embedded is None:
            return out, None
        a_shift, a_scale = (
            self.audio_scale_shift_table[None, None].to(a_embedded.dtype) + a_embedded[:, :, None]
        ).unbind(dim=2)
        audio_out = self.audio_proj_out(self.audio_norm_out(ax) * (1 + a_scale) + a_shift)
        return out, audio_out.reshape(batch, -1, AUDIO_CHANNELS, AUDIO_BINS).transpose(1, 2)
