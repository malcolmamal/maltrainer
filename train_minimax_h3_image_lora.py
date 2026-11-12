#!/usr/bin/env python3
"""MiniMax H3 Image LoRA training for Maltrainer: H3's transformer without audio, distilled to 6 CFG-free steps with
one output head per step. Trains stills at the checkpoint's own 6 noise levels, each through its own head."""

from __future__ import annotations

from typing import Any

import torch

from maltrainer.models.minimax_h3_image.grid import PDD_SIGMAS
from maltrainer.trainers.minimax_h3.sample import H3Sample
from maltrainer.trainers.minimax_h3_image.levels import clean_share, draw_level
from maltrainer.trainers.minimax_h3_image.model import load_image_transformer
from maltrainer.trainers.minimax_h3_image.step import training_step
from maltrainer.training.cli import start_run, training_arguments
from maltrainer.utils.config import load_config
from train_minimax_h3_lora import MiniMaxH3LoRATrainer, _say


class MiniMaxH3ImageTrainer(MiniMaxH3LoRATrainer):
    """Trainer for MiniMax H3 Image LoRA models (the H3 trainer's caching, loop, adapters and saving)."""

    checkpoint_prefix = "minimax_h3_image_lora"

    def __init__(self, config: dict[str, Any], dry_run: bool = False, dry_run_step_time: float = 0.025):
        config["dataset"] = {**config.get("dataset", {}), "media": ["image"]}  # no audio and no clips here
        super().__init__(config, dry_run, dry_run_step_time)
        self.run_metadata["model_type"] = "minimax_h3_image"
        self.sigmas: tuple[float, ...] = PDD_SIGMAS
        self._steps_taken = 0

    def _train_audio(self) -> bool:
        return False

    def _do_guidance_loss(self) -> bool:
        return False  # distilled without CFG: there is no unconditional prediction to push away from

    def load_transformer_and_lora(self) -> None:
        print("=== Step 2/2: Loading MiniMax H3 Image INT8 Transformer Base ===", flush=True)
        loaded = load_image_transformer(self.config, (self.device, self.dtype), self.run_metadata, _say)
        self.transformer, self.lora_network, self.lora_dtype, self.sigmas = loaded
        print("Transformer and LoRA network ready on GPU!", flush=True)

    def _epoch(self) -> int:
        """The 1-based epoch the next step belongs to."""
        return self._steps_taken // max(len(self._epoch_order()), 1) + 1

    def _draw_level(self) -> int:
        likeness = bool(self.config["train"].get("likeness_opt", False))
        return draw_level(self.sigmas[:-1], clean_share(self._epoch(), likeness))

    def _resolve_flow_shift(self, latent_h: int, latent_w: int) -> float:  # noqa: ARG002 - the loop's signature
        return 1.0  # sigmas are the checkpoint's levels, not drawn from a shifted distribution

    def _schedule_note(self, flow_shift: float, latent_h: int, latent_w: int) -> str:  # noqa: ARG002 - the loop's signature
        levels = ", ".join(f"{sigma:.3f}" for sigma in self.sigmas[:-1])
        mode = "likeness optimization" if self.config["train"].get("likeness_opt") else "an even draw"
        return f"Timestep schedule: the checkpoint's levels ({levels}), {mode}; latent {latent_h}x{latent_w}"

    def _forward_image(
        self, video_patches: Any, text_embeds: Any, sigma: float, layout: dict[str, Any], level: int
    ) -> Any:
        return self.lora_network(
            video_hidden_states=video_patches,
            encoder_hidden_states=text_embeds,
            timestep=torch.tensor([1.0 - sigma], device=self.device),
            timestep_indices=layout["timestep_indices"],
            token_tags=layout["token_tags"],
            position_ids=layout["position_ids"],
            video_indices=layout["video_indices"],
            text_indices=layout["text_indices"],
            level=level,
        )

    def _training_step(
        self,
        sample: H3Sample,
        flow_shift: float,  # noqa: ARG002 - the loop's signature
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss, parts = training_step(self, sample)
        self._steps_taken += 1
        return loss, parts


def main() -> None:
    args = training_arguments("Train MiniMax H3 Image LoRA in Maltrainer").parse_args()
    config = load_config(args.config)
    start_run(config)
    MiniMaxH3ImageTrainer(config, dry_run=args.dry_run, dry_run_step_time=args.dry_run_step_time).run()


if __name__ == "__main__":
    main()
