#!/usr/bin/env python3
"""
LTX-2.5 (Lightricks, 22B audio-video DiT) LoRA training for maltrainer.

Trains the dev model on ComfyUI's single files: the DiT (ltx-2.5-22b-dev-transformer, INT8 ConvRot or bf16), the video
and audio VAEs and the Gemma 4 12B text encoder with its projection. Stills train as one-frame clips, videos as clips of
8k + 1 frames with their soundtrack. Latents and caption features are cached once, then the encoders are unloaded before
the DiT loads. A clip with sound trains the audio stream too; without it (or with ``dataset.audio: false``) only video.

Conventions follow ComfyUI's LTXAV (ModelType.FLUX) and the official LTX trainer. The DiT gets sigma in [0, 1] and
predicts the velocity noise - x0 of both streams; sigma comes from a logit-normal shifted by the video token count.

    python train_ltx25_lora.py --config templates/ltx25_template.yaml
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from maltrainer.data.minimax_dataset import MediaSource, count_media_files, scan_media_sources
from maltrainer.models.ltx25.loading import load_ltx25_transformer
from maltrainer.models.ltx25.lora import LTX25LoRANetwork
from maltrainer.models.ltx25.text_encoder import load_ltx25_text_encoder
from maltrainer.trainers.ltx25.cache import cache_audio, cache_video
from maltrainer.trainers.ltx25.data import MediaSettings
from maltrainer.trainers.ltx25.step import ltx_loss
from maltrainer.training.cli import start_run, training_arguments
from maltrainer.training.image_lora import ImageLoRATrainer
from maltrainer.training.session import free_memory
from maltrainer.utils.config import load_config

logger = logging.getLogger(__name__)

KINDS = ["image", "clip"]


class LTX25LoRATrainer(ImageLoRATrainer):
    model_type = "ltx25"
    label = "LTX-2.5"
    default_name = "ltx25_lora"
    batch_one_reason = (
        "LTX-2.5 trains at batch_size 1 (clips differ in size and length); "
        "use train.gradient_accumulation_steps for a larger effective batch."
    )
    polarity_history = 16
    key_commands = False

    def __init__(self, config: dict[str, Any], dry_run: bool = False, dry_run_step_time: float = 0.025) -> None:
        super().__init__(config, dry_run, dry_run_step_time)
        self.audio_cache: dict[str, torch.Tensor] = {}
        self.with_audio = False
        self.fps = float(config.get("dataset", {}).get("fps", 24))

    # ------------------------------------------------------------------ data

    def _sources(self) -> list[MediaSource]:
        dataset_config = self.config["dataset"]
        return scan_media_sources(
            dataset_config["path"], KINDS, dataset_config.get("trigger"), dataset_config.get("default_caption", "")
        )

    def _dry_run_samples(self) -> int:
        dataset_config = self.config["dataset"]
        repeats = max(int(dataset_config.get("repeats", 1) or 1), 1)
        return max(count_media_files(dataset_config.get("path", ""), KINDS), 1) * repeats

    def precache_dataset(self) -> None:
        model_config, dataset_config = self.config["model"], self.config["dataset"]
        sources = self._sources()
        if not sources:
            raise ValueError(f"No images or videos found under {dataset_config['path']}")
        print("=== Step 1/2: Pre-caching Dataset (Video & Audio Latents, Text Features) ===", flush=True)
        settings = MediaSettings(
            int(dataset_config.get("resolution", 512)),
            self.fps,
            int(dataset_config.get("clip_frames", 49)),
            bool(dataset_config.get("audio", True)),
        )
        cache, soundtracks = cache_video(sources, model_config, settings, self.device, self.is_main_process)
        cache_audio(cache, soundtracks, model_config, self.fps, self.device, self.is_main_process)
        self.latent_cache, self.audio_cache = cache.video, cache.audio
        self.samples = [(str(s.path), s.caption) for s in sources if str(s.path) in cache.video]
        self.with_audio = bool(cache.audio)
        print("Latents pre-cached and the VAEs unloaded from VRAM.", flush=True)

        prompts = {caption for _, caption in self.samples}
        if float(self.config.get("train", {}).get("caption_dropout", 0.0) or 0.0) > 0:
            prompts.add("")
        encoder = load_ltx25_text_encoder(model_config["text_encoder_path"], self.device, self.dtype)
        self._encode_prompts(encoder, prompts)
        del encoder
        free_memory()
        print("Text features pre-cached and the text encoder unloaded from VRAM.", flush=True)
        logger.info(
            "Pre-cached %d samples (%d with sound) and %d prompts.",
            len(self.samples),
            len(self.audio_cache),
            len(self.text_cache),
        )

    # ------------------------------------------------------------------ model

    def load_transformer_and_lora(self) -> None:
        print("=== Step 2/2: Loading LTX-2.5 DiT ===", flush=True)
        lora_config, train_config = self.config.get("lora", {}), self.config.get("train", {})
        self.transformer, load_info = load_ltx25_transformer(
            self.config["model"]["transformer_path"],
            self.device,
            self.dtype,
            self.with_audio,
            str(train_config.get("int8_grad_precision", "bf16")),
        )
        self.run_metadata["base_load"] = load_info
        if bool(train_config.get("gradient_checkpointing", True)):
            self.transformer.enable_gradient_checkpointing()
        targets = str(lora_config.get("targets", "auto"))
        if targets == "auto":
            targets = "attn" if self.with_audio else "video"
        self.lora_network = LTX25LoRANetwork(
            self.transformer,
            rank=int(lora_config.get("rank", 32)),
            alpha=float(lora_config.get("alpha", lora_config.get("rank", 32))),
            targets=targets,
            prefix=lora_config.get("prefix", "diffusion_model."),
        )
        self._lora_ready()

    # ------------------------------------------------------------------ training

    def _training_step(self, index: int) -> torch.Tensor:
        # caption dropout trains on the empty prompt, which encodes as just the BOS token
        key, caption = self._sample(index)
        video = self.latent_cache[key].to(self.device, torch.float32).unsqueeze(0)
        audio = self.audio_cache.get(key)
        features = self.text_cache[caption].to(self.device, self.dtype).unsqueeze(0)
        return ltx_loss(
            self.lora_network,
            self.config["train"],
            video,
            features,
            None if audio is None else audio.to(self.device, torch.float32).unsqueeze(0),
            self.fps,
            self.dtype,
        )


def main() -> None:
    args = training_arguments("Train an LTX-2.5 LoRA in maltrainer").parse_args()
    config = load_config(args.config)
    start_run(config)
    LTX25LoRATrainer(config, dry_run=args.dry_run, dry_run_step_time=args.dry_run_step_time).run()


if __name__ == "__main__":
    main()
