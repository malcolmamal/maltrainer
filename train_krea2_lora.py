#!/usr/bin/env python3
"""
Krea 2 LoRA training for maltrainer.

Trains on ComfyUI's single files: the RAW DiT (fp8_scaled, or bf16 quantized to fp8 on
load), the Qwen-Image / Wan 2.1 VAE and the Qwen3-VL-4B text encoder. Latents and caption
embeddings are cached once, then the encoders are unloaded before the DiT loads.

Conventions follow ComfyUI's Krea 2 implementation (comfy/ldm/krea2, ModelType.FLUX): the DiT
gets sigma in [0, 1] and predicts v = noise - x0.

    python train_krea2_lora.py --config templates/krea2_template.yaml
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from tqdm import tqdm

from maltrainer.data.krea2_dataset import load_krea2_image
from maltrainer.models.krea2.loading import load_krea2_transformer
from maltrainer.models.krea2.lora import Krea2LoRANetwork
from maltrainer.models.krea2.text_encoder import load_krea2_text_encoder
from maltrainer.models.krea2.timesteps import sample_sigma
from maltrainer.models.krea2.vae import load_krea2_vae
from maltrainer.training.cli import start_run, training_arguments
from maltrainer.training.dop import check_dop_config, dop_captions, dop_loss
from maltrainer.training.image_lora import ImageLoRATrainer, free_memory
from maltrainer.utils.config import load_config

logger = logging.getLogger(__name__)


class Krea2LoRATrainer(ImageLoRATrainer):
    model_type = "krea2"
    label = "Krea 2"
    default_name = "krea2_lora"
    batch_one_reason = (
        "Krea 2 trains at batch_size 1 (every caption keeps its own unpadded length, as in "
        "ComfyUI); use train.gradient_accumulation_steps for a larger effective batch."
    )
    # Fizgig: a 16-step sign window, because at 8 the rate "crawled from 1e-6 to 4.3e-6 across a whole
    # epoch" on batch-1 Krea 2 gradients.
    polarity_history = 16

    def __init__(self, config: dict[str, Any], dry_run: bool = False, dry_run_step_time: float = 0.025):
        check_dop_config(config)
        super().__init__(config, dry_run=dry_run, dry_run_step_time=dry_run_step_time)
        self.dop = bool(config.get("train", {}).get("dop", False))
        self.dop_captions: dict[str, str] = {}  # sample key -> its caption with the trigger replaced
        self.micro_step = 0
        self.last_dop_loss: float | None = None
        self._shift_logged = [False]

    # ------------------------------------------------------------------ caching

    def precache_dataset(self) -> None:
        model_config = self.config["model"]
        dataset_config = self.config["dataset"]
        sources = self._sources()
        if not sources:
            raise ValueError(f"No images found under {dataset_config['path']}")

        print("=== Step 1/2: Pre-caching Dataset (VAE Latents & Text Embeddings) ===", flush=True)
        resolution = int(dataset_config.get("resolution", 512))
        aspect_mode = str(dataset_config.get("aspect_mode", "bucket")).lower()

        vae = load_krea2_vae(model_config["vae_path"], self.device)
        for source in tqdm(sources, desc="Caching latents", disable=not self.is_main_process):
            try:
                pixels = load_krea2_image(source.path, resolution, aspect_mode).unsqueeze(0).to(self.device)
                self.latent_cache[str(source.path)] = vae.encode(pixels).cpu()
                self.samples.append((str(source.path), source.caption))
            except (OSError, ValueError, RuntimeError) as e:
                logger.warning("Skipping %s: %s", source.path.name, e)
        del vae
        free_memory()
        print("VAE latents pre-cached and VAE unloaded from VRAM.", flush=True)

        prompts = {caption for _, caption in self.samples}
        if float(self.config.get("train", {}).get("caption_dropout", 0.0) or 0.0) > 0:
            prompts.add("")
        if self.dop:
            self._plan_dop_captions()
            prompts.update(self.dop_captions.values())
        encoder = load_krea2_text_encoder(
            model_config["text_encoder_path"], self.device, self.dtype, model_config.get("tokenizer_path")
        )
        self._encode_prompts(encoder, prompts)
        del encoder
        free_memory()
        print("Text embeddings pre-cached and Text Encoder unloaded from VRAM.", flush=True)
        logger.info("Pre-cached %d images and %d prompt embeddings.", len(self.samples), len(self.text_cache))

    def _plan_dop_captions(self) -> None:
        """DOP's caption per sample: the trigger replaced by the class word, as ai-toolkit does."""
        train_config = self.config["train"]
        cls = str(train_config["dop_class"]).strip()
        self.dop_captions = dop_captions(self.samples, str(self.config["dataset"]["trigger"]).strip(), cls)
        self.run_metadata["dop"] = {
            "class": cls,
            "multiplier": float(train_config.get("dop_multiplier", 1.0)),
            "every": int(train_config.get("dop_every", 1) or 1),
            "captions": len(self.dop_captions),
        }

    # ------------------------------------------------------------------ model

    def load_transformer_and_lora(self) -> None:
        print("=== Step 2/2: Loading Krea 2 DiT ===", flush=True)
        model_config = self.config["model"]
        lora_config = self.config.get("lora", {})
        self.transformer, load_info = load_krea2_transformer(model_config["transformer_path"], self.device, self.dtype)
        self.run_metadata["base_load"] = load_info
        self.transformer.enable_gradient_checkpointing()

        self.lora_network = Krea2LoRANetwork(
            self.transformer,
            rank=int(lora_config.get("rank", 16)),
            alpha=float(lora_config.get("alpha", lora_config.get("rank", 16))),
            targets=str(lora_config.get("targets", "attn-mlp")),
            block_range=lora_config.get("block_range"),
            prefix=lora_config.get("prefix", "diffusion_model."),
        )
        self._lora_ready()

    # ------------------------------------------------------------------ training

    def _sample_sigma(self, image_tokens: int) -> tuple[torch.Tensor, float]:
        """Video-free flow-matching sigma for one sample, plus the shift used (maltrainer.models.krea2.timesteps)."""
        return sample_sigma(self.config["train"], image_tokens, self.device)

    def _training_step(
        self, index: int, shift_logged: list[bool] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The flow-matching loss for one sample, and the noised latent and sigma it used. ``shift_logged``:
        whether the sigma shift was logged yet (the run's own flag when not given)."""
        logged = self._shift_logged if shift_logged is None else shift_logged
        train_config = self.config["train"]
        key, caption = self._sample(index)
        x0 = self.latent_cache[key].to(self.device, torch.float32)
        context = self.text_cache[caption].to(self.device, self.dtype).unsqueeze(0)

        tokens = (x0.shape[-2] // 2) * (x0.shape[-1] // 2)
        sigma, shift = self._sample_sigma(tokens)
        if not logged[0]:
            logged[0] = True
            self.run_metadata["resolved_sigma_shift"] = round(shift, 6)
            distribution = train_config.get("timestep_distribution", "krea2_shift")
            logger.info("Timestep schedule: %s, sigma shift %.3f at %d image tokens", distribution, shift, tokens)

        noise = torch.randn_like(x0)
        s = sigma.view(1, 1, 1, 1)
        noisy = ((1.0 - s) * x0 + s * noise).to(self.dtype)
        pred = self.lora_network(noisy, sigma, context)
        loss = torch.nn.functional.mse_loss(pred.float(), (noise - x0).float())
        strength = float(train_config.get("loss_weight_strength", 1.0))
        return (loss * strength if strength != 1.0 else loss), noisy, sigma

    def _dop_loss(self, index: int, noisy: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor | None:
        """DOP (maltrainer.training.dop) on the same noised latent and sigma; None when this sample's caption
        has no trigger, or it isn't a DOP step (``train.dop_every``)."""
        train_config = self.config["train"]
        caption = self.dop_captions.get(self.samples[index][0])
        every = max(int(train_config.get("dop_every", 1) or 1), 1)
        if caption is None or self.micro_step % every:
            return None
        context = self.text_cache[caption].to(self.device, self.dtype).unsqueeze(0)
        adapters = self.accelerator.unwrap_model(self.lora_network)
        multiplier = float(train_config.get("dop_multiplier", 1.0))
        return dop_loss(self.lora_network, adapters, (noisy, sigma, context), multiplier)

    def _micro_step(self, index: int) -> torch.Tensor:
        loss, noisy, sigma = self._training_step(index, self._shift_logged)
        self.accelerator.backward(loss)
        if self.dop:
            # its own backward, so the first pass's graph is freed before the second
            dop = self._dop_loss(index, noisy, sigma)
            if dop is not None:
                self.accelerator.backward(dop)
                self.last_dop_loss = dop.item()
        self.micro_step += 1
        return loss

    def _step_note(self) -> str:
        return f"DOP: {self.last_dop_loss:.6f}, " if self.last_dop_loss is not None else ""


def main() -> None:
    args = training_arguments("Train a Krea 2 LoRA in maltrainer").parse_args()
    config = load_config(args.config)
    start_run(config)
    Krea2LoRATrainer(config, dry_run=args.dry_run, dry_run_step_time=args.dry_run_step_time).run()


if __name__ == "__main__":
    main()
