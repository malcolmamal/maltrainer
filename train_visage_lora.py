#!/usr/bin/env python3
"""
Visage-Image (Sherstnev/Visage-Image) LoRA training for maltrainer, 256 px text-to-image faces.

Trains the 237M single-stream DiT on ComfyUI's single files: the DiT (visage-image), the f16 x 96 VAE (visage_vae) and
the Qwen3-0.6B text encoder (qwen3_0.6b, first 8 layers). Latents and caption states are cached once, then the
encoders are unloaded before the DiT loads.

Conventions follow the model's own scripts/train_dit.py: rectified flow x_t = (1 - t) x0 + t noise with the DiT
predicting noise - x0, t = sigmoid(N(0, 1) + log sqrt(tokens / 64)), and caption dropout replacing a caption's states
with zeros (the unconditional branch of its classifier-free guidance), keeping the caption's length.

    python train_visage_lora.py --config templates/visage_template.yaml
"""

from __future__ import annotations

import logging

import torch
from tqdm import tqdm

from maltrainer.control.launch import make_keys
from maltrainer.control.stats import RunView
from maltrainer.data.flux3_dataset import load_flux3_image
from maltrainer.models.visage.loading import load_visage_transformer
from maltrainer.models.visage.lora import VisageLoRANetwork
from maltrainer.models.visage.text_encoder import load_visage_text_encoder
from maltrainer.models.visage.vae import load_visage_vae
from maltrainer.trainers.visage.previews import VisagePreviews
from maltrainer.trainers.visage.timesteps import sample_sigma
from maltrainer.training.cli import start_run, training_arguments
from maltrainer.training.image_lora import ImageLoRATrainer
from maltrainer.training.session import free_memory
from maltrainer.utils.config import load_config
from maltrainer.utils.keyboard import KeyCommands

logger = logging.getLogger(__name__)


class VisageLoRATrainer(VisagePreviews, ImageLoRATrainer):
    model_type = "visage"
    label = "Visage-Image"
    default_name = "visage_lora"
    batch_one_reason = (
        "Visage-Image trains at batch_size 1 (every caption has its own length); "
        "use train.gradient_accumulation_steps for a larger effective batch."
    )
    # attention, MLP and the rest each adapt their own rate, over a 16-step sign window (batch-1 gradients).
    polarity_history = 16
    schedule = ("visage", 1.0)
    last_sigma: float | None = None

    # ------------------------------------------------------------------ caching

    def precache_dataset(self) -> None:
        model_config = self.config["model"]
        dataset_config = self.config["dataset"]
        sources = self._sources()
        if not sources:
            raise ValueError(f"No images found under {dataset_config['path']}")

        print("=== Step 1/2: Pre-caching Dataset (VAE Latents & Text States) ===", flush=True)
        resolution = int(dataset_config.get("resolution", 256))
        aspect_mode = str(dataset_config.get("aspect_mode", "square")).lower()
        vae = load_visage_vae(model_config["vae_path"], self.device)
        for source in tqdm(sources, desc="Caching latents", disable=not self.is_main_process):
            try:
                pixels = load_flux3_image(source.path, resolution, aspect_mode).unsqueeze(0).to(self.device)
                self.latent_cache[str(source.path)] = vae.encode(pixels)[0].cpu()
                self.samples.append((str(source.path), source.caption))
            except (OSError, ValueError, RuntimeError) as e:
                logger.warning("Skipping %s: %s", source.path.name, e)
        del vae
        free_memory()
        print("VAE latents pre-cached and VAE unloaded from VRAM.", flush=True)

        encoder = load_visage_text_encoder(
            model_config["text_encoder_path"], self.device, self.dtype, model_config.get("tokenizer_path")
        )
        self._encode_prompts(encoder, {caption for _, caption in self.samples} | self.preview_texts())
        del encoder
        free_memory()
        print("Text states pre-cached and Text Encoder unloaded from VRAM.", flush=True)
        logger.info("Pre-cached %d images and %d prompt states.", len(self.samples), len(self.text_cache))

    # ------------------------------------------------------------------ model

    def load_transformer_and_lora(self) -> None:
        print("=== Step 2/2: Loading Visage-Image DiT ===", flush=True)
        lora_config = self.config.get("lora", {})
        self.transformer, load_info = load_visage_transformer(
            self.config["model"]["transformer_path"], self.device, self.dtype
        )
        self.run_metadata["base_load"] = load_info
        self.lora_network = VisageLoRANetwork(
            self.transformer,
            rank=int(lora_config.get("rank", 16)),
            alpha=float(lora_config.get("alpha", lora_config.get("rank", 16))),
            targets=str(lora_config.get("targets", "blocks")),
            prefix=lora_config.get("prefix", "diffusion_model."),
        )
        self._lora_ready()

    # ------------------------------------------------------------------ keys

    def _make_keys(self) -> KeyCommands:
        """s / q / p / h, and the preview keys of the ``sample`` section (w, m, g ...)."""
        return make_keys(KeyCommands, self.config, RunView(self), extra=self.preview_keys())

    def _after_log(self, step: int, total: int) -> None:
        del total
        self.current_step = step

    # ------------------------------------------------------------------ training

    def _training_step(self, index: int) -> torch.Tensor:
        key, caption = self.samples[index]
        x0 = self.latent_cache[key].to(self.device, torch.float32).unsqueeze(0)
        text = self.text_cache[caption].to(self.device, self.dtype).unsqueeze(0)
        dropout = float(self.config["train"].get("caption_dropout", 0.0) or 0.0)
        if dropout > 0 and torch.rand(()).item() < dropout:
            text = torch.zeros_like(text)

        sigma = sample_sigma(self.config["train"], x0.shape[2] * x0.shape[3], self.device)
        self.last_sigma = sigma.item()
        noise = torch.randn_like(x0)
        noisy = (1.0 - sigma.view(1, 1, 1, 1)) * x0 + sigma.view(1, 1, 1, 1) * noise
        pred = self.lora_network(noisy.to(self.dtype), sigma.view(1, 1), text)
        return torch.nn.functional.mse_loss(pred.float(), noise - x0)


def main() -> None:
    args = training_arguments("Train a Visage-Image LoRA in maltrainer").parse_args()
    config = load_config(args.config)
    start_run(config)
    VisageLoRATrainer(config, dry_run=args.dry_run, dry_run_step_time=args.dry_run_step_time).run()


if __name__ == "__main__":
    main()
