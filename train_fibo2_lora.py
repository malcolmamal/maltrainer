#!/usr/bin/env python3
"""
Fibo 2.0 (briaai/Fibo-2.0-dev) LoRA training for maltrainer, text-to-image.

Trains the full, non-distilled Dev model on ComfyUI's single files: the bf16 DiT (fibo2_dev_bf16), the Flux2 VAE and
the Qwen3-VL-4B text encoder (qwen3vl_4b_int8_convrot, or _bf16). Latents and caption embeddings are cached once, then
the encoders are unloaded before the DiT loads. Captions are plain sentences or Fibo's structured JSON (a .txt file
holding the JSON object); a JSON caption is cleaned and compacted as in training. Edit training and the images of
``text_render`` entries are not supported.

Conventions follow BRIA's fine_tuning/train_lora.py and ComfyUI's Fibo 2.0 (ModelType.FLUX). The DiT gets sigma in
[0, 1] and predicts v = noise - x0. Sigma is sigmoid(N(shift, 1)) with the shift rising with the number of tokens
(10% uniform), as in BRIA's recipe.

    python train_fibo2_lora.py --config templates/fibo2_template.yaml
"""

from __future__ import annotations

import logging

import torch
from tqdm import tqdm

from maltrainer.control.launch import make_keys
from maltrainer.control.stats import RunView
from maltrainer.data.flux3_dataset import load_flux3_image
from maltrainer.models.fibo2.loading import load_fibo2_transformer
from maltrainer.models.fibo2.lora import Fibo2LoRANetwork
from maltrainer.models.fibo2.text_encoder import load_fibo2_text_encoder
from maltrainer.models.flux3.vae import load_flux2_vae_encoder
from maltrainer.trainers.fibo2.previews import FiboPreviews
from maltrainer.trainers.fibo2.timesteps import sample_sigma
from maltrainer.training.cli import start_run, training_arguments
from maltrainer.training.image_lora import ImageLoRATrainer
from maltrainer.training.session import free_memory
from maltrainer.utils.config import load_config
from maltrainer.utils.keyboard import KeyCommands

logger = logging.getLogger(__name__)


class Fibo2LoRATrainer(FiboPreviews, ImageLoRATrainer):
    model_type = "fibo2"
    label = "Fibo 2.0"
    default_name = "fibo2_lora"
    batch_one_reason = (
        "Fibo 2.0 trains at batch_size 1 (every caption has its own gist-token count, as at inference); "
        "use train.gradient_accumulation_steps for a larger effective batch."
    )
    # Fibo's attention, MLP and I/O each adapt their own rate, over a 16-step sign window (batch-1 gradients).
    polarity_history = 16
    schedule = ("bria", 1.0)

    # ------------------------------------------------------------------ caching

    def precache_dataset(self) -> None:
        model_config = self.config["model"]
        dataset_config = self.config["dataset"]
        train_config = self.config.get("train", {})
        sources = self._sources()
        if not sources:
            raise ValueError(f"No images found under {dataset_config['path']}")

        print("=== Step 1/2: Pre-caching Dataset (VAE Latents & Text Embeddings) ===", flush=True)
        resolution = int(dataset_config.get("resolution", 1024))
        aspect_mode = str(dataset_config.get("aspect_mode", "bucket")).lower()

        vae = load_flux2_vae_encoder(model_config["vae_path"], self.device)
        sizes: dict[tuple[int, int], int] = {}
        for source in tqdm(sources, desc="Caching latents", disable=not self.is_main_process):
            try:
                pixels = load_flux3_image(source.path, resolution, aspect_mode).unsqueeze(0).to(self.device)
                self.latent_cache[str(source.path)] = vae.encode(pixels)[0].cpu()
                self.samples.append((str(source.path), source.caption))
                size = (pixels.shape[-1], pixels.shape[-2])
                sizes[size] = sizes.get(size, 0) + 1
            except (OSError, ValueError, RuntimeError) as e:
                logger.warning("Skipping %s: %s", source.path.name, e)
        del vae
        free_memory()
        logger.info(
            "Training sizes (w x h: images): %s", ", ".join(f"{w}x{h}: {n}" for (w, h), n in sorted(sizes.items()))
        )
        print("VAE latents pre-cached and VAE unloaded from VRAM.", flush=True)

        prompts = {caption for _, caption in self.samples}
        if float(train_config.get("caption_dropout", 0.0) or 0.0) > 0:
            prompts.add("")
        prompts |= self.preview_texts()
        encoder = load_fibo2_text_encoder(
            model_config["text_encoder_path"], self.device, self.dtype, model_config.get("tokenizer_path")
        )
        self._encode_prompts(encoder, prompts)
        del encoder
        free_memory()
        print("Text embeddings pre-cached and Text Encoder unloaded from VRAM.", flush=True)
        logger.info("Pre-cached %d images and %d prompt embeddings.", len(self.samples), len(self.text_cache))

    # ------------------------------------------------------------------ model

    def load_transformer_and_lora(self) -> None:
        print("=== Step 2/2: Loading Fibo 2.0 DiT ===", flush=True)
        lora_config = self.config.get("lora", {})
        train_config = self.config.get("train", {})
        self.transformer, load_info = load_fibo2_transformer(
            self.config["model"]["transformer_path"], self.device, self.dtype
        )
        self.run_metadata["base_load"] = load_info
        if bool(train_config.get("gradient_checkpointing", True)):
            self.transformer.enable_gradient_checkpointing()

        self.lora_network = Fibo2LoRANetwork(
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
        # caption dropout trains on the empty prompt, which encodes as the null caption the CFG negative uses
        key, caption = self._sample(index)
        x0 = self.latent_cache[key].to(self.device, torch.float32).unsqueeze(0)
        context = self.text_cache[caption].to(self.device, self.dtype).unsqueeze(0)

        sigma = sample_sigma(self.config["train"], x0.shape[2] * x0.shape[3], self.device)
        noise = torch.randn_like(x0)
        noisy = (1.0 - sigma.view(1, 1, 1, 1)) * x0 + sigma.view(1, 1, 1, 1) * noise
        pred = self.lora_network(noisy.to(self.dtype), sigma, context)
        loss = torch.nn.functional.mse_loss(pred.float(), (noise - x0).float())
        strength = float(self.config["train"].get("loss_weight_strength", 1.0))
        return loss * strength if strength != 1.0 else loss


def main() -> None:
    args = training_arguments("Train a Fibo 2.0 LoRA in maltrainer").parse_args()
    config = load_config(args.config)
    start_run(config)
    Fibo2LoRATrainer(config, dry_run=args.dry_run, dry_run_step_time=args.dry_run_step_time).run()


if __name__ == "__main__":
    main()
