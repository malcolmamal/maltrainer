#!/usr/bin/env python3
"""
MiniMax H3 LoRA Training Script for Maltrainer.
Trains LoRA adapters for MiniMax H3 using INT8 ConvRot transformer, FP16 Video VAE, FP32 audio
VAE and Qwen3-VL text encoder, on stills, video clips with sound, and audio-only recordings.
Uses automated dataset pre-caching for 40x step acceleration (~0.8s/step).
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from accelerate import Accelerator, init_empty_weights
from accelerate.utils import set_seed
from safetensors.torch import save_file
from torch.optim import AdamW
from tqdm import tqdm

from maltrainer.data.media import read_audio, read_video_frames
from maltrainer.data.minimax_dataset import (
    AUDIO_SAMPLE_RATE,
    FPS,
    MEDIA_KINDS,
    MIN_AUDIO_FRAMES,
    MIN_CLIP_FRAMES,
    MediaSource,
    audio_latent_frames,
    count_media_files,
    frames_tensor,
    load_image,
    plan_segments,
    scan_media_sources,
    slice_audio,
    summarize_kinds,
)
from maltrainer.models.minimax_h3.audio_vae import load_minimax_h3_audio_encoder
from maltrainer.models.minimax_h3.int8_convrot import (
    enable_int8_convrot,
    load_comfy_int8_convrot_state_dict,
    prepare_int8_convrot_modules,
)
from maltrainer.models.minimax_h3.lora import MiniMaxH3LoRANetwork
from maltrainer.models.minimax_h3.model import (
    MiniMaxH3TokenTag,
    MiniMaxH3Transformer,
    MiniMaxH3TransformerConfig,
)
from maltrainer.models.minimax_h3.text_encoder import load_minimax_h3_text_encoder
from maltrainer.models.minimax_h3.video_vae import load_minimax_h3_video_vae, video_latent_frames
from maltrainer.utils.config import load_config, save_config_to_output
from maltrainer.utils.ema import ParameterEMA
from maltrainer.utils.logging_utils import setup_logging
from maltrainer.utils.lora_metadata import build_metadata, git_commit

logger = logging.getLogger(__name__)


def patchify_video(latent: torch.Tensor, patch_size=(1, 2, 2)) -> torch.Tensor:
    """Patchify video/image latents matching ComfyUI / MiniMax H3 layout [B, C, T, H, W] -> [B, N_tokens, C*1*2*2]."""
    b, c, t_full, h_full, w_full = latent.shape
    pt, ph, pw = patch_size
    t, h, w = t_full // pt, h_full // ph, w_full // pw
    x = latent.reshape(b, c, t, pt, h, ph, w, pw)
    x = torch.einsum("nctrhpwq->nthwcrpq", x)
    return x.reshape(b, t * h * w, c * pt * ph * pw)


def apply_flow_shift(t: torch.Tensor, shift: float = 12.0) -> torch.Tensor:
    """Apply discrete flow shift mapping: t_shifted = (shift * t) / (1 + (shift - 1) * t)."""
    if shift == 1.0:
        return t
    return (shift * t) / (1.0 + (shift - 1.0) * t)


# H3 denoises its two streams on different schedules: video at flow shift 12, audio at 3
# (ComfyUI's sigma_shift_video / sigma_shift_audio). They advance together, so at inference
# the audio sigma is always the closed-form image of the video sigma below -- the model has
# only ever seen (t_video, t_audio) pairs on that curve. Training keeps that coupling no
# matter how the video sigma itself is sampled: e.g. video sigma 0.5 pairs with audio 0.2.
VIDEO_SIGMA_SHIFT = 12.0
AUDIO_SIGMA_SHIFT = 3.0


def remap_sigma(
    sigma: torch.Tensor, from_shift: float = VIDEO_SIGMA_SHIFT, to_shift: float = AUDIO_SIGMA_SHIFT
) -> torch.Tensor:
    """A sigma on the ``from_shift`` schedule -> the same base-grid point on ``to_shift``.

    Mirrors comfy/ldm/minimax/model.py ``time_shift_sigma``.
    """
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def pack_audio_rows(latent: torch.Tensor) -> torch.Tensor:
    """[B, 32, 2, T] audio latent -> [B, 2*T, 32] channel-major rows (left t0..T-1, then right)."""
    b, c, ch, t = latent.shape
    return latent.permute(0, 2, 3, 1).reshape(b, ch * t, c)


# --- Packed-sequence RoPE geometry -------------------------------------------------
# These mirror comfy/ldm/minimax/model.py (_axis_from_sqrt_area / _frame_grid /
# _video_t_grid / PackedLayout) exactly. H3 does NOT index video tokens by raw patch
# index: each axis is normalized by the frame's sqrt(area), centered, and scaled by 32,
# and the video segment's temporal origin sits directly after the text segment. Training
# on a plain arange() grid puts every token at the wrong RoPE frequency.

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0


def _axis_from_sqrt_area(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    ratio = dim / sqrt_area
    n = dim // patch
    return (torch.arange(n, dtype=torch.float64) * (ratio / n) + (1.0 - ratio) / 2.0) * 32.0


def _frame_grid(latent_h: int, latent_w: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Area-normalized (h, w) coordinates of one latent frame's 2x2-patch rows, plus the w axis."""
    area = math.sqrt(latent_h * latent_w)
    w_axis = _axis_from_sqrt_area(latent_w, 2, area)
    hh, ww = torch.meshgrid(
        _axis_from_sqrt_area(latent_h, 2, area),
        w_axis,
        indexing="ij",
    )
    return torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1), w_axis


def _video_t_grid(latent_t: int, origin: float) -> torch.Tensor:
    spans = torch.tensor(
        [FRAME_RESCALE * FRAME_PER_TOKEN[index % len(FRAME_PER_TOKEN)] for index in range(latent_t)],
        dtype=torch.float64,
    )
    return float(origin) + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def build_packed_position_ids(
    text_len: int, latent_t: int, latent_h: int, latent_w: int, audio_t: int = 0
) -> torch.Tensor:
    """Position ids for H3's [text | audio | video] packed sequence, as [S, 3] float64.

    Matches ComfyUI's PackedLayout (t2va, no references): text tokens advance along the
    temporal axis; audio and video share the media clock's origin at text_len. Audio rows are
    channel-major stereo -- t advances one unit per latent frame (40 per second), h stays 0,
    and w is pinned to the video grid's first column for the left channel and its last column
    for the right one.
    """
    text_pos = torch.zeros(text_len, 3, dtype=torch.float64)
    text_pos[:, 0] = torch.arange(text_len, dtype=torch.float64)

    frame, w_axis = _frame_grid(latent_h, latent_w)
    audio_pos = torch.zeros(audio_t * 2, 3, dtype=torch.float64)
    if audio_t:
        audio_pos[:, 0] = (float(text_len) + torch.arange(audio_t, dtype=torch.float64)).repeat(2)
        audio_pos[:audio_t, 2] = float(w_axis[0])
        audio_pos[audio_t:, 2] = float(w_axis[-1])

    video_pos = torch.empty(latent_t, frame.shape[0], 3, dtype=torch.float64)
    video_pos[:, :, 0] = _video_t_grid(latent_t, float(text_len))[:, None]
    video_pos[:, :, 1:] = frame[None]

    return torch.cat([text_pos, audio_pos, video_pos.reshape(-1, 3)], dim=0)


@dataclass
class H3Sample:
    """One cached training item. Every kind carries a video latent and an audio row count."""

    key: str
    kind: str  # "image" | "clip" | "audio"
    caption: str
    frames: int  # 24 fps pixel frames the item spans (1 for a still)
    video: torch.Tensor  # [1, 24, T_lat, H/16, W/16] normalized, CPU
    audio: Optional[torch.Tensor] = None  # [1, 2*A, 32] normalized rows, CPU; None = silence
    audio_only: bool = False  # video is a zeros placeholder; drop the video loss

    @property
    def audio_latents(self) -> int:
        return audio_latent_frames(self.frames)


class MiniMaxH3LoRATrainer:
    """Trainer for MiniMax H3 LoRA models."""

    def __init__(self, config: dict, dry_run: bool = False, dry_run_step_time: float = 0.025):
        self.config = config
        self.dry_run = dry_run
        self.dry_run_step_time = dry_run_step_time
        self.last_saved_path = None
        self.lr_scheduler = None
        # Resolved facts about this run, embedded into every checkpoint's metadata. Populated
        # as they become known, since several are only decided at load/first-step time.
        self.run_metadata: Dict[str, object] = {
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_commit": git_commit(),
            "torch_version": torch.__version__,
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, config.get("dtype", "bfloat16"))

        self.samples: List[H3Sample] = []
        self._layout_cache: Dict[tuple, Dict[str, torch.Tensor]] = {}
        self.text_cache: Dict[str, torch.Tensor] = {}
        self.ema: Optional[ParameterEMA] = None

        train_config = config.get("train", {})
        if int(train_config.get("batch_size", 1) or 1) != 1:
            # Every item packs its own sequence (text length, frame count, aspect and audio
            # length all vary) and H3's per-timestep modulation is shared across the batch.
            raise ValueError(
                "MiniMax H3 trains at batch_size 1; use train.gradient_accumulation_steps for a larger effective batch."
            )

        # Initialize accelerator
        self.accelerator = Accelerator(
            mixed_precision=config.get("mixed_precision", "bf16"),
            gradient_accumulation_steps=config.get("train", {}).get("gradient_accumulation_steps", 1),
        )

        self.is_main_process = self.accelerator.is_main_process

        if self.is_main_process:
            device_name = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU"
            logger.info("Training on device: %s (%s)", self.device, device_name)
            if self.device.type == "cuda":
                logger.info("Total VRAM: %.2f GB", torch.cuda.get_device_properties(self.device).total_memory / 1024**3)

        log_dir = Path(config.get("output", {}).get("path", "./output"))
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "training.log")
        file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logging.getLogger().addHandler(file_handler)

        self.progress_logger = logging.getLogger("progress")
        self.progress_logger.setLevel(logging.INFO)
        self.progress_logger.addHandler(file_handler)
        self.progress_logger.propagate = False

    def _train_audio(self) -> bool:
        return bool(self.config.get("train", {}).get("train_audio", True))

    def _do_guidance_loss(self) -> bool:
        train_config = self.config.get("train", {})
        return bool(train_config.get("do_guidance_loss", train_config.get("do_differential_guidance", False)))

    def precache_dataset(self):
        """Encode every dataset item once -- video latents, audio rows, caption embeddings --
        into memory, unloading each encoder before the next loads so peak VRAM stays at one model.
        """
        model_config = self.config.get("model", {})
        dataset_config = self.config["dataset"]
        train_audio = self._train_audio()

        print("=== Step 1/2: Pre-caching Dataset (VAE Latents & Text Embeddings) ===", flush=True)
        logger.info("Starting dataset pre-caching...")

        kinds = dataset_config.get("media", list(MEDIA_KINDS))
        if isinstance(kinds, str):
            kinds = [kinds]
        if not train_audio:
            # an audio-only item has nothing but its audio to train
            kinds = [kind for kind in kinds if kind != "audio"]
        sources = scan_media_sources(
            dataset_config["path"],
            kinds=kinds,
            trigger_token=dataset_config.get("trigger", None),
            default_caption=dataset_config.get("default_caption", ""),
        )
        if not sources:
            raise ValueError(f"No training media ({', '.join(kinds)}) found under {dataset_config['path']}")
        logger.info("Dataset: %s", summarize_kinds(sources))

        resolution = int(dataset_config.get("resolution", 512))
        clip_resolution = int(dataset_config.get("clip_resolution", resolution))
        aspect_mode = str(dataset_config.get("aspect_mode", "bucket")).lower()
        max_frames = int(dataset_config.get("max_frames", 73))
        long_media = str(dataset_config.get("long_media", "first")).lower()
        clip_audio = bool(dataset_config.get("clip_audio", True)) and train_audio
        placeholder_px = int(dataset_config.get("audio_placeholder_resolution", 128))

        # Audio waits for the audio encoder: (sample index, source, start frame).
        pending_audio: List[Tuple[int, MediaSource, int]] = []
        audio_sources = [s for s in sources if s.kind == "audio"]
        visual_sources = [s for s in sources if s.kind != "audio"]

        # 1. Video VAE: stills and clip frames
        if visual_sources:
            vae_path = model_config["vae_path"]
            print(f"Loading Video VAE from {vae_path} for pre-caching...", flush=True)
            vae = load_minimax_h3_video_vae(vae_path, device=self.device)
            for source in tqdm(visual_sources, desc="Caching video latents", disable=not self.is_main_process):
                try:
                    if source.kind == "image":
                        pixels = load_image(source.path, resolution, aspect_mode).unsqueeze(0)
                        with torch.no_grad():
                            latent = vae.encode_image(pixels.to(self.device)).detach().cpu()
                        self.samples.append(H3Sample(str(source.path), "image", source.caption, 1, latent))
                        continue

                    budget = (
                        max_frames
                        if long_media != "split"
                        else max_frames * int(dataset_config.get("max_segments_per_file", 16))
                    )
                    frames = read_video_frames(source.path, clip_resolution, budget)
                    segments = plan_segments(len(frames), max_frames, long_media, MIN_CLIP_FRAMES)
                    if not segments:
                        logger.warning(
                            "Skipping %s: %d frames is shorter than one H3 clip", source.path.name, len(frames)
                        )
                        continue
                    for start, count in segments:
                        with torch.no_grad():
                            latent = vae.encode_video(frames_tensor(frames[start : start + count]).to(self.device))
                        latent = latent.detach().cpu()
                        if latent.shape[2] != video_latent_frames(count):
                            raise ValueError(
                                f"{source.path.name}: {count} frames encoded to {latent.shape[2]} latent frames, "
                                f"expected {video_latent_frames(count)}"
                            )
                        key = f"{source.path}@{start}" if len(segments) > 1 else str(source.path)
                        self.samples.append(H3Sample(key, "clip", source.caption, count, latent))
                        if clip_audio and not source.muted:
                            pending_audio.append((len(self.samples) - 1, source, start))
                except Exception as e:
                    if isinstance(e, ImportError):
                        raise
                    logger.warning("Skipping %s: %s", source.path.name, e)
            del vae
            gc.collect()
            torch.cuda.empty_cache()
            print("Video VAE latents pre-cached and VAE unloaded from VRAM.", flush=True)

        # 2. Audio VAE: clip soundtracks and audio-only recordings
        if pending_audio or audio_sources:
            audio_vae_path = model_config.get("audio_vae_path")
            if not audio_vae_path:
                raise ValueError(
                    "model.audio_vae_path is required to train on clip soundtracks or audio files "
                    "(ComfyUI's minimax_h3_audio_vae_fp32.safetensors). Set train.train_audio: false "
                    "to train video only."
                )
            print(f"Loading Audio VAE from {audio_vae_path} for pre-caching...", flush=True)
            audio_vae = load_minimax_h3_audio_encoder(audio_vae_path, device=self.device)

            def encode(wav) -> torch.Tensor:
                waveform = torch.from_numpy(wav).unsqueeze(0).to(self.device)
                return pack_audio_rows(audio_vae.encode(waveform)).detach().cpu()

            decoded: Dict[str, object] = {}
            for index, source, start in tqdm(
                pending_audio, desc="Caching clip audio", disable=not self.is_main_process
            ):
                key = str(source.path)
                if key not in decoded:
                    decoded = {key: read_audio(source.path)}  # one file at a time in memory
                wav = decoded[key]
                if wav is None:
                    continue  # no track / digital silence: trains as a muted clip
                sample = self.samples[index]
                rows = encode(slice_audio(wav, start, sample.frames))
                self._check_audio_rows(sample, rows)
                sample.audio = rows
            decoded.clear()

            budget_frames = (
                max_frames
                if long_media != "split"
                else max_frames * int(dataset_config.get("max_segments_per_file", 16))
            )
            for source in tqdm(audio_sources, desc="Caching audio items", disable=not self.is_main_process):
                try:
                    wav = read_audio(source.path)
                    if wav is None:
                        logger.warning("Skipping %s: no decodable sound (or digital silence)", source.path.name)
                        continue
                    total = min(int(wav.shape[1] / AUDIO_SAMPLE_RATE * FPS), budget_frames)
                    segments = plan_segments(total, max_frames, long_media, MIN_AUDIO_FRAMES)
                    if not segments:
                        logger.warning("Skipping %s: shorter than %.2f s", source.path.name, MIN_AUDIO_FRAMES / FPS)
                        continue
                    for start, count in segments:
                        # A zeros video latent is the dataset mean in normalized space. It sizes
                        # the packed sequence and gives the audio rows something to attend
                        # across; the video loss is dropped for these items.
                        side = max(32, placeholder_px // 32 * 32) // 16
                        placeholder = torch.zeros(1, 24, video_latent_frames(count), side, side)
                        key = f"{source.path}@{start}" if len(segments) > 1 else str(source.path)
                        sample = H3Sample(key, "audio", source.caption, count, placeholder, audio_only=True)
                        sample.audio = encode(slice_audio(wav, start, count))
                        self._check_audio_rows(sample, sample.audio)
                        self.samples.append(sample)
                except Exception as e:
                    if isinstance(e, ImportError):
                        raise
                    logger.warning("Skipping %s: %s", source.path.name, e)
            del audio_vae
            gc.collect()
            torch.cuda.empty_cache()
            print("Audio latents pre-cached and Audio VAE unloaded from VRAM.", flush=True)

        if not self.samples:
            raise ValueError("Every dataset item was skipped; see the warnings above.")

        # 3. Text encoder: every unique caption, plus the empty prompt when a pass needs it
        text_encoder_path = model_config["text_encoder_path"]
        print(f"Loading Text Encoder from {text_encoder_path} for pre-caching...", flush=True)
        text_encoder = load_minimax_h3_text_encoder(
            text_encoder_path,
            tokenizer_path=model_config.get("tokenizer_path"),
            device=self.device,
            dtype=self.dtype,
        )

        train_config = self.config.get("train", {})
        prompts = {sample.caption for sample in self.samples}
        if self._do_guidance_loss():
            prompts.add(str(train_config.get("unconditional_prompt", "")))
        if float(train_config.get("caption_dropout", 0.0) or 0.0) > 0:
            prompts.add("")
        print(f"Caching text embeddings for {len(prompts)} unique prompts...", flush=True)
        for caption in tqdm(sorted(prompts), desc="Caching Prompts", disable=not self.is_main_process):
            with torch.no_grad():
                self.text_cache[caption] = (
                    text_encoder.encode_prompts([caption], device=self.device, dtype=self.dtype).detach().cpu()
                )

        del text_encoder
        gc.collect()
        torch.cuda.empty_cache()
        print("Text embeddings pre-cached and Text Encoder unloaded from VRAM.", flush=True)

        counts = {
            "images": sum(1 for s in self.samples if s.kind == "image"),
            "clips": sum(1 for s in self.samples if s.kind == "clip"),
            "clips_with_audio": sum(1 for s in self.samples if s.kind == "clip" and s.audio is not None),
            "audio_items": sum(1 for s in self.samples if s.kind == "audio"),
        }
        self.run_metadata["dataset_composition"] = counts
        logger.info(
            "Pre-cached %d samples (%s) and %d prompt embeddings. Unloaded feature extractors from VRAM.",
            len(self.samples),
            ", ".join(f"{k}={v}" for k, v in counts.items()),
            len(self.text_cache),
        )

    @staticmethod
    def _check_audio_rows(sample: H3Sample, rows: torch.Tensor) -> None:
        # The packed layout's audio grid is sized from the frame count, the rows from the
        # waveform. They agree by construction (hop-exact slicing); a silent mismatch would
        # train every audio row against a shifted position, so check rather than trust it.
        expected = sample.audio_latents * 2
        if rows.shape[1] != expected:
            raise ValueError(
                f"{sample.key}: audio encoded to {rows.shape[1]} rows, the {sample.frames}-frame "
                f"layout packs {expected}"
            )

    def load_transformer_and_lora(self):
        """Load INT8 ConvRot transformer base and inject LoRA network."""
        print("=== Step 2/2: Loading MiniMax H3 INT8 Transformer Base ===", flush=True)
        logger.info("Loading Transformer base and LoRA network...")
        model_config = self.config.get("model", {})
        transformer_path = model_config["transformer_path"]

        int8_state_dict, num_quant_layers = load_comfy_int8_convrot_state_dict(
            transformer_path, device=torch.device("cpu")
        )
        logger.info("Parsed %d INT8 ConvRot quantized layers", num_quant_layers)

        if "adaln_t_table" in int8_state_dict:
            table = int8_state_dict["adaln_t_table"]
            table_size, embed_dim = table.shape[0], table.shape[1]
            logger.info("Detected pruned AdaLN table: size=%d, embed_dim=%d", table_size, embed_dim)
            transformer_cfg = MiniMaxH3TransformerConfig(adaln_t_table_size=table_size, time_embed_dim=embed_dim)
        else:
            transformer_cfg = MiniMaxH3TransformerConfig()

        print("Instantiating Transformer structure on GPU...", flush=True)
        with init_empty_weights(include_buffers=True):
            transformer = MiniMaxH3Transformer(transformer_cfg)

        prepare_int8_convrot_modules(transformer, int8_state_dict)
        info = transformer.load_state_dict(int8_state_dict, strict=False, assign=True)
        # Everything is built under init_empty_weights, so an unloaded tensor stays on the
        # meta device. rope.inv_freq and adaln_t_table in particular carry checkpoint values
        # that cannot be recomputed -- silently keeping a placeholder would misplace every
        # RoPE frequency, so fail loudly instead.
        if info.missing_keys:
            raise ValueError(
                f"Transformer checkpoint is missing {len(info.missing_keys)} tensor(s); "
                f"examples: {info.missing_keys[:5]}"
            )
        if info.unexpected_keys:
            raise ValueError(
                f"Transformer checkpoint has {len(info.unexpected_keys)} tensor(s) the model does "
                f"not expect; examples: {info.unexpected_keys[:5]}"
            )
        grad_precision = self.config.get("train", {}).get("int8_grad_precision")
        enable_int8_convrot(transformer, grad_precision=grad_precision)
        self.run_metadata["int8_grad_precision"] = grad_precision or "bf16"

        transformer.enable_gradient_checkpointing()
        transformer.requires_grad_(False)
        transformer.to(self.device)
        self.transformer = transformer

        # Inject LoRA
        lora_config = self.config.get("lora", {})
        self.lora_network = MiniMaxH3LoRANetwork(
            transformer=self.transformer,
            rank=lora_config.get("rank", 16),
            alpha=lora_config.get("alpha", 16.0),
            prefix=lora_config.get("prefix", "diffusion_model."),
            target_modules=lora_config.get("target_modules", ["qkv_proj", "out_proj", "fc1", "fc2"]),
            include_adaln=bool(lora_config.get("include_adaln", False)),
            include_token_refiner=bool(lora_config.get("include_token_refiner", True)),
            block_range=lora_config.get("block_range"),
        )
        # Adapters train in fp32 by default. bf16 carries ~8 mantissa bits (~0.4% relative
        # precision), and a LoRA update is small next to the weight it lands on, so in bf16 a
        # large share of every step rounds away -- the subject trains "a bit" and then stops
        # sharpening. Cast only the adapters: the base keeps its checkpoint dtypes, including
        # the tensors H3 holds in fp32.
        self.lora_dtype = getattr(torch, str(lora_config.get("dtype", "float32")))
        self.lora_network.set_lora_dtype(self.device, self.lora_dtype)
        adapter_path = model_config.get("training_adapter_path") or lora_config.get("training_adapter_path")
        if adapter_path:
            attached = self.lora_network.load_assistant_adapter(
                adapter_path,
                multiplier=float(lora_config.get("training_adapter_multiplier", 1.0)),
                device=self.device,
                dtype=self.dtype,
                adaln_grid_path=model_config.get("training_adapter_adaln_grid"),
            )
            self.run_metadata["training_adapter"] = Path(str(adapter_path)).name
            self.run_metadata["training_adapter_modules"] = attached

        trainable = sum(p.numel() for p in self.lora_network.trainable_parameters())
        self.run_metadata["lora_dtype"] = str(self.lora_dtype).replace("torch.", "")
        self.run_metadata["trainable_parameters"] = trainable
        self.run_metadata["lora_layers"] = len(self.lora_network.lora_layers)
        logger.info(
            "LoRA adapters: %.1fM trainable parameters in %s (base left at checkpoint dtypes)",
            trainable / 1e6,
            self.lora_dtype,
        )
        print("Transformer and LoRA network ready on GPU!", flush=True)

    def _epoch_order(self) -> List[int]:
        """Shuffled sample indices for one epoch (each sample ``repeats`` times), this rank's share."""
        repeats = max(int(self.config["dataset"].get("repeats", 1) or 1), 1)
        order = [index for index in range(len(self.samples)) for _ in range(repeats)]
        random.shuffle(order)
        world = self.accelerator.num_processes
        return order[self.accelerator.process_index :: world] if world > 1 else order

    def _count_dataset_samples(self) -> int:
        """Files only: the dry run does not decode, so split long clips count once."""
        dataset_config = self.config.get("dataset", {})
        kinds = dataset_config.get("media", list(MEDIA_KINDS))
        if isinstance(kinds, str):
            kinds = [kinds]
        sample_count = count_media_files(str(dataset_config.get("path", "")), kinds)
        repeats = int(dataset_config.get("repeats", 1) or 1)
        return max(sample_count, 1) * max(repeats, 1)

    def dry_run_training(self):
        """Simulate training without downloading or loading models."""
        train_config = self.config.get("train", {})
        num_epochs = int(train_config.get("num_epochs", 10) or 1)
        batch_size = int(train_config.get("batch_size", 1) or 1)
        grad_accum = int(train_config.get("gradient_accumulation_steps", 1) or 1)
        num_samples = self._count_dataset_samples()
        steps_per_epoch = math.ceil((num_samples / max(batch_size, 1)) / max(grad_accum, 1))
        total_steps = max(num_epochs * steps_per_epoch, 1)
        save_every = int(train_config.get("save_every", 500) or 0)
        step_time = max(float(self.dry_run_step_time), 0.0)
        output_dir = Path(self.config.get("output", {}).get("path", "./output"))
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Dry run enabled: simulating %d steps.", total_steps)

        start_time = time.time()
        progress_bar = tqdm(total=total_steps, desc="Dry Run Steps", disable=not self.is_main_process)
        for step in range(1, total_steps + 1):
            if step_time > 0:
                time.sleep(step_time)
            progress_bar.update(1)

            if save_every > 0 and step % save_every == 0:
                self._write_dry_run_checkpoint(output_dir, step, total_steps, steps_per_epoch, num_epochs, start_time)

        progress_bar.close()
        self._write_dry_run_checkpoint(output_dir, "final", total_steps, steps_per_epoch, num_epochs, start_time)
        logger.info("Dry run completed successfully.")

    def _write_dry_run_checkpoint(
        self, output_dir: Path, step, total_steps: int, steps_per_epoch: int, num_epochs: int, start_time: float
    ):
        elapsed = max(time.time() - start_time, 0.0)
        completed_steps = total_steps if step == "final" else int(step)
        progress = min(completed_steps / max(total_steps, 1), 1.0)

        payload = {
            "mode": "dry_run",
            "kind": "minimax_h3_lora",
            "step": step,
            "total_steps": total_steps,
            "steps_per_epoch": steps_per_epoch,
            "num_epochs": num_epochs,
            "progress": {
                "completed_steps": completed_steps,
                "percent": round(progress * 100, 2),
                "elapsed_seconds": round(elapsed, 4),
            },
            "config": self.config,
        }

        save_path = output_dir / f"minimax_h3_lora_step_{step}.safetensors"
        save_path.write_text(json.dumps(payload, indent=2))
        self.last_saved_path = str(save_path)

    def _build_packed_layout(
        self, num_text_tokens: int, latent_t: int, latent_h: int, latent_w: int, audio_t: int = 0
    ) -> Dict[str, torch.Tensor]:
        """Index/tag/position tensors for one [text | audio | video] packed sequence.

        Everything here is a function of the text length, the latent size and the audio
        length, so the conditional and unconditional passes need SEPARATE layouts whenever
        their prompts tokenize to different lengths -- which they always do when the
        unconditional prompt is empty (a single BOS token). Cached per shape signature.

        Timestep rows: 0 = video sigma (text and video tokens), 1 = audio sigma. The model
        indexes its AdaLN table by ``timestep_index * 3 + modality_tag``, as ComfyUI does.
        """
        key = (num_text_tokens, latent_t, latent_h, latent_w, audio_t)
        cached = self._layout_cache.get(key)
        if cached is not None:
            return cached

        num_audio_tokens = audio_t * 2
        num_video_tokens = latent_t * (latent_h // 2) * (latent_w // 2)
        audio_start = num_text_tokens
        video_start = audio_start + num_audio_tokens
        seq_len = video_start + num_video_tokens

        text_indices = torch.arange(0, audio_start, device=self.device)
        audio_indices = torch.arange(audio_start, video_start, device=self.device)
        video_indices = torch.arange(video_start, seq_len, device=self.device)
        token_tags = torch.zeros((seq_len,), device=self.device, dtype=torch.long)
        token_tags[text_indices] = MiniMaxH3TokenTag.TEXT
        token_tags[audio_indices] = MiniMaxH3TokenTag.AUDIO
        token_tags[video_indices] = MiniMaxH3TokenTag.VIDEO
        timestep_indices = torch.zeros((seq_len,), device=self.device, dtype=torch.long)
        timestep_indices[audio_indices] = 1

        layout = {
            "text_indices": text_indices,
            "video_indices": video_indices,
            "audio_indices": audio_indices,
            "timestep_indices": timestep_indices,
            "token_tags": token_tags,
            "position_ids": build_packed_position_ids(num_text_tokens, latent_t, latent_h, latent_w, audio_t).to(
                device=self.device
            ),
        }
        self._layout_cache[key] = layout
        return layout

    def _sample_timesteps(self, batch_size: int) -> torch.Tensor:
        # fp32: in bf16 a sigma near 1 lands on a 1/256 grid, and the audio remap below
        # amplifies that step several-fold at the low-noise end of the audio schedule.
        train_config = self.config.get("train", {})
        distribution = str(train_config.get("timestep_distribution", "krea2_shift")).lower()
        if distribution in {"sigmoid", "krea2_shift", "krea2"}:
            # Logit-normal base, as musubi-tuner's krea2_shift does before shifting.
            scale = float(train_config.get("sigmoid_scale", 1.0))
            z = torch.randn(batch_size, device=self.device, dtype=torch.float32) * scale
            timesteps = torch.sigmoid(z).clamp(0.001, 0.999)
        elif distribution in {"logit_normal", "logit-normal", "logitnormal"}:
            mean = float(train_config.get("timestep_logit_mean", 0.0))
            std = float(train_config.get("timestep_logit_std", 1.0))
            z = torch.randn(batch_size, device=self.device, dtype=torch.float32) * std + mean
            timesteps = torch.sigmoid(z).clamp(0.001, 0.999)
        else:
            timesteps = torch.rand(batch_size, device=self.device, dtype=torch.float32).clamp(0.001, 0.999)
        return timesteps

    def _resolve_flow_shift(self, latent_h: int, latent_w: int) -> float:
        """Sigma shift for this latent size.

        H3's 12.0 is its *video* flow shift. Applied to still images it pushes ~57% of
        samples above sigma 0.9 and leaves only ~8% below 0.5, so almost all training
        happens in the high-noise range that sets composition rather than the low-noise
        range where facial identity and fine detail are resolved.

        krea2_shift instead derives the shift from the packed token count, matching
        musubi-tuner (x1=256 -> mu 0.5, x2=6400 -> mu 1.15, shift = exp(mu)). At 512px that
        is ~1.65, which puts ~31% of samples below sigma 0.5.
        """
        train_config = self.config.get("train", {})
        distribution = str(train_config.get("timestep_distribution", "krea2_shift")).lower()
        if distribution == "sigmoid":
            # No shift at all: sigma = sigmoid(N(0,1)), median 0.5. Community reports on
            # H3 character LoRAs single this out as important, paired with contrastive
            # guidance -- which changes the dynamics that made unshifted densities
            # overdrive adapters in trainers that ran them without it.
            return 1.0
        if distribution == "lownoise":
            # Fizgig's "Likeness and Style" default: uniform u through the shift map, with the
            # shift chosen so that exactly lownoise_pct % of steps land below sigma 0.5
            # (P(sigma < 0.5) = 1 / (1 + shift)). 60 % -> shift 0.667.
            share = float(train_config.get("lownoise_pct", 60)) / 100.0
            if not 0.0 < share < 1.0:
                raise ValueError("train.lownoise_pct must be between 0 and 100 (exclusive)")
            return (1.0 - share) / share
        if distribution not in {"krea2_shift", "krea2"}:
            return float(train_config.get("timestep_shift", 12.0))

        packed_tokens = (latent_h // 2) * (latent_w // 2)
        slope = (1.15 - 0.5) / (6400 - 256)
        mu = 0.5 + slope * (packed_tokens - 256)
        return math.exp(mu)

    def setup_optimizer(self):
        """Setup optimizer over the LoRA adapters only."""
        train_config = self.config["train"]
        lr = float(train_config.get("learning_rate", 1e-4))
        optimizer_name = str(train_config.get("optimizer", "adamw8bit")).lower()
        weight_decay = float(train_config.get("weight_decay", 0.01))
        params = list(self.lora_network.trainable_parameters())

        if optimizer_name in {"adamw8bit", "adam8bit"}:
            try:
                import bitsandbytes as bnb

                optimizer_cls = bnb.optim.AdamW8bit if optimizer_name == "adamw8bit" else bnb.optim.Adam8bit
                self.optimizer = optimizer_cls(params, lr=lr)
                logger.info("Using 8-bit optimizer: %s (lr=%.2e)", optimizer_name, lr)
                self.run_metadata["optimizer_resolved"] = {"name": optimizer_name, "lr": lr}
            except (ImportError, OSError, RuntimeError) as e:
                logger.warning("Bitsandbytes not available (%s). Falling back to AdamW.", e)
                self.optimizer = AdamW(params, lr=lr)
        elif optimizer_name.startswith("prodigy"):
            try:
                import inspect

                from prodigyopt import Prodigy

                # Prodigy's `lr` is a multiplier on the step size it estimates, so it belongs
                # near 1.0. A config that sets learning_rate to an AdamW-scale value (<0.01)
                # clearly did not mean it as a multiplier, so fall back to 1.0 there; a value
                # at or above 0.01 is taken at face value. train.prodigy_lr overrides either way.
                configured_lr = float(train_config.get("learning_rate", 1.0))
                default_multiplier = configured_lr if configured_lr >= 0.01 else 1.0
                requested = {
                    "lr": float(train_config.get("prodigy_lr", default_multiplier)),
                    "betas": (
                        float(train_config.get("adam_beta1", 0.9)),
                        float(train_config.get("prodigy_beta2", 0.99)),
                    ),
                    # EMA horizon for the d estimate. None -> prodigyopt uses sqrt(beta2).
                    "beta3": train_config.get("prodigy_beta3"),
                    "weight_decay": weight_decay,
                    "eps": float(train_config.get("prodigy_eps", 1e-8)),
                    # Starting step-size guess. Raising it shortens the ramp before Prodigy
                    # finds its rate, at the risk of an early overshoot.
                    "d0": float(train_config.get("prodigy_d0", 1e-6)),
                    # The single most useful knob: multiplies the adapted rate. 1.0 is the
                    # library default, ~2.0 the usual LoRA recommendation; higher trains
                    # harder and starts trading likeness for texture damage.
                    "d_coef": float(train_config.get("prodigy_d_coef", 2.0)),
                    # Per-step ceiling on how fast d may grow (inf = uncapped). Values just
                    # above 1 (e.g. 1.02) tame a d that spikes and then overshoots.
                    "growth_rate": float(train_config.get("prodigy_growth_rate", float("inf"))),
                    "decouple": bool(train_config.get("prodigy_decouple", True)),
                    "use_bias_correction": bool(train_config.get("prodigy_use_bias_correction", True)),
                    "safeguard_warmup": bool(
                        train_config.get(
                            "prodigy_safeguard_warmup",
                            bool(train_config.get("learning_rate_warmup_steps", 0)),
                        )
                    ),
                }
                if train_config.get("prodigy_slice_p") is not None:
                    requested["slice_p"] = int(train_config["prodigy_slice_p"])

                # prodigyopt's signature varies across releases (slice_p and growth_rate are
                # not in every version). Pass only what this install accepts instead of
                # crashing on an unexpected keyword.
                supported = set(inspect.signature(Prodigy.__init__).parameters)
                kwargs = {k: v for k, v in requested.items() if k in supported and v is not None}
                unsupported = sorted(set(requested) - set(kwargs) - {"beta3"})
                if unsupported:
                    logger.warning(
                        "Installed prodigyopt does not accept %s; those settings are ignored.",
                        unsupported,
                    )

                self.optimizer = Prodigy(params, **kwargs)
                logger.info(
                    "Using Prodigy: %s",
                    ", ".join(f"{k}={v}" for k, v in sorted(kwargs.items())),
                )
                self.run_metadata["optimizer_resolved"] = {"name": "prodigy", **kwargs}
                if configured_lr < 0.01 and "prodigy_lr" not in train_config:
                    logger.info(
                        "train.learning_rate (%.2e) is AdamW-scale, so Prodigy's multiplier is "
                        "1.0 instead. Set train.prodigy_lr to control it explicitly.",
                        configured_lr,
                    )
                if self.lora_dtype is not torch.float32:
                    logger.warning(
                        "Prodigy adapts on dot products of gradients and parameter deltas; in "
                        "%s those are too coarse to steer it. Set lora.dtype: float32.",
                        self.lora_dtype,
                    )
            except (ImportError, OSError, RuntimeError) as e:
                logger.warning("Prodigy not available (%s). Falling back to AdamW.", e)
                self.optimizer = AdamW(params, lr=lr)
        elif optimizer_name in {"automagic3", "automagic", "automagic_v3"}:
            from maltrainer.optimizers import Automagic3

            configured_lr = float(train_config.get("learning_rate", 1e-6))
            automagic_params = {
                "lr": configured_lr,
                "min_lr": float(train_config.get("automagic_min_lr", train_config.get("min_lr", 1e-8))),
                "max_lr": float(train_config.get("automagic_max_lr", train_config.get("max_lr", 1e3))),
                "beta2": float(train_config.get("automagic_beta2", train_config.get("beta2", 0.999))),
                "eps": float(train_config.get("automagic_eps", train_config.get("eps", 1e-30))),
                "clip_threshold": float(
                    train_config.get("automagic_clip_threshold", train_config.get("clip_threshold", 1.0))
                ),
                "weight_decay": weight_decay,
                "polarity_history": int(
                    train_config.get("automagic_polarity_history", train_config.get("polarity_history", 8))
                ),
                "fused": bool(train_config.get("automagic_fused", False)),
            }
            self.optimizer = Automagic3(params, **automagic_params)
            logger.info(
                "Using Automagic3 optimizer: start_lr=%.2e, history=%d, clip_threshold=%.2f",
                automagic_params["lr"],
                automagic_params["polarity_history"],
                automagic_params["clip_threshold"],
            )
            self.run_metadata["optimizer_resolved"] = {"name": "automagic3", **automagic_params}
        else:
            self.optimizer = AdamW(params, lr=lr)

        warmup_steps = int(train_config.get("learning_rate_warmup_steps", 0) or 0)
        if warmup_steps > 0:

            def lr_lambda(step: int) -> float:
                return min((step + 1) / warmup_steps, 1.0)

            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lr_lambda,
            )
            logger.info("Configured learning rate warmup for %d steps.", warmup_steps)

    def _forward(self, video_patches, audio_rows, text_embeds, timestep, layout):
        return self.lora_network(
            video_hidden_states=video_patches,
            audio_hidden_states=audio_rows,
            encoder_hidden_states=text_embeds,
            timestep=timestep,
            timestep_indices=layout["timestep_indices"],
            token_tags=layout["token_tags"],
            position_ids=layout["position_ids"],
            video_indices=layout["video_indices"],
            audio_indices=layout["audio_indices"],
            text_indices=layout["text_indices"],
        )

    @staticmethod
    def _guided_target(target, uncond, sigma, scale, schedule, cfg_zero):
        """Contrastive guidance target for one stream: uncond + scale * (target - uncond).

        H3 is guidance-distilled: CFG is baked into the weights, so a plain flow-matching
        target trains the conditional direction too weakly and the distillation degrades as
        the LoRA takes hold. The fix extrapolates the TARGET away from the model's own
        (detached) unconditional prediction, rather than applying CFG to the prediction.
        Each stream uses its own sigma for the schedule, so audio decays on the audio clock.
        """
        uncond = uncond.detach().float()
        guided = target.float()
        bsz = guided.shape[0]
        if cfg_zero:
            # CFG-Zero*: rescale the unconditional direction onto the conditional one before
            # extrapolating, so a poorly-aligned uncond prediction does not drag the target sideways.
            flat_cond = guided.reshape(bsz, -1)
            flat_uncond = uncond.reshape(bsz, -1)
            dot = (flat_cond * flat_uncond).sum(dim=1, keepdim=True)
            squared_norm = (flat_uncond**2).sum(dim=1, keepdim=True) + 1e-8
            uncond = uncond * (dot / squared_norm).view(bsz, 1, 1)
        # (target - uncond) carries sigma * fresh_noise, which nothing can predict at low
        # sigma, so the "sigma" schedule decays toward a plain flow target there.
        scale_eff = 1.0 + (scale - 1.0) * sigma.detach().float().view(bsz, 1, 1) if schedule == "sigma" else scale
        return (uncond + scale_eff * (guided - uncond)).detach().to(target.dtype)

    def _training_step(self, sample: H3Sample, flow_shift: float) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Loss for one cached sample. Returns (loss, parts for logging)."""
        train_config = self.config["train"]

        latents = sample.video.to(device=self.device, dtype=torch.float32)  # [1, 24, T, h, w]
        _, _, f_lat, h_lat, w_lat = latents.shape

        caption = sample.caption
        dropout = float(train_config.get("caption_dropout", 0.0) or 0.0)
        if dropout > 0 and torch.rand(()).item() < dropout:
            caption = ""
        text_embeds = self.text_cache[caption].to(device=self.device, dtype=self.dtype)

        # Sigmas: video from the configured schedule, audio on its coupled shift-3 curve.
        sigma_v = apply_flow_shift(self._sample_timesteps(1), shift=flow_shift)
        sigma_a = remap_sigma(sigma_v)

        # x_t = (1 - sigma) * x0 + sigma * noise; the H3 checkpoint predicts (x0 - noise).
        noise = torch.randn_like(latents)
        noisy_latents = (1.0 - sigma_v.view(1, 1, 1, 1, 1)) * latents + sigma_v.view(1, 1, 1, 1, 1) * noise
        video_patches = patchify_video(noisy_latents).to(self.dtype)
        video_target = patchify_video(latents - noise)

        # Audio rows. ComfyUI never runs H3 without them, so even a still packs its 2-latent
        # (0.083 s) stereo block: silence (x0 = 0) noised at the audio sigma, no loss. A sample
        # with real sound gets a real target. pack_silent_audio: false reproduces the old
        # audio-free layout for A/B comparisons.
        use_audio = sample.audio is not None and self._train_audio()
        pack_audio = use_audio or bool(train_config.get("pack_silent_audio", True))
        audio_t = sample.audio_latents if pack_audio else 0
        audio_target = None
        if use_audio:
            a0 = sample.audio.to(device=self.device, dtype=torch.float32)
            a_noise = torch.randn_like(a0)
            audio_rows = (1.0 - sigma_a.view(1, 1, 1)) * a0 + sigma_a.view(1, 1, 1) * a_noise
            audio_target = a0 - a_noise
        else:
            audio_rows = sigma_a.view(1, 1, 1) * torch.randn(1, audio_t * 2, 32, device=self.device)
        audio_rows = audio_rows.to(self.dtype)

        # H3's DiT is conditioned on t = 1 - sigma (comfy/ldm/minimax/model.py). Row 0 is the
        # video/text timestep, row 1 the audio one (see _build_packed_layout).
        timestep = torch.cat([1.0 - sigma_v, 1.0 - sigma_a]) if audio_t else 1.0 - sigma_v
        layout = self._build_packed_layout(text_embeds.shape[1], f_lat, h_lat, w_lat, audio_t)

        model_output = self._forward(video_patches, audio_rows, text_embeds, timestep, layout)

        train_video = not sample.audio_only
        if self._do_guidance_loss():
            uncond_prompt = str(train_config.get("unconditional_prompt", ""))
            uncond_embeds = self.text_cache[uncond_prompt].to(device=self.device, dtype=self.dtype)
            # The empty prompt is one BOS token where the caption is several, and the packed
            # layout is sized from the text length -- indices, tags and every media row's
            # RoPE origin shift with it. Build its own.
            uncond_layout = self._build_packed_layout(uncond_embeds.shape[1], f_lat, h_lat, w_lat, audio_t)
            with torch.no_grad():
                uncond_output = self._forward(video_patches, audio_rows, uncond_embeds, timestep, uncond_layout)
            scale = float(train_config.get("guidance_loss_target", 3.5))
            schedule = str(train_config.get("guidance_loss_schedule", "sigma")).lower()
            cfg_zero = bool(train_config.get("guidance_loss_cfg_zero", False))
            if train_video:
                video_target = self._guided_target(
                    video_target, uncond_output.video, sigma_v, scale, schedule, cfg_zero
                )
            if audio_target is not None:
                audio_target = self._guided_target(
                    audio_target, uncond_output.audio, sigma_a, scale, schedule, cfg_zero
                )

        parts: Dict[str, float] = {"sigma": float(sigma_v)}
        loss = None
        if train_video:
            # Not 0 * loss for audio-only items: their video is a placeholder, and even a
            # zero-weighted term would build (and walk) the whole video backward graph.
            video_loss = torch.nn.functional.mse_loss(model_output.video.float(), video_target.float())
            parts["video"] = float(video_loss.detach())
            loss = video_loss
        if audio_target is not None:
            audio_loss = torch.nn.functional.mse_loss(model_output.audio.float(), audio_target.float())
            parts["audio"] = float(audio_loss.detach())
            weighted = float(train_config.get("audio_loss_weight", 1.0)) * audio_loss
            loss = weighted if loss is None else loss + weighted

        loss_weight_strength = float(train_config.get("loss_weight_strength", 1.0))
        if loss_weight_strength != 1.0:
            loss = loss * loss_weight_strength
        return loss, parts

    def train(self):
        """Main MiniMax H3 LoRA training loop (~0.8s/step)."""
        train_config = self.config["train"]
        num_epochs = train_config.get("num_epochs", 10)
        save_every = train_config.get("save_every", 500)

        self.lora_network, self.optimizer = self.accelerator.prepare(self.lora_network, self.optimizer)

        ema_decay = float(train_config.get("ema_decay", 0.0) or 0.0)
        if ema_decay > 0:
            self.ema = ParameterEMA(
                self.accelerator.unwrap_model(self.lora_network).trainable_parameters(),
                decay=ema_decay,
                warmup=bool(train_config.get("ema_warmup", True)),
            )
            self.run_metadata["ema_decay"] = ema_decay
            logger.info("EMA of the LoRA weights: decay %.4f (checkpoints save the average)", ema_decay)

        global_step = 0
        grad_accum = self.accelerator.gradient_accumulation_steps
        items_per_epoch = len(self._epoch_order())
        steps_per_epoch = math.ceil(items_per_epoch / max(grad_accum, 1))
        total_steps = num_epochs * steps_per_epoch
        self.run_metadata["steps_per_epoch"] = steps_per_epoch
        self.run_metadata["planned_total_steps"] = total_steps
        self.run_metadata["dataset_samples"] = len(self.samples)

        logger.info(
            "Starting accelerated MiniMax H3 LoRA training for %d epochs (%d total steps)...", num_epochs, total_steps
        )

        progress_bar = tqdm(total=total_steps, desc="Training Steps", disable=not self.is_main_process)
        shift_logged = False
        loss = None

        for epoch in range(num_epochs):
            self.lora_network.train()

            for index in self._epoch_order():
                sample = self.samples[index]
                _, _, _, h_lat, w_lat = sample.video.shape
                flow_shift = self._resolve_flow_shift(h_lat, w_lat)
                if not shift_logged:
                    shift_logged = True
                    self.run_metadata["resolved_sigma_shift"] = round(flow_shift, 6)
                    self.run_metadata["latent_size"] = f"{h_lat}x{w_lat}"
                    logger.info(
                        "Timestep schedule: %s, sigma shift %.3f (latent %dx%d); audio sigma follows "
                        "the inference coupling (video shift %.0f -> audio shift %.0f)",
                        train_config.get("timestep_distribution", "krea2_shift"),
                        flow_shift,
                        h_lat,
                        w_lat,
                        VIDEO_SIGMA_SHIFT,
                        AUDIO_SIGMA_SHIFT,
                    )

                with self.accelerator.accumulate(self.lora_network):
                    loss, parts = self._training_step(sample, flow_shift)
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.lora_network.trainable_parameters(), train_config.get("max_grad_norm", 1.0)
                        )
                        self.optimizer.step()
                        if self.lr_scheduler is not None:
                            self.lr_scheduler.step()
                        self.optimizer.zero_grad()
                        if self.ema is not None:
                            self.ema.update()

                if self.accelerator.sync_gradients:
                    global_step += 1
                    progress_bar.update(1)

                    detail = "".join(f", {name}: {parts[name]:.6f}" for name in ("video", "audio") if name in parts)
                    detail += f" [{sample.kind}]"

                    # For Prodigy, param_groups['lr'] is only the multiplier and sits at 1.0
                    # once warmup ends; the step size it actually adapted is lr * d. Report
                    # the effective rate so a stalled d (stuck near d0) is visible early
                    # rather than after the whole run.
                    group = self.optimizer.param_groups[0]
                    current_lr = group["lr"]
                    prodigy_d = group.get("d")
                    # accelerate's AcceleratedOptimizer does not forward custom methods
                    inner_optimizer = getattr(self.optimizer, "optimizer", self.optimizer)
                    if hasattr(inner_optimizer, "get_avg_learning_rate"):
                        adapted_lr = inner_optimizer.get_avg_learning_rate()
                        self.progress_logger.info(
                            "Step: %d/%d [%d%%], Loss: %.6f, Epoch: %d/%d, LR: %.2e (automagic3)%s",
                            global_step,
                            total_steps,
                            int(global_step / total_steps * 100),
                            loss.item(),
                            epoch + 1,
                            num_epochs,
                            adapted_lr,
                            detail,
                        )
                    elif prodigy_d is not None:
                        self.progress_logger.info(
                            "Step: %d/%d [%d%%], Loss: %.6f, Epoch: %d/%d, LR: %.2e (mult %.3f x d %.2e)%s",
                            global_step,
                            total_steps,
                            int(global_step / total_steps * 100),
                            loss.item(),
                            epoch + 1,
                            num_epochs,
                            current_lr * prodigy_d,
                            current_lr,
                            prodigy_d,
                            detail,
                        )
                    else:
                        self.progress_logger.info(
                            "Step: %d/%d [%d%%], Loss: %.6f, Epoch: %d/%d, LR: %.2e%s",
                            global_step,
                            total_steps,
                            int(global_step / total_steps * 100),
                            loss.item(),
                            epoch + 1,
                            num_epochs,
                            current_lr,
                            detail,
                        )

                    if global_step % save_every == 0:
                        self.save_checkpoint(global_step)

                progress_bar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "step": global_step, "epoch": f"{epoch + 1}/{num_epochs}"}
                )

        progress_bar.close()
        self.save_checkpoint("final")
        logger.info("MiniMax H3 LoRA training completed!")

    def save_checkpoint(self, step):
        """Save LoRA checkpoint safetensors file."""
        if not self.is_main_process:
            return

        output_dir = Path(self.config["output"]["path"])
        output_dir.mkdir(parents=True, exist_ok=True)

        save_path = output_dir / f"minimax_h3_lora_step_{step}.safetensors"
        save_dtype = getattr(torch, str(self.config.get("output", {}).get("dtype", "bfloat16")))
        network = self.accelerator.unwrap_model(self.lora_network)
        if self.ema is not None:
            # Checkpoints carry the averaged weights; training continues on the live ones.
            with self.ema.applied():
                state_dict = network.get_state_dict(save_dtype=save_dtype)
        else:
            state_dict = network.get_state_dict(save_dtype=save_dtype)

        runtime = dict(self.run_metadata)
        runtime["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        runtime["save_dtype"] = str(save_dtype).replace("torch.", "")
        metadata = build_metadata(
            self.config,
            step=step,
            runtime=runtime,
            embed_config=bool(self.config.get("output", {}).get("embed_config", True)),
        )

        save_file(state_dict, str(save_path), metadata=metadata)
        self.last_saved_path = str(save_path)
        logger.info("Saved LoRA checkpoint to %s", save_path)

    def run(self):
        """Run training pipeline."""
        try:
            set_seed(self.config.get("seed", 42))

            if self.dry_run:
                self.dry_run_training()
                return

            self.precache_dataset()
            self.load_transformer_and_lora()
            self.setup_optimizer()
            self.train()
        except KeyboardInterrupt:
            logger.info("Training interrupted by user. Exiting...")
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error("An error occurred during training: %s", e, exc_info=True)
            raise e


def main():
    parser = argparse.ArgumentParser(description="Train MiniMax H3 LoRA in Maltrainer")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--dry-run", action="store_true", help="Simulate training without loading models")
    parser.add_argument(
        "--dry-run-step-time", type=float, default=0.025, help="Seconds per simulated step during dry run"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.get("logging", {}))

    if config.get("output"):
        save_config_to_output(config, config["output"]["path"])

    trainer = MiniMaxH3LoRATrainer(
        config,
        dry_run=args.dry_run,
        dry_run_step_time=args.dry_run_step_time,
    )
    trainer.run()


if __name__ == "__main__":
    main()
