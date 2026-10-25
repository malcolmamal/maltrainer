"""The half of an image LoRA trainer that does not depend on the model (FLUX.3 [klein], Krea 2, MOSS-Image).

A trainer adds the model side: ``precache_dataset`` (fills ``samples``, ``latent_cache`` and ``text_cache``),
``load_transformer_and_lora`` (sets ``lora_network``), and ``_training_step`` (one sample's loss). The step
loop and its hooks are in loop.py.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any, ClassVar

import torch
from accelerate.utils import set_seed
from tqdm import tqdm

from maltrainer.data.minimax_dataset import MediaSource, count_media_files, scan_media_sources
from maltrainer.training.loop import StepLoop
from maltrainer.training.optimizer import build_lora_optimizer, warmup_scheduler
from maltrainer.training.progress import simulate_steps
from maltrainer.training.saving import lora_save_path, write_lora
from maltrainer.training.session import log_device, make_accelerator, open_training_log, start_metadata
from maltrainer.training.steps import epoch_order, plan_training_steps
from maltrainer.utils.ema import OffloadedParameterEMA, ParameterEMA

logger = logging.getLogger(__name__)


class ImageLoRATrainer(StepLoop):
    model_type: ClassVar[str]  # run_metadata["model_type"]
    default_name: ClassVar[str]  # the checkpoint name when output.name is not set
    batch_one_reason: ClassVar[str]  # the error for train.batch_size != 1
    polarity_history: ClassVar[int] = 16  # Automagic3's default sign window
    strict_optimizer: ClassVar[bool] = False  # an unknown train.optimizer: an error, or AdamW

    def __init__(self, config: dict[str, Any], dry_run: bool = False, dry_run_step_time: float = 0.025) -> None:
        self.config = config
        self.dry_run, self.dry_run_step_time = dry_run, dry_run_step_time
        self.last_saved_path: str | None = None
        self.lr_scheduler: Any = None
        self.optimizer: Any = None
        self.ema: ParameterEMA | OffloadedParameterEMA | None = None
        self.run_metadata = start_metadata(self.model_type)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, config.get("dtype", "bfloat16"))
        self.samples: list[tuple[str, str]] = []  # (key, caption)
        self.latent_cache: dict[str, torch.Tensor] = {}
        self.text_cache: dict[str, torch.Tensor] = {}
        if int(config.get("train", {}).get("batch_size", 1) or 1) != 1:
            raise ValueError(self.batch_one_reason)
        self.accelerator = make_accelerator(config)
        self.is_main_process = self.accelerator.is_main_process
        if self.is_main_process:
            log_device(self.device)
        self.progress_logger = open_training_log(config)

    # ------------------------------------------------------------------ data

    def _sources(self) -> list[MediaSource]:
        dataset_config = self.config["dataset"]
        return scan_media_sources(
            dataset_config["path"],
            kinds=["image"],
            trigger_token=dataset_config.get("trigger"),
            default_caption=dataset_config.get("default_caption", ""),
        )

    def _encode_prompts(self, encoder: Any, prompts: Iterable[str]) -> None:
        """Each prompt's embedding into text_cache, in sorted order."""
        for caption in tqdm(sorted(prompts), desc="Caching prompts", disable=not self.is_main_process):
            self.text_cache[caption] = encoder.encode(caption, self.device, self.dtype).cpu()

    def _extra_prompts(self) -> list[str]:
        """Captions to encode beside the samples' own (previews, a finetune's class captions); none here."""
        return []

    def _sample(self, index: int) -> tuple[str, str]:
        """The sample's cache key and caption; ``train.caption_dropout`` swaps the caption for the empty prompt."""
        key, caption = self.samples[index]
        dropout = float(self.config["train"].get("caption_dropout", 0.0) or 0.0)
        if dropout > 0 and torch.rand(()).item() < dropout:
            caption = ""
        return key, caption

    def _epoch_order(self) -> list[int]:
        repeats = self.config["dataset"].get("repeats", 1)
        return epoch_order(len(self.samples), repeats, self.accelerator.num_processes, self.accelerator.process_index)

    # ------------------------------------------------------------------ model and optimizer

    def _lora_ready(self) -> None:
        """lora_network built: its adapters into ``lora.dtype`` on the device, their size into the metadata."""
        self.lora_dtype = getattr(torch, str(self.config.get("lora", {}).get("dtype", "float32")))
        self.lora_network.set_lora_dtype(self.device, self.lora_dtype)
        trainable = sum(p.numel() for p in self.lora_network.trainable_parameters())
        self.run_metadata.update(
            lora_layers=len(self.lora_network.lora_layers),
            trainable_parameters=trainable,
            lora_dtype=str(self.lora_dtype).replace("torch.", ""),
        )
        logger.info("LoRA: %.1fM trainable parameters in %s", trainable / 1e6, self.lora_dtype)

    def setup_optimizer(self) -> None:
        train_config = self.config["train"]
        self.optimizer, resolved = build_lora_optimizer(
            self.lora_network, train_config, self.polarity_history, self.strict_optimizer
        )
        self.run_metadata["optimizer_resolved"] = resolved
        logger.info("Optimizer: %s", resolved)
        self.lr_scheduler = warmup_scheduler(self.optimizer, train_config)

    # ------------------------------------------------------------------ saving

    def save_checkpoint(self, step: int | str) -> bool | None:
        if not self.is_main_process:
            return None
        output_config = self.config.get("output", {})
        save_path = lora_save_path(output_config, step, self.default_name)
        save_dtype = getattr(torch, str(output_config.get("dtype", "bfloat16")))
        network = self.accelerator.unwrap_model(self.lora_network)
        if self.ema is not None:
            with self.ema.applied():
                state_dict = network.get_state_dict(save_dtype)
        else:
            state_dict = network.get_state_dict(save_dtype)
        write_lora(self.config, state_dict, save_path, step, self.run_metadata, save_dtype)
        self.last_saved_path = str(save_path)
        logger.info("Saved LoRA checkpoint to %s", save_path)
        return None

    # ------------------------------------------------------------------ dry run / entry

    def _dry_run_samples(self) -> int:
        """One epoch's samples, counted from the dataset folder without decoding anything."""
        dataset_config = self.config["dataset"]
        repeats = max(int(dataset_config.get("repeats", 1) or 1), 1)
        return max(count_media_files(dataset_config.get("path", ""), ["image"]), 1) * repeats

    def simulate(self) -> None:
        """Plan from the real dataset folder and tick through the steps without loading models."""
        samples = self._dry_run_samples()
        plan = plan_training_steps(samples, self.config["train"])
        logger.info("Dry run: %d samples, %d steps/epoch -> %d epochs, %d total steps", samples, *plan)
        simulate_steps(self.progress_logger, plan, self.dry_run_step_time, self.is_main_process)

    def run(self) -> None:
        try:
            set_seed(self.config.get("seed", 42))
            if self.dry_run:
                self.simulate()
                return
            self.precache_dataset()
            if not self.samples:
                raise ValueError("No usable images: every file in the dataset failed to load")
            self.load_transformer_and_lora()
            self.setup_optimizer()
            self.train()
        except KeyboardInterrupt:
            logger.info("Training interrupted by user. Exiting...")
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error("An error occurred during training: %s", e, exc_info=True)
            raise

    def precache_dataset(self) -> None:
        raise NotImplementedError

    def load_transformer_and_lora(self) -> None:
        raise NotImplementedError
