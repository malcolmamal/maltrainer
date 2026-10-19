#!/usr/bin/env python3
"""
Z-Image LoRA Training Script
Trains LoRA adapters for Z-Image (base) models with YAML configuration support.
"""

import argparse
import inspect
import json
import logging
import math
import os
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL
from diffusers.models.transformers import ZImageTransformer2DModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from maltrainer.data.dataset import ZImageDataset
from maltrainer.data.walk import count_files
from maltrainer.models.lora import LoRANetwork
from maltrainer.models.scheduler import get_flow_match_scheduler
from maltrainer.utils.config import load_config, save_config_to_output
from maltrainer.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


class ZImageLoRATrainer:
    """Trainer for Z-Image LoRA models (base or turbo)."""

    def __init__(self, config: dict, dry_run: bool = False, dry_run_step_time: float = 0.025):
        self.config = config
        self.dry_run = dry_run
        self.dry_run_step_time = dry_run_step_time
        self.last_saved_path = None
        self.lr_scheduler = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, config.get("dtype", "bfloat16"))

        # Initialize accelerator
        self.accelerator = Accelerator(
            mixed_precision=config.get("mixed_precision", "bf16"),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        )

        self.is_main_process = self.accelerator.is_main_process

        # Log device info
        if self.is_main_process:
            device_name = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU"
            logger.info(f"Training on device: {self.device} ({device_name})")
            if self.device.type == "cuda":
                logger.info(
                    f"Total VRAM: {torch.cuda.get_device_properties(self.device).total_memory / 1024**3:.2f} GB"
                )

        # Add file logging if configured
        log_dir = Path(config.get("output", {}).get("path", "./output"))
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "training.log")
        file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logging.getLogger().addHandler(file_handler)

        # Create a dedicated logger for progress updates that writes only to file
        self.progress_logger = logging.getLogger("progress")
        self.progress_logger.setLevel(logging.INFO)
        self.progress_logger.addHandler(file_handler)
        self.progress_logger.propagate = False

    def _resolve_hf_path(self, path_value: str) -> str:
        if os.path.exists(path_value):
            return path_value
        parts = path_value.split("/")
        if len(parts) < 3:
            raise ValueError(f"Adapter path {path_value} is not a valid local path or hub path.")
        repo_id = "/".join(parts[:2])
        filename = "/".join(parts[2:])
        return hf_hub_download(repo_id=repo_id, filename=filename)

    def _merge_lora_into_transformer(self, lora_state_dict: dict, merge_weight: float = 1.0) -> int:
        module_map = dict(self.transformer.named_modules())
        merged = 0
        for key, lora_a in lora_state_dict.items():
            if not key.endswith(".lora_A.weight"):
                continue
            base_key = key[: -len(".lora_A.weight")]
            lora_b_key = f"{base_key}.lora_B.weight"
            alpha_key = f"{base_key}.alpha"
            if lora_b_key not in lora_state_dict:
                continue

            module_name = base_key
            if module_name.startswith("diffusion_model."):
                module_name = module_name[len("diffusion_model.") :]
            if module_name.startswith("transformer."):
                module_name = module_name[len("transformer.") :]

            module = module_map.get(module_name)
            if module is None or not isinstance(module, torch.nn.Linear):
                continue

            lora_b = lora_state_dict[lora_b_key]
            rank = int(lora_a.shape[0])
            alpha = float(lora_state_dict.get(alpha_key, rank))
            scale = (alpha / rank) * float(merge_weight)

            with torch.no_grad():
                lora_a = lora_a.to(device=module.weight.device, dtype=module.weight.dtype)
                lora_b = lora_b.to(device=module.weight.device, dtype=module.weight.dtype)
                module.weight.add_(torch.matmul(lora_b, lora_a) * scale)
            merged += 1
        return merged

    def load_model(self):
        """Load Z-Image model components."""
        logger.info("Loading Z-Image model...")

        model_config = self.config.get("model", {})
        model_path = model_config["name_or_path"]
        base_model_path = model_config.get("extras_name_or_path", model_path)
        assistant_lora_path = model_config.get("assistant_lora_path", None)

        # Load transformer
        transformer = ZImageTransformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=self.dtype,
        )
        transformer.enable_gradient_checkpointing()

        # Merge training adapter (for Z-Image Turbo) if provided
        if assistant_lora_path:
            resolved_path = self._resolve_hf_path(str(assistant_lora_path))
            lora_state_dict = load_file(resolved_path)
            merged = self._merge_lora_into_transformer(lora_state_dict, merge_weight=1.0)
            logger.info("Merged %d adapter layers from %s", merged, assistant_lora_path)

        # Load text encoder
        text_encoder = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            subfolder="text_encoder",
            torch_dtype=self.dtype,
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            subfolder="tokenizer",
        )

        # Load VAE
        vae = AutoencoderKL.from_pretrained(
            base_model_path,
            subfolder="vae",
            torch_dtype=torch.float32,  # VAE always float32
        )

        # Set requires_grad
        transformer.requires_grad_(False)
        text_encoder.requires_grad_(False)
        vae.requires_grad_(False)

        # Move everything to GPU for the 5090
        transformer.to(self.device)
        text_encoder.to(self.device)
        vae.to(self.device)

        self.transformer = transformer
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.vae = vae

        logger.info("Model loaded successfully")

    def create_lora_network(self):
        """Create LoRA network for the transformer."""
        logger.info("Creating LoRA network...")

        lora_config = self.config.get("lora", {})

        self.lora_network = LoRANetwork(
            transformer=self.transformer,
            rank=lora_config.get("rank", 4),
            alpha=lora_config.get("alpha", 1.0),
            target_modules=lora_config.get("target_modules", ["to_q", "to_k", "to_v", "to_out"]),
        )
        self.lora_network.prefix = lora_config.get("prefix", "")

        self.lora_network.to(self.device, dtype=self.dtype)
        logger.info(f"LoRA network created with rank={lora_config.get('rank', 4)}")

    def create_dataset(self):
        """Create training dataset."""
        logger.info("Creating dataset...")

        dataset_config = self.config["dataset"]

        self.train_dataset = ZImageDataset(
            data_root=dataset_config["path"],
            resolution=dataset_config.get("resolution", 512),
            center_crop=dataset_config.get("center_crop", True),
            random_flip=dataset_config.get("random_flip", False),
            trigger_token=dataset_config.get("trigger", None),
            repeats=dataset_config.get("repeats", 1),
        )

        default_caption = dataset_config.get("default_caption", "")
        trigger = dataset_config.get("trigger", "").strip()

        if default_caption:
            # Smart replacement: if caption is empty or just the trigger, use default_caption
            # Note: Dataset already prepends trigger if it exists
            new_captions = []
            for c in self.train_dataset.captions:
                c_clean = c.strip()
                # If it is empty or just the trigger token, it means no specific caption was found
                if not c_clean or c_clean == trigger:
                    # Construct full caption
                    if trigger and default_caption and trigger not in default_caption:
                        new_captions.append(f"{trigger} {default_caption}")
                    else:
                        new_captions.append(default_caption)
                else:
                    new_captions.append(c)
            self.train_dataset.captions = new_captions

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config["train"].get("batch_size", 1),
            shuffle=True,
            num_workers=dataset_config.get("num_workers", 0),
        )

        trigger = dataset_config.get("trigger", None)
        empty_captions = sum(1 for c in self.train_dataset.captions if not c)
        if not trigger and empty_captions > 0:
            logger.warning(
                "Detected %d empty captions and no trigger token. "
                "Training may fail to learn a consistent concept. "
                "Add dataset.trigger or provide caption .txt files.",
                empty_captions,
            )

        logger.info(f"Dataset created with {len(self.train_dataset)} samples")

    def _count_dataset_samples(self) -> int:
        dataset_config = self.config.get("dataset", {})
        data_root = Path(str(dataset_config.get("path", "")))
        if not data_root.exists():
            logger.warning("Dataset path does not exist: %s", data_root)
            return 1
        supported_exts = {".jpg", ".jpeg", ".png", ".webp"}
        sample_count = count_files(data_root, supported_exts)
        repeats = int(dataset_config.get("repeats", 1) or 1)
        sample_count = max(sample_count, 1) * max(repeats, 1)
        return sample_count

    def _sample_timesteps(self, batch_size: int, num_train_timesteps: int) -> torch.Tensor:
        train_config = self.config.get("train", {})
        distribution = str(train_config.get("timestep_distribution", "uniform")).lower()
        if distribution in {"logit_normal", "logit-normal", "logitnormal"}:
            mean = float(train_config.get("timestep_logit_mean", 0.0))
            std = float(train_config.get("timestep_logit_std", 1.0))
            z = torch.randn(batch_size, device=self.device, dtype=self.dtype) * std + mean
            timesteps = torch.sigmoid(z).clamp(0.0, 1.0)
        else:
            timesteps = torch.rand(batch_size, device=self.device, dtype=self.dtype)

        shift = float(train_config.get("timestep_shift", 1.0))
        if shift != 1.0:
            timesteps = (timesteps * shift).clamp(0.0, 1.0)

        if num_train_timesteps > 0:
            timesteps = (timesteps * num_train_timesteps).floor() / num_train_timesteps
        return timesteps

    def dry_run_training(self):
        """Simulate training without loading or downloading models."""
        train_config = self.config.get("train", {})
        num_epochs = int(train_config.get("num_epochs", 10) or 1)
        batch_size = int(train_config.get("batch_size", 1) or 1)
        grad_accum = int(
            train_config.get("gradient_accumulation_steps", self.config.get("gradient_accumulation_steps", 1)) or 1
        )
        num_samples = self._count_dataset_samples()
        steps_per_epoch = math.ceil((num_samples / max(batch_size, 1)) / max(grad_accum, 1))
        total_steps = max(num_epochs * steps_per_epoch, 1)
        save_every = int(train_config.get("save_every", 500) or 0)
        step_time = max(float(self.dry_run_step_time), 0.0)
        output_dir = Path(self.config.get("output", {}).get("path", "./output"))
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Dry run enabled: skipping model download/loading and training compute.")
        logger.info(
            "Simulating %d steps (epochs=%d, steps/epoch=%d, batch_size=%d, grad_accum=%d, step_time=%.2fs).",
            total_steps,
            num_epochs,
            steps_per_epoch,
            batch_size,
            grad_accum,
            step_time,
        )

        start_time = time.time()
        progress_bar = tqdm(
            total=total_steps,
            desc="Dry Run Steps",
            disable=not self.is_main_process,
        )
        for step in range(1, total_steps + 1):
            if step_time > 0:
                time.sleep(step_time)
            progress_bar.update(1)

            if save_every > 0 and step % save_every == 0:
                self._write_dry_run_checkpoint(
                    output_dir=output_dir,
                    step=step,
                    total_steps=total_steps,
                    steps_per_epoch=steps_per_epoch,
                    num_epochs=num_epochs,
                    start_time=start_time,
                )
        progress_bar.close()
        self._write_dry_run_checkpoint(
            output_dir=output_dir,
            step="final",
            total_steps=total_steps,
            steps_per_epoch=steps_per_epoch,
            num_epochs=num_epochs,
            start_time=start_time,
        )
        logger.info("Dry run completed.")

    def _write_dry_run_checkpoint(
        self, output_dir: Path, step, total_steps: int, steps_per_epoch: int, num_epochs: int, start_time: float
    ):
        dataset_config = self.config.get("dataset", {})
        train_config = self.config.get("train", {})
        elapsed = max(time.time() - start_time, 0.0)
        completed_steps = total_steps if step == "final" else int(step)
        progress = min(completed_steps / max(total_steps, 1), 1.0)
        eta_seconds = max((total_steps - completed_steps) * float(self.dry_run_step_time), 0.0)

        payload = {
            "mode": "dry_run",
            "kind": "lora",
            "step": step,
            "total_steps": total_steps,
            "steps_per_epoch": steps_per_epoch,
            "num_epochs": num_epochs,
            "progress": {
                "completed_steps": completed_steps,
                "percent": round(progress * 100, 2),
                "elapsed_seconds": round(elapsed, 4),
                "eta_seconds": round(eta_seconds, 4),
                "step_time_seconds": float(self.dry_run_step_time),
            },
            "config": self.config,
            "derived": {
                "dataset_path": str(dataset_config.get("path", "")),
                "dataset_resolution": dataset_config.get("resolution", 512),
                "batch_size": train_config.get("batch_size", 1),
                "gradient_accumulation_steps": train_config.get(
                    "gradient_accumulation_steps", self.config.get("gradient_accumulation_steps", 1)
                ),
                "save_every": train_config.get("save_every", 500),
                "dtype": str(self.config.get("dtype", "bfloat16")),
                "mixed_precision": str(self.config.get("mixed_precision", "bf16")),
                "output_path": str(output_dir),
            },
        }

        save_path = output_dir / f"zimage_lora_step_{step}.safetensors"
        save_path.write_text(json.dumps(payload, indent=2))
        self.last_saved_path = str(save_path)

    def setup_optimizer(self):
        """Setup optimizer and scheduler."""
        train_config = self.config["train"]

        def _to_float(value, default: float) -> float:
            if value is None:
                return float(default)
            if isinstance(value, (int, float)):
                return float(value)
            return float(str(value))

        def _to_int(value, default: int) -> int:
            if value is None:
                return int(default)
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)
            return int(str(value))

        def _to_bool(value, default: bool) -> bool:
            if value is None:
                return bool(default)
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            value_str = str(value).strip().lower()
            if value_str in {"true", "1", "yes", "y", "on"}:
                return True
            if value_str in {"false", "0", "no", "n", "off"}:
                return False
            return bool(default)

        lr = _to_float(train_config.get("learning_rate", 1e-4), 1e-4)
        beta1 = _to_float(train_config.get("adam_beta1", 0.9), 0.9)
        beta2 = _to_float(train_config.get("adam_beta2", 0.999), 0.999)
        weight_decay = _to_float(train_config.get("weight_decay", 0.01), 0.01)
        eps = _to_float(train_config.get("adam_epsilon", 1e-8), 1e-8)

        optimizer_name = str(train_config.get("optimizer", "adamw")).lower()
        use_weight_decay = 0.0 if train_config.get("weight_decay_exclude_lora", False) else weight_decay

        if optimizer_name.startswith("prodigy"):
            try:
                from prodigyopt import Prodigy

                beta3_raw = train_config.get("prodigy_beta3", None)
                beta3 = None if beta3_raw is None else _to_float(beta3_raw, 0.0)
                decouple = _to_bool(train_config.get("prodigy_decouple", True), True)
                use_bias_correction = _to_bool(train_config.get("prodigy_use_bias_correction", False), False)
                safeguard_warmup = _to_bool(train_config.get("prodigy_safeguard_warmup", False), False)
                d0 = _to_float(train_config.get("prodigy_d0", 1e-6), 1e-6)
                d_coef = _to_float(train_config.get("prodigy_d_coef", 1.0), 1.0)
                growth_rate = _to_float(train_config.get("prodigy_growth_rate", float("inf")), float("inf"))
                slice_p = _to_int(train_config.get("prodigy_slice_p", 1), 1)

                self.optimizer = Prodigy(
                    self.lora_network.parameters(),
                    lr=lr,
                    betas=(beta1, beta2),
                    beta3=beta3,
                    eps=eps,
                    weight_decay=use_weight_decay,
                    decouple=decouple,
                    use_bias_correction=use_bias_correction,
                    safeguard_warmup=safeguard_warmup,
                    d0=d0,
                    d_coef=d_coef,
                    growth_rate=growth_rate,
                    slice_p=slice_p,
                )
                logger.info("Using optimizer: %s", optimizer_name)
            except Exception as e:
                logger.warning(
                    "Failed to load prodigy optimizer (%s). Falling back to AdamW. Error: %s",
                    optimizer_name,
                    e,
                )
                self.optimizer = AdamW(
                    self.lora_network.parameters(),
                    lr=lr,
                    betas=(beta1, beta2),
                    weight_decay=use_weight_decay,
                    eps=eps,
                )
        elif optimizer_name in {"adamw8bit", "adam8bit"}:
            try:
                import bitsandbytes as bnb

                optimizer_cls = bnb.optim.AdamW8bit if optimizer_name == "adamw8bit" else bnb.optim.Adam8bit
                self.optimizer = optimizer_cls(
                    self.lora_network.parameters(),
                    lr=lr,
                    betas=(beta1, beta2),
                    weight_decay=use_weight_decay,
                    eps=eps,
                )
                logger.info("Using optimizer: %s", optimizer_name)
            except Exception as e:
                logger.warning(
                    "Failed to load bitsandbytes optimizer (%s). Falling back to AdamW. Error: %s",
                    optimizer_name,
                    e,
                )
                self.optimizer = AdamW(
                    self.lora_network.parameters(),
                    lr=lr,
                    betas=(beta1, beta2),
                    weight_decay=use_weight_decay,
                    eps=eps,
                )
        else:
            self.optimizer = AdamW(
                self.lora_network.parameters(),
                lr=lr,
                betas=(beta1, beta2),
                weight_decay=use_weight_decay,
                eps=eps,
            )

        self.noise_scheduler = get_flow_match_scheduler()

        warmup_steps = int(train_config.get("learning_rate_warmup_steps", 0) or 0)
        if warmup_steps > 0:

            def lr_lambda(step: int) -> float:
                return min((step + 1) / warmup_steps, 1.0)

            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lr_lambda,
            )

    def train(self):
        """Main training loop."""
        train_config = self.config["train"]
        aitk_weighting = [
            0.905706524848938,
            0.9097874164581299,
            0.9251009821891785,
            0.9399133920669556,
            0.9497355818748474,
            0.962195873260498,
            0.9691638946533203,
            0.9927358627319336,
            1.01659095287323,
            1.0392684936523438,
            1.0429067611694336,
            1.0677919387817383,
            1.092896580696106,
            1.1149251461029053,
            1.1015851497650146,
            1.1209120750427246,
            1.1399472951889038,
            1.1559219360351562,
            1.143755555152893,
            1.160578727722168,
            1.1761468648910522,
            1.1895899772644043,
            1.1867371797561646,
            1.1993824243545532,
            1.2113513946533203,
            1.2201216220855713,
            1.221794605255127,
            1.2329251766204834,
            1.2426395416259766,
            1.251118540763855,
            1.2562459707260132,
            1.2668883800506592,
            1.2760146856307983,
            1.2822540998458862,
            1.2892439365386963,
            1.2972776889801025,
            1.3042182922363281,
            1.3104143142700195,
            1.3190876245498657,
            1.3253265619277954,
            1.3295484781265259,
            1.3324649333953857,
            1.3436106443405151,
            1.3482736349105835,
            1.351183533668518,
            1.3556060791015625,
            1.359933614730835,
            1.3625402450561523,
            1.3641390800476074,
            1.3723753690719604,
            1.3748365640640259,
            1.3776681423187256,
            1.3802807331085205,
            1.3848408460617065,
            1.3877429962158203,
            1.390969157218933,
            1.3927756547927856,
            1.4031920433044434,
            1.4064390659332275,
            1.4097232818603516,
            1.4132285118103027,
            1.4208053350448608,
            1.4254504442214966,
            1.4294012784957886,
            1.4323071241378784,
            1.44380521774292,
            1.4465110301971436,
            1.4490883350372314,
            1.451446533203125,
            1.458960771560669,
            1.460111379623413,
            1.4603426456451416,
            1.4595434665679932,
            1.4694839715957642,
            1.470526099205017,
            1.4705318212509155,
            1.4697961807250977,
            1.477265477180481,
            1.4771337509155273,
            1.4768965244293213,
            1.4760032892227173,
            1.4871419668197632,
            1.4877341985702515,
            1.488408088684082,
            1.489357829093933,
            1.4884581565856934,
            1.4882304668426514,
            1.4878745079040527,
            1.4949846267700195,
            1.4957915544509888,
            1.4951081275939941,
            1.4950779676437378,
            1.4995663166046143,
            1.4994632005691528,
            1.4986882209777832,
            1.4976568222045898,
            1.5038518905639648,
            1.503743052482605,
            1.502982258796692,
            1.502566933631897,
            1.5068002939224243,
            1.506575584411621,
            1.5067178010940552,
            1.5050104856491089,
            1.508437991142273,
            1.5073161125183105,
            1.506223440170288,
            1.504751205444336,
            1.512485384941101,
            1.5121595859527588,
            1.5116060972213745,
            1.5102864503860474,
            1.5156408548355103,
            1.5157924890518188,
            1.5142825841903687,
            1.5141233205795288,
            1.5191627740859985,
            1.518998622894287,
            1.5177574157714844,
            1.516713261604309,
            1.5186537504196167,
            1.5179635286331177,
            1.5159885883331299,
            1.5150446891784668,
            1.5226575136184692,
            1.5215232372283936,
            1.5201103687286377,
            1.5225480794906616,
            1.5215368270874023,
            1.520209789276123,
            1.5184354782104492,
            1.5220975875854492,
            1.5209708213806152,
            1.519667625427246,
            1.5177332162857056,
            1.5203157663345337,
            1.519271731376648,
            1.5175830125808716,
            1.5161852836608887,
            1.517741322517395,
            1.5163099765777588,
            1.5150357484817505,
            1.513055682182312,
            1.5154107809066772,
            1.513993501663208,
            1.5126358270645142,
            1.510875940322876,
            1.5125701427459717,
            1.510590672492981,
            1.5086392164230347,
            1.5068501234054565,
            1.5094420909881592,
            1.5080277919769287,
            1.5060293674468994,
            1.5040230751037598,
            1.504892110824585,
            1.5031654834747314,
            1.5013214349746704,
            1.499170184135437,
            1.500420331954956,
            1.4979915618896484,
            1.4959659576416016,
            1.494127631187439,
            1.4961415529251099,
            1.494597315788269,
            1.4926577806472778,
            1.4904958009719849,
            1.4913445711135864,
            1.489889144897461,
            1.4880430698394775,
            1.4873780012130737,
            1.485144019126892,
            1.4830609560012817,
            1.48123037815094,
            1.483134150505066,
            1.4808504581451416,
            1.4793643951416016,
            1.4773707389831543,
            1.4774049520492554,
            1.4752962589263916,
            1.4733281135559082,
            1.4712235927581787,
            1.470260500907898,
            1.4684780836105347,
            1.4657323360443115,
            1.4639259576797485,
            1.4644291400909424,
            1.4621423482894897,
            1.4604870080947876,
            1.4585931301116943,
            1.458650827407837,
        ]
        num_epochs = train_config.get("num_epochs", 10)
        save_every = train_config.get("save_every", 500)

        # Log device and model info right before training
        if self.is_main_process:
            device_name = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU"
            print("--- Training Configuration ---")
            print(f"Device: {device_name}")
            print(f"Dtype: {self.dtype}")
            print(f"Transformer: {sum(p.numel() for p in self.transformer.parameters()) / 1e9:.2f}B params")
            print(f"Text Encoder: {sum(p.numel() for p in self.text_encoder.parameters()) / 1e9:.2f}B params")
            print("-------------------------------")

        # Prepare for training
        self.lora_network, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.lora_network, self.optimizer, self.train_dataloader
        )

        global_step = 0
        grad_accum = self.accelerator.gradient_accumulation_steps
        steps_per_epoch = math.ceil(len(self.train_dataloader) / max(grad_accum, 1))
        total_steps = num_epochs * steps_per_epoch
        num_train_timesteps = self.noise_scheduler.config.num_train_timesteps

        logger.info("Starting training...")

        progress_bar = tqdm(
            total=total_steps,
            desc="Training Steps",
            disable=not self.is_main_process,
        )

        for epoch in range(num_epochs):
            self.lora_network.train()

            for batch in self.train_dataloader:
                with self.accelerator.accumulate(self.lora_network):
                    # Get pixel values and captions from batch
                    pixel_values = batch["pixel_values"].to(self.device)
                    captions = batch["caption"]

                    # Encode with VAE
                    with torch.no_grad():
                        latents = self.vae.encode(pixel_values).latent_dist.sample()
                        # Z-Image specific scaling/shifting logic
                        # Standard Diffusers VAE scaling is usually latents * config.scaling_factor
                        # But Z-Image uses specific shift and scale factors
                        # Source: Musubi-Tuner zimage_config.py
                        zimage_shift = 0.1159
                        zimage_scale = 0.3611
                        latents = (latents - zimage_shift) * zimage_scale
                        latents = latents.to(dtype=self.dtype)

                        # Tokenize and encode text
                        # Use "longest" padding so for batch=1 we have no padding tokens
                        # This avoids the issue where the model doesn't support attention_mask
                        tokens = self.tokenizer(
                            [
                                self.tokenizer.apply_chat_template(
                                    [{"role": "user", "content": c}],
                                    tokenize=False,
                                    add_generation_prompt=True,
                                    # Z-Image specific argument found in reference implementation
                                    **(
                                        {"enable_thinking": True}
                                        if "enable_thinking"
                                        in inspect.signature(self.tokenizer.apply_chat_template).parameters
                                        or "kwargs" in inspect.signature(self.tokenizer.apply_chat_template).parameters
                                        else {}
                                    ),
                                )
                                if hasattr(self.tokenizer, "apply_chat_template")
                                else c
                                for c in captions
                            ],
                            padding="longest",
                            truncation=True,
                            max_length=self.tokenizer.model_max_length,
                            return_tensors="pt",
                        ).to(self.device)

                    with torch.no_grad():
                        # Use the base model to get hidden states (last layer outputs)
                        base_text_encoder = getattr(self.text_encoder, "model", self.text_encoder)
                        te_out = base_text_encoder(
                            tokens.input_ids,
                            attention_mask=tokens.attention_mask,
                            output_hidden_states=True,
                        )
                        text_embeds = te_out.hidden_states[-2]

                    # Sample noise
                    noise = torch.randn_like(latents)

                    # Optional noise/latent transforms (AI-Toolkit style)
                    latent_multiplier = float(train_config.get("latent_multiplier", 1.0))
                    if latent_multiplier != 1.0:
                        latents = latents * latent_multiplier

                    noise_multiplier = float(train_config.get("noise_multiplier", 1.0))
                    if noise_multiplier != 1.0:
                        noise = noise * noise_multiplier

                    if train_config.get("dynamic_noise_offset", False):
                        latents_channel_mean = latents.mean(dim=(2, 3), keepdim=True) / 2
                        noise = noise + latents_channel_mean

                    random_noise_multiplier = float(train_config.get("random_noise_multiplier", 0.0))
                    if random_noise_multiplier > 0.0:
                        s = (noise.shape[0], noise.shape[1], 1, 1)
                        noise = noise * (
                            1 + torch.randn(s, device=noise.device, dtype=noise.dtype) * random_noise_multiplier
                        )

                    random_noise_shift = float(train_config.get("random_noise_shift", 0.0))
                    if random_noise_shift > 0.0:
                        s = (noise.shape[0], noise.shape[1], 1, 1)
                        noise = noise + torch.randn(s, device=noise.device, dtype=noise.dtype) * random_noise_shift

                    # Sample timestep
                    bsz = latents.shape[0]
                    timesteps = self._sample_timesteps(bsz, num_train_timesteps)

                    # Add noise to latents (flow matching formulation)
                    noisy_latents = (1 - timesteps[:, None, None, None]) * latents + timesteps[
                        :, None, None, None
                    ] * noise
                    noisy_latent_multiplier = float(train_config.get("noisy_latent_multiplier", 1.0))
                    if noisy_latent_multiplier != 1.0:
                        noisy_latents = noisy_latents * noisy_latent_multiplier

                    # Predict velocity
                    transformer_kwargs = {}
                    forward_params = inspect.signature(self.transformer.forward).parameters
                    if "cap_mask" in forward_params:
                        transformer_kwargs["cap_mask"] = tokens.attention_mask.bool()

                    # Ensure we pass all required arguments that might be missing
                    # Z-Image expects cap_feats (which corresponds to text_embeds here)
                    # We map text_embeds to cap_feats via positional args if needed,
                    # but here we pass text_embeds as the 3rd arg which matches cap_feats position in signature:
                    # forward(x, t, cap_feats, cap_mask...)

                    # NOTE: ZImage model expects t in range 0-1 and scales it internally by 1000.
                    # It also seems to expect t=0 to be noise and t=1 to be clean (based on musubi code (1000-t)/1000)
                    # Our 'timesteps' is 0 (clean) -> 1 (noise).
                    # So we pass 1.0 - timesteps.
                    output = self.lora_network(
                        noisy_latents.unsqueeze(2),
                        1.0 - timesteps,
                        text_embeds,
                        **transformer_kwargs,
                    )

                    # Handle different return types (BaseOutput, tuple, list)
                    if hasattr(output, "sample"):
                        model_pred = output.sample
                    elif isinstance(output, (list, tuple)):
                        model_pred = output[0]
                    else:
                        model_pred = output

                    # If it's still a list (some models return [tensor]), take the first element
                    if isinstance(model_pred, (list, tuple)):
                        model_pred = model_pred[0]

                    # Squeeze the frame dimension if it exists
                    if len(model_pred.shape) == 5:
                        model_pred = model_pred.squeeze(2)

                    # Fix potential channel/batch swap to avoid broadcasting
                    if (
                        model_pred.ndim == 4
                        and model_pred.shape[0] == latents.shape[1]
                        and model_pred.shape[1] == latents.shape[0]
                        and model_pred.shape[2:] == latents.shape[2:]
                    ):
                        model_pred = model_pred.permute(1, 0, 2, 3).contiguous()

                    # Optional differential guidance (CFG-style)
                    if train_config.get("do_differential_guidance", False):
                        unconditional_prompt = train_config.get("unconditional_prompt", "")
                        uncond_tokens = self.tokenizer(
                            [
                                self.tokenizer.apply_chat_template(
                                    [{"role": "user", "content": unconditional_prompt}],
                                    tokenize=False,
                                    add_generation_prompt=True,
                                )
                                if hasattr(self.tokenizer, "apply_chat_template")
                                else unconditional_prompt
                                for _ in range(len(captions))
                            ],
                            padding="longest",
                            max_length=self.tokenizer.model_max_length,
                            truncation=True,
                            return_tensors="pt",
                        ).to(self.device)
                        with torch.no_grad():
                            uncond_out = base_text_encoder(
                                uncond_tokens.input_ids,
                                attention_mask=uncond_tokens.attention_mask,
                                output_hidden_states=True,
                            )
                            uncond_embeds = uncond_out.hidden_states[-2]
                        uncond_kwargs = {}
                        if "cap_mask" in forward_params:
                            uncond_kwargs["cap_mask"] = uncond_tokens.attention_mask.bool()
                        uncond_output = self.lora_network(
                            noisy_latents.unsqueeze(2),
                            1.0 - timesteps,
                            uncond_embeds,
                            **uncond_kwargs,
                        )
                        if hasattr(uncond_output, "sample"):
                            uncond_pred = uncond_output.sample
                        elif isinstance(uncond_output, (list, tuple)):
                            uncond_pred = uncond_output[0]
                        else:
                            uncond_pred = uncond_output
                        if isinstance(uncond_pred, (list, tuple)):
                            uncond_pred = uncond_pred[0]
                        if len(uncond_pred.shape) == 5:
                            uncond_pred = uncond_pred.squeeze(2)
                        if (
                            uncond_pred.ndim == 4
                            and uncond_pred.shape[0] == latents.shape[1]
                            and uncond_pred.shape[1] == latents.shape[0]
                            and uncond_pred.shape[2:] == latents.shape[2:]
                        ):
                            uncond_pred = uncond_pred.permute(1, 0, 2, 3).contiguous()

                        guidance_scale = float(train_config.get("differential_guidance_scale", 3.0))
                        model_pred = uncond_pred + guidance_scale * (model_pred - uncond_pred)

                    # Compute loss
                    # Z-Image / Musubi uses target = latents - noise
                    # Standard flow matching is noise - latents. We align with Musubi.
                    target = latents - noise

                    loss_map = torch.nn.functional.mse_loss(
                        model_pred.float(),
                        target.float(),
                        reduction="none",
                    )

                    per_sample_loss = loss_map.mean(dim=(1, 2, 3))

                    timestep_weighting = str(train_config.get("timestep_weighting", "none")).lower()
                    if timestep_weighting == "linear":
                        weights = 0.5 + timesteps
                    elif timestep_weighting == "sqrt":
                        weights = 0.5 + torch.sqrt(timesteps)
                    elif timestep_weighting == "cosine":
                        weights = 0.5 + torch.cos((1.0 - timesteps) * torch.pi * 0.5)
                    elif timestep_weighting == "aitk":
                        w = torch.tensor(aitk_weighting, device=timesteps.device, dtype=timesteps.dtype)
                        idx = (timesteps * (w.numel() - 1)).long().clamp(0, w.numel() - 1)
                        weights = w[idx]
                    else:
                        weights = None

                    loss = (per_sample_loss * weights).mean() if weights is not None else per_sample_loss.mean()

                    loss_weight_fn = str(train_config.get("loss_weight_fn", "none")).lower()
                    if loss_weight_fn == "constant":
                        loss_weight_strength = float(train_config.get("loss_weight_strength", 1.0))
                        loss = loss * loss_weight_strength

                    # Backpropagate
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.lora_network.parameters(), train_config.get("max_grad_norm", 1.0)
                        )

                    if self.accelerator.sync_gradients:
                        self.optimizer.step()
                        if self.lr_scheduler is not None:
                            self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    global_step += 1
                    progress_bar.update(1)

                    # Log progress to file, every step
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    self.progress_logger.info(
                        f"Step: {global_step}/{total_steps} [{int(global_step / total_steps * 100)}%], "
                        f"Loss: {loss.item():.6f}, Epoch: {epoch + 1}/{num_epochs}, LR: {current_lr:.2e}"
                    )

                    # Save checkpoint
                    if global_step % save_every == 0:
                        self.save_checkpoint(global_step)

                # Update progress bar
                progress_bar.set_postfix(
                    {"loss": loss.item(), "step": global_step, "epoch": f"{epoch + 1}/{num_epochs}"}
                )

        # Save final checkpoint
        progress_bar.close()
        self.save_checkpoint("final")
        logger.info("Training completed!")

    def save_checkpoint(self, step):
        """Save LoRA checkpoint."""
        if not self.is_main_process:
            return

        output_dir = Path(self.config["output"]["path"])
        output_dir.mkdir(parents=True, exist_ok=True)

        save_path = output_dir / f"zimage_lora_step_{step}.safetensors"

        # Get LoRA state dict
        state_dict = self.lora_network.get_state_dict()

        # Add metadata
        metadata = {
            "step": str(step),
            "rank": str(self.config.get("lora", {}).get("rank", 4)),
            "alpha": str(self.config.get("lora", {}).get("alpha", 1.0)),
        }

        save_file(state_dict, str(save_path), metadata=metadata)
        self.last_saved_path = str(save_path)
        logger.info(f"Saved checkpoint to {save_path}")

    def _log_summary(self, start_time: float):
        elapsed = max(time.time() - start_time, 0.0)
        output_path = self.last_saved_path or self.config.get("output", {}).get("path", "")
        logger.info("Summary: ran for %.2fs, last output saved to %s", elapsed, output_path)

    def run(self):
        """Run the complete training pipeline."""
        try:
            set_seed(self.config.get("seed", 42))
            start_time = time.time()

            if self.dry_run:
                self.dry_run_training()
                self._log_summary(start_time)
                return

            self.load_model()
            self.create_lora_network()
            self.create_dataset()
            self.setup_optimizer()
            self.train()
            self._log_summary(start_time)
        except KeyboardInterrupt:
            logger.info("Training interrupted by user. Exiting...")
            # Attempt to clean up
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"An error occurred: {e}", exc_info=True)
            raise e


def main():
    parser = argparse.ArgumentParser(description="Train Z-Image LoRA")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--dry-run", action="store_true", help="Simulate training without downloading/loading models")
    parser.add_argument(
        "--dry-run-step-time", type=float, default=0.025, help="Seconds per simulated step during dry run"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup logging
    setup_logging(config.get("logging", {}))

    # Save config to output directory
    if config.get("output"):
        save_config_to_output(config, config["output"]["path"])

    # Create trainer and run
    trainer = ZImageLoRATrainer(
        config,
        dry_run=args.dry_run,
        dry_run_step_time=args.dry_run_step_time,
    )
    trainer.run()


if __name__ == "__main__":
    main()
