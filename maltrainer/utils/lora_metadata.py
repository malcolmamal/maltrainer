"""Training metadata embedded into LoRA safetensors files.

safetensors metadata is a flat ``Dict[str, str]``, so everything is stringified and
anything structured is JSON-encoded. Two kinds of key are written:

``ss_*``
    Flat scalars, following the kohya/musubi-tuner convention, so existing inspectors
    and model managers that already look for ``ss_`` keys show something useful.

``maltrainer_*``
    This trainer's own record. ``maltrainer_config`` is the complete YAML config as JSON,
    which is what makes a checkpoint self-describing: it can be dumped back out to a
    runnable config file long after the run, when the original YAML has moved on.

Values recorded here are the *resolved* ones where they differ from the config -- the
sigma shift krea2_shift computed, the dtype the adapters actually trained in, the optimizer
arguments that survived the installed library's signature. Those are exactly the details
that get lost, and re-deriving them from a YAML months later is guesswork.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)

CONFIG_KEY = "maltrainer_config"
SCHEMA_VERSION = "1"

# model_type -> (trainer script, ss_base_model_version, ss_network_module)
ARCHITECTURES = {
    "minimax_h3": ("train_minimax_h3_lora.py", "minimax_h3_fl2va", "maltrainer.models.minimax_h3.lora"),
    "qwen_image_2_1": ("train_qwen_image_2_1_lora.py", "qwen_image_2_1", "maltrainer.models.qwen_image_2_1.lora"),
    "krea2": ("train_krea2_lora.py", "krea2_raw", "maltrainer.models.krea2.lora"),
    "flux3_klein": ("train_flux3_klein_lora.py", "flux3_image_klein_base", "maltrainer.models.flux3.lora"),
    "moss_image": ("train_moss_image_lora.py", "moss_image_base", "maltrainer.models.moss_image.lora"),
    "moss_image_finetune": ("train_moss_image_finetune.py", "moss_image_base", "maltrainer.models.moss_image.finetune"),
    "krea2_finetune": ("train_krea2_finetune.py", "krea2_raw", "maltrainer.models.krea2.finetune"),
}


def git_commit(repo_root: Path | None = None) -> str | None:
    """Short commit of the trainer that produced a checkpoint, if it is a git checkout.

    Worth recording: this trainer's numerics have changed materially between runs, so
    "which code made this file" is not answerable from the hyperparameters alone.
    """
    root = repo_root or Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    if not commit:
        return None
    try:
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if dirty.returncode == 0 and dirty.stdout.strip():
            commit += "-dirty"
    except (OSError, subprocess.SubprocessError):
        pass
    return commit


def _flatten(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, (str, int, float)):
        return str(value)
    return json.dumps(value, default=str, sort_keys=True)


def build_metadata(
    config: dict[str, Any],
    *,
    step: Any,
    runtime: dict[str, Any] | None = None,
    embed_config: bool = True,
) -> dict[str, str]:
    """Assemble the safetensors metadata dict for a checkpoint.

    ``runtime`` carries resolved values the config cannot express (total steps, the sigma
    shift chosen for this latent size, trainable parameter count, and so on).
    """
    runtime = dict(runtime or {})
    lora = config.get("lora", {}) or {}
    train = config.get("train", {}) or {}
    dataset = config.get("dataset", {}) or {}
    model = config.get("model", {}) or {}

    arch = runtime.get("model_type") or config.get("model_type") or "minimax_h3"
    if arch not in ARCHITECTURES:
        logger.warning("Metadata: no labels for model type %r; trainer, base model and network module stay empty", arch)
    trainer_script, base_model_version, network_module = ARCHITECTURES.get(arch, ("", "", ""))

    metadata: dict[str, Any] = {
        "maltrainer_metadata_version": SCHEMA_VERSION,
        "maltrainer_architecture": arch,
        "maltrainer_trainer": trainer_script,
        # kohya/musubi-style flat keys
        "ss_base_model_version": base_model_version,
        "ss_network_module": network_module,
        "ss_network_dim": lora.get("rank"),
        "ss_network_alpha": lora.get("alpha"),
        "ss_network_args": {
            "target_modules": lora.get("target_modules"),
            "prefix": lora.get("prefix"),
            "dtype": lora.get("dtype", "float32"),
        },
        "ss_optimizer": train.get("optimizer"),
        "ss_learning_rate": train.get("learning_rate"),
        "ss_weight_decay": train.get("weight_decay"),
        "ss_lr_warmup_steps": train.get("learning_rate_warmup_steps", 0),
        "ss_num_epochs": train.get("num_epochs"),
        "ss_batch_size": train.get("batch_size"),
        "ss_gradient_accumulation_steps": train.get("gradient_accumulation_steps", 1),
        "ss_max_grad_norm": train.get("max_grad_norm"),
        "ss_mixed_precision": config.get("mixed_precision"),
        "ss_seed": config.get("seed"),
        "ss_resolution": dataset.get("resolution"),
        "ss_timestep_sampling": train.get("timestep_distribution"),
        "ss_sigmoid_scale": train.get("sigmoid_scale"),
        "ss_discrete_flow_shift": train.get("timestep_shift"),
        # base model identities by filename only -- the full paths live in the config blob
        "ss_transformer": Path(str(model.get("transformer_path") or "")).name,
        "ss_vae": Path(str(model.get("vae_path") or "")).name,
        "ss_text_encoder": Path(str(model.get("text_encoder_path") or "")).name,
    }

    metadata["step"] = step
    metadata["rank"] = lora.get("rank")
    metadata["alpha"] = lora.get("alpha")
    metadata["architecture"] = arch

    for key, value in runtime.items():
        metadata[f"maltrainer_{key}"] = value

    if embed_config:
        metadata[CONFIG_KEY] = json.dumps(config, default=str, sort_keys=False)

    return {key: _flatten(value) for key, value in metadata.items()}


def read_metadata(path: str | Path) -> dict[str, str]:
    """Read just the metadata header of a safetensors file, without loading tensors."""
    from safetensors import SafetensorError, safe_open

    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            return dict(handle.metadata() or {})
    except SafetensorError as error:  # one type for the callers to catch with the OS and value errors
        raise ValueError(f"not a readable safetensors file: {error}") from error


def extract_config(metadata: dict[str, str]) -> dict[str, Any] | None:
    """Recover the embedded training config, or None if the file predates embedding."""
    blob = metadata.get(CONFIG_KEY)
    if not blob:
        return None
    try:
        return cast(dict[str, Any] | None, json.loads(blob))
    except json.JSONDecodeError as error:
        logger.warning("Embedded config is not valid JSON: %s", error)
        return None
