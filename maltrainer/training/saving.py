"""Where a LoRA checkpoint goes, and writing it with the run's metadata."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from maltrainer.utils.lora_metadata import build_metadata


def lora_save_path(
    output_config: dict[str, Any], step: int | str, default_name: str, name_keys: Sequence[str] = ("name", "lora_name")
) -> Path:
    """<output.path>/<name>.safetensors for the final save, <name>_step_<N>.safetensors before it; the name is
    the first of ``name_keys`` that is set. Creates the folder."""
    output_dir = Path(output_config["path"])
    output_dir.mkdir(parents=True, exist_ok=True)
    name = next((output_config[key] for key in name_keys if output_config.get(key)), default_name)
    name = name[: -len(".safetensors")] if name.endswith(".safetensors") else name
    return output_dir / (f"{name}.safetensors" if step == "final" else f"{name}_step_{step}.safetensors")


def write_lora(
    config: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
    save_path: Path,
    step: int | str,
    run_metadata: dict[str, object],
    save_dtype: torch.dtype,
) -> None:
    runtime = dict(
        run_metadata, saved_at=time.strftime("%Y-%m-%dT%H:%M:%S"), save_dtype=str(save_dtype).replace("torch.", "")
    )
    embed_config = bool(config.get("output", {}).get("embed_config", True))
    metadata = build_metadata(config, step=step, runtime=runtime, embed_config=embed_config)
    save_file(state_dict, str(save_path), metadata=metadata)
