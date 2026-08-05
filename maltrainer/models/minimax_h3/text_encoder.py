"""
Qwen3-VL text encoder wrapper for MiniMax H3.
Loads BF16 or Comfy NVFP4 AWQ text encoder checkpoints and extracts layer-50 hidden states.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Union

import torch
from torch import nn
from transformers import AutoTokenizer, Qwen3VLConfig, Qwen3VLModel
from accelerate import init_empty_weights

from maltrainer.models.minimax_h3.comfy_quant import has_comfy_quantized_layers, load_comfy_quantized_state_dict

logger = logging.getLogger(__name__)


def _text_encoder_key(source_prefix: str) -> str:
    if source_prefix.startswith("visual"):
        return source_prefix
    if source_prefix == "model":
        return "language_model"
    if source_prefix.startswith("model."):
        return f"language_model.{source_prefix.removeprefix('model.')}"
    return source_prefix


class MiniMaxH3TextEncoder(nn.Module):
    """Wrapper for Qwen3-VL text encoder extracting layer-50 hidden states."""

    def __init__(self, text_encoder_model: Qwen3VLModel, tokenizer: AutoTokenizer):
        super().__init__()
        self.model = text_encoder_model
        self.tokenizer = tokenizer
        self._prompt_cache: dict[str, torch.Tensor] = {}
        self.eval()
        self.requires_grad_(False)

    @torch.no_grad()
    def encode_prompts(
        self,
        prompts: List[str],
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        max_length: int = 512,
    ) -> torch.Tensor:
        """Encode prompts to layer-50 text hidden states [B, S, 5120]."""
        cache_key = f"{dtype}_" + "||".join(prompts)
        if cache_key in self._prompt_cache:
            return self._prompt_cache[cache_key].to(device)

        formatted_prompts = []
        for prompt in prompts:
            if hasattr(self.tokenizer, "apply_chat_template"):
                formatted = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                formatted = prompt
            formatted_prompts.append(formatted)

        tokens = self.tokenizer(
            formatted_prompts,
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        language_model = getattr(self.model, "language_model", self.model)
        outputs = language_model(
            tokens.input_ids,
            attention_mask=tokens.attention_mask,
            output_hidden_states=True,
        )

        if hasattr(outputs, "hidden_states") and len(outputs.hidden_states) >= 50:
            hidden_states = outputs.hidden_states[50]
        elif hasattr(outputs, "hidden_states"):
            hidden_states = outputs.hidden_states[-1]
        else:
            hidden_states = outputs[0]

        res = hidden_states.to(dtype=dtype)
        self._prompt_cache[cache_key] = res
        return res


def load_minimax_h3_text_encoder(
    text_encoder_path: Union[str, Path],
    tokenizer_path: Union[str, Path, None] = None,
    device: Union[str, torch.device] = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> MiniMaxH3TextEncoder:
    """Load Qwen3-VL text encoder (BF16 or NVFP4/AWQ) and tokenizer."""
    text_encoder_path = Path(text_encoder_path)
    device = torch.device(device)

    # Use bundled asset tokenizer or HF
    asset_dir = Path("C:/Development/trainers/akane/musubi-tuner-akane-minimax/src/musubi_tuner/minimax_h3/assets/text_encoder")
    if tokenizer_path is not None:
        tokenizer_source = str(tokenizer_path)
    elif asset_dir.exists():
        tokenizer_source = str(asset_dir)
    else:
        tokenizer_source = "Qwen/Qwen2-VL-7B-Instruct"

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Could not load tokenizer from {tokenizer_source}: {e}.")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build Qwen3-VL config for 50-layer 32B model
    config_source = asset_dir if asset_dir.exists() else "Qwen/Qwen2-VL-7B-Instruct"
    try:
        config = Qwen3VLConfig.from_pretrained(config_source)
    except Exception:
        config = Qwen3VLConfig()

    config.text_config.num_hidden_layers = 50
    config.text_config.hidden_size = 5120
    config.text_config.use_cache = False

    with init_empty_weights(include_buffers=True):
        model = Qwen3VLModel(config)
        model.language_model.norm = nn.Identity()

    if has_comfy_quantized_layers(text_encoder_path):
        logger.info("Loading Comfy NVFP4/AWQ quantized text encoder from %s", text_encoder_path)
        load_comfy_quantized_state_dict(
            model,
            text_encoder_path,
            key_map=_text_encoder_key,
            output_dtype=dtype,
        )
    else:
        logger.info("Loading text encoder safetensors from %s", text_encoder_path)
        from safetensors.torch import load_file
        sd = load_file(str(text_encoder_path))
        mapped_sd = {}
        for k, v in sd.items():
            if k.startswith(("visual.", "model.")):
                prefix, _, suffix = k.rpartition(".")
                target_key = f"{_text_encoder_key(prefix)}.{suffix}"
                mapped_sd[target_key] = v
        model.load_state_dict(mapped_sd, strict=False)

    vision_config = config.vision_config
    vision_head_dim = vision_config.hidden_size // vision_config.num_heads
    model.visual.rotary_pos_emb = type(model.visual.rotary_pos_emb)(vision_head_dim // 2).to(device)
    model.language_model.rotary_emb = type(model.language_model.rotary_emb)(
        config.text_config,
        device=device,
    )

    model.to(device)
    model.eval()
    model.requires_grad_(False)

    return MiniMaxH3TextEncoder(model, tokenizer)
