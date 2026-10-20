"""
Qwen3-VL Text Encoder for Qwen-Image-2.1.
Supports INT8 ConvRot quantized Qwen3-VL 8B checkpoint.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import torch
from accelerate import init_empty_weights
from torch import nn
from transformers import AutoTokenizer, PreTrainedTokenizerBase, Qwen3VLConfig, Qwen3VLModel

from maltrainer.models.minimax_h3.int8_convrot import (
    enable_int8_convrot,
    load_comfy_int8_convrot_state_dict,
    prepare_int8_convrot_modules,
)
from maltrainer.models.qwen_image_2_1.int8_embedding import Int8Embedding

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."
DEFAULT_DROP_IDX = 14

__all__ = [
    "DEFAULT_DROP_IDX",
    "DEFAULT_SYSTEM_PROMPT",
    "Int8Embedding",
    "Qwen3VLTextEncoder",
    "load_qwen_image_2_1_text_encoder",
]


class Qwen3VLTextEncoder(nn.Module):
    """Wrapper around Qwen3-VL 8B text encoder."""

    def __init__(
        self,
        model: Qwen3VLModel,
        tokenizer: PreTrainedTokenizerBase,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        drop_idx: int = DEFAULT_DROP_IDX,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.drop_idx = drop_idx
        self.template = (
            f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        )

    def format_prompt(self, prompt: str) -> str:
        return self.template.format(prompt)

    @torch.no_grad()
    def encode_prompts(
        self,
        prompts: str | list[str],
        max_length: int = 512,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Encode a list of text prompts into Qwen-Image-2.1 text embeddings [B, S, 4096]."""
        if isinstance(prompts, str):
            prompts = [prompts]

        formatted_prompts = [self.format_prompt(p) for p in prompts]
        encoded = self.tokenizer(
            formatted_prompts,
            padding="longest",
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt",
        )

        input_ids = encoded.input_ids
        attention_mask = encoded.attention_mask

        if device is not None:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
        else:
            dev = next(self.model.parameters()).device
            input_ids = input_ids.to(dev)
            attention_mask = attention_mask.to(dev)

        outputs = self.model.language_model(input_ids=input_ids, attention_mask=attention_mask)

        hidden_states = outputs.last_hidden_state
        # Drop system prompt tokens
        if self.drop_idx > 0 and hidden_states.shape[1] > self.drop_idx:
            hidden_states = hidden_states[:, self.drop_idx :]

        return cast(torch.Tensor, hidden_states)


def _map_te_key(k: str) -> str:
    if k.startswith("model.visual."):
        return k[len("model.") :]
    if k.startswith("model."):
        return "language_model." + k[len("model.") :]
    return k


def load_qwen_image_2_1_text_encoder(
    text_encoder_path: str | Path,
    asset_dir: str | Path | None = None,
    device: str | torch.device = "cuda",
    grad_precision: str = "bf16",
) -> Qwen3VLTextEncoder:
    """Load Qwen3-VL 8B text encoder from INT8 ConvRot safetensors (no dtype: the file fixes it)."""
    text_encoder_path = Path(text_encoder_path)
    if not text_encoder_path.exists():
        raise FileNotFoundError(f"Text encoder checkpoint not found at {text_encoder_path}")

    if asset_dir is None:
        # Default to bundled text encoder assets
        asset_dir = Path(__file__).resolve().parent.parent / "minimax_h3" / "assets" / "text_encoder"

    asset_dir = Path(asset_dir)
    logger.info("Loading tokenizer from %s...", asset_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(asset_dir), trust_remote_code=True)

    config = cast(Any, Qwen3VLConfig.from_pretrained(str(asset_dir)))  # typed as a dict|config|None union
    config.text_config.num_hidden_layers = 36
    config.text_config.hidden_size = 4096
    config.text_config.intermediate_size = 12288
    config.text_config.num_attention_heads = 32
    config.text_config.num_key_value_heads = 8
    config.text_config.use_cache = False
    config.vision_config.out_hidden_size = 4096

    with init_empty_weights(include_buffers=True):
        model: Any = cast(Any, Qwen3VLModel)(config)
        model.language_model.norm = nn.Identity()

    logger.info("Loading INT8 ConvRot state dict for text encoder from %s...", text_encoder_path)
    int8_state_dict, _num_quant = load_comfy_int8_convrot_state_dict(str(text_encoder_path), device=torch.device("cpu"))

    mapped_sd = {}
    for k, v in int8_state_dict.items():
        if k.startswith("lm_head"):
            continue
        mapped_sd[_map_te_key(k)] = v

    embed_group_size = int(mapped_sd["language_model.embed_tokens.int8_convrot_groupsize"].item())
    embed_tokens = Int8Embedding(
        config.text_config.vocab_size,
        config.text_config.hidden_size,
        group_size=embed_group_size,
    )
    embed_tokens.weight.data.copy_(mapped_sd["language_model.embed_tokens.weight"])
    embed_tokens.get_buffer("scale_weight").data.copy_(mapped_sd["language_model.embed_tokens.scale_weight"])
    model.language_model.embed_tokens = embed_tokens
    logger.info("Embedding table: INT8 ConvRot, un-rotating with group size %d", embed_group_size)

    clean_sd = {k: v for k, v in mapped_sd.items() if not k.startswith("language_model.embed_tokens.")}
    prepare_int8_convrot_modules(model, clean_sd)
    model.load_state_dict(clean_sd, strict=False, assign=True)

    enable_int8_convrot(model, grad_precision=grad_precision)

    vision_config = config.vision_config
    vision_head_dim = vision_config.hidden_size // vision_config.num_heads
    model.visual.rotary_pos_emb = type(cast(Any, model.visual.rotary_pos_emb))(vision_head_dim // 2).to(device)
    model.language_model.rotary_emb = type(model.language_model.rotary_emb)(config.text_config, device=device)

    model.requires_grad_(False)
    model.to(device=device)
    model.eval()

    logger.info("Loaded Qwen3-VL 8B Text Encoder successfully on %s", device)
    return Qwen3VLTextEncoder(model=model, tokenizer=tokenizer)
