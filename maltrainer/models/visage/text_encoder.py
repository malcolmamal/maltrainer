"""
Visage-Image's text conditioning: the first 8 layers of Qwen3-0.6B, no final norm (hidden_states[8] of the full model).

Mirrors the model's own ``TextEncoder``: the caption is tokenized with a leading space and runs behind the fixed
prefix "Describe the image." (a causal LM's first tokens are an attention sink with outsized norms; the prefix takes
that instead of the caption's first word), the prefix's states are dropped, and the states are rounded through float8
e4m3 as the training cache stored them. Weights come from ComfyUI's ``qwen3_0.6b.safetensors``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file
from torch import nn
from transformers import AutoTokenizer, Qwen3Config, Qwen3Model

logger = logging.getLogger(__name__)

PREFIX = "Describe the image."
LAYERS = 8
MAX_LENGTH = 128  # caption tokens; the DiT's largest length bucket

# The Qwen3 tokenizer is shared by every size and by Qwen3-VL; reuse the copy vendored for MiniMax H3.
TOKENIZER_ASSETS = Path(__file__).resolve().parents[1] / "minimax_h3" / "assets" / "text_encoder"
CONFIG_ASSETS = Path(__file__).resolve().parent / "assets" / "text_encoder"


class VisageTextEncoder:
    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.prefix_ids: list[int] = tokenizer(PREFIX, add_special_tokens=False).input_ids

    @torch.no_grad()
    def encode(self, caption: str, device: torch.device, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """One caption -> its token states [L, 1024] (L <= 128), float8-rounded, in ``dtype``."""
        ids = self.tokenizer(" " + caption, add_special_tokens=False).input_ids[:MAX_LENGTH]
        input_ids = torch.tensor([self.prefix_ids + ids], device=device)
        out = self.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
        hidden = out.last_hidden_state[0, len(self.prefix_ids) :]
        return cast(torch.Tensor, hidden.float().to(torch.float8_e4m3fn).to(dtype))


def text_key(key: str) -> str | None:
    """A checkpoint key as the 8-layer ``Qwen3Model`` names it; None for what it drops (the head, the deeper layers
    and the final norm)."""
    key = key.removeprefix("model.")
    if key.startswith(("lm_head.", "norm.")):
        return None
    if key.startswith("layers.") and int(key.split(".", 2)[1]) >= LAYERS:
        return None
    return key


def load_visage_text_encoder(
    path: str | Path,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    tokenizer_path: str | Path | None = None,
) -> VisageTextEncoder:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Qwen3-0.6B text encoder not found at {path}")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path or TOKENIZER_ASSETS))
    config: Any = Qwen3Config.from_pretrained(str(CONFIG_ASSETS))
    config.num_hidden_layers = LAYERS
    config.layer_types = config.layer_types[:LAYERS]
    config.use_cache = False
    with init_empty_weights(include_buffers=False):
        model: Any = Qwen3Model(config)
    model.norm = nn.Identity()
    state = {target: value for key, value in load_file(str(path)).items() if (target := text_key(key)) is not None}
    info = model.load_state_dict(state, strict=False, assign=True)
    missing = [key for key in info.missing_keys if "rotary_emb" not in key]
    if missing or info.unexpected_keys:
        raise ValueError(
            f"{path.name} does not look like Qwen3-0.6B (missing {missing[:5]}, unexpected {info.unexpected_keys[:5]})"
        )
    # Non-persistent buffers (RoPE inv_freq) were never materialized under init_empty_weights.
    model.rotary_emb = type(model.rotary_emb)(config=config)
    model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    logger.info("Visage-Image text encoder: Qwen3-0.6B, first %d layers", LAYERS)
    return VisageTextEncoder(model, tokenizer)
