"""
Krea 2 text conditioning: a 12-layer tap of Qwen3-VL-4B hidden states.

Mirrors comfy/text_encoders/krea2.py exactly:
- the prompt goes through Qwen-Image's system template (no <think> block)
- hidden_states[k] for k in 2, 5, ..., 35 -- the input to decoder layer k, unnormalized
- the system + "<|im_start|>user\\n" prefix is dropped from the OUTPUT only; the prompt
  tokens still attended to it, and the trailing "<|im_end|>\\n<|im_start|>assistant\\n"
  stays in
Result: (seq, 12, 2560) per prompt, which the DiT's txtfusion adapter consumes.

Weights come from ComfyUI's qwen3vl_4b(_fp8_scaled).safetensors; fp8 layers are dequantized
to bf16 on load (the encoder only runs once per caption, at caching time).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from accelerate import init_empty_weights
from torch import nn
from transformers import AutoTokenizer, Qwen3VLConfig

from maltrainer.models.comfy_fp8 import dequantize_checkpoint, read_comfy_checkpoint

logger = logging.getLogger(__name__)

TAP_LAYERS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
IM_START, USER, NEWLINE = 151644, 872, 198

# The Qwen3-VL tokenizer is shared by every size; reuse the copy vendored for MiniMax H3.
TOKENIZER_ASSETS = Path(__file__).resolve().parents[1] / "minimax_h3" / "assets" / "text_encoder"

# comfy/text_encoders/llama.py Qwen3VL_4BConfig (the rest -- heads, rope, eps -- is shared
# with the vendored Qwen3-VL config and inherited from it).
QWEN3VL_4B_TEXT = {
    "hidden_size": 2560,
    "intermediate_size": 9728,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
}


def _prefix_end(token_ids: list[int]) -> int:
    """Index of the first prompt token: ComfyUI's Krea2TEModel prefix strip."""
    template_end, seen = -1, 0
    for i, token in enumerate(token_ids):
        if token == IM_START and seen < 2:
            template_end, seen = i, seen + 1
    if (
        len(token_ids) > template_end + 3
        and token_ids[template_end + 1] == USER
        and token_ids[template_end + 2] == NEWLINE
    ):
        template_end += 3
    return max(template_end, 0)


def _text_key(key: str) -> str | None:
    key = key.removeprefix("text_encoders.qwen3vl_4b.transformer.")  # a combined ComfyUI file, then its own spelling
    for prefix in (
        "model.language_model.",
        "language_model.model.",
        "language_model.",
        "model.",
    ):
        if key.startswith(prefix):
            key = key[len(prefix) :]
            break
    if key.startswith(("visual.", "lm_head.", "model.visual")):
        return None
    return key


class Krea2TextEncoder(nn.Module):
    def __init__(self, model: Any, tokenizer: Any) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self._captured: dict[int, torch.Tensor] = {}
        for k in TAP_LAYERS[:-1]:
            model.layers[k].register_forward_pre_hook(self._capture(k), with_kwargs=True)

    def _capture(self, index: int) -> Any:
        def hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            hidden = args[0] if args else kwargs["hidden_states"]
            self._captured[index] = hidden.detach()

        return hook

    @torch.no_grad()
    def encode(
        self, prompt: str, device: torch.device, dtype: torch.dtype = torch.bfloat16, max_prompt_tokens: int = 512
    ) -> torch.Tensor:
        """One prompt -> (seq, 12, 2560) conditioning on CPU-ready ``dtype``.

        Prompts longer than ``max_prompt_tokens`` are cut (ai-toolkit and OneTrainer cap at 512;
        ComfyUI does not cap, but captions that long are rare).
        """
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        if len(prompt_ids) > max_prompt_tokens:
            prompt = self.tokenizer.decode(prompt_ids[:max_prompt_tokens])
        ids = self.tokenizer(TEMPLATE.format(prompt), add_special_tokens=False).input_ids
        start = _prefix_end(ids)
        input_ids = torch.tensor([ids], device=device)
        self._captured.clear()
        out = self.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
        # The model is truncated to 35 layers with an Identity norm, so its output IS
        # hidden_states[35]; the earlier taps come from the pre-hooks.
        last = getattr(out, "last_hidden_state", None)
        last = out[0] if last is None else last
        taps = [self._captured[k] for k in TAP_LAYERS[:-1]] + [last]
        stacked = torch.stack([t[0] for t in taps], dim=1)  # (seq, 12, 2560)
        return stacked[start:].to(dtype)


def load_krea2_text_encoder(
    path: str | Path,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    tokenizer_path: str | Path | None = None,
) -> Krea2TextEncoder:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    assets = Path(tokenizer_path) if tokenizer_path else TOKENIZER_ASSETS
    tokenizer = AutoTokenizer.from_pretrained(str(assets), trust_remote_code=True)

    config: Any = Qwen3VLConfig.from_pretrained(str(TOKENIZER_ASSETS)).text_config
    for name, setting in QWEN3VL_4B_TEXT.items():
        setattr(config, name, setting)
    config.num_hidden_layers = TAP_LAYERS[-1]  # 35: the deepest tap is the input to layer 35
    config.use_cache = False

    with init_empty_weights(include_buffers=False):
        model: Any = Qwen3VLTextModel(config)
    model.norm = nn.Identity()

    tensors, layers = read_comfy_checkpoint(path)
    state = dequantize_checkpoint(tensors, layers, dtype)
    mapped = {}
    for key, value in state.items():
        target = _text_key(key)
        if target is None or target.startswith("norm."):
            continue
        if target.startswith("layers.") and int(target.split(".", 2)[1]) >= config.num_hidden_layers:
            continue
        mapped[target] = value
    info = model.load_state_dict(mapped, strict=False, assign=True)
    missing = [key for key in info.missing_keys if "rotary_emb" not in key]
    if missing or info.unexpected_keys:
        raise ValueError(
            f"{Path(path).name} does not look like a Qwen3-VL-4B text encoder "
            f"(missing {missing[:5]}, unexpected {info.unexpected_keys[:5]})"
        )
    # Non-persistent buffers (RoPE inv_freq) were never materialized under init_empty_weights.
    model.rotary_emb = type(model.rotary_emb)(config=config)
    model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    logger.info(
        "Krea 2 text encoder: Qwen3-VL-4B, %d layers, %d fp8 tensors dequantized", config.num_hidden_layers, len(layers)
    )
    return Krea2TextEncoder(model, tokenizer)
