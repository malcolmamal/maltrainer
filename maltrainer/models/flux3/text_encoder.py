"""
FLUX.3 text conditioning: an 8-layer tap of Qwen3-VL-4B-Instruct hidden states.

Mirrors comfy/text_encoders/flux.py (Flux3Tokenizer / Flux3TEModel) and the diffusers
FLUX.3 pipeline:
- the instruct chat template, no system turn and no <think> block:
  ``<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n``
- right-padded with <|endoftext|> (151643) to a multiple of 80 tokens. The padding is masked
  inside Qwen3-VL but kept in the output: the DiT attends to it, so the window size is part of
  the conditioning.
- padding tokens sit at rope position 1. transformers 4.57, which the reference pins, puts
  them there, and ComfyUI reproduces it.
- hidden_states[k] for k in 4, 8, ..., 32 is the input to decoder layer k, unnormalized. The 8
  taps are concatenated per token: (seq, 8 x 2560 = 20480).

Weights come from ComfyUI's qwen3vl_4b_int8_convrot.safetensors (run through the INT8 ConvRot
kernels) or the plain qwen3vl_4b_bf16. Only the first 32 decoder layers are built.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file
from torch import nn
from transformers import AutoTokenizer, Qwen3VLConfig

from maltrainer.models.flux3.loading import is_int8_convrot
from maltrainer.models.minimax_h3.int8_convrot import (
    enable_int8_convrot,
    load_comfy_int8_convrot_state_dict,
    prepare_int8_convrot_modules,
)
from maltrainer.models.qwen_image_2_1.text_encoder import Int8Embedding

logger = logging.getLogger(__name__)

TAP_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
PAD_TOKEN = 151643
PAD_MULTIPLE = 80
MAX_SEQUENCE_LENGTH = 4096  # the reference's cap (BFL text_encoder.py); ComfyUI does not cap

# The Qwen3-VL tokenizer is shared by every size; reuse the copy vendored for MiniMax H3.
TOKENIZER_ASSETS = Path(__file__).resolve().parents[1] / "minimax_h3" / "assets" / "text_encoder"

# comfy/text_encoders/llama.py Qwen3VL_4BConfig. Heads, rope and eps are shared with the
# vendored Qwen3-VL config and inherited from it.
QWEN3VL_4B_TEXT = {
    "hidden_size": 2560,
    "intermediate_size": 9728,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
}


def _text_key(key: str) -> str | None:
    for prefix in (
        "text_encoders.qwen3vl_4b.transformer.",
        "model.language_model.",
        "language_model.model.",
        "language_model.",
        "model.",
    ):
        if key.startswith(prefix):
            key = key[len(prefix) :]
            break
    if key.startswith(("visual.", "lm_head.")):
        return None
    return key


class Flux3TextEncoder(nn.Module):
    def __init__(self, model: nn.Module, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self._captured: dict[int, torch.Tensor] = {}
        for k in TAP_LAYERS[:-1]:
            model.layers[k].register_forward_pre_hook(self._capture(k), with_kwargs=True)

    def _capture(self, index: int):
        def hook(_module, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            self._captured[index] = hidden.detach()

        return hook

    @torch.no_grad()
    def encode(self, prompt: str, device: torch.device, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """One prompt -> (window, 20480) conditioning, window a multiple of 80."""
        ids = self.tokenizer(TEMPLATE.format(prompt), add_special_tokens=False).input_ids[:MAX_SEQUENCE_LENGTH]
        real = len(ids)
        window = min(math.ceil(real / PAD_MULTIPLE) * PAD_MULTIPLE, MAX_SEQUENCE_LENGTH)
        ids = ids + [PAD_TOKEN] * (window - real)

        input_ids = torch.tensor([ids], device=device)
        attention_mask = torch.zeros_like(input_ids)
        attention_mask[:, :real] = 1
        position_ids = torch.arange(window, device=device)[None].masked_fill(attention_mask == 0, 1)

        self._captured.clear()
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
        # The model stops after layer 31 with an Identity norm, so its output IS hidden_states[32];
        # the earlier taps come from the pre-hooks.
        last = getattr(out, "last_hidden_state", None)
        last = out[0] if last is None else last
        taps = [self._captured[k] for k in TAP_LAYERS[:-1]] + [last]
        return torch.cat([t[0] for t in taps], dim=-1).to(dtype)


def load_flux3_text_encoder(
    path: str | Path,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    tokenizer_path: str | Path | None = None,
    grad_precision: str = "bf16",
) -> Flux3TextEncoder:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Qwen3-VL-4B text encoder not found at {path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(tokenizer_path) if tokenizer_path else TOKENIZER_ASSETS), trust_remote_code=True
    )

    config = Qwen3VLConfig.from_pretrained(str(TOKENIZER_ASSETS)).text_config
    for name, value in QWEN3VL_4B_TEXT.items():
        setattr(config, name, value)
    config.num_hidden_layers = TAP_LAYERS[-1]
    config.use_cache = False

    quantized = is_int8_convrot(path)
    if quantized:
        raw, _ = load_comfy_int8_convrot_state_dict(str(path), device=torch.device("cpu"))
    else:
        raw = load_file(str(path))
    state = {}
    for key, value in raw.items():
        target = _text_key(key)
        if target is None or target.startswith("norm."):
            continue
        if target.startswith("layers.") and int(target.split(".", 2)[1]) >= config.num_hidden_layers:
            continue
        state[target] = value

    with init_empty_weights(include_buffers=False):
        model = Qwen3VLTextModel(config)
    model.norm = nn.Identity()

    if quantized and "embed_tokens.int8_convrot_groupsize" in state:
        # The table is stored Hadamard-rotated; Int8Embedding un-rotates it on lookup.
        embed = Int8Embedding(
            config.vocab_size, config.hidden_size, group_size=int(state["embed_tokens.int8_convrot_groupsize"].item())
        )
        embed.weight.data.copy_(state.pop("embed_tokens.weight"))
        embed.scale_weight.data.copy_(state.pop("embed_tokens.scale_weight"))
        state.pop("embed_tokens.int8_convrot_groupsize")
        model.embed_tokens = embed
    if quantized:
        prepare_int8_convrot_modules(model, state)

    info = model.load_state_dict(state, strict=False, assign=True)
    replaced_embed = isinstance(model.embed_tokens, Int8Embedding)
    missing = [
        key
        for key in info.missing_keys
        if "rotary_emb" not in key and not (replaced_embed and key.startswith("embed_tokens."))
    ]
    if missing or info.unexpected_keys:
        raise ValueError(
            f"{path.name} does not look like a Qwen3-VL-4B text encoder "
            f"(missing {missing[:5]}, unexpected {info.unexpected_keys[:5]})"
        )
    if quantized:
        enable_int8_convrot(model, grad_precision=grad_precision)

    # Non-persistent buffers (RoPE inv_freq) were never materialized under init_empty_weights.
    model.rotary_emb = type(model.rotary_emb)(config=config)
    for parameter in model.parameters():
        if parameter.is_floating_point():
            parameter.data = parameter.data.to(dtype)
    model.to(device=device).eval().requires_grad_(False)
    logger.info(
        "FLUX.3 text encoder: Qwen3-VL-4B, %d layers, %s",
        config.num_hidden_layers,
        "INT8 ConvRot" if quantized else "plain",
    )
    return Flux3TextEncoder(model, tokenizer)
