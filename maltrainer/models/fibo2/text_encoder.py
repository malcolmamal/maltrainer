"""
Fibo 2.0 text conditioning: Qwen3-VL-4B hidden states in six bundles of three layers.

Mirrors comfy/text_encoders/fibo2.py and diffusers' ``BriaFibo2Pipeline``:
- the instruct chat template, no system turn: ``<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n``
- no padding: the DiT's gist-token count follows the true token length
- 15 hidden layers (``hidden_states[k]`` is the input of decoder layer k, unnormalised), each scaled to unit RMS in
  fp32 and rounded to bf16. Bundle b concatenates its three layers; the six bundles are concatenated per token:
  (tokens, 6 x 3 x 2560 = 46080). The first bundle feeds the Perceiver, the others the five injection blocks.

Weights come from ComfyUI's qwen3vl_4b_int8_convrot.safetensors or the plain bf16 file. Only the first 35 decoder
layers are built; layer 35's input is the last tap.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from accelerate import init_empty_weights
from torch import nn
from transformers import AutoTokenizer

from maltrainer.models.fibo2.prompt import prompt_text
from maltrainer.models.flux3.loading import is_int8_convrot
from maltrainer.models.flux3.text_encoder import TEMPLATE, TOKENIZER_ASSETS, text_config, text_state_dict
from maltrainer.models.flux3.text_state import install_int8_embedding, load_text_state
from maltrainer.models.minimax_h3.int8_convrot import enable_int8_convrot, prepare_int8_convrot_modules

logger = logging.getLogger(__name__)

BUNDLE_LAYERS = ((9, 20, 31), (5, 16, 27), (7, 18, 29), (9, 20, 31), (11, 22, 33), (13, 24, 35))
TAP_LAYERS = tuple(sorted({layer for bundle in BUNDLE_LAYERS for layer in bundle}))
MAX_SEQUENCE_LENGTH = 4096
NORM_EPS = 1e-6


def unit_rms(states: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Each token's vector scaled to unit RMS in fp32, then rounded."""
    states = states.float()
    return (states * torch.rsqrt(states.pow(2).mean(-1, keepdim=True) + NORM_EPS)).to(dtype)


class Fibo2TextEncoder(nn.Module):
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
    def encode(self, prompt: str, device: torch.device, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """One prompt -> (tokens, 46080) conditioning."""
        ids = self.tokenizer(TEMPLATE.format(prompt_text(prompt)), add_special_tokens=False).input_ids
        input_ids = torch.tensor([ids[:MAX_SEQUENCE_LENGTH]], device=device)
        position_ids = torch.arange(input_ids.shape[1], device=device)[None]
        self._captured.clear()
        out = self.model(input_ids=input_ids, position_ids=position_ids, use_cache=False)
        # The model stops after layer 34 with an Identity norm, so its output IS hidden_states[35]; the earlier
        # taps come from the pre-hooks.
        last = getattr(out, "last_hidden_state", None)
        last = out[0] if last is None else last
        taps = {**self._captured, TAP_LAYERS[-1]: last}
        return torch.cat([unit_rms(taps[layer][0], dtype) for bundle in BUNDLE_LAYERS for layer in bundle], dim=-1)


def load_fibo2_text_encoder(
    path: str | Path,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    tokenizer_path: str | Path | None = None,
    grad_precision: str = "bf16",
) -> Fibo2TextEncoder:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Qwen3-VL-4B text encoder not found at {path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(tokenizer_path) if tokenizer_path else TOKENIZER_ASSETS), trust_remote_code=True
    )
    config = text_config(TOKENIZER_ASSETS, TAP_LAYERS[-1])

    quantized = is_int8_convrot(path)
    state = text_state_dict(path, quantized, config.num_hidden_layers)

    with init_empty_weights(include_buffers=False):
        model: Any = Qwen3VLTextModel(config)
    model.norm = nn.Identity()
    if quantized and "embed_tokens.int8_convrot_groupsize" in state:
        install_int8_embedding(model, state, config)
    if quantized:
        prepare_int8_convrot_modules(model, state)
    load_text_state(model, state, path.name)
    if quantized:
        enable_int8_convrot(model, grad_precision=grad_precision)

    # Non-persistent buffers (RoPE inv_freq) were never materialized under init_empty_weights.
    model.rotary_emb = type(model.rotary_emb)(config=config)
    for parameter in model.parameters():
        if parameter.is_floating_point():
            parameter.data = parameter.data.to(dtype)
    model.to(device=device).eval().requires_grad_(False)
    logger.info(
        "Fibo 2.0 text encoder: Qwen3-VL-4B, %d layers, %s",
        config.num_hidden_layers,
        "INT8 ConvRot" if quantized else "plain",
    )
    return Fibo2TextEncoder(model, tokenizer)
