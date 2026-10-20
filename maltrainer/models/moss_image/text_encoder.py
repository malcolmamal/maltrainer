"""
MOSS-Image text conditioning: Qwen3.5-9B's last hidden state.

Mirrors comfy/text_encoders/moss_image.py (MOSSImageTokenizer / MOSSImageTEModel) and
comfy/text_encoders/qwen35.py:
- the prompt is stripped and wrapped in the chat template with thinking enabled, which leaves an
  open think block: ``<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n``
- no padding and no truncation of the window: one prompt is encoded at its true length, and the
  DiT's RoPE follows that length. Capped at 4096 tokens, the final pretraining stage's limit.
- the output is every token's final hidden state after the model's last RMSNorm, (L, 4096)

Qwen3.5 is a hybrid: of its 32 decoder layers, every 4th is gated full attention (QK-norm,
partial RoPE on 64 of 256 head dims, a sigmoid output gate); the rest are Gated DeltaNet linear
attention (a causal depthwise conv, then the chunked gated delta rule). The port below is the
text path of ComfyUI's, without the KV cache, the vision tower, the MTP head and lm_head.

Weights come from ComfyUI's qwen3.5_9b_int8_convrot.safetensors (its Linears run through the
INT8 ConvRot kernels) or a plain bf16 file. They stay in their stored dtype, but activations run
in fp32 and each weight is upcast per op: ComfyUI's SDClipModel embeds tokens with out_dtype
float32 and runs the encoder with dtype=float32, which keeps its bf16 drift ~3x below an
all-bf16 run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, cast

import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from safetensors.torch import load_file
from torch import Tensor, nn

from maltrainer.models.flux3.loading import is_int8_convrot
from maltrainer.models.minimax_h3.int8_convrot import (
    enable_int8_convrot,
    load_comfy_int8_convrot_state_dict,
    prepare_int8_convrot_modules,
)
from maltrainer.models.qwen_image_2_1.text_encoder import Int8Embedding

logger = logging.getLogger(__name__)

TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n"
MAX_SEQUENCE_LENGTH = 4096  # OpenMOSS's ComfyUI node and the report's final-stage cap
TOKENIZER_ASSETS = Path(__file__).resolve().parent / "assets" / "tokenizer"

KEY_PREFIXES = (
    "text_encoders.qwen35_9b.transformer.model.",
    "text_encoders.qwen35_9b.transformer.",
    "model.language_model.",
    "language_model.model.",
    "language_model.",
    "model.",
)
SKIPPED = ("visual.", "lm_head.", "mtp.")


@dataclass(frozen=True)
class Qwen35TextConfig:
    """Qwen3.5-9B's text model (comfy/text_encoders/qwen35.py QWEN35_MODELS["qwen35_9b"])."""

    vocab_size: int = 248320
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    partial_rotary_factor: float = 0.25
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    conv_kernel_size: int = 4
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000000.0
    full_attention_layers: tuple[int, ...] = tuple(i for i in range(32) if (i + 1) % 4 == 0)

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor]) -> Qwen35TextConfig:
        """Read the sizes off the weights, so a mis-picked file fails here and not deep in a matmul."""
        if "layers.0.linear_attn.A_log" not in state:
            raise ValueError("Not a Qwen3.5 text encoder (no layers.0.linear_attn.A_log)")
        layers = sorted({int(k.split(".", 2)[1]) for k in state if k.startswith("layers.")})
        full = tuple(i for i in layers if f"layers.{i}.self_attn.k_proj.weight" in state)
        first_full = full[0] if full else 3
        head_dim = state[f"layers.{first_full}.self_attn.q_norm.weight"].shape[0]
        value_heads = state["layers.0.linear_attn.A_log"].shape[0]
        value_dim = state["layers.0.linear_attn.in_proj_z.weight"].shape[0]
        conv_dim = state["layers.0.linear_attn.conv1d.weight"].shape[0]
        key_head_dim = cls.linear_key_head_dim
        return cls(
            vocab_size=state["embed_tokens.weight"].shape[0],
            hidden_size=state["embed_tokens.weight"].shape[1],
            intermediate_size=state["layers.0.mlp.gate_proj.weight"].shape[0],
            num_hidden_layers=len(layers),
            num_attention_heads=state[f"layers.{first_full}.self_attn.q_proj.weight"].shape[0] // (2 * head_dim),
            num_key_value_heads=state[f"layers.{first_full}.self_attn.k_proj.weight"].shape[0] // head_dim,
            head_dim=head_dim,
            linear_num_key_heads=(conv_dim - value_dim) // (2 * key_head_dim),
            linear_num_value_heads=value_heads,
            linear_value_head_dim=value_dim // value_heads,
            conv_kernel_size=state["layers.0.linear_attn.conv1d.weight"].shape[-1],
            full_attention_layers=full,
        )


# ---------------------------------------------------------------- building blocks


class RMSNorm(nn.Module):
    """``add=True`` scales by (1 + weight), Qwen3.5's convention for every norm but the gated one."""

    def __init__(self, dim: int, eps: float, add: bool):
        super().__init__()
        self.eps = eps
        self.add = add
        self.weight = nn.Parameter(torch.zeros(dim) if add else torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight + 1.0 if self.add else self.weight
        return F.rms_norm(x, weight.shape, weight.to(x.dtype), self.eps)


class RMSNormGated(RMSNorm):
    def __init__(self, dim: int, eps: float):
        super().__init__(dim, eps, add=False)

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:  # type: ignore[override]  # gated variant: second arg by design
        return super().forward(x) * F.silu(gate.to(x.dtype))


class MLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


def chunk_gated_delta_rule(
    query: Tensor, key: Tensor, value: Tensor, g: Tensor, beta: Tensor, chunk_size: int = 64
) -> Tensor:
    """ComfyUI's torch_chunk_gated_delta_rule from a zero state. [B, L, H, D] in, [B, L, H, Dv] out."""
    initial_dtype = query.dtype
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query, key, value = [F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value)]
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    query = query * (1 / (query.shape[-1] ** 0.5))

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    state = torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for i in range(total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_new = v_i - k_cumdecay[:, :, i] @ state
        core_attn_out[:, :, i] = (q_i * g[:, :, i, :, None].exp()) @ state + attn @ v_new
        state = (
            state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    return core_attn_out[:, :, :sequence_length].transpose(1, 2).contiguous().to(initial_dtype)


class GatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen35TextConfig):
        super().__init__()
        hidden = config.hidden_size
        self.num_key_heads = config.linear_num_key_heads
        self.num_value_heads = config.linear_num_value_heads
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.key_dim = self.num_key_heads * self.key_head_dim
        self.value_dim = self.num_value_heads * self.value_head_dim
        conv_dim = self.key_dim * 2 + self.value_dim

        self.in_proj_qkv = nn.Linear(hidden, conv_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(hidden, self.num_value_heads, bias=False)
        self.in_proj_a = nn.Linear(hidden, self.num_value_heads, bias=False)
        self.out_proj = nn.Linear(self.value_dim, hidden, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(self.num_value_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_value_heads))
        self.conv1d = nn.Conv1d(
            conv_dim,
            conv_dim,
            kernel_size=config.conv_kernel_size,
            groups=conv_dim,
            padding=config.conv_kernel_size - 1,
            bias=False,
        )
        self.norm = RMSNormGated(self.value_head_dim, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        z = self.in_proj_z(x)
        mixed = self.in_proj_qkv(x).transpose(1, 2)
        mixed = F.silu(
            F.conv1d(
                mixed, self.conv1d.weight.to(mixed.dtype), None, padding=self.conv1d.padding, groups=self.conv1d.groups
            )[:, :, :L]
        ).transpose(1, 2)
        query, key, value = mixed.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)

        beta = self.in_proj_b(x).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(self.in_proj_a(x).float() + self.dt_bias.float())
        query = query.reshape(B, L, self.num_key_heads, self.key_head_dim)
        key = key.reshape(B, L, self.num_key_heads, self.key_head_dim)
        value = value.reshape(B, L, self.num_value_heads, self.value_head_dim)
        if self.num_value_heads != self.num_key_heads:
            rep = self.num_value_heads // self.num_key_heads
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)

        core = chunk_gated_delta_rule(query, key, value, g=g, beta=beta)
        core = self.norm(core.reshape(-1, self.value_head_dim), z.reshape(-1, self.value_head_dim))
        return cast(torch.Tensor, self.out_proj(core.reshape(B, L, -1)))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class GatedAttention(nn.Module):
    def __init__(self, config: Qwen35TextConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rotary_dim = int(config.head_dim * config.partial_rotary_factor)
        inner = self.num_heads * self.head_dim
        self.q_proj = nn.Linear(config.hidden_size, inner * 2, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(inner, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps, add=True)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps, add=True)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        qg = self.q_proj(x).view(B, L, self.num_heads, self.head_dim * 2)
        q, gate = qg[..., : self.head_dim], qg[..., self.head_dim :].reshape(B, L, -1)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Partial split-half RoPE on the first rotary_dim dims, in fp32, rounded once.
        r = self.rotary_dim
        q_rot, k_rot = q[..., :r].float(), k[..., :r].float()
        q = torch.cat([(q_rot * cos + rotate_half(q_rot) * sin).to(v.dtype), q[..., r:]], dim=-1)
        k = torch.cat([(k_rot * cos + rotate_half(k_rot) * sin).to(v.dtype), k[..., r:]], dim=-1)

        rep = self.num_heads // self.num_kv_heads
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, L, -1)
        return cast(torch.Tensor, self.o_proj(out * gate.sigmoid()))


class DecoderLayer(nn.Module):
    def __init__(self, config: Qwen35TextConfig, index: int):
        super().__init__()
        if index in config.full_attention_layers:
            self.self_attn = GatedAttention(config)
        else:
            self.linear_attn = GatedDeltaNet(config)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, add=True)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, add=True)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        h = self.input_layernorm(x)
        h = self.self_attn(h, cos, sin) if hasattr(self, "self_attn") else self.linear_attn(h)
        x = x + h
        return cast(torch.Tensor, x + self.mlp(self.post_attention_layernorm(x)))


class Qwen35TextModel(nn.Module):
    def __init__(self, config: Qwen35TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens: nn.Module = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, add=True)
        self.compute_dtype = torch.float32

    def rope(self, length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Text-only positions: Qwen3.5's interleaved MRoPE collapses to 1D RoPE (all three axes equal)."""
        dim = int(self.config.head_dim * self.config.partial_rotary_factor)
        inv_freq = 1.0 / (self.config.rope_theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
        freqs = torch.arange(length, device=device).float()[:, None] * inv_freq[None]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos()[None, None], emb.sin()[None, None]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if isinstance(self.embed_tokens, nn.Embedding):
            x = F.embedding(input_ids, self.embed_tokens.weight).to(self.compute_dtype)
        else:
            cast(Any, self.embed_tokens).out_dtype = self.compute_dtype
            x = self.embed_tokens(input_ids)
        cos, sin = self.rope(input_ids.shape[1], input_ids.device)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return cast(torch.Tensor, self.norm(x))


# ---------------------------------------------------------------- encoder


class MOSSImageTextEncoder(nn.Module):
    def __init__(self, model: Qwen35TextModel, tokenizer: Any):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def token_ids(self, prompt: str) -> list[Any]:
        ids = self.tokenizer(TEMPLATE.format(prompt.strip()), add_special_tokens=False).input_ids
        if len(ids) > MAX_SEQUENCE_LENGTH:
            logger.warning("Prompt is %d tokens; truncated to %d.", len(ids), MAX_SEQUENCE_LENGTH)
        return cast(list[Any], ids[:MAX_SEQUENCE_LENGTH])

    @torch.no_grad()
    def encode(self, prompt: str, device: torch.device, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """One prompt -> (L, 4096) conditioning at its true length."""
        input_ids = torch.tensor([self.token_ids(prompt)], device=device)
        return cast(torch.Tensor, self.model(input_ids)[0].to(dtype))


def _upcast_linear_forward(module: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """A plain Linear in the input's dtype, as comfy.ops casts a bf16 weight up to fp32 activations."""
    bias = module.bias.to(x.dtype) if module.bias is not None else None
    return F.linear(x, module.weight.to(x.dtype), bias)


def _text_key(key: str) -> str | None:
    for prefix in KEY_PREFIXES:
        if key.startswith(prefix):
            key = key[len(prefix) :]
            break
    if key.startswith(SKIPPED):
        return None
    return key


def load_moss_image_text_encoder(
    path: str | Path,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    tokenizer_path: str | Path | None = None,
    grad_precision: str = "bf16",
    compute_dtype: torch.dtype = torch.float32,
) -> MOSSImageTextEncoder:
    """``dtype`` is the storage dtype of the float weights; ``compute_dtype`` that of the activations."""
    from transformers import Qwen2Tokenizer

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Qwen3.5-9B text encoder not found at {path}")
    tokenizer = Qwen2Tokenizer.from_pretrained(str(Path(tokenizer_path) if tokenizer_path else TOKENIZER_ASSETS))

    quantized = is_int8_convrot(path)
    if quantized:
        raw, _ = load_comfy_int8_convrot_state_dict(str(path), device=torch.device("cpu"))
    else:
        raw = load_file(str(path))
    state = {target: value for key, value in raw.items() if (target := _text_key(key)) is not None}
    del raw
    config = Qwen35TextConfig.from_state_dict(state)

    with init_empty_weights(include_buffers=False):
        model = Qwen35TextModel(config)

    if quantized and "embed_tokens.int8_convrot_groupsize" in state:
        # The table is stored Hadamard-rotated; Int8Embedding un-rotates it on lookup.
        embed = Int8Embedding(
            config.vocab_size, config.hidden_size, group_size=int(state["embed_tokens.int8_convrot_groupsize"].item())
        )
        embed.weight.data.copy_(state.pop("embed_tokens.weight"))
        cast(Any, embed).scale_weight.data.copy_(state.pop("embed_tokens.scale_weight"))
        state.pop("embed_tokens.int8_convrot_groupsize")
        model.embed_tokens = embed
    if quantized:
        prepare_int8_convrot_modules(model, state)

    info = model.load_state_dict(state, strict=False, assign=True)
    replaced_embed = isinstance(model.embed_tokens, Int8Embedding)
    missing = [key for key in info.missing_keys if not (replaced_embed and key.startswith("embed_tokens."))]
    if missing or info.unexpected_keys:
        raise ValueError(
            f"{path.name} does not look like a Qwen3.5 text encoder "
            f"(missing {missing[:5]}, unexpected {info.unexpected_keys[:5]})"
        )
    if quantized:
        enable_int8_convrot(model, grad_precision=grad_precision)

    for module in model.modules():
        if isinstance(module, nn.Linear) and not hasattr(module, "scale_weight"):
            cast(Any, module).forward = MethodType(_upcast_linear_forward, module)
    for parameter in model.parameters():
        if parameter.is_floating_point():
            parameter.data = parameter.data.to(dtype)
    model.compute_dtype = compute_dtype
    model.to(device=device).eval().requires_grad_(False)
    logger.info(
        "MOSS-Image text encoder: Qwen3.5, %d layers (%d full attention), hidden %d, %s",
        config.num_hidden_layers,
        len(config.full_attention_layers),
        config.hidden_size,
        "INT8 ConvRot" if quantized else "plain",
    )
    return MOSSImageTextEncoder(model, tokenizer)
