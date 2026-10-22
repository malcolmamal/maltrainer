from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from maltrainer.models.minimax_h3.model import (
    MiniMaxH3FeedForward,
    MiniMaxH3FinalLayer,
    MiniMaxH3RotaryPosEmbed,
    MiniMaxH3TokenRefiner,
    MiniMaxH3TransformerBlock,
)
from tests.maltrainer.models.minimax_h3.h3_tiny import HIDDEN, tiny_config

CHUNKS = ("shift_attn", "scale_attn", "gate_attn", "shift_mlp", "scale_mlp", "gate_mlp")


def _block_with_rows(rows: list[dict[str, float]]) -> MiniMaxH3TransformerBlock:
    torch.manual_seed(0)
    block = MiniMaxH3TransformerBlock(tiny_config()).eval()
    bias = torch.zeros(len(rows), len(CHUNKS), HIDDEN)
    for row, values in enumerate(rows):
        for chunk, name in enumerate(CHUNKS):
            bias[row, chunk] = values.get(name, 0.0)
    with torch.no_grad():
        block.adaln_proj.linear.weight.zero_()
        block.adaln_proj.linear.bias.copy_(bias.flatten())
    return block


def test_the_feed_forward_is_a_swiglu_with_the_gate_in_the_first_half_of_fc1() -> None:
    ffn = MiniMaxH3FeedForward(2, 1)
    with torch.no_grad():
        ffn.fc1.weight.copy_(torch.eye(2))
        ffn.fc2.weight.copy_(torch.tensor([[2.0], [3.0]]))

    out = ffn(torch.tensor([[1.0, 4.0]]))

    activated = 4.0 / (1.0 + math.exp(-1.0))
    torch.testing.assert_close(out, torch.tensor([[2.0 * activated, 3.0 * activated]]))


def test_the_adaln_zero_block_modulates_and_gates_each_token_by_its_timestep_and_tag_row() -> None:
    rows = [
        {
            "shift_attn": 0.25,
            "scale_attn": 0.5,
            "gate_attn": 1.0,
            "shift_mlp": -0.5,
            "scale_mlp": 0.25,
            "gate_mlp": 2.0,
        },
        {"gate_attn": 0.0, "gate_mlp": 0.5, "scale_mlp": -0.5},
        {"gate_attn": -1.0, "shift_attn": 1.0},
    ]
    block = _block_with_rows(rows)
    hidden = torch.randn(1, 4, HIDDEN, generator=torch.Generator().manual_seed(1))
    indices = torch.tensor([0, 1, 2, 1])
    rotary = MiniMaxH3RotaryPosEmbed(1, 10000.0)(torch.arange(4.0)[:, None].expand(4, 3))

    out = block(hidden, torch.zeros(1, 10), indices, rotary, None)

    mod = {name: torch.stack([torch.full((HIDDEN,), r.get(name, 0.0)) for r in rows])[indices] for name in CHUNKS}
    attn_in = block.norm1(hidden) * (1 + mod["scale_attn"]) + mod["shift_attn"]
    mid = hidden + mod["gate_attn"] * block.attn(attn_in, rotary, None)
    mlp_in = block.norm2(mid) * (1 + mod["scale_mlp"]) + mod["shift_mlp"]
    torch.testing.assert_close(out, mid + mod["gate_mlp"] * block.mlp(mlp_in))


def test_a_zero_gate_makes_the_block_an_identity_for_that_token() -> None:
    block = _block_with_rows([{}] * 3)
    hidden = torch.randn(1, 3, HIDDEN, generator=torch.Generator().manual_seed(2))
    rotary = MiniMaxH3RotaryPosEmbed(1, 10000.0)(torch.zeros(3, 3))

    out = block(hidden, torch.zeros(1, 10), torch.tensor([0, 1, 2]), rotary, None)

    torch.testing.assert_close(out, hidden)


def test_the_block_modulation_is_computed_in_the_dtype_of_the_activations() -> None:
    block = _block_with_rows([{"gate_attn": 1.0, "gate_mlp": 1.0}] * 3).to(torch.bfloat16)
    rotary = MiniMaxH3RotaryPosEmbed(1, 10000.0)(torch.zeros(2, 3))

    out = block(
        torch.randn(1, 2, HIDDEN, dtype=torch.bfloat16), torch.zeros(1, 10), torch.zeros(2).long(), rotary, None
    )

    assert out.dtype == torch.bfloat16


def test_the_final_layer_modulates_per_timestep_row_then_projects_each_modality_to_its_own_width() -> None:
    torch.manual_seed(0)
    layer = MiniMaxH3FinalLayer(tiny_config()).eval()
    hidden = torch.randn(1, 4, HIDDEN, generator=torch.Generator().manual_seed(3))
    embedding = torch.randn(2, 10, generator=torch.Generator().manual_seed(4))

    out = layer(hidden, embedding, torch.tensor([0, 0, 1, 1]), torch.tensor([0, 2]), torch.tensor([1, 3]))

    shift, scale = layer.adaln_proj(embedding).chunk(2, dim=-1)
    modulated = layer.norm(hidden) * (1 + scale[[0, 0, 1, 1]]) + shift[[0, 0, 1, 1]]
    torch.testing.assert_close(out.video, layer.video_out(modulated)[:, [0, 2]])
    torch.testing.assert_close(out.audio, layer.audio_out(modulated)[:, [1, 3]])
    assert (out.video.shape, out.audio.shape) == ((1, 2, 8), (1, 2, 3))


def test_each_head_of_the_final_layer_answers_in_its_own_weight_dtype() -> None:
    layer = MiniMaxH3FinalLayer(tiny_config()).eval()
    layer.video_out.to(torch.bfloat16)

    out = layer(
        torch.randn(1, 2, HIDDEN), torch.zeros(1, 10), torch.zeros(2).long(), torch.tensor([0]), torch.tensor([1])
    )

    assert (out.video.dtype, out.audio.dtype) == (torch.bfloat16, torch.float32)


def test_the_token_refiner_is_residual_blocks_followed_by_a_final_norm() -> None:
    torch.manual_seed(0)
    refiner = MiniMaxH3TokenRefiner(tiny_config(num_refiner_layers=2)).eval()
    hidden = torch.randn(1, 3, HIDDEN)
    for block in refiner.blocks:
        torch.nn.init.zeros_(block.attn.out_proj.weight)
        torch.nn.init.zeros_(block.mlp.fc2.weight)

    out = refiner(hidden)

    assert len(refiner.blocks) == 2
    torch.testing.assert_close(out, F.rms_norm(hidden, (HIDDEN,), refiner.final_norm.weight, 1e-5))
