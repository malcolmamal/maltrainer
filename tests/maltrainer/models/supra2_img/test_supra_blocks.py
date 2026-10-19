from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from maltrainer.models.supra2_img.model import Attention, DiTBlock, FinalLayer, TimestepEmbedder, modulate

DIM, HEADS, CTX = 16, 4, 12


def _zero(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        torch.nn.init.zeros_(parameter)


def _attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None, heads: int) -> torch.Tensor:
    batch, length, channels = q.shape

    def split(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(batch, t.shape[1], heads, channels // heads).transpose(1, 2)

    scores = split(q) @ split(k).transpose(-1, -2) / math.sqrt(channels // heads)
    if mask is not None:
        scores = scores.masked_fill(~mask.bool()[:, None, None, :], float("-inf"))
    return (scores.softmax(dim=-1) @ split(v)).transpose(1, 2).reshape(batch, length, channels)


def test_adaln_scales_by_one_plus_scale_and_shifts_per_sample() -> None:
    x = torch.ones(2, 3, 2)
    shift, scale = torch.tensor([[1.0, 0.0], [0.0, 0.0]]), torch.tensor([[1.0, 2.0], [-1.0, 0.0]])

    out = modulate(x, shift, scale)

    torch.testing.assert_close(out[0], torch.tensor([[3.0, 3.0]] * 3))
    torch.testing.assert_close(out[1], torch.tensor([[0.0, 1.0]] * 3))


def test_timestep_embedding_is_cosines_then_sines_of_scaled_frequencies() -> None:
    embedder = TimestepEmbedder(8, freq_dim=8)

    emb = embedder._sinusoidal(torch.tensor([0.0, 0.001]))

    freqs = torch.tensor([1.0, 0.1, 0.01, 0.001])
    torch.testing.assert_close(emb[0], torch.tensor([1.0] * 4 + [0.0] * 4))
    torch.testing.assert_close(emb[1], torch.cat([freqs.cos(), freqs.sin()]), rtol=1e-5, atol=1e-6)


def test_an_odd_frequency_dimension_is_padded_with_a_zero_column() -> None:
    embedder = TimestepEmbedder(8, freq_dim=9)

    emb = embedder._sinusoidal(torch.tensor([0.3]))

    assert emb.shape == (1, 9)
    assert emb[0, -1] == 0.0
    assert embedder(torch.tensor([0.3, 0.9])).shape == (2, 8)


def test_self_attention_matches_a_hand_written_multi_head_softmax_attention() -> None:
    torch.manual_seed(0)
    attention, x = Attention(DIM, HEADS), torch.randn(2, 5, DIM)

    out = attention(x)

    q, k, v = F.linear(x, attention.qkv.weight, attention.qkv.bias).chunk(3, dim=-1)
    torch.testing.assert_close(out, attention.proj(_attend(q, k, v, None, HEADS)), rtol=1e-4, atol=1e-5)
    assert sorted(name for name, _ in attention.named_children()) == ["proj", "qkv"]


def test_cross_attention_attends_over_the_unmasked_context_only() -> None:
    torch.manual_seed(1)
    attention, x, ctx = Attention(DIM, HEADS, ctx_dim=CTX), torch.randn(2, 5, DIM), torch.randn(2, 7, CTX)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1]]).float()

    out = attention(x, ctx=ctx, ctx_mask=mask)

    k, v = F.linear(ctx, attention.kv.weight, attention.kv.bias).chunk(2, dim=-1)
    expected = attention.proj(_attend(attention.q(x), k, v, mask, HEADS))
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-5)
    assert sorted(name for name, _ in attention.named_children()) == ["kv", "proj", "q"]


def test_masked_context_tokens_do_not_change_the_output() -> None:
    torch.manual_seed(2)
    attention, x, ctx = Attention(DIM, HEADS, ctx_dim=CTX), torch.randn(1, 3, DIM), torch.randn(1, 6, CTX)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0]])
    changed = ctx.clone()
    changed[:, 3:] = 100.0

    assert torch.allclose(attention(x, ctx, mask), attention(x, changed, mask), atol=1e-6)
    assert not torch.allclose(attention(x, ctx), attention(x, changed), atol=1e-3)


def test_a_block_with_zeroed_conditioning_only_adds_cross_attention() -> None:
    torch.manual_seed(3)
    block, x, c, ctx = (
        DiTBlock(DIM, HEADS, DIM, 4.0),
        torch.randn(2, 5, DIM),
        torch.randn(2, DIM),
        torch.randn(2, 4, DIM),
    )
    _zero(block.adaln)

    out = block(x, c, ctx, None)

    torch.testing.assert_close(out, x + block.cross_attn(block.norm_ca(x), ctx=ctx), rtol=1e-4, atol=1e-5)


def test_a_block_keeps_its_input_shape_and_its_mlp_is_widened_by_the_ratio() -> None:
    block = DiTBlock(DIM, HEADS, DIM, 2.5)

    out = block(torch.randn(2, 5, DIM), torch.randn(2, DIM), torch.randn(2, 4, DIM), torch.ones(2, 4))

    assert out.shape == (2, 5, DIM)
    assert (block.mlp[0].out_features, block.adaln[1].out_features) == (40, 6 * DIM)


def test_the_final_layer_projects_the_normalized_tokens_when_conditioning_is_zero() -> None:
    layer, x = FinalLayer(DIM, 8), torch.randn(2, 5, DIM)
    _zero(layer.adaln)

    out = layer(x, torch.randn(2, DIM))

    torch.testing.assert_close(out, layer.linear(layer.norm(x)))
    assert out.shape == (2, 5, 8)
