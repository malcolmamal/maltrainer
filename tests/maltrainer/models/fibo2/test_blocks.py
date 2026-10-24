from __future__ import annotations

import torch

from maltrainer.models.fibo2.blocks import Block, modulate
from maltrainer.models.fibo2.layers import FeedForward, RMSNorm
from maltrainer.models.fibo2.rope import grid_ids, rope_angles

from .tiny_model import CAP, TINY


def test_modulate_scales_only_the_span_by_the_noisy_value_and_the_rest_by_the_clean_one() -> None:
    x = torch.ones(1, 5, 2)

    out = modulate(x, torch.full((1, 2), 3.0), torch.full((1, 2), 7.0), (1, 3))

    assert out[0, :, 0].tolist() == [7.0, 3.0, 3.0, 7.0, 7.0]


def test_modulate_without_a_span_scales_every_token_by_the_noisy_value() -> None:
    x = torch.ones(1, 3, 2)

    assert modulate(x, torch.full((1, 2), 3.0), torch.full((1, 2), 7.0), None).unique().tolist() == [3.0]
    assert modulate(x, torch.full((1, 2), 3.0), None, (1, 2)).unique().tolist() == [3.0]


def _block(injection: bool = False) -> Block:
    torch.manual_seed(1)
    block = Block(TINY, injection)
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.05)
    return block.eval()


def _rope(tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    return rope_angles(grid_ids(0, 1, tokens, torch.device("cpu")))


def test_a_span_with_equal_noisy_and_clean_modulation_is_the_same_as_no_span() -> None:
    block = _block()
    x = torch.randn(1, 5, TINY.dim)
    adaln = torch.randn(1, 256)

    plain = block(x, _rope(5), adaln, None)
    spanned = block(x, _rope(5), adaln.repeat(2, 1), (1, 3))

    assert torch.allclose(plain, spanned, atol=1e-5)


def test_the_clean_tokens_are_modulated_by_the_second_adaln_row() -> None:
    block = _block()
    x = torch.randn(1, 5, TINY.dim)
    noisy, clean_a, clean_b = torch.randn(1, 256), torch.randn(1, 256), torch.randn(1, 256)

    with_a = block(x, _rope(5), torch.cat((noisy, clean_a)), (1, 3))
    with_b = block(x, _rope(5), torch.cat((noisy, clean_b)), (1, 3))

    assert not torch.allclose(with_a[:, 0], with_b[:, 0])


def test_an_injection_block_attends_to_its_text_bundle_and_the_gate_scales_it() -> None:
    block = _block(injection=True)
    x, adaln = torch.randn(1, 4, TINY.dim), torch.randn(1, 256)
    text = torch.randn(1, 3, CAP)
    with torch.no_grad():
        block.cross_attn_gate.fill_(0.0)
    closed = block(x, _rope(4), adaln, None, text)
    with torch.no_grad():
        block.cross_attn_gate.fill_(1.0)
    opened = block(x, _rope(4), adaln, None, text)

    assert torch.allclose(closed, block(x, _rope(4), adaln, None, None))
    assert not torch.allclose(closed, opened)


def test_a_plain_block_has_no_cross_attention_and_ignores_a_text_bundle() -> None:
    block = _block()
    x, adaln = torch.randn(1, 4, TINY.dim), torch.randn(1, 256)

    assert block.cross_attention is None
    assert torch.equal(block(x, _rope(4), adaln, None, torch.randn(1, 3, CAP)), block(x, _rope(4), adaln, None))


def test_rms_norm_normalises_in_fp32_and_rounds_before_scaling() -> None:
    norm = RMSNorm(4, 1e-5)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    x = torch.tensor([[1.0, 1.0, 1.0, 1.0]]).bfloat16()

    out = norm(x)

    assert out.dtype == torch.bfloat16
    assert out.float().tolist() == [[1.0, 2.0, 3.0, 4.0]]


def test_the_feed_forward_is_swiglu_with_the_gate_in_the_first_half() -> None:
    ffn = FeedForward(2, 3)
    with torch.no_grad():
        ffn.w13.weight.copy_(torch.cat((torch.eye(3, 2) + 1, -torch.ones(3, 2))))
        ffn.w2.weight.copy_(torch.ones(2, 3))
    x = torch.tensor([[1.0, 2.0]])
    gate, up = (ffn.w13.weight[:3] @ x[0]), (ffn.w13.weight[3:] @ x[0])

    assert torch.allclose(ffn(x)[0], torch.nn.functional.silu(gate).mul(up).sum().expand(2))


def test_the_positions_of_the_tokens_change_the_attention() -> None:
    block = _block()
    x, adaln = torch.randn(1, 4, TINY.dim), torch.randn(1, 256)
    ids = grid_ids(0, 1, 4, torch.device("cpu"))
    moved = ids.clone()
    moved[3, 2] = 9  # one token elsewhere: a uniform shift would leave RoPE's relative positions alone

    assert not torch.allclose(block(x, rope_angles(ids), adaln, None), block(x, rope_angles(moved), adaln, None))
