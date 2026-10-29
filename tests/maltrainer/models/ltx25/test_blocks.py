from __future__ import annotations

from dataclasses import replace

import torch

from maltrainer.models.ltx25.blocks import BasicAVTransformerBlock, Stream, modulate
from maltrainer.models.ltx25.config import LTX25Config
from maltrainer.models.ltx25.rope import make_rope

from .tiny_model import TINY, randomise

V, A = TINY.inner_dim, TINY.audio_inner_dim


def _rope(tokens: int, dim: int, heads: int) -> tuple[torch.Tensor, torch.Tensor, bool]:
    return make_rope(torch.arange(float(tokens))[None, None, :], dim, heads, (20,), 10000.0, True, True, False)


def _stream(dim: int, heads: int, context_dim: int, tokens: int, ada: int = 9, seed: int = 0) -> Stream:
    generator = torch.Generator().manual_seed(seed)
    rope = _rope(tokens, dim, heads)
    return Stream(
        context=torch.randn(1, 5, context_dim, generator=generator),
        timestep=torch.randn(1, 1, ada * dim, generator=generator),
        pe=rope,
        prompt_timestep=torch.randn(1, 1, 2 * dim, generator=generator),
        cross_pe=_rope(tokens, TINY.audio_cross_attention_dim, TINY.audio_num_heads),
        cross_shift_timestep=torch.randn(1, 1, 4 * dim, generator=generator),
        cross_gate_timestep=torch.randn(1, 1, dim, generator=generator),
    )


def _block(config: LTX25Config = TINY, with_audio: bool = True) -> BasicAVTransformerBlock:
    block = BasicAVTransformerBlock(config, with_audio)
    randomise(block)
    return block.eval()


def test_modulate_scales_by_one_plus_scale_and_shifts() -> None:
    table = torch.zeros(2, 4)
    timestep = torch.cat((torch.full((1, 1, 4), 3.0), torch.full((1, 1, 4), 0.5)), dim=-1)  # shift, then scale
    x = torch.ones(1, 2, 4) * 2

    out = modulate(x, table, timestep, slice(0, 2))

    assert out.flatten().tolist() == [(1 + 0.5) * 1.0 + 3.0] * 8  # plain_rms of a constant row is 1


def test_a_block_without_audio_runs_the_video_stream_alone() -> None:
    block = _block(with_audio=False)
    vx = torch.randn(1, 6, V)

    out, audio = block(vx, None, _stream(V, TINY.num_heads, TINY.cross_attention_dim, 6), None)

    assert out.shape == vx.shape and audio is None
    assert not hasattr(block, "audio_attn1") and not hasattr(block, "scale_shift_table_a2v_ca_audio")


def test_audio_passed_to_a_block_changes_the_video_through_the_cross_modal_attention() -> None:
    block = _block()
    vx, ax = torch.randn(1, 6, V), torch.randn(1, 4, A)
    video = _stream(V, TINY.num_heads, TINY.cross_attention_dim, 6)
    audio = _stream(A, TINY.audio_num_heads, TINY.audio_cross_attention_dim, 4, seed=1)

    alone, _ = block(vx, None, video, None)
    joint_video, joint_audio = block(vx, ax, video, audio)

    assert joint_audio is not None and joint_audio.shape == ax.shape
    assert not torch.allclose(alone, joint_video)


def test_a_zero_gate_row_silences_the_branch_it_gates() -> None:
    block = _block(with_audio=False)
    vx = torch.randn(1, 6, V)
    stream = _stream(V, TINY.num_heads, TINY.cross_attention_dim, 6)
    stream.timestep.zero_()  # the gates are then exactly the table rows
    with torch.no_grad():
        block.scale_shift_table.zero_()
    block.eval()

    out, _ = block(vx, None, stream, None)

    assert torch.equal(out, vx)  # self-attention, text cross-attention and feed-forward are all gated shut


def test_the_text_cross_attention_without_adaln_uses_the_plain_normalised_query() -> None:
    plain_config = replace(TINY, cross_attention_adaln=False)
    block = _block(plain_config, with_audio=False)
    vx = torch.randn(1, 6, V)
    stream = _stream(V, TINY.num_heads, TINY.cross_attention_dim, 6, ada=6)
    stream.prompt_timestep = None

    out, _ = block(vx, None, stream, None)

    assert out.shape == vx.shape
    assert not hasattr(block, "prompt_scale_shift_table") and block.scale_shift_table.shape[0] == 6


def test_the_text_context_changes_the_result_in_both_streams() -> None:
    block = _block()
    vx, ax = torch.randn(1, 6, V), torch.randn(1, 4, A)
    video = _stream(V, TINY.num_heads, TINY.cross_attention_dim, 6)
    audio = _stream(A, TINY.audio_num_heads, TINY.audio_cross_attention_dim, 4, seed=1)
    base_v, base_a = block(vx, ax, video, audio)

    video.context = video.context + 1.0
    audio.context = audio.context + 1.0
    new_v, new_a = block(vx, ax, video, audio)

    assert not torch.allclose(base_v, new_v) and base_a is not None and new_a is not None
    assert not torch.allclose(base_a, new_a)
