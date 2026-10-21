"""The packed sign history and the stochastic rounding of Automagic3, bit by bit."""

from __future__ import annotations

import itertools

import torch

from maltrainer.optimizers.automagic3_ops import SignBits, StochasticRounding


def packed(*steps: list[int]) -> torch.Tensor:
    """One packed row per step, from lists of 0/1 per element."""
    return torch.stack([SignBits._pack_bits(torch.tensor(step)) for step in steps])


def test_bit_i_of_a_byte_is_element_i_and_the_last_byte_is_padded_with_zeros() -> None:
    assert SignBits._pack_bits(torch.tensor([1, 0, 0, 0, 0, 0, 0, 0])).tolist() == [1]
    assert SignBits._pack_bits(torch.tensor([0, 0, 0, 0, 0, 0, 0, 1])).tolist() == [128]
    assert SignBits._pack_bits(torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])).tolist() == [170]
    assert SignBits._pack_bits(torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])).tolist() == [255, 3]


def test_unpacking_returns_exactly_the_zeros_and_ones_that_were_packed() -> None:
    assert SignBits._unpack_bits(torch.tensor([170], dtype=torch.uint8), 8).tolist() == [0, 1, 0, 1, 0, 1, 0, 1]
    assert SignBits._unpack_bits(torch.tensor([255, 3], dtype=torch.uint8), 10).tolist() == [1] * 10
    every_byte = torch.arange(256, dtype=torch.uint8)
    bits = torch.tensor(list(itertools.chain.from_iterable([(b >> i) & 1 for i in range(8)] for b in range(256))))
    assert torch.equal(SignBits._unpack_bits(every_byte, 2048).long(), bits)
    assert torch.equal(SignBits._pack_bits(bits), every_byte)


def test_up_is_the_same_sign_at_every_step_and_down_is_a_flip_at_every_step() -> None:
    # elements A..D over three steps: A 0,1,1  B 0,1,0  C 1,1,1  D 0,0,0
    hist = packed([0, 0, 1, 0], [1, 1, 1, 0], [1, 0, 1, 0])

    up, down = SignBits._vote_masks(hist, 0, 4)

    assert up.tolist() == [0, 0, 1, 1]
    assert down.tolist() == [0, 1, 0, 0]


def test_the_history_is_read_oldest_first_from_where_the_ring_starts() -> None:
    # the same three steps stored rotated by one: the oldest row is at index 1
    hist = packed([1, 0, 1, 0], [0, 0, 1, 0], [1, 1, 1, 0])

    up, down = SignBits._vote_masks(hist, 1, 4)

    assert up.tolist() == [0, 0, 1, 1]
    assert down.tolist() == [0, 1, 0, 0]
    assert SignBits._vote_masks(hist, 0, 4)[1].tolist() != down.tolist()


def test_two_steps_of_history_vote_up_when_equal_and_down_when_different() -> None:
    up, down = SignBits._vote_masks(packed([0, 1, 0, 1], [0, 1, 1, 0]), 0, 4)

    assert up.tolist() == [1, 1, 0, 0]
    assert down.tolist() == [0, 0, 1, 1]


def test_truncating_to_fp16_precision_keeps_what_fits_and_moves_the_rest_by_at_most_a_step() -> None:
    fits = torch.arange(1, 2001, dtype=torch.float32) * 0.0625
    assert torch.equal(StochasticRounding._sr_truncate(fits.clone(), 13), fits)

    off = torch.full((4000,), 1.0 + 3 * 2.0**-23 + 2.0**-11)
    rounded = StochasticRounding._sr_truncate(off.clone(), 13)
    step = 2.0**-10
    assert set(rounded.tolist()) == {1.0, 1.0 + step}


def test_copying_into_half_precision_rounds_stochastically_and_never_by_more_than_one_step() -> None:
    src = torch.full((20000,), 1.0 + 2.0**-11)
    for dtype, step in ((torch.float16, 2.0**-10), (torch.bfloat16, 2.0**-7)):
        dst = torch.empty(20000, dtype=dtype)
        StochasticRounding._stochastic_copy_(dst, src.clone())

        assert set(dst.float().tolist()) <= {1.0, 1.0 + step}
    low = torch.empty(20000, dtype=torch.float16)
    StochasticRounding._stochastic_copy_(low, src.clone())
    assert 0.3 < (low.float() > 1.0).float().mean() < 0.7


def test_copying_into_another_dtype_adds_noise_below_one_ulp() -> None:
    src = torch.full((1000,), 1.5)
    dst = torch.empty(1000, dtype=torch.float64)

    StochasticRounding._stochastic_copy_(dst, src.clone())

    assert torch.allclose(dst.float(), src, atol=2e-7)
