from __future__ import annotations

import pytest

from maltrainer.models.fibo2.config import BUNDLES, Fibo2Config, gist_count

from .tiny_model import TINY, tiny_transformer

REAL = Fibo2Config()


@pytest.mark.parametrize(
    ("length", "count"),
    [(1, 256), (256, 256), (257, 320), (300, 320), (1152, 384), (1500, 448), (2047, 512), (2048, 512), (4096, 512)],
)
def test_the_gist_count_rises_from_256_at_256_tokens_to_512_at_2048_rounded_up_to_steps_of_64(
    length: int, count: int
) -> None:
    assert gist_count(length, REAL) == count


def test_the_gist_count_never_exceeds_the_perceivers_queries() -> None:
    assert gist_count(4096, TINY) == TINY.max_gist_tokens


def test_the_sizes_are_read_off_the_state_dict() -> None:
    state = tiny_transformer().state_dict()

    assert Fibo2Config.from_state_dict(state) == TINY


def test_the_real_release_has_the_published_shape() -> None:
    assert (REAL.dim, REAL.n_heads, REAL.n_layers, REAL.injection_layers) == (3584, 28, 44, (6, 14, 22, 30, 38))
    assert BUNDLES * REAL.cap_feat_dim == 46080


def test_heads_that_do_not_fill_the_rope_axes_are_refused() -> None:
    with pytest.raises(ValueError, match="RoPE axes"):
        Fibo2Config(dim=256, n_heads=1)


def test_a_wrong_number_of_injection_layers_is_refused() -> None:
    with pytest.raises(ValueError, match="6 text bundles, got 2"):
        Fibo2Config(injection_layers=(1, 2))
