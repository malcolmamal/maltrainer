from __future__ import annotations

import pytest
import torch

from maltrainer.models.visage.config import VisageConfig

from .tiny_model import TINY, tiny_transformer


def test_the_default_config_is_the_released_237m_dit() -> None:
    config = VisageConfig()

    assert (config.width, config.depth, config.heads, config.in_channels) == (768, 18, 12, 96)
    assert config.text_length_buckets == (32, 64, 96, 128)


def test_a_state_dict_gives_back_the_config_it_was_built_from() -> None:
    state = tiny_transformer().state_dict()

    config = VisageConfig.from_state_dict(state)

    assert config.width == 128 and config.depth == 2 and config.mlp_ratio == 2
    assert (config.in_channels, config.patch_size, config.text_dim) == (8, 1, 16)
    assert config.noise_embed_dims == 16 and config.n_text_registers == 2
    assert config.text_length_buckets == (8,)  # only the longest is known: the default buckets below it are kept


def test_the_buckets_below_the_longest_pad_row_count_are_the_default_ones() -> None:
    state = tiny_transformer().state_dict()
    state["text_pad_embed"] = torch.zeros(128, TINY.width)

    assert VisageConfig.from_state_dict(state).text_length_buckets == (32, 64, 96, 128)
    state["text_pad_embed"] = torch.zeros(64, TINY.width)
    assert VisageConfig.from_state_dict(state).text_length_buckets == (32, 64)


def test_a_file_that_is_not_the_dit_is_refused_naming_what_is_missing() -> None:
    with pytest.raises(ValueError, match=r"not a Visage-Image DiT checkpoint \(missing text_proj.weight"):
        VisageConfig.from_state_dict({"blocks.0.mlp.fc2.weight": torch.zeros(1, 1)})
