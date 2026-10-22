"""``MOSSImageConfig`` defaults and the checkpoint sniffing, pinned (mutation testing)."""

from __future__ import annotations

import pytest
import torch

from maltrainer.models.moss_image.model_config import MOSSImageConfig


def test_the_default_config_is_the_shipped_model() -> None:
    config = MOSSImageConfig()

    assert (config.dim, config.ffn_dim, config.num_heads, config.num_layers, config.text_dim) == (
        3072,
        8192,
        24,
        42,
        4096,
    )
    assert config.fused_qkv is False


def test_the_config_is_frozen_and_so_hashable() -> None:
    assert hash(MOSSImageConfig()) == hash(MOSSImageConfig())  # an unfrozen dataclass with eq has no hash


@pytest.mark.parametrize("present", ["shared_block_modulation.weight", "layers.0.proj.ffn_up_gate.weight"])
def test_a_checkpoint_with_only_one_of_the_two_marker_keys_is_refused(present: str) -> None:
    with pytest.raises(ValueError, match="Not a MOSS-Image DiT checkpoint"):
        MOSSImageConfig.from_state_dict({present: torch.empty(12, 4)})
