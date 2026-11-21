from __future__ import annotations

import json

import pytest

from maltrainer.models.ltx25.config import ConnectorConfig, LTX25Config, config_from_state, transformer_metadata

from .tiny_model import TINY, tiny_transformer

HEADS = {
    "config": json.dumps(
        {
            "transformer": {
                "num_attention_heads": 2,
                "audio_num_attention_heads": 2,
                "connector_attention_head_dim": 48,
                "audio_connector_attention_head_dim": 24,
            }
        }
    )
}


def test_the_sizes_are_read_off_the_state_dict() -> None:
    config = config_from_state(tiny_transformer().state_dict())

    assert config.num_layers == 2
    assert (config.num_heads, config.head_dim) == (32, 3)  # 32 heads unless the metadata says otherwise (ComfyUI)
    assert (config.cross_attention_dim, config.audio_cross_attention_dim) == (96, 48)
    assert config.gated_attention and config.cross_attention_adaln and config.prompt_adaln and config.ff_bias
    assert not config.caption_projection


def test_the_heads_and_the_connectors_are_read_off_the_metadata_and_the_tensors() -> None:
    config = config_from_state(tiny_transformer().state_dict(), HEADS)

    assert (config.num_heads, config.head_dim, config.audio_num_heads, config.audio_head_dim) == (2, 48, 2, 24)
    assert config.video_connector == ConnectorConfig(num_heads=2, head_dim=48, num_layers=1, registers=128, gated=True)
    assert config.audio_connector == ConnectorConfig(num_heads=2, head_dim=24, num_layers=1, registers=128, gated=True)
    assert config.video_connector.inner_dim == 96


def test_a_file_without_the_optional_layers_is_detected_as_such() -> None:
    state = {
        k: v
        for k, v in tiny_transformer().state_dict().items()
        if not k.endswith(("to_gate_logits.weight", "to_gate_logits.bias", "prompt_scale_shift_table"))
        and not k.startswith(("prompt_adaln_single.", "audio_prompt_adaln_single."))
    }

    config = config_from_state(state)

    assert not config.gated_attention and not config.cross_attention_adaln and not config.prompt_adaln
    assert config.ada_params == 6


def test_the_ada_params_are_nine_with_cross_attention_adaln() -> None:
    assert LTX25Config().ada_params == 9
    assert TINY.ada_params == 9


def test_the_metadata_overrides_the_rope_and_the_timestep_scales() -> None:
    meta = {
        "config": json.dumps(
            {
                "transformer": {
                    "rope_type": "interleaved",
                    "frequencies_precision": "float32",
                    "positional_embedding_theta": 5000.0,
                    "positional_embedding_max_pos": [10, 100, 100],
                    "use_middle_indices_grid": False,
                    "causal_temporal_positioning": False,
                    "timestep_scale_multiplier": 500,
                    "av_ca_timestep_scale_multiplier": 1,
                    "vae_scale_factors": [4, 16, 16],
                    "unknown_key": 1,
                }
            }
        )
    }

    config = config_from_state(tiny_transformer().state_dict(), meta)

    assert not config.split_rope and not config.double_precision_rope
    assert config.theta == 5000.0 and config.max_pos == (10, 100, 100) and config.vae_scale_factors == (4, 16, 16)
    assert not config.use_middle_indices_grid and not config.causal_temporal_positioning
    assert (config.timestep_scale, config.av_ca_timestep_scale) == (500, 1)


def test_without_metadata_the_ltx_2_3_defaults_hold() -> None:
    config = config_from_state(tiny_transformer().state_dict(), None)

    assert config.split_rope and config.double_precision_rope and config.use_middle_indices_grid
    assert config.causal_temporal_positioning and (config.timestep_scale, config.av_ca_timestep_scale) == (
        1000.0,
        1000.0,
    )


@pytest.mark.parametrize(
    "meta", [None, {}, {"other": "x"}, {"config": json.dumps({"vae": {}})}, {"config": json.dumps({"transformer": 3})}]
)
def test_metadata_without_a_transformer_section_gives_nothing(meta: dict[str, str] | None) -> None:
    assert transformer_metadata(meta) == {}


def test_the_caption_projection_is_detected_by_its_tensor() -> None:
    state = tiny_transformer().state_dict()
    state["caption_projection.linear_1.weight"] = state["patchify_proj.weight"]

    assert config_from_state(state).caption_projection
