from __future__ import annotations

import pytest
import torch

from maltrainer.models.minimax_h3 import model as h3_model
from maltrainer.models.minimax_h3.model import MiniMaxH3TokenTag, MiniMaxH3Transformer, MiniMaxH3TransformerConfig
from tests.maltrainer.models.minimax_h3.h3_tiny import HIDDEN, Packed, tiny_config, tiny_transformer


def _capture_inputs(module: torch.nn.Module) -> list[tuple]:
    seen: list[tuple] = []
    module.register_forward_pre_hook(lambda _module, args: seen.append(args))
    return seen


def test_the_outputs_are_video_patches_and_audio_rows_for_every_batch_item() -> None:
    packed = Packed(text=3, audio=2, video=4)

    out = tiny_transformer()(**packed.inputs(batch=2))

    assert out.video.shape == (2, 4, 8)
    assert out.audio.shape == (2, 2, 3)


def test_the_packed_sequence_puts_each_modality_at_its_own_indices() -> None:
    model = tiny_transformer(num_layers=0)
    inputs = Packed(text=2, audio=2, video=3).inputs()
    inputs.update(
        text_indices=torch.tensor([0, 4]),
        audio_indices=torch.tensor([1, 5]),
        video_indices=torch.tensor([2, 3, 6]),
        token_tags=torch.tensor([1, 2, 0, 0, 1, 2, 0]),
        timestep_indices=torch.tensor([0, 1, 0, 0, 0, 1, 0]),
        position_ids=torch.zeros(7, 3),
    )
    packed = _capture_inputs(model.final_layer)

    model(**inputs)

    hidden = packed[0][0]
    text = model.token_refiner(model.condition_proj(inputs["encoder_hidden_states"]))
    torch.testing.assert_close(hidden[:, [0, 4]], text)
    torch.testing.assert_close(hidden[:, [1, 5]], model.audio_patch_proj(inputs["audio_hidden_states"]))
    torch.testing.assert_close(hidden[:, [2, 3, 6]], model.video_patch_proj(inputs["video_hidden_states"]))


def test_each_token_picks_the_adaln_row_of_its_timestep_times_three_plus_its_modality_tag() -> None:
    model = tiny_transformer(num_layers=1)
    packed = Packed(text=2, audio=2, video=2)
    inputs = packed.inputs()
    seen = _capture_inputs(model.blocks[0])

    model(**inputs)

    expected = [1, 1, 3 + 2, 3 + 2, 0, 0]
    assert seen[0][2].tolist() == expected
    assert (MiniMaxH3TokenTag.TEXT, MiniMaxH3TokenTag.AUDIO) == (1, 2)


def test_a_negative_modality_tag_is_treated_as_video() -> None:
    model = tiny_transformer(num_layers=1)
    inputs = Packed(text=1, audio=1, video=2).inputs()
    inputs["token_tags"] = torch.tensor([1, -1, -1, 0])
    seen = _capture_inputs(model.blocks[0])

    model(**inputs)

    assert seen[0][2].tolist() == [1, 3, 0, 0]


def test_the_two_timestep_rows_matter_only_when_they_differ() -> None:
    model = tiny_transformer()
    inputs = Packed(text=2, audio=2, video=2).inputs()
    all_video_row = {**inputs, "timestep_indices": torch.zeros_like(inputs["timestep_indices"])}

    equal_rows = model(**{**inputs, "timestep": torch.tensor([0.4, 0.4])})
    one_row_used = model(**{**all_video_row, "timestep": torch.tensor([0.4, 0.4])})
    different_rows = model(**{**inputs, "timestep": torch.tensor([0.4, 0.9])})

    torch.testing.assert_close(equal_rows.video, one_row_used.video)
    assert not torch.allclose(equal_rows.video, different_rows.video, atol=1e-4)


def test_a_diagonal_attention_mask_isolates_the_video_tokens_from_the_text() -> None:
    model = tiny_transformer()
    packed = Packed(text=2, audio=1, video=2)
    open_attention = packed.inputs()
    isolated = {**open_attention, "attention_mask": torch.eye(packed.length, dtype=torch.bool)}
    shift = {"encoder_hidden_states": open_attention["encoder_hidden_states"] + 3.0}

    isolated_video = model(**isolated).video
    isolated_video_new_text = model(**{**isolated, **shift}).video

    torch.testing.assert_close(isolated_video, isolated_video_new_text)
    assert not torch.allclose(model(**open_attention).video, model(**{**open_attention, **shift}).video, atol=1e-4)


def test_inputs_are_cast_to_the_dtype_of_the_weights() -> None:
    model = tiny_transformer().to(torch.float64)

    out = model(**Packed(text=2, audio=1, video=2).inputs())

    assert (out.video.dtype, out.audio.dtype) == (torch.float64, torch.float64)
    assert (model.dtype, model.device) == (torch.float64, torch.device("cpu"))


@pytest.mark.parametrize(("enabled", "expected_calls"), [(True, 3 + 1), (False, 0)])
def test_gradient_checkpointing_wraps_every_block_and_refiner_block_while_grads_are_on(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, expected_calls: int
) -> None:
    calls: list[int] = []

    def spy(function, *args, **_kwargs):
        calls.append(1)
        return function(*args)

    monkeypatch.setattr(h3_model, "checkpoint", spy)
    model = tiny_transformer()
    model.enable_gradient_checkpointing()
    if not enabled:
        model.disable_gradient_checkpointing()

    model(**Packed(text=2, audio=1, video=2).inputs())

    assert model.gradient_checkpointing is enabled
    assert len(calls) == expected_calls


def test_checkpointing_is_skipped_without_gradients(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(h3_model, "checkpoint", lambda *_a, **_k: pytest.fail("checkpoint used under no_grad"))
    model = tiny_transformer()
    model.enable_gradient_checkpointing()

    with torch.no_grad():
        model(**Packed(text=2, audio=1, video=2).inputs())


def test_checkpointing_gives_the_same_outputs_and_gradients() -> None:
    inputs = Packed(text=2, audio=1, video=3).inputs()
    plain, checkpointed = tiny_transformer(), tiny_transformer()
    checkpointed.enable_gradient_checkpointing()

    plain_out, checkpointed_out = plain(**inputs), checkpointed(**inputs)
    (plain_out.video.sum() + plain_out.audio.sum()).backward()
    (checkpointed_out.video.sum() + checkpointed_out.audio.sum()).backward()

    torch.testing.assert_close(checkpointed_out.video, plain_out.video)
    name = "blocks.0.attn.qkv_proj.weight"
    torch.testing.assert_close(
        dict(checkpointed.named_parameters())[name].grad, dict(plain.named_parameters())[name].grad
    )


def test_a_rotary_width_wider_than_the_attention_head_is_rejected() -> None:
    with pytest.raises(ValueError, match="rotary width 12 exceeds attention head width 8"):
        MiniMaxH3Transformer(tiny_config(rope_freq_dim=2))


def test_the_default_config_builds_fifty_blocks() -> None:
    with torch.device("meta"):
        model = MiniMaxH3Transformer()

    assert model.config == MiniMaxH3TransformerConfig()
    assert len(model.blocks) == 50
    assert model.blocks[0].adaln_proj.linear.weight.shape == (6 * 5376 * 3, 2688)


def test_the_state_dict_names_follow_the_checkpoint_layout() -> None:
    keys = set(tiny_transformer(num_layers=1).state_dict())

    assert {
        "rope.inv_freq",
        "video_patch_proj.weight",
        "audio_patch_proj.bias",
        "condition_proj.weight",
        "time_embedder.proj_in.weight",
        "token_refiner.blocks.0.attn.qkv_proj.weight",
        "token_refiner.final_norm.weight",
        "blocks.0.attn.q_norm.weight",
        "blocks.0.mlp.fc1.weight",
        "blocks.0.adaln_proj.linear.bias",
        "final_layer.adaln_proj.linear.weight",
        "final_layer.audio_out.bias",
    } <= keys
    assert "blocks.0.attn.qkv_proj.bias" not in keys
    assert "adaln_t_table" not in keys
    assert tiny_transformer().blocks[0].hidden_size == HIDDEN
