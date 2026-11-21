from __future__ import annotations

import pytest
import torch

from maltrainer.models.ltx25.config import AUDIO_BINS, AUDIO_CHANNELS

from .tiny_model import TINY, tiny_inputs, tiny_transformer


def test_the_outputs_have_the_latents_shapes() -> None:
    video, sigma, features, audio, audio_sigma = tiny_inputs(frames=2, height=3, width=2, audio_frames=6)

    video_out, audio_out = tiny_transformer()(video, sigma, features, 24.0, audio, audio_sigma)

    assert video_out.shape == video.shape
    assert audio_out is not None and audio_out.shape == (1, AUDIO_CHANNELS, 6, AUDIO_BINS)


def test_the_prediction_is_the_one_comfyuis_ltxav_module_gives_for_the_same_weights() -> None:
    """Golden values taken from comfy/ldm/lightricks/av_model.py (checkout of 2026-10-24) run on this tiny model's
    weights; the two agreed to 2e-6 at the time. They pin the RoPE, the timestep modulation of both streams, the
    cross-modal attention and the connectors."""
    video, sigma, features, audio, audio_sigma = tiny_inputs()

    video_out, audio_out = tiny_transformer()(video, sigma, features, 24.0, audio, audio_sigma)

    assert video_out.flatten()[:6].tolist() == pytest.approx(
        [1.27823, -0.03461, 0.60781, 1.08219, 0.13606, 1.66918], abs=1e-4
    )
    assert video_out.sum().item() == pytest.approx(118.8459, abs=5e-3)
    assert video_out.abs().mean().item() == pytest.approx(1.250901, abs=1e-5)
    assert audio_out is not None
    assert audio_out.flatten()[:6].tolist() == pytest.approx(
        [-1.12773, -0.18062, -1.24151, 0.32758, 0.26992, 0.46642], abs=1e-4
    )
    assert audio_out.sum().item() == pytest.approx(6.0774, abs=5e-3)


def test_without_audio_it_is_a_video_model_that_the_audio_stream_does_not_influence() -> None:
    video, sigma, features, audio, audio_sigma = tiny_inputs()
    model = tiny_transformer()

    alone, no_audio = model(video, sigma, features)
    joint, _ = model(video, sigma, features, 24.0, audio, audio_sigma)

    assert no_audio is None
    assert not torch.allclose(alone, joint)


def test_a_model_built_without_audio_has_no_audio_modules_and_ignores_audio_input() -> None:
    video, sigma, features, audio, audio_sigma = tiny_inputs()
    model = tiny_transformer(with_audio=False)

    out, audio_out = model(video, sigma, features, 24.0, audio, audio_sigma)

    assert audio_out is None and out.shape == video.shape
    assert not any("audio" in name or "av_ca" in name or "a2v_ca" in name for name in model.state_dict())


def test_the_audio_stream_depends_on_the_video_and_the_video_on_the_audio() -> None:
    video, sigma, features, audio, audio_sigma = tiny_inputs()
    model = tiny_transformer()
    base_video, base_audio = model(video, sigma, features, 24.0, audio, audio_sigma)

    _, moved_audio = model(video + 1.0, sigma, features, 24.0, audio, audio_sigma)
    video_after_audio, _ = model(video, sigma, features, 24.0, audio + 1.0, audio_sigma)

    assert base_audio is not None and moved_audio is not None
    assert not torch.allclose(base_audio, moved_audio)
    assert not torch.allclose(base_video, video_after_audio)


def test_sigma_the_text_and_the_frame_rate_all_change_the_prediction() -> None:
    video, sigma, features, audio, audio_sigma = tiny_inputs()
    model = tiny_transformer()
    base, _ = model(video, sigma, features, 24.0, audio, audio_sigma)

    assert not torch.allclose(base, model(video, sigma + 0.3, features, 24.0, audio, audio_sigma + 0.3)[0])
    assert not torch.allclose(base, model(video, sigma, features + 1.0, 24.0, audio, audio_sigma)[0])
    assert not torch.allclose(base, model(video, sigma, features, 12.0, audio, audio_sigma)[0])


def test_the_audio_sigma_defaults_to_the_video_sigma() -> None:
    video, sigma, features, audio, _ = tiny_inputs()
    model = tiny_transformer()

    assert torch.equal(
        model(video, sigma, features, 24.0, audio)[0], model(video, sigma, features, 24.0, audio, sigma)[0]
    )


def test_a_per_token_sigma_that_is_constant_equals_a_per_sample_one() -> None:
    video, sigma, features, _, _ = tiny_inputs()
    model = tiny_transformer()
    tokens = video.shape[2] * video.shape[3] * video.shape[4]

    per_sample, _ = model(video, sigma, features)
    per_token, _ = model(video, sigma.expand(1, tokens), features)

    torch.testing.assert_close(per_sample, per_token, atol=1e-5, rtol=1e-4)


def test_a_batch_is_the_samples_run_one_after_another() -> None:
    video, sigma, features, audio, audio_sigma = tiny_inputs()
    model = tiny_transformer()
    batch = (
        video.repeat(2, 1, 1, 1, 1),
        sigma.repeat(2),
        features.repeat(2, 1, 1),
        audio.repeat(2, 1, 1, 1),
        audio_sigma.repeat(2),
    )

    batched_video, batched_audio = model(batch[0], batch[1], batch[2], 24.0, batch[3], batch[4])
    single_video, single_audio = model(video, sigma, features, 24.0, audio, audio_sigma)

    torch.testing.assert_close(batched_video[0], single_video[0], atol=1e-5, rtol=1e-4)
    assert batched_audio is not None and single_audio is not None
    torch.testing.assert_close(batched_audio[1], single_audio[0], atol=1e-5, rtol=1e-4)


def test_gradient_checkpointing_changes_neither_the_output_nor_the_gradients() -> None:
    video, sigma, features, audio, audio_sigma = tiny_inputs()
    video.requires_grad_(True)
    plain, checkpointed = tiny_transformer(), tiny_transformer()
    checkpointed.enable_gradient_checkpointing()

    grads = []
    for model in (plain, checkpointed):
        out, audio_out = model(video, sigma, features, 24.0, audio, audio_sigma)
        assert audio_out is not None
        grads.append(torch.autograd.grad(out.square().sum() + audio_out.square().sum(), video)[0])

    assert torch.allclose(grads[0], grads[1], atol=1e-4)


def test_the_state_dict_uses_comfyuis_module_names() -> None:
    keys = set(tiny_transformer().state_dict())

    assert {
        "patchify_proj.weight",
        "adaln_single.emb.timestep_embedder.linear_1.weight",
        "adaln_single.linear.bias",
        "prompt_adaln_single.linear.weight",
        "audio_patchify_proj.weight",
        "av_ca_a2v_gate_adaln_single.linear.weight",
        "av_ca_audio_scale_shift_adaln_single.emb.timestep_embedder.linear_2.bias",
        "video_embeddings_connector.learnable_registers",
        "audio_embeddings_connector.transformer_1d_blocks.0.attn1.to_gate_logits.weight",
        "transformer_blocks.0.attn1.to_out.0.weight",
        "transformer_blocks.0.attn1.q_norm.weight",
        "transformer_blocks.1.ff.net.0.proj.weight",
        "transformer_blocks.0.prompt_scale_shift_table",
        "transformer_blocks.0.scale_shift_table_a2v_ca_video",
        "transformer_blocks.1.audio_to_video_attn.to_k.weight",
        "scale_shift_table",
        "audio_scale_shift_table",
        "proj_out.weight",
        "audio_proj_out.bias",
    } <= keys
    assert TINY.num_layers == 2


def test_an_audio_model_refuses_nothing_when_the_text_has_a_different_length() -> None:
    video, sigma, features, audio, audio_sigma = tiny_inputs(length=1500)

    out, _ = tiny_transformer()(video, sigma, features, 24.0, audio, audio_sigma)

    assert out.shape == video.shape


def test_the_compute_dtype_decides_the_activation_dtype() -> None:
    model = tiny_transformer()
    model.compute_dtype = torch.bfloat16
    model.to(torch.bfloat16)
    video, sigma, features, audio, audio_sigma = tiny_inputs()

    out, audio_out = model(video, sigma, features, 24.0, audio, audio_sigma)

    assert out.dtype == torch.bfloat16 and audio_out is not None and audio_out.dtype == torch.bfloat16
