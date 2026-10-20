"""One training step on a cached sample: what the model is fed and how the loss is built, per kind of item."""

from __future__ import annotations

import pytest
import torch

import train_minimax_h3_lora as h3

from .steps import SHIFT, Calls, make_trainer, sample_of, step
from .world import Workspace


def test_a_still_packs_two_latents_of_noised_silence_and_trains_video_only(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = make_trainer(workspace)
    calls = Calls(trainer, monkeypatch)

    _, parts = step(trainer, "image")

    video, audio, text, _timestep, layout = calls.args[0]
    assert set(parts) == {"sigma", "video"}
    assert video.shape == (1, 1, 96) and audio.shape == (1, 4, 32) and text.shape == (1, 4, 6)
    assert layout["audio_indices"].numel() == 4 and layout["text_indices"].numel() == 4


def test_video_and_audio_are_conditioned_on_one_minus_their_coupled_sigmas(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = make_trainer(workspace)
    calls = Calls(trainer, monkeypatch)

    _, parts = step(trainer, "image")

    sigma = torch.tensor([parts["sigma"]])
    timestep = calls.args[0][3]
    torch.testing.assert_close(timestep, torch.cat([1.0 - sigma, 1.0 - h3.remap_sigma(sigma)]))
    torch.manual_seed(4)
    expected = h3.apply_flow_shift(torch.sigmoid(torch.randn(1)).clamp(0.001, 0.999), SHIFT)
    assert parts["sigma"] == pytest.approx(expected.item())


def test_pack_silent_audio_false_drops_the_audio_block_and_its_timestep_row(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = make_trainer(workspace, pack_silent_audio=False)
    calls = Calls(trainer, monkeypatch)

    step(trainer, "image")

    _, audio, _, timestep, layout = calls.args[0]
    assert audio.shape == (1, 0, 32) and timestep.shape == (1,) and layout["audio_indices"].numel() == 0


def test_the_video_loss_is_the_mse_to_the_clean_latent_minus_the_noise(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = make_trainer(workspace, pack_silent_audio=False)
    calls = Calls(trainer, monkeypatch)
    loss, parts = step(trainer, "image")
    latents = sample_of(trainer, "image").video
    torch.manual_seed(4)
    sigma = h3.apply_flow_shift(torch.sigmoid(torch.randn(1)).clamp(0.001, 0.999), SHIFT)
    noise = torch.randn_like(latents)
    noisy = (1.0 - sigma.view(1, 1, 1, 1, 1)) * latents + sigma.view(1, 1, 1, 1, 1) * noise

    video, audio, text, timestep, layout = calls.args[0]
    torch.testing.assert_close(video, h3.patchify_video(noisy))
    with torch.no_grad():
        output = trainer._forward(video, audio, text, timestep, layout)
    expected = torch.nn.functional.mse_loss(output.video, h3.patchify_video(latents - noise))
    torch.testing.assert_close(loss, expected)
    assert parts["video"] == pytest.approx(expected.item())


def test_a_clip_with_sound_adds_the_weighted_audio_loss_to_the_video_loss(workspace: Workspace) -> None:
    trainer = make_trainer(workspace, audio_loss_weight=0.5)

    loss, parts = step(trainer, "clip")

    assert set(parts) == {"sigma", "video", "audio"}
    assert loss.item() == pytest.approx(parts["video"] + 0.5 * parts["audio"], rel=1e-5)


def test_a_recording_trains_audio_only_and_builds_no_video_loss(workspace: Workspace) -> None:
    trainer = make_trainer(workspace, audio_loss_weight=2.0)

    loss, parts = step(trainer, "audio")

    assert set(parts) == {"sigma", "audio"} and loss.item() == pytest.approx(2.0 * parts["audio"], rel=1e-5)


def test_loss_weight_strength_scales_the_whole_loss(workspace: Workspace) -> None:
    plain, _ = step(make_trainer(workspace), "clip")

    strong, _ = step(make_trainer(workspace, loss_weight_strength=3.0), "clip")

    assert strong.item() == pytest.approx(3.0 * plain.item(), rel=1e-5)


def test_with_train_audio_off_a_clip_keeps_its_silent_rows_and_loses_the_audio_loss(workspace: Workspace) -> None:
    trainer = make_trainer(workspace)
    trainer.config["train"]["train_audio"] = False

    _, parts = step(trainer, "clip")

    assert set(parts) == {"sigma", "video"}


def test_a_recording_with_train_audio_off_has_no_loss_term_at_all(workspace: Workspace) -> None:
    trainer = make_trainer(workspace)
    trainer.config["train"]["train_audio"] = False

    with pytest.raises(RuntimeError, match="no loss term for this sample"):
        step(trainer, "audio")


def test_caption_dropout_swaps_in_the_empty_prompt_embedding(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = make_trainer(workspace, caption_dropout=1.0)
    calls = Calls(trainer, monkeypatch)

    step(trainer, "image")

    assert calls.args[0][2].shape == (1, 1, 6) and torch.equal(calls.args[0][2], trainer.text_cache[""])


def test_the_step_is_deterministic_for_a_seed_and_the_seed_changes_the_noise(workspace: Workspace) -> None:
    trainer = make_trainer(workspace)

    first, _ = step(trainer, "clip", seed=1)
    again, _ = step(trainer, "clip", seed=1)
    other, _ = step(trainer, "clip", seed=2)

    assert first.item() == again.item() and first.item() != other.item()
