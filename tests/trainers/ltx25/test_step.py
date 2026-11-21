"""One training step: rectified-flow noising and the velocity loss of both streams."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from maltrainer.trainers.ltx25 import step
from maltrainer.trainers.ltx25.step import flow_inputs, ltx_loss

CPU = torch.device("cpu")


class Oracle:
    """A network that predicts the exact velocity of whatever it is shown (so the loss is zero), plus a fixed error."""

    def __init__(
        self,
        video_error: float = 0.0,
        audio_error: float = 0.0,
        clean_video: torch.Tensor | None = None,
        clean_audio: torch.Tensor | None = None,
    ):
        self.errors = (video_error, audio_error)
        self.clean = (clean_video, clean_audio)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        video: torch.Tensor,
        sigma: torch.Tensor,
        features: torch.Tensor,
        fps: float,
        audio: torch.Tensor | None,
        audio_sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self.calls.append(
            {
                "video": video,
                "sigma": sigma,
                "features": features,
                "fps": fps,
                "audio": audio,
                "audio_sigma": audio_sigma,
            }
        )
        # x_t = (1 - s) x0 + s noise  =>  v = noise - x0 = (x_t - x0) / s
        pred_video = (video - self.clean[0]) / sigma.view(1, 1, 1, 1, 1) + self.errors[0]
        if audio is None:
            return pred_video, None
        return pred_video, (audio - self.clean[1]) / sigma.view(1, 1, 1, 1) + self.errors[1]


def _latents() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(2)
    return (
        torch.randn(1, 128, 2, 2, 3, generator=generator),
        torch.randn(1, 8, 5, 16, generator=generator),
        torch.randn(1, 4, 6, generator=generator),
    )


def test_the_noisy_latent_interpolates_between_the_clean_one_and_noise_and_the_target_is_the_velocity() -> None:
    clean = torch.randn(1, 4, 2, 2, 2)
    torch.manual_seed(0)
    noisy, target = flow_inputs(clean, torch.tensor(0.25).view(1, 1, 1, 1, 1))

    noise = target + clean
    assert torch.allclose(noisy, 0.75 * clean + 0.25 * noise, atol=1e-6)
    assert noise.std().item() == pytest.approx(1.0, abs=0.4)


def test_a_network_that_knows_the_velocity_has_zero_loss_and_sees_the_noised_inputs() -> None:
    video, audio, features = _latents()
    network = Oracle(clean_video=video, clean_audio=audio)

    loss = ltx_loss(network, {}, video, features, audio, 24.0, torch.float32)

    assert loss.item() == pytest.approx(0.0, abs=1e-5)
    call = network.calls[0]
    assert call["video"].shape == video.shape and call["audio"].shape == audio.shape and call["fps"] == 24.0
    assert torch.equal(call["sigma"], call["audio_sigma"]) and not torch.allclose(call["video"], video)


def test_a_constant_error_on_each_stream_is_squared_and_weighted() -> None:
    video, audio, features = _latents()
    train = {"video_loss_weight": 2.0, "audio_loss_weight": 0.5}

    loss = ltx_loss(Oracle(1.0, 3.0, video, audio), train, video, features, audio, 24.0, torch.float32)

    assert loss.item() == pytest.approx(2.0 * 1.0 + 0.5 * 9.0, abs=1e-4)


def test_without_audio_only_the_video_is_scored_and_the_audio_never_reaches_the_network() -> None:
    video, _, features = _latents()
    network = Oracle(2.0, 100.0, video, None)

    loss = ltx_loss(network, {"audio_loss_weight": 5.0}, video, features, None, 24.0, torch.float32)

    assert loss.item() == pytest.approx(4.0, abs=1e-4)
    assert network.calls[0]["audio"] is None


def test_the_network_receives_the_compute_dtype_and_the_loss_is_fp32() -> None:
    video, audio, features = _latents()
    network = Oracle(clean_video=video, clean_audio=audio)

    loss = ltx_loss(network, {}, video, features, audio, 24.0, torch.bfloat16)

    assert network.calls[0]["video"].dtype == torch.bfloat16 and network.calls[0]["audio"].dtype == torch.bfloat16
    assert loss.dtype == torch.float32


def test_sigma_is_drawn_for_the_number_of_video_latent_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int] = []

    def spy(_config: dict[str, Any], tokens: int, _device: torch.device) -> torch.Tensor:
        seen.append(tokens)
        return torch.tensor([0.5])

    monkeypatch.setattr(step, "sample_sigma", spy)
    video, _, features = _latents()

    ltx_loss(Oracle(clean_video=video), {}, video, features, None, 24.0, torch.float32)

    assert seen == [2 * 2 * 3]
