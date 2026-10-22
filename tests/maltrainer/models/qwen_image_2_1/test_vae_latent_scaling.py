from __future__ import annotations

import pytest
import torch

from maltrainer.models.qwen_image_2_1.vae import LATENTS_MEAN, LATENTS_STD

from ._vae_helpers import stub_vae


def test_there_is_one_mean_and_one_positive_std_for_each_of_the_64_latent_channels() -> None:
    assert len(LATENTS_MEAN) == len(LATENTS_STD) == 64
    assert min(LATENTS_STD) > 2.5


@pytest.mark.parametrize(
    ("index", "mean", "std"),
    [(0, 0.5126, 3.2001), (15, 3.8363, 5.0756), (52, -5.0179, 5.5614), (63, 0.0304, 3.8161)],
)
def test_the_normalisation_constants_match_the_published_values(index: int, mean: float, std: float) -> None:
    assert LATENTS_MEAN[index] == mean
    assert LATENTS_STD[index] == std


def test_the_constants_are_kept_as_non_persistent_channel_first_buffers(monkeypatch: pytest.MonkeyPatch) -> None:
    vae = stub_vae(monkeypatch)

    assert vae.latents_mean.shape == vae.latents_std.shape == (1, 64, 1, 1)
    assert vae.latents_mean.flatten().tolist() == pytest.approx(LATENTS_MEAN)
    assert not {"latents_mean", "latents_std"} & set(vae.state_dict())


def test_normalising_subtracts_the_channel_mean_and_divides_by_the_channel_std(monkeypatch: pytest.MonkeyPatch) -> None:
    vae = stub_vae(monkeypatch)
    latents = torch.zeros(1, 64, 1, 1)
    latents[0, 1] = 0.7721 + 3.2936

    normalised = vae.normalize_latents(latents)

    assert normalised[0, 1].item() == pytest.approx(1.0, abs=1e-6)
    assert normalised[0, 0].item() == pytest.approx(-0.5126 / 3.2001)


def test_a_latent_equal_to_the_mean_normalises_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    vae = stub_vae(monkeypatch)

    normalised = vae.normalize_latents(vae.latents_mean.expand(2, 64, 3, 3))

    torch.testing.assert_close(normalised, torch.zeros(2, 64, 3, 3))


def test_unnormalising_undoes_normalising(monkeypatch: pytest.MonkeyPatch) -> None:
    vae = stub_vae(monkeypatch)
    latents = torch.randn(2, 64, 3, 2, generator=torch.Generator().manual_seed(0))

    restored = vae.unnormalize_latents(vae.normalize_latents(latents))

    torch.testing.assert_close(restored, latents, atol=1e-5, rtol=1e-5)


def test_the_scaling_follows_the_latent_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    vae = stub_vae(monkeypatch)

    out = vae.normalize_latents(torch.zeros(1, 64, 1, 1, dtype=torch.bfloat16))

    assert out.dtype == torch.bfloat16
    assert vae.unnormalize_latents(out).dtype == torch.bfloat16
