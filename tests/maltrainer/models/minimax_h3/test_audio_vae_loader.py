from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from maltrainer.models.minimax_h3 import audio_vae
from tests.maltrainer.models.minimax_h3.h3_audio import tiny_audio_encoder

CPU = torch.device("cpu")


def _state(monkeypatch: pytest.MonkeyPatch) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in tiny_audio_encoder(monkeypatch, seed=5).state_dict().items()}


def _save(tmp_path: Path, state: dict[str, torch.Tensor]) -> Path:
    path = tmp_path / "audio.safetensors"
    save_file(state, str(path))
    return path


def _weight_normed(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in state.items():
        if key.endswith(".weight") and value.ndim == 3:
            stem = key[: -len("weight")]
            out[stem + "weight_g"] = value.flatten(1).norm(dim=1).view(-1, 1, 1)
            out[stem + "weight_v"] = value * 3.0
        else:
            out[key] = value
    return out


def test_weight_norm_pairs_fold_into_g_times_v_over_the_norm_of_v() -> None:
    v = torch.tensor([[[3.0], [4.0]]])
    state = {"a.weight_g": torch.tensor([[[10.0]]]), "a.weight_v": v, "a.bias": torch.ones(1)}

    folded = audio_vae._fold_weight_norm(state)

    assert set(folded) == {"a.weight", "a.bias"}
    torch.testing.assert_close(folded["a.weight"], torch.tensor([[[6.0], [8.0]]]))


def test_a_comfy_single_file_loads_by_name_with_its_own_statistics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = _state(monkeypatch)
    state["latents_mean"] = torch.full((32,), 0.25)
    path = _save(tmp_path, {**state, "decoder.block.0.weight": torch.zeros(1)})

    model = audio_vae.load_minimax_h3_audio_encoder(path, CPU)

    torch.testing.assert_close(model.mean_proj.weight, state["mean_proj.weight"])
    assert model.latents_mean.tolist() == [0.25] * 32
    assert (model.training, any(p.requires_grad for p in model.parameters())) == (False, False)


def test_the_original_release_with_weight_norm_loads_and_falls_back_to_the_config_statistics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    state = {k: v for k, v in _state(monkeypatch).items() if not k.startswith("latents")}
    path = _save(tmp_path, _weight_normed(state))

    with caplog.at_level("INFO", logger=audio_vae.logger.name):
        model = audio_vae.load_minimax_h3_audio_encoder(path, CPU)

    torch.testing.assert_close(model.mean_proj.weight, state["mean_proj.weight"], atol=1e-5, rtol=1e-5)
    assert model.latents_mean.tolist() == pytest.approx(list(audio_vae.DEFAULT_LATENTS_MEAN))
    assert "carries no latent statistics" in caplog.text


def test_half_precision_files_are_loaded_as_float32(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    state = {k: v.half() for k, v in _state(monkeypatch).items()}

    model = audio_vae.load_minimax_h3_audio_encoder(_save(tmp_path, state), CPU)

    assert {p.dtype for p in model.parameters()} == {torch.float32}


def test_a_file_missing_encoder_tensors_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    state = _state(monkeypatch)
    del state["mean_proj.bias"]

    with pytest.raises(ValueError, match=r"audio.safetensors did not supply 1 .*mean_proj.bias.*audio VAE"):
        audio_vae.load_minimax_h3_audio_encoder(_save(tmp_path, state), CPU)


def test_a_file_with_unknown_encoder_tensors_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    state = {**_state(monkeypatch), "encoder.extra.weight": torch.zeros(1)}

    with pytest.raises(ValueError, match=r"has 1 encoder tensor.*extra.weight"):
        audio_vae.load_minimax_h3_audio_encoder(_save(tmp_path, state), CPU)
