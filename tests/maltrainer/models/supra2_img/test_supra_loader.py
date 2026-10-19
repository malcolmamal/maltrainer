from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from maltrainer.models.supra2_img import model as model_module
from maltrainer.models.supra2_img.model import SupraDiT, load_supra2_img

CPU = torch.device("cpu")


def _tiny() -> SupraDiT:
    return SupraDiT(latent_ch=4, d_model=32, depth=1, n_heads=4, ctx_dim=16, num_tokens=16)


@pytest.fixture(autouse=True)
def tiny_architecture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_module, "SupraDiT", _tiny)


def _checkpoint(tmp_path: Path, extra: dict[str, torch.Tensor] | None = None, metadata: dict | None = None) -> Path:
    torch.manual_seed(0)
    tensors = {**_tiny().state_dict(), **(extra or {})}
    path = tmp_path / "supra.safetensors"
    save_file({key: value.contiguous() for key, value in tensors.items()}, str(path), metadata=metadata)
    return path


def test_the_published_weights_load_strictly_into_the_model(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path)
    torch.manual_seed(0)
    saved = _tiny().state_dict()

    model, sidecar = load_supra2_img(path, CPU, torch.float32)

    assert sidecar == {}
    assert sorted(model.state_dict()) == sorted(saved)
    assert all(torch.equal(value, saved[key]) for key, value in model.state_dict().items())


def test_the_unconditional_embedding_and_metadata_are_returned_as_a_sidecar(tmp_path: Path) -> None:
    uncond = {"__uncond_text__": torch.randn(1, 128, 16), "__uncond_mask__": torch.ones(1, 128)}
    path = _checkpoint(tmp_path, uncond, {"ctx_len": "128", "name": "supra", "ratio": "1.5"})

    model, sidecar = load_supra2_img(path, CPU, torch.float32)

    assert sidecar["ctx_len"] == 128 and sidecar["name"] == "supra" and sidecar["ratio"] == "1.5"
    assert torch.equal(sidecar["uncond_text"], uncond["__uncond_text__"])
    assert torch.equal(sidecar["uncond_mask"], uncond["__uncond_mask__"])
    assert not any(key.startswith("__") for key in model.state_dict())


def test_a_sidecar_with_only_the_text_embedding_has_no_mask(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path, {"__uncond_text__": torch.zeros(1, 4, 16)})

    _, sidecar = load_supra2_img(path, CPU, torch.float32)

    assert sorted(sidecar) == ["uncond_text"]


def test_a_frozen_model_is_cast_and_set_to_evaluation(tmp_path: Path) -> None:
    model, _ = load_supra2_img(_checkpoint(tmp_path), CPU, torch.bfloat16)

    assert {p.dtype for p in model.parameters()} == {torch.bfloat16}
    assert not any(p.requires_grad for p in model.parameters())
    assert not model.training


def test_a_finetune_load_leaves_the_weights_trainable(tmp_path: Path) -> None:
    model, _ = load_supra2_img(_checkpoint(tmp_path), CPU, torch.float32, freeze=False)

    assert all(p.requires_grad for p in model.parameters())
    assert model.training


@pytest.mark.parametrize("damage", ["extra", "missing"])
def test_a_checkpoint_that_does_not_match_the_architecture_is_rejected(tmp_path: Path, damage: str) -> None:
    tensors = _tiny().state_dict()
    if damage == "extra":
        tensors["rogue.weight"] = torch.zeros(1)
    else:
        del tensors["pos_embed"]
    path = tmp_path / "bad.safetensors"
    save_file(tensors, str(path))

    with pytest.raises(RuntimeError, match="Error\\(s\\) in loading state_dict"):
        load_supra2_img(path, CPU, torch.float32)
