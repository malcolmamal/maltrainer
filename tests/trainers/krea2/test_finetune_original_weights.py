"""The original model in place under the window rotator (maltrainer/finetune/rotation_base.py): what a resumed run
with new images records their prior from. Inside, the model is the freshly loaded base; after, the trained one,
bit for bit."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from maltrainer.finetune.rotation import WindowRotator
from maltrainer.finetune.rotation_base import original_weights
from maltrainer.models.comfy_fp8 import quantize_rows
from maltrainer.models.krea2.finetune import DEFAULT_WINDOWS, always_on_linears, dequantize_always_on
from maltrainer.models.krea2.loading import load_krea2_transformer
from maltrainer.models.krea2.model import Krea2Transformer

from .tiny import CONFIG
from .world import PREFIX, Workspace, write_checkpoint

CPU = torch.device("cpu")


def _loaded(path: Path) -> tuple[Krea2Transformer, WindowRotator, list[tuple[str, torch.nn.Parameter]]]:
    master: dict[str, torch.Tensor] = {}
    model, _ = load_krea2_transformer(path, CPU, torch.bfloat16, master=master)
    rotator = WindowRotator(
        model, master, DEFAULT_WINDOWS, CPU, torch.bfloat16, lambda params, _name: torch.optim.SGD(params, lr=0.1)
    )
    dequantize_always_on(model, ["txtfusion"], master)
    return model, rotator, always_on_linears(model, ["txtfusion"])


def _trained(path: Path) -> tuple[Krea2Transformer, WindowRotator, list[tuple[str, torch.nn.Parameter]]]:
    """The base after a rotation through two windows and a change to the text fusion, as training leaves it."""
    model, rotator, always = _loaded(path)
    with torch.no_grad():
        for window in (0, 2):
            rotator.activate(window)
            for group in rotator.optimizer.param_groups:
                for param in group["params"]:
                    param.add_(0.05)
        rotator.deactivate()
        for _, param in always:
            param.mul_(1.5)
    return model, rotator, always


def _predict(model: Krea2Transformer) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(1, CONFIG.channels, 4, 4, generator=generator)
    context = torch.randn(1, 3, CONFIG.txtlayers, CONFIG.txtdim, generator=generator)
    with torch.no_grad():
        return model(x, torch.tensor([0.5]), context)


def _state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def test_inside_the_model_is_the_freshly_loaded_base_and_after_it_is_the_trained_one_bit_for_bit(
    workspace: Workspace,
) -> None:
    base, _, _ = _loaded(workspace.checkpoint)
    model, rotator, always = _trained(workspace.checkpoint)
    trained = _state(model)
    assert not torch.equal(_predict(model), _predict(base))

    with original_weights(rotator, always, workspace.checkpoint):
        inside = _predict(model)

    assert torch.equal(inside, _predict(base))
    after = _state(model)
    assert after.keys() == trained.keys()
    assert all(torch.equal(after[name], trained[name]) for name in trained)


def test_the_trained_weights_come_back_when_the_recording_fails(workspace: Workspace) -> None:
    model, rotator, always = _trained(workspace.checkpoint)
    trained = _state(model)

    with (
        pytest.raises(RuntimeError, match="out of memory"),
        original_weights(rotator, always, workspace.checkpoint),
    ):
        raise RuntimeError("out of memory")

    assert all(torch.equal(tensor, trained[name]) for name, tensor in model.state_dict().items())


def test_an_fp8_base_file_is_dequantized_and_quantized_again_per_row(workspace: Workspace, tmp_path: Path) -> None:
    fp8 = write_checkpoint(tmp_path / "fp8.safetensors", fp8=True)
    model, rotator, always = _trained(workspace.checkpoint)
    key = next(iter(rotator.master))
    module = model.get_submodule(key[: -len(".weight")])
    fp8_model, _, _ = _loaded(fp8)
    stored = fp8_model.get_submodule(key[: -len(".weight")])
    dequantized = (stored.weight.float() * stored.weight_scale).to(torch.bfloat16)

    with original_weights(rotator, always, fp8):
        expected_weight, expected_scale = quantize_rows(dequantized)
        assert torch.equal(module.weight.view(torch.uint8), expected_weight.view(torch.uint8))
        assert torch.equal(module.weight_scale, expected_scale)


def test_an_active_window_is_refused(workspace: Workspace) -> None:
    _, rotator, always = _trained(workspace.checkpoint)
    rotator.activate(1)

    with (
        pytest.raises(RuntimeError, match="every window frozen"),
        original_weights(rotator, always, workspace.checkpoint),
    ):
        pass


def test_a_tensor_missing_from_the_base_file_is_named_and_the_trained_weights_come_back(
    workspace: Workspace, tmp_path: Path
) -> None:
    tensors = load_file(str(workspace.checkpoint))
    del tensors[PREFIX + "blocks.1.mlp.up.weight"]
    short = tmp_path / "short.safetensors"
    save_file(tensors, str(short))
    model, rotator, always = _trained(workspace.checkpoint)
    trained = _state(model)

    with (
        pytest.raises(KeyError, match=r"short\.safetensors has no tensor for blocks\.1\.mlp\.up\.weight"),
        original_weights(rotator, always, short),
    ):
        pass

    assert all(torch.equal(tensor, trained[name]) for name, tensor in model.state_dict().items())


def test_a_base_with_its_text_fusion_in_fp8_puts_the_dequantized_values_in_and_the_trained_ones_back(
    workspace: Workspace, tmp_path: Path
) -> None:
    fp8 = write_checkpoint(tmp_path / "fp8.safetensors", fp8=True, fp8_text=True)
    stored = load_file(str(fp8))
    _, rotator, always = _trained(workspace.checkpoint)
    trained = {name: param.detach().clone() for name, param in always}

    with original_weights(rotator, always, fp8):
        for name, param in always:
            weight, scale = stored[PREFIX + name], stored.get(PREFIX + name[: -len(".weight")] + ".weight_scale")
            expected = weight.to(torch.bfloat16) if scale is None else (weight.float() * scale).to(torch.bfloat16)
            assert torch.equal(param.detach(), expected), name
        assert any(PREFIX + name[: -len(".weight")] + ".weight_scale" in stored for name, _ in always)

    assert all(torch.equal(param.detach(), trained[name]) for name, param in always)
