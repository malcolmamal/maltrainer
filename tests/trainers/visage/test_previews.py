"""The preview keys of a run: w / m / g render prompts from the live LoRA without touching the training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image

from maltrainer.trainers.visage import previews
from maltrainer.utils.keyboard import KeyCommands
from train_visage_lora import VisageLoRATrainer

from .fakes import Fakes
from .runs import prepared
from .world import Workspace, named_weights

SAMPLE = {"keys": {"w": "photo of a woman", "m": "photo of a man"}, "steps": 2, "guidance": 3.0, "resolution": 32}


class FakeVAE:
    """Latents [1, 8, h, w] -> pixels [1, 3, 16 h, 16 w] in [0, 1] from their mean (so images differ)."""

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        value = torch.sigmoid(latents.mean(dim=1, keepdim=True)).repeat(1, 3, 1, 1)
        return value.repeat_interleave(16, dim=2).repeat_interleave(16, dim=3)


@pytest.fixture(autouse=True)
def decoder(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    loads: list[str] = []

    def load(path: str, _device: torch.device) -> FakeVAE:
        loads.append(path)
        return FakeVAE()

    monkeypatch.setattr(previews, "load_visage_vae", load)
    return loads


def _trainer(ws: Workspace, sample: dict[str, Any] | None = None, **sections: Any) -> VisageLoRATrainer:
    ws.add_images(1)
    return prepared(ws.config(sample=SAMPLE if sample is None else sample, **sections))


def _size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return (image.width, image.height)


def test_the_configured_keys_get_their_prompts_and_g_renders_them_all(workspace: Workspace) -> None:
    keys = _trainer(workspace).preview_keys()

    assert set(keys) == {"w", "m", "g"}
    assert keys["w"][0] == 'render "photo of a woman"' and keys["g"][0] == "render every preview prompt (2)"


def test_without_a_sample_section_there_are_no_preview_keys_but_the_standard_ones_remain(workspace: Workspace) -> None:
    workspace.add_images(1)
    trainer = prepared(workspace.config())

    keys = trainer._make_keys()

    assert trainer.preview_keys() == {} and trainer.preview_texts() == set()
    assert isinstance(keys, KeyCommands) and keys.keys == set("sqph")


def test_the_run_offers_the_keys_in_its_key_commands(workspace: Workspace) -> None:
    keys = _trainer(workspace)._make_keys()

    assert keys.keys == set("sqphwmg") and keys.actions.keys() == {"w", "m", "g"}


@pytest.mark.parametrize("key", ["s", "q", "p", "e", "g", "h", "ww", ""])
def test_a_preview_key_that_is_a_command_key_or_not_one_character_is_refused(workspace: Workspace, key: str) -> None:
    workspace.add_images(1)

    with pytest.raises(ValueError, match=r"sample\.keys must be single characters"):
        prepared(workspace.config(sample={"keys": {key: "photo"}}))


def test_the_prompts_are_encoded_while_the_text_encoder_is_loaded(workspace: Workspace, fakes: Fakes) -> None:
    trainer = _trainer(workspace)

    assert {"photo of a woman", "photo of a man"} <= set(trainer.text_cache)
    assert {"photo of a woman", "photo of a man"} <= set(fakes.text.prompts)
    assert "" not in trainer.text_cache  # the unconditional branch is zeros, not the empty prompt


def test_a_press_renders_one_image_named_by_step_and_key_at_the_requested_size(workspace: Workspace) -> None:
    trainer = _trainer(workspace)
    trainer.current_step = 12

    trainer.preview_keys()["w"][1]()

    files = sorted(p.name for p in (workspace.out / "samples").iterdir())
    assert files == ["step000012_w.png"]
    assert _size(workspace.out / "samples" / files[0]) == (32, 32)


def test_the_size_defaults_to_the_dataset_resolution(workspace: Workspace) -> None:
    trainer = _trainer(workspace, {**SAMPLE, "resolution": None}, dataset={"resolution": 48})
    del trainer.config["sample"]["resolution"]

    trainer.preview_keys()["w"][1]()

    assert _size(workspace.out / "samples" / "step000000_w.png") == (48, 48)


def test_g_renders_every_prompt_and_loads_the_decoder_once(workspace: Workspace, decoder: list[str]) -> None:
    trainer = _trainer(workspace)

    trainer.preview_keys()["g"][1]()

    assert sorted(p.name for p in (workspace.out / "samples").iterdir()) == ["step000000_m.png", "step000000_w.png"]
    assert decoder == [str(workspace.root / "vae.safetensors")]


def test_the_same_seed_gives_the_same_image_and_the_prompts_differ(workspace: Workspace) -> None:
    trainer = _trainer(workspace)
    folder = workspace.out / "samples"

    trainer.preview_keys()["g"][1]()
    first = {p.name: p.read_bytes() for p in folder.iterdir()}
    trainer.current_step = 1
    trainer.preview_keys()["g"][1]()

    assert (folder / "step000001_w.png").read_bytes() == first["step000000_w.png"]
    assert first["step000000_w.png"] != first["step000000_m.png"]


def test_the_press_leaves_the_training_untouched(workspace: Workspace) -> None:
    trainer = _trainer(workspace)
    trainer.lora_network.train()
    weights = named_weights(trainer.lora_network)
    torch.manual_seed(3)
    state = torch.get_rng_state()

    trainer.preview_keys()["g"][1]()

    assert trainer.lora_network.training and torch.equal(torch.get_rng_state(), state)
    assert all(torch.equal(weights[k], v) for k, v in named_weights(trainer.lora_network).items())
    assert all(p.grad is None for p in trainer.lora_network.parameters())


def test_a_lora_that_was_trained_changes_the_preview(workspace: Workspace) -> None:
    trainer = _trainer(workspace)
    trainer.preview_keys()["w"][1]()
    before = (workspace.out / "samples" / "step000000_w.png").read_bytes()
    for layer in trainer.lora_network.lora_layers.values():
        layer.lora_up.weight.data.normal_(std=0.5)

    trainer.current_step = 1
    trainer.preview_keys()["w"][1]()

    assert (workspace.out / "samples" / "step000001_w.png").read_bytes() != before


def test_with_ema_the_preview_uses_the_averaged_weights_unless_switched_off(workspace: Workspace) -> None:
    workspace.add_images(1)
    config = workspace.config(sample=SAMPLE, train={"ema_decay": 0.5, "ema_warmup": False})
    trainer = prepared(config)
    trainer._start_ema(trainer.lora_network, trainer.config["train"])  # what train() does before the first step
    assert trainer.ema is not None
    for layer in trainer.lora_network.lora_layers.values():
        layer.lora_up.weight.data.normal_(std=0.5)
    trainer.ema.update()
    trainer.ema.update()

    trainer.preview_keys()["w"][1]()
    trainer.current_step = 1
    trainer.config["sample"]["ema"] = False
    trainer.preview_keys()["w"][1]()

    folder = workspace.out / "samples"
    assert (folder / "step000000_w.png").read_bytes() != (folder / "step000001_w.png").read_bytes()


def test_only_the_main_process_renders(workspace: Workspace) -> None:
    trainer = _trainer(workspace)
    trainer.is_main_process = False

    trainer.preview_keys()["g"][1]()

    assert not (workspace.out / "samples").exists()
