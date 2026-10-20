"""Prior preservation: recorded from the original weights before training, then a loss term on trigger images."""

from __future__ import annotations

import re
from typing import Any

import pytest
import torch

import train_moss_image_finetune as finetune
from maltrainer.models.moss_image.loading import load_moss_image_transformer

from .fakes import CPU
from .runs import base_prediction, record_sample_order, same_values
from .world import Workspace, base_weights, named_weights

PRIOR = {"weight": 1.0, "classes": {"zed": "person"}, "samples_per_image": 2}


def _trainer(ws: Workspace, **sections: Any) -> finetune.MOSSImageFinetuneTrainer:
    ws.add_images("p", 2, caption="a photo of zed")
    ws.add_images("q", 2, caption="a photo of someone")
    return finetune.MOSSImageFinetuneTrainer(ws.config(prior=PRIOR, **sections))


def _step_log(ws: Workspace) -> list[str]:
    return [line for line in (ws.out / "training.log").read_text(encoding="utf-8").splitlines() if "Step:" in line]


def _logged(label: str, ws: Workspace) -> float:
    found = re.search(rf"{label}: ([\d.]+)", _step_log(ws)[-1])
    assert found is not None
    return float(found.group(1))


def test_the_original_models_predictions_are_recorded_before_the_first_step(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = _trainer(workspace)
    assert trainer.prior is not None
    seen: dict[str, Any] = {}
    build = trainer.prior.build

    def spy(*args: Any, **kwargs: Any) -> None:
        seen["weights"], seen["step"] = named_weights(trainer.transformer), trainer.global_step
        build(*args, **kwargs)

    monkeypatch.setattr(trainer.prior, "build", spy)
    trainer.run()

    assert seen["step"] == 0
    assert same_values(seen["weights"], base_weights(workspace))
    assert not same_values(named_weights(trainer.transformer), base_weights(workspace))


def test_each_recorded_target_is_the_unchanged_models_answer_to_its_noised_input(workspace: Workspace) -> None:
    trainer = _trainer(workspace)
    trainer.run()
    prior, base = trainer.prior, load_moss_image_transformer(workspace.layout, CPU, torch.bfloat16)[0]
    assert prior is not None

    assert sorted(prior.captions) == ["p/img0.png", "p/img1.png"]
    for rel in prior.captions:
        assert prior.captions[rel] == "a photo of person"
        x0 = trainer.latent_cache[str(workspace.data / rel)]
        context = trainer.text_cache[prior.captions[rel]]
        for i in range(prior.samples_per_image):
            expected = base_prediction(base, x0, prior.sigmas[rel][i], prior.seeds[rel][i], context)
            assert torch.equal(prior.targets[rel][i], expected.to(torch.bfloat16))


def test_only_images_whose_caption_names_a_trigger_get_the_prior_term(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    order = record_sample_order(monkeypatch)
    trainer = _trainer(workspace, train={"max_steps": 8})

    trainer.run()

    with_prior = ["Prior:" in line for line in _step_log(workspace)]
    assert [("/p/" in trainer.samples[i][0].replace("\\", "/")) for i in order] == with_prior
    assert any(with_prior) and not all(with_prior)


@pytest.mark.parametrize(("every", "expected"), [(1, [True] * 4), (2, [True, False, True, False])])
def test_the_prior_term_runs_every_nth_step(workspace: Workspace, every: int, expected: list[bool]) -> None:
    workspace.add_images("p", 1, caption="a photo of zed")
    config = workspace.config(prior={**PRIOR, "every": every}, train={"max_steps": 4})

    finetune.MOSSImageFinetuneTrainer(config).run()

    assert ["Prior:" in line for line in _step_log(workspace)] == expected


def test_the_prior_term_is_added_to_the_loss_with_its_weight(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_images("p", 1, caption="a photo of zed")
    totals: list[float] = []
    real_backward = torch.Tensor.backward

    def spy(self: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        totals.append(self.item())
        real_backward(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "backward", spy)
    trainer = finetune.MOSSImageFinetuneTrainer(
        workspace.config(prior={**PRIOR, "weight": 3.0}, train={"max_steps": 2, "monitor_every": 1})
    )

    trainer.run()

    prior_value, plain = _logged("Prior", workspace), _logged("Loss", workspace)
    assert totals[-1] == pytest.approx(plain + 3.0 * prior_value, abs=2e-5)


def test_the_record_is_saved_beside_the_model_and_into_the_resume_state(workspace: Workspace) -> None:
    trainer = _trainer(workspace)

    trainer.run()

    assert trainer.prior_path == workspace.out / "ft_prior.safetensors"
    assert (workspace.out / "ft_state" / "prior.safetensors").stat().st_size == trainer.prior_path.stat().st_size
    assert trainer.run_metadata["prior_images"] == 2
    assert trainer.run_metadata["prior_classes"] == {"zed": "person"}
