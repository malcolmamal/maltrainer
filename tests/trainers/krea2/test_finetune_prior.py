"""Prior preservation in the Krea 2 finetune: the original model's predictions recorded before the first step, the
term on the log line, kept in the resume state, and recorded from the original weights for images added on resume."""

from __future__ import annotations

import logging
from typing import Any

import pytest
import torch
from safetensors.torch import load_file

from maltrainer.finetune.prior import PriorPreservation, _noise
from maltrainer.models.krea2.loading import load_krea2_transformer
from maltrainer.trainers.krea2.finetune_host import STATE_PRIOR

from .fakes import Fakes
from .runs import final_weights, progress_lines, run_finetune
from .world import Workspace, write_image

PRIOR = {"weight": 1.0, "class_word": "person", "samples_per_image": 2}
CPU = torch.device("cpu")


def _config(ws: Workspace, **sections: Any) -> dict[str, Any]:
    return ws.finetune_config(**{"prior": PRIOR, **sections})


def test_the_prior_is_recorded_before_the_first_step_and_kept_beside_the_model_and_in_the_state(
    workspace: Workspace, fakes: Fakes
) -> None:
    workspace.add_images(3)

    trainer = run_finetune(_config(workspace))

    assert trainer.prior is not None and sorted(trainer.prior.keys()) == ["img0.png", "img1.png", "img2.png"]
    assert set(trainer.prior.captions.values()) == {"a photo of person"}
    assert trainer.prior.targets["img0.png"].shape == (2, 4, 4, 4)
    assert "a photo of person" in fakes.text.prompts
    saved = workspace.out / "ft_prior.safetensors"
    assert saved.samefile(workspace.out / "ft_state" / STATE_PRIOR)
    assert torch.equal(load_file(str(saved))["0.target"], trainer.prior.targets["img0.png"])
    assert trainer.run_metadata["prior_images"] == 3 and trainer.run_metadata["prior_classes"] == {"zed": "person"}


def test_the_prior_terms_mean_is_on_the_log_line_of_the_steps_it_is_due(workspace: Workspace) -> None:
    workspace.add_images(2)

    run_finetune(_config(workspace, prior={**PRIOR, "every": 2}))

    lines = progress_lines(workspace)
    assert ["Prior: " in line for line in lines] == [True, False, True]  # steps 0 and 2 of 0, 1, 2


def test_the_prior_term_reaches_the_weights_in_the_fused_backward(workspace: Workspace) -> None:
    # the same prior draws (so the same epoch orders) at two weights: only the term in the backward differs
    workspace.add_images(2)

    for name, weight in (("faint", 1e-6), ("strong", 100.0)):
        run_finetune(_config(workspace, prior={**PRIOR, "weight": weight}, output={"path": str(workspace.root / name)}))

    faint, strong = final_weights(workspace.root / "faint"), final_weights(workspace.root / "strong")
    assert any(not torch.equal(faint[key], strong[key]) for key in faint)


def test_a_prior_whose_triggers_no_caption_names_is_refused(workspace: Workspace) -> None:
    workspace.add_images(1)
    with pytest.raises(ValueError, match=r"no caption names a trigger of prior\.classes \['nobody'\]"):
        run_finetune(_config(workspace, prior={"weight": 1.0, "classes": {"nobody": "person"}}))


def _resume(ws: Workspace, **sections: Any) -> Any:
    state = str(ws.out / "ft_state")
    return run_finetune(_config(ws, train={"max_steps": 5, "resume_from": state}, **sections))


def test_a_resume_reloads_the_recorded_prior_instead_of_recording_it_again(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_images(2)
    first = run_finetune(_config(workspace, train={"max_steps": 2}))
    builds: list[int] = []
    monkeypatch.setattr(PriorPreservation, "build", lambda *_args: builds.append(1))

    resumed = _resume(workspace)

    assert builds == [] and resumed.global_step == 5
    assert resumed.prior is not None and first.prior is not None
    assert all(torch.equal(resumed.prior.targets[key], first.prior.targets[key]) for key in first.prior.keys())


def test_images_added_before_a_resume_get_their_prior_from_the_original_model(
    workspace: Workspace, fakes: Fakes, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(2)
    run_finetune(_config(workspace, train={"max_steps": 4}))
    write_image(workspace.data / "img9.png", "a photo of zed")

    with caplog.at_level(logging.INFO):
        resumed = _resume(workspace)

    assert "Prior: 1 image(s) added since it was recorded; recording theirs from the original weights in krea2" in (
        caplog.text
    )
    prior = resumed.prior
    assert prior is not None and sorted(prior.keys()) == ["img0.png", "img1.png", "img9.png"]
    base, _ = load_krea2_transformer(workspace.checkpoint, CPU, torch.bfloat16)
    x0 = resumed.latent_cache[str(workspace.data / "img9.png")].float()  # [1, C, H, W]
    context = fakes.text.encode("a photo of person", CPU, torch.bfloat16).unsqueeze(0)
    for i in range(2):
        sigma = prior.sigmas["img9.png"][i : i + 1]
        noisy = PriorPreservation.noisy(x0, _noise(int(prior.seeds["img9.png"][i]), x0.shape), sigma)
        with torch.no_grad():
            expected = base(noisy.to(torch.bfloat16), sigma, context)[0].to(torch.bfloat16)
        assert torch.equal(prior.targets["img9.png"][i], expected)


def test_resuming_with_a_prior_from_a_run_that_had_none_is_refused(workspace: Workspace) -> None:
    workspace.add_images(2)
    run_finetune(workspace.finetune_config(train={"max_steps": 2}))

    with pytest.raises(FileNotFoundError, match=r"prior\.safetensors is missing: the resumed run had no prior"):
        _resume(workspace)
