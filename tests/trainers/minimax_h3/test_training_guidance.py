"""The contrastive guidance target inside a training step."""

from __future__ import annotations

import pytest
import torch

from .steps import Calls, make_trainer, step
from .world import Workspace


def test_guidance_loss_runs_a_second_detached_pass_on_the_empty_prompt_with_its_own_layout(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = make_trainer(workspace, do_guidance_loss=True)
    calls = Calls(trainer, monkeypatch)

    loss, _ = step(trainer, "image")

    (cond, uncond) = calls.args
    assert cond[2].shape[1] == 4 and uncond[2].shape[1] == 1
    assert (
        uncond[4]["text_indices"].numel() == 1
        and uncond[4]["position_ids"].shape[0] == cond[4]["position_ids"].shape[0] - 3
    )
    assert uncond[0] is cond[0] and uncond[1] is cond[1] and loss.requires_grad


def test_guidance_with_a_scale_of_one_leaves_the_target_unchanged_and_a_bigger_scale_moves_it(
    workspace: Workspace,
) -> None:
    plain, _ = step(make_trainer(workspace), "clip")

    unit, _ = step(make_trainer(workspace, do_guidance_loss=True, guidance_loss_target=1.0), "clip")
    boosted, _ = step(make_trainer(workspace, do_guidance_loss=True, guidance_loss_target=3.5), "clip")

    assert unit.item() == pytest.approx(plain.item(), rel=1e-4) and boosted.item() != pytest.approx(plain.item())


@pytest.mark.parametrize("kind", ["image", "clip", "audio"])
@pytest.mark.parametrize("schedule", ["sigma", "scale"])
@pytest.mark.parametrize("cfg_zero", [False, True])
def test_guidance_trains_every_kind_under_either_schedule_with_or_without_cfg_zero(
    workspace: Workspace, kind: str, schedule: str, cfg_zero: bool
) -> None:
    trainer = make_trainer(
        workspace, do_guidance_loss=True, guidance_loss_schedule=schedule.upper(), guidance_loss_cfg_zero=cfg_zero
    )

    loss, _ = step(trainer, kind)

    assert torch.isfinite(loss)
