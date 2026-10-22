"""--suggest-steps for the MOSS-Image trainers: the folders an epoch draws from, and the report on them. The
finetune adds its per-folder repeats, the regularisation folder and its notes."""

from __future__ import annotations

from typing import Any

from maltrainer.utils.step_planner import Group, folder_counts
from maltrainer.utils.step_planner import report as step_report


class StepAdvice:
    @classmethod
    def step_groups(cls, config: dict[str, Any]) -> list[Group]:
        """The image folders an epoch draws from, for --suggest-steps."""
        dataset_config = config["dataset"]
        root = dataset_config.get("path", "")
        return [Group("Subjects", root, float(dataset_config.get("repeats", 1) or 1), folder_counts(root))]

    @classmethod
    def step_notes(cls, config: dict[str, Any]) -> list[str]:  # noqa: ARG003 - overridden with the config in use
        return []

    @classmethod
    def suggest_steps(cls, config: dict[str, Any], epochs: float) -> str:
        return step_report(cls.step_groups(config), config.get("train", {}), epochs, cls.step_notes(config))
