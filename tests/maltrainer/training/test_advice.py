"""--suggest-steps' StepAdvice moved from the MOSS-Image trainers to ``training`` (shared with Krea 2)."""

from __future__ import annotations

from maltrainer.trainers.moss_image import advice as moss_advice
from maltrainer.training.advice import StepAdvice


def test_the_old_import_path_names_the_same_class() -> None:
    assert moss_advice.StepAdvice is StepAdvice
    assert moss_advice.__all__ == ["StepAdvice"]
