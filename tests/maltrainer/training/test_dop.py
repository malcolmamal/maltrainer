"""The DOP caption swap and its log lines, without a trainer."""

from __future__ import annotations

import logging

import pytest

from maltrainer.training.dop import dop_captions

SAMPLES = [("a", "a photo of sks"), ("b", "a landscape")]


def test_the_swap_is_reported_with_an_example_and_the_captions_without_the_trigger_warned_about(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="maltrainer.training.dop"):
        captions = dop_captions(SAMPLES, "sks", "person")

    assert captions == {"a": "a photo of person"}
    info, warning = caplog.records
    assert info.getMessage() == (
        "DOP: 1 of 2 captions name the trigger 'sks'; their DOP caption swaps it for 'person'"
        " (e.g. 'a photo of sks' -> 'a photo of person')"
    )
    assert warning.levelno == logging.WARNING
    assert warning.getMessage().startswith("DOP: 1 caption(s) don't contain the trigger 'sks' and get no DOP pass (")


def test_no_warning_when_every_caption_names_the_trigger(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="maltrainer.training.dop"):
        dop_captions(SAMPLES[:1], "sks", "person")

    assert [record.levelno for record in caplog.records] == [logging.INFO]
