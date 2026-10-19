from __future__ import annotations

import io
import logging
import sys
from typing import Any, Dict, List

import pytest

from maltrainer.utils import logging_utils
from maltrainer.utils.logging_utils import log_level, setup_logging


@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    monkeypatch.setattr(logging_utils.logging, "basicConfig", lambda **kwargs: calls.append(kwargs))
    return calls


def test_the_default_logs_info_to_stdout_with_time_and_logger(
    configured: List[Dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)

    setup_logging({})

    (call,) = configured
    assert call["level"] == logging.INFO
    assert call["format"] == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    (handler,) = call["handlers"]
    assert isinstance(handler, logging.StreamHandler)
    assert handler.stream is stdout


def test_the_level_and_format_come_from_the_config(configured: List[Dict[str, Any]]) -> None:
    setup_logging({"level": "DEBUG", "format": "%(message)s"})

    assert (configured[0]["level"], configured[0]["format"]) == (logging.DEBUG, "%(message)s")


@pytest.mark.parametrize(
    ("level", "expected"),
    [("INFO", logging.INFO), ("debug", logging.DEBUG), (" Warning ", logging.WARNING), (5, 5), (logging.ERROR, 40)],
)
def test_a_level_is_a_name_in_any_case_or_a_number(level: object, expected: int) -> None:
    assert log_level(level) == expected


@pytest.mark.parametrize("level", ["verbose", "", "basicConfig", True])
def test_an_unknown_level_names_the_ones_there_are(level: object) -> None:
    with pytest.raises(ValueError, match="logging.level must be one of CRITICAL, ERROR, WARNING, INFO, DEBUG"):
        log_level(level)


def test_a_lower_case_level_configures_logging(configured: List[Dict[str, Any]]) -> None:
    setup_logging({"level": "debug"})

    assert configured[0]["level"] == logging.DEBUG
