from __future__ import annotations

from typing import Any

import pytest

from maltrainer.control.settings import ControlSettings


def test_no_section_or_a_section_not_enabled_means_no_api() -> None:
    assert ControlSettings.from_config({}) is None
    assert ControlSettings.from_config({"control_api": None}) is None
    assert ControlSettings.from_config({"control_api": {"port": 9000}}) is None
    assert ControlSettings.from_config({"control_api": {"enabled": False}}) is None


def test_enabled_alone_uses_loopback_and_the_default_port() -> None:
    settings = ControlSettings.from_config({"control_api": {"enabled": True}})

    assert settings == ControlSettings(host="127.0.0.1", port=8765, token="")


def test_every_key_is_read() -> None:
    section = {"enabled": True, "host": "localhost", "port": 0, "token": "s3cret"}

    assert ControlSettings.from_config({"control_api": section}) == ControlSettings("localhost", 0, "s3cret")


@pytest.mark.parametrize(
    ("section", "message"),
    [
        ({"enabled": True, "prot": 1}, "unknown key"),
        ({"enabled": True, "port": "8765"}, "whole number"),
        ({"enabled": True, "port": True}, "whole number"),
        ({"enabled": True, "port": 70000}, "whole number"),
        ({"enabled": True, "port": -1}, "whole number"),
        ({"enabled": True, "host": ""}, "address"),
        ({"enabled": True, "token": 5}, "must be a string"),
        ({"enabled": True, "host": "10.1.2.3"}, "set control_api.token"),
        ({"enabled": True, "host": "192.168.1.5", "token": ""}, "set control_api.token"),
    ],
)
def test_a_bad_section_is_an_error_naming_the_problem(section: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ControlSettings.from_config({"control_api": section})


def test_a_section_that_is_not_a_mapping_is_an_error() -> None:
    with pytest.raises(ValueError, match="mapping"):
        ControlSettings.from_config({"control_api": True})


def test_another_address_is_allowed_with_a_token_and_is_a_name_the_api_answers_to() -> None:
    settings = ControlSettings.from_config({"control_api": {"enabled": True, "host": "10.0.0.7", "token": "t"}})

    assert settings is not None
    assert settings.loopback_names == {"127.0.0.1", "localhost", "::1", "10.0.0.7"}


def test_ipv6_loopback_needs_no_token() -> None:
    assert ControlSettings.from_config({"control_api": {"enabled": True, "host": "::1"}}) is not None


def test_the_token_is_not_in_the_text_of_the_settings() -> None:
    settings = ControlSettings.from_config({"control_api": {"enabled": True, "token": "s3cret"}})

    assert "s3cret" not in repr(settings)
