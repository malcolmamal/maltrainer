"""The ``control_api:`` section of a training config, checked once, before anything listens.

control_api:
  enabled: true      # off unless set
  host: 127.0.0.1    # loopback by default; any other address needs a token
  port: 8765         # 0 = a free port, logged at the start
  token: ""          # when set, every call must send "Authorization: Bearer <token>"
"""

from __future__ import annotations

import ipaddress
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

KEYS = {"enabled", "host", "port", "token"}


@dataclass(frozen=True)
class ControlSettings:
    host: str = "127.0.0.1"
    port: int = 8765
    token: str = field(default="", repr=False)  # a secret: never in a log line or a traceback

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ControlSettings | None:
        """The settings when the run asks for the API, None when it does not; ValueError for a bad section."""
        section = config.get("control_api")
        if section is None:
            return None
        if not isinstance(section, Mapping):
            raise ValueError("control_api must be a mapping")
        unknown = sorted(set(section) - KEYS)
        if unknown:
            raise ValueError(f"control_api: unknown key(s) {', '.join(map(str, unknown))} (known: {sorted(KEYS)})")
        if section.get("enabled", False) is not True:
            return None
        host, port, token = section.get("host", cls.host), section.get("port", cls.port), section.get("token", "")
        if not isinstance(host, str) or not host:
            raise ValueError("control_api.host must be an address such as 127.0.0.1")
        if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
            raise ValueError("control_api.port must be a whole number from 0 to 65535")
        if not isinstance(token, str):
            raise ValueError("control_api.token must be a string")
        if not token and not _is_loopback(host):
            raise ValueError(f"control_api.host {host!r} is not a loopback address: set control_api.token")
        return cls(host, port, token)

    @property
    def loopback_names(self) -> set[str]:
        """The Host header names the API answers to."""
        return {"127.0.0.1", "localhost", "::1", self.host.lower()}


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
