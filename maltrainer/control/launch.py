"""Gives a trainer its key handling: the plain ``KeyCommands`` it always had, or, when the config asks for the control
API, ``ControlledKeys`` with the API running beside the training."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from maltrainer.control.app import create_app
from maltrainer.control.hub import Capabilities, ControlHub
from maltrainer.control.keys import ControlledKeys, PreviewTarget
from maltrainer.control.server import start_server
from maltrainer.control.settings import ControlSettings
from maltrainer.control.stats import RunView
from maltrainer.utils.keyboard import KeyCommands
from maltrainer.utils.previews import Previews
from maltrainer.utils.train_monitor import TrainMonitor

logger = logging.getLogger(__name__)


class PreviewHost(Protocol):
    """A trainer with the MOSS-Image preview hooks (maltrainer/trainers/moss_image/previews.py)."""

    config: dict[str, Any]
    previews: Previews

    def _preview_menu(self) -> None: ...

    def _preview_row_labels(self) -> list[str]: ...


def make_keys(
    factory: Callable[..., KeyCommands],
    config: Mapping[str, Any],
    view: RunView,
    *,
    menu: Callable[[], None] | None = None,
    gallery: Callable[[], None] | None = None,
    previews: Previews | None = None,
    row_labels: Callable[[], list[str]] = list,
) -> KeyCommands:
    """``factory`` builds the plain keys (the trainer's own ``KeyCommands`` name, so tests can replace it).
    A run that asks for the API but cannot bind its port logs that and carries on without it: the run comes first."""
    plain = {name: value for name, value in (("menu", menu), ("gallery", gallery)) if value is not None}
    settings = ControlSettings.from_config(config)
    if settings is None:
        return factory(**plain)
    target = None if previews is None else PreviewTarget(previews, row_labels)
    hub = ControlHub(Capabilities(gallery=gallery is not None, preview_settings=target is not None))
    try:
        running = start_server(create_app(hub, settings), settings)
    except OSError as error:
        logger.error(
            "The control API could not start on %s:%d (%s); training goes on without it",
            settings.host,
            settings.port,
            error,
        )
        return factory(**plain)
    keys = ControlledKeys(hub, view, target, menu=menu, gallery=gallery)
    keys.on_close = running.stop
    keys.url = running.url
    logger.info(
        "Control API: %s (Swagger UI at /docs)%s", running.url, ", bearer token required" if settings.token else ""
    )
    return keys


def make_preview_host_keys(
    factory: Callable[..., KeyCommands],
    host: PreviewHost,
    monitor: TrainMonitor | None = None,
    gallery: Callable[[], None] | None = None,
) -> KeyCommands:
    """``make_keys`` for a trainer with the preview hooks: its menu, preview settings and (a finetune) gallery."""
    return make_keys(
        factory,
        host.config,
        RunView(host, monitor),
        menu=host._preview_menu,
        gallery=gallery,
        previews=host.previews,
        row_labels=host._preview_row_labels,
    )
