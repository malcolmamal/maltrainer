"""The control API: FastAPI routes over a ``ControlHub``. Swagger UI at ``/docs``, the schema at ``/openapi.json``.

Commands answer 202 at once with a command id; they take effect at the training loop's next step boundary (at once while
paused) and ``GET /commands/{id}`` says when they did, or that they failed."""

from __future__ import annotations

import secrets
from collections.abc import Awaitable, Callable
from typing import Annotated
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from maltrainer.control.hub import ControlHub
from maltrainer.control.routes import command_routes, preview_routes, status_routes
from maltrainer.control.settings import ControlSettings

DESCRIPTION = """Check on and steer the training run that serves this API. The numbers refresh once per optimizer
step. Commands are queued and carried out at the next step boundary, the same place the console keys act (`s`, `q`, `p`,
`g`, and the `e` menu's preview settings)."""

bearer = HTTPBearer(auto_error=False, description="Needed only when control_api.token is set")
Guard = Callable[[Request, Callable[[Request], Awaitable[Response]]], Awaitable[Response]]


def _authorizer(token: str) -> Callable[[HTTPAuthorizationCredentials | None], None]:
    def authorize(credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]) -> None:
        if token and (
            credentials is None or not secrets.compare_digest(credentials.credentials.encode(), token.encode())
        ):
            raise HTTPException(401, "missing or wrong bearer token", headers={"WWW-Authenticate": "Bearer"})

    return authorize


def _same_origin_only(settings: ControlSettings) -> Guard:
    """A browser page on another site (or a rebound DNS name) must not steer the run: a cross-origin ``Origin`` is
    refused, and without a token the Host must be one of ours (with a token a request has to carry it, which a page
    on another site cannot, and the API may be reached under any name). Tools without an Origin header pass."""

    async def guard(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        host = request.headers.get("host", "")
        origin = request.headers.get("origin")
        foreign_host = not settings.token and (urlsplit(f"//{host}").hostname or "") not in settings.loopback_names
        if foreign_host or (origin is not None and origin != f"http://{host}"):
            return JSONResponse({"detail": "cross-origin request refused"}, status_code=403)
        return await call_next(request)

    return guard


def create_app(hub: ControlHub, settings: ControlSettings) -> FastAPI:
    authorize = _authorizer(settings.token)
    app = FastAPI(title="maltrainer control API", description=DESCRIPTION, version="1.0")
    app.middleware("http")(_same_origin_only(settings))
    for router in (status_routes(hub), command_routes(hub), preview_routes(hub)):
        app.include_router(router, dependencies=[Depends(authorize)])
    return app
