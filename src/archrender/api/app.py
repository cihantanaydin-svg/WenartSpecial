"""FastAPI application: API under /api/v1, health endpoints, and the static UI on one port."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from archrender.api.routes import auth, plans, projects, runs, system, understand
from archrender.api.security import FailureLimiter
from archrender.core.config import Settings, get_settings
from archrender.core.errors import ArchRenderError, ErrorCode
from archrender.core.logging import configure_logging, get_logger
from archrender.ops.readiness import ReadinessState, evaluate
from archrender.pipeline.services import Services

log = get_logger(__name__)
VERSION = "0.1.0"
STREAMING_PATHS = ("/events",)


class RequestTimeout:
    """Pure-ASGI guard: if a handler has not started its response within ``timeout`` seconds, answer
    504 with a coded error. The RunPod proxy cuts at 100 s, so we stay well below it. Streaming
    responses (SSE, downloads) start immediately and are not cut."""

    def __init__(self, app: ASGIApp, timeout: float) -> None:
        self.app = app
        self.timeout = timeout

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = asyncio.Event()

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                started.set()
            await send(message)

        async def run_app() -> None:
            await self.app(scope, receive, send_wrapper)

        task = asyncio.create_task(run_app())
        waiter = asyncio.create_task(started.wait())
        done, _ = await asyncio.wait(
            {task, waiter}, timeout=self.timeout, return_when=asyncio.FIRST_COMPLETED
        )
        if not done:
            task.cancel()
            waiter.cancel()
            body = (
                ArchRenderError(
                    ErrorCode.REQUEST_TIMEOUT,
                    f"Request exceeded {self.timeout:.0f} s.",
                    "Long operations run as jobs: poll /api/v1/jobs/{id} or subscribe to its events.",
                )
                .to_info()
                .model_dump_json()
            )
            await send(
                {
                    "type": "http.response.start",
                    "status": 504,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": f'{{"error": {body}}}'.encode()})
            return
        waiter.cancel()
        await task


class SecurityHeaders:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                names = {h[0].lower() for h in headers}
                extra = [
                    (b"x-content-type-options", b"nosniff"),
                    (b"referrer-policy", b"no-referrer"),
                    (b"x-frame-options", b"DENY"),
                    (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
                ]
                headers += [h for h in extra if h[0] not in names]
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_wrapper)


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    settings = settings or get_settings()
    svc = services or Services.create(settings)
    app = FastAPI(
        title="ArchRender",
        version=VERSION,
        docs_url="/api/docs",
        openapi_url="/api/v1/openapi.json",
        redoc_url=None,
    )
    app.state.services = svc
    app.state.auth_limiter = FailureLimiter()
    app.state.readiness = ReadinessState()

    @app.exception_handler(ArchRenderError)
    async def _arch_error(_: Request, exc: ArchRenderError) -> JSONResponse:
        return JSONResponse(
            {"error": exc.to_info().model_dump(mode="json")}, status_code=exc.http_status
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        info = ArchRenderError(
            ErrorCode.VALIDATION,
            "Request validation failed.",
            "Fix the fields listed in context.errors.",
            context={"errors": [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]},
        ).to_info()
        return JSONResponse({"error": info.model_dump(mode="json")}, status_code=422)

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled API error")
        info = ArchRenderError(
            ErrorCode.INTERNAL,
            f"Unexpected {type(exc).__name__}.",
            "This is a bug; see the API log.",
        ).to_info()
        return JSONResponse({"error": info.model_dump(mode="json")}, status_code=500)

    for r in (
        auth.router,
        projects.router,
        runs.router,
        understand.router,
        plans.router,
        system.router,
    ):
        app.include_router(r, prefix="/api/v1")

    @app.get("/healthz", tags=["system"])
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", tags=["system"])
    def readyz() -> JSONResponse:
        state = evaluate(svc, app.state.readiness)
        # no details without authentication (see /api/v1/system/info)
        return JSONResponse({"ready": state.ready}, status_code=200 if state.ready else 503)

    ui_dist = settings.ui_dist
    index = ui_dist / "index.html"

    @app.get("/{path:path}", include_in_schema=False)
    def ui(path: str) -> Any:
        if path.startswith("api/"):
            raise ArchRenderError(ErrorCode.NOT_FOUND, f"No API route /{path}.", "See /api/docs.")
        if not index.exists():
            return JSONResponse(
                {
                    "error": {
                        "code": "NOT_FOUND",
                        "message": "UI not built.",
                        "fix_hint": "Run `make ui`.",
                    }
                },
                status_code=404,
            )
        candidate = (ui_dist / path).resolve()
        if path and candidate.is_file() and ui_dist.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(index, headers={"Cache-Control": "no-cache"})

    app.add_middleware(SecurityHeaders)
    app.add_middleware(RequestTimeout, timeout=settings.request_timeout_s)
    return app


def main() -> None:
    import uvicorn

    configure_logging()
    settings = get_settings()
    uvicorn.run(
        create_app(settings),
        host="0.0.0.0",  # noqa: S104 - container port, reached only via the RunPod proxy
        port=8000,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
