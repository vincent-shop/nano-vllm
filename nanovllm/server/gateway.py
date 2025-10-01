from __future__ import annotations

from time import perf_counter
from typing import AsyncGenerator, Awaitable, Callable

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from .config import ServerConfig
from .controller import ControllerError, OpenAIResponse, RequestController
from .lifecycle import LifecycleManager
from .schemas import ChatCompletionRequest, CompletionRequest
from .telemetry import Telemetry


def create_app(config: ServerConfig) -> FastAPI:
    lifecycle = LifecycleManager(config)
    telemetry = Telemetry()

    app = FastAPI(title="Nano-vLLM Server", version="0.1")

    @app.on_event("startup")
    async def _startup() -> None:  # pragma: no cover - exercised at runtime
        await lifecycle.startup()

    @app.on_event("shutdown")
    async def _shutdown() -> None:  # pragma: no cover - exercised at runtime
        await lifecycle.shutdown()

    def get_controller() -> RequestController:
        controller = lifecycle.controller
        if controller is None:
            raise ControllerError(503, "Server initializing", "server_error")
        return controller

    def check_auth(request: Request) -> None:
        if config.api_key is None:
            return
        auth_header = request.headers.get("authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise ControllerError(401, "Missing bearer token", "authentication_error")
        token = auth_header.split(" ", 1)[1]
        if token != config.api_key:
            raise ControllerError(403, "Invalid API key", "authentication_error")

    async def respond(
        request: Request,
        handler: Callable[[RequestController], Awaitable[OpenAIResponse]],
        controller: RequestController,
    ):
        start = perf_counter()
        try:
            check_auth(request)
            result = await handler(controller)
        except ControllerError as exc:
            telemetry.observe(perf_counter() - start, exc.status_code)
            return JSONResponse(status_code=exc.status_code, content=exc.to_dict())
        if result.stream:
            assert result.stream_generator is not None

            async def _instrumented() -> AsyncGenerator[str, None]:
                status_code = 200
                try:
                    async for chunk in result.stream_generator:
                        yield chunk
                except ControllerError as exc:
                    status_code = exc.status_code
                    raise
                except Exception:
                    status_code = 500
                    raise
                finally:
                    telemetry.observe(perf_counter() - start, status_code)

            headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            }
            return StreamingResponse(_instrumented(), headers=headers, media_type="text/event-stream")

        telemetry.observe(perf_counter() - start, 200)
        assert result.payload is not None
        return JSONResponse(status_code=200, content=result.payload)

    @app.post("/v1/completions")
    async def completions(
        request: Request,
        payload: CompletionRequest,
        controller: RequestController = Depends(get_controller),
    ):
        async def handler(ctrl: RequestController) -> OpenAIResponse:
            return await ctrl.handle_completion(payload)

        return await respond(request, handler, controller)

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: Request,
        payload: ChatCompletionRequest,
        controller: RequestController = Depends(get_controller),
    ):
        async def handler(ctrl: RequestController) -> OpenAIResponse:
            return await ctrl.handle_chat_completion(payload)

        return await respond(request, handler, controller)

    @app.get(config.health_endpoint)
    async def health() -> dict[str, str]:
        return {"status": "ok", "uptime_s": f"{telemetry.uptime():.2f}"}

    if config.metrics_endpoint:

        @app.get(config.metrics_endpoint)
        async def metrics() -> PlainTextResponse:
            return PlainTextResponse(content=telemetry.render_prometheus(lifecycle.engine))

    return app
