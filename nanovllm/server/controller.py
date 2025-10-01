from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator
from uuid import uuid4

from nanovllm.sampling_params import SamplingParams

from .config import ServerConfig
from .engine_facade import EngineFacade
from .schemas import ChatCompletionRequest, CompletionRequest
from .types import EngineStreamEvent, ServeRequest


class ControllerError(Exception):
    def __init__(self, status_code: int, message: str, type_: str = "invalid_request_error") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.type = type_

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": {
                "message": str(self),
                "type": self.type,
            }
        }


@dataclass(slots=True)
class OpenAIResponse:
    stream: bool
    payload: dict[str, Any] | None = None
    stream_generator: AsyncGenerator[str, None] | None = None


class RequestController:
    def __init__(self, config: ServerConfig, engine: EngineFacade) -> None:
        self.config = config
        self.engine = engine
        self.model_id = engine.model_id
        self._tokenizer = engine.tokenizer

    # --------------------------------------------------------------------- utils
    def _ensure_capacity(self) -> None:
        if self.engine.active_count() >= self.config.max_concurrent_requests:
            raise ControllerError(503, "Too many in-flight requests", "server_error")
        if self.config.max_queue_size > 0 and self.engine.queue_backlog() >= self.config.max_queue_size:
            raise ControllerError(503, "Request queue is full", "server_error")

    def _sampling_from_completion(self, req: CompletionRequest) -> SamplingParams:
        temperature = req.temperature if req.temperature is not None else 1.0
        temperature = max(temperature, 1e-3)
        max_tokens = req.max_tokens if req.max_tokens is not None else 64
        return SamplingParams(temperature=temperature, max_tokens=max_tokens)

    def _sampling_from_chat(self, req: ChatCompletionRequest) -> SamplingParams:
        temperature = req.temperature if req.temperature is not None else 1.0
        temperature = max(temperature, 1e-3)
        max_tokens = req.max_tokens if req.max_tokens is not None else 64
        return SamplingParams(temperature=temperature, max_tokens=max_tokens)

    def _extract_prompt(self, req: CompletionRequest) -> str:
        prompt = req.prompt
        if isinstance(prompt, list):
            prompt = prompt[0]
        assert isinstance(prompt, str)
        return prompt

    def _render_chat_prompt(self, req: ChatCompletionRequest) -> str:
        messages = [msg.dict() for msg in req.messages]
        if hasattr(self._tokenizer, "apply_chat_template"):
            return self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        rendered = []
        for msg in messages:
            rendered.append(f"{msg['role']}: {msg['content']}")
        rendered.append("assistant:")
        return "\n".join(rendered)

    def _create_serve_request(
        self, *, prompt: str, sampling: SamplingParams, stream: bool, kind: str, metadata: dict[str, Any]
    ) -> tuple[ServeRequest, str, int]:
        request_id = metadata.get("request_id") or f"req-{uuid4().hex}"
        created = int(time.time())
        serve_request = ServeRequest(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling,
            stream=stream,
            kind=kind,  # type: ignore[arg-type]
            created_at=time.perf_counter(),
            metadata=metadata,
        )
        return serve_request, request_id, created

    # -------------------------------------------------------------- entrypoints
    async def handle_completion(self, req: CompletionRequest) -> OpenAIResponse:
        self._ensure_capacity()
        if req.stream and not self.config.enable_streaming:
            raise ControllerError(400, "Streaming is disabled by server configuration")
        sampling = self._sampling_from_completion(req)
        prompt = self._extract_prompt(req)
        serve_request, request_id, created = self._create_serve_request(
            prompt=prompt,
            sampling=sampling,
            stream=req.stream,
            kind="completion",
            metadata={"model": req.model},
        )
        if req.stream:
            generator = self._completion_stream(serve_request, request_id, created)
            return OpenAIResponse(stream=True, stream_generator=generator)
        payload = await self._completion_response(serve_request, request_id, created)
        return OpenAIResponse(stream=False, payload=payload)

    async def handle_chat_completion(self, req: ChatCompletionRequest) -> OpenAIResponse:
        self._ensure_capacity()
        if req.stream and not self.config.enable_streaming:
            raise ControllerError(400, "Streaming is disabled by server configuration")
        sampling = self._sampling_from_chat(req)
        prompt = self._render_chat_prompt(req)
        serve_request, request_id, created = self._create_serve_request(
            prompt=prompt,
            sampling=sampling,
            stream=req.stream,
            kind="chat",
            metadata={"model": req.model},
        )
        if req.stream:
            generator = self._chat_stream(serve_request, request_id, created)
            return OpenAIResponse(stream=True, stream_generator=generator)
        payload = await self._chat_response(serve_request, request_id, created)
        return OpenAIResponse(stream=False, payload=payload)

    # ----------------------------------------------------------- response wiring
    async def _completion_response(self, serve_request: ServeRequest, request_id: str, created: int) -> dict[str, Any]:
        text_parts: list[str] = []
        final_event: EngineStreamEvent | None = None
        async for event in self.engine.generate(serve_request):
            if event.text:
                text_parts.append(event.text)
            if event.finished:
                final_event = event
        if final_event is None:
            raise ControllerError(500, "Engine returned no completion", "server_error")
        response = {
            "id": request_id,
            "object": "text_completion",
            "created": created,
            "model": serve_request.metadata.get("model", self.model_id),
            "choices": [
                {
                    "index": 0,
                    "text": "".join(text_parts),
                    "logprobs": None,
                    "finish_reason": final_event.reason or "stop",
                }
            ],
        }
        if final_event.usage:
            response["usage"] = self._usage_to_dict(final_event.usage)
        return response

    async def _chat_response(self, serve_request: ServeRequest, request_id: str, created: int) -> dict[str, Any]:
        text_parts: list[str] = []
        final_event: EngineStreamEvent | None = None
        async for event in self.engine.generate(serve_request):
            if event.text:
                text_parts.append(event.text)
            if event.finished:
                final_event = event
        if final_event is None:
            raise ControllerError(500, "Engine returned no completion", "server_error")
        response = {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": serve_request.metadata.get("model", self.model_id),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "".join(text_parts)},
                    "finish_reason": final_event.reason or "stop",
                    "logprobs": None,
                }
            ],
        }
        if final_event.usage:
            response["usage"] = self._usage_to_dict(final_event.usage)
        return response

    async def _completion_stream(
        self,
        serve_request: ServeRequest,
        request_id: str,
        created: int,
    ) -> AsyncGenerator[str, None]:
        async for event in self.engine.generate(serve_request):
            payload = {
                "id": request_id,
                "object": "text_completion",
                "created": created,
                "model": serve_request.metadata.get("model", self.model_id),
                "choices": [
                    {
                        "index": 0,
                        "text": event.text,
                        "logprobs": None,
                        "finish_reason": event.reason if event.finished else None,
                    }
                ],
            }
            if event.finished and event.usage:
                payload["usage"] = self._usage_to_dict(event.usage)
            yield self._as_sse(payload)
        yield "data: [DONE]\n\n"

    async def _chat_stream(
        self,
        serve_request: ServeRequest,
        request_id: str,
        created: int,
    ) -> AsyncGenerator[str, None]:
        first = True
        async for event in self.engine.generate(serve_request):
            delta: dict[str, Any] = {}
            if first:
                delta["role"] = "assistant"
                first = False
            if event.text:
                delta["content"] = event.text
            payload = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": serve_request.metadata.get("model", self.model_id),
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": event.reason if event.finished else None,
                        "logprobs": None,
                    }
                ],
            }
            if event.finished and event.usage:
                payload["usage"] = self._usage_to_dict(event.usage)
            yield self._as_sse(payload)
        yield "data: [DONE]\n\n"

    def _usage_to_dict(self, usage) -> dict[str, int]:
        return {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }

    @staticmethod
    def _as_sse(payload: dict[str, Any]) -> str:
        return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
