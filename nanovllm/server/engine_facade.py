from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import AsyncGenerator, Callable

from nanovllm.engine.sequence import Sequence
from nanovllm.llm import LLM

from .types import EngineStreamEvent, EngineUsage, ServeRequest


_LOGGER = logging.getLogger("nanovllm.server.engine")


@dataclass(slots=True)
class _Command:
    action: str
    request: ServeRequest | None = None
    queue: asyncio.Queue[EngineStreamEvent] | None = None
    ack: asyncio.Future[int] | None = None
    seq_id: int | None = None


@dataclass(slots=True)
class _RequestState:
    request: ServeRequest
    sequence: Sequence
    queue: asyncio.Queue[EngineStreamEvent] | None
    created_at: float = field(default_factory=time.perf_counter)
    delivered_tokens: int = 0
    cancelled: bool = False


class EngineFacade:
    """Adapter that exposes the blocking :class:`LLM` as an async generator."""

    def __init__(self, engine: LLM, *, max_queue_size: int = 128) -> None:
        self._engine = engine
        self._loop: asyncio.AbstractEventLoop | None = None
        self._command_queue: queue.Queue[_Command] = queue.Queue()
        self._active: dict[int, _RequestState] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._max_queue_size = max_queue_size
        self._worker = threading.Thread(target=self._worker_loop, name="EngineFacade", daemon=True)

    # lifecycle -----------------------------------------------------------------
    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._loop is not None and self._loop is not loop:
            raise RuntimeError("EngineFacade already bound to an event loop")
        self._loop = loop
        if not self._worker.is_alive():
            self._worker.start()

    async def stop(self) -> None:
        if self._loop is None:
            return
        self._stop.set()
        self._command_queue.put(_Command(action="shutdown"))
        if self._worker.is_alive():
            self._worker.join(timeout=5)

    # status --------------------------------------------------------------------
    def active_count(self) -> int:
        with self._lock:
            return len(self._active)

    def queue_backlog(self) -> int:
        return self._command_queue.qsize()

    @property
    def tokenizer(self):  # pragma: no cover - simple accessor
        return self._engine.tokenizer

    @property
    def model_id(self) -> str:  # pragma: no cover - simple accessor
        return self._engine.config.model

    # public API ----------------------------------------------------------------
    async def generate(self, request: ServeRequest) -> AsyncGenerator[EngineStreamEvent, None]:
        if self._loop is None:
            raise RuntimeError("EngineFacade must be started before use")
        queue_ = asyncio.Queue[EngineStreamEvent](maxsize=self._max_queue_size)
        ack = self._loop.create_future()
        self._command_queue.put(_Command(action="add", request=request, queue=queue_, ack=ack))
        seq_id = await ack

        try:
            while True:
                event = await queue_.get()
                yield event
                if event.finished:
                    break
        except asyncio.CancelledError:
            await self._cancel_request(seq_id)
            raise
        finally:
            await self._cancel_request(seq_id, graceful=True)

    async def _cancel_request(self, seq_id: int, *, graceful: bool = False) -> None:
        if self._loop is None:
            return
        ack = self._loop.create_future()
        self._command_queue.put(_Command(action="cancel", seq_id=seq_id, ack=ack))
        try:
            await asyncio.wait_for(ack, timeout=0.5 if graceful else 0)
        except asyncio.TimeoutError:
            pass

    # worker --------------------------------------------------------------------
    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                cmd = self._command_queue.get(timeout=0.01 if self._active else None)
            except queue.Empty:
                cmd = None

            if cmd is not None:
                if cmd.action == "shutdown":
                    break
                try:
                    self._handle_command(cmd)
                except Exception:  # pragma: no cover - defensive
                    _LOGGER.exception("Failed to handle engine command %s", cmd.action)

            if not self._active:
                continue

            try:
                self._engine.step()
            except Exception as exc:  # pragma: no cover - propagate failure
                _LOGGER.exception("Engine step failed")
                self._broadcast_error(str(exc))
                continue

            self._emit_new_tokens()

    def _handle_command(self, cmd: _Command) -> None:
        if cmd.action == "add":
            assert cmd.request is not None and cmd.queue is not None and cmd.ack is not None
            seq = self._engine.add_request(cmd.request.prompt, cmd.request.sampling_params)
            state = _RequestState(request=cmd.request, sequence=seq, queue=cmd.queue)
            with self._lock:
                self._active[seq.seq_id] = state
            self._loop_call(cmd.ack.set_result, seq.seq_id)
        elif cmd.action == "cancel":
            assert cmd.seq_id is not None and cmd.ack is not None
            with self._lock:
                state = self._active.get(cmd.seq_id)
                if state is not None:
                    state.cancelled = True
                    state.queue = None
            self._loop_call(cmd.ack.set_result, cmd.seq_id)

    def _emit_new_tokens(self) -> None:
        with self._lock:
            states = list(self._active.values())
        for state in states:
            seq = state.sequence
            new_token_ids = seq.completion_token_ids[state.delivered_tokens :]
            if new_token_ids:
                state.delivered_tokens += len(new_token_ids)
                if not state.cancelled:
                    text = self._engine.tokenizer.decode(new_token_ids, skip_special_tokens=False)
                    event = EngineStreamEvent(
                        request_id=state.request.request_id,
                        text=text,
                        token_ids=list(new_token_ids),
                    )
                    self._enqueue_event(state.queue, event)
            if seq.is_finished:
                finish_reason = "length" if seq.num_completion_tokens >= seq.max_tokens else "stop"
                usage = EngineUsage(
                    prompt_tokens=seq.num_prompt_tokens,
                    completion_tokens=seq.num_completion_tokens,
                )
                if not state.cancelled:
                    event = EngineStreamEvent(
                        request_id=state.request.request_id,
                        text="",
                        token_ids=[],
                        finished=True,
                        reason=finish_reason,
                        usage=usage,
                    )
                    self._enqueue_event(state.queue, event)
                with self._lock:
                    self._active.pop(seq.seq_id, None)

    def _broadcast_error(self, message: str) -> None:
        with self._lock:
            states = list(self._active.values())
            self._active.clear()
        for state in states:
            if state.cancelled or state.queue is None:
                continue
            error = EngineStreamEvent(
                request_id=state.request.request_id,
                text="",
                token_ids=[],
                finished=True,
                reason="error",
                usage=None,
            )
            self._enqueue_event(state.queue, error)

    # helpers -------------------------------------------------------------------
    def _enqueue_event(self, queue_: asyncio.Queue[EngineStreamEvent] | None, event: EngineStreamEvent) -> None:
        if queue_ is None or self._loop is None:
            return
        fut = asyncio.run_coroutine_threadsafe(queue_.put(event), self._loop)
        if fut is not None:

            def _log_exception(f: Future) -> None:  # pragma: no cover - debug helper
                try:
                    f.result()
                except Exception:  # noqa: BLE001
                    _LOGGER.debug("Failed to deliver engine event", exc_info=True)

            fut.add_done_callback(_log_exception)

    def _loop_call(self, func: Callable[..., object], *args) -> None:
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(func, *args)
