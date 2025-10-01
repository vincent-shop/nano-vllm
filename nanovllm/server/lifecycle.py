from __future__ import annotations

import asyncio
import logging

from nanovllm.llm import LLM

from .config import ServerConfig
from .controller import RequestController
from .engine_facade import EngineFacade


_LOGGER = logging.getLogger("nanovllm.server.lifecycle")


class LifecycleManager:
    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self._engine: EngineFacade | None = None
        self._llm: LLM | None = None
        self.controller: RequestController | None = None

    async def startup(self) -> None:
        loop = asyncio.get_running_loop()
        model_kwargs = self.config.model_kwargs()
        _LOGGER.info("Loading model from %s", self.config.model)
        self._llm = LLM(self.config.model, **model_kwargs)
        self._engine = EngineFacade(self._llm, max_queue_size=self.config.max_queue_size)
        self._engine.start(loop)
        self.controller = RequestController(self.config, self._engine)
        _LOGGER.info("Server ready: host=%s port=%s", self.config.host, self.config.port)

    async def shutdown(self) -> None:
        if self._engine is not None:
            await self._engine.stop()
        if self._llm is not None:
            try:
                self._llm.exit()
            except Exception:  # pragma: no cover - best effort cleanup
                _LOGGER.exception("Error during engine shutdown")

    @property
    def engine(self) -> EngineFacade:
        if self._engine is None:
            raise RuntimeError("LifecycleManager not started")
        return self._engine

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            raise RuntimeError("LifecycleManager not started")
        return self._llm
