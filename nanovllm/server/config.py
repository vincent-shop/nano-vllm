from __future__ import annotations

import os
from dataclasses import dataclass


def _get_env(key: str, default: str | None = None) -> str | None:
    value = os.getenv(key)
    return value if value is not None else default


@dataclass(slots=True)
class ServerConfig:
    model: str
    host: str = _get_env("NANOVLLM_HOST", "0.0.0.0") or "0.0.0.0"
    port: int = int(_get_env("NANOVLLM_PORT", "8000") or 8000)
    tensor_parallel_size: int = int(_get_env("NANOVLLM_TP", "1") or 1)
    max_model_len: int | None = None
    max_concurrent_requests: int = int(_get_env("NANOVLLM_MAX_CONCURRENCY", "32") or 32)
    max_queue_size: int = int(_get_env("NANOVLLM_MAX_QUEUE", "128") or 128)
    request_timeout_s: float = float(_get_env("NANOVLLM_REQUEST_TIMEOUT_S", "60") or 60)
    enable_streaming: bool = (_get_env("NANOVLLM_ENABLE_STREAMING", "true") or "true").lower() != "false"
    api_key: str | None = _get_env("NANOVLLM_API_KEY")
    log_level: str = _get_env("NANOVLLM_LOG_LEVEL", "info") or "info"
    health_endpoint: str = _get_env("NANOVLLM_HEALTH_ENDPOINT", "/healthz") or "/healthz"
    metrics_endpoint: str | None = _get_env("NANOVLLM_METRICS_ENDPOINT")
    heartbeat_interval_s: float = float(_get_env("NANOVLLM_HEARTBEAT_INTERVAL_S", "30") or 30)

    def model_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {"tensor_parallel_size": self.tensor_parallel_size}
        if self.max_model_len is not None:
            kwargs["max_model_len"] = self.max_model_len
        return kwargs

    @property
    def allow_anonymous(self) -> bool:
        return self.api_key is None
