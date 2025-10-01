from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from .engine_facade import EngineFacade


@dataclass(slots=True)
class _Counters:
    requests_total: int = 0
    requests_failed: int = 0
    latency_sum: float = 0.0
    latency_count: int = 0


class Telemetry:
    def __init__(self) -> None:
        self._counters = _Counters()
        self._lock = threading.Lock()
        self._started = time.time()

    def observe(self, duration_s: float, status_code: int) -> None:
        with self._lock:
            self._counters.requests_total += 1
            if status_code >= 400:
                self._counters.requests_failed += 1
            self._counters.latency_sum += duration_s
            self._counters.latency_count += 1

    def snapshot(self) -> _Counters:
        with self._lock:
            return _Counters(
                requests_total=self._counters.requests_total,
                requests_failed=self._counters.requests_failed,
                latency_sum=self._counters.latency_sum,
                latency_count=self._counters.latency_count,
            )

    def uptime(self) -> float:
        return time.time() - self._started

    def render_prometheus(self, engine: EngineFacade | None = None) -> str:
        counters = self.snapshot()
        avg_latency = counters.latency_sum / counters.latency_count if counters.latency_count else 0.0
        lines = [
            "# HELP nanovllm_requests_total Total number of processed requests",
            "# TYPE nanovllm_requests_total counter",
            f"nanovllm_requests_total {counters.requests_total}",
            "# HELP nanovllm_requests_failed Total number of failed requests",
            "# TYPE nanovllm_requests_failed counter",
            f"nanovllm_requests_failed {counters.requests_failed}",
            "# HELP nanovllm_request_latency_seconds_avg Rolling average latency in seconds",
            "# TYPE nanovllm_request_latency_seconds_avg gauge",
            f"nanovllm_request_latency_seconds_avg {avg_latency}",
        ]
        if engine is not None:
            lines.extend(
                [
                    "# HELP nanovllm_inflight_requests Current number of in-flight engine requests",
                    "# TYPE nanovllm_inflight_requests gauge",
                    f"nanovllm_inflight_requests {engine.active_count()}",
                ]
            )
        return "\n".join(lines) + "\n"
