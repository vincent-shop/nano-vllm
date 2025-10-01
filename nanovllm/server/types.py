from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from nanovllm.sampling_params import SamplingParams


RequestKind = Literal["completion", "chat"]


@dataclass(slots=True)
class ServeRequest:
    request_id: str
    prompt: str | list[int]
    sampling_params: SamplingParams
    stream: bool
    kind: RequestKind
    created_at: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EngineUsage:
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(slots=True)
class EngineStreamEvent:
    request_id: str
    text: str
    token_ids: list[int]
    finished: bool = False
    reason: str | None = None
    usage: EngineUsage | None = None


@dataclass(slots=True)
class EngineError:
    request_id: str
    message: str
    recoverable: bool = False
