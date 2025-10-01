from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, root_validator, validator


class CompletionRequest(BaseModel):
    model: str
    prompt: str | list[str] | None = Field(default=None)
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: bool = False
    n: int = 1
    stop: Optional[str | list[str]] = None
    logprobs: Optional[int] = None
    best_of: Optional[int] = None

    @validator("n")
    def validate_n(cls, value: int) -> int:
        if value != 1:
            raise ValueError("Only n=1 is supported")
        return value

    @validator("best_of")
    def validate_best_of(cls, value: Optional[int]) -> Optional[int]:
        if value not in (None, 1):
            raise ValueError("best_of is not supported")
        return value

    @root_validator
    def validate_prompt(cls, values: dict[str, Any]) -> dict[str, Any]:
        prompt = values.get("prompt")
        if prompt is None:
            raise ValueError("prompt is required")
        if isinstance(prompt, list) and len(prompt) != 1:
            raise ValueError("Only a single prompt is supported")
        return values


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: bool = False
    n: int = 1

    @validator("n")
    def validate_n(cls, value: int) -> int:
        if value != 1:
            raise ValueError("Only n=1 is supported")
        return value

    @validator("messages")
    def validate_messages(cls, value: list[ChatMessage]) -> list[ChatMessage]:
        if not value:
            raise ValueError("messages must not be empty")
        return value
