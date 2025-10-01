"""Server package exposing an OpenAI-compatible HTTP interface."""

from .config import ServerConfig
from .gateway import create_app


__all__ = ["ServerConfig", "create_app"]
