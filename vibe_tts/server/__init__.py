"""Server package for vibe-tts HTTP/WebSocket API."""

from .app import create_app
from .schemas import (
    ErrorResponse,
    HealthResponse,
    ServerInfoResponse,
    SynthesizeRequest,
    SynthesizeScriptRequest,
)

__all__ = [
    "create_app",
    "SynthesizeRequest",
    "SynthesizeScriptRequest",
    "ServerInfoResponse",
    "HealthResponse",
    "ErrorResponse",
]
