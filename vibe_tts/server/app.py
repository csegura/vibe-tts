"""FastAPI application factory for vibe-tts server."""

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

from .. import __version__
from ..config import Config
from ..model import get_device, load_model
from .middleware import setup_middleware
from .routes import router
from .websocket import websocket_router

# Default streaming buffer in seconds
DEFAULT_STREAM_BUFFER = 1.5


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - pre-warm model at startup."""
    config: Config = app.state.config
    device = app.state.device
    model_name = app.state.model_name

    print(f"Loading model: {model_name} on {app.state.actual_device}...")
    model, processor, is_realtime = load_model(
        model_name=model_name,
        device=device,
        config=config,
        verbose=True,
    )
    app.state.model = model
    app.state.processor = processor
    app.state.is_realtime = is_realtime
    app.state.model_loaded = True
    print("Model loaded successfully, server ready.")

    yield

    app.state.model_loaded = False
    print("Server shutting down.")


def create_app(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    config: Optional[Config] = None,
    stream_buffer: Optional[float] = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        model_name: Model to use (e.g., 'vibe-1.5b')
        device: Device to use ('auto', 'cpu', 'cuda')
        config: Configuration object
        stream_buffer: Streaming audio buffer in seconds (default: 1.5)

    Returns:
        Configured FastAPI application
    """
    cfg = config or Config.load()

    # Get stream buffer from param, env var, or default
    if stream_buffer is None:
        stream_buffer = float(os.environ.get("VIBE_STREAM_BUFFER", DEFAULT_STREAM_BUFFER))

    app = FastAPI(
        title="vibe-tts Server",
        description="Text-to-Speech API using Microsoft VibeVoice",
        version=__version__,
        lifespan=lifespan,
    )

    app.state.config = cfg
    app.state.model_name = model_name or cfg.default_model
    app.state.device = device if device != "auto" else None
    app.state.actual_device = get_device(app.state.device)
    app.state.model_loaded = False
    app.state.model = None
    app.state.processor = None
    app.state.is_realtime = False
    app.state.stream_buffer = stream_buffer

    setup_middleware(app)
    app.include_router(router)
    app.include_router(websocket_router)

    return app
