"""FastAPI application factory for vibe-tts server."""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

from .. import __version__
from ..config import Config
from ..model import get_device, load_model
from .middleware import setup_middleware
from .routes import router
from .websocket import websocket_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - pre-warm model at startup."""
    config: Config = app.state.config
    device = app.state.device
    model_name = app.state.model_name

    print(f"Loading model: {model_name} on {app.state.actual_device}...")
    model, processor = load_model(
        model_name=model_name,
        device=device,
        config=config,
        verbose=True,
    )
    app.state.model = model
    app.state.processor = processor
    app.state.model_loaded = True
    print("Model loaded successfully, server ready.")

    yield

    app.state.model_loaded = False
    print("Server shutting down.")


def create_app(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    config: Optional[Config] = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        model_name: Model to use (e.g., 'vibe-1.5b')
        device: Device to use ('auto', 'cpu', 'cuda')
        config: Configuration object

    Returns:
        Configured FastAPI application
    """
    cfg = config or Config.load()

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

    setup_middleware(app)
    app.include_router(router)
    app.include_router(websocket_router)

    return app
