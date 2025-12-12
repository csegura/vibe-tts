"""FastAPI dependency injection for vibe-tts server."""

from typing import Any

from fastapi import HTTPException, Request


def get_model(request: Request) -> tuple[Any, Any]:
    """Get the loaded model and processor from app state.

    Returns:
        Tuple of (model, processor)

    Raises:
        HTTPException: If model is not loaded yet
    """
    if not request.app.state.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return request.app.state.model, request.app.state.processor


def get_config(request: Request) -> Any:
    """Get the configuration from app state."""
    return request.app.state.config
