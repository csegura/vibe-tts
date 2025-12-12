"""Middleware setup for vibe-tts server."""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..exceptions import VibeTTSError


def setup_middleware(app: FastAPI) -> None:
    """Configure middleware for the FastAPI application."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(VibeTTSError)
    async def vibetss_exception_handler(
        request: Request, exc: VibeTTSError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(exc),
                "code": exc.exit_code,
            },
        )
