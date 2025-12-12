"""HTTP route handlers for vibe-tts server."""

import torch
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from .. import __version__
from ..config import MODELS
from ..exceptions import VibeTTSError
from ..script_parser import parse_script
from ..synthesis import SAMPLE_RATE, synthesize_script, synthesize_text
from .audio_utils import numpy_to_wav_bytes
from .schemas import (
    HealthResponse,
    ServerInfoResponse,
    SynthesizeRequest,
    SynthesizeScriptRequest,
)

router = APIRouter()


@router.post(
    "/synthesize",
    response_class=Response,
    responses={
        200: {"content": {"audio/wav": {}}},
        400: {"description": "Invalid request"},
        500: {"description": "Synthesis error"},
        503: {"description": "Model not loaded"},
    },
)
async def synthesize(request: Request, body: SynthesizeRequest) -> Response:
    """Synthesize text to speech and return WAV audio."""
    if not request.app.state.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        config = request.app.state.config
        device = request.app.state.actual_device
        model_name = request.app.state.model_name

        result = synthesize_text(
            text=body.text,
            model_name=model_name,
            device=device,
            voice_sample=body.voice,
            max_duration=body.max_duration,
            config=config,
            verbose=False,
        )

        wav_bytes = numpy_to_wav_bytes(result.audio, result.sample_rate)

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "X-Audio-Duration": str(result.duration),
                "X-Sample-Rate": str(result.sample_rate),
            },
        )
    except VibeTTSError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post(
    "/synthesize-script",
    response_class=Response,
    responses={
        200: {"content": {"audio/wav": {}}},
        400: {"description": "Invalid script format"},
        500: {"description": "Synthesis error"},
        503: {"description": "Model not loaded"},
    },
)
async def synthesize_script_endpoint(
    request: Request, body: SynthesizeScriptRequest
) -> Response:
    """Synthesize multi-speaker script to speech and return WAV audio."""
    if not request.app.state.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        config = request.app.state.config
        device = request.app.state.actual_device
        model_name = request.app.state.model_name

        lines = parse_script(body.script)
        speaker_map = body.speaker_map or {}

        result = synthesize_script(
            lines=lines,
            speaker_map=speaker_map,
            model_name=model_name,
            device=device,
            max_duration=body.max_duration,
            config=config,
            verbose=False,
        )

        wav_bytes = numpy_to_wav_bytes(result.audio, result.sample_rate)

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "X-Audio-Duration": str(result.duration),
                "X-Sample-Rate": str(result.sample_rate),
            },
        )
    except VibeTTSError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/info", response_model=ServerInfoResponse)
async def info(request: Request) -> ServerInfoResponse:
    """Get server information."""
    model_name = request.app.state.model_name
    device = request.app.state.actual_device

    cuda_device = None
    if device == "cuda" and torch.cuda.is_available():
        cuda_device = torch.cuda.get_device_name(0)

    return ServerInfoResponse(
        version=__version__,
        model=model_name,
        model_id=MODELS.get(model_name, model_name),
        device=device,
        cuda_available=torch.cuda.is_available(),
        cuda_device=cuda_device,
        sample_rate=SAMPLE_RATE,
        status="ready" if request.app.state.model_loaded else "loading",
    )


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if request.app.state.model_loaded else "starting",
        model_loaded=request.app.state.model_loaded,
    )
