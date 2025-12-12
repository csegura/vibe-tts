"""Pydantic request/response models for the server API."""

from typing import Optional

from pydantic import BaseModel, Field


class SynthesizeRequest(BaseModel):
    """Request model for text-to-speech synthesis."""

    text: str = Field(..., min_length=1, max_length=100000)
    voice: Optional[str] = Field(default=None, description="Voice preset name (e.g., 'en-Emma')")
    max_duration: int = Field(default=3600, ge=1, le=7200)


class SynthesizeScriptRequest(BaseModel):
    """Request model for multi-speaker script synthesis."""

    script: str = Field(..., min_length=1)
    speaker_map: Optional[dict[str, str]] = Field(default=None)
    max_duration: int = Field(default=3600, ge=1, le=7200)


class ServerInfoResponse(BaseModel):
    """Response model for server info endpoint."""

    version: str
    model: str
    model_id: str
    device: str
    cuda_available: bool
    cuda_device: Optional[str]
    sample_rate: int
    status: str


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str
    model_loaded: bool


class ErrorResponse(BaseModel):
    """Response model for error responses."""

    error: str
    code: int
    detail: Optional[str] = None
