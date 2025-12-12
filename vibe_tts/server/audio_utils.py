"""Audio conversion utilities for the server."""

import io
import wave

import numpy as np


def numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert numpy audio array to WAV bytes.

    Args:
        audio: Float32 audio array normalized to [-1, 1]
        sample_rate: Sample rate in Hz

    Returns:
        WAV file bytes with proper headers
    """
    audio_int16 = (audio * 32767).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    return buffer.getvalue()


def numpy_to_pcm_bytes(audio: np.ndarray) -> bytes:
    """Convert numpy audio array to raw PCM bytes (no WAV header).

    Args:
        audio: Float32 audio array normalized to [-1, 1]

    Returns:
        Raw 16-bit PCM bytes
    """
    audio_int16 = (audio * 32767).astype(np.int16)
    return audio_int16.tobytes()
