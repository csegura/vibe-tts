"""WebSocket handler for real-time streaming synthesis."""

import asyncio
import json
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..synthesis import (
    SAMPLE_RATE,
    split_text_into_chunks,
    synthesize_text,
    synthesize_text_streaming,
)
from .audio_utils import numpy_to_pcm_bytes

websocket_router = APIRouter()

CHUNK_SIZE = 4096


@websocket_router.websocket("/stream")
async def websocket_stream(websocket: WebSocket) -> None:
    """Handle WebSocket connections for streaming TTS."""
    await websocket.accept()

    config = websocket.app.state.config
    device = websocket.app.state.actual_device
    model_name = websocket.app.state.model_name

    cancel_event = asyncio.Event()

    try:
        while True:
            data = await websocket.receive_text()
            message: dict[str, Any] = json.loads(data)
            msg_type = message.get("type")

            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if msg_type == "cancel":
                cancel_event.set()
                continue

            if msg_type == "synthesize":
                cancel_event.clear()
                text = message.get("text", "")
                voice = message.get("voice")

                if not text:
                    await websocket.send_json(
                        {"type": "error", "error": "Empty text", "code": 1}
                    )
                    continue

                if not websocket.app.state.model_loaded:
                    await websocket.send_json(
                        {"type": "error", "error": "Model not loaded", "code": 2}
                    )
                    continue

                await websocket.send_json(
                    {"type": "start", "sample_rate": SAMPLE_RATE, "format": "pcm_s16le"}
                )

                try:
                    is_realtime = websocket.app.state.is_realtime

                    if is_realtime:
                        # True streaming - audio starts within seconds
                        total_bytes = 0
                        total_samples = 0

                        def stream_audio():
                            return synthesize_text_streaming(
                                text=text,
                                model_name=model_name,
                                device=device,
                                voice_sample=voice,
                                config=config,
                                verbose=False,
                            )

                        # Run streaming generator in thread pool
                        loop = asyncio.get_event_loop()
                        gen = await loop.run_in_executor(None, stream_audio)

                        # Wrap blocking iterator for async consumption
                        def get_next_chunk(generator):
                            try:
                                return next(generator)
                            except StopIteration:
                                return None

                        while True:
                            if cancel_event.is_set():
                                await websocket.send_json({"type": "cancelled"})
                                break

                            audio_chunk = await loop.run_in_executor(
                                None, get_next_chunk, gen
                            )
                            if audio_chunk is None:
                                break

                            pcm_bytes = numpy_to_pcm_bytes(audio_chunk)
                            total_bytes += len(pcm_bytes)
                            total_samples += len(audio_chunk)
                            await websocket.send_bytes(pcm_bytes)
                            await asyncio.sleep(0)

                        if not cancel_event.is_set():
                            duration = total_samples / SAMPLE_RATE
                            await websocket.send_json(
                                {
                                    "type": "complete",
                                    "duration": duration,
                                    "total_bytes": total_bytes,
                                }
                            )
                    else:
                        # Batch mode for standard model
                        chunks = split_text_into_chunks(text)
                        total_bytes = 0
                        total_samples = 0

                        for i, chunk in enumerate(chunks):
                            if cancel_event.is_set():
                                await websocket.send_json({"type": "cancelled"})
                                break

                            await websocket.send_json(
                                {
                                    "type": "progress",
                                    "chunk": i + 1,
                                    "total_chunks": len(chunks),
                                }
                            )

                            result = await asyncio.to_thread(
                                synthesize_text,
                                text=chunk,
                                model_name=model_name,
                                device=device,
                                voice_sample=voice,
                                config=config,
                                verbose=False,
                            )

                            pcm_bytes = numpy_to_pcm_bytes(result.audio)
                            total_bytes += len(pcm_bytes)
                            total_samples += len(result.audio)

                            for j in range(0, len(pcm_bytes), CHUNK_SIZE):
                                if cancel_event.is_set():
                                    break
                                audio_chunk = pcm_bytes[j : j + CHUNK_SIZE]
                                await websocket.send_bytes(audio_chunk)
                                await asyncio.sleep(0)

                        if not cancel_event.is_set():
                            duration = total_samples / SAMPLE_RATE
                            await websocket.send_json(
                                {
                                    "type": "complete",
                                    "duration": duration,
                                    "total_bytes": total_bytes,
                                }
                            )

                except Exception as e:
                    await websocket.send_json(
                        {"type": "error", "error": str(e), "code": 3}
                    )

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "error": str(e), "code": 500})
        except Exception:
            pass
