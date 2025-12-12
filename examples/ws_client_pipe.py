#!/usr/bin/env python3
"""WebSocket client that outputs raw PCM to stdout for piping to audio players.

Usage:
    python ws_client_pipe.py "Hello world" | ffplay -f s16le -ar 24000 -ac 1 -nodisp -autoexit -
    python ws_client_pipe.py --voice en-Emma "Hello" | aplay -f S16_LE -r 24000 -c 1
    python ws_client_pipe.py "Test" > output.pcm
"""

import argparse
import asyncio
import json
import sys

try:
    import websockets
except ImportError:
    print("Please install websockets: pip install websockets", file=sys.stderr)
    sys.exit(1)


async def synthesize_to_stdout(
    text: str,
    url: str = "ws://localhost:8000/stream",
    voice: str | None = None,
):
    """Connect to vibe-tts server and write PCM audio to stdout."""
    async with websockets.connect(url, ping_timeout=300, ping_interval=30) as ws:
        request = {"type": "synthesize", "text": text}
        if voice:
            request["voice"] = voice

        await ws.send(json.dumps(request))
        print(f"Sent: {text[:50]}{'...' if len(text) > 50 else ''}", file=sys.stderr)

        while True:
            message = await ws.recv()

            if isinstance(message, bytes):
                sys.stdout.buffer.write(message)
                sys.stdout.buffer.flush()
            else:
                msg = json.loads(message)
                msg_type = msg.get("type")

                if msg_type == "start":
                    sample_rate = msg.get("sample_rate", 24000)
                    print(f"Sample rate: {sample_rate} Hz", file=sys.stderr)
                elif msg_type == "progress":
                    print(f"Progress: {msg['chunk']}/{msg['total_chunks']}", file=sys.stderr)
                elif msg_type == "complete":
                    print(f"Complete: {msg['duration']:.2f}s", file=sys.stderr)
                    break
                elif msg_type == "error":
                    print(f"Error: {msg['error']}", file=sys.stderr)
                    sys.exit(1)
                elif msg_type == "cancelled":
                    print("Cancelled", file=sys.stderr)
                    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="WebSocket client that outputs PCM to stdout",
        epilog="Example: python ws_client_pipe.py 'Hello' | ffplay -f s16le -ar 24000 -ac 1 -nodisp -autoexit -"
    )
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("--url", default="ws://localhost:8000/stream", help="WebSocket URL")
    parser.add_argument("--voice", "-v", help="Voice preset (e.g., en-Emma)")
    args = parser.parse_args()

    asyncio.run(synthesize_to_stdout(
        text=args.text,
        url=args.url,
        voice=args.voice,
    ))


if __name__ == "__main__":
    main()
