#!/usr/bin/env python3
"""Simple WebSocket client for vibe-tts server.

Usage:
    python ws_client.py "Hello, this is a test"
    python ws_client.py --voice en-Emma "Hello world"
    python ws_client.py --url ws://localhost:8000/stream "Hello"
"""

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import tempfile

try:
    import websockets
except ImportError:
    print("Please install websockets: pip install websockets")
    sys.exit(1)

try:
    import sounddevice as sd
    sd.query_devices()
    HAS_SOUNDDEVICE = True
except (ImportError, Exception):
    HAS_SOUNDDEVICE = False

AUDIO_PLAYERS = ["play", "ffplay", "aplay", "paplay", "mpv", "vlc"]


def find_audio_player() -> str | None:
    """Find first available command-line audio player."""
    for player in AUDIO_PLAYERS:
        if shutil.which(player):
            return player
    return None


def start_streaming_player(sample_rate: int) -> subprocess.Popen | None:
    """Start ffplay process for streaming PCM input."""
    if not shutil.which("ffplay"):
        return None
    return subprocess.Popen(
        [
            "ffplay",
            "-f", "s16le",
            "-ar", str(sample_rate),
            "-ac", "1",
            "-nodisp",
            "-autoexit",
            "-probesize", "32",        # Minimal probe size for faster start
            "-analyzeduration", "0",   # Don't analyze, we know the format
            "-fflags", "nobuffer",     # Reduce buffering latency
            "-flags", "low_delay",     # Low delay mode
            "-infbuf",                 # Don't limit input buffer
            "-",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def play_with_command(audio_data: bytes, sample_rate: int, player: str) -> None:
    """Play audio using a command-line player."""
    import wave

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        with wave.open(tmp, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)

    try:
        if player == "ffplay":
            cmd = [player, "-nodisp", "-autoexit", tmp_path]
        elif player == "play":
            cmd = [player, tmp_path]
        elif player == "aplay":
            cmd = [player, tmp_path]
        elif player == "paplay":
            cmd = [player, tmp_path]
        elif player == "mpv":
            cmd = [player, "--no-video", tmp_path]
        elif player == "vlc":
            cmd = [player, "--intf", "dummy", "--play-and-exit", tmp_path]
        else:
            cmd = [player, tmp_path]

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finally:
        import os
        os.unlink(tmp_path)


async def synthesize_and_play(
    text: str,
    url: str = "ws://localhost:8000/stream",
    voice: str | None = None,
    output_file: str | None = None,
    stream: bool = False,
):
    """Connect to vibe-tts server and stream audio."""
    audio_data = bytearray()
    sample_rate = 24000
    player_proc = None

    async with websockets.connect(url, ping_timeout=300, ping_interval=30) as ws:
        request = {"type": "synthesize", "text": text}
        if voice:
            request["voice"] = voice

        await ws.send(json.dumps(request))
        print(f"Sent: {text[:50]}{'...' if len(text) > 50 else ''}")

        while True:
            message = await ws.recv()

            if isinstance(message, bytes):
                audio_data.extend(message)
                if stream and player_proc and player_proc.stdin:
                    player_proc.stdin.write(message)
            else:
                msg = json.loads(message)
                msg_type = msg.get("type")

                if msg_type == "start":
                    sample_rate = msg.get("sample_rate", 24000)
                    print(f"Streaming started (sample rate: {sample_rate} Hz)")
                    if stream:
                        player_proc = start_streaming_player(sample_rate)
                        if player_proc:
                            print("Streaming playback started (ffplay)")
                        else:
                            print("Warning: ffplay not found, will play after download")
                elif msg_type == "progress":
                    print(f"Progress: chunk {msg['chunk']}/{msg['total_chunks']}")
                elif msg_type == "complete":
                    print(f"Complete! Duration: {msg['duration']:.2f}s, Size: {msg['total_bytes']} bytes")
                    if player_proc and player_proc.stdin:
                        player_proc.stdin.close()
                        player_proc.wait()
                        player_proc = None
                    break
                elif msg_type == "error":
                    print(f"Error: {msg['error']}")
                    return
                elif msg_type == "cancelled":
                    print("Cancelled")
                    return

    if not audio_data:
        print("No audio received")
        return

    if output_file:
        import wave
        with wave.open(output_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)
        print(f"Saved to: {output_file}")

    # Play audio (skip if already streamed)
    if stream and shutil.which("ffplay"):
        return  # Already played via streaming

    played = False
    if HAS_SOUNDDEVICE:
        try:
            import array
            samples = array.array('h', audio_data)
            audio_float = [s / 32768.0 for s in samples]
            print("Playing audio (sounddevice)...")
            sd.play(audio_float, sample_rate)
            sd.wait()
            played = True
        except Exception as e:
            print(f"sounddevice failed: {e}")

    if not played:
        player = find_audio_player()
        if player:
            print(f"Playing audio ({player})...")
            play_with_command(bytes(audio_data), sample_rate, player)
        elif not output_file:
            print("No audio player available. Use --output to save to file.")


def main():
    parser = argparse.ArgumentParser(description="WebSocket client for vibe-tts")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("--url", default="ws://localhost:8000/stream", help="WebSocket URL")
    parser.add_argument("--voice", "-v", help="Voice preset (e.g., en-Emma)")
    parser.add_argument("--output", "-o", help="Save audio to WAV file")
    parser.add_argument("--stream", "-s", action="store_true",
                        help="Play audio as it arrives (requires ffplay)")
    args = parser.parse_args()

    asyncio.run(synthesize_and_play(
        text=args.text,
        url=args.url,
        voice=args.voice,
        output_file=args.output,
        stream=args.stream,
    ))


if __name__ == "__main__":
    main()
