"""Cross-platform audio playback."""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from .exceptions import PlaybackError


def play_audio_sounddevice(audio: np.ndarray, sample_rate: int) -> None:
    try:
        import sounddevice as sd

        sd.play(audio, sample_rate)
        sd.wait()
    except Exception as e:
        raise PlaybackError(f"sounddevice playback failed: {e}") from e


def play_audio_file_system(path: Path, player_command: Optional[str] = None) -> None:
    if player_command:
        cmd = player_command.split() + [str(path)]
    elif sys.platform == "win32":
        powershell = shutil.which("powershell")
        if powershell:
            cmd = [
                powershell,
                "-Command",
                f'(New-Object Media.SoundPlayer "{path}").PlaySync()',
            ]
        else:
            raise PlaybackError("No audio player available on Windows")
    elif sys.platform == "darwin":
        afplay = shutil.which("afplay")
        if afplay:
            cmd = [afplay, str(path)]
        else:
            raise PlaybackError("afplay not found on macOS")
    else:
        for player in ["play", "ffplay", "aplay", "paplay", "mpv", "vlc"]:
            player_path = shutil.which(player)
            if player_path:
                if player == "ffplay":
                    cmd = [player_path, "-nodisp", "-autoexit", str(path)]
                elif player in ("mpv", "vlc"):
                    cmd = [player_path, "--no-video", str(path)]
                else:
                    cmd = [player_path, str(path)]
                break
        else:
            raise PlaybackError(
                "No audio player found. Install ffplay, aplay, paplay, mpv, or vlc"
            )

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise PlaybackError(f"Playback command failed: {e}") from e
    except FileNotFoundError as e:
        raise PlaybackError(f"Player not found: {e}") from e


def play_audio(
    audio: np.ndarray,
    sample_rate: int,
    player_command: Optional[str] = None,
    use_file: bool = False,
    verbose: bool = False,
) -> None:
    if use_file or player_command:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = Path(f.name)

        try:
            from .synthesis import SynthesisResult, save_audio

            result = SynthesisResult(audio=audio, sample_rate=sample_rate, duration=0)
            save_audio(result, temp_path, verbose=False)
            if verbose:
                print(f"Playing audio using system player...")
            play_audio_file_system(temp_path, player_command)
        finally:
            temp_path.unlink(missing_ok=True)
    else:
        if verbose:
            print("Playing audio using sounddevice...")
        play_audio_sounddevice(audio, sample_rate)
