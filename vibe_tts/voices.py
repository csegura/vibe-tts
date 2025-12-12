"""Voice preset management for vibe-tts."""

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from .config import get_config_dir

VOICE_PRESETS = {
    # English voices
    "en-Emma": "en-Emma_woman",
    "en-Carter": "en-Carter_man",
    "en-Davis": "en-Davis_man",
    "en-Frank": "en-Frank_man",
    "en-Grace": "en-Grace_woman",
    "en-Mike": "en-Mike_man",
    # German
    "de-Spk0": "de-Spk0_man",
    "de-Spk1": "de-Spk1_woman",
    # French
    "fr-Spk0": "fr-Spk0_man",
    "fr-Spk1": "fr-Spk1_woman",
    # Italian
    "it-Spk0": "it-Spk0_woman",
    "it-Spk1": "it-Spk1_man",
    # Japanese
    "jp-Spk0": "jp-Spk0_man",
    "jp-Spk1": "jp-Spk1_woman",
    # Korean
    "kr-Spk0": "kr-Spk0_woman",
    "kr-Spk1": "kr-Spk1_man",
    # Dutch
    "nl-Spk0": "nl-Spk0_man",
    "nl-Spk1": "nl-Spk1_woman",
    # Polish
    "pl-Spk0": "pl-Spk0_man",
    "pl-Spk1": "pl-Spk1_woman",
    # Portuguese
    "pt-Spk0": "pt-Spk0_woman",
    "pt-Spk1": "pt-Spk1_man",
    # Spanish
    "sp-Spk0": "sp-Spk0_woman",
    "sp-Spk1": "sp-Spk1_man",
    # Indian English
    "in-Samuel": "in-Samuel_man",
}

VOICE_REPO_URL = "https://huggingface.co/microsoft/VibeVoice-1.5B/resolve/main"
VOICE_GITHUB_URL = "https://raw.githubusercontent.com/microsoft/VibeVoice/main/demo/voices/streaming_model"


def get_voices_dir() -> Path:
    """Get the directory for cached voice presets."""
    return get_config_dir() / "voices"


def list_available_voices() -> list[str]:
    """List all available voice preset names."""
    return sorted(VOICE_PRESETS.keys())


def get_voice_path(voice_name: str) -> Optional[Path]:
    """Get the path to a voice preset file.

    Args:
        voice_name: Voice name (e.g., 'en-Emma') or full filename

    Returns:
        Path to the voice file if it exists, None otherwise
    """
    if voice_name in VOICE_PRESETS:
        filename = f"{VOICE_PRESETS[voice_name]}.pt"
    elif voice_name.endswith(".pt"):
        filename = voice_name
    else:
        filename = f"{voice_name}.pt"

    voice_path = get_voices_dir() / filename
    if voice_path.exists():
        return voice_path

    # Check if it's a direct path
    direct_path = Path(voice_name)
    if direct_path.exists():
        return direct_path

    return None


def download_voice(voice_name: str, verbose: bool = False) -> Path:
    """Download a voice preset from the VibeVoice repository.

    Args:
        voice_name: Voice name (e.g., 'en-Emma')
        verbose: Print progress

    Returns:
        Path to the downloaded voice file
    """
    import urllib.request
    import urllib.error

    if voice_name in VOICE_PRESETS:
        filename = f"{VOICE_PRESETS[voice_name]}.pt"
    else:
        filename = f"{voice_name}.pt"

    voices_dir = get_voices_dir()
    voices_dir.mkdir(parents=True, exist_ok=True)
    voice_path = voices_dir / filename

    if voice_path.exists():
        if verbose:
            print(f"Voice preset already cached: {voice_path}")
        return voice_path

    url = f"{VOICE_GITHUB_URL}/{filename}"
    if verbose:
        print(f"Downloading voice preset: {voice_name}...")
        print(f"  URL: {url}")

    try:
        urllib.request.urlretrieve(url, voice_path)
        if verbose:
            print(f"  Saved to: {voice_path}")
        return voice_path
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Failed to download voice '{voice_name}': {e}") from e


def ensure_voice(voice_name: str, verbose: bool = False) -> Path:
    """Ensure a voice preset is available, downloading if necessary.

    Args:
        voice_name: Voice name or path to voice file
        verbose: Print progress

    Returns:
        Path to the voice file
    """
    # Check if it's already a valid path
    path = get_voice_path(voice_name)
    if path is not None:
        return path

    # Try to download it
    return download_voice(voice_name, verbose=verbose)


def load_voice_sample(voice_name: str, verbose: bool = False) -> np.ndarray:
    """Load a voice sample as a numpy array.

    This function loads the .pt voice file with weights_only=False to handle
    PyTorch 2.6+ compatibility issues.

    Args:
        voice_name: Voice name or path to voice file
        verbose: Print progress

    Returns:
        Voice sample as numpy array
    """
    voice_path = ensure_voice(voice_name, verbose=verbose)

    if voice_path.suffix == ".pt":
        # Load with weights_only=False for PyTorch 2.6+ compatibility
        audio_tensor = torch.load(voice_path, map_location="cpu", weights_only=False)
        audio_tensor = audio_tensor.squeeze()
        if isinstance(audio_tensor, torch.Tensor):
            audio_array = audio_tensor.numpy()
        else:
            audio_array = np.array(audio_tensor)
        return audio_array.astype(np.float32)
    elif voice_path.suffix == ".npy":
        return np.load(voice_path).astype(np.float32)
    else:
        # For audio files, use librosa
        import librosa
        audio_array, _ = librosa.load(str(voice_path), sr=24000, mono=True)
        return audio_array.astype(np.float32)
