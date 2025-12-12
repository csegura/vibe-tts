"""VibeVoice model loading and caching."""

from typing import Optional

import torch

from .config import Config
from .exceptions import ModelLoadError

_model_cache: dict[str, object] = {}


def get_device(device: Optional[str] = None) -> str:
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ModelLoadError("CUDA requested but not available")
        return "cuda"
    if device == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    config: Optional[Config] = None,
    verbose: bool = False,
) -> tuple[object, object]:
    cfg = config or Config.load()
    model_id = cfg.get_model_id(model_name)
    actual_device = get_device(device)
    cache_key = f"{model_id}:{actual_device}"

    if cache_key in _model_cache:
        if verbose:
            print(f"Using cached model: {model_id} on {actual_device}")
        return _model_cache[cache_key]

    if verbose:
        print(f"Loading model: {model_id} on {actual_device}...")

    try:
        from vibevoice.modular.modeling_vibevoice_inference import (
            VibeVoiceForConditionalGenerationInference,
        )
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

        model = VibeVoiceForConditionalGenerationInference.from_pretrained(model_id)
        model = model.to(actual_device)
        model.eval()

        processor = VibeVoiceProcessor.from_pretrained(model_id)

        _model_cache[cache_key] = (model, processor)

        if verbose:
            print(f"Model loaded successfully on {actual_device}")

        return model, processor
    except ImportError as e:
        raise ModelLoadError(
            "vibevoice package not installed. "
            "Install it from: https://github.com/microsoft/VibeVoice"
        ) from e
    except Exception as e:
        raise ModelLoadError(f"Failed to load model {model_id}: {e}") from e


def clear_cache() -> None:
    _model_cache.clear()
