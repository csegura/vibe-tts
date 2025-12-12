"""VibeVoice model loading and caching."""

from typing import Optional, Tuple

import torch

from .config import Config
from .exceptions import ModelLoadError

_model_cache: dict[str, Tuple[object, object, bool]] = {}

# Clear cache at module load to ensure fresh model/processor on restart
_model_cache.clear()


def get_device(device: Optional[str] = None) -> str:
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ModelLoadError("CUDA requested but not available")
        return "cuda"
    if device == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def is_realtime_model(model_id: str) -> bool:
    """Check if model is the realtime/streaming variant."""
    model_lower = model_id.lower()
    return "realtime" in model_lower or "streaming" in model_lower


def load_model(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    config: Optional[Config] = None,
    verbose: bool = False,
) -> Tuple[object, object, bool]:
    """Load VibeVoice model and processor.

    Returns:
        Tuple of (model, processor, is_realtime)
    """
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

    realtime = is_realtime_model(model_id)

    try:
        if realtime:
            # Use streaming inference class and processor for realtime model
            from vibevoice.modular.modeling_vibevoice_streaming_inference import (
                VibeVoiceStreamingForConditionalGenerationInference,
            )
            from vibevoice.processor.vibevoice_streaming_processor import (
                VibeVoiceStreamingProcessor,
            )

            # Device-specific configuration for realtime model
            if actual_device == "cuda":
                load_dtype = torch.bfloat16
                # Try flash_attention_2, fall back to sdpa if not available
                try:
                    import flash_attn  # noqa: F401
                    attn_impl = "flash_attention_2"
                except ImportError:
                    attn_impl = "sdpa"
            else:
                load_dtype = torch.float32
                attn_impl = "sdpa"

            if verbose:
                print(f"  Using streaming model with dtype={load_dtype}, attn={attn_impl}")

            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                model_id,
                torch_dtype=load_dtype,
                device_map=actual_device,
                attn_implementation=attn_impl,
            )
            processor = VibeVoiceStreamingProcessor.from_pretrained(model_id)
            if verbose:
                print(f"  Processor class: {type(processor).__name__}")
        else:
            # Use standard inference class and processor for non-realtime model
            from vibevoice.modular.modeling_vibevoice_inference import (
                VibeVoiceForConditionalGenerationInference,
            )
            from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

            model = VibeVoiceForConditionalGenerationInference.from_pretrained(model_id)
            model = model.to(actual_device)
            processor = VibeVoiceProcessor.from_pretrained(model_id)

        model.eval()

        _model_cache[cache_key] = (model, processor, realtime)

        if verbose:
            print(f"Model loaded successfully on {actual_device}")

        return model, processor, realtime
    except ImportError as e:
        raise ModelLoadError(
            "vibevoice package not installed. "
            "Install it from: https://github.com/microsoft/VibeVoice"
        ) from e
    except Exception as e:
        raise ModelLoadError(f"Failed to load model {model_id}: {e}") from e


def clear_cache() -> None:
    _model_cache.clear()
