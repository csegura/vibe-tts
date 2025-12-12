"""Audio generation from text."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .config import Config
from .exceptions import SynthesisError
from .model import get_device, load_model
from .script_parser import ScriptLine

SAMPLE_RATE = 24000
MAX_CHUNK_CHARS = 5000


@dataclass
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int
    duration: float


def split_text_into_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    chunks = []
    paragraphs = re.split(r"\n\s*\n", text)

    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= max_chars:
            current_chunk += ("\n\n" if current_chunk else "") + para
        else:
            if current_chunk:
                chunks.append(current_chunk)
            if len(para) <= max_chars:
                current_chunk = para
            else:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                current_chunk = ""
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 1 <= max_chars:
                        current_chunk += (" " if current_chunk else "") + sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def text_to_script_format(text: str, voice: Optional[str] = None) -> str:
    """Convert plain text to 'Speaker N:' format expected by VibeVoice.

    VibeVoice only accepts 'Speaker 1:', 'Speaker 2:', etc. as speaker labels.
    Custom voice names are not supported by the model.
    """
    lines = text.strip().split("\n")
    script_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not re.match(r"^Speaker\s+\d+\s*:", line, re.IGNORECASE):
            line = f"Speaker 1: {line}"
        script_lines.append(line)
    return "\n".join(script_lines)


def _extract_audio_from_output(output, verbose: bool = False) -> list[np.ndarray]:
    """Extract audio segments from model output."""
    audio_segments = []

    if hasattr(output, "speech_outputs") and output.speech_outputs:
        for audio_tensor in output.speech_outputs:
            if audio_tensor is not None:
                if isinstance(audio_tensor, torch.Tensor):
                    # Convert to float32 before numpy (bfloat16 not supported by numpy)
                    audio = audio_tensor.cpu().float().numpy()
                else:
                    audio = np.array(audio_tensor)
                audio = audio.flatten().astype(np.float32)
                if len(audio) > 0:
                    max_val = max(abs(audio.max()), abs(audio.min()))
                    if max_val > 1.0:
                        audio = audio / max_val
                    audio_segments.append(audio)
    elif verbose:
        available_attrs = [a for a in dir(output) if not a.startswith("_")]
        print(f"  Warning: No speech_outputs. Available attributes: {available_attrs}")

    return audio_segments


def synthesize_text(
    text: str,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    voice: Optional[str] = None,
    voice_sample: Optional[str] = None,
    max_duration: int = 3600,
    config: Optional[Config] = None,
    verbose: bool = False,
) -> SynthesisResult:
    """Synthesize text to speech.

    Args:
        text: Text to synthesize
        model_name: Model name to use
        device: Device to use (cpu, cuda, auto)
        voice: Voice name (deprecated, use voice_sample)
        voice_sample: Path to voice sample file or voice preset name (e.g., 'en-Emma')
        max_duration: Maximum audio duration in seconds
        config: Configuration object
        verbose: Print progress
    """
    from .voices import load_voice_sample

    cfg = config or Config.load()
    actual_device = get_device(device)
    model, processor, is_realtime = load_model(model_name, device, cfg, verbose)

    # Handle voice sample - load as numpy array to avoid PyTorch 2.6 weights_only issue
    voice_samples = None
    if voice_sample:
        try:
            voice_array = load_voice_sample(voice_sample, verbose=verbose)
            voice_samples = [voice_array]
            if verbose:
                print(f"Using voice: {voice_sample}")
        except Exception as e:
            if verbose:
                print(f"Warning: Could not load voice '{voice_sample}': {e}")

    chunks = split_text_into_chunks(text)
    audio_segments = []

    if verbose:
        model_type = "realtime" if is_realtime else "standard"
        print(f"Processing {len(chunks)} text chunk(s) with {model_type} model...")

    try:
        import copy

        for i, chunk in enumerate(chunks):
            if verbose:
                print(f"  Generating chunk {i + 1}/{len(chunks)}...")

            if is_realtime:
                # Realtime model doesn't need "Speaker N:" prefix - pass text directly
                script_text = chunk
            else:
                script_text = text_to_script_format(chunk)

            if is_realtime:
                # Streaming model uses different input preparation
                # Determine model dtype
                model_dtype = torch.bfloat16 if actual_device == "cuda" else torch.float32

                # Load voice embeddings if available
                all_prefilled_outputs = None
                if voice_sample:
                    from .voices import get_voice_preset_path
                    voice_path = get_voice_preset_path(voice_sample)
                    if voice_path and voice_path.exists():
                        all_prefilled_outputs = torch.load(voice_path, weights_only=False)

                if all_prefilled_outputs is not None:
                    inputs = processor.process_input_with_cached_prompt(
                        text=script_text,
                        cached_prompt=all_prefilled_outputs,
                        padding=True,
                        return_tensors="pt",
                        return_attention_mask=True,
                    )
                else:
                    inputs = processor(text=script_text, return_tensors="pt")

                # Move all tensors to device with correct dtype
                for k, v in inputs.items():
                    if torch.is_tensor(v):
                        if v.dtype in (torch.float32, torch.float16, torch.bfloat16):
                            inputs[k] = v.to(device=actual_device, dtype=model_dtype)
                        else:
                            inputs[k] = v.to(actual_device)

                with torch.no_grad():
                    gen_kwargs = {
                        **inputs,
                        "tokenizer": processor.tokenizer,
                        "cfg_scale": 1.5,
                        "generation_config": {"do_sample": False},
                        "verbose": verbose,
                    }
                    if all_prefilled_outputs is not None:
                        # Convert cached prompt tensors to correct dtype
                        prefilled_copy = copy.deepcopy(all_prefilled_outputs)
                        for k, v in prefilled_copy.items():
                            if torch.is_tensor(v):
                                if v.dtype in (torch.float32, torch.float16, torch.bfloat16):
                                    prefilled_copy[k] = v.to(device=actual_device, dtype=model_dtype)
                                else:
                                    prefilled_copy[k] = v.to(actual_device)
                        gen_kwargs["all_prefilled_outputs"] = prefilled_copy

                    output = model.generate(**gen_kwargs)
            else:
                # Standard model
                inputs = processor(text=script_text, voice_samples=voice_samples, return_tensors="pt")

                input_ids = inputs["input_ids"].to(actual_device)
                attention_mask = inputs.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(actual_device)
                speech_input_mask = inputs.get("speech_input_mask")
                if speech_input_mask is not None:
                    speech_input_mask = speech_input_mask.to(actual_device)

                speech_tensors = inputs.get("speech_tensors")
                if speech_tensors is not None:
                    speech_tensors = speech_tensors.to(actual_device)
                else:
                    speech_tensors = torch.zeros(1, 1, device=actual_device)

                speech_masks = inputs.get("speech_masks")
                if speech_masks is not None:
                    speech_masks = speech_masks.to(actual_device)
                else:
                    speech_masks = torch.zeros(1, 1, dtype=torch.bool, device=actual_device)

                with torch.no_grad():
                    gen_kwargs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "speech_input_mask": speech_input_mask,
                        "speech_tensors": speech_tensors,
                        "speech_masks": speech_masks,
                        "tokenizer": processor.tokenizer,
                        "return_speech": True,
                        "show_progress_bar": verbose,
                        "verbose": verbose,
                    }

                    output = model.generate(**gen_kwargs)

            chunk_audio = _extract_audio_from_output(output, verbose)
            audio_segments.extend(chunk_audio)

            total_duration = sum(len(seg) for seg in audio_segments) / SAMPLE_RATE
            if total_duration >= max_duration:
                if verbose:
                    print(f"  Reached max duration ({max_duration}s), stopping")
                break

        if not audio_segments:
            raise SynthesisError("No audio was generated")

        final_audio = np.concatenate(audio_segments)
        duration = len(final_audio) / SAMPLE_RATE

        if verbose:
            print(f"Generated {duration:.1f}s of audio")

        return SynthesisResult(audio=final_audio, sample_rate=SAMPLE_RATE, duration=duration)
    except Exception as e:
        raise SynthesisError(f"Audio generation failed: {e}") from e


def synthesize_script(
    lines: list[ScriptLine],
    speaker_map: dict[str, str],
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    max_duration: int = 3600,
    config: Optional[Config] = None,
    verbose: bool = False,
) -> SynthesisResult:
    cfg = config or Config.load()
    actual_device = get_device(device)
    model, processor, is_realtime = load_model(model_name, device, cfg, verbose)

    if is_realtime and len(set(line.speaker for line in lines)) > 1:
        raise SynthesisError("Realtime model only supports single-speaker scripts")

    if verbose:
        print(f"Processing {len(lines)} script line(s)...")

    # Map script speakers to Speaker 1, Speaker 2, etc. (VibeVoice requirement)
    speaker_to_id: dict[str, int] = {}
    next_id = 1
    for line in lines:
        if line.speaker not in speaker_to_id:
            speaker_to_id[line.speaker] = next_id
            next_id += 1

    script_text = "\n".join(
        f"Speaker {speaker_to_id[line.speaker]}: {line.text}" for line in lines
    )

    try:
        inputs = processor(text=script_text, return_tensors="pt")

        input_ids = inputs["input_ids"].to(actual_device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(actual_device)
        speech_input_mask = inputs.get("speech_input_mask")
        if speech_input_mask is not None:
            speech_input_mask = speech_input_mask.to(actual_device)

        speech_tensors = inputs.get("speech_tensors")
        if speech_tensors is not None:
            speech_tensors = speech_tensors.to(actual_device)
        else:
            speech_tensors = torch.zeros(1, 1, device=actual_device)

        speech_masks = inputs.get("speech_masks")
        if speech_masks is not None:
            speech_masks = speech_masks.to(actual_device)
        else:
            speech_masks = torch.zeros(1, 1, dtype=torch.bool, device=actual_device)

        with torch.no_grad():
            if is_realtime:
                # Streaming model uses different API
                gen_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "speech_input_mask": speech_input_mask,
                    "speech_tensors": speech_tensors,
                    "speech_masks": speech_masks,
                    "tokenizer": processor.tokenizer,
                    "cfg_scale": 1.5,
                    "generation_config": {"do_sample": False},
                    "verbose": verbose,
                }
            else:
                # Standard model
                gen_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "speech_input_mask": speech_input_mask,
                    "speech_tensors": speech_tensors,
                    "speech_masks": speech_masks,
                    "tokenizer": processor.tokenizer,
                    "return_speech": True,
                    "show_progress_bar": verbose,
                    "verbose": verbose,
                }

            output = model.generate(**gen_kwargs)

        audio_segments = _extract_audio_from_output(output, verbose)

        if not audio_segments:
            raise SynthesisError("No audio was generated")

        final_audio = np.concatenate(audio_segments)
        duration = len(final_audio) / SAMPLE_RATE

        if verbose:
            print(f"Generated {duration:.1f}s of audio")

        return SynthesisResult(audio=final_audio, sample_rate=SAMPLE_RATE, duration=duration)
    except Exception as e:
        raise SynthesisError(f"Script synthesis failed: {e}") from e


def save_audio(result: SynthesisResult, output_path: Path, verbose: bool = False) -> None:
    import wave

    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio_int16 = (result.audio * 32767).astype(np.int16)

    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(result.sample_rate)
        wf.writeframes(audio_int16.tobytes())

    if verbose:
        print(f"Saved audio to {output_path}")
