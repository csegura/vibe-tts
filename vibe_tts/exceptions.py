"""Custom exception types for vibe-tts."""


class VibeTTSError(Exception):
    """Base exception for vibe-tts errors."""

    exit_code: int = 1


class InvalidArgumentError(VibeTTSError):
    """Raised when CLI arguments are invalid."""

    exit_code = 1


class ModelLoadError(VibeTTSError):
    """Raised when model initialization or loading fails."""

    exit_code = 2


class SynthesisError(VibeTTSError):
    """Raised when audio generation fails."""

    exit_code = 3


class PlaybackError(VibeTTSError):
    """Raised when audio playback fails."""

    exit_code = 4


class ServerError(VibeTTSError):
    """Raised when server-related operations fail."""

    exit_code = 5
