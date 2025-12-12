"""Configuration loading and management."""

import os
import sys
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

MODELS = {
    "vibe-1.5b": "microsoft/VibeVoice-1.5B",
    "realtime-0.5b": "microsoft/VibeVoice-Realtime-0.5B",
}

DEFAULT_MODEL = "vibe-1.5b"


def get_config_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "vibe-tts"


def get_config_path() -> Path:
    return get_config_dir() / "config.toml"


class Config(BaseModel):
    default_model: str = Field(default=DEFAULT_MODEL)
    default_voice: Optional[str] = Field(default=None)  # VibeVoice uses "Speaker N" format
    default_output_dir: Path = Field(default_factory=Path.cwd)
    player_command: Optional[str] = Field(default=None)
    max_duration: int = Field(default=3600)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Config":
        path = config_path or get_config_path()
        if not path.exists():
            return cls()

        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[import-not-found]

        with open(path, "rb") as f:
            data = tomllib.load(f)

        return cls(**data)

    def get_model_id(self, model_name: Optional[str] = None) -> str:
        name = model_name or self.default_model
        if name in MODELS:
            return MODELS[name]
        return name
