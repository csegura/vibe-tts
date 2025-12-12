"""Multi-speaker script parsing."""

import re
from dataclasses import dataclass
from pathlib import Path

from .exceptions import InvalidArgumentError

SPEAKER_PATTERN = re.compile(r"^(\w+)\s*:\s*(.+)$")


@dataclass
class ScriptLine:
    speaker: str
    text: str
    line_number: int


def parse_script(content: str) -> list[ScriptLine]:
    lines = []
    for i, raw_line in enumerate(content.splitlines(), start=1):
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        match = SPEAKER_PATTERN.match(raw_line)
        if match:
            speaker, text = match.groups()
            lines.append(ScriptLine(speaker=speaker, text=text.strip(), line_number=i))
        else:
            raise InvalidArgumentError(
                f"Line {i}: Invalid script format. Expected 'Speaker: text', got: {raw_line[:50]}"
            )

    if not lines:
        raise InvalidArgumentError("Script file is empty or contains no valid lines")

    return lines


def parse_script_file(path: Path) -> list[ScriptLine]:
    if not path.exists():
        raise InvalidArgumentError(f"Script file not found: {path}")

    content = path.read_text(encoding="utf-8")
    return parse_script(content)


def parse_speaker_map(mapping: str) -> dict[str, str]:
    result = {}
    for pair in mapping.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise InvalidArgumentError(
                f"Invalid speaker mapping: '{pair}'. Expected format: 'Name=voice_id'"
            )
        name, voice = pair.split("=", 1)
        result[name.strip()] = voice.strip()
    return result


def get_speakers(lines: list[ScriptLine]) -> set[str]:
    return {line.speaker for line in lines}
