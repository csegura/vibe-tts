# vibe-tts

A cross-platform command-line tool for text-to-speech using Microsoft VibeVoice.

## Features

- Convert text to natural-sounding speech using VibeVoice 1.5B model
- Multiple input methods: inline text, files, or stdin pipes
- Multi-speaker script support for dialogues and podcasts
- **Server mode** with HTTP REST API and WebSocket streaming
- Cross-platform audio playback
- GPU acceleration with CUDA support
- Configurable via TOML config files

## Installation

### Prerequisites

- Python 3.10+
- [VibeVoice](https://github.com/csegura/vibe-tts.git) package
- PyTorch (CUDA optional for GPU acceleration)

### Install with uv (recommended)

```bash
# Clone the repository
git clone https://github.com/csegura/vibe-tts.git
cd vibe-tts

# Install dependencies and package
uv sync

# Run the CLI
uv run vibe-tts --help
```

### Install with pip

```bash
pip install -e .
```

### Install VibeVoice

```bash
# Clone and install vibevoice
git clone https://github.com/csegura/vibe-tts.git
cd VibeVoice
pip install -e .
```

## Usage

### Basic Text-to-Speech

```bash
# From inline text
uv run vibe-tts --text "Hello, this is VibeVoice speaking." --output hello.wav

# From a text file
uv run vibe-tts --file article.txt --output article.wav

# From stdin (pipe mode)
cat document.txt | uv run vibe-tts --output document.wav

# With verbose output
uv run vibe-tts --text "Hello world" --output hello.wav -v
```

### Play Audio Immediately

```bash
uv run vibe-tts --text "Hello world" --output hello.wav --play
```

### Multi-Speaker Scripts

Create a script file `dialog.txt` with any speaker names you like:

```text
Alice: Hello, how are you today?
Bob: I'm doing great, thanks for asking!
Alice: That's wonderful to hear.
```

Generate speech:

```bash
uv run vibe-tts --script dialog.txt --output dialog.wav
```

Speaker names are automatically mapped to VibeVoice's `Speaker 1`, `Speaker 2`, etc. format (up to 4 speakers supported).

### Voice Presets

Use the `--voice` option to select a specific voice preset:

```bash
# Use Emma's voice (English, female)
uv run vibe-tts --text "Hello world" --voice en-Emma --output hello.wav

# Use Carter's voice (English, male)
uv run vibe-tts --text "Hello world" --voice en-Carter --output hello.wav

# List all available voices
uv run vibe-tts list-voices
```

Available voice presets:

| Language | Voices |
|----------|--------|
| English | en-Emma, en-Carter, en-Davis, en-Frank, en-Grace, en-Mike |
| German | de-Spk0, de-Spk1 |
| French | fr-Spk0, fr-Spk1 |
| Spanish | sp-Spk0, sp-Spk1 |
| Italian | it-Spk0, it-Spk1 |
| Japanese | jp-Spk0, jp-Spk1 |
| Korean | kr-Spk0, kr-Spk1 |
| Dutch | nl-Spk0, nl-Spk1 |
| Polish | pl-Spk0, pl-Spk1 |
| Portuguese | pt-Spk0, pt-Spk1 |
| Indian English | in-Samuel |

Voice presets are downloaded automatically on first use and cached locally.

### Model and Device Selection

```bash
# Use the default 1.5B model (recommended for quality)
uv run vibe-tts --text "Hello" --model vibe-1.5b

# Use the realtime 0.5B model (faster, lower quality)
uv run vibe-tts --text "Hello" --model realtime-0.5b

# Force CPU execution
uv run vibe-tts --text "Hello" --device cpu

# Use GPU (CUDA)
uv run vibe-tts --text "Hello" --device cuda
```

### Available Commands

```bash
# Show help
uv run vibe-tts --help

# Show version
uv run vibe-tts --version

# Show system info (PyTorch version, CUDA availability, etc.)
uv run vibe-tts info

# List available voices
uv run vibe-tts list-voices

# Start HTTP/WebSocket server
uv run vibe-tts serve --port 8000
```

## Server Mode

Server mode keeps the model loaded in memory, eliminating the 20-30 second model load time for each request.

### Start the Server

```bash
# Basic (localhost only)
uv run vibe-tts serve

# Allow external connections
uv run vibe-tts serve --host 0.0.0.0 --port 8000

# With specific model and device
uv run vibe-tts serve --model vibe-1.5b --device cuda

# Development mode with auto-reload
uv run vibe-tts serve --reload
```

### Server Options

| Option | Short | Description |
|--------|-------|-------------|
| `--host` | `-h` | Host to bind to (default: `127.0.0.1`) |
| `--port` | `-p` | Port to bind to (default: `8000`) |
| `--model` | `-m` | Model to use |
| `--device` | `-d` | Device (`auto`, `cpu`, `cuda`) |
| `--reload` | | Enable auto-reload for development |
| `--workers` | `-w` | Number of workers (default: `1`) |

### HTTP API

#### Synthesize Text

```bash
curl -X POST http://localhost:8000/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!", "max_duration": 60}' \
  --output hello.wav
```

#### Synthesize Multi-Speaker Script

```bash
curl -X POST http://localhost:8000/synthesize-script \
  -H "Content-Type: application/json" \
  -d '{
    "script": "Alice: Hello!\nBob: Hi there!",
    "speaker_map": {}
  }' \
  --output dialog.wav
```

#### Health Check

```bash
curl http://localhost:8000/health
# {"status":"healthy","model_loaded":true}
```

#### Server Info

```bash
curl http://localhost:8000/info
# {"version":"0.1.0","model":"vibe-1.5b","device":"cuda",...}
```

### WebSocket Streaming

Connect to `ws://localhost:8000/stream` for real-time audio streaming.

**Send a synthesis request:**
```json
{"type": "synthesize", "text": "Hello, world!"}
```

**Receive messages:**
- `{"type": "start", "sample_rate": 24000, "format": "pcm_s16le"}` - Synthesis started
- Binary frames with raw PCM audio chunks
- `{"type": "progress", "chunk": 1, "total_chunks": 3}` - Progress updates
- `{"type": "complete", "duration": 2.5, "total_bytes": 120000}` - Done

**Cancel synthesis:**
```json
{"type": "cancel"}
```

### Python Client Example

```python
import requests

response = requests.post(
    "http://localhost:8000/synthesize",
    json={"text": "Hello from Python!"}
)
with open("output.wav", "wb") as f:
    f.write(response.content)
```

## CLI Options

| Option | Short | Description |
|--------|-------|-------------|
| `--text` | `-t` | Text to synthesize |
| `--file` | `-f` | Path to text file to read |
| `--script` | `-s` | Path to multi-speaker script |
| `--output` | `-o` | Output WAV file path |
| `--model` | `-m` | Model to use (`vibe-1.5b`, `realtime-0.5b`) |
| `--device` | `-d` | Device to use (`auto`, `cpu`, `cuda`) |
| `--voice` | | Voice/speaker ID |
| `--speaker-map` | | Speaker to voice mapping |
| `--play` | `-p` | Play audio after generation |
| `--max-duration` | | Maximum audio duration in seconds |
| `--config` | | Path to config file |
| `--verbose` | `-v` | Show detailed progress |
| `--quiet` | `-q` | Show only errors |

## Configuration

Create a config file at:
- **Linux/macOS**: `~/.config/vibe-tts/config.toml`
- **Windows**: `%APPDATA%\vibe-tts\config.toml`

```toml
default_model = "vibe-1.5b"
default_voice = "default"
default_output_dir = "~/audio"
player_command = "mpv"
max_duration = 3600
```

CLI flags always override config file values.

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Invalid arguments/usage |
| 2 | Model initialization/load failure |
| 3 | Audio generation failure |
| 4 | Playback failure |
| 5 | Server error |

## Requirements

### Hardware

- **CPU**: Works on any modern CPU (slower generation)
- **GPU**: NVIDIA GPU with CUDA support recommended for faster generation
- **RAM**: ~16GB recommended for the 1.5B model

### Software

- Python 3.10+
- PyTorch 2.0+
- vibevoice package

## Docker

Run vibe-tts server in a Docker container with optional GPU support.

### Build the Image

```bash
docker build -t vibe-tts .
```

### Run the Container

```bash
# CPU mode (default)
docker run -p 8000:8000 -e VIBE_DEVICE=cpu vibe-tts

# GPU mode (requires nvidia-docker)
docker run --gpus all -p 8000:8000 -e VIBE_DEVICE=cuda vibe-tts

# Use smaller model (less memory)
docker run -p 8000:8000 -e VIBE_DEVICE=cpu -e VIBE_MODEL=realtime-0.5b vibe-tts
```

### Using Docker Compose

```bash
# Default settings
docker compose up

# With custom settings
VIBE_DEVICE=cpu VIBE_MODEL=realtime-0.5b docker compose up
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VIBE_DEVICE` | Device to use (`auto`, `cpu`, `cuda`) | `auto` |
| `VIBE_MODEL` | Model to use (`vibe-1.5b`, `realtime-0.5b`) | `vibe-1.5b` |

### Memory Requirements

| Model | GPU VRAM | System RAM |
|-------|----------|------------|
| `vibe-1.5b` | ~10-16GB | ~16GB |
| `realtime-0.5b` | ~4-6GB | ~8GB |

If you encounter CUDA out of memory errors, use `VIBE_DEVICE=cpu` or switch to `realtime-0.5b`.

## License

MIT

## Acknowledgments

- [Microsoft VibeVoice](https://github.com/csegura/vibe-tts.git) - The underlying TTS model
