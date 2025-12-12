"""CLI entrypoint for vibe-tts."""

import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from . import __version__
from .config import MODELS, Config
from .exceptions import VibeTTSError
from .playback import play_audio
from .script_parser import get_speakers, parse_script_file, parse_speaker_map
from .synthesis import save_audio, synthesize_script, synthesize_text

app = typer.Typer(
    name="vibe-tts",
    help="Text-to-speech CLI using Microsoft VibeVoice",
    add_completion=False,
    no_args_is_help=False,
)
console = Console()
err_console = Console(stderr=True)

DISCLAIMER = """
[yellow]Note:[/yellow] This tool uses AI-generated speech.
Do not use it to impersonate others or mislead people.
"""


def show_disclaimer() -> None:
    config_dir = Config().default_output_dir.parent
    disclaimer_shown = (config_dir / ".vibe-tts-disclaimer-shown").exists() if config_dir else False
    if not disclaimer_shown:
        console.print(DISCLAIMER)


def get_default_output() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path.cwd() / f"tts-output-{timestamp}.wav"


def version_callback(value: bool) -> None:
    if value:
        console.print(f"vibe-tts version {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    text: Annotated[
        Optional[str],
        typer.Option("--text", "-t", help="Text to synthesize"),
    ] = None,
    file: Annotated[
        Optional[Path],
        typer.Option("--file", "-f", help="Path to text file to read"),
    ] = None,
    script: Annotated[
        Optional[Path],
        typer.Option("--script", "-s", help="Path to multi-speaker script"),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output WAV file path"),
    ] = None,
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Model to use (vibe-1.5b, realtime-0.5b)"),
    ] = "vibe-1.5b",
    device: Annotated[
        str,
        typer.Option("--device", "-d", help="Device to use (auto, cpu, cuda)"),
    ] = "auto",
    voice: Annotated[
        Optional[str],
        typer.Option("--voice", help="Voice preset name (e.g., 'en-Emma', 'en-Carter') or path to .pt file"),
    ] = None,
    speaker_map: Annotated[
        Optional[str],
        typer.Option("--speaker-map", help="Speaker to voice mapping (e.g., 'Alice=v1,Bob=v2')"),
    ] = None,
    play: Annotated[
        bool,
        typer.Option("--play", "-p", help="Play audio after generation"),
    ] = False,
    max_duration: Annotated[
        int,
        typer.Option("--max-duration", help="Maximum audio duration in seconds"),
    ] = 3600,
    config_path: Annotated[
        Optional[Path],
        typer.Option("--config", help="Path to config file"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed progress"),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Show only errors"),
    ] = False,
    version: Annotated[
        Optional[bool],
        typer.Option("--version", callback=version_callback, is_eager=True),
    ] = None,
) -> None:
    """Convert text to speech using Microsoft VibeVoice."""
    if ctx.invoked_subcommand is not None:
        return

    try:
        if not quiet:
            show_disclaimer()

        config = Config.load(config_path)

        input_text: Optional[str] = None
        is_script = False

        if text:
            input_text = text
        elif file:
            if not file.exists():
                err_console.print(f"[red]Error:[/red] File not found: {file}")
                raise typer.Exit(1)
            input_text = file.read_text(encoding="utf-8")
        elif script:
            is_script = True
        elif not sys.stdin.isatty():
            input_text = sys.stdin.read()
        else:
            err_console.print(
                "[red]Error:[/red] No input provided. Use --text, --file, --script, or pipe text."
            )
            raise typer.Exit(1)

        if not input_text and not is_script:
            err_console.print("[red]Error:[/red] Empty input")
            raise typer.Exit(1)

        output_path = output or get_default_output()

        if verbose:
            console.print(f"Model: {model} ({MODELS.get(model, model)})")
            console.print(f"Device: {device}")
            console.print(f"Output: {output_path}")

        if is_script:
            assert script is not None
            lines = parse_script_file(script)
            speakers = get_speakers(lines)

            voice_map: dict[str, str] = {}
            if speaker_map:
                voice_map = parse_speaker_map(speaker_map)

            missing = speakers - set(voice_map.keys())
            if missing and verbose:
                console.print(f"[yellow]Warning:[/yellow] No voice mapping for: {missing}")

            result = synthesize_script(
                lines=lines,
                speaker_map=voice_map,
                model_name=model,
                device=device if device != "auto" else None,
                max_duration=max_duration,
                config=config,
                verbose=verbose,
            )
        else:
            assert input_text is not None
            result = synthesize_text(
                text=input_text,
                model_name=model,
                device=device if device != "auto" else None,
                voice_sample=voice or config.default_voice,
                max_duration=max_duration,
                config=config,
                verbose=verbose,
            )

        save_audio(result, output_path, verbose=verbose)

        if not quiet:
            console.print(f"[green]Audio saved:[/green] {output_path} ({result.duration:.1f}s)")

        if play:
            play_audio(
                result.audio,
                result.sample_rate,
                player_command=config.player_command,
                verbose=verbose,
            )

    except VibeTTSError as e:
        err_console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(e.exit_code)
    except KeyboardInterrupt:
        err_console.print("\n[yellow]Interrupted[/yellow]")
        raise typer.Exit(130)


@app.command("list-voices")
def list_voices() -> None:
    """List available voice presets."""
    from .voices import VOICE_PRESETS, get_voices_dir

    console.print("[bold]Available voice presets:[/bold]\n")

    voices_dir = get_voices_dir()
    cached_voices = set()
    if voices_dir.exists():
        cached_voices = {p.stem for p in voices_dir.glob("*.pt")}

    # Group by language
    languages = {}
    for short_name, full_name in VOICE_PRESETS.items():
        lang = short_name.split("-")[0]
        if lang not in languages:
            languages[lang] = []
        is_cached = full_name in cached_voices
        languages[lang].append((short_name, full_name, is_cached))

    lang_names = {
        "en": "English",
        "de": "German",
        "fr": "French",
        "it": "Italian",
        "jp": "Japanese",
        "kr": "Korean",
        "nl": "Dutch",
        "pl": "Polish",
        "pt": "Portuguese",
        "sp": "Spanish",
        "in": "Indian English",
    }

    for lang, voices in sorted(languages.items()):
        lang_name = lang_names.get(lang, lang.upper())
        console.print(f"[bold]{lang_name}:[/bold]")
        for short_name, full_name, is_cached in voices:
            status = "[green]cached[/green]" if is_cached else "[dim]not cached[/dim]"
            console.print(f"  {short_name:12} ({full_name}) {status}")
        console.print()

    console.print("[yellow]Usage:[/yellow] vibe-tts --voice en-Emma --text 'Hello world'")
    console.print("Voice presets are downloaded automatically on first use.")


@app.command("info")
def info() -> None:
    """Show information about the installation."""
    import torch

    console.print(f"[bold]vibe-tts[/bold] version {__version__}\n")

    console.print("[bold]Available models:[/bold]")
    for name, model_id in MODELS.items():
        console.print(f"  {name}: {model_id}")

    console.print(f"\n[bold]PyTorch version:[/bold] {torch.__version__}")
    console.print(f"[bold]CUDA available:[/bold] {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        console.print(f"[bold]CUDA device:[/bold] {torch.cuda.get_device_name(0)}")

    config = Config.load()
    console.print(f"\n[bold]Config file:[/bold] {Config().get_config_path() if hasattr(Config, 'get_config_path') else 'default'}")
    console.print(f"[bold]Default model:[/bold] {config.default_model}")
    console.print(f"[bold]Default voice:[/bold] {config.default_voice or '(none)'}")


@app.command("serve")
def serve(
    host: Annotated[
        str,
        typer.Option("--host", "-h", help="Host to bind to"),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="Port to bind to"),
    ] = 8000,
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Model to use"),
    ] = "vibe-1.5b",
    device: Annotated[
        str,
        typer.Option("--device", "-d", help="Device (auto, cpu, cuda)"),
    ] = "auto",
    voice: Annotated[
        Optional[str],
        typer.Option("--voice", help="Default voice preset (e.g., 'en-Emma')"),
    ] = None,
    reload: Annotated[
        bool,
        typer.Option("--reload", help="Enable auto-reload for development"),
    ] = False,
    workers: Annotated[
        int,
        typer.Option("--workers", "-w", help="Number of workers (default: 1)"),
    ] = 1,
) -> None:
    """Start the HTTP/WebSocket server."""
    try:
        import uvicorn
    except ImportError:
        err_console.print(
            "[red]Error:[/red] uvicorn not installed. "
            "Install with: pip install uvicorn"
        )
        raise typer.Exit(1)

    # Pre-download voice preset if specified
    if voice:
        from .voices import ensure_voice
        try:
            console.print(f"[dim]Pre-downloading voice preset: {voice}[/dim]")
            ensure_voice(voice, verbose=True)
        except Exception as e:
            err_console.print(f"[yellow]Warning:[/yellow] Could not download voice '{voice}': {e}")

    if workers > 1:
        console.print(
            "[yellow]Warning:[/yellow] Using multiple workers will load "
            "the model separately in each worker, increasing memory usage."
        )

    console.print("[bold]Starting vibe-tts server[/bold]")
    console.print(f"  Host: {host}")
    console.print(f"  Port: {port}")
    console.print(f"  Model: {model}")
    console.print(f"  Device: {device}")
    if voice:
        console.print(f"  Default voice: {voice}")
    console.print(f"  Workers: {workers}")
    if reload:
        console.print("  [yellow]Reload mode enabled[/yellow]")

    if reload:
        uvicorn.run(
            "vibe_tts.server.app:create_app",
            factory=True,
            host=host,
            port=port,
            reload=reload,
        )
    else:
        from .server.app import create_app

        config = Config.load()
        app_instance = create_app(
            model_name=model,
            device=device,
            config=config,
        )
        uvicorn.run(
            app_instance,
            host=host,
            port=port,
            workers=workers,
        )


if __name__ == "__main__":
    app()
