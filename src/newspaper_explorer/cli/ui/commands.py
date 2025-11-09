"""
CLI commands for running the UI
"""

import click
from click import echo


@click.group(name="ui")
def ui_commands():
    """Web interface commands"""
    pass


@ui_commands.command()
@click.option(
    "--host",
    default="0.0.0.0",
    help="Host to bind to (default: 0.0.0.0)",
    show_default=True,
)
@click.option(
    "--port",
    default=7860,
    type=int,
    help="Port to run on (default: 7860)",
    show_default=True,
)
@click.option(
    "--reload/--no-reload",
    default=True,
    help="Enable auto-reload on code changes",
    show_default=True,
)
def start(host: str, port: int, reload: bool):
    """
    Start the NiceGUI web interface

    Examples:
        # Start with defaults (http://localhost:7860)
        newspaper-explorer ui start

        # Custom port
        newspaper-explorer ui start --port 3000

        # Disable auto-reload (for production)
        newspaper-explorer ui start --no-reload
    """
    echo(f"🚀 Starting Historical Newspaper Explorer UI...")
    echo(f"   Host: {host}")
    echo(f"   Port: {port}")
    echo(f"   Auto-reload: {'enabled' if reload else 'disabled'}")
    echo()
    echo(f"   Open: http://localhost:{port}")
    echo()

    try:
        import subprocess
        import sys
        import os
        from pathlib import Path

        # Get the path to the app.py file
        ui_module_path = Path(__file__).parent.parent.parent / "ui" / "nicegui" / "app.py"

        if not ui_module_path.exists():
            echo(f"❌ Error: UI module not found at {ui_module_path}", err=True)
            raise click.Abort()

        echo("📦 Starting NiceGUI app...")

        # Run the app.py directly as a subprocess to avoid import issues
        env = {
            "NICEGUI_HOST": host,
            "NICEGUI_PORT": str(port),
            "NICEGUI_RELOAD": "1" if reload else "0",
        }

        # Run with current Python
        result = subprocess.run(
            [sys.executable, str(ui_module_path)], env={**os.environ, **env}, cwd=Path.cwd()
        )

        if result.returncode != 0:
            echo(f"❌ App exited with code {result.returncode}", err=True)
            raise click.Abort()

    except ImportError as e:
        echo("❌ Error: NiceGUI not installed", err=True)
        echo("", err=True)
        echo(f"Import error: {e}", err=True)
        echo("Install with: pip install nicegui", err=True)
        raise click.Abort()
    except KeyboardInterrupt:
        echo("\n👋 Shutting down UI...")
    except Exception as e:
        echo(f"❌ Error starting UI: {e}", err=True)
        import traceback

        traceback.print_exc()
        raise


@ui_commands.command()
def info():
    """
    Show UI information and status
    """
    from newspaper_explorer.config.base import get_config

    config = get_config()

    echo("📰 Historical Newspaper Explorer UI")
    echo()
    echo("Framework: NiceGUI (Vue.js + Quasar)")
    echo()
    echo("Configuration:")
    echo(f"  Data directory: {config.data_dir}")
    echo(f"  Results directory: {config.results_dir}")
    echo()
    echo("Available tabs:")
    echo("  • Entities - Entity timeline and distribution analysis")
    echo("  • Knowledge Graph - Interactive entity relationship network")
    echo("  • Images - Image gallery with caption matching")
    echo("  • Search - Full-text archive search")
    echo("  • Topics - Topic modeling and discovery")
    echo("  • Emotions - Emotion analysis over time")
    echo()
    echo("Start the UI with: newspaper-explorer ui start")
