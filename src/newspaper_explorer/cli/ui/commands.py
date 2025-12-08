"""
CLI commands for running the UI

Thin wrapper around ui.utils.commands for process management.
"""

import click
from click import echo

from newspaper_explorer.ui.utils.commands import get_ui_info, start_ui


@click.group(name="ui")
def ui_commands() -> None:
    """Web interface commands"""
    pass


@ui_commands.command()
@click.option(
    "--host",
    default="0.0.0.0",  # noqa: S104
    help="Host to bind to (default: 0.0.0.0)",
    show_default=True,
)
@click.option(
    "--backend-port",
    default=8005,
    type=int,
    help="Port for FastAPI backend (default: 8005)",
    show_default=True,
)
@click.option(
    "--frontend-port",
    default=7860,
    type=int,
    help="Port for Vue frontend (default: 7860)",
    show_default=True,
)
@click.option(
    "--reload/--no-reload",
    default=True,
    help="Enable auto-reload on code changes",
    show_default=True,
)
@click.option(
    "--backend-only",
    is_flag=True,
    default=False,
    help="Only start the backend (useful if running frontend separately)",
)
def start(
    host: str,
    backend_port: int,
    frontend_port: int,
    *,
    reload: bool,
    backend_only: bool,
) -> None:
    """
    Start the Historical Newspaper Explorer UI

    This command starts both the FastAPI backend and Vue frontend servers
    using a process supervisor. Use --backend-only if you're running the
    frontend development server separately (e.g., via npm run dev).

    Examples:
        # Start with defaults (backend: 8005, frontend: 7860)
        newspaper-explorer ui start

        # Custom ports
        newspaper-explorer ui start --backend-port 8000 --frontend-port 3000

        # Backend only (for separate frontend development)
        newspaper-explorer ui start --backend-only

        # Disable auto-reload (for production)
        newspaper-explorer ui start --no-reload
    """

    echo("Starting Historical Newspaper Explorer UI...")
    echo(f"   Host: {host}")
    echo(f"   Backend port: {backend_port}")
    if not backend_only:
        echo(f"   Frontend port: {frontend_port}")
    echo(f"   Auto-reload: {'enabled' if reload else 'disabled'}")
    echo()
    echo(f"   API docs: http://localhost:{backend_port}/docs")
    if not backend_only:
        echo(f"   Frontend: http://localhost:{frontend_port}")
    echo()

    try:
        start_ui(
            host=host,
            backend_port=backend_port,
            frontend_port=frontend_port,
            reload=reload,
            backend_only=backend_only,
        )
    except KeyboardInterrupt:
        echo("\nShutting down UI...")
    except Exception as e:
        echo(f"[ERROR] Error starting UI: {e}", err=True)
        raise click.Abort() from e


@ui_commands.command()
def info() -> None:
    """
    Show UI information and status
    """

    info_data = get_ui_info()

    echo("Historical Newspaper Explorer UI")
    echo()
    echo("Framework:")
    echo(f"  Backend:  {info_data['framework']['backend']}")
    echo(f"  Frontend: {info_data['framework']['frontend']}")
    echo()
    echo("Paths:")
    echo(f"  UI root:  {info_data['paths']['ui_root']}")
    echo(f"  Backend:  {info_data['paths']['backend']}")
    echo(f"  Frontend: {info_data['paths']['frontend']}")
    echo()
    echo("Configuration:")
    echo(f"  Data directory:    {info_data['config']['data_dir']}")
    echo(f"  Results directory: {info_data['config']['results_dir']}")
    echo()
    echo("Status:")
    echo(f"  Backend ready:  {'[OK]' if info_data['status']['backend_ready'] else '[ERROR]'}")
    echo(f"  Frontend ready: {'[OK]' if info_data['status']['frontend_ready'] else '[ERROR]'}")
    echo()
    echo("API Endpoints:")
    echo(f"  Documentation: {info_data['endpoints']['api_docs']}")
    echo(f"  Health check:  {info_data['endpoints']['api_health']}")
    echo()
    echo("Available features:")
    for feature in info_data["features"]:
        echo(f"  • {feature}")
    echo()
    echo("Start the UI with: newspaper-explorer ui start")
