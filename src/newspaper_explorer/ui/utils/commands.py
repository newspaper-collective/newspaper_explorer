"""
UI process management utilities

Manages FastAPI backend and Vue frontend processes for the Historical Newspaper Explorer.
Uses a supervisor-style approach to manage multiple processes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from types import FrameType

from newspaper_explorer.config.base import get_config

logger = logging.getLogger(__name__)


class ProcessState(Enum):
    """Process lifecycle states"""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"


@dataclass
class ProcessConfig:
    """Configuration for a managed process"""

    name: str
    command: list[str]
    cwd: Path | None = None
    env: dict[str, str] = field(default_factory=lambda: {})
    startup_timeout: float = 30.0
    health_check: Callable[[], bool] | None = None
    depends_on: list[str] = field(default_factory=lambda: [])


@dataclass
class ManagedProcess:
    """A managed subprocess with state tracking"""

    config: ProcessConfig
    process: subprocess.Popen[str] | None = None
    state: ProcessState = ProcessState.STOPPED
    exit_code: int | None = None

    @property
    def is_running(self) -> bool:
        """Check if process is currently running"""
        if self.process is None:
            return False
        return self.process.poll() is None

    def start(self) -> bool:
        """Start the process"""
        if self.is_running:
            logger.warning(f"{self.config.name} is already running")
            return True

        self.state = ProcessState.STARTING
        logger.info(f"Starting {self.config.name}...")

        try:
            env = {**os.environ, **self.config.env}
            # Commands are constructed internally by create_backend_config/create_frontend_config,
            self.process = subprocess.Popen(  # noqa: S603
                self.config.command,
                cwd=self.config.cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
                shell=False,  # Explicitly disable shell to prevent injection
            )
            self.state = ProcessState.RUNNING
            logger.info(f"{self.config.name} started (PID: {self.process.pid})")
            return True
        except (OSError, ValueError, subprocess.SubprocessError) as e:
            self.state = ProcessState.FAILED
            logger.error(f"Failed to start {self.config.name}: {e}")
            return False

    def stop(self, timeout: float = 5.0) -> bool:
        """Stop the process gracefully, then forcefully if needed"""
        if not self.is_running or self.process is None:
            self.state = ProcessState.STOPPED
            return True

        self.state = ProcessState.STOPPING
        logger.info(f"Stopping {self.config.name} (PID: {self.process.pid})...")

        try:
            # Try graceful termination first
            self.process.terminate()
            try:
                self.exit_code = self.process.wait(timeout=timeout)
                self.state = ProcessState.STOPPED
                logger.info(f"{self.config.name} stopped gracefully")
                return True
            except subprocess.TimeoutExpired:
                # Force kill
                logger.warning(f"{self.config.name} didn't stop gracefully, killing...")
                self.process.kill()
                self.exit_code = self.process.wait(timeout=2.0)
                self.state = ProcessState.STOPPED
                return True
        except (OSError, subprocess.SubprocessError) as e:
            logger.error(f"Error stopping {self.config.name}: {e}")
            self.state = ProcessState.FAILED
            return False


class ProcessSupervisor:
    """
    Manages multiple processes with dependency ordering and health monitoring.

    Similar to supervisord but implemented in pure Python for simplicity.
    """

    def __init__(self) -> None:
        self.processes: dict[str, ManagedProcess] = {}
        self._shutdown_requested = False

    def register(self, config: ProcessConfig) -> None:
        """Register a process to be managed"""
        self.processes[config.name] = ManagedProcess(config=config)

    def start_all(self) -> bool:
        """Start all registered processes in dependency order"""
        # Simple dependency resolution (topological sort)
        started: set[str] = set()
        to_start = list(self.processes.keys())

        while to_start:
            progress = False
            for name in to_start[:]:
                proc = self.processes[name]
                deps_satisfied = all(dep in started for dep in proc.config.depends_on)

                if deps_satisfied:
                    if proc.start():
                        started.add(name)
                        to_start.remove(name)
                        progress = True
                        # Give process time to initialize
                        time.sleep(0.5)
                    else:
                        logger.error(f"Failed to start {name}")
                        return False

            if not progress and to_start:
                logger.error(f"Circular dependency or missing process: {to_start}")
                return False

        return True

    def stop_all(self) -> None:
        """Stop all processes in reverse dependency order"""
        self._shutdown_requested = True

        # Stop in reverse order of registration
        for name in reversed(list(self.processes.keys())):
            self.processes[name].stop()

    def wait(self) -> None:
        """Wait for all processes, forwarding output"""
        try:
            while not self._shutdown_requested:
                all_stopped = True
                for proc in self.processes.values():
                    if proc.is_running:
                        all_stopped = False
                        # Read and forward output
                        if proc.process and proc.process.stdout:
                            line = proc.process.stdout.readline()
                            if line:
                                # Prefix output with process name
                                sys.stdout.write(f"[{proc.config.name}] {line}")
                                sys.stdout.flush()

                if all_stopped:
                    break

                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Interrupt received, shutting down...")
            self.stop_all()

    def get_status(self) -> dict[str, dict[str, Any]]:
        """Get status of all managed processes"""
        return {
            name: {
                "state": proc.state.value,
                "pid": proc.process.pid if proc.process else None,
                "running": proc.is_running,
                "exit_code": proc.exit_code,
            }
            for name, proc in self.processes.items()
        }


def get_ui_paths() -> dict[str, Path]:
    """Get paths to UI components"""
    ui_root = Path(__file__).parent.parent
    return {
        "ui_root": ui_root,
        "backend": ui_root / "backend",
        "frontend": ui_root / "frontend",
    }


def create_backend_config(
    *,
    host: str = "127.0.0.1",
    port: int = 8005,
    reload: bool = True,
) -> ProcessConfig:
    """Create configuration for the FastAPI backend process"""
    paths = get_ui_paths()

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "newspaper_explorer.ui.backend.main:app",
        "--host",
        host,
        "--port",
        str(port),
    ]

    if reload:
        cmd.append("--reload")

    return ProcessConfig(
        name="backend",
        command=cmd,
        cwd=paths["ui_root"].parent.parent.parent,  # project root
        env={},
        startup_timeout=30.0,
    )


def create_frontend_config(
    *,
    port: int = 7860,
    backend_url: str = "http://127.0.0.1:8005",
) -> ProcessConfig:
    """Create configuration for the Vue frontend process"""
    paths = get_ui_paths()
    frontend_path = paths["frontend"]

    # Check if we have a proper frontend setup
    package_json = frontend_path / "package.json"
    if not package_json.exists():
        # Fall back to serving static files via backend
        logger.warning(
            f"No package.json found at {frontend_path}, "
            "frontend will need to be built and served via backend"
        )
        return ProcessConfig(
            name="frontend",
            command=[sys.executable, "-c", "import time; time.sleep(86400)"],  # placeholder
            env={},
        )

    # Use npm/pnpm to run dev server
    cmd = ["npm", "run", "dev", "--", "--port", str(port)]

    return ProcessConfig(
        name="frontend",
        command=cmd,
        cwd=frontend_path,
        env={
            "VITE_API_URL": backend_url,
        },
        depends_on=["backend"],  # Frontend depends on backend
    )


# Default host - use 0.0.0.0 to allow external connections
DEFAULT_HOST = "0.0.0.0"  # noqa: S104


def start_ui(
    *,
    host: str = DEFAULT_HOST,
    backend_port: int = 8005,
    frontend_port: int = 7860,
    reload: bool = True,
    backend_only: bool = False,
) -> None:
    """
    Start the UI services (backend and optionally frontend)

    Args:
        host: Host to bind to
        backend_port: Port for FastAPI backend
        frontend_port: Port for Vue frontend
        reload: Enable auto-reload for development
        backend_only: Only start the backend (useful if frontend is served separately)
    """
    supervisor = ProcessSupervisor()

    # Register backend
    backend_config = create_backend_config(
        host=host,
        port=backend_port,
        reload=reload,
    )
    supervisor.register(backend_config)

    # Register frontend (unless backend-only mode)
    if not backend_only:
        frontend_config = create_frontend_config(
            port=frontend_port,
            backend_url=f"http://{host}:{backend_port}",
        )
        # Only register if we have a real frontend
        paths = get_ui_paths()
        if (paths["frontend"] / "package.json").exists():
            supervisor.register(frontend_config)
        else:
            logger.info("No frontend package.json found, running backend only")

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum: int, _frame: FrameType | None) -> None:
        logger.info(f"Received signal {signum}, shutting down...")
        supervisor.stop_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start all processes
    logger.info("Starting UI services...")
    if not supervisor.start_all():
        logger.error("Failed to start all services")
        supervisor.stop_all()
        sys.exit(1)

    # Print status
    status = supervisor.get_status()
    logger.info("Services started:")
    for name, info in status.items():
        logger.info(f"  {name}: {info['state']} (PID: {info['pid']})")

    # Wait for processes
    supervisor.wait()


def get_ui_info() -> dict[str, Any]:
    """Get information about the UI configuration and status"""
    config = get_config()
    paths = get_ui_paths()

    frontend_exists = (paths["frontend"] / "package.json").exists()
    backend_exists = (paths["backend"] / "main.py").exists()

    return {
        "framework": {
            "backend": "FastAPI + Uvicorn",
            "frontend": "Vue.js + Vite" if frontend_exists else "Not configured",
        },
        "paths": {
            "ui_root": str(paths["ui_root"]),
            "backend": str(paths["backend"]),
            "frontend": str(paths["frontend"]),
        },
        "config": {
            "data_dir": str(config.data_dir),
            "results_dir": str(config.results_dir),
        },
        "status": {
            "backend_ready": backend_exists,
            "frontend_ready": frontend_exists,
        },
        "endpoints": {
            "api_docs": "/docs",
            "api_health": "/health",
        },
        "features": [
            "Entities - Entity extraction and analysis",
            "Knowledge Graph - Interactive entity relationship network",
            "Images - Image gallery with caption matching",
            "Search - Full-text archive search",
            "Topics - Topic modeling and discovery",
            "Emotions - Emotion analysis over time",
            "Concepts - Historical concept analysis",
        ],
    }
