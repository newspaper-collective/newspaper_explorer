# Newspaper Explorer

A tool for exploring and analyzing historical newspaper data.

## Installation

### Using pip

```bash
pip install -e .
```

### Using uv (recommended for faster installs)

```bash
# Install uv if you haven't already
pip install uv

# Install the package
uv pip install -e .

# Or sync with pyproject.toml
uv sync
```

## Setup

1. Copy the example environment file:

   ```bash
   cp .env.example .env
   ```

2. Edit `.env` to customize your data directories and settings (optional)

3. The default configuration uses:
   - `data/` - Main data directory
   - `data/downloads/` - Downloaded archives (preserved for re-extraction)
   - `data/sources/` - Configuration files with dataset URLs and checksums
   - `data/raw/` - Organized newspaper data by year (temp extraction dirs auto-cleaned)
   - `results/` - Analysis results and outputs

## Quick Start

See main README.md for usage instructions.

## Environment Variables

All settings in `.env` are optional. The system uses sensible defaults:

- `DATA_DIR` - Main data directory (default: `data`)
- `DOWNLOAD_DIR` - Download location (default: `data/downloads`)
- `EXTRACTED_DIR` - Extraction location (default: `data/extracted`)
- `SOURCES_DIR` - Source configs (default: `data/sources`)
- `RESULTS_DIR` - Analysis results (default: `results`)
- `LOG_LEVEL` - Logging level (default: `INFO`)

Note: Download behavior (checksum verification, auto-extraction, error fixes) is controlled via CLI flags, not environment variables.

## Development with UV

```bash
# Create virtual environment with uv
uv venv

# Activate it (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install in editable mode with dev dependencies
uv pip install -e ".[dev]"

# Or use uv sync to install everything from pyproject.toml
uv sync --all-extras
```
