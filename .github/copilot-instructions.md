# GitHub Copilot Instructions for Newspaper Explorer

## Project Structure Guidelines

### Package Organization
- **NO `__init__.py` files**: This project uses explicit imports without `__init__.py` files
- Use explicit module imports: `from newspaper_explorer.utils.config import get_config`
- Never create or suggest creating `__init__.py` files

### Import Style
```python
# ✅ Correct - explicit imports
from newspaper_explorer.utils.config import get_config
from newspaper_explorer.data.download import ZenodoDownloader

# ❌ Avoid - package-level imports that require __init__.py
from newspaper_explorer.utils import get_config
```

### Code Style
- Use **Black** formatter with 100 character line length
- Follow PEP 8 naming conventions
- Use type hints where appropriate (but `disallow_untyped_defs = false`)
- Docstrings for all public functions and classes

### Environment & Configuration
- Configuration via `.env` file using `python-dotenv`
- All paths configurable through environment variables
- Config located in `src/newspaper_explorer/utils/config.py`

### CLI Structure
- Use **Click** for CLI commands
- Commands organized in `src/newspaper_explorer/cli/` by feature
- Main entry point: `src/newspaper_explorer/main.py`
- Each CLI module is a Click command group

### Directory Structure
```
src/newspaper_explorer/
├── main.py              # CLI entry point
├── cli/                 # CLI command modules (no __init__.py)
│   └── data.py         # Data management commands
├── data/               # Data handling (no __init__.py)
│   └── utils.py        # Download/extract utilities
├── utils/              # Utilities (no __init__.py)
│   └── config.py       # Configuration management
└── ui/                 # Future UI components (no __init__.py)
```

### Testing
- Place tests in `tests/` directory at project root
- Mirror the source structure without `__init__.py` files
- Use pytest for testing

### Dependencies
- Core: `requests`, `tqdm`, `click`, `python-dotenv`
- Dev: `pytest`, `black`, `ruff`, `mypy`
- Support both `pip` and `uv` package managers

## Specific Instructions

### When Creating New Modules
1. Create the `.py` file directly in the appropriate subdirectory
2. Do NOT create `__init__.py` files
3. Use explicit imports from the full module path
4. Add to `pyproject.toml` if it's a new top-level package

### When Adding CLI Commands
1. Create new command file in `cli/` directory
2. Import in `main.py` and register with CLI group
3. Use Click decorators and helpers
4. Include helpful docstrings and examples

### When Working with Configuration
1. Use `get_config()` from `newspaper_explorer.utils.config`
2. Add new settings to `.env.example`
3. Document in configuration class docstring

### Data Management
- Downloads go to configurable `DOWNLOAD_DIR` (default: `data/downloads/`)
- Extracted data to `EXTRACTED_DIR` (default: `data/extracted/`)
- Source configs in `SOURCES_DIR` (default: `data/sources/`)
- Always verify MD5 checksums when available

## Code Examples

### Adding a New CLI Command
```python
# In cli/analyze.py
import click

@click.group()
def analyze():
    """Analyze newspaper data."""
    pass

@analyze.command()
def summary():
    """Generate summary statistics."""
    click.echo("Analyzing...")

# In main.py
from newspaper_explorer.cli.analyze import analyze
cli.add_command(analyze)
```

### Using Configuration
```python
from newspaper_explorer.utils.config import get_config

config = get_config()
download_dir = config.download_dir
```

### Type Hints (Optional but Preferred)
```python
from pathlib import Path
from typing import Optional, List

def process_file(path: Path, limit: Optional[int] = None) -> List[str]:
    """Process a file and return results."""
    pass
```

## What NOT to Do
- ❌ Don't create `__init__.py` files
- ❌ Don't use relative imports
- ❌ Don't hardcode paths (use config)
- ❌ Don't create convenience scripts in project root
- ❌ Don't import entire packages when you need specific functions

## Remember
This project favors **explicit over implicit** and **configuration over convention**.
