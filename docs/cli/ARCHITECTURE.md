# CLI Architecture

## Entry Point

The CLI entry point is defined in [src/newspaper_explorer/main.py](../../src/newspaper_explorer/main.py):

```python
@click.group()
def cli() -> None:
    """Newspaper Explorer - Explore and analyze historical newspaper data."""
    pass
```

Registered via `pyproject.toml`:
```toml
[project.scripts]
newspaper-explorer = "newspaper_explorer.main:cli"
```

## Command Groups

The CLI has three main command groups:

1. **data** - Data management (download, load, preprocess)
2. **analyze** - Analysis tasks (entities, keywords, topics, emotions, layout)
3. **ui** - User interface commands

### Structure

```
cli/
├── main.py              # Entry point: cli()
├── data/
│   ├── commands.py      # Main data group + registration
│   ├── download.py      # register_download_commands()
│   ├── images.py        # register_image_commands()
│   ├── info.py          # register_info_commands()
│   ├── loading.py       # register_loading_commands()
│   ├── preprocessing.py # register_preprocessing_commands()
│   └── validation.py    # register_validation_commands()
├── analyze/
│   ├── commands.py      # Main analyze group + imports
│   ├── emotions/
│   │   └── commands.py  # emotions_group
│   ├── entities/
│   │   └── commands.py  # entities_group
│   ├── keywords/
│   │   └── commands.py  # keywords_group
│   ├── layout/
│   │   └── commands.py  # layout_group
│   ├── topics/
│   │   └── commands.py  # topics_group
│   └── captions/
│       └── commands.py  # captions_group
└── ui/
    └── commands.py      # ui_commands
```

## Current Organization: Two Patterns (To Be Unified)

### Legacy Pattern: Registration (Data CLI)

**Current implementation** (will be replaced):
```python
# cli/data/commands.py
from .download import register_download_commands

@click.group()
def data():
    """Manage newspaper data."""
    pass

register_info_commands(data)  # Legacy pattern
```

```python
# cli/data/download.py
def register_download_commands(data_group):
    @data_group.command()
    def download(...):
        """Download newspaper data."""
        pass
```

**Issues**:
- Inconsistent with analyze CLI
- Less flexible for grouping
- Commands defined inside functions
- Not the Click recommended approach

### Recommended Pattern: Group Nesting (All CLI)

**Implementation**:
```python
# cli/data/commands.py
from .download import download_commands  # Import group

@click.group()
def data():
    """Manage newspaper data."""
    pass

data.add_command(download_commands)  # Unified pattern
```

**For flat commands** (like current data CLI):
```python
# cli/data/download.py
@click.group(name="download")
def download_commands():
    """Download-related commands."""
    pass

@download_commands.command(name="sources")
def download_sources(...):
    """Download source data."""
    pass

@download_commands.command(name="images")
def download_images(...):
    """Download images."""
    pass
```

**For nested domains** (like analyze CLI):
```python
# cli/analyze/emotions/commands.py
@click.group(name="emotions")
def emotions_group():
    """Emotion analysis commands."""
    pass

@emotions_group.command(name="predict")
def predict(...):
    """Run emotion prediction."""
    pass
```

**Benefits of unified pattern**:
- ✅ Consistent everywhere
- ✅ More flexible (works for flat and nested)
- ✅ Click recommended approach
- ✅ Easier to extend and maintain
- ✅ Groups can have their own --help

## Current Problem: Embedded Logic

**Both patterns have logic embedded in commands files.**

Example from [cli/data/download.py](../../src/newspaper_explorer/cli/data/download.py#L50-L95):
```python
def download(part, parts, download_all, force, ...):
    """Download newspaper data parts."""

    # 45 lines of logic inside CLI command:
    # - Logging configuration
    # - Part name parsing
    # - CSV parsing
    # - Error handling
    # - Conditional workflows

    downloader = ZenodoDownloader()

    if download_all:
        part_names = None
        click.echo("Downloading ALL dataset parts...")
    elif part or parts:
        part_names = []
        if part:
            part_names.append(part)
        if parts:
            reader = csv.reader(io.StringIO(parts))
            for row in reader:
                part_names.extend([p.strip() for p in row if p.strip()])
        # ... more logic
```

**Why this is problematic**:
- **Not testable**: CLI commands are hard to unit test
- **Not reusable**: Logic can't be used programmatically
- **Violates SRP**: Commands handle both presentation and business logic
- **Hard to maintain**: Complex logic mixed with Click decorators

## Solution: Extract Logic to Utilities

### Proper Separation

**CLI Layer** (commands.py):
- Click decorators and options
- Argument validation (Click-level)
- Output formatting (click.echo, Rich)
- User interaction (prompts, confirmations)
- Progress display (tqdm)

**Utility Layer** (utils.py or core modules):
- Business logic
- Data validation
- Processing orchestration
- Error handling
- Return structured data

### Example Refactoring

**Before** - Embedded logic:
```python
# cli/analyze/emotions/commands.py (old)
@emotions_group.command(name="models")
def models():
    """Check emotion model availability."""

    # 60 lines of CUDA checking, file validation, status building
    import torch
    cuda_info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        ...
    }
    # ... more logic
    click.echo(...)
```

**After** - Clean separation:
```python
# cli/analyze/emotions/commands.py (new)
from newspaper_explorer.analyze.emotions.utils import get_model_status

@emotions_group.command(name="models")
@click.option("--model-dir", default="models/emotions")
def models(model_dir):
    """Check emotion model availability."""
    status = get_model_status(model_dir)  # Logic in utility

    # CLI just presents the data
    click.echo(f"CUDA: {status['cuda']['cuda_available']}")
    for emotion, available in status['models'].items():
        click.echo(f"  {emotion}: {'✓' if available else '✗'}")
```

```python
# analyze/emotions/utils.py (new)
from newspaper_explorer.data.utils.ml import check_cuda_available

def get_model_status(model_dir: str) -> dict[str, Any]:
    """Check emotion model availability and CUDA status.

    Returns:
        Dictionary with 'cuda' and 'models' keys containing status info.
    """
    cuda_info = check_cuda_available()
    models = check_emotion_models(model_dir)

    return {
        "cuda": cuda_info,
        "models": models,
        # ... structured data
    }
```

## Unified Pattern: Group Nesting

**Use Group Nesting pattern for ALL CLI modules** (data and analyze):

**Why**:
- More extensible (easy to add subcommands)
- Better organization (logical grouping)
- Clearer hierarchy in --help output
- Consistent with Click best practices

**How**:
1. Each domain gets its own group (`emotions_group`, `entities_group`)
2. Extract all logic to utilities or core modules
3. CLI commands stay thin - just presentation layer
4. Parent group imports and registers child groups

### Standard Structure

```
cli/
└── analyze/
    └── {domain}/
        ├── commands.py     # Click decorators only
        └── (no utils.py)   # Logic lives in analyze/{domain}/

analyze/
└── {domain}/
    ├── core.py            # Main processing logic
    └── utils.py           # Helper functions
```

**Commands file template**:
```python
"""CLI commands for {domain} analysis."""

import click
from newspaper_explorer.analyze.{domain}.utils import process_{task}

@click.group(name="{domain}")
def {domain}_group():
    """{Domain} analysis commands."""
    pass

@{domain}_group.command(name="{command}")
@click.option("--source", required=True)
@click.option("--option", default="value")
def {command}(source, option):
    """Do {task}."""
    result = process_{task}(source, option)  # Logic in utility

    # Presentation only
    if result["success"]:
        click.echo(f"Processed {result['count']} items")
    else:
        click.echo(f"Error: {result['error']}", err=True)
```

## Migration Strategy

### Phase 1: Unify CLI Patterns
1. **Convert data CLI to group nesting** - replace `register_*_commands()` pattern
2. **Keep command structure the same** - `newspaper-explorer data download` stays the same
3. **Update imports in cli/data/commands.py** - import groups instead of registration functions

### Phase 2: Extract Embedded Logic
1. **Start with emotions** - smallest module, already partially clean
2. **Extract logic to utilities** - move business logic out of CLI commands
3. **Add tests** - utilities are testable, commands are not
4. **Update one domain at a time** - emotions → keywords → entities → topics → layout

### Phase 3: Documentation
1. **Update CLI reference docs** - reflect new unified structure
2. **Add examples** - show proper separation of concerns

## See Also

- [CLI Commands Reference](README.md) - Complete command documentation
- [Output Standards](../../docs/OUTPUT_STANDARDS.md) - CLI output guidelines
- [Development Guide](../../.github/copilot-instructions.md) - Code style and patterns
