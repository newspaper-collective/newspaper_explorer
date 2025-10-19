# GitHub Copilot Instructions for Newspaper Explorer

## Architecture Overview

**Purpose**: Load, process, and analyze historical newspaper data from ALTO XML archives.

**Data Pipeline**: Zenodo archives → Download/Extract → Raw XML → Parse (ALTO+METS) → Polars DataFrame → Analysis

### Core Components

1. **ZenodoDownloader** (`data/download.py`) - Configuration-driven download/extraction
   - Reads from `data/sources/{source}.json` for URLs, checksums, metadata
   - Organizes archives into `data/raw/{source}/{data_type}/YYYY/` structure
   - Automatic error correction via `DataFixer` for known data issues
   
2. **DataLoader** (`data/loading.py`) - Configuration-driven XML parsing
   - **NEW**: Initialized with `source_name` (e.g., `DataLoader("der_tag")`)
   - Reads source config from `data/sources/{source}.json` for paths and patterns
   - Parallel ALTO XML parsing with METS metadata enrichment
   - Outputs line-level Polars DataFrames to `data/raw/{source}/text/{source}_lines.parquet`
   - Resume functionality: skips already-processed files by default

3. **ALTO/METS Parsers** (`data/utils/`) - XML extraction layer
   - ALTOParser: Extracts text lines with coordinates from fulltext pages
   - METSParser: Extracts issue-level metadata (title, date, volume, page count)
   - Automatic namespace detection for ALTO version compatibility

### Configuration-Driven Pattern

**Source Configuration** (`data/sources/{source}.json`):
```json
{
  "dataset_name": "der_tag",
  "data_type": "xml_ocr",
  "metadata": {
    "newspaper_title": "Der Tag",
    "language": "de",
    "years_available": "1900-1920"
  },
  "loading": {
    "pattern": "**/fulltext/*.xml",
    "compression": "zstd"
  },
  "parts": [...]  // Download parts with URLs, MD5s
}
```

**Why**: Single source of truth enables both download and loading to reference the same configuration. CLI commands work with `--source {name}` instead of raw paths.

## Development Patterns

### Package Organization
- **NO `__init__.py` files** - Use explicit imports
- Import style: `from newspaper_explorer.data.loading import DataLoader`
- Never: `from newspaper_explorer.data import DataLoader`

### CLI Commands (Click)
All commands follow: `newspaper-explorer {group} {command} --option value`

**Data commands** (in `cli/data.py`):
- `download --source der_tag` - Download from Zenodo
- `load --source der_tag` - Parse XML to Parquet
- `sources` - List available sources
- `load-status --source der_tag` - Check processing status
- `status` - Show download/extraction status

### Configuration Management
```python
from newspaper_explorer.utils.config import get_config

config = get_config()
# Paths: data_dir, download_dir, sources_dir, results_dir
# All configurable via .env (defaults to data/)
```

### Working with DataFrames (Polars, NOT Pandas)
```python
from newspaper_explorer.data.loading import DataLoader

# Load via source name (recommended)
loader = DataLoader(source_name="der_tag")
df = loader.load_source()

# Or load saved parquet
df = DataLoader.load_parquet("data/raw/der_tag/text/der_tag_lines.parquet")

# Polars operations
df.filter(df["year"] == 1901)
df.group_by("text_block_id").agg(df["text"].str.concat(" "))
```

### Line-Level Data Schema
Each DataFrame row = one text line from ALTO:
```python
{
    "line_id": str,          # Unique identifier
    "text": str,             # OCR text (cleaned)
    "text_block_id": str,    # Block grouping
    "filename": str,         # Source XML file
    "date": datetime,        # Publication date
    "x", "y", "width", "height": int,  # Coordinates
    "newspaper_title": str,  # From METS
    "year_volume": str,      # From METS
    "page_count": int,       # From METS
}
```

### Parallel Processing Pattern
- DataLoader uses `ProcessPoolExecutor` with `cpu_count() - 1` workers
- METS cache pre-built before parallel parsing (shared metadata)
- Worker function: `_parse_file_worker(filepath, mets_cache)`

### Error Correction (DataFixer)
Known data issues are automatically corrected during extraction:
- Mislabeled dates in filenames (e.g., 1900 → 1902)
- Relocates issues to correct year directories
- Updates METS XML with corrected dates
- Dataset-specific fixes in `_fix_{dataset}_{issue}` methods

## Code Style

- **Black**: 100 char line length
- **Type hints**: Optional but preferred (mypy with `disallow_untyped_defs = false`)
- **Docstrings**: Required for public APIs
- **CLI**: Rich examples in docstrings

### Output Standards

**Context-based output handling - NEVER use `print()`:**

1. **CLI Commands** (`cli/*.py`):
   - Use `click.echo()` for all user-facing messages
   - Use `click.echo(..., err=True)` for errors
   - Use `tqdm` for progress bars
   
2. **Library Code** (`data/*.py`, `analysis/*.py`):
   - Use `logging` module exclusively
   - `logger.info()` for informational messages
   - `logger.debug()` for verbose details
   - `logger.warning()` for warnings
   - `logger.error()` for errors
   - Configure logging in CLI entry points, not in library code
   
3. **Progress Tracking**:
   - Use `tqdm` for long-running operations
   - Works alongside logging

**Why**: Separates user-facing CLI output from internal library logging, enables proper testing, and allows output control via log levels.

See `docs/OUTPUT_STANDARDS.md` for detailed guidelines.

## Common Operations

### Adding a Source
1. Create `data/sources/{source}.json` with metadata and loading config
2. Both download and load commands will auto-discover it
3. Test with: `newspaper-explorer data sources`

### Adding Analysis
1. Create module in `analysis/{type}/`
2. Accept Polars DataFrame as input
3. Output to `results/{type}/`
4. Add CLI command in `cli/analyze.py` (create if needed)

### Testing
- Place in `tests/` mirroring `src/` structure
- Use pytest, mark integration tests: `@pytest.mark.integration`
- Sample data in `tests/data/`

## Critical Patterns

✅ **DO**:
- Initialize classes with source names: `DataLoader("der_tag")`
- Use Polars for DataFrames (not Pandas)
- Parse METS for metadata enrichment
- Check loading status before re-processing
- Use configuration for all paths

❌ **AVOID**:
- `__init__.py` files anywhere
- Hardcoded paths (use `get_config()`)
- Pandas operations (use Polars)
- Re-parsing already processed files (use resume)
- Path-based CLI args (use `--source name`)
- **`print()` statements** (use `click.echo()` in CLI, `logging` in libraries)

## No Legacy Support

**This project does NOT maintain backwards compatibility or legacy support.**

- When improving code, replace old patterns completely—don't add compatibility layers
- Remove deprecated code paths rather than wrapping them
- Update all call sites when changing APIs
- No `if legacy_mode:` branches or `use_old_api` flags
- Breaking changes are acceptable and expected as the project evolves
- Focus on the best current solution, not supporting old approaches

**Example**: When we moved from path-based to source-based loading, we completely replaced the CLI interface. The old `newspaper-explorer data load <path>` was removed in favor of `newspaper-explorer data load --source <name>`. No compatibility mode, no warnings, clean break.

## Key Files to Reference
- `data/sources/der_tag.json` - Source configuration example
- `data/loading.py` - Configuration-driven loading pattern
- `data/utils/alto_parser.py` - XML parsing with dataclasses
- `cli/data.py` - CLI command patterns with Click
- `docs/DATA_LOADER.md` - Detailed loading architecture
