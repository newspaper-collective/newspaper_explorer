# GitHub Copilot Instructions for Newspaper Explorer

## Architecture Overview

**Purpose**: Load, process, and analyze historical newspaper data from ALTO XML archives.

**Data Pipeline**: Zenodo archives → Download/Extract → Raw XML → Parse (ALTO+METS) → Polars DataFrame → Analysis

### Core Components

1. **ZenodoDownloader** (`data/download/text.py`) - Configuration-driven archive download/extraction
   - Reads from `data/sources/{source}.json` for URLs, checksums, metadata
   - Organizes archives into `data/raw/{source}/{data_type}/YYYY/` structure
   - Automatic error correction via `DataFixer` for known data issues
   
2. **ImageDownloader** (`data/download/images.py`) - Page image download from METS
   - Downloads high-resolution newspaper page images from METS XML references
   - Stores in `data/raw/{source}/images/` mirroring XML directory structure
   - Parallel download with progress tracking and resume support
   - Provides download status via `get_download_status()` method
   
3. **DataIngester** (`data/ingest/`) - Configuration-driven XML parsing (modular)
   - `loader.py`: Main DataIngester class, initialized with `source_name`
   - `workers.py`: Parallel processing workers for ALTO/METS parsing
   - Reads source config from `data/sources/{source}.json` for paths and patterns
   - Parallel ALTO XML parsing with METS metadata enrichment
   - Outputs line-level Polars DataFrames to `data/raw/{source}/text/{source}_lines.parquet`
   - Resume functionality: skips already-processed files by default

4. **ALTO/METS Parsers** (`data/parser/`) - XML extraction layer
   - ALTOParser: Extracts text lines with coordinates from fulltext pages
   - METSParser: Extracts issue-level metadata (title, date, volume, page count)
   - Automatic namespace detection for ALTO version compatibility

5. **Text Aggregation** (`data/processing/aggregation.py`) - Post-processing utilities
   - Aggregates line-level data into text blocks
   - Operates on already-loaded parquet files

6. **QueryEngine** (`analyze/query/engine.py`) - DuckDB-based query interface
   - Unified query engine for both source data and analysis results
   - UI-focused methods: get_stats(), search_text_simple(), get_sample_data()
   - Analysis methods: find_entity_mentions(), entity_frequency(), etc.
   - Replaces old data/query.py (now deleted)

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
- Import style: `from newspaper_explorer.data.loading.loader import DataLoader`
- Never: `from newspaper_explorer.data import DataLoader`

### Module Structure
```
src/newspaper_explorer/
├── config/                 # Configuration management
│   ├── environment.py     # Environment variable loading
│   └── base.py           # Main Config class
├── llm/                   # LLM functionality (first-class module)
│   ├── client.py         # LLM client with retry & validation
│   ├── examples.py       # Usage examples
│   ├── prompts/          # Individual prompt templates (direct import)
│   │   ├── base.py      # PromptTemplate base class
│   │   ├── entity_extraction.py
│   │   ├── topic_analysis.py
│   │   ├── emotion_analysis.py
│   │   ├── concept_extraction.py
│   │   ├── summarization.py
│   │   └── text_quality.py
│   └── schemas/         # Individual response schemas (direct import)
│       ├── entity_extraction.py
│       ├── topic_analysis.py
│       ├── emotion_analysis.py
│       ├── concept_extraction.py
│       ├── summarization.py
│       └── text_quality.py
├── data/                 # Data acquisition & processing
│   ├── download/        # Download modules
│   │   ├── text.py     # Archive downloads (ZenodoDownloader)
│   │   └── images.py   # Image downloads (ImageDownloader)
│   ├── parser/         # XML parsers
│   │   ├── alto.py    # ALTO fulltext parser
│   │   └── mets.py    # METS metadata parser
│   ├── ingest/         # Data ingestion (one-time ETL)
│   │   ├── loader.py   # Main DataIngester class
│   │   └── workers.py  # Parallel processing workers
│   ├── processing/      # Data transformations
│   │   └── aggregation.py  # Text aggregation (lines → blocks)
│   ├── preprocessing/       # Text preprocessing (modular)
│   │   ├── pipeline.py      # Main preprocessing pipeline
│   │   ├── normalization.py # Historical text normalization
│   │   ├── cleaning.py      # Text cleaning functions
│   │   ├── filtering.py     # Filtering functions
│   │   └── linguistic.py    # Lemmatization, dehyphenation
│   └── utils/        # Data utilities
│       ├── fixes.py  # DataFixer for corrections
│       ├── text.py   # Text utilities
│       └── validation.py # Data quality validation
├── cli/               # Command-line interface
│   ├── data/         # Data management commands (modular)
│   │   ├── commands.py      # Main group registration
│   │   ├── download.py      # download, unpack, verify
│   │   ├── images.py        # download-images
│   │   ├── info.py          # info, list-sources
│   │   ├── loading.py       # parse, aggregate, find-empty
│   │   ├── preprocessing.py # preprocess
│   │   └── validation.py    # validate-alto-mets
│   ├── analyze/       # Analysis commands (nested structure)
│   │   ├── commands.py      # Main analyze group
│   │   ├── entities/        # Entity extraction commands
│   │   │   └── commands.py  # entities group
│   │   └── layout/          # Layout analysis commands
│   │       └── commands.py  # layout group
│   └── main.py        # CLI entry point
├── analysis/          # Analysis modules
│   ├── concepts/     # Concept extraction
│   ├── embeddings/   # Embedding generation
│   ├── emotions/     # Emotion analysis
│   ├── entities/     # Entity extraction
│   ├── layout/       # Layout analysis
│   ├── topics/       # Topic modeling
│   └── query/        # Query engine for analysis
│       ├── engine.py    # DuckDB query engine (unified)
│       └── examples.py  # Usage examples
├── utils/             # Lean utilities (infrastructure only)
│   ├── config.py     # DEPRECATED: compatibility shim
│   └── sources.py    # Source configuration
└── ui/                # User interface (if any)
```

### CLI Commands (Click)
All commands follow: `newspaper-explorer {group} {command} --option value`

**Data commands** (modular structure in `cli/data/`):
- `download` - Download archives from Zenodo
- `unpack --source <name>` - Extract downloaded archives
- `parse --source <name>` - Parse XML to Parquet
- `aggregate --source <name>` - Aggregate lines into text blocks
- `info --source <name>` - Show comprehensive source status
- `list-sources` - List available sources
- `download-images --source <name>` - Download page images
- `find-empty --source <name>` - Find empty XML files
- `preprocess --source <name>` - Preprocess text data
- `verify` - Verify checksums

### Configuration Management
```python
from newspaper_explorer.config.base import get_config

config = get_config()
# Paths: data_dir, download_dir, sources_dir, results_dir
# All configurable via .env (defaults to data/)
```

### LLM Usage with Metadata
```python
from newspaper_explorer.llm.client import LLMClient
from newspaper_explorer.llm.prompts.entity_extraction import ENTITY_EXTRACTION
from newspaper_explorer.llm.schemas.entity_extraction import EntityResponse

# Get prompt template (direct import, no wrapper)
prompt = ENTITY_EXTRACTION

# Format with text and metadata for better context
metadata = {
    "source": "Der Tag",
    "newspaper_title": "Der Tag",
    "date": "1920-01-15",
    "year_volume": "1920/15",
    "page_number": 3,
}
prompts = prompt.format(text="...", metadata=metadata)

# Make LLM request
with LLMClient() as client:
    response = client.complete(
        prompt=prompts["user"],
        system_prompt=prompts["system"],
        response_schema=EntityResponse,
    )
```

**Metadata Support**: All prompts accept optional metadata dict with fields:
- `source` - Source identifier (e.g., "der_tag")
- `newspaper_title` - Full newspaper title (e.g., "Der Tag")
- `date` - Publication date (ISO format: "1920-01-15")
- `year_volume` - Year and volume info
- `page_number` - Page number in the issue
- Any custom fields relevant to the analysis

Prompts with `include_metadata=True` automatically append context section.

**Direct Imports**: No wrapper files - import prompts and schemas directly from their modules.

### Working with DataFrames (Polars, NOT Pandas)
```python
from newspaper_explorer.data.ingest.loader import DataIngester

# Load via source name (recommended)
ingester = DataIngester(source_name="der_tag")
df = ingester.load_source()

# Or load saved parquet
df = DataIngester.load_parquet("data/raw/der_tag/text/der_tag_lines.parquet")

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
- DataIngester uses `ProcessPoolExecutor` with `cpu_count() - 1` workers
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
- **Emojis**: Do not use emojis in any generated code, comments, docstrings, log messages, CLI output, documentation, or commit messages.

### Data Validation with Pydantic

**Always validate external data immediately at boundaries using Pydantic models:**

```python
# BAD: Raw dict from API/file
def process_user_data(data: dict) -> None:
    name = data["name"]  # Could fail, no validation
    age = data.get("age", 0)  # Type is Any

# GOOD: Pydantic model + immediate validation
from pydantic import BaseModel, Field

class UserData(BaseModel):
    name: str = Field(min_length=1)
    age: int = Field(ge=0, le=150)

def process_user_data(data: dict) -> None:
    user = UserData.model_validate(data)  # Fails fast with clear errors
    # Now user.name and user.age are fully typed and validated
```

**Apply this pattern to:**
- **API responses**: `response.json()` → immediate `Model.model_validate()`
- **Config files**: `json.load()` → immediate `Model.model_validate()`
- **CLI arguments**: `argparse.Namespace` → convert to Pydantic model
- **Environment variables**: Use `pydantic-settings` instead of raw `os.getenv()`
- **JSON metadata**: Always validate with Pydantic schemas

**Why**: Catches errors early, provides type safety, generates clear validation errors, and prevents working with untyped `dict[str, Any]` data throughout the codebase.

### Avoid Boolean Traps

**Never use positional boolean parameters - they make code unreadable:**

```python
# BAD: What does True mean here?
send_email(user, True, False)

# GOOD: Use enums or keyword-only args
from enum import Enum

class EmailFormat(Enum):
    HTML = "html"
    PLAIN = "plain"

def send_email(
    user: User,
    *,  # Force keyword-only
    format: EmailFormat,
    async_send: bool = False
) -> None: ...

send_email(user, format=EmailFormat.HTML, async_send=False)
```

**Key patterns:**
- Use `*` to force keyword-only arguments for booleans
- Prefer enums over booleans when there are multiple states
- Use descriptive parameter names that make the boolean meaning clear
- Configuration objects are better than multiple boolean flags

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

**DO**:
- Initialize classes with source names: `DataIngester("der_tag")`
- Use Polars for DataFrames (not Pandas)
- Parse METS for metadata enrichment
- Check ingestion status before re-processing
- Use configuration for all paths
- Use `QueryEngine` for all data queries (source + analysis)

**AVOID**:
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

### Configuration & Data
- `data/sources/der_tag.json` - Source configuration example
- `data/ingest/loader.py` - Main DataIngester class
- `data/ingest/workers.py` - Parallel processing workers
- `data/processing/aggregation.py` - Text aggregation utilities
- `data/download/text.py` - Archive download (ZenodoDownloader)
- `data/download/images.py` - Image download (ImageDownloader)
- `data/parser/alto.py` - ALTO XML parsing with dataclasses
- `data/parser/mets.py` - METS metadata extraction
- `data/preprocessing/pipeline.py` - Main preprocessing pipeline
- `data/preprocessing/normalization.py` - Historical German text normalization
- `data/preprocessing/cleaning.py` - Text cleaning functions
- `data/preprocessing/filtering.py` - Filtering functions
- `data/preprocessing/linguistic.py` - Lemmatization, dehyphenation
- `data/utils/fixes.py` - DataFixer for automatic error correction
- `data/utils/validation.py` - Data quality validation utilities

### CLI
- `cli/main.py` - Main CLI entry point
- `cli/data/commands.py` - Main data command group registration
- `cli/data/download.py` - Download & unpack commands
- `cli/data/images.py` - Image download command
- `cli/data/loading.py` - Parse, aggregate, find-empty commands
- `cli/data/info.py` - Info & list-sources commands (with image status)
- `cli/data/preprocessing.py` - Preprocess command
- `cli/data/validation.py` - Validate ALTO-METS command
- `cli/analyze/commands.py` - Main analyze group
- `cli/analyze/entities/commands.py` - Entity extraction commands
- `cli/analyze/layout/commands.py` - Layout analysis commands

### Analysis
- `analyze/query/engine.py` - DuckDB query engine (unified for all queries)
- `analyze/query/examples.py` - Query usage examples

### Documentation
- `docs/DATA_LOADER.md` - Detailed loading architecture
- `docs/CLI.md` - Complete CLI reference
- `docs/IMAGES.md` - Image download documentation
- `docs/QUERY_ARCHITECTURE.md` - Query engine documentation
- `docs/LLM.md` - Complete LLM utilities guide (client, prompts, schemas, metadata)
- `docs/data/preprocessing/PREPROCESSING.md` - Complete preprocessing guide (all methods, pipelines, workflows)
- `docs/data/preprocessing/NORMALIZATION.md` - Historical text normalization guide (Transnormer, DTA-CAB)
- `docs/analysis/ENTITIES.md` - Entity extraction documentation
- `docs/analysis/LAYOUT.md` - Complete layout analysis guide (detection, extraction, matching, filtering)
- `docs/analysis/EMOTIONS.md` - Emotion classification documentation
- `docs/analysis/TOPICS.md` - Topic modeling documentation
