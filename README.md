# Newspaper Explorer

A comprehensive toolkit for downloading, processing, and analyzing historical newspaper data from ALTO XML archives with METS metadata enrichment. Built with a configuration-driven architecture and modern data tools (Polars, DuckDB, LLMs).

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Key Features

- **📥 Smart Data Pipeline**: Configuration-driven download → extract → parse → analyze workflow
- **🚀 High Performance**: Parallel processing with Polars DataFrames and DuckDB queries
- **🤖 LLM Integration**: Structured prompts for entity extraction, topic analysis, emotion detection, and more
- **📊 Query Engine**: DuckDB-based analysis layer for efficient multi-GB data queries
- **🔄 Resume Support**: Automatic tracking of processed files to avoid reprocessing
- **🖼️ Image Support**: Download high-resolution newspaper page scans from METS references
- **🔧 Modular CLI**: Clean command structure for all operations

## 📑 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [CLI Commands Overview](#cli-commands-overview)
- [Python API Examples](#python-api-examples)
- [Documentation](#-documentation)
- [Data Schemas](#data-schemas)
- [Project Structure](#️-project-structure)
- [Features](#-features)
- [Usage Examples](#-usage-examples)
- [Available Datasets](#-available-datasets)
- [Development](#️-development)
- [Contributing](#-contributing)
- [License](#-license)

## 📦 Installation

### Quick Start (Recommended: uv)

```bash
# Install uv if needed
pip install uv

# Clone and setup
git clone https://github.com/newspaper-collective/newspaper_explorer.git
cd newspaper_explorer

# Copy environment template and configure
cp .env.example .env
# Edit .env to add your LLM API credentials if using LLM features

# Create venv and install
uv venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\Activate.ps1  # Windows PowerShell

uv pip install -e .

# Optional: Install additional features
uv pip install -e ".[nlp]"        # For sentence splitting (spacy)
uv pip install -e ".[normalize]"  # For text normalization
uv pip install transformers torch # For entity extraction (GLiNER)
```

### Alternative: pip

```bash
# Clone the repository
git clone https://github.com/newspaper-collective/newspaper_explorer.git
cd newspaper_explorer

# Setup environment
cp .env.example .env
python -m venv .venv
source .venv/bin/activate  # Linux/Mac

# Install
pip install -e .

# Optional features
pip install -e ".[nlp]"
pip install -e ".[normalize]"
pip install transformers torch
```

See [INSTALL.md](docs/INSTALL.md) for detailed installation instructions.

## 🚀 Quick Start

### Complete Workflow Example

```bash
# 1. List available sources
newspaper-explorer data list-sources

# 2. Download newspaper data
newspaper-explorer data download --part dertag_1900-1902

# 3. Unpack archives
newspaper-explorer data unpack --source der_tag

# 4. Parse ALTO XML to Parquet with METS metadata
newspaper-explorer data parse --source der_tag

# 5. Aggregate lines into text blocks
newspaper-explorer data aggregate --source der_tag

# 6. Check comprehensive status
newspaper-explorer data info --source der_tag

# 7. Optional: Download page images
newspaper-explorer data download-images --source der_tag --max-workers 8

# 8. Preprocess text (normalize, clean, etc.)
newspaper-explorer data preprocess --source der_tag --normalize --lemmatize
```

Then analyze with Python:

```python
import polars as pl
from newspaper_explorer.data.loading.loader import DataLoader

# Load the parsed line-level data
loader = DataLoader(source_name="der_tag")
df = loader.load_parquet("data/raw/der_tag/text/der_tag_lines.parquet")

# Or load text blocks
blocks_df = loader.load_parquet("data/raw/der_tag/text/der_tag_text_blocks.parquet")

# Filter to a specific year
df_1901 = df.filter(pl.col("year") == 1901)

print(f"Found {len(df_1901)} lines in 1901")
print(f"Date range: {df_1901['date'].min()} to {df_1901['date'].max()}")

# Query with DuckDB for complex analytics
from newspaper_explorer.analysis.query.engine import QueryEngine

engine = QueryEngine(source_name="der_tag")
result = engine.query("""
    SELECT 
        strftime(date, '%Y-%m') as month,
        COUNT(*) as line_count,
        COUNT(DISTINCT text_block_id) as block_count
    FROM lines
    WHERE year = 1901
    GROUP BY month
    ORDER BY month
""")
print(result)
```

### CLI Commands Overview

**Data Management:**
```bash
# Source management
newspaper-explorer data list-sources              # List configured sources
newspaper-explorer data info --source der_tag     # Comprehensive status

# Download pipeline
newspaper-explorer data download --part dertag_1900-1902
newspaper-explorer data verify                     # Verify checksums
newspaper-explorer data unpack --source der_tag    # Extract archives

# Processing pipeline
newspaper-explorer data parse --source der_tag     # Parse XML to Parquet
newspaper-explorer data aggregate --source der_tag # Aggregate lines to blocks
newspaper-explorer data preprocess --source der_tag --normalize --lemmatize

# Image downloads
newspaper-explorer data download-images --source der_tag --max-workers 8

# Utilities
newspaper-explorer data find-empty --source der_tag  # Find empty XML files
```

**Analysis:**
```bash
# Entity extraction with GLiNER
newspaper-explorer analyze extract-entities \
    --source der_tag \
    --input data/raw/der_tag/text/der_tag_text_blocks.parquet \
    --normalize \
    --threshold 0.6

# More analysis commands coming soon (topics, emotions, etc.)
```

**Get Help:**
```bash
newspaper-explorer --help
newspaper-explorer data --help
newspaper-explorer data parse --help
newspaper-explorer analyze --help
```

### Python API Examples

#### 1. Data Loading

```python
from newspaper_explorer.data.loading.loader import DataLoader
import polars as pl

# Initialize loader with source name
loader = DataLoader(source_name="der_tag")

# Load line-level data (automatically finds parquet file)
lines_df = loader.load_source()

# Load text blocks (aggregated lines)
blocks_df = DataLoader.load_parquet(
    "data/raw/der_tag/text/der_tag_text_blocks.parquet"
)

# Work with Polars DataFrames (NOT Pandas!)
df_1901 = lines_df.filter(pl.col("year") == 1901)
print(f"Found {len(df_1901)} lines in 1901")

# Access line-level data with coordinates
print(lines_df.select([
    "line_id", "text", "x", "y", 
    "newspaper_title", "date"
]).head())
```

#### 2. Query Engine (DuckDB)

```python
from newspaper_explorer.analysis.query.engine import QueryEngine

# Initialize query engine
engine = QueryEngine(source_name="der_tag")

# SQL queries on Parquet files (no memory loading!)
result = engine.query("""
    SELECT 
        year,
        COUNT(*) as total_lines,
        COUNT(DISTINCT date) as unique_dates
    FROM lines
    WHERE year BETWEEN 1900 AND 1905
    GROUP BY year
    ORDER BY year
""")
print(result)

# Query with parameters
result = engine.query("""
    SELECT text, date
    FROM lines
    WHERE text LIKE ? AND year = ?
    LIMIT 10
""", params=["%Kaiser%", 1901])

# Join source data with analysis results
result = engine.query("""
    SELECT 
        l.date,
        e.entity_text,
        e.entity_type,
        e.confidence
    FROM lines l
    JOIN entities e ON l.line_id = e.line_id
    WHERE e.entity_type = 'person'
    LIMIT 10
""")
```

#### 3. LLM Analysis

```python
from newspaper_explorer.llm.client import LLMClient
from newspaper_explorer.llm.prompts.entity_extraction import ENTITY_EXTRACTION
from newspaper_explorer.llm.schemas.entity_extraction import EntityResponse

# Text to analyze
text = "Kaiser Wilhelm II empfing Bernhard von Bülow in Berlin."

# Format prompt with metadata for better context
metadata = {
    "source": "der_tag",
    "newspaper_title": "Der Tag",
    "date": "1901-01-15",
    "page_number": 3
}

prompt = ENTITY_EXTRACTION
formatted = prompt.format(text=text, metadata=metadata)

# Make LLM request with structured output
with LLMClient() as client:
    response = client.complete(
        prompt=formatted["user"],
        system_prompt=formatted["system"],
        response_schema=EntityResponse,
        temperature=0.3  # Low for extraction tasks
    )

# Type-safe access to results
print(response.persons)      # ["Kaiser Wilhelm II", "Bernhard von Bülow"]
print(response.locations)    # ["Berlin"]
print(response.confidence)   # 0.95
```

#### 4. Downloading Archives

```python
from newspaper_explorer.data.download.text import ZenodoDownloader

# Initialize downloader
downloader = ZenodoDownloader(source_name="der_tag")

# Check what's available
parts = downloader.list_parts()
for part in parts:
    print(f"{part['name']}: {part['years']}")

# Download specific part
downloader.download_part("dertag_1900-1902")

# Check status
status = downloader.get_status()
print(f"Downloaded: {status['downloaded']}/{status['total']}")
```

#### 5. Image Downloads

```python
from newspaper_explorer.data.download.images import ImageDownloader

# Initialize downloader
downloader = ImageDownloader(
    source_name="der_tag",
    max_workers=8,
    max_retries=3
)

# Download images for specific date range
stats = downloader.download_images(
    year_start=1901,
    year_end=1902,
    skip_existing=True
)

print(f"Downloaded: {stats['downloaded']}")
print(f"Skipped: {stats['skipped']}")
print(f"Failed: {stats['failed']}")

# Get download status
status = downloader.get_download_status()
print(f"Total images: {status['total_images']}")
print(f"Downloaded: {status['downloaded_count']}")
```

## 📚 Documentation

### Guides
- **[Installation Guide](docs/INSTALL.md)** - Setup for pip and uv
- **[CLI Reference](docs/CLI.md)** - Complete command reference
- **[Data Management](docs/DATA.md)** - Download, extraction, source configuration
- **[Data Loader](docs/DATA_LOADER.md)** - ALTO/METS parsing, DataFrames, resume functionality
- **[LLM Utilities](docs/LLM.md)** - Complete guide to prompts, schemas, client usage
- **[Query Architecture](docs/QUERY_ARCHITECTURE.md)** - DuckDB query engine for analysis
- **[Normalization](docs/NORMALIZATION.md)** - Historical German text normalization
- **[Entity Extraction](docs/ENTITIES.md)** - Named entity recognition with GLiNER
- **[Image Downloads](docs/IMAGES.md)** - High-resolution page scans

### Architecture
- **[Data Architecture](docs/DATA_ARCHITECTURE.md)** - Overall data pipeline design
- **[Configuration Philosophy](docs/CONFIGURATION_PHILOSOPHY.md)** - Config vs CLI flags

### Key Concepts

**Configuration-Driven Architecture**: Sources are defined in `data/sources/{source}.json` with download URLs, patterns, and metadata. Single source of truth for all operations.

**Modular CLI Structure**: Commands organized by function (`data/`, `analyze/`) with clean subcommands. No legacy compatibility—always the best current approach.

**Source-Based Operations**: All commands work with `--source {name}` instead of raw paths. Example: `newspaper-explorer data parse --source der_tag`

**METS Metadata Enrichment**: Every newspaper issue has METS XML with metadata (title, date, volume). Automatically parsed and merged with ALTO text data.

**Line-Level + Text Blocks**: Parse creates line-level data (each text line from ALTO). Aggregate command creates text blocks (coherent paragraphs/articles).

**Resume Support**: Automatic tracking of processed files. Commands skip already-processed data unless you use `--force` flags.

**Query Layer**: DuckDB engine queries Parquet files directly (no loading into memory). Enables complex SQL analytics on multi-GB datasets.

**LLM Integration**: Structured prompts with Pydantic schemas for reliable, type-safe responses. Supports entity extraction, topic analysis, emotion detection, summarization, and more.

### Data Schemas

**Line-Level Data** (`{source}_lines.parquet`):

| Field             | Type     | Description                           |
| ----------------- | -------- | ------------------------------------- |
| `line_id`         | str      | Unique identifier (FK for analysis)   |
| `text`            | str      | OCR-extracted text content (cleaned)  |
| `text_block_id`   | str      | Identifier for grouping related lines |
| `filename`        | str      | Source ALTO XML filename              |
| `date`            | datetime | Publication date (from METS)          |
| `x`, `y`          | int      | Coordinates of text line on page      |
| `width`, `height` | int      | Dimensions of text line               |
| `newspaper_title` | str      | Name of newspaper (from METS)         |
| `year_volume`     | str      | Year and volume number (from METS)    |
| `page_count`      | int      | Number of pages in issue (from METS)  |
| `year`            | int      | Extracted year for filtering          |

**Text Block Data** (`{source}_text_blocks.parquet`):

| Field             | Type     | Description                              |
| ----------------- | -------- | ---------------------------------------- |
| `text_block_id`   | str      | Unique block identifier                  |
| `text`            | str      | Aggregated text from all lines in block  |
| `line_count`      | int      | Number of lines in this block            |
| `date`            | datetime | Publication date                         |
| `newspaper_title` | str      | Name of newspaper                        |
| `filename`        | str      | Source XML filename                      |
| `year`            | int      | Year for filtering                       |

**Analysis Results** (varies by type, e.g., `results/{source}/entities/{method}/entities.parquet`):

| Field         | Type  | Description                        |
| ------------- | ----- | ---------------------------------- |
| `line_id`     | str   | Foreign key to source data         |
| `entity_text` | str   | Extracted entity text              |
| `entity_type` | str   | Type (person, location, org, etc.) |
| `confidence`  | float | Confidence score 0-1               |
| `method`      | str   | Extraction method/model used       |

## 🗂️ Project Structure

```
newspaper_explorer/
├── src/newspaper_explorer/
│   ├── main.py                      # CLI entry point
│   ├── config/                      # Configuration management
│   │   ├── base.py                  # Main Config class
│   │   └── environment.py           # Environment variable loading
│   ├── cli/                         # Modular CLI commands
│   │   ├── data/                    # Data management commands
│   │   │   ├── commands.py          # Main group registration
│   │   │   ├── download.py          # download, unpack, verify
│   │   │   ├── images.py            # download-images
│   │   │   ├── info.py              # info, list-sources
│   │   │   ├── loading.py           # parse, aggregate, find-empty
│   │   │   ├── preprocessing.py     # preprocess
│   │   │   └── common.py            # Shared constants
│   │   └── analyze.py               # Analysis commands
│   ├── data/                        # Data acquisition & processing
│   │   ├── download/
│   │   │   ├── text.py              # ZenodoDownloader
│   │   │   └── images.py            # ImageDownloader
│   │   ├── parser/
│   │   │   ├── alto.py              # ALTO XML parser
│   │   │   └── mets.py              # METS metadata parser
│   │   ├── loading/
│   │   │   ├── loader.py            # Main DataLoader class
│   │   │   ├── workers.py           # Parallel processing
│   │   │   └── aggregation.py       # Text aggregation
│   │   ├── preprocessing/           # Text preprocessing
│   │   └── utils/                   # Data utilities
│   ├── llm/                         # LLM integration (first-class)
│   │   ├── client.py                # LLM client with retry
│   │   ├── prompts/                 # Prompt templates (direct import)
│   │   │   ├── base.py              # PromptTemplate base
│   │   │   ├── entity_extraction.py
│   │   │   ├── topic_analysis.py
│   │   │   ├── emotion_analysis.py
│   │   │   └── ...
│   │   └── schemas/                 # Response schemas (direct import)
│   │       ├── entity_extraction.py
│   │       ├── topic_analysis.py
│   │       └── ...
│   ├── analysis/                    # Analysis modules
│   │   ├── concepts/                # Concept extraction
│   │   ├── embeddings/              # Embedding generation
│   │   ├── emotions/                # Emotion analysis
│   │   ├── entities/                # Entity extraction (GLiNER)
│   │   ├── layout/                  # Layout analysis
│   │   ├── topics/                  # Topic modeling
│   │   └── query/                   # DuckDB query engine
│   │       ├── engine.py            # QueryEngine class
│   │       └── examples.py          # Usage examples
│   ├── utils/                       # Lean utilities
│   │   └── sources.py               # Source config utilities
│   └── ui/                          # User interface (future)
├── data/
│   ├── sources/                     # Source configurations
│   │   └── der_tag.json             # URLs, patterns, metadata
│   ├── downloads/                   # Downloaded archives
│   │   └── der_tag/xml_ocr/
│   ├── extracted/                   # Temporary extraction
│   └── raw/                         # Organized data
│       └── der_tag/
│           ├── xml_ocr/             # Raw XML by year
│           ├── images/              # Page images by year/month
│           └── text/                # Parsed Parquet files
├── results/                         # Analysis outputs
│   └── der_tag/
│       ├── entities/{method}/       # Entity extraction by method
│       ├── topics/{method}/         # Topic modeling by method
│       ├── emotions/{method}/       # Emotion analysis by method
│       └── .../                     # Other analysis types
├── docs/                            # Comprehensive documentation
├── tests/                           # Test suite
└── notebooks/                       # Jupyter notebooks
```

**Development Patterns:**

- **NO `__init__.py` files** - Use explicit imports only
- Import style: `from newspaper_explorer.data.loading.loader import DataLoader`
- **Polars, not Pandas** - All DataFrames are `polars.DataFrame`
- **Configuration-driven** - Sources defined in `data/sources/{name}.json`
- **Modular CLI** - Clean command organization without legacy support
- **Method tracking** - Results stored in `{type}/{method}/` directories
- **Foreign keys** - All analysis results have `line_id` FK to source data
- See `.github/copilot-instructions.md` for complete coding guidelines

## 🎯 Features

### ✅ Data Pipeline

**Download & Extract:**
- Configuration-driven Zenodo downloads from `data/sources/{source}.json`
- MD5 checksum verification
- Automatic archive extraction with error correction
- Smart caching (skip existing downloads)
- Progress tracking with detailed status reporting

**Parse & Load:**
- Parallel ALTO XML parsing with multiprocessing
- Automatic METS metadata enrichment (titles, dates, volumes)
- Line-level output with text coordinates
- Text block aggregation (coherent paragraphs/articles)
- Resume functionality (skip processed files)
- Auto-save to compressed Parquet (zstd)
- Fast Polars DataFrames (not Pandas!)

**Images:**
- Parallel download of high-resolution page scans from METS references
- Organized by year/month matching XML structure
- Skip existing files, resume support
- Progress tracking and statistics

### ✅ Text Processing

**Preprocessing Pipeline:**
- Historical German text normalization (19th/20th century models)
- Lemmatization with German morphology support
- Sentence splitting (spaCy)
- Configurable pipeline via CLI flags

**Text Utilities:**
- Line-to-block aggregation
- Sentence segmentation for German
- Text cleaning and normalization
- Custom preprocessing functions

### ✅ Analysis & LLM Integration

**Query Engine (DuckDB):**
- SQL queries on multi-GB Parquet files without memory loading
- Join source data with analysis results
- Complex analytics with standard SQL
- Foreign key support via `line_id`

**LLM Client:**
- Structured prompts with Pydantic validation
- Retry logic with exponential backoff
- Support for any OpenAI-compatible API
- Temperature control for different tasks
- Metadata-aware prompts (date, source, page context)

**Available Prompts & Schemas:**
- Entity extraction (persons, locations, organizations)
- Topic analysis (themes, subjects)
- Emotion detection (sentiment, tone)
- Concept extraction (ideas, themes)
- Text summarization
- Text quality assessment

**Entity Extraction:**
- GLiNER-based named entity recognition
- Supports persons, organizations, locations, events
- Configurable confidence thresholds
- Batch processing
- Output to Parquet and JSON

### ✅ CLI Commands

**Data Management:**
- `list-sources` - Show configured sources
- `info` - Comprehensive status (XML, Parquet, images)
- `download` - Download archives
- `verify` - Verify checksums
- `unpack` - Extract archives

**Data Processing:**
- `parse` - ALTO XML to Parquet
- `aggregate` - Lines to text blocks
- `preprocess` - Text normalization pipeline
- `find-empty` - Find empty XML files
- `download-images` - Download page scans

**Analysis:**
- `extract-entities` - Entity extraction with GLiNER

### 📝 Roadmap

- Topic modeling with LLMs and traditional methods
- Emotion/sentiment analysis
- Layout analysis (column detection, article segmentation)
- Concept extraction
- Web UI (Streamlit or FastAPI)
- More analysis types

## 💡 Usage Examples

### Text Preprocessing

```python
# Use the CLI for the full preprocessing pipeline
# newspaper-explorer data preprocess --source der_tag --normalize --lemmatize

# Or use Python API
from newspaper_explorer.data.preprocessing.pipeline import PreprocessingPipeline
import polars as pl

# Load text blocks
df = pl.read_parquet("data/raw/der_tag/text/der_tag_text_blocks.parquet")

# Create pipeline with desired steps
pipeline = PreprocessingPipeline(
    normalize=True,
    normalize_model="20c",  # 1900-1999 model
    lemmatize=True,
    sentence_split=True
)

# Process the data
result = pipeline.process(df, text_column="text")

# Save results
result.write_parquet("data/processed/der_tag/preprocessed.parquet")
```

**Requirements:**
```bash
pip install -e ".[nlp,normalize]"
python -m spacy download de_core_news_sm
```

### Entity Extraction with GLiNER

**CLI (Recommended):**
```bash
newspaper-explorer analyze extract-entities \
    --source der_tag \
    --input data/raw/der_tag/text/der_tag_text_blocks.parquet \
    --normalize \
    --threshold 0.6 \
    --labels Person,Organisation,Ort,Ereignis
```

**Python API:**
```python
from newspaper_explorer.analysis.entities.extraction import EntityExtractor

# Initialize extractor
extractor = EntityExtractor(
    source_name="der_tag",
    model_name="urchade/gliner_multi-v2.1",
    labels=["Person", "Organisation", "Ereignis", "Ort"],
    threshold=0.5
)

# Extract entities
results = extractor.extract_and_save(
    input_path="data/raw/der_tag/text/der_tag_text_blocks.parquet",
    text_column="text",
    id_column="text_block_id",
    normalize=True,
    output_format="both"  # Parquet + JSON
)

# Results in: results/der_tag/entities/{method}/
```

**Requirements:**
```bash
pip install transformers torch
```

### Advanced Queries with DuckDB

```python
from newspaper_explorer.analysis.query.engine import QueryEngine

engine = QueryEngine(source_name="der_tag")

# Find all mentions of a person with context
result = engine.query("""
    SELECT 
        l.date,
        l.text,
        l.newspaper_title,
        e.entity_text,
        e.confidence
    FROM lines l
    JOIN entities e ON l.line_id = e.line_id
    WHERE e.entity_type = 'person' 
    AND e.entity_text LIKE '%Wilhelm%'
    ORDER BY l.date
    LIMIT 100
""")

# Aggregate entity counts by year
result = engine.query("""
    SELECT 
        YEAR(l.date) as year,
        e.entity_type,
        COUNT(DISTINCT e.entity_text) as unique_entities,
        COUNT(*) as total_mentions
    FROM lines l
    JOIN entities e ON l.line_id = e.line_id
    GROUP BY year, e.entity_type
    ORDER BY year, e.entity_type
""")

# Compare different extraction methods
result = engine.query("""
    SELECT 
        e1.entity_text,
        COUNT(*) as gliner_count,
        SUM(CASE WHEN e2.entity_text IS NOT NULL THEN 1 ELSE 0 END) as llm_count
    FROM entities_gliner e1
    LEFT JOIN entities_llm e2 ON e1.line_id = e2.line_id 
        AND e1.entity_text = e2.entity_text
    WHERE e1.entity_type = 'person'
    GROUP BY e1.entity_text
    HAVING COUNT(*) > 10
    ORDER BY gliner_count DESC
""")
```

## 📊 Available Datasets

### Der Tag (1900-1920)

Historical Austrian newspaper "Der Tag" - ALTO XML with METS metadata.

**Collection**: [Der Tag on Zenodo](https://zenodo.org/records/17232177)

**Coverage**: 1900-1920 (multiple archive parts)  
**Format**: ALTO XML (OCR fulltext) + METS XML (metadata)  
**Language**: German (historical orthography)

Use `newspaper-explorer data list-sources` to see configured sources and `newspaper-explorer data info --source der_tag` for detailed status.

### Adding New Sources

Create `data/sources/{source_name}.json`:

```json
{
  "dataset_name": "my_newspaper",
  "data_type": "xml_ocr",
  "metadata": {
    "newspaper_title": "My Newspaper",
    "language": "de",
    "years_available": "1850-1900"
  },
  "loading": {
    "pattern": "**/fulltext/*.xml",
    "compression": "zstd"
  },
  "parts": [
    {
      "name": "mynews_1850-1860",
      "url": "https://zenodo.org/records/...",
      "md5": "...",
      "years": "1850-1860"
    }
  ]
}
```

The system will automatically discover and use your new source configuration.

## 🛠️ Development

### Setup Development Environment

```bash
# Clone and setup
git clone https://github.com/newspaper-collective/newspaper_explorer.git
cd newspaper_explorer

# Install with dev dependencies
pip install -e ".[dev]"

# Install optional features for development
pip install -e ".[nlp,normalize]"
pip install transformers torch
python -m spacy download de_core_news_sm
```

### Development Workflow

```bash
# Run tests
pytest

# Run specific test file
pytest tests/data/test_loading.py

# Run with coverage
pytest --cov=newspaper_explorer --cov-report=html

# Format code (Black - 100 char line length)
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking (optional - mypy allows untyped defs)
mypy src/newspaper_explorer/
```

### Code Style & Patterns

- **NO `__init__.py` files** - Use explicit imports
- **Polars, not Pandas** - For all DataFrame operations
- **Click for CLI** - Rich docstrings with examples
- **Logging in libraries** - `click.echo()` only in CLI commands
- **Configuration-driven** - Sources in `data/sources/{name}.json`
- **No legacy support** - Replace old patterns completely, no compatibility layers

See `.github/copilot-instructions.md` for comprehensive coding guidelines.

### Adding New Commands

1. Create module in `cli/data/` or `cli/analyze/`
2. Implement commands with Click decorators
3. Register in `commands.py` or `analyze.py`
4. Update documentation

### Adding New Analysis

1. Create module in `analysis/{type}/`
2. Accept Polars DataFrame, return with `line_id` FK
3. Save to `results/{source}/{type}/{method}/`
4. Include `metadata.json` with method info
5. Add CLI command for user access

## 🤝 Contributing

Contributions are welcome! This project prioritizes:

- **Clean architecture** over backwards compatibility
- **Modern tools** (Polars, DuckDB) over legacy approaches
- **Type safety** with Pydantic schemas
- **Configuration over convention** for flexibility

Please feel free to submit Pull Requests or open issues for bugs and feature requests.

## � License

MIT License - see [LICENSE](LICENSE) file for details.

## �📧 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/newspaper-collective/newspaper_explorer/issues)
- **Documentation**: See `docs/` directory for comprehensive guides
- **Examples**: Check `notebooks/` for Jupyter notebook examples

## 🙏 Acknowledgments

- **ALTO XML Format**: [ALTO XML Schema](https://www.loc.gov/standards/alto/)
- **METS Format**: [METS Schema](http://www.loc.gov/standards/mets/)
- **Der Tag Dataset**: [Zenodo Collection](https://zenodo.org/records/17232177)
- **Tools**: Built with Polars, DuckDB, Click, Pydantic, and modern Python
