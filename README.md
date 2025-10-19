# Newspaper Explorer

A toolkit for exploring and analyzing historical newspaper data from ALTO XML archives with additional METS metadata.

## 📦 Installation

### Using pip

```bash
# Clone the repository
git clone https://github.com/newspaper-collective/newspaper_explorer.git
cd newspaper_explorer

# Copy environment template
cp .env.example .env

# Install in development mode
pip install -e .

# Optional: Install additional features
pip install -e ".[nlp]"        # For sentence splitting
pip install -e ".[normalize]"  # For text normalization
pip install transformers torch # For entity extraction
```

### Using uv (recommended - faster!)

```bash
# Install uv if needed
pip install uv

# Clone and setup
git clone https://github.com/newspaper-collective/newspaper_explorer.git
cd newspaper_explorer

# Copy environment template
cp .env.example .env

# Create venv and install
uv venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
# or: source .venv/bin/activate  # Linux/Mac

uv pip install -e .

# Optional: Install additional features
uv pip install -e ".[nlp]"        # For sentence splitting
uv pip install -e ".[normalize]"  # For text normalization
uv pip install transformers torch # For entity extraction
```

See [INSTALL.md](INSTALL.md) for detailed installation instructions.

## 🚀 Quick Start

### Complete Workflow Example

```bash
# 1. Download newspaper data
newspaper-explorer data download --part dertag_1900-1902

# 2. Parse ALTO XML to Parquet with METS metadata
newspaper-explorer data load --source der_tag

# 3. Check the results
newspaper-explorer data load-status --source der_tag
```

Then analyze with Python:

```python
import polars as pl
from newspaper_explorer.data.loading import DataLoader

# Load the parsed data
df = DataLoader.load_parquet("data/raw/der_tag/text/der_tag_lines.parquet")

# Filter to a specific year
df_1901 = df.filter(pl.col("year") == 1901)

# Aggregate lines into text blocks
blocks = df_1901.group_by("text_block_id").agg(
    pl.col("text").str.concat(" ").alias("block_text"),
    pl.col("date").first(),
    pl.col("newspaper_title").first()
)

print(f"Found {len(blocks)} text blocks in 1901")
```

### CLI Usage

```bash
# 1. Download Data
# List available dataset parts
newspaper-explorer data list

# Check current status
newspaper-explorer data status

# Download and extract specific parts
newspaper-explorer data download --part dertag_1900-1902

# Download multiple parts in parallel
newspaper-explorer data download --parts dertag_1900-1902,dertag_1903-1905 --parallel

# 2. Load and Parse Data
# List available sources
newspaper-explorer data sources

# Load a source (parses ALTO XML to Parquet with METS metadata)
newspaper-explorer data load --source der_tag

# Check loading status
newspaper-explorer data load-status --source der_tag

# Limit files for testing
newspaper-explorer data load --source der_tag --max-files 100

# Control worker processes
newspaper-explorer data load --source der_tag --workers 4

# 3. Get Help
newspaper-explorer --help
newspaper-explorer data --help
```

### Using the Python API

#### Downloading Data

```python
from newspaper_explorer.data.download import ZenodoDownloader

# Initialize downloader
downloader = ZenodoDownloader()

# List available parts
for part in downloader.list_available_parts():
    print(f"{part['name']} - {part['years']}")

# Download and extract a single part
downloader.download_and_extract(['dertag_1900-1902'])

# Download multiple parts in parallel
downloader.download_and_extract(
    ['dertag_1900-1902', 'dertag_1903-1905'],
    parallel=True
)

# Check status
downloader.print_status_summary()
```

#### Loading and Parsing Data

```python
from newspaper_explorer.data.loading import DataLoader
import polars as pl

# Initialize loader with source name
loader = DataLoader(source_name="der_tag")

# Load source (parses ALTO XML to Polars DataFrame with METS metadata)
df = loader.load_source()

# Resume functionality: skips already processed files by default
df = loader.load_source(skip_processed=True)

# Or force reprocess all files
df = loader.load_source(skip_processed=False)

# Limit files for testing
df = loader.load_source(max_files=100)

# Check loading status
status = loader.get_loading_status()
print(f"XML files: {status['xml_files_count']}")
print(f"Parquet exists: {status['parquet_exists']}")
if status['parquet_exists']:
    print(f"Rows: {status.get('parquet_rows', 0)}")
    print(f"Date range: {status.get('date_range', 'N/A')}")

# Work with the DataFrame (uses Polars, not Pandas!)
df_1901 = df.filter(pl.col("year") == 1901)
print(df_1901.head())

# Access line-level data with coordinates and METS metadata
print(df.select(["line_id", "text", "x", "y", "newspaper_title", "date"]).head())

# Group by text blocks and concatenate lines
blocks = df.group_by("text_block_id").agg(
    pl.col("text").str.concat(" ").alias("block_text"),
    pl.col("date").first(),
    pl.col("newspaper_title").first()
)

# Load pre-saved parquet directly
df = DataLoader.load_parquet("data/raw/der_tag/text/der_tag_lines.parquet")
```

## 📚 Documentation

- [CLI Reference](docs/CLI.md) - Complete CLI command reference
- [Data Management Guide](docs/DATA.md) - Detailed guide for data downloading, extraction, and source configuration
- [Data Loader Guide](docs/DATA_LOADER.md) - ALTO XML parsing, METS metadata, DataFrame operations, and resume functionality
- [Normalization Guide](docs/NORMALIZATION.md) - Historical German text normalization to modern orthography
- [Entity Extraction Guide](docs/ENTITIES.md) - Named entity recognition with GLiNER
- [Image Downloading Guide](docs/IMAGES.md) - Download high-resolution page scans
- [Installation Guide](docs/INSTALL.md) - Setup instructions for pip and uv
- [Configuration Philosophy](docs/CONFIGURATION_PHILOSOPHY.md) - Design decisions for config vs CLI flags

### Key Concepts

**Source Configuration**: All newspaper sources are defined in `data/sources/{source}.json` files with download URLs, extraction patterns, and metadata. This enables both download and loading to reference the same configuration.

**METS Metadata**: Each newspaper issue has a METS XML file containing metadata (title, date, volume). The loader automatically finds and parses METS files to enrich ALTO text data.

**Line-Level Data**: Each row in the output DataFrame represents one text line from ALTO XML, including the OCR text, coordinates (x, y, width, height), and METS metadata (newspaper title, date, volume).

**Text Blocks**: Related lines are automatically grouped into text blocks during parsing. Use the `text_block_id` field to aggregate lines into coherent text blocks.

**Resume Functionality**: The loader tracks which files have been processed and skips them by default. Use `--no-resume` to force reprocessing.

### Data Schema

The parsed line-level DataFrame includes:

| Field             | Type     | Description                           |
| ----------------- | -------- | ------------------------------------- |
| `line_id`         | str      | Unique identifier for each line       |
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

## 🗂️ Project Structure

```
newspaper_explorer/
├── src/
│   └── newspaper_explorer/
│       ├── main.py              # Main CLI entry point
│       ├── cli/                 # CLI commands
│       │   ├── data.py          # Data management commands
│       │   └── analyze.py       # Analysis commands
│       ├── data/                # Data handling
│       │   ├── download.py      # Download/extract from Zenodo
│       │   ├── fixes.py         # Error correction utilities
│       │   ├── loading.py       # ALTO XML parser with METS integration
│       │   ├── cleaning.py      # Data cleaning (placeholder)
│       │   └── utils/
│       │       ├── alto_parser.py   # ALTO XML parsing
│       │       ├── mets_parser.py   # METS metadata extraction
│       │       ├── images.py        # Image downloading from METS
│       │       └── text.py          # Text aggregation & sentence splitting
│       ├── analysis/            # Analysis modules
│       │   ├── concepts/        # Concept extraction (placeholder)
│       │   ├── emotions/        # Emotion analysis (placeholder)
│       │   ├── entities/        # Entity extraction (GLiNER)
│       │   │   └── extraction.py
│       │   ├── layout/          # Layout analysis (placeholder)
│       │   └── topics/          # Topic modeling (placeholder)
│       ├── ui/                  # UI components (future)
│       └── utils/               # Utilities (no __init__.py)
│           ├── config.py        # Configuration management
│           ├── sources.py       # Source configuration utilities
│           └── llm.py           # LLM utilities
├── data/
│   ├── sources/
│   │   └── der_tag.json         # Source config (URLs, paths, metadata)
│   ├── downloads/               # Downloaded .tar.gz files (preserved)
│   └── raw/                     # Organized extracted data
│       └── der_tag/
│           ├── xml_ocr/         # Raw XML files by year
│           ├── images/          # Downloaded page scans by year/month
│           └── text/            # Parsed Parquet files
├── results/                     # Analysis results and outputs
│   └── der_tag/
│       ├── entities/            # Entity extraction results
│       ├── concepts/            # Concept extraction results
│       ├── emotions/            # Emotion analysis results
│       ├── layout/              # Layout analysis results
│       └── topics/              # Topic modeling results
├── docs/                        # Documentation
└── tests/                       # Test suite
```

**Important Notes:**

- This project uses **explicit imports** without `__init__.py` files
- Use full module paths: `from newspaper_explorer.utils.config import get_config`
- **Uses Polars, not Pandas** - DataFrames are `polars.DataFrame` objects
- **Source-based architecture** - All operations reference source names (e.g., `der_tag`)
- **Resume by default** - Loading skips already processed files unless `--no-resume` is used
- **METS metadata enrichment** - All parsed data includes newspaper metadata from METS files
- See `.github/copilot-instructions.md` for detailed coding guidelines

## 🎯 Features

### Data Management

- ✅ **Configuration-driven architecture** - Source configs in `data/sources/*.json`
- ✅ Download newspaper data from Zenodo
- ✅ MD5 checksum verification
- ✅ Automatic extraction of tar.gz archives
- ✅ Smart caching (won't re-download existing files)
- ✅ Progress bars for downloads and processing
- ✅ Automatic error correction for known data issues
- ✅ Status tracking and reporting
- ✅ Parallel downloads for faster multi-part downloads
- ✅ Configurable directories via environment variables

### Data Loading & Parsing

- ✅ **ALTO XML parsing** with multiprocessing
- ✅ **METS metadata integration** - Enriches data with newspaper title, dates, volumes
- ✅ **Line-level DataFrame output** with text coordinates and metadata
- ✅ **Resume functionality** - Skips already processed files
- ✅ **Auto-save to Parquet** - Compressed output with zstd
- ✅ **Polars DataFrames** - Fast, memory-efficient data structures
- ✅ **Automatic text block aggregation** - Groups related lines
- ✅ **Source configuration system** - Easy to add new newspaper sources
- ✅ **Image downloading** - Parallel downloads from METS XML references

### CLI Commands

#### Data Management

- `data list` - List all available dataset parts with metadata
- `data status` - Show download/extraction status (detailed with `--verbose`)
- `data download` - Download one or more parts (with `--all`, `--parallel`, `--no-extract`, `--no-fix` options)
- `data extract` - Extract already downloaded archives
- `data verify` - Verify MD5 checksums of downloaded files

#### Data Loading

- `data sources` - List available newspaper sources
- `data load` - Parse ALTO XML to Parquet with METS metadata
  - `--source <name>` - Source to load (required)
  - `--max-files <n>` - Limit number of files (for testing)
  - `--workers <n>` - Number of parallel workers
  - `--no-resume` - Reprocess all files (ignores already processed)
- `data load-status` - Check loading status for a source

#### Analysis

- `analyze extract-entities` - Extract named entities (persons, organizations, locations, events) using GLiNER
  - `--source <name>` - Source name (required)
  - `--input <path>` - Input parquet file (required)
  - `--text-column <col>` - Column containing text (default: "text")
  - `--id-column <col>` - Column to use as identifier (default: "text_block_id")
  - `--normalize/--no-normalize` - Normalize text before extraction (default: normalize)
  - `--model <name>` - GLiNER model from Hugging Face (default: "urchade/gliner_multi-v2.1")
  - `--labels <labels>` - Comma-separated entity labels (default: Person,Organisation,Ereignis,Ort)
  - `--threshold <float>` - Confidence threshold 0-1 (default: 0.5)
  - `--batch-size <n>` - Processing batch size (default: 32)
  - `--format <format>` - Output format: parquet, json, or both (default: both)

### Analysis Modules

- ✅ **Text processing utilities** - Aggregate text blocks and split into sentences (German)
- ✅ **Text normalization** - Normalize historical German text to modern orthography
- ✅ **ALTO XML parsing** - Extract text with coordinates from newspaper pages
- ✅ **METS metadata extraction** - Newspaper titles, dates, volumes, page counts
- ✅ **Entity extraction** - Named entity recognition using GLiNER (persons, organizations, locations, events)
- ✅ **Image downloading** - Download high-resolution page scans from METS references
- 📝 Text cleaning and preprocessing
- 📝 Concept extraction
- 📝 Emotion analysis
- 📝 Layout analysis
- 📝 Topic modeling

### Text Processing

Process newspaper text with utilities for aggregation, sentence splitting, and normalization:

```python
from newspaper_explorer.data.utils.text import (
    load_and_aggregate_textblocks,
    split_into_sentences,
    normalize_text,
    load_aggregate_and_split
)

# Load line-level parquet and aggregate into text blocks
blocks_df = load_and_aggregate_textblocks("data/raw/der_tag/text/der_tag_lines.parquet")

# Split German text into sentences (requires spacy)
sentences_df = split_into_sentences(blocks_df, text_column="text")

# Normalize historical German text to modern orthography (requires transformers)
normalized_df = normalize_text(
    sentences_df,
    text_column="sentence",
    model="19c"  # For 1780-1899 texts, use "20c" for 1900-1999
)

# Or do everything in one step
normalized_df = load_aggregate_and_split(
    "data/raw/der_tag/text/der_tag_lines.parquet",
    normalize=True,              # Enable normalization
    normalize_model="19c",       # Use 19th century model
    save_path="data/processed/sentences.parquet"
)
```

**Requirements for sentence splitting:**

```bash
pip install -e ".[nlp]"
python -m spacy download de_core_news_sm
```

**Requirements for text normalization:**

```bash
pip install -e ".[normalize]"
```

See [Normalization Guide](docs/NORMALIZATION.md) for detailed usage.

### Entity Extraction

Extract named entities from historical newspaper text using GLiNER:

```python
from newspaper_explorer.analysis.entities.extraction import EntityExtractor

# Initialize extractor
extractor = EntityExtractor(
    source_name="der_tag",
    model_name="urchade/gliner_multi-v2.1",
    labels=["Person", "Organisation", "Ereignis", "Ort"],
    threshold=0.5,
    batch_size=32
)

# Extract entities from text blocks or sentences
results = extractor.extract_and_save(
    input_path="data/processed/der_tag/text/textblocks.parquet",
    text_column="text",
    id_column="text_block_id",
    normalize=True,  # Normalize before extraction
    output_format="both"  # Save as parquet and json
)

# Results saved to results/{source}/entities/
# - entities_raw.parquet: All extracted entities with IDs
# - entities_grouped.json: Entities grouped by ID and label
```

**CLI usage:**

```bash
newspaper-explorer analyze extract-entities \
    --source der_tag \
    --input data/processed/der_tag/text/textblocks.parquet \
    --normalize \
    --threshold 0.6
```

**Requirements for entity extraction:**

```bash
pip install transformers torch
```

### Image Downloading

Download high-resolution page scans from METS XML references:

```python
from newspaper_explorer.data.utils.images import ImageDownloader

# Initialize downloader
downloader = ImageDownloader(
    source_name="der_tag",
    max_workers=8,  # Parallel downloads
    max_retries=3,
    timeout=30
)

# Find all METS files
mets_files = downloader.find_mets_files()
print(f"Found {len(mets_files)} METS files")

# Download images from specific METS file
from pathlib import Path
mets_file = Path("data/raw/der_tag/xml_ocr/1901/01/18010101.xml")
downloaded, failed = downloader.download_images_from_mets(mets_file)

# Or download all images from all METS files
stats = downloader.download_all_images(skip_existing=True)
print(f"Downloaded: {stats['downloaded']}")
print(f"Skipped: {stats['skipped']}")
print(f"Failed: {stats['failed']}")
```

**Image organization:**

Images are saved to `data/raw/{source}/images/{year}/{month}/` with filenames matching the METS file structure.

## 📊 Available Dataset

The "Der Tag" newspaper collection from Zenodo.

**Collection**: [Der Tag on Zenodo](https://zenodo.org/records/17232177)

Use `newspaper-explorer data list` to see all available dataset parts with years, sizes, and checksums.

## 🛠️ Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests (when implemented)
pytest

# Format code
black src/

# Lint code
ruff check src/
```

## 📝 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub.
