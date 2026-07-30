# CLI Reference

Command-line interface for Newspaper Explorer.

## Installation

First, install the package in development mode:

```bash
pip install -e .
```

This will make the `newspaper-explorer` command available in your terminal.

## Command Structure

The CLI uses a hybrid structure with grouped and flat commands:

```
newspaper-explorer
├── data                        # Data management
│   ├── text                    # Text data pipeline
│   │   ├── download            # Download archives from Zenodo
│   │   ├── unpack              # Extract downloaded archives
│   │   ├── verify              # Verify checksums
│   │   ├── parse               # Parse ALTO XML to Parquet
│   │   └── aggregate           # Aggregate lines into text blocks
│   ├── images                  # Image data pipeline
│   │   ├── download            # Download page images from METS
│   │   └── build-index         # Build image index
│   ├── validation              # Data validation
│   │   ├── validate-alto-mets  # Validate ALTO/METS consistency
│   │   ├── images              # Validate downloaded images
│   │   └── generate-wordlist   # Generate German wordlist
│   ├── list-sources            # List available sources (flat)
│   ├── info                    # Show source status (flat)
│   ├── analyze-chars           # Character analysis (flat)
│   ├── analyze-tokens          # Token analysis (flat)
│   ├── longest-tokens          # Find longest tokens (flat)
│   ├── preprocess              # Run preprocessing pipeline (flat)
│   └── list-pipelines          # List preprocessing pipelines (flat)
└── analyze                     # Analysis commands
    ├── entities                # Entity extraction
    ├── emotions                # Emotion classification
    ├── topics                  # Topic modeling
    ├── keywords                # Keyword extraction
    └── layout                  # Layout detection
```

## Commands

### Data Management

#### Text Pipeline

Text data commands are grouped under `data text`:

```bash
# Download a single part
newspaper-explorer data text download --part dertag_1900-1902

# Download multiple parts in parallel
newspaper-explorer data text download --parts dertag_1900-1902,dertag_1903-1905 --parallel

# Download all parts
newspaper-explorer data text download --all

# Force re-download
newspaper-explorer data text download --part dertag_1900-1902 --force

# Unpack archives
newspaper-explorer data text unpack --source der_tag

# Verify checksums
newspaper-explorer data text verify --source der_tag

# Parse ALTO XML to Parquet
newspaper-explorer data text parse --source der_tag

# Aggregate lines into text blocks
newspaper-explorer data text aggregate --source der_tag
```

#### Image Pipeline

Image commands are grouped under `data images`:

```bash
# Download high-resolution page images
newspaper-explorer data images download --source der_tag

# Customize parallel workers (default: 8)
newspaper-explorer data images download --source der_tag --max-workers 16

# Build image index
newspaper-explorer data images build-index --source der_tag
```

Images are stored in `data/raw/{source}/images/` with the same directory structure as XML files.

See [IMAGES.md](../data/IMAGES.md) for detailed image downloading documentation.

#### Validation

Validation commands are grouped under `data validation`:

```bash
# Validate ALTO/METS consistency
newspaper-explorer data validation validate-alto-mets --source der_tag

# Validate downloaded images
newspaper-explorer data validation images --source der_tag

# Generate German wordlist for quality validation
newspaper-explorer data validation generate-wordlist --source-type hunspell
```

#### Flat Data Commands

Commonly used commands are available directly under `data`:

```bash
# List available sources
newspaper-explorer data list-sources

# Show detailed source status
newspaper-explorer data info --source der_tag

# Character analysis
newspaper-explorer data analyze-chars --source der_tag

# Token analysis
newspaper-explorer data analyze-tokens --source der_tag

# Find longest tokens
newspaper-explorer data longest-tokens --source der_tag

# Run preprocessing pipeline
newspaper-explorer data preprocess --source der_tag --normalize --lemmatize

# List available preprocessing pipelines
newspaper-explorer data list-pipelines
```

## Help

Get help for any command:

```bash
# General help
newspaper-explorer --help

# Help for data commands
newspaper-explorer data --help

# Help for text pipeline
newspaper-explorer data text --help

# Help for specific command
newspaper-explorer data text download --help
```

## Examples

### Typical Workflow

```bash
# 1. See what's available
newspaper-explorer data list-sources

# 2. Check current status
newspaper-explorer data info --source der_tag

# 3. Download and extract a specific time period
newspaper-explorer data text download --parts dertag_1900-1902,dertag_1903-1905

# 4. Verify the downloads
newspaper-explorer data text verify --source der_tag

# 5. Parse to Parquet
newspaper-explorer data text parse --source der_tag

# 6. Aggregate into text blocks
newspaper-explorer data text aggregate --source der_tag

# 7. Download images
newspaper-explorer data images download --source der_tag

# 8. Run preprocessing
newspaper-explorer data preprocess --source der_tag --normalize --lemmatize
```

### Download Everything

```bash
# Download and extract all parts (be patient - this is a lot of data!)
newspaper-explorer data text download --all
```

## Python API

You can also use the downloader programmatically:

```python
from newspaper_explorer.data.download import ZenodoDownloader

# Initialize
downloader = ZenodoDownloader()

# List parts
parts = downloader.list_available_parts()

# Download and extract
downloader.download_and_extract(['dertag_1900-1902'])

# Parallel downloads
downloader.download_and_extract(
    ['dertag_1900-1902', 'dertag_1903-1905'],
    parallel=True
)

# Status
downloader.print_status_summary()
```

See [DATA.md](../data/DATA.md) for complete Python API documentation.
