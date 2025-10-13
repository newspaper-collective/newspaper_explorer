# CLI Reference

Command-line interface for Newspaper Explorer.

## Installation

First, install the package in development mode:

```bash
pip install -e .
```

This will make the `newspaper-explorer` command available in your terminal.

## Commands

### Data Management

All data-related commands are under the `data` subcommand.

#### List Available Parts

```bash
newspaper-explorer data list
```

Shows all available dataset parts with their years, sizes, and MD5 checksums.

#### Check Status

```bash
# Basic status
newspaper-explorer data status

# Detailed status with paths and checksums
newspaper-explorer data status --verbose
```

Shows what's downloaded and extracted.

#### Download Data

```bash
# Download a single part
newspaper-explorer data download --part dertag_1900-1902

# Download multiple parts (comma-separated)
newspaper-explorer data download --parts dertag_1900-1902,dertag_1903-1905

# Download multiple parts in parallel (faster)
newspaper-explorer data download --parts dertag_1900-1902,dertag_1903-1905 --parallel

# Download all parts
newspaper-explorer data download --all

# Force re-download
newspaper-explorer data download --part dertag_1900-1902 --force

# Download without extracting
newspaper-explorer data download --part dertag_1900-1902 --no-extract

# Download without error fixes
newspaper-explorer data download --part dertag_1900-1902 --no-fix

# Control number of parallel workers
newspaper-explorer data download --all --parallel --max-workers 5
```

#### Extract Data

```bash
# Extract a downloaded part
newspaper-explorer data extract dertag_1900-1902

# Extract multiple parts
newspaper-explorer data extract dertag_1900-1902 dertag_1903-1905

# Extract without error fixes
newspaper-explorer data extract dertag_1900-1902 --no-fix
```

#### Verify Checksums

```bash
# Verify one part
newspaper-explorer data verify dertag_1900-1902

# Verify multiple parts
newspaper-explorer data verify dertag_1900-1902 dertag_1903-1905
```

## Help

Get help for any command:

```bash
# General help
newspaper-explorer --help

# Help for data commands
newspaper-explorer data --help

# Help for specific command
newspaper-explorer data download --help
```

## Examples

### Typical Workflow

```bash
# 1. See what's available
newspaper-explorer data list

# 2. Check current status
newspaper-explorer data status

# 3. Download and extract a specific time period
newspaper-explorer data download --parts dertag_1900-1902,dertag_1903-1905

# 4. Verify the downloads
newspaper-explorer data verify --parts dertag_1900-1902,dertag_1903-1905

# 5. Check status again
newspaper-explorer data status -v
```

### Download Everything

```bash
# Download and extract all parts (be patient - this is a lot of data!)
newspaper-explorer data download --all
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

See [DATA.md](DATA.md) for complete Python API documentation.
