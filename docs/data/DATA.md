# Data Management Guide

Comprehensive guide for downloading and managing newspaper data from Zenodo.

## Quick Start

### Using the CLI (Recommended)

```bash
# List available dataset parts
newspaper-explorer data list

# Download and extract a specific time period
newspaper-explorer data download --part dertag_1900-1902

# Check status
newspaper-explorer data status
```

### Using the Python API

```python
from newspaper_explorer.data.download import ZenodoDownloader

# Initialize downloader
downloader = ZenodoDownloader()

# Download and extract a single part
downloader.download_and_extract(['dertag_1900-1902'])

# Download multiple parts in parallel (faster)
downloader.download_and_extract(
    ['dertag_1900-1902', 'dertag_1903-1905'],
    parallel=True,
    max_workers=3
)

# Download without error fixes
downloader.download_and_extract(['dertag_1900-1902'], fix_errors=False)

# Check status
downloader.print_status_summary()
```

## Available Dataset Parts

The "Der Tag" Zenodo collection is divided into parts covering different time periods.

Use `newspaper-explorer data list` to see all available parts with their:

- Year ranges
- File sizes
- MD5 checksums (for verification)

## Data Structure

The data is organized by dataset name and data type:

```
data/
├── sources/            # Configuration files
│   └── der_tag.json    # Zenodo links, checksums, metadata
├── downloads/          # Downloaded .tar.gz archives (preserved)
│   └── der_tag/
│       └── xml_ocr/
│           ├── dertag_1900-1902.tar.gz
│           └── ...
└── raw/               # Organized newspaper data by year (after extraction)
    └── der_tag/
        └── xml_ocr/
            ├── 1900/   # Year directories with newspaper files
            ├── 1901/
            └── ...
```

**Important Notes:**

- **Automatic Cleanup**: Temporary extraction directories are automatically cleaned up after organizing data into `raw/`, including empty parent directories
- **Year-Based Organization**: Data is reorganized from the archive structure into year-based directories
- **Space Efficiency**: Only the original `.tar.gz` files and organized `raw/` data are kept - `extracted/` directory remains completely empty
- All paths are configurable via `.env` file (see [Configuration](#configuration))

## Error Correction

The downloader includes automatic error correction for known data issues in the Zenodo collection.

### Current Fixes

**dertag_1900-1902 Part:**

- Fixes mislabeled files in the 1900 directory
- Known issue: Some files labeled as 1900-01-02 are actually from 1902-01-02
- Automatically relocates files to correct year directories
- Updates metadata (METS XML) with correct dates

### Disabling Error Corrections

If you want to download data without automatic fixes:

```bash
# CLI
newspaper-explorer data download --part dertag_1900-1902 --no-fix

# Python API
downloader.download_and_extract(['dertag_1900-1902'], fix_errors=False)
```

### Adding New Corrections

To add fixes for newly discovered issues:

1. Edit `src/newspaper_explorer/data/fixes.py`
2. Add your fix method to the `DataFixer` class
3. Call it from the `apply_fixes()` method with appropriate conditions
4. Follow the existing pattern for part-specific corrections

## Configuration

### Directory Configuration

Configure data directories in your `.env` file:

```bash
# Main data directory
DATA_DIR=data

# Specific directories (defaults to DATA_DIR subdirectories)
DOWNLOAD_DIR=data/downloads
EXTRACTED_DIR=data/extracted
SOURCES_DIR=data/sources
```

All paths can be absolute or relative to the project root.

### Dataset Source Configuration

Dataset links and checksums are stored in `data/sources/der_tag.json`.

**Configuration File Structure:**

```json
{
  "collection_id": "17232177",
  "collection_url": "https://zenodo.org/records/17232177",
  "description": "Der Tag newspaper collection from Zenodo",
  "dataset_name": "der_tag",
  "data_type": "xml_ocr",
  "parts": [
    {
      "name": "dertag_1900-1902",
      "url": "https://zenodo.org/records/17232177/files/dertag_1900-1902.tar.gz?download=1",
      "years": "1900-1902",
      "md5": "83408fdd6963bd12e2edd59809d2571f",
      "size": "1.4 GB"
    }
    // ... more parts
  ]
}
```

**Key Fields:**

- `collection_id` - Zenodo record ID
- `collection_url` - URL to the Zenodo collection page
- `dataset_name` - Name used for organizing files (e.g., "der_tag")
- `data_type` - Type of data (e.g., "xml_ocr")
- `parts` - Array of dataset parts with:
  - `name` - Identifier for the part
  - `url` - Direct download URL
  - `years` - Year range covered
  - `md5` - MD5 checksum for verification (optional but recommended)
  - `size` - Human-readable file size (optional)

**MD5 Checksums:**

MD5 checksums verify data integrity after download:

```bash
# Calculate MD5 on Windows (PowerShell)
Get-FileHash -Algorithm MD5 filename.tar.gz

# Calculate MD5 on Linux/Mac
md5sum filename.tar.gz
```

**Adding New Sources:**

1. Create a new JSON file in `data/sources/` (e.g., `newspaper_name.json`)
2. Follow the structure above
3. Include MD5 checksums when available
4. Update the downloader code to support the new source if needed

## Advanced Usage

### Parallel Downloads

Download multiple parts simultaneously for faster processing:

```bash
# CLI with parallel flag
newspaper-explorer data download --parts dertag_1900-1902,dertag_1903-1905 --parallel

# Python API
downloader.download_parts_parallel(
    ['dertag_1900-1902', 'dertag_1903-1905'],
    max_workers=3
)
```

### Verifying Downloads

Ensure data integrity by verifying MD5 checksums:

```bash
newspaper-explorer data verify --part dertag_1900-1902
```

### Extraction Status

Get detailed information about what's been downloaded and extracted:

```bash
# Basic status
newspaper-explorer data status

# Detailed status with paths and checksums
newspaper-explorer data status --verbose
```

## API Reference

### ZenodoDownloader Class

```python
from newspaper_explorer.data.download import ZenodoDownloader

downloader = ZenodoDownloader(data_dir=None)
```

**Methods:**

- `list_available_parts()` - Get list of all available dataset parts
- `download_part(part_name, force_redownload=False)` - Download a single part
- `extract_part(part_name, fix_errors=True)` - Extract a downloaded archive
- `download_and_extract(part_names=None, fix_errors=True, parallel=False)` - Download and extract parts
- `download_parts_parallel(part_names, force_redownload=False, max_workers=3)` - Parallel downloads
- `get_extraction_status()` - Get detailed status dictionary
- `print_status_summary()` - Print human-readable status
