# Newspaper Explorer

A tool for exploring and analyzing historical newspaper data from the "Der Tag" collection.

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
```

See [INSTALL.md](INSTALL.md) for detailed installation instructions.

## 🚀 Quick Start

### Using the CLI

```bash
# List available dataset parts
newspaper-explorer data list

# Check current status
newspaper-explorer data status

# Download and extract a specific time period
newspaper-explorer data download --part dertag_1900-1902

# Download multiple parts
newspaper-explorer data download --parts dertag_1900-1902,dertag_1903-1905

# Download all parts at once
newspaper-explorer data download --all

# Download without extraction
newspaper-explorer data download --part dertag_1900-1902 --no-extract

# Download without automatic error fixes
newspaper-explorer data download --part dertag_1900-1902 --no-fix

# Parallel downloads (faster for multiple parts)
newspaper-explorer data download --parts dertag_1900-1902,dertag_1903-1905 --parallel

# Extract already downloaded data
newspaper-explorer data extract --part dertag_1900-1902

# Verify checksums
newspaper-explorer data verify --part dertag_1900-1902

# Get help
newspaper-explorer --help
newspaper-explorer data download --help
```

### Using the Python API

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

# Download without extraction
downloader.download_part('dertag_1900-1902')

# Extract already downloaded data
downloader.extract_part('dertag_1900-1902')

# Check status
downloader.print_status_summary()
```

## 📚 Documentation

- [CLI Reference](docs/CLI.md) - Complete CLI command reference
- [Data Management Guide](docs/DATA.md) - Detailed guide for data downloading, extraction, and source configuration
- [Installation Guide](docs/INSTALL.md) - Setup instructions for pip and uv
- [Configuration Philosophy](docs/CONFIGURATION_PHILOSOPHY.md) - Design decisions for config vs CLI flags

## 🗂️ Project Structure

```
newspaper_explorer/
├── src/
│   └── newspaper_explorer/
│       ├── main.py              # Main CLI entry point
│       ├── cli/                 # CLI command modules (no __init__.py)
│       │   └── data.py          # Data management commands
│       ├── data/                # Data handling (no __init__.py)
│       │   ├── download.py      # Download/extract functionality
│       │   ├── fixes.py         # Error correction utilities
│       │   ├── loading.py       # Data loading (placeholder)
│       │   └── cleaning.py      # Data cleaning (placeholder)
│       ├── analysis/            # Analysis modules (placeholders)
│       │   ├── concepts/
│       │   ├── emotions/
│       │   ├── entities/
│       │   ├── layout/
│       │   └── topics/
│       ├── ui/                  # UI components (future)
│       └── utils/               # Utilities (no __init__.py)
│           └── config.py        # Configuration management
├── data/
│   ├── sources/
│   │   └── der_tag.json         # Dataset configuration with checksums
│   ├── downloads/               # Downloaded .tar.gz files (preserved)
│   ├── extracted/               # Temp extraction (auto-cleaned, kept empty)
│   └── raw/                     # Organized data by year
├── results/                     # Analysis results and outputs
├── docs/                        # Documentation
└── tests/                       # Test suite
```

**Important Notes:**

- This project uses **explicit imports** without `__init__.py` files
- Use full module paths: `from newspaper_explorer.utils.config import get_config`
- See `.github/copilot-instructions.md` for coding guidelines
- Analysis modules (`analysis/`, `data/cleaning.py`, `data/loading.py`) are placeholders for future development

## 🎯 Features

### Data Management

- ✅ Download newspaper data from Zenodo
- ✅ MD5 checksum verification
- ✅ Automatic extraction of tar.gz archives
- ✅ Smart caching (won't re-download existing files)
- ✅ Progress bars for downloads
- ✅ Automatic error correction for known data issues
- ✅ Status tracking and reporting
- ✅ Parallel downloads for faster multi-part downloads
- ✅ Configurable directories via environment variables

### CLI Commands

- `data list` - List all available dataset parts with metadata
- `data status` - Show download/extraction status (detailed with `--verbose`)
- `data download` - Download one or more parts (with `--all`, `--parallel`, `--no-extract`, `--no-fix` options)
- `data extract` - Extract already downloaded archives
- `data verify` - Verify MD5 checksums of downloaded files

### Analysis Modules (Planned)

- 📝 Text cleaning and normalization
- 📝 Data loading from XML/OCR files
- 📝 Concept extraction
- 📝 Emotion analysis
- 📝 Entity recognition
- 📝 Layout analysis
- 📝 Topic modeling

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
