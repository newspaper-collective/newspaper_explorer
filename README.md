# Newspaper Explorer

> **Initially created at [culture.explore(data)](https://lab.sbb.berlin/culture-explore-data/) Hackathon**  
> *7-8 October 2025, Staatsbibliothek zu Berlin*

A toolkit for exploring historical newspapers through computational analysis. Built during a two-day hackathon as an **initial starting point for researchers, students, and cultural heritage professionals** to explore large newspaper datasets under a **unified interface** that combines traditional data analysis with modern AI approaches. This project serves as a showcase for different computational methods, from statistical analysis and named entity recognition to topic modeling, layout detection, and LLM-powered insights, demonstrating how diverse approaches can work together on cultural heritage data. It provides researchers with a **first step into unknown datasets**, generating visualizations and insights that surface patterns and possibilities before specific research questions have been formulated.

**The Challenge**: Historical newspaper archives contain millions of pages locked in ALTO XML format. Exploring this data requires downloading, parsing, cleaning, and analyzing—with no standardized tools connecting the raw data to various analysis methods (statistics, NLP, computer vision, LLMs).

**The Solution**: Newspaper Explorer provides a complete pipeline from raw digitized newspapers to actionable insights:
- **Unified data interface**: Query millions of text lines with SQL, Python, or CLI commands
- **Multiple analysis approaches**: Combine statistical analysis, entity extraction, topic modeling, layout detection, and LLM-powered insights
- **Reproducible workflows**: Configuration-driven architecture ensures consistent processing across datasets
- **Automatic error correction**: Fixes known dataset issues (mislabeled dates, incorrect file locations) during processing
- **Extensible design**: Add new analysis methods without rewriting data loading code

Built with open cultural data from the Stiftung Preußischer Kulturbesitz and the University of Oxford, this project unlocks stories hidden in digitized newspaper archives by bridging the gap between historical sources and computational methods.

**What you can do:**
- Download and process millions of OCR text lines from historical newspaper archives
- Extract named entities, analyze topics, and detect emotions in 19th/20th century texts
- Query multi-gigabyte datasets with SQL without loading them into memory
- Leverage LLMs for structured information extraction with type-safe outputs
- Detect layout components with computer vision (YOLOv11)
- Track complete data lineage from raw ALTO XML to analysis results
- Combine multiple analysis methods seamlessly through unified foreign keys

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Screenshots from the Hackathon

<!-- TODO: Add hackathon screenshots -->

<details>
<summary>📸 Click to view screenshots from culture.explore(data) hackathon (October 2025)</summary>

### Data Processing Pipeline
<!-- ![Data Pipeline](docs/images/hackathon/01-pipeline.png) -->
*Screenshot: Real-time processing of 134,867 ALTO XML files from "Der Tag" newspaper*

### Entity Extraction Dashboard
<!-- ![Entity Extraction](docs/images/hackathon/02-entities.png) -->
*Screenshot: Named entity recognition extracting persons, organizations, and locations*

### Query Interface
<!-- ![Query Interface](docs/images/hackathon/03-query.png) -->
*Screenshot: DuckDB SQL queries on multi-gigabyte Parquet files*

### Layout Analysis
<!-- ![Layout Analysis](docs/images/hackathon/04-layout.png) -->
*Screenshot: YOLOv11-based layout detection identifying headlines, text blocks, and images*

### Text Normalization
<!-- ![Text Normalization](docs/images/hackathon/05-normalization.png) -->
*Screenshot: Historical German text normalization (19th/20th century orthography)*

### LLM Integration
<!-- ![LLM Analysis](docs/images/hackathon/06-llm.png) -->
*Screenshot: Structured entity extraction with LLM prompts and Pydantic schemas*

### Timeline Visualization
<!-- ![Timeline](docs/images/hackathon/07-timeline.png) -->
*Screenshot: Entity mentions over time across newspaper issues*

</details>

---

## Key Features

- **Smart Data Pipeline**: Configuration-driven download → extract → parse → analyze workflow
- **High Performance**: Parallel processing with Polars DataFrames and DuckDB queries
- **LLM Integration**: Structured prompts for entity extraction, topic analysis, emotion detection
- **Query Engine**: DuckDB-based analysis layer for efficient multi-GB data queries
- **Foreign Keys**: Complete traceability from raw XML to analysis results
- **Resume Support**: Automatic tracking of processed files to avoid reprocessing
- **Image Support**: Download high-resolution newspaper page scans from METS references
- **Modular CLI**: Clean command structure for all operations

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Python API](#python-api)
- [Documentation](#documentation)
- [Available Datasets](#available-datasets)
- [Development](#development)
- [License](#license)

## Installation

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

# Install with all dependencies
uv pip install -e .

# Optional: Install development tools
uv pip install -e ".[dev]"
```

See **[Installation Guide](docs/INSTALL.md)** for detailed instructions and troubleshooting.

---

## Quick Start

**Complete workflow from download to analysis:**

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

# 8. Optional: Preprocess text (normalize, clean, lemmatize)
newspaper-explorer data preprocess --source der_tag --normalize --lemmatize
```

**Then analyze with Python** - see **[Python API Examples](docs/PYTHON_API.md)** for:
- Loading and querying data with Polars/DuckDB
- LLM analysis with structured outputs
- Entity extraction
- Text preprocessing

**Or explore via CLI** - see **[CLI Reference](docs/cli/CLI.md)** for all commands.

---

## Core Concepts

### Configuration-Driven Architecture

Sources are defined in `data/sources/{source}.json` with download URLs, file patterns, and metadata—a single source of truth for all operations.

### Data Pipeline

```
Download → Extract → Parse → Aggregate → Preprocess → Analyze
 (XML)      (ALTO)   (Parquet)  (Blocks)   (Clean)    (Results)
```

### Key Features

- **Line-Level + Text Blocks**: Parse creates line-level data (each text line from ALTO). Aggregate creates text blocks (coherent paragraphs).
- **METS Metadata**: Every issue has METS XML with metadata (title, date, volume) automatically merged with text data.
- **Foreign Keys**: Unified ID system (`source_id` → `issue_id` → `page_id` → `text_block_id` → `line_id`) enables complete traceability. See [ID Generation](docs/development/ID_GENERATION.md).
- **Resume Support**: Commands skip already-processed data automatically (use `--force` to override).
- **Query Layer**: DuckDB engine queries Parquet files directly—no loading into memory.

### Architecture Documentation

- **[Data Architecture](docs/data/DATA_ARCHITECTURE.md)** - Overall pipeline design
- **[Data Loader](docs/data/DATA_LOADER.md)** - ALTO/METS parsing, schemas, resume functionality
- **[ID Generation](docs/development/ID_GENERATION.md)** - Unified ID system and foreign keys
- **[Configuration](docs/development/CONFIGURATION.md)** - Config philosophy

---

## Python API

Use the newspaper explorer as a Python library:

```python
# Load parsed newspaper data
from newspaper_explorer.data.loading.loader import DataLoader
import polars as pl

loader = DataLoader(source_name="der_tag")
df = loader.load_source()
df_1901 = df.filter(pl.col("year") == 1901)

# Query with SQL (DuckDB)
from newspaper_explorer.analysis.query.engine import QueryEngine

engine = QueryEngine(source_name="der_tag")
result = engine.query("""
    SELECT year, COUNT(*) as lines
    FROM lines
    WHERE year BETWEEN 1900 AND 1905
    GROUP BY year
""")

# LLM analysis with structured output
from newspaper_explorer.llm.client import LLMClient
from newspaper_explorer.llm.prompts.entity_extraction import ENTITY_EXTRACTION
from newspaper_explorer.llm.schemas.entity_extraction import EntityResponse

text = "Kaiser Wilhelm II empfing Bernhard von Bülow in Berlin."
prompt = ENTITY_EXTRACTION.format(text=text)

with LLMClient() as client:
    response = client.complete(
        prompt=prompt["user"],
        response_schema=EntityResponse
    )
print(response.persons)  # ["Kaiser Wilhelm II", "Bernhard von Bülow"]
```

See **[Python API Examples](docs/PYTHON_API.md)** for comprehensive documentation with examples for:
- Data loading and schemas
- Query engine (DuckDB)
- LLM analysis
- Text preprocessing
- Entity extraction
- Advanced queries
- Working with images

---

## Documentation

### Getting Started
- **[Installation Guide](docs/INSTALL.md)** - Detailed setup for pip and uv
- **[Python API Examples](docs/PYTHON_API.md)** - Comprehensive Python library usage
- **[CLI Reference](docs/cli/CLI.md)** - Complete command reference with examples
- **[Quick Start Tutorial](docs/data/DATA.md)** - Download, extract, parse, analyze

### Data Pipeline
- **[Data Architecture](docs/data/DATA_ARCHITECTURE.md)** - Overall pipeline design
- **[Data Loader](docs/data/DATA_LOADER.md)** - ALTO/METS parsing, schemas, DataFrames
- **[Image Downloads](docs/data/IMAGES.md)** - High-resolution page scans
- **[Preprocessing](docs/preprocessing/NORMALIZATION.md)** - Historical text normalization

### Analysis
- **[Query Engine](docs/analysis/QUERY_ARCHITECTURE.md)** - DuckDB analytics on Parquet
- **[LLM Utilities](docs/analysis/LLM.md)** - Prompts, schemas, client usage
- **[Entity Extraction](docs/analysis/ENTITIES.md)** - Named entity recognition with GLiNER

### Development
- **[ID Generation](docs/development/ID_GENERATION.md)** - Unified ID system and foreign keys
- **[Configuration](docs/development/CONFIGURATION.md)** - Config philosophy and patterns
- **[Copilot Instructions](.github/copilot-instructions.md)** - Comprehensive coding guidelines

---

## Available Datasets

### Der Tag (1900-1920)

Historical German newspaper from the Staatsbibliothek zu Berlin collection, used during the hackathon as the primary dataset.

- **Collection**: [Der Tag on Zenodo](https://zenodo.org/records/17232177)
- **Coverage**: 1900-1920 (multiple archive parts, ~134k ALTO XML files)
- **Format**: ALTO XML (OCR fulltext) + METS XML (metadata)
- **Language**: German (historical orthography)

```bash
# List available sources
newspaper-explorer data list-sources

# Get detailed status
newspaper-explorer data info --source der_tag
```

### Adding New Sources

The system is designed to work with any ALTO/METS newspaper archive. Create `data/sources/{source_name}.json` with download URLs, file patterns, and metadata. The system automatically discovers new sources.

Perfect for other cultural heritage collections from libraries and archives!

See **[Data Management](docs/data/DATA.md)** for source configuration details.

## Development

```bash
# Clone and install with dev dependencies
git clone https://github.com/newspaper-collective/newspaper_explorer.git
cd newspaper_explorer
pip install -e ".[dev]"

# Run tests
pytest

# Format and lint
black src/ tests/
ruff check src/ tests/
```

### Code Principles

- **NO `__init__.py` files** - Use explicit imports
- **Polars, not Pandas** - For all DataFrame operations  
- **Configuration-driven** - Sources in `data/sources/{name}.json`
- **No legacy support** - Replace old patterns completely, no compatibility layers
- **Foreign keys** - All analysis results link back via `line_id`

See **[Copilot Instructions](.github/copilot-instructions.md)** for comprehensive coding guidelines.

### Contributing

Contributions welcome! This project prioritizes clean architecture over backwards compatibility, modern tools (Polars, DuckDB) over legacy approaches, and type safety with Pydantic schemas.

Submit Pull Requests or open issues for bugs and feature requests.

## Acknowledgments

**Hackathon**: This project was developed at [**culture.explore(data)**](https://cultureexploredata.de/), an open cultural data hackathon held on 7-8 October 2025 at the Staatsbibliothek zu Berlin.

**Organizers**:
- [Staatsbibliothek zu Berlin](https://staatsbibliothek-berlin.de/) - Stiftung Preußischer Kulturbesitz
- [Bodleian Libraries](https://www.bodleian.ox.ac.uk/) - University of Oxford

**Funding**: Supported by a GLAM-SPK Seed Funding grant.

**Dataset**: "Der Tag" newspaper collection provided by the Staatsbibliothek zu Berlin via [Zenodo](https://zenodo.org/records/17232177).

**Challenge**: *"Unlock hidden stories by rethinking cultural heritage and experimenting with digital forms, such as apps, games, storytelling, data visualizations or interactive exhibitions."*

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Built with**: Polars, DuckDB, Click, Pydantic, and modern Python tools.

**Issues**: [GitHub Issues](https://github.com/newspaper-collective/newspaper_explorer/issues)
