# Newspaper Explorer

> **Initially created at [culture.explore(data)](https://lab.sbb.berlin/culture-explore-data/) Hackathon**  
> *7-8 October 2025, Staatsbibliothek zu Berlin*

A toolkit for exploring historical newspapers through computational analysis. Historical newspaper archives contain thousands of pages locked in ALTO XML format. Exploring this data requires downloading, parsing, cleaning, and analyzing, yet standardized tools connecting raw data to statistical, NLP, computer vision, and LLM workflows remain scarce. 

Built during a two-day hackathon, this project provides an **initial starting point for researchers, students, and cultural heritage professionals** to explore large newspaper datasets under a **unified interface** combining traditional data analysis with modern AI approaches. It showcases how diverse computational methods (statistical analysis, named entity recognition, topic modeling, layout detection, and LLM-powered insights) can work together on cultural heritage data. The toolkit gives researchers a **first step into unknown datasets**, generating visualizations and insights that surface patterns and possibilities before specific research questions have been formulated. 

During the hackathon, we extensively experimented with various data analysis approaches, testing GLiNER for named entity extraction, Google Gemini for knowledge graph generation, YOLOv11 for layout detection, LLM-based topic modeling via OpenRouter, and a custom emotion classification model from Universität Würzburg. However, this exploration quickly revealed the central challenge: the "Der Tag" dataset for the years 1900-1920 contains **~148,000 XML files** with **61+ million text lines** from **135,000 high-resolution page images**, totaling over **10 GB of compressed XML archives** and **200 GB of JPEG images**. Working with a dataset of this scale proved a significant challenge during the two days of the hackathon, so some analyses were performed on data subsets due to computational and time constraints. Future development will focus on optimizing the codebase to efficiently process the complete dataset.

Built with open cultural data from the Stiftung Preußischer Kulturbesitz, this project unlocks stories hidden in digitized newspaper archives by bridging the gap between historical sources and computational methods.


**Our Vision**: Newspaper Explorer aims to provide a complete, scalable pipeline from raw digitized newspapers to actionable insights:
- **Unified data interface**: Load and query millions of text lines using Polars DataFrames and DuckDB SQL
- **Multiple analysis approaches**: Framework for combining statistical analysis, entity extraction, topic modeling, layout detection, and LLM-powered insights
- **Reproducible workflows**: Configuration-driven architecture with source definitions in JSON for consistent processing
- **Automatic error correction**: Built-in fixes for known dataset issues (mislabeled dates, incorrect file locations)
- **Extensible design**: Modular structure allows adding new analysis methods independently of data loading


**What's currently implemented:**
- Download and parse ALTO/METS XML archives into structured Parquet files
- Load and query multi-gigabyte datasets with Polars DataFrames and DuckDB SQL
- Preprocess historical German text with normalization, cleaning, and lemmatization
- Use LLM client with structured prompts and Pydantic schemas for type-safe outputs
- Track complete data lineage through unified ID system (source → issue → page → text block → line)
- Download high-resolution page images from METS references
- Validate data quality with ALTO-METS consistency checks

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Hackathon Demo: Streamlit UI

During the two-day hackathon, we built a Streamlit-based web interface to showcase the first results of our various analysis approaches on the "Der Tag" dataset. These screenshots demonstrate the exploratory analysis capabilities we tested, it helped us identify computational challenges and informed the development of the current CLI-based architecture.

<details>
<summary>📸 View screenshots from the Streamlit demo (October 2025)</summary>

### Entity Extraction Dashboard
![Entity Extraction](docs/hackathon/screenshots/01_entities.png)
*Screenshot: Named entity recognition extracting persons, organizations, locations, events with GLiNER*

### Concept Extraction
![Concept Extraction](docs/hackathon/screenshots/02_concepts.png)
*Screenshot: Knowledge Graph created with automated concept extraction from newspaper text using Google Gemini*

### Layout Analysis
![Layout Analysis](docs/hackathon/screenshots/03_layout.png)
*Screenshot: YOLOv11-based layout detection identifying headlines, text blocks, and images*

### Layout Pictures Gallery
![Layout Pictures Gallery](docs/hackathon/screenshots/04_layout_pictures_gallery.png)
*Screenshot: Gallery view of detected image regions from newspaper pages*

### Layout Pictures with Captions
![Layout Pictures Detail](docs/hackathon/screenshots/05_layout_pictures_with_captions_detail.png)
*Screenshot: Detailed view of extracted images with their captions*

### Search and Word Cloud
![Search and Word Cloud](docs/hackathon/screenshots/06_search_and_word_cloud.png)
*Screenshot: Caption search interface with word cloud visualization settings*

### Entities Word Cloud
![Entities Word Cloud](docs/hackathon/screenshots/07_entities_word_cloud.png)
*Screenshot: Word cloud visualization of extracted named entities*

### Topic Modeling
![Topics Overview](docs/hackathon/screenshots/08_topics_01.png)
*Screenshot: Topic modeling analysis showing thematic clusters, topics where extracted using LLMs via OpenRouter*

### Topic Modeling
![Topics Temporal](docs/hackathon/screenshots/09_topics_02.png)
*Screenshot: Topic evolution over time across newspaper issues*

### Topic Modeling
![Topics Details](docs/hackathon/screenshots/10_topics_03.png)
*Screenshot: Detailed view of individual topics with representative texts*

### Emotion Analysis
![Emotions Distribution](docs/hackathon/screenshots/12_emotions_02.png)
*Screenshot: Distribution of emotional content in the corpus, a classification model developed at Universität Würzburg was used (https://github.com/LeKonArD/Gattungen_und_Emotionen_dhd2023/tree/main)*

### Emotion Analysis
![Emotions Timeline](docs/hackathon/screenshots/13_emotions_03.png)
*Screenshot: Emotional patterns over time*

### Emotion Analysis
![Emotions Details](docs/hackathon/screenshots/14_emotions_04.png)
*Screenshot: Detailed emotion analysis with text examples*

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
from newspaper_explorer.analyze.query.engine import QueryEngine

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
