# Python API Examples

This guide shows how to use the newspaper explorer as a Python library for data loading, querying, and analysis.

## Table of Contents

- [Data Loading](#data-loading)
- [Query Engine (DuckDB)](#query-engine-duckdb)
- [LLM Analysis](#llm-analysis)
- [Text Preprocessing](#text-preprocessing)
- [Entity Extraction](#entity-extraction)
- [Advanced Queries](#advanced-queries)

---

## Data Loading

Load parsed newspaper data as Polars DataFrames:

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

**Schema Documentation**: See [Data Loader](data/DATA_LOADER.md) for complete schema details.

---

## Query Engine (DuckDB)

Use SQL to query Parquet files efficiently without loading into memory:

```python
from newspaper_explorer.analyze.query.engine import QueryEngine

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

See **[Query Architecture](analysis/QUERY_ARCHITECTURE.md)** for advanced examples.

---

## LLM Analysis

Extract structured information using LLM prompts:

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

See **[LLM Utilities Guide](analysis/LLM.md)** for all available prompts and schemas.

---

## Text Preprocessing

Process historical German text with normalization, lemmatization, and sentence splitting:

```python
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

**CLI Alternative:**
```bash
newspaper-explorer data preprocess --source der_tag --normalize --lemmatize
```

See **[Normalization Guide](preprocessing/NORMALIZATION.md)** for details on historical text processing.

---

## Entity Extraction

Extract named entities using GLiNER:

```python
from newspaper_explorer.analyze.entities.extraction import EntityExtractor

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

# Results saved in: results/der_tag/entities/gliner/
print(f"Extracted {len(results['entities'])} entities")
```

**Requirements:**
```bash
pip install transformers torch
```

**CLI Alternative:**
```bash
newspaper-explorer analyze entities extract \
    --source der_tag \
    --input data/raw/der_tag/text/der_tag_text_blocks.parquet \
    --normalize \
    --threshold 0.6 \
    --labels Person,Organisation,Ort,Ereignis
```

See **[Entity Extraction Guide](analysis/ENTITIES.md)** for detailed documentation.

---

## Advanced Queries

### Find mentions with context

```python
from newspaper_explorer.analyze.query.engine import QueryEngine

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
```

### Aggregate entity counts by year

```python
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
```

### Compare extraction methods

```python
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

---

## Working with Images

Download and reference newspaper page images:

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

See **[Image Downloads](data/IMAGES.md)** for more details.

---

## Archive Downloads

Download newspaper archives from Zenodo:

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

---

## Foreign Keys and Traceability

All data maintains foreign key relationships for complete traceability:

```python
# Every line has foreign keys to parent entities
line = lines_df.filter(pl.col("line_id") == "3074409-X_1901-01-08_006_1_001_r_1_1_tl_1").first()

print(line["source_id"])      # "3074409-X" (ZDB ID)
print(line["issue_id"])       # "3074409-X_1901-01-08_006_1"
print(line["page_id"])        # "3074409-X_1901-01-08_006_1_001"
print(line["text_block_id"])  # "3074409-X_1901-01-08_006_1_001_r_1_1"
print(line["line_id"])        # Full ID

# Query all data from a specific issue
issue_data = lines_df.filter(
    pl.col("issue_id") == "3074409-X_1901-01-08_006_1"
)
```

See **[ID Generation](development/ID_GENERATION.md)** for the complete ID system specification.

### Reconstructing Filenames from IDs

When you need to reconstruct original ALTO/METS filenames from `source_id`:

```python
from newspaper_explorer.data.utils.ids import source_id_to_filename_prefix

# source_id uses hyphen: "3074409-X"
# But filenames don't: "3074409X_1902-09-05_000_415_H_2_005.xml"

source_id = "3074409-X"
filename_prefix = source_id_to_filename_prefix(source_id)  # "3074409X"

# Build ALTO filename
alto_filename = f"{filename_prefix}_1902-09-05_000_415_H_2_005.xml"
# Result: "3074409X_1902-09-05_000_415_H_2_005.xml"
```

**Why the difference?**
- `source_id` follows standardized format with hyphens for consistency across the codebase
- Original filenames use the raw format without hyphens as provided in the archives
- Use `source_id_to_filename_prefix()` when you need to match/reconstruct filenames

---

## Tips and Best Practices

### Use Polars, Not Pandas

```python
import polars as pl  # Fast, memory-efficient
import pandas as pd  # Not used in this project

# Polars syntax
df = pl.read_parquet("data.parquet")
filtered = df.filter(pl.col("year") == 1901)
grouped = df.group_by("text_block_id").agg(pl.col("text").str.concat(" "))
```

### Query Large Datasets with DuckDB

```python
# Use QueryEngine for large datasets (avoids memory issues)
engine = QueryEngine(source_name="der_tag")
result = engine.query("SELECT * FROM lines WHERE year = 1901")

# Avoid loading entire datasets into memory
df = pl.read_parquet("huge_file.parquet")  # Memory intensive!
```

### Leverage Foreign Keys

```python
# Join using foreign keys
result = engine.query("""
    SELECT l.*, e.entity_text
    FROM lines l
    JOIN entities e ON l.line_id = e.line_id
""")

# Filter by hierarchical IDs
issue_data = df.filter(pl.col("issue_id") == "...")
page_data = df.filter(pl.col("page_id").str.starts_with("3074409-X_1901-01-08"))
```

### Preserve Foreign Keys in Analysis

```python
# When creating analysis results, always preserve FKs
analysis_results = df.select([
    # Keep all foreign keys
    "line_id", "text_block_id", "page_id", "issue_id", "source_id",
    # Keep metadata
    "date", "newspaper_title",
    # Add your analysis columns
    pl.col("text").alias("processed_text"),
    # ... your results
])
```

---

## See Also

- **[CLI Reference](cli/CLI.md)** - Command-line interface documentation
- **[Data Architecture](data/DATA_ARCHITECTURE.md)** - Pipeline overview
- **[Query Architecture](analysis/QUERY_ARCHITECTURE.md)** - Advanced querying
- **[LLM Utilities](analysis/LLM.md)** - All prompts and schemas
