# ID Generation System

This document describes the unified ID system used throughout the newspaper explorer codebase. All data from parsing (ALTO/METS) through analysis maintains a consistent foreign key hierarchy enabling complete traceability and efficient querying.

**Module**: `newspaper_explorer.data.utils.ids`

---

## Design Principles

1. **Hierarchical** - IDs form a clear parent-child relationship
2. **Stable** - IDs don't change across re-runs (UUID-based where needed)
3. **Traceable** - Any analysis result can be linked back to source data
4. **Efficient** - Foreign keys enable fast filtering and joining
5. **Human-readable** - IDs contain semantic information (source name, date, page)

---

## ID Hierarchy

```
source_id (e.g., "der_tag")
  └── issue_id (e.g., "der_tag_1902-09-05_415_2")
      └── page_id (e.g., "der_tag_1902-09-05_415_2_005")
          ├── text_block_id (e.g., "der_tag_1902-09-05_415_2_005_TB_1")
          │   └── line_id (e.g., "der_tag_1902-09-05_415_2_005_TB_1_TL_1")
          ├── detection_id (e.g., "der_tag_1902-09-05_415_2_005_headline_a3f9c2")
          └── article_id (e.g., "der_tag_1902-09-05_415_2_005_art_b7e4d1")
```

---

## ID Specifications

### 1. source_id

**Type**: String (primary key for sources)

- **Format**: `{source_name}`
- **Example**: `der_tag`
- **Purpose**: Identify the newspaper source (human-readable, matches directory structure)
- **Generation**: Same as source name from configuration
- **Stored in**: All parquet files

```python
from newspaper_explorer.data.utils.ids import generate_source_id

source_id = generate_source_id("der_tag")  # Returns "der_tag"
```

**Note**: The source_id uses the canonical source name (e.g., "der_tag") which is human-readable and matches the directory structure. The ZDB ID is stored separately in source config for provenance but is NOT used in ID generation.

**Filename Compatibility**: When reconstructing ALTO/METS filenames from ZDB-style IDs, use `source_id_to_filename_prefix()` to remove hyphens:

```python
from newspaper_explorer.data.utils.ids import source_id_to_filename_prefix

# Only needed for ZDB-style IDs (removes hyphens)
source_id_to_filename_prefix("3074409-X")  # Returns "3074409X"
source_id_to_filename_prefix("der_tag")    # Returns "der_tag" (unchanged)
```

---

### 2. issue_id

**Type**: String (foreign key to source_id)

- **Format**: `{source}_{YYYY-MM-DD}_{issue:03d}_{edition}`
- **Example**: `der_tag_1902-09-05_415_2`
- **Components**:
  - `source`: Source name (e.g., "der_tag")
  - `YYYY-MM-DD`: Publication date
  - `issue`: Sequential publication number (e.g., 415)
  - `edition`: Edition of the day (1=morning, 2=midday, 3=evening)
- **Purpose**: Group all pages from the same newspaper issue/edition
- **Stored in**: All parquet files

```python
from newspaper_explorer.data.utils.ids import generate_issue_id
from datetime import datetime

issue_id = generate_issue_id(
    source="der_tag",
    date=datetime(1902, 9, 5),
    issue_number=415,
    edition=2
)
# Returns: "der_tag_1902-09-05_415_2"
```

**Note**: Multiple editions can share the same `issue_number`. For example, morning (edition=1) and midday (edition=2) might both be issue 37, while evening (edition=3) is issue 38.

---

### 3. page_id

**Type**: String (foreign key to issue_id)

- **Format**: `{source}_{YYYY-MM-DD}_{issue:03d}_{edition}_{page:03d}`
- **Example**: `der_tag_1902-09-05_415_2_005`
- **Purpose**: Unique identifier for a newspaper page
- **Stored in**: All parquet files

```python
from newspaper_explorer.data.utils.ids import generate_page_id
from datetime import datetime

page_id = generate_page_id(
    source="der_tag",
    date=datetime(1902, 9, 5),
    issue_number=415,
    edition=2,
    page_number=5
)
# Returns: "der_tag_1902-09-05_415_2_005"
```

---

### 4. text_block_id

**Type**: String (foreign key to page_id)

- **Format**: `{page_id}_{block_id}`
- **Example**: `der_tag_1902-09-05_415_2_005_TB_1`
- **Purpose**: Unique text block identifier across entire dataset
- **Stored in**: `lines.parquet`, `text_blocks.parquet`, detection results

```python
from newspaper_explorer.data.utils.ids import generate_text_block_id

text_block_id = generate_text_block_id(
    page_id="der_tag_1902-09-05_415_2_005",
    block_id="TB_1"
)
# Returns: "der_tag_1902-09-05_415_2_005_TB_1"
```

**Note**: `block_id` comes from ALTO XML and varies by document structure (e.g., `TB_1`, `r_1_1`).

---

### 5. line_id

**Type**: String (foreign key to text_block_id)

- **Format**: `{text_block_id}_{line_id}`
- **Example**: `der_tag_1902-09-05_415_2_005_TB_1_TL_1`
- **Purpose**: Unique text line identifier across entire dataset
- **Stored in**: `lines.parquet`, entity extraction results

```python
from newspaper_explorer.data.utils.ids import generate_line_id

line_id = generate_line_id(
    text_block_id="der_tag_1902-09-05_415_2_005_TB_1",
    line_id="TL_1"
)
# Returns: "der_tag_1902-09-05_415_2_005_TB_1_TL_1"
```

---

### 6. detection_id

**Type**: String (UUID-based, foreign key to page_id)

- **Format**: `{page_id}_{class}_{uuid_short}`
- **Example**: `der_tag_1902-09-05_415_2_005_headline_a3f9c2`
- **Purpose**: Stable unique ID for detected layout elements
- **Stored in**: Detection parquet files, headline results

```python
from newspaper_explorer.data.utils.ids import generate_detection_id

detection_id = generate_detection_id(
    page_id="der_tag_1902-09-05_415_2_005",
    class_name="headline"
)
# Returns: "der_tag_1902-09-05_415_2_005_headline_a3f9c2"
```

**Important**: Uses 6-character UUID suffix for uniqueness. The same visual element gets different IDs in different runs, but this is acceptable as detections are typically regenerated.

---

### 7. article_id

**Type**: String (UUID-based, foreign key to page_id)

- **Format**: `{page_id}_art_{uuid_short}`
- **Example**: `der_tag_1902-09-05_415_2_005_art_b7e4d1`
- **Purpose**: Unique identifier for reconstructed articles
- **Stored in**: Article reconstruction results

```python
from newspaper_explorer.data.utils.ids import generate_article_id

article_id = generate_article_id(
    page_id="der_tag_1902-09-05_415_2_005"
)
# Returns: "der_tag_1902-09-05_415_2_005_art_b7e4d1"
```

---

### 8. entity_id

**Type**: String (UUID-based, foreign key to line_id)

- **Format**: `{line_id}_ent_{uuid_short}`
- **Example**: `der_tag_1902-09-05_415_2_005_TB_1_TL_1_ent_c5a8b3`
- **Purpose**: Unique identifier for extracted entities
- **Stored in**: Entity extraction results

```python
from newspaper_explorer.data.utils.ids import generate_entity_id

entity_id = generate_entity_id(
    line_id="der_tag_1902-09-05_415_2_005_TB_1_TL_1"
)
# Returns: "der_tag_1902-09-05_415_2_005_TB_1_TL_1_ent_c5a8b3"
```

---

## Filename Parsing

### ALTO Filename Format

ALTO files follow this naming convention:
```
{zdb_prefix}_{YYYY-MM-DD}_{unknown}_{issue}_{separator}_{edition}_{page}.xml
```

Example: `3074409X_1902-09-05_000_415_H_2_005.xml`

| Component | Example | Description |
|-----------|---------|-------------|
| zdb_prefix | `3074409X` | ZDB ID without hyphen |
| date | `1902-09-05` | Publication date |
| unknown | `000` | Always "000", purpose unknown |
| issue | `415` | Sequential publication number |
| separator | `H` | "Heft" = issue marker |
| edition | `2` | Edition (1=morning, 2=midday, 3=evening) |
| page | `005` | Page number |

### parse_alto_filename

Parse an ALTO filename into its components:

```python
from newspaper_explorer.data.utils.ids import parse_alto_filename

parsed = parse_alto_filename("3074409X_1902-09-05_000_415_H_2_005.xml")
# Returns ParsedFilename:
#   zdb_prefix="3074409X"
#   date=datetime(1902, 9, 5)
#   unknown_field="000"
#   issue_number=415
#   separator="H"
#   edition=2
#   page_number=5
```

### extract_edition

Extract edition from filename or folder path (authoritative function):

```python
from newspaper_explorer.data.utils.ids import extract_edition

# From ALTO filename
extract_edition(filename="3074409X_1902-09-05_000_415_H_2_005.xml")  # Returns 2

# From METS filename (no page suffix)
extract_edition(filename="3074409X_1920-03-03_000_53_H_1.xml")  # Returns 1

# From folder path
extract_edition(folder_path="1920/03/03/02")  # Returns 2
```

---

## Parquet Schemas with Foreign Keys

#### **lines.parquet** (Source data)
```python
{
    # Primary key
    "line_id": str,           # PRIMARY: der_tag_1902-09-05_415_2_005_TB_1_TL_1

    # Foreign keys
    "source_id": str,         # FK: der_tag
    "issue_id": str,          # FK: der_tag_1902-09-05_415_2
    "page_id": str,           # FK: der_tag_1902-09-05_415_2_005
    "text_block_id": str,     # FK: der_tag_1902-09-05_415_2_005_TB_1

    # Data
    "text": str,
    "x": int, "y": int, "width": int, "height": int,

    # Metadata (denormalized for convenience)
    "date": datetime,
    "newspaper_title": str,
    "issue_number": int,
    "edition": int,           # Edition of the day (1=morning, 2=midday, 3=evening)
    "page_number": int,
    "year_volume": str,
    "page_count": int,

    # Original reference (for debugging)
    "filename": str,          # Original ALTO filename
}
```

#### **text_blocks.parquet** (Aggregated data)
```python
{
    # Primary key
    "text_block_id": str,     # PRIMARY: der_tag_1902-09-05_415_2_005_TB_1

    # Foreign keys
    "source_id": str,         # FK: der_tag
    "issue_id": str,          # FK: der_tag_1902-09-05_415_2
    "page_id": str,           # FK: der_tag_1902-09-05_415_2_005

    # Data
    "text": str,
    "line_count": int,
    "bbox": struct {          # Bounding box of entire block
        "x": int, "y": int,
        "width": int, "height": int
    },

    # Metadata (denormalized)
    "date": datetime,
    "newspaper_title": str,
    "year": int,
}
```

#### **preprocessed.parquet** (Processed text)
```python
{
    # Primary key (from original)
    "line_id": str,           # PRIMARY (if line-level)
    "text_block_id": str,     # PRIMARY (if block-level)

    # Foreign keys (MUST PRESERVE)
    "source_id": str,         # FK: der_tag
    "issue_id": str,          # FK: der_tag_1902-09-05_415_2
    "page_id": str,           # FK: der_tag_1902-09-05_415_2_005

    # Data
    "text_original": str,     # Original text
    "text_processed": str,    # Processed text

    # Processing metadata
    "preprocessing_steps": list[str],
    "preprocessing_date": datetime,
}
```

#### **detections.parquet** (Layout analysis)
```python
{
    # Primary key
    "detection_id": str,      # PRIMARY: der_tag_1902-09-05_415_2_005_headline_a3f9c2

    # Foreign keys
    "source_id": str,         # FK: der_tag
    "issue_id": str,          # FK: der_tag_1902-09-05_415_2
    "page_id": str,           # FK: der_tag_1902-09-05_415_2_005

    # Detection data
    "class_name": str,
    "confidence": float,
    "bbox_x1": float, "bbox_y1": float,
    "bbox_x2": float, "bbox_y2": float,

    # Linked ALTO elements
    "text_block_ids": list[str],  # FKs: List of linked text_block_ids
    "line_ids": list[str],        # FKs: List of linked line_ids
    "text_content": str,          # Aggregated text

    # Image reference
    "image_path": str,

    # Metadata
    "date": datetime,
    "year": int,
}
```

#### **headlines.parquet** (Matched headlines)
```python
{
    # Primary key
    "headline_id": str,       # PRIMARY: Same as detection_id

    # Foreign keys
    "source_id": str,         # FK: der_tag
    "issue_id": str,          # FK: der_tag_1902-09-05_415_2
    "page_id": str,           # FK: der_tag_1902-09-05_415_2_005
    "detection_id": str,      # FK: Link to detection

    # Data
    "ocr_text": str,
    "text_block_ids": list[str],  # FKs: Linked text blocks
    "confidence": float,
    "match_score": float,

    # Metadata
    "date": datetime,
    "newspaper_title": str,
}
```

#### **articles.parquet** (Reconstructed articles)
```python
{
    # Primary key
    "article_id": str,        # PRIMARY: der_tag_1902-09-05_415_2_005_art_b7e4d1

    # Foreign keys
    "source_id": str,         # FK: der_tag
    "issue_id": str,          # FK: der_tag_1902-09-05_415_2
    "page_id": str,           # FK: der_tag_1902-09-05_415_2_005 (main page)
    "headline_id": str,       # FK: Link to headline

    # Data
    "full_text": str,
    "text_block_ids": list[str],   # FKs: All text blocks in article
    "line_ids": list[str],         # FKs: All lines (optional)

    # Associated media
    "image_detection_ids": list[str],    # FKs: Linked images
    "table_detection_ids": list[str],    # FKs: Linked tables
    "formula_detection_ids": list[str],  # FKs: Linked formulas

    # Spatial extent
    "bbox_x1": float, "bbox_y1": float,
    "bbox_x2": float, "bbox_y2": float,

    # Quality
    "completeness_score": float,

    # Metadata
    "date": datetime,
    "newspaper_title": str,
}
```

#### **entities.parquet** (Entity extraction)
```python
{
    # Primary key
    "entity_id": str,         # PRIMARY: der_tag_..._TB_1_TL_1_ent_c5a8b3

    # Foreign keys (CRITICAL)
    "source_id": str,         # FK: der_tag
    "issue_id": str,          # FK: der_tag_1902-09-05_415_2
    "page_id": str,           # FK: der_tag_1902-09-05_415_2_005
    "text_block_id": str,     # FK: der_tag_1902-09-05_415_2_005_TB_1
    "line_id": str,           # FK: der_tag_1902-09-05_415_2_005_TB_1_TL_1

    # Can also link to articles later
    "article_id": str,        # FK (optional): Link if article reconstruction done

    # Entity data
    "entity_text": str,
    "entity_type": str,
    "confidence": float,

    # Context
    "context_text": str,      # Full line text for context

    # Metadata
    "date": datetime,
    "newspaper_title": str,
    "extraction_model": str,
    "extraction_date": datetime,
}
```

---

## Usage Examples

### Parsing IDs

Extract components from compound IDs using NamedTuple-based parsers:

```python
from newspaper_explorer.data.utils.ids import (
    parse_page_id,
    parse_issue_id,
    parse_line_id,
)

# Parse page_id - returns PageIdComponents NamedTuple
components = parse_page_id("der_tag_1902-09-05_415_2_005")
print(components.source)       # "der_tag"
print(components.date)         # "1902-09-05"
print(components.issue_number) # 415
print(components.edition)      # 2
print(components.page_number)  # 5

# Parse issue_id - returns IssueIdComponents NamedTuple
components = parse_issue_id("der_tag_1902-09-05_415_2")
print(components.source)       # "der_tag"
print(components.issue_number) # 415
print(components.edition)      # 2

# Parse line_id - returns LineIdComponents NamedTuple
components = parse_line_id("der_tag_1902-09-05_415_2_005_TB_1_TL_1")
print(components.source)            # "der_tag"
print(components.block_id)          # "TB_1"
print(components.line_id)           # "TL_1"
print(components.full_page_id)      # "der_tag_1902-09-05_415_2_005"
print(components.full_text_block_id) # "der_tag_1902-09-05_415_2_005_TB_1"
```

### ID Type Identification

Identify what type of ID you have:

```python
from newspaper_explorer.data.utils.ids import identify_id_type

identify_id_type("der_tag")  # "source_id"
identify_id_type("der_tag_1902-09-05_415_2")  # "issue_id"
identify_id_type("der_tag_1902-09-05_415_2_005")  # "page_id"
identify_id_type("der_tag_1902-09-05_415_2_005_TB_1")  # "text_block_id"
identify_id_type("der_tag_1902-09-05_415_2_005_TB_1_TL_1")  # "line_id"
identify_id_type("der_tag_1902-09-05_415_2_005_headline_a3f9c2")  # "detection_id"
identify_id_type("der_tag_1902-09-05_415_2_005_art_b7e4d1")  # "article_id"
identify_id_type("der_tag_1902-09-05_415_2_005_TB_1_TL_1_ent_c5a8b3")  # "entity_id"
```

### Extracting Foreign Keys

Extract parent IDs from any ID (cached for performance):

```python
from newspaper_explorer.data.utils.ids import extract_foreign_keys

# Returns ForeignKeys NamedTuple (cached with lru_cache)
fks = extract_foreign_keys("der_tag_1902-09-05_415_2_005_TB_1_TL_1")

print(fks.source_id)      # "der_tag"
print(fks.issue_id)       # "der_tag_1902-09-05_415_2"
print(fks.page_id)        # "der_tag_1902-09-05_415_2_005"
print(fks.text_block_id)  # "der_tag_1902-09-05_415_2_005_TB_1"
print(fks.line_id)        # "der_tag_1902-09-05_415_2_005_TB_1_TL_1"

# Convert to dict if needed
fks_dict = fks.as_dict()
```

### Adding FK Columns to DataFrames

Efficiently add foreign key columns to analysis results:

```python
from newspaper_explorer.data.utils.ids import add_foreign_key_columns
import polars as pl

# Preferred: Join from source DataFrame (no parsing needed)
source_df = pl.read_parquet("data/raw/der_tag/text/der_tag_lines.parquet")
results_df = add_foreign_key_columns(
    results_df,
    id_column="text_block_id",
    source_df=source_df
)

# Fallback: Parse from ID column (slower but works without source)
results_df = add_foreign_key_columns(results_df, id_column="doc_id")
```

---

### Query Examples

#### Get all text for a specific issue:
```python
import polars as pl

lines = pl.read_parquet("data/raw/der_tag/text/der_tag_lines.parquet")
issue_lines = lines.filter(pl.col("issue_id") == "der_tag_1902-09-05_415_2")
```

#### Link detections back to source text:

```python
detections = pl.read_parquet("results/der_tag/layout/detections.parquet")
lines = pl.read_parquet("data/raw/der_tag/text/der_tag_lines.parquet")

# Get headline detection
headline_det = detections.filter(
    pl.col("detection_id") == "der_tag_1902-09-05_415_2_005_headline_a3f9c2"
).first()

# Get linked text blocks
text_block_ids = headline_det["text_block_ids"]
headline_lines = lines.filter(pl.col("text_block_id").is_in(text_block_ids))
```

#### Find all entities in an article (future):

```python
# When article reconstruction and entity extraction are production-ready:
articles = pl.read_parquet("results/der_tag/articles.parquet")
entities = pl.read_parquet("results/der_tag/entities.parquet")

article = articles.filter(
    pl.col("article_id") == "der_tag_1902-09-05_415_2_005_art_b7e4d1"
).first()

# Get all entities in article's text blocks
article_entities = entities.filter(
    pl.col("text_block_id").is_in(article["text_block_ids"])
)
```

#### Cross-reference detections and entities (future):

```python
# When entity extraction is production-ready:
headline_dets = detections.filter(pl.col("class_name") == "headline")

for headline in headline_dets.iter_rows(named=True):
    headline_entities = entities.filter(
        pl.col("page_id") == headline["page_id"]
    ).filter(
        pl.col("text_block_id").is_in(headline["text_block_ids"])
    )
```

### ID Extraction Utilities

Extract parent IDs from compound IDs:

```python
from newspaper_explorer.data.utils.ids import (
    extract_issue_id_from_page_id,
    extract_page_id_from_text_block_id,
    extract_text_block_id_from_line_id,
    extract_page_id_from_detection_or_article_id,
)

# Extract issue_id from page_id
extract_issue_id_from_page_id("der_tag_1902-09-05_415_2_005")
# Returns: "der_tag_1902-09-05_415_2"

# Extract page_id from text_block_id
extract_page_id_from_text_block_id("der_tag_1902-09-05_415_2_005_TB_1")
# Returns: "der_tag_1902-09-05_415_2_005"

# Extract text_block_id from line_id
extract_text_block_id_from_line_id("der_tag_1902-09-05_415_2_005_TB_1_TL_1")
# Returns: "der_tag_1902-09-05_415_2_005_TB_1"

# Extract page_id from detection or article ID
extract_page_id_from_detection_or_article_id("der_tag_1902-09-05_415_2_005_headline_a3f9c2")
# Returns: "der_tag_1902-09-05_415_2_005"
```

---

## Implementation Guidelines

### For Analysis Module Developers

When implementing new analysis modules (entities, topics, emotions, concepts), follow these patterns:

#### 1. Always Include Foreign Keys

```python
# Example: Entity extraction output
from newspaper_explorer.data.utils.ids import generate_entity_id

entity_df = pl.DataFrame({
    "entity_id": [generate_entity_id(line_id) for line_id in line_ids],
    "entity_text": [...],
    "entity_type": [...],
    "confidence": [...],

    # Foreign keys (REQUIRED)
    "source_id": [...],           # From source data
    "issue_id": [...],            # From source data
    "page_id": [...],             # From source data
    "text_block_id": [...],       # From source data
    "line_id": [...],             # From source data

    # Metadata
    "date": [...],
    "newspaper_title": [...],
})
```

#### 2. Use add_foreign_key_columns for Efficient FK Addition

```python
from newspaper_explorer.data.utils.ids import add_foreign_key_columns

# Preferred: Join from source DataFrame (no parsing)
df = pl.read_parquet("data/raw/der_tag/text/der_tag_lines.parquet")
results = add_foreign_key_columns(
    results,
    id_column="text_block_id",
    source_df=df
)
```

#### 3. Preserve FKs Through Pipeline

```python
# When loading data for analysis, keep FK columns
df = pl.read_parquet("data/raw/der_tag/text/der_tag_lines.parquet")

# Your analysis adds new columns but preserves existing ones
results = df.select([
    # Keep all foreign keys
    "line_id", "text_block_id", "page_id", "issue_id", "source_id",
    # Keep metadata
    "date", "newspaper_title",
    # Add your analysis results
    pl.col("text").alias("analyzed_text"),
    # ... your new columns
])
```

#### 4. Use UUID for Analysis-Generated IDs

```python
from newspaper_explorer.data.utils.ids import (
    generate_detection_id,
    generate_article_id,
    generate_entity_id,
)

# For stable, unique IDs:
detection_id = generate_detection_id(page_id, "headline")
article_id = generate_article_id(page_id)
entity_id = generate_entity_id(line_id)
```

---

## Benefits

1. **Complete Traceability** - Link any analysis result back to original ALTO/METS
2. **Efficient Queries** - Foreign keys enable fast filtering and joining with Polars/DuckDB
3. **Issue-Level Analysis** - Easily analyze complete newspaper issues
4. **Cross-Module Integration** - Link entities, topics, emotions to articles
5. **Stable IDs** - UUID-based IDs for detection/analysis outputs
6. **Human-Readable** - IDs contain semantic information (source name, date, page)
7. **Resume Friendly** - Partial processing can resume using existing IDs
8. **Cached Performance** - `extract_foreign_keys()` uses LRU cache for repeated lookups

---

## Architecture Decisions

### 1. Source ID: Human-Readable Source Name

**Decision**: Use canonical source name (e.g., "der_tag") instead of ZDB ID.

**Rationale**:
- Human-readable and matches directory structure
- ZDB ID stored separately in source config for provenance
- Simpler to work with in queries and debugging

**Implementation**: Source name is passed directly to ID generation functions:

```python
source_id = generate_source_id("der_tag")  # Returns "der_tag"
```

### 2. Edition vs Daily Issue Number

**Decision**: Use `edition` (1, 2, 3) instead of `daily_issue_number`.

**Rationale**:
- Clearer semantics: 1=morning, 2=midday, 3=evening
- Multiple editions can share the same issue_number
- Aligns with historical newspaper publishing patterns

### 3. UUID-Based Detection IDs

**Decision**: Use 6-character UUID suffix for detection/analysis IDs instead of sequential indices.

**Rationale**:
- Sequential indices change when re-running detection
- UUIDs enable robust resume functionality
- Short (6 chars) for readability while maintaining uniqueness

**Tradeoff**: IDs are less human-readable, but stability is more important.

### 4. Hierarchical Structure

**Decision**: Each ID contains its parent ID as prefix.

**Rationale**:
- Self-documenting (you can see the hierarchy in the ID)
- Enables efficient prefix-based filtering
- No need for separate FK columns in some cases

**Example**: `line_id` contains `text_block_id` which contains `page_id` which contains `issue_id` which contains `source_id`.

### 5. NamedTuple Return Types

**Decision**: Parse functions return NamedTuples instead of dicts.

**Rationale**:
- Type-safe attribute access
- Hashable (can be used in sets/dicts)
- Immutable
- Clear field names in IDE autocomplete

### 6. LRU Cache for FK Extraction

**Decision**: `extract_foreign_keys()` uses `@lru_cache(maxsize=10000)`.

**Rationale**:
- FK extraction is frequently called with same IDs
- Caching avoids repeated regex parsing
- 10K cache size balances memory vs performance

---

## API Reference

### Generation Functions

| Function | Parameters | Returns | Example |
|----------|-----------|---------|---------|
| `generate_source_id` | `source_name: str` | `str` | `"der_tag"` |
| `generate_issue_id` | `source, date, issue_number, edition` | `str` | `"der_tag_1902-09-05_415_2"` |
| `generate_page_id` | `source, date, issue_number, edition, page_number` | `str` | `"der_tag_1902-09-05_415_2_005"` |
| `generate_text_block_id` | `page_id, block_id` | `str` | `"der_tag_..._005_TB_1"` |
| `generate_line_id` | `text_block_id, line_id` | `str` | `"der_tag_..._TB_1_TL_1"` |
| `generate_detection_id` | `page_id, class_name` | `str` | `"der_tag_..._headline_a3f9c2"` |
| `generate_article_id` | `page_id` | `str` | `"der_tag_..._art_b7e4d1"` |
| `generate_entity_id` | `line_id` | `str` | `"der_tag_..._TL_1_ent_c5a8b3"` |

### Parsing Functions

| Function | Parameters | Returns |
|----------|-----------|---------|
| `parse_alto_filename` | `filename: str` | `ParsedFilename` or `None` |
| `parse_page_id` | `page_id: str` | `PageIdComponents` |
| `parse_issue_id` | `issue_id: str` | `IssueIdComponents` |
| `parse_line_id` | `line_id: str` | `LineIdComponents` |

### Extraction Functions

| Function | Parameters | Returns |
|----------|-----------|---------|
| `extract_edition` | `filename=None, folder_path=None` | `int` or `None` |
| `extract_foreign_keys` | `id_string: str` | `ForeignKeys` (cached) |
| `extract_issue_id_from_page_id` | `page_id: str` | `str` |
| `extract_page_id_from_text_block_id` | `text_block_id: str` | `str` |
| `extract_text_block_id_from_line_id` | `line_id: str` | `str` |
| `extract_page_id_from_detection_or_article_id` | `id_string: str` | `str` |

### Utility Functions

| Function | Parameters | Returns |
|----------|-----------|---------|
| `identify_id_type` | `id_string: str` | `str` (type name) |
| `source_id_to_filename_prefix` | `source_id: str` | `str` |
| `add_foreign_key_columns` | `df, id_column, source_df=None` | `pl.DataFrame` |

### NamedTuple Types

```python
class ParsedFilename:
    zdb_prefix: str
    date: datetime
    unknown_field: str
    issue_number: int
    separator: str
    edition: int
    page_number: int

class PageIdComponents(NamedTuple):
    source: str
    date: str
    issue_number: int
    edition: int
    page_number: int

class IssueIdComponents(NamedTuple):
    source: str
    date: str
    issue_number: int
    edition: int

class LineIdComponents(NamedTuple):
    source: str
    date: str
    issue_number: int
    edition: int
    page_number: int
    block_id: str
    line_id: str
    full_page_id: str
    full_text_block_id: str

class ForeignKeys(NamedTuple):
    source_id: Optional[str]
    issue_id: Optional[str]
    page_id: Optional[str]
    text_block_id: Optional[str]
    line_id: Optional[str]

    def as_dict(self) -> dict[str, Optional[str]]: ...
```

---

## Testing

Tests are located in `tests/data/`:
- `test_ids.py` - Tests for ID generation and parsing

Run tests:
```bash
pytest tests/data/test_ids.py -v
```
