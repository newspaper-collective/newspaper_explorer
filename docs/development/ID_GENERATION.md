# ID Generation System

> **Status**: Production-ready (November 2025)  
> **Implementation**: `src/newspaper_explorer/data/utils/ids.py`  
> **Tests**: 42 tests covering all aspects

This document describes the unified ID system used throughout the newspaper explorer codebase. All data from parsing (ALTO/METS) through analysis maintains a consistent foreign key hierarchy enabling complete traceability and efficient querying.

---

## Design Principles

1. **Hierarchical** - IDs form a clear parent-child relationship
2. **Stable** - IDs don't change across re-runs (UUID-based where needed)
3. **Traceable** - Any analysis result can be linked back to source data
4. **Efficient** - Foreign keys enable fast filtering and joining
5. **Human-readable** - IDs contain semantic information when possible

---

## ID Hierarchy

```
source_id (e.g., "3074409-X")
  └── issue_id (e.g., "3074409-X_1901-01-08_006_1")
      └── page_id (e.g., "3074409-X_1901-01-08_006_1_001")
          ├── text_block_id (e.g., "3074409-X_1901-01-08_006_1_001_r_1_1")
          │   └── line_id (e.g., "3074409-X_1901-01-08_006_1_001_r_1_1_tl_1")
          ├── detection_id (e.g., "3074409-X_1901-01-08_006_1_001_headline_a3f9c2")
          └── article_id (e.g., "3074409-X_1901-01-08_006_1_001_art_b7e4d1")
```

---

## ID Specifications

### 1. source_id

**Type**: String (primary key for sources)

- **Format**: `{zdb_source_id}` or `{source_name}` (fallback)
- **Examples**: 
  - `3074409-X` (ZDB ID - preferred)
  - `der_tag` (fallback when ZDB ID unavailable)
- **Purpose**: Identify the newspaper source
- **Generation**: Automatic from source configuration (`data/sources/{source}.json`)
- **Stored in**: All parquet files

```python
from newspaper_explorer.data.utils.ids import generate_source_id
from newspaper_explorer.utils.sources import load_source_config

config = load_source_config("der_tag")
source_id = generate_source_id(config)  # Returns "3074409-X" or "der_tag"
```

**Note**: System prefers ZDB (Zeitschriftendatenbank) ID from source config for standardization.

**Filename Compatibility**: When reconstructing ALTO/METS filenames, use `source_id_to_filename_prefix()` to remove hyphens:

```python
from newspaper_explorer.data.utils.ids import source_id_to_filename_prefix

# Convert source_id to filename prefix
source_id = "3074409-X"
filename_prefix = source_id_to_filename_prefix(source_id)  # Returns "3074409X"

# Example: Build ALTO filename
# 3074409X_1902-09-05_000_415_H_2_005.xml
alto_filename = f"{filename_prefix}_{date}_000_{issue}_H_{daily}_{page:03d}.xml"
```

---

### 2. issue_id

**Type**: String (foreign key to source_id)

- **Format**: `{source_id}_{date}_{issue:03d}_{daily}`
- **Example**: `3074409-X_1902-09-05_415_2`
- **Purpose**: Group all pages from the same newspaper issue/edition
- **Stored in**: All parquet files

```python
from newspaper_explorer.data.utils.ids import generate_issue_id
from datetime import datetime

issue_id = generate_issue_id(
    source_id="3074409-X",
    date=datetime(1902, 9, 5),
    issue_number=415,
    daily_issue_number=2
)
# Returns: "3074409-X_1902-09-05_415_2"
```

---

### 3. page_id

**Type**: String (foreign key to issue_id)

- **Format**: `{issue_id}_{page:03d}`
- **Example**: `3074409-X_1902-09-05_415_2_005`
- **Purpose**: Unique identifier for a newspaper page
- **Stored in**: All parquet files

```python
from newspaper_explorer.data.utils.ids import generate_page_id

page_id = generate_page_id(
    issue_id="3074409-X_1902-09-05_415_2",
    page_number=5
)
# Returns: "3074409-X_1902-09-05_415_2_005"
```

---

### 4. text_block_id

**Type**: String (foreign key to page_id)

- **Format**: `{page_id}_{block_id}`
- **Example**: `3074409-X_1901-01-08_006_1_001_r_1_1`
- **Purpose**: Unique text block identifier across entire dataset
- **Stored in**: `lines.parquet`, `text_blocks.parquet`, detection results

```python
from newspaper_explorer.data.utils.ids import generate_text_block_id

text_block_id = generate_text_block_id(
    page_id="3074409-X_1901-01-08_006_1_001",
    block_id="r_1_1"
)
# Returns: "3074409-X_1901-01-08_006_1_001_r_1_1"
```

**Note**: `block_id` comes from ALTO XML and varies by document structure (e.g., `r_1_1`, `TB_1`).

---

### 5. line_id

**Type**: String (foreign key to text_block_id)

- **Format**: `{text_block_id}_{line_id}`
- **Example**: `3074409-X_1901-01-08_006_1_001_r_1_1_tl_1`
- **Purpose**: Unique text line identifier across entire dataset
- **Stored in**: `lines.parquet`, entity extraction results

```python
from newspaper_explorer.data.utils.ids import generate_line_id

line_id = generate_line_id(
    text_block_id="3074409-X_1901-01-08_006_1_001_r_1_1",
    line_id_from_alto="tl_1"
)
# Returns: "3074409-X_1901-01-08_006_1_001_r_1_1_tl_1"
```

---

### 6. detection_id

**Type**: String (UUID-based, foreign key to page_id)

- **Format**: `{page_id}_{class}_{uuid_short}`
- **Example**: `3074409-X_1901-01-08_006_1_001_headline_a3f9c2`
- **Purpose**: Stable unique ID for detected layout elements
- **Stored in**: Detection parquet files, headline results

```python
from newspaper_explorer.data.utils.ids import generate_detection_id

detection_id = generate_detection_id(
    page_id="3074409-X_1901-01-08_006_1_001",
    class_name="headline"
)
# Returns: "3074409-X_1901-01-08_006_1_001_headline_a3f9c2"
```

**Important**: Uses UUID for stability across re-runs. The same visual element may get different IDs in different runs, but this is acceptable as detections are regenerated.

---

### 7. article_id

**Type**: String (UUID-based, foreign key to page_id)

- **Format**: `{page_id}_art_{uuid_short}`
- **Example**: `3074409-X_1901-01-08_006_1_001_art_b7e4d1`
- **Purpose**: Unique identifier for reconstructed articles
- **Stored in**: Article reconstruction results

```python
from newspaper_explorer.data.utils.ids import generate_article_id

article_id = generate_article_id(
    page_id="3074409-X_1901-01-08_006_1_001"
)
# Returns: "3074409-X_1901-01-08_006_1_001_art_b7e4d1"
```

**Note**: Currently proto type implementation. Will be integrated when article reconstruction is production-ready.

---

## Parquet Schemas with Foreign Keys

#### **lines.parquet** (Source data)
```python
{
    # Primary key
    "line_id": str,           # PRIMARY: 1902-09-05_415_2_005_TB_1_TL_1
    
    # Foreign keys (NEW)
    "source_id": str,         # FK: der_tag
    "issue_id": str,          # FK: der_tag_1902-09-05_415_2
    "page_id": str,           # FK: 1902-09-05_415_2_005
    "text_block_id": str,     # FK: 1902-09-05_415_2_005_TB_1
    
    # Data
    "text": str,
    "x": int, "y": int, "width": int, "height": int,
    
    # Metadata (denormalized for convenience)
    "date": datetime,
    "newspaper_title": str,
    "issue_number": int,
    "daily_issue_number": int,
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
    "text_block_id": str,     # PRIMARY: 1902-09-05_415_2_005_TB_1
    
    # Foreign keys (NEW)
    "source_id": str,         # FK: der_tag
    "issue_id": str,          # FK: der_tag_1902-09-05_415_2
    "page_id": str,           # FK: 1902-09-05_415_2_005
    
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
    "page_id": str,           # FK: 1902-09-05_415_2_005
    
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
    "detection_id": str,      # PRIMARY: 1902-09-05_415_2_005_headline_a3f9c2
    
    # Foreign keys (NEW)
    "source_id": str,         # FK: der_tag
    "issue_id": str,          # FK: der_tag_1902-09-05_415_2
    "page_id": str,           # FK: 1902-09-05_415_2_005
    
    # Detection data
    "class_name": str,
    "confidence": float,
    "bbox_x1": float, "bbox_y1": float,
    "bbox_x2": float, "bbox_y2": float,
    
    # Linked ALTO elements (NEW - improved)
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
    "page_id": str,           # FK: 1902-09-05_415_2_005
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
    "article_id": str,        # PRIMARY: 1902-09-05_415_2_005_art_b7e4d1
    
    # Foreign keys
    "source_id": str,         # FK: der_tag
    "issue_id": str,          # FK: der_tag_1902-09-05_415_2
    "page_id": str,           # FK: 1902-09-05_415_2_005 (main page)
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
    "entity_id": str,         # NEW: {line_id}_ent_{uuid_short}
    
    # Foreign keys (CRITICAL)
    "source_id": str,         # FK: der_tag
    "issue_id": str,          # FK: der_tag_1902-09-05_415_2
    "page_id": str,           # FK: 1902-09-05_415_2_005
    "text_block_id": str,     # FK: 1902-09-05_415_2_005_TB_1
    "line_id": str,           # FK: 1902-09-05_415_2_005_TB_1_TL_1
    
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

Extract components from compound IDs:

```python
from newspaper_explorer.data.utils.ids import parse_page_id, parse_line_id

# Parse page_id
components = parse_page_id("3074409-X_1901-01-08_006_1_001")
print(components)
# {
#     'source_id': '3074409-X',
#     'date': '1901-01-08',
#     'issue_number': '006',
#     'daily_issue_number': '1',
#     'page_number': '001'
# }

# Parse line_id
components = parse_line_id("3074409-X_1901-01-08_006_1_001_r_1_1_tl_1")
print(components)
# {
#     'page_id': '3074409-X_1901-01-08_006_1_001',
#     'text_block_id': '3074409-X_1901-01-08_006_1_001_r_1_1',
#     'line_id_component': 'tl_1'
# }
```

---

### Query Examples

#### Get all text for a specific issue:
```python
import polars as pl

lines = pl.read_parquet("data/raw/der_tag/text/der_tag_lines.parquet")
issue_lines = lines.filter(pl.col("issue_id") == "3074409-X_1902-09-05_415_2")
```

#### Link detections back to source text:

```python
detections = pl.read_parquet("results/der_tag/layout/detections.parquet")
lines = pl.read_parquet("data/raw/der_tag/text/der_tag_lines.parquet")

# Get headline detection
headline_det = detections.filter(
    pl.col("detection_id") == "3074409-X_1902-09-05_415_2_005_headline_a3f9c2"
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
    pl.col("article_id") == "3074409-X_1902-09-05_415_2_005_art_b7e4d1"
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

---

## Implementation Guidelines

### For Analysis Module Developers

When implementing new analysis modules (entities, topics, emotions, concepts), follow these patterns:

#### 1. Always Include Foreign Keys

```python
# Example: Entity extraction output
entity_df = pl.DataFrame({
    "entity_id": [...],           # Generate with UUID
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

#### 2. Preserve FKs Through Pipeline

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

#### 3. Use UUID for Analysis-Generated IDs

```python
import uuid
from newspaper_explorer.data.utils.ids import generate_detection_id

# For stable, unique IDs that survive re-runs:
entity_id = f"{line_id}_ent_{str(uuid.uuid4())[:6]}"
topic_id = f"{text_block_id}_topic_{str(uuid.uuid4())[:6]}"
```

---

## Benefits

1. **Complete Traceability** - Link any analysis result back to original ALTO/METS
2. **Efficient Queries** - Foreign keys enable fast filtering and joining with Polars/DuckDB
3. **Issue-Level Analysis** - Easily analyze complete newspaper issues
4. **Cross-Module Integration** - Link entities, topics, emotions to articles
5. **Stable IDs** - UUID-based IDs survive re-runs (for detection/analysis outputs)
6. **Human-Readable** - IDs contain semantic information when possible
7. **Resume Friendly** - Partial processing can resume using existing IDs

---

## Architecture Decisions

### 1. Source ID: ZDB When Available

**Decision**: Prefer ZDB (Zeitschriftendatenbank) ID over source name.

**Rationale**: Standardized IDs across German newspaper archives enable cross-dataset integration.

**Implementation**: Source configuration (`data/sources/{source}.json`) includes optional `zdb_source_id` field:

```json
{
  "dataset_name": "der_tag",
  "metadata": {
    "zdb_source_id": "3074409-X",
    "newspaper_title": "Der Tag"
  }
}
```

### 2. UUID-Based Detection IDs

**Decision**: Use UUID suffix for detection/analysis IDs instead of sequential indices.

**Rationale**: 
- Sequential indices change when re-running detection
- UUIDs remain stable across partial re-runs
- Enables robust resume functionality

**Tradeoff**: IDs are less human-readable, but stability is more important.

### 3. Hierarchical Structure

**Decision**: Each ID contains its parent ID as prefix.

**Rationale**:
- Self-documenting (you can see the hierarchy in the ID)
- Enables efficient prefix-based filtering
- No need for separate FK columns in some cases

**Example**: `line_id` contains `text_block_id` which contains `page_id` which contains `issue_id` which contains `source_id`.

---

## Testing

Tests are located in `tests/data/`:
- `test_ids.py` - 19 tests for ID generation and parsing
- `test_detection.py` - 23 tests for detection IDs and FK extraction
- `test_preprocessing_fk.py` - 3 tests for FK preservation through preprocessing

Run tests:
```bash
pytest tests/data/test_ids.py -v
pytest tests/data/test_detection.py -v
pytest tests/data/test_preprocessing_fk.py -v
```


