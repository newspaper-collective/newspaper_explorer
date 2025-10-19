# Data Architecture & Query Strategy

## Overview

This document describes the data architecture for storing source text, analysis results, and enabling efficient UI queries without loading multi-GB files into memory.

## Core Design: **Parquet + DuckDB**

### Philosophy

- **Parquet for storage** - Columnar, compressed, efficient
- **DuckDB for queries** - SQL interface, queries Parquet directly
- **No data duplication** - DuckDB reads Parquet files in place
- **Maintain provenance** - Track which method/model produced each result

### Data Flow

```
XML Archives → DataLoader → Parquet (source)
                                ↓
                          Analysis Pipeline
                                ↓
                    Parquet (results) ← method/model metadata
                                ↓
                            DuckDB queries
                                ↓
                          UI (FastAPI/Streamlit)
```

## Storage Structure

### Source Data (Current)

```
data/raw/{source}/text/{source}_lines.parquet
```

**Schema:**
```python
{
    "line_id": str,          # Primary key (e.g., "der_tag_1901_01_15_001_block_003_line_001")
    "text": str,             # Full text content
    "text_block_id": str,    # Block grouping
    "filename": str,         # Source XML file
    "date": datetime,        # Publication date
    "x": int, "y": int,      # Coordinates
    "width": int, "height": int,
    "newspaper_title": str,  # From METS
    "year_volume": str,      # From METS
    "page_count": int,       # From METS
}
```

### Analysis Results (New Structure)

```
results/{source}/{analysis_type}/{method_id}/results.parquet
```

Example structure:
```
results/der_tag/
├── entities/
│   ├── llm_gpt4o_mini_20241019_143022/
│   │   └── entities.parquet
│   └── spacy_de_core_news_lg_20241019_150000/
│       └── entities.parquet
├── topics/
│   ├── llm_gpt4o_mini_20241019_143022/
│   │   └── topics.parquet
│   └── bertopic_20241019_150000/
│       └── topics.parquet
└── emotions/
    └── llm_gpt4o_mini_20241019_143022/
        └── emotions.parquet
```

### Analysis Metadata

Each analysis run creates a `metadata.json` alongside results:

```json
{
    "analysis_id": "llm_gpt4o_mini_20241019_143022",
    "analysis_type": "entities",
    "method_type": "llm",
    "model_name": "gpt-4o-mini",
    "model_version": null,
    "parameters": {
        "temperature": 0.3,
        "max_tokens": 2000,
        "prompt_template": "entity_extraction"
    },
    "source": "der_tag",
    "created_at": "2024-10-19T14:30:22Z",
    "line_count": 15420,
    "duration_seconds": 3600
}
```

## Result Schemas

### Entity Extraction Results

**File:** `results/{source}/entities/{method_id}/entities.parquet`

```python
{
    "line_id": str,          # FK to source_lines
    "entity_text": str,      # "Kaiser Wilhelm II"
    "entity_type": str,      # "person", "location", "organization"
    "confidence": float,     # 0.0-1.0 (if available)
    "start_pos": int,        # Character position in text (optional)
    "end_pos": int,          # Character position in text (optional)
}
```

### Topic Analysis Results

**File:** `results/{source}/topics/{method_id}/topics.parquet`

```python
{
    "line_id": str,
    "primary_topic": str,
    "secondary_topics": list[str],  # JSON or separate rows
    "confidence": float,
}
```

### Emotion Analysis Results

**File:** `results/{source}/emotions/{method_id}/emotions.parquet`

```python
{
    "line_id": str,
    "sentiment": str,        # "positive", "negative", "neutral", "mixed"
    "emotions": list[str],   # JSON list
    "intensity": float,
    "tone": str,
}
```

### Concept Extraction Results

**File:** `results/{source}/concepts/{method_id}/concepts.parquet`

```python
{
    "line_id": str,
    "concept": str,
}
```

**Relationships:** `results/{source}/concepts/{method_id}/relationships.parquet`

```python
{
    "source_concept": str,
    "target_concept": str,
    "relationship_type": str,  # "leads_to", "causes", etc.
}
```

## Query Layer: DuckDB

### Why DuckDB?

- ✅ **Queries Parquet directly** - No data loading/duplication
- ✅ **SQL interface** - Familiar, powerful
- ✅ **Blazing fast** - Built for analytics
- ✅ **Polars compatible** - Can convert to/from Polars DataFrames
- ✅ **Join across files** - Query multiple Parquet files as tables
- ✅ **Python-friendly** - Excellent Python API
- ✅ **Small footprint** - No server, embedded database

### Installation

```bash
pip install duckdb
```

### Basic Usage

```python
import duckdb

# Create connection (in-memory or persistent)
con = duckdb.connect()  # In-memory
# OR
con = duckdb.connect('results/der_tag/query_cache.duckdb')  # Persistent

# Query Parquet directly
result = con.execute("""
    SELECT line_id, text, date
    FROM 'data/raw/der_tag/text/der_tag_lines.parquet'
    WHERE date >= '1901-01-01'
    LIMIT 10
""").df()  # Returns Pandas DataFrame

# Or convert to Polars
import polars as pl
result_pl = pl.from_pandas(result)
```

### Example Queries

#### Find All Mentions of an Entity

```python
# Query entities and join with source text
query = """
SELECT 
    e.entity_text,
    e.entity_type,
    s.text,
    s.date,
    s.filename
FROM 'results/der_tag/entities/llm_gpt4o_mini_20241019/entities.parquet' e
JOIN 'data/raw/der_tag/text/der_tag_lines.parquet' s
    ON e.line_id = s.line_id
WHERE e.entity_text = 'Kaiser Wilhelm II'
ORDER BY s.date
"""

mentions = con.execute(query).df()
```

#### Compare Methods

```python
# Compare entity extraction from two methods
query = """
SELECT 
    llm.entity_text AS llm_entity,
    spacy.entity_text AS spacy_entity,
    s.text,
    s.date
FROM 'results/der_tag/entities/llm_gpt4o_mini/entities.parquet' llm
FULL OUTER JOIN 'results/der_tag/entities/spacy_de/entities.parquet' spacy
    ON llm.line_id = spacy.line_id 
    AND llm.entity_text = spacy.entity_text
JOIN 'data/raw/der_tag/text/der_tag_lines.parquet' s
    ON COALESCE(llm.line_id, spacy.line_id) = s.line_id
WHERE llm.entity_text IS NULL OR spacy.entity_text IS NULL
"""

differences = con.execute(query).df()
```

#### Aggregate Statistics

```python
# Entity frequency by year
query = """
SELECT 
    YEAR(s.date) as year,
    e.entity_text,
    COUNT(*) as mention_count
FROM 'results/der_tag/entities/llm_gpt4o_mini/entities.parquet' e
JOIN 'data/raw/der_tag/text/der_tag_lines.parquet' s
    ON e.line_id = s.line_id
GROUP BY year, e.entity_text
HAVING mention_count > 5
ORDER BY year, mention_count DESC
"""

freq = con.execute(query).df()
```

## UI Backend: FastAPI + DuckDB

### Architecture

```
Frontend (React/Vue/Streamlit)
    ↓ HTTP API
FastAPI Server
    ↓ SQL Queries
DuckDB
    ↓ Direct Read
Parquet Files (source + results)
```

### API Endpoints Example

```python
from fastapi import FastAPI
import duckdb

app = FastAPI()
con = duckdb.connect('results/der_tag/query_cache.duckdb')

@app.get("/api/entities/{entity_name}")
def get_entity_mentions(entity_name: str, method: str = "llm_gpt4o_mini"):
    """Get all mentions of an entity with context."""
    query = f"""
    SELECT 
        e.entity_text,
        e.entity_type,
        s.text,
        s.date,
        s.filename,
        s.line_id
    FROM 'results/der_tag/entities/{method}/entities.parquet' e
    JOIN 'data/raw/der_tag/text/der_tag_lines.parquet' s
        ON e.line_id = s.line_id
    WHERE e.entity_text = ?
    ORDER BY s.date
    """
    result = con.execute(query, [entity_name]).df()
    return result.to_dict(orient='records')

@app.get("/api/line/{line_id}")
def get_line_fulltext(line_id: str):
    """Get full text for a specific line_id."""
    query = """
    SELECT *
    FROM 'data/raw/der_tag/text/der_tag_lines.parquet'
    WHERE line_id = ?
    """
    result = con.execute(query, [line_id]).df()
    if len(result) == 0:
        return {"error": "Line not found"}
    return result.iloc[0].to_dict()

@app.get("/api/search")
def search_text(q: str, start_date: str = None, end_date: str = None):
    """Full-text search with date filtering."""
    query = """
    SELECT line_id, text, date, filename
    FROM 'data/raw/der_tag/text/der_tag_lines.parquet'
    WHERE text LIKE ?
    """
    params = [f"%{q}%"]
    
    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)
    
    query += " LIMIT 100"
    
    result = con.execute(query, params).df()
    return result.to_dict(orient='records')
```

## Alternative: Streamlit UI

For simpler use cases, Streamlit with DuckDB is excellent:

```python
import streamlit as st
import duckdb
import plotly.express as px

con = duckdb.connect()

st.title("Newspaper Explorer")

# Entity search
entity = st.text_input("Search entity:")
if entity:
    query = f"""
    SELECT 
        e.entity_text,
        s.text,
        s.date
    FROM 'results/der_tag/entities/llm_gpt4o_mini/entities.parquet' e
    JOIN 'data/raw/der_tag/text/der_tag_lines.parquet' s
        ON e.line_id = s.line_id
    WHERE e.entity_text LIKE '%{entity}%'
    LIMIT 50
    """
    result = con.execute(query).df()
    st.dataframe(result)
    
    # Visualization
    if len(result) > 0:
        fig = px.histogram(result, x='date', title=f"Mentions of {entity} over time")
        st.plotly_chart(fig)
```

## Migration Path

### Phase 1: Current State
- ✅ Parquet source files exist
- Results saved as Parquet (implement new schema)

### Phase 2: Query Layer
- Add DuckDB for ad-hoc queries
- Create query utilities in `utils/queries.py`

### Phase 3: UI Backend
- Implement FastAPI endpoints
- OR build Streamlit app
- Use DuckDB to query Parquet files

### Phase 4: Optimization
- Create DuckDB views for common queries
- Add indexes if needed
- Cache frequently accessed data

## Best Practices

### 1. Always Use `line_id` as Foreign Key

```python
# ✅ Good - maintains link to source
df = pl.DataFrame({
    "line_id": source_df["line_id"],
    "entity_text": entities,
    "entity_type": entity_types,
})
```

### 2. Include Method Metadata

Store analysis metadata alongside results for reproducibility.

### 3. Partition Large Results

For very large result sets, partition by date:

```
results/der_tag/entities/llm_gpt4o_mini/
├── year=1901/
│   └── entities.parquet
├── year=1902/
│   └── entities.parquet
...
```

DuckDB can query partitioned data efficiently:

```sql
SELECT * FROM 'results/der_tag/entities/llm_gpt4o_mini/year=*/entities.parquet'
WHERE year >= 1901 AND year <= 1905
```

### 4. Use Views for Common Queries

Create DuckDB views to simplify repeated queries:

```python
con.execute("""
CREATE VIEW entity_mentions AS
SELECT 
    e.line_id,
    e.entity_text,
    e.entity_type,
    s.text,
    s.date,
    s.filename
FROM 'results/der_tag/entities/llm_gpt4o_mini/entities.parquet' e
JOIN 'data/raw/der_tag/text/der_tag_lines.parquet' s
    ON e.line_id = s.line_id
""")

# Now query the view
result = con.execute("SELECT * FROM entity_mentions WHERE entity_text = 'Berlin'").df()
```

## Performance Considerations

### File Size
- **Parquet** is highly compressed (typically 5-10x smaller than CSV)
- **DuckDB** reads only required columns (columnar format)
- Multi-GB files are fine - DuckDB doesn't load entire file

### Memory Usage
- DuckDB streams data, doesn't load everything
- Only result set needs to fit in memory
- Use `LIMIT` for large result sets in UI

### Query Speed
- DuckDB is **very fast** on Parquet (C++ implementation)
- Expect millisecond queries on GB-scale data
- Add indexes for repeated filter columns if needed

## Summary

**Storage:** Parquet files (source + results)  
**Query:** DuckDB (SQL interface, reads Parquet directly)  
**UI:** FastAPI + DuckDB OR Streamlit + DuckDB  
**Provenance:** Method ID in directory structure + metadata.json  
**Relations:** Foreign keys via `line_id`  

This architecture is:
- ✅ Scalable (handles multi-GB files)
- ✅ Efficient (no data duplication)
- ✅ Queryable (SQL interface)
- ✅ Flexible (compare methods, track provenance)
- ✅ UI-friendly (fast queries, paginated results)
