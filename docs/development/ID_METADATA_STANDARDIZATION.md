# ID & Metadata Standardization Plan

**Date:** 2025-12-07
**Status:** Mostly complete
**Decision:** Use `source_name` (e.g., `der_tag`) as canonical `source_id`

## Overview

Unify ID formats, column naming, and metadata handling across:
- Source data (lines parquet)
- Preprocessed datasets
- Analysis results (entities, keywords, emotions, layout, topics)
- Image index

**Goal:** Eliminate redundant parsing, fix naming inconsistencies, ensure reliable cross-dataset joins.

---

## Completed Tasks

### ✅ Core Infrastructure

1. **Schema definitions** (`models/data/schema.py`)
   - `SourceLinesSchema` - Line-level source data
   - `TextBlocksSchema` - Aggregated text blocks
   - `ImageIndexSchema` - Image index records
   - `PreprocessedSchema` - Preprocessed text data

2. **Unified parsing** (`data/utils/ids.py`)
   - `ParsedFilename` dataclass for ALTO filename components
   - `parse_alto_filename()` - Single source of truth for filename parsing
   - `add_foreign_key_columns()` - Efficient FK column addition (joins from source or parses from IDs)

3. **Entity model standardization** (`models/analysis/entities.py`)
   - Changed primary FK from `line_id` to `text_block_id` (entities extracted from blocks, not lines)
   - Renamed `line_ids` → `text_block_ids` in aggregated records

4. **Image index** (`data/indexing/workers.py`)
   - Added `source_id` column to all image index records

### ✅ Legacy Code Cleanup

Removed `line_id` fallback patterns from:
- `data/utils/ids.py` - `add_foreign_key_columns()`
- `analyze/keywords/yake.py`
- `analyze/keywords/rake.py`
- `analyze/keywords/keybert.py`
- `analyze/keywords/llm.py`
- `analyze/keywords/tf_idf.py`
- `analyze/layout/text_linker.py` - Removed filename fallback for page filtering

---

## Key Decision: Source Identifier

**Chosen format:** `der_tag` (source_name)

| Alternative | Example | Rejected Because |
|-------------|---------|------------------|
| ZDB ID with hyphen | `3074409-X` | Not in filenames, requires config lookup |
| ZDB ID no hyphen | `3074409X` | Ugly, not human-readable |
| **Source name** | `der_tag` | ✅ Human-readable, matches directories/CLI |

The ZDB ID (`3074409-X`) remains in config as `metadata.zdb_source_id` for provenance.

**New ID format:**
```
source_id:     "der_tag"
issue_id:      "der_tag_1920-03-03_053_1"
page_id:       "der_tag_1920-03-03_053_1_001"
text_block_id: "der_tag_1920-03-03_053_1_001_r_1_1"
line_id:       "der_tag_1920-03-03_053_1_001_r_1_1_TL_1"
```

---

## Current Parsing Functions

All parsing functions use the centralized `_DATE_PATTERN` regex and `_find_date_index()` helper.
`extract_foreign_keys()` is cached with `@lru_cache` for 24x speedup on repeated calls.

### Filename Parsing

| Location | Function | Input | Output |
|----------|----------|-------|--------|
| `alto.py:63` | `_parse_filename()` | `3074409X_1902-09-05_000_415_H_2_005.xml` | Uses `parse_alto_filename()` from ids.py |

### ID Parsing

| Location | Function | Input | Output |
|----------|----------|-------|--------|
| `ids.py:391` | `parse_page_id()` | `der_tag_1902-09-05_415_2_005` | `PageIdComponents` |
| `ids.py:438` | `parse_issue_id()` | `der_tag_1902-09-05_415_2` | `IssueIdComponents` |
| `ids.py:483` | `parse_line_id()` | `der_tag_1902-09-05_415_2_005_TB_1_TL_1` | `LineIdComponents` |

### ID Extraction (parent from child)

| Location | Function | Input | Output |
|----------|----------|-------|--------|
| `ids.py:566` | `extract_issue_id_from_page_id()` | page_id | issue_id |
| `ids.py:585` | `extract_page_id_from_text_block_id()` | text_block_id | page_id |
| `ids.py:608` | `extract_text_block_id_from_line_id()` | line_id | text_block_id |
| `ids.py:831` | `extract_foreign_keys()` | any ID | Dict of all parent IDs (**cached**) |
| `ids.py:921` | `extract_page_id_from_detection_or_article_id()` | detection_id | page_id |

### ID Type Detection

| Location | Function | Input | Output |
|----------|----------|-------|--------|
| `ids.py:643` | `identify_id_type()` | any ID | "line_id", "page_id", etc. |

---

## Problems with Current Approach

### 1. ~~Two Separate Parsing Domains~~ ✅ FIXED

~~**Filename parsing** (`alto.py`) and **ID parsing** (`ids.py`) used different implementations.~~

Now unified:
- `alto.py` imports and uses `parse_alto_filename()` from `ids.py`
- Single source of truth for filename parsing
- ID parsing functions in `ids.py` handle generated IDs (different format is intentional)

### 2. ~~Duplicate Parsing Logic~~ ✅ FIXED

~~`extract_foreign_keys()` re-parses IDs that already have FK columns in DataFrame~~

Solved by `add_foreign_key_columns()` which:
1. **Joins from source_df** if FK columns exist (preferred, no parsing)
2. **Falls back to parsing** only when necessary

### 3. ~~Inconsistent Return Types~~ ✅ FIXED

All parsing functions now return NamedTuples:
- `parse_page_id()` → `PageIdComponents`
- `parse_issue_id()` → `IssueIdComponents`
- `parse_line_id()` → `LineIdComponents`

### 4. ~~Fragile Date Detection~~ ✅ FIXED

All date detection now uses a single compiled regex pattern:
```python
# Centralized in ids.py
_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")

def _find_date_index(parts: list[str]) -> Optional[int]:
    for i, part in enumerate(parts):
        if _DATE_PATTERN.match(part):
            return i
    return None
```

All parsing functions (`parse_page_id`, `parse_issue_id`, `parse_line_id`, `identify_id_type`, `extract_page_id_from_detection_or_article_id`) now use `_find_date_index()`.

---

## Implementation Steps

### Step 1: Create Canonical Schema Definitions

**File:** `src/newspaper_explorer/models/data/schema.py`

Define Pydantic models specifying required/optional columns for each data store.

### Step 2: Unify Parsing Functions

**New structure in `ids.py`:**

```python
# ============================================================================
# Core Parsing (single source of truth)
# ============================================================================

@dataclass
class ParsedFilename:
    """Components from ALTO filename."""
    zdb_prefix: str      # "3074409X" (not used for ID generation)
    date: datetime
    unknown_field: str   # "000" (always)
    issue_number: int
    separator: str       # "H"
    daily_issue_number: int
    page_number: int

@dataclass
class ParsedId:
    """Components from any generated ID."""
    source: str          # "der_tag"
    date: str            # "1902-09-05"
    issue_number: int
    daily_issue_number: int
    page_number: Optional[int] = None
    block_id: Optional[str] = None
    line_id: Optional[str] = None

def parse_alto_filename(filename: str) -> ParsedFilename:
    """Parse ALTO filename. Single implementation."""

def parse_id(id_string: str) -> ParsedId:
    """Parse any generated ID. Single implementation."""
```

### Step 3: Remove Duplicate Parsing

- Keep extraction functions but implement via `parse_id()`

### Step 4: Update ALTO Parser

Change `alto.py:_parse_filename()` to use the unified function:
```python
from newspaper_explorer.data.utils.ids import parse_alto_filename

def parse_file(self, filepath, source_name, ...):
    parsed = parse_alto_filename(filepath.name)
    # Use source_name (not parsed.zdb_prefix) for ID generation
    issue_id = generate_issue_id(
        source=source_name,  # "der_tag"
        date=parsed.date,
        issue_number=parsed.issue_number,
        daily_issue_number=parsed.daily_issue_number,
    )
```

### Step 5: Regenerate All Data

With `source_name` as canonical `source_id`:
1. Re-run `newspaper-explorer data parse --source der_tag`
2. Re-run `newspaper-explorer data aggregate --source der_tag`
3. Rebuild image index
4. Re-run all analysis

---

## Schema Definitions

### ID Hierarchy

```
source_id (der_tag)
└── issue_id (der_tag_1902-09-05_415_2)
    └── page_id (der_tag_1902-09-05_415_2_005)
        ├── text_block_id (der_tag_1902-09-05_415_2_005_r_1_1)
        │   └── line_id (der_tag_1902-09-05_415_2_005_r_1_1_TL_1)
        └── detection_id (der_tag_1902-09-05_415_2_005_text_ae5fd8)
            └── article_id (der_tag_1902-09-05_415_2_005_article_b3c9d1)
```

### Common FK Columns (required in ALL data stores)

| Column | Format | Example |
|--------|--------|---------|
| `source_id` | source_name | `der_tag` |
| `issue_id` | `{source}_{date}_{issue:03d}_{daily}` | `der_tag_1902-09-05_415_2` |
| `page_id` | `{issue_id}_{page:03d}` | `der_tag_1902-09-05_415_2_005` |

### Metadata Columns

| Column | Type | Source | Required In |
|--------|------|--------|-------------|
| `date` | datetime | ALTO filename | lines, blocks, results |
| `newspaper_title` | str | METS | lines, blocks (optional) |
| `year_volume` | str | METS | lines, blocks (optional) |
| `page_number` | int | ALTO filename | lines, blocks, images |
| `issue_number` | int | ALTO filename | lines, blocks |
| `daily_issue_number` | int | ALTO filename | lines, blocks |
| `page_count` | int | METS | lines, blocks (optional) |

---

## Further Considerations (Deferred)

1. **Missing 1900 METS handling** - Create stub METS or handle gracefully
2. **Mixed issue directories** (March 19, 1901) - Fix via DataFixer
3. **Test fixtures** - Update to match real filename format

---

## Files Modified

### High Priority - ✅ DONE

- [x] `src/newspaper_explorer/data/utils/ids.py` - Added `ParsedFilename`, `parse_alto_filename()`, `add_foreign_key_columns()`
- [x] `src/newspaper_explorer/data/parser/alto.py` - Uses unified parsing from ids.py
- [x] `src/newspaper_explorer/models/data/schema.py` - Created with canonical schemas
- [x] `src/newspaper_explorer/data/indexing/workers.py` - Added source_id to image index
- [x] `src/newspaper_explorer/models/analysis/entities.py` - line_id → text_block_id
- [x] `src/newspaper_explorer/analyze/entities/gliner.py` - Uses text_block_id
- [x] `src/newspaper_explorer/ui/v2/backend/routers/entities.py` - Uses text_block_id

### Legacy Cleanup - ✅ DONE

- [x] `src/newspaper_explorer/analyze/keywords/yake.py` - Removed line_id fallback
- [x] `src/newspaper_explorer/analyze/keywords/rake.py` - Removed line_id fallback
- [x] `src/newspaper_explorer/analyze/keywords/keybert.py` - Removed line_id fallback
- [x] `src/newspaper_explorer/analyze/keywords/llm.py` - Removed line_id fallback
- [x] `src/newspaper_explorer/analyze/keywords/tf_idf.py` - Removed line_id fallback in text_block extraction
- [x] `src/newspaper_explorer/analyze/layout/text_linker.py` - Removed filename fallback

### Still Supporting Line-Level (intentional, NOT legacy)

The following files correctly support BOTH line-level and text-block-level extraction:
- `analyze/keywords/tf_idf.py` - `elif "line_id" in results_df.columns` branch for line-level extraction
- `cli/analyze/entities/commands.py` - Auto-detects `text_block_id` or `line_id` based on input type
- `analyze/emotions/bert_classifier.py` - Extracts FKs from line_id for line-level emotion analysis

### Low Priority - Deferred

- [ ] `tests/` - Update fixtures and tests (some are outdated)
- [ ] `data/sources/der_tag.json` - Document ZDB ID purpose
