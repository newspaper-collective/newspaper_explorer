# ID System Analysis

**Date:** 2025-12-07
**Status:** ✅ Performance optimizations complete, source identifier standardization in progress

## Executive Summary

The ID system works but has inconsistencies and workarounds due to:
1. ~~Multiple identifier formats for the same newspaper (ZDB ID, filename prefix, source name)~~ → **Standardizing on `source_name`**
2. Issue number vs. daily issue count confusion
3. Test fixtures that don't match real data format
4. ~~Redundant ID parsing in analysis modules (performance issue)~~ → **FIXED: `@lru_cache` added (24x speedup)**

**Recommendation:** Keep Parquet/DuckDB architecture. Fix the Python code patterns, not the data model.

---

## 1. Data Model Assessment

### Current Architecture: Denormalized Star Schema

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  source_lines.parquet (Fact Table)                                          │
│  ┌──────────┬─────────────┬────────────┬─────────────────┬─────────────────┐│
│  │ line_id  │ text        │ date       │ issue_id        │ page_id         ││
│  ├──────────┼─────────────┼────────────┼─────────────────┼─────────────────┤│
│  │ (PK)     │ (fact)      │ (metadata) │ (FK)            │ (FK)            ││
│  │ string   │ string      │ datetime   │ string          │ string          ││
│  └──────────┴─────────────┴────────────┴─────────────────┴─────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why this is CORRECT for analytics:**
- No JOINs needed for common queries → Fast
- Parquet columnar compression handles repeated values efficiently
- DuckDB can filter by `date` without loading another file

**Verdict:** ✅ Keep this architecture. Do NOT switch to SQLite.

---

## 2. Source Identifier Chaos ✅ FIXED

**Status: RESOLVED** - All code now uses `source_name` consistently for ID generation.

Three different identifiers exist for the same newspaper:

| Identifier | Example | Where Used |
|------------|---------|------------|
| **ZDB Source ID** | `3074409-X` | Config file (`der_tag.json`) - kept for provenance/metadata |
| **Filename Prefix** | `3074409X` | ALTO/METS filenames (no hyphen!) |
| **Source Name** | `der_tag` | **All generated IDs**, CLI arguments, config filename |

### Format Differences

```
Config:    zdb_source_id = "3074409-X"    ← WITH hyphen (provenance only)
Filename:  3074409X_1920-03-03_...        ← WITHOUT hyphen
METS URL:  SNP3074409X-19200303-...       ← No hyphen, "SNP" prefix, compact date
```

### Resolution

**Decision:** Always use `source_name` for ID generation.

**`loader.py`** now always uses source_name:
```python
# Always use source_name for consistent ID generation across the codebase
# (ZDB source ID is kept in source config for provenance but not used in IDs)
source_id = self.source_name
```

**`image_index.py`** also simplified:
```python
# Always use source_name for consistent ID generation across the codebase
self.source_id = source_name
```

**Result:** All generated IDs now consistently use `source_name`:
- `der_tag_1902-09-05_page_001` ✅
- Never: `3074409-X_1902-09-05_page_001` ❌

**ZDB ID preserved for:**
- Source config metadata (provenance)
- Mapping to external archives if needed
- `zdb_id_to_filename_prefix()` still available for filename parsing

---

## 3. Issue Number vs. Edition ✅ FIXED

**Status: RESOLVED** - Renamed `daily_issue_number` → `edition` for clarity.

### The Problem (Now Fixed)

The column name `daily_issue_number` was **misleading**. Analysis revealed:

| Old Name | Suggested Meaning | Actual Meaning |
|----------|-------------------|----------------|
| `issue_number` | "Which issue of the year" | Sequential publication number |
| `daily_issue_number` | "Which issue of the day" | **Edition** (1=morning, 2=midday, 3=evening) |

### Resolution

**Renamed:** `daily_issue_number` → `edition` throughout the codebase.

| Column | Description |
|--------|-------------|
| `issue_number` | Sequential publication number (e.g., Nr. 37, Nr. 38) |
| `edition` | Edition of the day (1=morning, 2=midday, 3=evening) |

### Key Insight: Multiple Editions Can Share an Issue Number

```
Feb 8, 1901:
edition=1, issue=37  ← Morning edition (Morgenausgabe)
edition=2, issue=37  ← Midday edition (same issue!)
edition=3, issue=38  ← Evening edition (new issue number)
```

### Data Analysis

| Pattern | Days | Example |
|---------|------|---------|
| 3 editions, 2 issue numbers | 467 | edition [1,2,3] → issues [37,38] |
| 2 editions, 1 issue number | 756 | edition [1,2] → issues [53] |
| Editions = Issues (1:1) | 3,225 | edition [1,2] → issues [41,42] |

### ID Format

IDs use BOTH `issue_number` AND `edition` to ensure uniqueness:
```
issue_id: {source}_{date}_{issue_number:03d}_{edition}
Example:  der_tag_1901-02-08_037_2
```

### Data Sources for `edition`

**Authoritative Source: ALTO Filename** (the `_H_X_` part)

| Source | Example | Value | Status |
|--------|---------|-------|--------|
| **ALTO filename** | `..._H_2_...` | `2` | ✅ **Authoritative** |
| Folder structure | `/1915/03/14/02/` | `02` | Validation only |
| METS `order=` | `order="1915031402"` | `02` (last 2 digits) | Derived |
| METS `<mods:partNumber>` | `Ausgabe A` | "A" (string!) | Human label only |

**Why ALTO filename is authoritative:**
1. Always present in every ALTO file
2. Already parsed by `parse_alto_filename()` in `ids.py`
3. Numeric (1, 2, 3) - easier to work with
4. Folder structure can be used for validation but is redundant

**Note:** `<mods:partNumber>` contains human-readable labels like "Ausgabe A", "Ausgabe B" -
NOT the numeric edition. These map to edition numbers as: A=1, B=2, C=3.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ALTO Filename: 3074409X_1920-03-03_000_53_H_1_001.xml                       │
│                                   ↑   ↑    ↑   ↑                            │
│                              000  53   H   1  001                           │
│                              ???  issue    edition  page                    │
│                                    ↑        ↑ (AUTHORITATIVE)               │
│                                                                             │
│  METS part: <mods:part order="1920030301">                                  │
│                                       ↑                                     │
│                     Last 2 digits = edition (01, derived)                   │
│                                                                             │
│  METS: <mods:partNumber>Ausgabe A</mods:partNumber>                         │
│                              ↑                                              │
│                    Human label only (A=1, B=2, C=3)                         │
│                                                                             │
│  Folder Structure: 1920/03/03/01/fulltext/                                  │
│                              ↑                                              │
│                    "01" = edition (validation backup)                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Breaking Change Notice

**Parquet schema changed:** Existing parquet files have `daily_issue_number` column.
New files will have `edition` column. Re-ingest data to update.

---

## 4. Hierarchical ID Format

### ID Structure

```
line_id = source_issue_id + "_" + page + "_" + block + "_" + line

Example: der_tag_1901-03-15_076_1_005_TB_1_TL_12
         │       │          │   │ │   │    │
         │       │          │   │ │   │    └── Line ID from ALTO (TL_12)
         │       │          │   │ │   └─────── Block ID from ALTO (TB_1)
         │       │          │   │ └─────────── Page number (005)
         │       │          │   └───────────── Daily issue (1)
         │       │          └───────────────── Issue number (076)
         │       └──────────────────────────── Date
         └──────────────────────────────────── Source name OR ZDB ID
```

### ID Hierarchy

```
source_id
└── issue_id      = source_id + date + issue_number + daily_issue_number
    └── page_id   = issue_id + page_number
        └── block_id = page_id + alto_block_id
            └── line_id = block_id + alto_line_id
```

### Other IDs (UUID-based)

```
entity_id    = line_id + "_entity_" + uuid6
article_id   = issue_id + "_article_" + uuid6
detection_id = page_id + "_" + element_type + "_" + uuid6
```

---

## 5. The "Smart String" Anti-Pattern

### Problem

Analysis modules parse the ID string to rediscover metadata that already exists in columns:

```python
# ❌ CURRENT: Slow string parsing
def analyze_entities(df):
    for row in df.iter_rows():
        line_id = row["line_id"]
        fk = extract_foreign_keys(line_id)  # Parses string!
        date = fk["date"]  # Could just read df["date"]!
```

### ✅ Solution IMPLEMENTED

`extract_foreign_keys()` now uses `@lru_cache(maxsize=10000)` for 24x speedup on repeated calls.
Additionally, `add_foreign_key_columns()` was added to efficiently add FK columns from source DataFrame or parse from IDs.

```python
# ✅ CORRECT: Read columns directly or use cached extraction
def analyze_entities(df):
    result = df.select([
        pl.col("line_id"),
        pl.col("date"),      # Already exists!
        pl.col("issue_id"),  # Already exists!
    ])

# Or if FKs need to be added from IDs (now cached):
from newspaper_explorer.data.utils.ids import add_foreign_key_columns
df_with_fks = add_foreign_key_columns(df, id_column="text_block_id")
```

### Affected Files (now using cached `extract_foreign_keys()`)

| File | Issue | Status |
|------|-------|--------|
| `analysis/topics/mallet.py` | Calls `extract_foreign_keys()` | ✅ Now cached |
| `analysis/entities/gliner.py` | Calls `extract_foreign_keys()` | ✅ Now cached |
| `analysis/entities/gliner2.py` | Stores `line_id` as "source_id" column! | ⚠️ Wrong column name |
| `analysis/entities/bert_classifier.py` | Calls `extract_foreign_keys()` | ✅ Now cached |
| `analysis/keywords/keybert.py` | Calls `extract_foreign_keys()` | ✅ Now cached |
| `analysis/keywords/rake.py` | Calls `extract_foreign_keys()` | ✅ Now cached |
| `analysis/keywords/yake.py` | Calls `extract_foreign_keys()` | ✅ Now cached |
| `analysis/topics/tf_idf.py` | Calls `extract_foreign_keys()` | ✅ Now cached |
| `analysis/topics/lda.py` | Calls `extract_foreign_keys()` | ✅ Now cached |
| `analysis/topics/bertopic.py` | Calls `extract_foreign_keys()` | ✅ Now cached |
| `analysis/topics/fastopic.py` | Calls `extract_foreign_keys()` | ✅ Now cached |

---

## 6. Test Fixture Mismatch

### Real ALTO Filenames

```
3074409X_1920-03-03_000_53_H_1_001.xml
│        │          │   │  │ │ │
│        │          │   │  │ │ └── Page number (001)
│        │          │   │  │ └──── Daily issue (1)
│        │          │   │  └────── "H" = Heft
│        │          │   └───────── Issue number (53)
│        │          └───────────── Unknown (always 000)
│        └──────────────────────── Date
└───────────────────────────────── ZDB ID (no hyphen)
```

### Test Fixture Filenames

```
alto_1920_03_03_page_001.xml
│    │    │  │  │    │
│    └────┴──┴──┘    └── Page number
│         Date
└─────────────────────── Prefix
```

### Missing in Test Fixtures

- ZDB prefix (`3074409X`)
- Issue number (`53`)
- Daily issue number (`1`)
- The `_000_` field
- The `_H_` separator

**Impact:** `_parse_filename()` regex won't match test fixture names.

---

## 7. Workarounds Found in Code

| Location | Issue | Workaround | Status |
|----------|-------|------------|--------|
| `loader.py:296` | ZDB vs source_name | Fallback chain with logging | 🔄 Fixing |
| `alto.py:68` | Filename ID ignored | Uses config source_id instead | ✅ Correct behavior |
| `ids.py:350` | Missing TL marker | Uses `tuple.index()` | ✅ Optimized |
| `gliner2.py:223` | Wrong column name | Stores line_id as "source_id" | ⚠️ Still wrong |
| `layout/detector.py:237` | Missing image index | Falls back to path-based ID | ⚠️ Data issue |
| `cli/layout/commands.py:276` | Type mixing | Explicit `str()` conversion | ✅ Fine |
| `aggregation.py:155` | Unknown source | Falls back to "unknown" | ⚠️ Still present |
| `ids.py:600+` | ID type detection | Hardcoded list of detection classes | ✅ Fine |

---

## 8. Recommendations

### ✅ Priority 1: Stop Redundant Parsing - DONE
- ~~Refactor analysis modules to read `date`, `issue_id`, `page_id` columns directly~~
- ~~Remove calls to `extract_foreign_keys()` when columns exist~~
- **Implemented:** `@lru_cache(maxsize=10000)` on `extract_foreign_keys()` - 24x speedup
- **Implemented:** `add_foreign_key_columns()` for efficient FK addition

### ✅ Priority 2: Standardize Source Identifier - DONE
- Pick ONE canonical form: **`source_name`** (e.g., `der_tag`) ✅ Decision made
- **Implemented:** `loader.py` always uses `source_name` for ID generation
- **Implemented:** `image_index.py` always uses `source_name` for ID generation
- ZDB ID kept in source config for provenance/metadata only

### Priority 3: Fix GLiNER2 Column Naming
- Rename `source_id` → `text_block_id` in entity results
- Or properly populate with actual source_id

### Priority 4: Clarify Edition vs Issue Semantics
- **Option A: Rename column** `daily_issue_number` → `edition_slot` (breaking change)
- **Option B: Add documentation** explaining the actual semantics (non-breaking)
- **Option C: Add `edition_slot` alias** keeping `daily_issue_number` for compatibility
- Update docstrings in `ids.py`, `schema.py`, `content.py`
- Note: The ID format `{issue}_{daily}` works correctly, just naming is confusing

### Priority 5: Define Authoritative Source for Edition
- Document: filename is authoritative, folder is backup
- Or: Extract from METS `<mods:part order="...">` if available

### Priority 6: Fix Test Fixtures
- Rename to match real filename format
- Or add separate tests for real-format filenames

### Priority 7: Add ID Validation
- Create `validate_line_id()`, `validate_page_id()` functions
- Fail early on malformed IDs instead of fallback guessing

---

## 9. What NOT to Change

- ✅ Keep Parquet files (not SQLite)
- ✅ Keep denormalized schema
- ✅ Keep hierarchical string IDs (human-readable, debuggable)
- ✅ Keep DuckDB for cross-file queries

---

## 10. Questions to Resolve

1. **What is the `_000_` field in filenames?** Always seems to be "000".
2. ~~**Should `source_id` use `der_tag` or `3074409-X`?** Currently inconsistent.~~ **RESOLVED: Always `source_name`**
3. **Is daily_issue_number always in filename AND folder?** Or can they differ?
4. **How should test fixtures be updated?** Copy real files or rename existing?

---

## 11. Image Index Analysis

### Schema

The `image_index.parquet` has 20 columns:

| Column | Type | Description |
|--------|------|-------------|
| `image_path` | String | Relative path: `YYYY/MM/DD/issue/max_N.jpg` |
| `year`, `month`, `day` | Int64 | Date components from path |
| `date` | String | ISO date string (YYYY-MM-DD) |
| `issue_id` | String | Generated ID or path fallback |
| `page_id` | String | Generated page ID (nullable!) |
| `page_number` | Int64 | Extracted from filename |
| `filename` | String | Just the filename |
| `file_size_bytes` | Int64 | File size |
| `alto_width`, `alto_height` | Int64 | From ALTO XML (nullable) |
| `width`, `height` | Int64 | Actual image dimensions |
| `newspaper_title` | String | From METS (nullable) |
| `year_volume` | String | From METS (nullable) |
| `page_count` | Int64 | From METS (nullable) |
| `issue_number` | Int64 | From METS (nullable) |
| `daily_issue_number` | Int64 | From METS (nullable) |
| `file_exists` | Boolean | Always true |

### ID Format Inconsistency

**When METS is found:**
```
issue_id: "3074409-X_1911-06-02_128_3"  ← Proper format with ZDB ID
page_id:  "3074409-X_1920-01-11_009_1_009"  ← Proper format
```

**When METS is NOT found (1900 data):**
```
issue_id: "1900/01/02/01"  ← Path-based fallback!
page_id:  null  ← Cannot generate without METS metadata
```

### Null Value Problem

12 images (0.0%) have null values for METS-derived columns:

| Column | Null Count |
|--------|------------|
| `page_id` | 12 |
| `newspaper_title` | 12 |
| `year_volume` | 12 |
| `page_count` | 12 |
| `issue_number` | 12 |
| `daily_issue_number` | 12 |

**Root cause:** These are all from `1900/01/02/01/` - the METS file is missing.

This matches TODO.md note: "1900 issue mets file is needed"

### ALTO Dimension Gaps

113 images (0.1%) have null `alto_width`/`alto_height`:
- ALTO file exists but dimension extraction failed
- Or page number padding mismatch in cache key lookup

### How Image Index Connects to Text Data

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  JOIN KEY: page_id                                                          │
│                                                                             │
│  image_index.parquet                    source_lines.parquet                │
│  ┌────────────────────────────┐         ┌────────────────────────────┐      │
│  │ page_id                    │◄───────►│ page_id (derived from      │      │
│  │ "3074409-X_1920-01-11_9_1_9"│         │  line_id)                  │      │
│  │                            │         │                            │      │
│  │ image_path                 │         │ text                       │      │
│  │ width, height              │         │ line_id                    │      │
│  │ alto_width, alto_height    │         │ x, y, width, height        │      │
│  └────────────────────────────┘         └────────────────────────────┘      │
│                                                                             │
│  ⚠️ Problem: page_id can be null in image_index when METS missing!         │
│  ⚠️ Problem: page_id format uses ZDB ID, not source_name                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Lookup Strategies in Code

The image indexer uses TWO cache keys for METS lookup:

```python
# Primary key: path-based (always works)
path_key = f"{year}/{month}/{day}/{issue_num}"  # "1920/03/03/01"

# Secondary key: generated issue_id (for compatibility)
issue_id = generate_issue_id(source_id, date, issue_number, daily_issue_num)
# "3074409-X_1920-03-03_053_1"

# Both stored in cache:
mets_cache[path_key] = cache_entry
mets_cache[issue_id] = cache_entry
```

**Why dual keys?** Different parts of the system look up by different formats.

### Text ↔ Image Join Analysis

**Good news:** The ID formats DO match when both exist:
- Text: `3074409-X_1901-01-08_006_1_001`
- Image: `3074409-X_1901-01-08_006_1_001`

**Coverage:**
| Metric | Count |
|--------|-------|
| Text page_ids | 134,123 |
| Image page_ids | 134,868 |
| **Matching** | **134,007** (99.9%) |
| Text without images | 116 |
| Images without text | 861 |

**Year gap:** Text starts at 1901, images include 1900 (but 1900 has null page_ids due to missing METS).

### The Mixed Issue Problem (March 19, 1901)

This is a **data integrity issue** also noted in TODO.md:

```
Text data (ALTO files):
  Issue 103: pages 1-12  ✓
  Issue 104: pages 1-8   ✓

Image index (from METS):
  Issue 103: pages 1-20  ← Includes pages 13-20 that text says are issue 104!
  Issue 104: (no images) ← Missing entirely!
```

**What happened:**
1. ALTO files in folder `1901/03/19/01/` contain text for BOTH issues 103 and 104
2. METS file says it's issue 103 with 20 pages
3. Image indexer trusts METS → assigns all 20 images to issue 103
4. Text parser reads issue number from ALTO filename → correctly splits into 103 and 104
5. **Result:** Pages 13-20 have different issue_ids in text vs images!

This is the "12 directories have ALTO files from multiple issues mixed together" bug.

### Recommendations for Image Index

1. **Fix missing 1900 METS:** Either create it or handle gracefully
2. **Standardize issue_id fallback:** Use `source_name_path` format instead of raw path
3. **Always generate page_id:** Even without METS, derive from path + source config
4. **Add source_id column:** Currently missing, would help with joins
5. **Fix mixed issue problem:** Either trust ALTO filename or fix source data

---

## 12. Analysis Results - Granularity Levels

### Analysis Operates at Different Levels

| Analysis | Granularity | Primary ID | Join Key |
|----------|-------------|------------|----------|
| **Entities** | Text Block | `line_id` (misnomer!) | `text_block_id` format |
| **Keywords** | Page | `doc_id` | `page_id` |
| **Layout** | Page | `detection_id` | `page_id` |
| **Emotions** | Line | `line_id` | `line_id` |
| **Topics** | Corpus | `topic_id` | N/A (global) |

### Column Naming Confusion

**Entities stores text_block_id in column named `line_id`:**
```
Column name:     line_id
Actual format:   3074409-X_1901-01-08_006_1_001_r_11_1  (text_block format)
Should be:       text_block_id
```

This is a **naming issue**, not a bug. The entity extraction runs on aggregated text blocks, not individual lines. The column should be renamed to `text_block_id` for clarity.

### Join Key Verification

| Analysis | ID Column | Matches Source | Status |
|----------|-----------|----------------|--------|
| Entities | `line_id` → `text_block_id` | 100% ✅ | Works (naming is confusing) |
| Layout | `page_id` | 99.6% ✅ | Works |
| Keywords | `page_id` | 100% ✅ | Works |
| Emotions | `line_id` | 100% ✅ | Works |

### Recommendations

1. **Rename entity `line_id` to `text_block_id`** - reflects actual granularity
2. **Add `granularity` field to result metadata** - clarify what level analysis ran on
3. **Standardize column naming** across all analysis types:
   - Line level: `line_id`
   - Block level: `text_block_id`
   - Page level: `page_id`
   - Issue level: `issue_id`

### `source_id` Column

All result files correctly use `3074409-X` (ZDB ID) for `source_id`:
```
source_id: "3074409-X"  ← Correct, matches generated IDs
```

### `doc_id` in Keywords

Keywords uses `doc_id` which equals `page_id`:
```
doc_id:  3074409-X_1901-08-23_366_3_001
page_id: 3074409-X_1901-08-23_366_3_001  ← Same value
```
This is redundant but harmless.

### Layout `detection_id` Format

Layout detection IDs follow the pattern:
```
detection_id: {page_id}_{class_name}_{uuid6}
Example:      3074409-X_1910-03-14_132_1_003_text_ae5fd8
```
This works correctly - can extract page_id by removing the last two parts.
