# ID System Analysis

**Date:** 2025-12-07
**Status:** Analysis complete, no code changes yet

## Executive Summary

The ID system works but has inconsistencies and workarounds due to:
1. Multiple identifier formats for the same newspaper (ZDB ID, filename prefix, source name)
2. Issue number vs. daily issue count confusion
3. Test fixtures that don't match real data format
4. Redundant ID parsing in analysis modules (performance issue)

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

## 2. Source Identifier Chaos

Three different identifiers exist for the same newspaper:

| Identifier | Example | Where Used |
|------------|---------|------------|
| **ZDB Source ID** | `3074409-X` | Config file (`der_tag.json`), preferred for generated IDs |
| **Filename Prefix** | `3074409X` | ALTO/METS filenames (no hyphen!) |
| **Source Name** | `der_tag` | CLI arguments, config filename, fallback |

### Format Differences

```
Config:    zdb_source_id = "3074409-X"    ← WITH hyphen
Filename:  3074409X_1920-03-03_...        ← WITHOUT hyphen
METS URL:  SNP3074409X-19200303-...       ← No hyphen, "SNP" prefix, compact date
```

### Code Handling

**`ids.py`** has a conversion function:
```python
def zdb_id_to_filename_prefix(zdb_id: str) -> str:
    """Convert ZDB ID to filename prefix format."""
    return zdb_id.replace("-", "")  # "3074409-X" → "3074409X"
```

**`loader.py:296-302`** chooses which to use:
```python
source_id = self.source_name
if self.config_data and self.config_data.metadata.zdb_source_id:
    source_id = self.config_data.metadata.zdb_source_id
```

**Problem:** Generated IDs can be either:
- `3074409-X_1902-09-05_...` (with ZDB)
- `der_tag_1902-09-05_...` (without ZDB)

---

## 3. Issue Number vs. Daily Issue Count

### Data Sources (Three Different Places!)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ALTO Filename: 3074409X_1920-03-03_000_53_H_1_001.xml                       │
│                                   ↑   ↑    ↑   ↑                            │
│                              000  53   H   1  001                           │
│                              ???  issue    daily  page                      │
│                                                                             │
│  METS XML: <mods:number>Nr. 53, 03. März 1920</mods:number>                 │
│                          ↑                                                  │
│                     issue=53  (NO daily count in METS!)                     │
│                                                                             │
│  METS part: <mods:part order="1920030301">                                  │
│                                       ↑                                     │
│                     Last 2 digits might be daily issue (01)                 │
│                                                                             │
│  Folder Structure: 1920/03/03/01/fulltext/                                  │
│                              ↑                                              │
│                         "01" = daily issue number                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Insight

- **Issue number** (e.g., 53) = Annual sequential number (issue 53 of year 1920)
- **Daily issue number** (e.g., 1 or 2) = Which edition that day (Morgen/Abend)

Example with 2nd daily issue:
```
Filename: 3074409X_1915-03-14_000_62_H_2_009.xml
                                    ↑
                               2nd edition of the day
Folder:   1915/03/14/02/fulltext/
                    ↑
               Also shows "02"
```

### Current Behavior

- METS parser extracts `issue_number` but NOT `daily_issue_number`
- ALTO parser gets both from filename
- Image indexer gets `daily_issue_number` from folder structure
- **No authoritative source defined** → potential for mismatches

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

### Solution

```python
# ✅ CORRECT: Read columns directly
def analyze_entities(df):
    result = df.select([
        pl.col("line_id"),
        pl.col("date"),      # Already exists!
        pl.col("issue_id"),  # Already exists!
    ])
```

### Affected Files

| File | Issue |
|------|-------|
| `analysis/topics/mallet.py` | Calls `extract_foreign_keys()` |
| `analysis/entities/gliner.py` | Calls `extract_foreign_keys()` |
| `analysis/entities/gliner2.py` | Stores `line_id` as "source_id" column! |
| `analysis/entities/bert_classifier.py` | Calls `extract_foreign_keys()` |
| `analysis/keywords/keybert.py` | Calls `extract_foreign_keys()` |
| `analysis/keywords/rake.py` | Calls `extract_foreign_keys()` |
| `analysis/keywords/yake.py` | Calls `extract_foreign_keys()` |
| `analysis/topics/tf_idf.py` | Calls `extract_foreign_keys()` |
| `analysis/topics/lda.py` | Calls `extract_foreign_keys()` |
| `analysis/topics/bertopic.py` | Calls `extract_foreign_keys()` |
| `analysis/topics/fastopic.py` | Calls `extract_foreign_keys()` |

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

| Location | Issue | Workaround |
|----------|-------|------------|
| `loader.py:296` | ZDB vs source_name | Fallback chain with logging |
| `alto.py:68` | Filename ID ignored | Uses config source_id instead |
| `ids.py:350` | Missing TL marker | Guesses block/line split |
| `gliner2.py:223` | Wrong column name | Stores line_id as "source_id" |
| `layout/detector.py:237` | Missing image index | Falls back to path-based ID |
| `cli/layout/commands.py:276` | Type mixing | Explicit `str()` conversion |
| `aggregation.py:155` | Unknown source | Falls back to "unknown" |
| `ids.py:600+` | ID type detection | Hardcoded list of detection classes |

---

## 8. Recommendations

### Priority 1: Stop Redundant Parsing
- Refactor analysis modules to read `date`, `issue_id`, `page_id` columns directly
- Remove calls to `extract_foreign_keys()` when columns exist

### Priority 2: Standardize Source Identifier
- Pick ONE canonical form (recommend: `source_name` like `der_tag`)
- Keep ZDB ID as optional metadata only
- Remove `zdb_id_to_filename_prefix()` complexity

### Priority 3: Fix GLiNER2 Column Naming
- Rename `source_id` → `origin_id` or `parent_id` in entity results
- Or properly populate with actual source_id

### Priority 4: Define Authoritative Source for Daily Issue
- Document: filename is authoritative, folder is backup
- Or: Extract from METS `<mods:part order="...">` if available

### Priority 5: Fix Test Fixtures
- Rename to match real filename format
- Or add separate tests for real-format filenames

### Priority 6: Add ID Validation
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
2. **Should `source_id` use `der_tag` or `3074409-X`?** Currently inconsistent.
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
