## Current Priorities
- update requirements.txt and requirements-dev.txt
- unify ids, generation and usage, more linability
- remove emojis from cli
- unifiy output styling

### ID System Optimization
- **Redundant FK parsing**: Analysis modules re-parse IDs via `extract_foreign_keys()` even though FK columns already exist in source parquet
  - Affects: mallet.py, gliner.py, bert_classifier.py, keybert.py, rake.py, yake.py, tf_idf.py, lda.py, bertopic.py, fastopic.py
  - Fix: Check if FK columns exist before parsing, preserve FKs through aggregation
- **Expensive string parsing**: `parse_line_id()` uses O(n) loops to find date/TL markers
  - Fix: Use vectorized Polars string operations or compiled regex
- **No caching**: Same IDs parsed repeatedly (e.g., multiple entities from same line)
  - Fix: Add `@lru_cache` to `extract_foreign_keys()`
- **Storage bloat**: Hierarchical IDs embed parent info redundantly
  - Consider: Normalized integer IDs with lookup tables for long-term

### CLI Refactoring - Move Logic to Library
- **Problem**: CLI contains ~8,500 lines with significant business logic that should be in library modules
- **Worst offenders**:
  - `cli/analyze/topics/commands.py` (1,787 lines) - topic modeling orchestration
  - `cli/analyze/layout/commands.py` (1,446 lines) - detection flattening, stats
  - `cli/analyze/entities/commands.py` (961 lines) - entity extraction orchestration
  - `cli/analyze/keywords/commands.py` (923 lines) - keyword extraction orchestration
  - `cli/data/info.py` (897 lines) - token/char analysis, completeness checks
- **What to move**:
  - DataFrame transformations → library modules
  - Statistics/analysis functions → `data/utils/` or `analyze/`
  - Result saving patterns → shared utilities
  - Progress bar wrapping → decorators/context managers
- **CLI should only**:
  - Parse arguments
  - Call library functions
  - Format output with click.echo()
  - Handle errors gracefully
- **Benefits**: Testable, reusable, cleaner separation of concerns

- add source_name to aggregated blocks
- spacy models?
- yolo, spacy, hf cache dir move to .env
- unify sample/limit cmd
- do bertopic year by year

- normalization output always named textblocks

### UI Issues
- use image index consitently
- datamanager tab?
- permalink to pages in issue browser to jump to them from other tabs
- page samples not reactive slightly buggy
- add drop shadows all around to flat elements
- support preprocessing metadata
### Data Issues
- Images need fixing too
- 1900 issue mets file is needded

- **Data integrity: Mixed issues in same directory**
  - 12 directories have ALTO files from multiple issues mixed together (0.09% of data)
  - Example: `1901/03/19/01` has issue 103 METS but contains ALTO files from both 103 and 104
  - Issue 103: Missing pages 13-20 (expected by METS, not found)
  - Issue 104: Has pages 1-8 in wrong directory with no METS file
  - Images are also affected/mixed up
  - Need to investigate if files are mislabeled or if data is genuinely incomplete
  - Validation command exists: `newspaper-explorer data validate-alto-mets --source der_tag`
  - Detection in fixes.py exists but no automatic repair (returns 0)
  - **Action**: Manual investigation of source data or contact data provider

- Finish downloading and fixing dataset -> Final testing

### Features
- CLI option to cleanup downloads
- Page Classification using DONUT -> Detect Ads?
- Update EntityExtractor to use preprocessed data (remove inline normalization)



]Skipping 3074409X_1901-07-14_000_297_H_1_-01.xml: Missing required ID components (date=None, issue=None, daily=None, page=None)
march 1901


Transkribus OCR -> Problem with alto-export

- validate xml files


- combine results with/in knowdledge graph

- keywords ui broken

- unify results output

- sort dataset by filename when loading

- jsonrepair libary for llm client

-  "- und" nicht dehyphen

- null values in image_index.parquet

- option to collapse analysis header

- year picker in date range filter
- slider in date range

pagination for entiteies list

refactor captions cli

remove title from charts or improve/unify?

picture detail dialog should link to page.

thumbnail gallery lazy loading broken

- emotions: link to pages in noticable peaks


page reader -> click line scroll text


filter textlines that contain only one or two repeating characters
filter textlines that contain only numbers

sorting broken in search -> 1910 Daten problem?

brwose link layout detection detailed region in analysis tab to overlay

issuereader, link emotion detections to line

simple stats page? word count etc?

drowpdon filters in browse month view

date range filter custom range text of

support external image urls -> include in index