## Current Priorities

### CLI Refactoring: Unify output styling, input options, and command patterns (Issue #16)

Consolidate CLI improvements into a cohesive refactoring effort with reusable utilities.

**Phase 1: Infrastructure (COMPLETED)**
- [x] Create `cli/utils/output.py` - Unified output formatting (240 lines)
  - [x] header(), subheader(), result(), status(), success(), error(), warning(), tip(), complete()
  - [x] Standardize separator width (70), character (─), status markers ([OK]/[MISSING])
- [x] Create `cli/utils/options.py` - Reusable option decorators (339 lines, 12 decorators)
  - [x] source_option(), limit_option(), year_option(), input_type_option()
  - [x] text_column_option(), resume_option(), batch_size_option()
  - [x] output_path_option(), force_option()
  - [x] input_file_option(), output_name_option(), model_dir_option()
- [x] Create `cli/utils/paths.py` - Input path resolution (160 lines)
  - [x] resolve_input_path(), detect_id_column(), resolve_output_path()
  - [x] get_source_text_path(), get_source_images_path(), get_analysis_results_dir()
- [x] Create `cli/utils/errors.py` - Standard error handling (220 lines)
  - [x] handle_error(), handle_validation_error(), require_file(), require_directory()
  - [x] warn_if_missing(), confirm_overwrite()

**Phase 2: Option Unification (COMPLETED)**
- [x] Unified `--sample` → `--limit` in data/preprocessing.py
- [x] Unified `--sample-size` → `--limit` in data/info.py (analyze_chars, analyze_tokens)

**Phase 3: Reference Implementation (COMPLETED)**
- [x] Refactor `cli/analyze/emotions/commands.py` (280 lines)
  - [x] Apply all option decorators (source, input_file, text_column, batch_size, limit, model_dir, output_name)
  - [x] Use errors module (handle_error, handle_validation_error)
  - [x] Use output module throughout (header, key_value, success, error, code_block)
  - [x] Add `--limit` support to EmotionPredictor.predict() and predict_from_source()

**Phase 4: Migrate Analyze Commands (2/5 complete)**
- [x] `cli/analyze/entities/commands.py` (863 lines) - Entity extraction & network analysis - **COMPLETED**
  - [x] **Extraction commands** (gliner, gliner2, llm):
    - [x] Apply decorators: source_option, input_file_option, text_column_option, id_column_option, limit_option, batch_size_option, threshold_option, model_option, num_gpus_option, min_length_option, max_length_option, temperature_option, max_tokens_option
    - [x] Update error handling (errors.handle_error, errors.handle_validation_error)
    - [x] Update output styling (output.header, output.key_value, output.success, output.info)
    - [x] Extract input path resolution to cli/utils/paths.py (resolve_input_path)
    - [x] Create reusable option decorators in cli/utils/options.py (7 new options)
    - [x] **Removed output_name_option** - now uses standardized auto-naming via save_analysis_results()
  - [x] **Network commands** (network-stats, find-path, entity-connections):
    - [x] Replace click.echo() with output module (header, section, key_value, info)
    - [x] Replace manual "=" separators with output.section()
    - [x] Replace click.echo(err=True) with errors.handle_validation_error()
    - [x] Add return type annotations
    - [x] Standardize tips/help messages with muted info
  - [x] **Overall**:
    - [x] File reduced from 1003 to 863 lines (140 line reduction, 14%)
    - [x] All extraction methods support limit parameter
    - [x] All commands follow consistent output patterns
    - [x] **Bonus**: Refactored emotions to also use save_analysis_results() for consistency
- [ ] `cli/analyze/topics/commands.py` (1,787 lines) - LDA, BERTopic
  - [ ] Apply decorators: source_option, input_file_option, output_name_option, limit_option
  - [ ] Update error handling and output styling
  - [ ] Standardize across lda, bertopic, query commands
- [x] `cli/analyze/keywords/commands.py` (748 lines) - Keyword extraction - **COMPLETED (Phase 4: 2/5)**
  - [x] **Fully refactored with decorators and utilities**:
    - [x] Applied 16 decorator types across 4 commands (tfidf, rake, yake, keybert)
    - [x] Decorators: source_option, input_file_option, text_column_option, limit_option, top_k_option, group_by_option, output_name_option, batch_size_option, num_workers_option, min_length_option, max_length_option, device_option, use_chunking_option, chunk_size_option, chunk_overlap_option, compile_model_option
    - [x] Renamed --stopwords to --custom-stopwords in tfidf for clarity
    - [x] Unified num_workers_option across all CLI commands (replaced max_workers_option)
    - [x] Replaced 100% of click.echo() calls with output module (header, key_value, success, info, error)
    - [x] Removed 124 lines (14.2% reduction: 872 → 748 lines)
    - [x] All algorithm-specific options kept inline (no-stopwords, document_level, min_df, max_df, ngram_range, language, model, diversity, mmr, etc.)
- [ ] `cli/analyze/topics/commands.py` (1,777 lines) - Topic modeling - **IN PROGRESS (Phase 4: 3/5)**
  - [ ] **Commands**: lda, mallet, bertopic, fastopic, bertopic_batch
  - [x] Created analyze/topics/utils.py with 4 helper functions (parsing, formatting)
  - [x] Started LDA refactoring: Applied decorators (source, input_file, text_column, top_k, limit), renamed --stopwords → --custom-stopwords, partial output module conversion
  - [ ] Complete LDA refactoring: Convert remaining click.echo() to output module
  - [ ] Refactor mallet command with same pattern
  - [ ] Refactor bertopic command with same pattern
  - [ ] Refactor fastopic command with same pattern
  - [ ] Refactor bertopic_batch command with same pattern
  - [ ] Extract more CLI logic to analyze/topics/utils.py (workflow orchestration)
  - [ ] Target: 10-15% line reduction (similar to keywords)
- [ ] `cli/analyze/layout/commands.py` (1,447 lines) - YOLO layout detection
  - [ ] Apply decorators where applicable (different pattern - image-focused)
  - [ ] Update error handling and output styling
- [ ] `cli/analyze/captions/commands.py` (627 lines) - Image caption extraction
  - [ ] Apply decorators where applicable
  - [ ] Update error handling and output styling

**Phase 5: Migrate Data Commands (6/6 complete) ✅**
- [x] `cli/data/info.py` (664 lines) - COMPLETED
  - [x] Apply output module to analyze_chars, analyze_tokens
  - [x] Standardize table/stats formatting
- [x] `cli/data/preprocessing.py` (311 lines) - COMPLETED
  - [x] Apply output module for progress and results
- [x] `cli/data/download.py` (275 lines) - COMPLETED
  - [x] Update progress indicators and status messages
- [x] `cli/data/loading.py` (272 lines) - COMPLETED
  - [x] Update parsing output formatting
- [x] `cli/data/images.py` (208 lines) - COMPLETED
  - [x] Update download progress and status
- [x] `cli/data/validation.py` (313 lines) - COMPLETED
  - [x] Update validation output with consistent styling
  - [x] Replace bullet_list with numbered output.info()
  - [x] Add VALIDATING/RESULTS sections

**Phase 6: Extract Business Logic (Future)**
- [ ] Move analysis orchestration from CLI to library modules
- [ ] Create `analyze/{type}/core.py` modules for main processing
- [ ] Create `analyze/{type}/utils.py` for helpers
- [ ] CLI becomes thin presentation layer (Click decorators + output only)

**Breaking Changes**
- [x] `--sample` → `--limit` (data preprocess)
- [x] `--sample-size` → `--limit` (data analyze-tokens, analyze-chars)

---

### Backend Refactoring: Extract queries.py modules (Issue #2)

Move business logic from FastAPI routers to `analyze/*/queries.py` modules.
CLI and API will both call these functions. Routers become thin wrappers.

**Phase 0: Centralized result loading (COMPLETED)**
- [x] Add `load_analysis_results()` to `data/utils/results.py`
- [x] Add `list_analysis_runs()` to `data/utils/results.py`
- [x] Add `load_analysis_metadata()` to `data/utils/results.py`
- [x] Add `AnalysisType` literal to `models/data/metadata.py`
- [x] Create `models/api/results.py` for API response models
- [x] Create `models/api/entities.py` for entity API models (placeholder)
- [x] Handles both flat structure (legacy) and timestamped runs
- [x] Refactor `routers/results.py` to use centralized functions & models
- Pattern: All analysis modules use these loaders + API models from models/api/

**Phase 1: Create query modules**
- [x] `analyze/emotions/queries.py` - from routers/emotions.py (173 lines → 57 lines)
  - [x] `get_statistics(df) -> dict[str, Any]`
  - [x] `get_timeline(df, granularity) -> list[dict[str, Any]]`
  - [x] `get_peaks(df, emotion, limit) -> list[dict[str, Any]]`
  - [x] Move EMOTIONS constants to `models/analysis/emotions.py`
- [x] `analyze/entities/queries.py` - from routers/entities.py (224 lines → 85 lines)
  - [x] `aggregate_entities(df, entity_type, limit) -> pl.DataFrame`
  - [x] `get_entity_types(df) -> list[str]`
  - [x] `get_entity_occurrences(df, entity_text, entity_type, limit) -> pl.DataFrame`
  - [x] `get_timeline(df, entity_type, aggregation) -> dict`
- [ ] `analyze/keywords/queries.py` - from routers/keywords.py (511 lines)
  - [ ] `get_stats(df, filters) -> KeywordStats`
  - [ ] `get_keywords_paginated(df, filters, page, page_size) -> PaginatedKeywords`
  - [ ] `get_timeline(df, keyword) -> list[dict]`
  - [ ] `get_cooccurrences(df, keyword, limit) -> list[dict]`

**Phase 2: Refactor routers to use query modules**
- [x] routers/emotions.py - call analyze/emotions/queries.py (uses load_analysis_results)
- [x] routers/entities.py - call analyze/entities/queries.py (uses load_analysis_results)
- [ ] routers/keywords.py - call analyze/keywords/queries.py (use load_analysis_results)

**Phase 3: Larger routers (lower priority)**
- [ ] routers/layout.py (811 lines) - extract to analyze/layout/queries.py
- [ ] routers/data.py (797 lines) - extract to data/utils/browse.py
- [ ] routers/sources.py (224 lines) - extract to data/utils/status.py

**Phase 4: Cleanup (after all modules refactored)**
- [x] Remove ui/backend/utils/results.py (ResultsLoader now redundant - no longer used)
- [ ] Update all remaining routers to use load_analysis_results from data/utils/results.py

---

- update requirements.txt and requirements-dev.txt

- textblock coords?

+ Bohnſtedt (Graudenz) des Danziger Inſ.⸗Regts. Nr. 128, -> wrong long s test


- add source_name to aggregated blocks

- yolo, spacy, hf cache dir move to .env


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



]Skipping 3074409X_1901-07-14_000_297_H_1_-01.xml: Missing required ID components (date=None, issue=None, daily=None, page=None)
march 1901


Transkribus OCR -> Problem with alto-export

- validate xml files


- combine results with/in knowdledge graph

- keywords ui broken


- sort dataset by filename when loading

- jsonrepair libary for llm client


- null values in image_index.parquet

- option to collapse analysis header


pagination for entiteies list

refactor captions cli

picture detail dialog should link to page.

thumbnail gallery lazy loading broken

- emotions: link to pages in noticable peaks


page reader -> click line scroll text


sorting broken in search -> 1910 Daten problem?

brwose link layout detection detailed region in analysis tab to overlay

issuereader, link emotion detections to line

simple stats page? word count etc?

drowpdon filters in browse month view

support external image urls -> include in index


emotions cli: use list-models

keywords commands: tfidf gourp by / document level clarification

keywords commands: tfidf min-df and max-df do what?

topics commands: rework merge-yearly