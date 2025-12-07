# Preprocessing Tab - Design Document

## Overview

A new interactive tab for the v2 UI that allows users to:
1. **Visualize** the effects of preprocessing steps on sample text
2. **Configure** custom preprocessing pipelines via drag-and-drop
3. **Create** preprocessed datasets directly from the UI

## User Interface Design

### Layout Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  [Header: Preprocessing Pipeline Builder]                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─ Available Steps ──────────────────────────────────────────────────────┐ │
│  │                                                                         │ │
│  │  [Normalization]  [Cleaning]  [Filtering]  [Linguistic]   [Presets ▼]  │ │
│  │                                                                         │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     │ │
│  │  │ Unicode  │ │ Long S   │ │ Whitespc │ │ Casing   │ │ Umlauts  │ ... │ │
│  │  │ Normal.  │ │ Normal.  │ │ Normal.  │ │ Normal.  │ │ Normal.  │     │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘     │ │
│  │                                                                         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│                              ↓ Drag & Drop ↓                               │
│                                                                             │
│  ┌─ Active Pipeline ───────────────────────────────────────────────────────┐ │
│  │                                                                         │ │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐          │ │
│  │  │ Unicode  │ → │ Long S   │ → │ Dehyphen │ → │ Lowercase│  [+ Add]  │ │
│  │  │ [⚙] [×] │    │ [⚙] [×] │    │ [⚙] [×] │    │ [⚙] [×] │          │ │
│  │  └──────────┘    └──────────┘    └──────────┘    └──────────┘          │ │
│  │                                                                         │ │
│  │  Drag to reorder • Click ⚙ to configure • Click × to remove            │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌─ Live Preview ──────────────────────────────────────────────────────────┐ │
│  │                                                                         │ │
│  │  ┌─ Original ────────────────┐  ┌─ After Pipeline ──────────────────┐  │ │
│  │  │ Der Reichstag iſt am      │  │ Der Reichstag ist am              │  │ │
│  │  │ Sonnabend, den 15. Januar │  │ sonnabend den 15 januar           │  │ │
│  │  │ 1910, zu⸗                 │  │ 1910 zusammengetreten             │  │ │
│  │  │ ſammengetreten.           │  │                                   │  │ │
│  │  └───────────────────────────┘  └───────────────────────────────────┘  │ │
│  │                                                                         │ │
│  │  [Random Sample] [Use Custom Text]  Changes: 15 chars • -2 lines       │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌─ Step-by-Step Diff (collapsible) ───────────────────────────────────────┐ │
│  │  Step 1: normalize_unicode    →  "Der Reichstag iſt am..."             │ │
│  │  Step 2: normalize_long_s     →  "Der Reichstag ist am..."  [+1 char]  │ │
│  │  Step 3: dehyphenate_auto     →  "...zusammengetreten"      [merged]   │ │
│  │  Step 4: normalize_casing     →  "der reichstag ist..."     [lowered]  │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌─ Export Options ────────────────────────────────────────────────────────┐ │
│  │                                                                         │ │
│  │  Source: [der_tag ▼]    Output: preprocessed_custom_2024-12-07         │ │
│  │                                                                         │ │
│  │  ☑ Include original text column                                        │ │
│  │  ☑ Save pipeline configuration as preset                               │ │
│  │                                                                         │ │
│  │  [Run Pipeline on Full Dataset]     Est. time: ~15 min for 2.3M lines  │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Available Steps Panel (`PreprocessingStepPalette.vue`)

Horizontal scrollable list of all available preprocessing steps, organized by category:

**Categories** (tabs or filter):
- **Normalization**: `normalize_unicode`, `normalize_long_s`, `normalize_umlauts`, `normalize_whitespace`, `normalize_casing`, `dehyphenate_auto`
- **Modernization**: `modernization_transnormer`, `modernization_dta-cab`
- **Cleaning**: `remove_diacritics`, `remove_punctuation`, `remove_numbers`, `remove_stopwords`, `only_keep_allowed_chars`
- **Filtering**: `filter_by_total_character_length`, `filter_by_word_count`, `filter_repeating_chars`, `filter_number_only_lines`, `filter_empty_lines`, `filter_by_char_token_ratio`, `filter_by_max_word_length`
- **Linguistic**: `lemmatize_spacy`, `lemmatize_germalemma`
- **Quality**: `calculate_quality_metrics`, `filter_by_quality_score`

**Each step card shows**:
- Icon representing category
- Step name (human-readable)
- Tooltip with description
- Draggable handle

**Presets dropdown**: Load predefined pipelines (minimal, basic, standard, advanced, full, entities, topics, etc.)

### 2. Active Pipeline Area (`PreprocessingPipelineBuilder.vue`)

Drop zone where users build their pipeline:

**Features**:
- Drag-and-drop from palette
- Reorderable via drag-and-drop
- Each step shows:
  - Name
  - Configure button (⚙) → opens modal for step-specific args
  - Remove button (×)
- Visual arrows showing data flow
- "Empty pipeline" placeholder when no steps

**Step Configuration Modal** (`PreprocessingStepConfig.vue`):
- Different form fields based on step type
- Example for `filter_by_word_count`:
  - `min_words`: number input (default: 0)
  - `max_words`: number input (default: ∞)
- Example for `normalize_whitespace`:
  - `keep_newlines`: checkbox (default: false)

### 3. Live Preview Panel (`PreprocessingPreview.vue`)

Side-by-side comparison of original vs. processed text:

**Left side**: Original text
**Right side**: Processed text (after all pipeline steps)

**Features**:
- Auto-updates on pipeline changes (debounced)
- "Random Sample" button → fetches random lines from source
- "Use Custom Text" toggle → allows pasting custom text for testing
- Diff highlighting (optional): show changed characters
- Statistics: character count diff, line count diff

### 4. Step-by-Step Diff (`PreprocessingStepDiff.vue`)

Collapsible section showing intermediate results:

- Shows output after each step
- Highlights what changed in that step
- Useful for debugging pipeline order

### 5. Export Panel (`PreprocessingExport.vue`)

Controls for running the pipeline on the full dataset:

**Options**:
- Source selector (from available sources)
- Output name (auto-generated or custom)
- Include original text column (checkbox)
- Save as preset (checkbox + preset name input)

**Actions**:
- "Run Pipeline" button → starts background job
- Progress indicator (if running)
- Estimated time based on line count and step complexity

## API Endpoints

### New Backend Routes (`routers/preprocessing.py`)

```python
# GET /api/preprocessing/steps
# Returns list of all available preprocessing steps with metadata
@router.get("/steps")
async def get_preprocessing_steps() -> list[PreprocessingStepInfo]:
    """Get all available preprocessing steps with descriptions and parameters."""
    pass

# GET /api/preprocessing/presets
# Returns list of all available presets
@router.get("/presets")
async def get_preprocessing_presets() -> dict[str, PresetInfo]:
    """Get all available preprocessing pipeline presets."""
    pass

# POST /api/preprocessing/preview
# Apply pipeline to sample text and return result
@router.post("/preview")
async def preview_preprocessing(
    request: PreprocessingPreviewRequest
) -> PreprocessingPreviewResponse:
    """Apply pipeline to sample text and return step-by-step results."""
    pass

# POST /api/preprocessing/run
# Run pipeline on full dataset (background task)
@router.post("/{source_name}/run")
async def run_preprocessing_pipeline(
    source_name: str,
    request: PreprocessingRunRequest,
    background_tasks: BackgroundTasks
) -> PreprocessingRunResponse:
    """Start preprocessing pipeline as background task."""
    pass

# GET /api/preprocessing/{source_name}/status
# Check status of running preprocessing job
@router.get("/{source_name}/status")
async def get_preprocessing_status(source_name: str) -> PreprocessingStatusResponse:
    """Get status of running or completed preprocessing job."""
    pass
```

### Request/Response Models

```python
from pydantic import BaseModel
from typing import Any, Optional

class PreprocessingStepInfo(BaseModel):
    name: str
    display_name: str
    description: str
    category: str  # normalization, cleaning, filtering, linguistic
    parameters: list[StepParameter]
    is_filter: bool  # Does it filter rows vs transform text?

class StepParameter(BaseModel):
    name: str
    type: str  # "int", "float", "bool", "string", "select"
    default: Any
    description: str
    options: Optional[list[str]] = None  # For select type

class PresetInfo(BaseModel):
    name: str
    description: str
    use_case: str
    steps: list[dict]  # Step configs with args

class PipelineStep(BaseModel):
    name: str
    args: dict[str, Any] = {}

class PreprocessingPreviewRequest(BaseModel):
    text: str
    steps: list[PipelineStep]
    show_intermediate: bool = False

class StepResult(BaseModel):
    step_name: str
    output: str
    changes_description: str

class PreprocessingPreviewResponse(BaseModel):
    original: str
    final: str
    intermediate_steps: list[StepResult]
    stats: dict[str, Any]  # char_diff, line_diff, etc.

class PreprocessingRunRequest(BaseModel):
    steps: list[PipelineStep]
    output_name: Optional[str] = None
    include_original: bool = True
    save_as_preset: Optional[str] = None

class PreprocessingRunResponse(BaseModel):
    job_id: str
    estimated_time_seconds: int
    output_path: str

class PreprocessingStatusResponse(BaseModel):
    job_id: str
    status: str  # "pending", "running", "completed", "failed"
    progress: float  # 0.0 - 1.0
    current_step: Optional[str]
    error: Optional[str]
```

## Frontend Implementation

### New Files

```
frontend/src/
├── views/
│   └── PreprocessingView.vue           # Main view
├── components/
│   └── preprocessing/
│       ├── StepPalette.vue             # Draggable step cards
│       ├── PipelineBuilder.vue         # Drop zone for building pipeline
│       ├── StepCard.vue                # Individual step in pipeline
│       ├── StepConfigModal.vue         # Configure step parameters
│       ├── PreviewPanel.vue            # Live preview
│       ├── StepDiffPanel.vue           # Step-by-step diff
│       └── ExportPanel.vue             # Run on full dataset
├── stores/
│   └── preprocessing.ts                # Pinia store for state
└── types/
    └── preprocessing.ts                # TypeScript interfaces
```

### Pinia Store (`stores/preprocessing.ts`)

```typescript
import { defineStore } from 'pinia'
import api from '@/lib/api'

interface PipelineStep {
  id: string  // Unique ID for drag-and-drop
  name: string
  args: Record<string, any>
}

interface PreprocessingState {
  availableSteps: PreprocessingStepInfo[]
  presets: Record<string, PresetInfo>
  pipeline: PipelineStep[]
  previewText: string
  previewResult: PreprocessingPreviewResponse | null
  isLoading: boolean
  runningJob: PreprocessingStatusResponse | null
}

export const usePreprocessingStore = defineStore('preprocessing', {
  state: (): PreprocessingState => ({
    availableSteps: [],
    presets: {},
    pipeline: [],
    previewText: '',
    previewResult: null,
    isLoading: false,
    runningJob: null,
  }),

  actions: {
    async loadSteps() { /* ... */ },
    async loadPresets() { /* ... */ },
    addStep(stepName: string) { /* ... */ },
    removeStep(stepId: string) { /* ... */ },
    reorderSteps(fromIndex: number, toIndex: number) { /* ... */ },
    updateStepArgs(stepId: string, args: Record<string, any>) { /* ... */ },
    loadPreset(presetName: string) { /* ... */ },
    async preview() { /* ... */ },
    async runPipeline(sourceName: string, options: RunOptions) { /* ... */ },
    async checkJobStatus(sourceName: string) { /* ... */ },
  },
})
```

### Drag-and-Drop

Use Vue's native drag-and-drop or a library like `@vueuse/core` draggable:

```vue
<script setup lang="ts">
import { useDraggable, useDropZone } from '@vueuse/core'

// In StepPalette.vue - make steps draggable
// In PipelineBuilder.vue - accept drops and allow reordering
</script>
```

Or consider using `vue-draggable-next` for more complex DnD:

```json
// Add to package.json dependencies
"vuedraggable": "^4.1.0"
```

## Router Update

```typescript
// router/index.ts - add to children array
{
  path: 'preprocessing',
  name: 'preprocessing',
  component: () => import('@/views/PreprocessingView.vue'),
},
```

## Navigation Update

```typescript
// layouts/MainLayout.vue - add to navigation array
{ name: 'Preprocessing', to: '/preprocessing', icon: Wand },  // or Settings, Sliders icon
```

## Step Metadata Registry

Create a registry that maps step names to UI metadata:

```typescript
// lib/preprocessing-steps.ts
export const STEP_REGISTRY: Record<string, StepUIConfig> = {
  normalize_unicode: {
    displayName: 'Unicode Normalization',
    description: 'Normalize Unicode characters (NFKC), fix quotes, spaces, and control characters',
    category: 'normalization',
    icon: 'Type',
    color: '#2E5EFF',
    parameters: [],
  },
  normalize_long_s: {
    displayName: 'Long S → Modern S',
    description: 'Convert historical long s (ſ) to modern s',
    category: 'normalization',
    icon: 'Replace',
    color: '#2E5EFF',
    parameters: [
      { name: 'mode', type: 'select', default: 'simple', options: ['simple', 'context', 'preserve'] },
    ],
  },
  filter_by_word_count: {
    displayName: 'Filter by Word Count',
    description: 'Remove lines with too few or too many words',
    category: 'filtering',
    icon: 'Filter',
    color: '#FF9100',
    parameters: [
      { name: 'min_words', type: 'int', default: 0, description: 'Minimum word count' },
      { name: 'max_words', type: 'int', default: null, description: 'Maximum word count (null = no limit)' },
    ],
  },
  // ... more steps
}
```

## Implementation Priority

### Phase 1: Preview Demo (MVP)
1. Backend: `/preprocessing/steps` and `/preprocessing/preview` endpoints
2. Frontend: Basic view with step palette, pipeline builder, preview panel
3. No persistence, no full dataset processing

### Phase 2: Full Pipeline Execution
1. Backend: `/preprocessing/run` and `/preprocessing/status` endpoints
2. Frontend: Export panel with progress tracking
3. Background job management

### Phase 3: Presets & Persistence
1. Load/save presets from UI
2. Custom preset creation
3. Pipeline history

## Technical Considerations

### Performance
- **Preview debouncing**: Wait 300ms after pipeline changes before re-running preview
- **Sample size**: Use 5-10 random lines for preview, not full dataset
- **Streaming**: For large datasets, consider streaming progress updates via WebSocket or SSE

### Step Complexity Indicator
Some steps are fast (normalize_unicode), others are slow (transnormer). Show visual indicator:
- ⚡ Fast (< 1ms per line)
- 🔄 Medium (1-10ms per line)
- ⏳ Slow (GPU/API required)

### Error Handling
- Invalid step order (e.g., filter after filter removes all data)
- API failures for DTA-CAB
- GPU out of memory for Transnormer

## Future Enhancements

1. **A/B Comparison**: Compare two different pipelines side-by-side
2. **Pipeline Templates**: Share pipelines with other users
3. **Batch Preview**: Preview on multiple random samples simultaneously
4. **Quality Metrics**: Show OCR quality metrics before/after pipeline
5. **Undo/Redo**: Pipeline editing history
6. **Export as CLI**: Generate CLI command for the configured pipeline

---

## Appendix: Backend Integration Details

### Existing Pipeline Infrastructure

The preprocessing module already provides everything needed for the backend:

```python
# From pipeline.py - STEP_CONFIG maps step names to functions
STEP_CONFIG: dict[str, dict[str, Any]] = {
    "normalize_unicode": {"func": normalize_unicode},
    "normalize_whitespace": {"func": normalize_whitespace},
    "normalize_casing": {"func": normalize_casing},
    "normalize_umlauts": {"func": normalize_umlauts},
    "normalize_long_s": {"func": normalize_long_s, "extra_args": {"mode": "simple"}},
    "dehyphenate_auto": {"func": dehyphenate_auto},
    "modernization_transnormer": {"func": transnormer, "special": "transnormer"},
    "modernization_dta-cab": {"func": dta_cab},
    "remove_diacritics": {"func": remove_diacritics},
    "remove_punctuation": {"func": remove_punctuation},
    "remove_numbers": {"func": remove_numbers},
    "remove_stopwords": {"func": remove_stopwords},
    "only_keep_allowed_chars": {"func": only_keep_allowed_chars},
    "lemmatize_spacy": {"func": lemmatize_spacy},
    "lemmatize_germalemma": {"func": lemmatize_germalemma},
    "filter_by_total_character_length": {"func": filter_by_total_character_length, "filter_only": True},
    "filter_by_word_count": {"func": filter_by_word_count, "filter_only": True},
    "filter_repeating_chars": {"func": filter_repeating_chars},
    "filter_number_only_lines": {"func": filter_number_only_lines, "filter_only": True},
    "filter_by_char_token_ratio": {"func": filter_by_char_token_ratio, "filter_only": True},
    "filter_by_max_word_length": {"func": filter_by_max_word_length, "filter_only": True},
    "filter_empty_lines": {"func": filter_empty_lines, "filter_only": True},
    "calculate_quality_metrics": {"func": calculate_quality_metrics, "filter_only": True},
    "filter_by_quality_score": {"func": filter_by_quality_score, "filter_only": True, "no_args": True},
}
```

### Using Existing Presets

```python
# From presets.py - can be loaded directly
from newspaper_explorer.data.preprocessing.presets import (
    get_preset,           # Get steps for a preset name
    list_presets,         # List all available presets
    get_preset_info,      # Get preset description, use_case, steps
    ALL_PIPELINES,        # All pipelines dict
    GENERAL_PIPELINES,    # minimal, basic, standard, advanced, full
    ANALYSIS_PIPELINES,   # entities, topics, emotions, keywords, embeddings, concepts
)
```

### Using run_preprocessing for Full Dataset

```python
# From pipeline.py - high-level function for background tasks
from newspaper_explorer.data.preprocessing.pipeline import run_preprocessing

result = run_preprocessing(
    source="der_tag",
    steps=[
        "normalize_unicode",
        {"name": "filter_by_word_count", "args": {"min_words": 2}},
        "normalize_casing",
    ],
    text_column="text",
    output_column="text_processed",
    sample=None,  # None = full dataset
    save=True,
)

# Result contains:
# - result.metadata: PreprocessingMetadata
# - result.results_path: Path to output parquet
# - result.input_rows / output_rows: Row counts
# - result.duration_seconds: Processing time
# - result.sample_original / sample_processed: Sample text for preview
```

### Preview Implementation (New Code Needed)

For the preview endpoint, we need a lightweight version that doesn't save:

```python
# New function for routers/preprocessing.py
async def preview_pipeline(
    text: str,
    steps: list[dict],
    show_intermediate: bool = False,
) -> dict:
    """
    Apply pipeline to sample text without saving.

    Returns step-by-step results for visualization.
    """
    import polars as pl
    from newspaper_explorer.data.preprocessing.pipeline import TextPreprocessor
    from newspaper_explorer.data.preprocessing.presets import get_step_name

    # Create single-row DataFrame
    df = pl.DataFrame({"text": [text]})

    preprocessor = TextPreprocessor(text_column="text")
    intermediate_results = []
    current_text = text

    for i, step in enumerate(steps):
        step_name = get_step_name(step)

        # Apply single step
        df_result = preprocessor.pipeline(
            df,
            steps=[step],
            output_column="text_out",
        )

        new_text = df_result["text_out"][0]

        if show_intermediate:
            intermediate_results.append({
                "step_name": step_name,
                "output": new_text,
                "changes_description": _describe_changes(current_text, new_text),
            })

        # Update for next step
        df = df_result.rename({"text_out": "text"})
        current_text = new_text

    return {
        "original": text,
        "final": current_text,
        "intermediate_steps": intermediate_results,
        "stats": {
            "char_diff": len(current_text) - len(text),
            "original_length": len(text),
            "final_length": len(current_text),
        },
    }
```

---

## Appendix: Step Parameter Documentation

Each step's configurable parameters (for the UI config modal):

| Step | Parameter | Type | Default | Description |
|------|-----------|------|---------|-------------|
| `normalize_whitespace` | `keep_newlines` | bool | false | Preserve line breaks |
| `normalize_long_s` | `mode` | select | "simple" | Options: simple, context, preserve |
| `filter_by_word_count` | `min_words` | int | 0 | Minimum word count |
| `filter_by_word_count` | `max_words` | int | null | Maximum word count |
| `filter_by_total_character_length` | `min_chars` | int | 0 | Minimum character count |
| `filter_by_total_character_length` | `max_chars` | int | null | Maximum character count |
| `filter_repeating_chars` | `max_repeats` | int | 3 | Max consecutive identical chars |
| `filter_by_char_token_ratio` | `max_ratio` | float | 10.0 | Max chars per token |
| `filter_by_max_word_length` | `max_length` | int | 30 | Max word length |
| `modernization_transnormer` | `model` | select | "19c" | Options: 19c, 18-19c |
| `modernization_transnormer` | `batch_size` | int | 32 | Batch size for inference |
| `modernization_transnormer` | `num_beams` | int | 4 | Beam search width |
| `modernization_dta-cab` | `batch_size` | int | 100 | Batch size for API calls |
| `modernization_dta-cab` | `timeout` | int | 30 | API timeout in seconds |
| `lemmatize_spacy` | `model` | select | "de_core_news_sm" | spaCy model |
| `remove_stopwords` | `language` | select | "german" | Stopword language |

Most other steps have no configurable parameters.
