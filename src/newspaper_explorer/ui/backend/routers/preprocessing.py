"""
Preprocessing pipeline endpoints for the UI.

Provides API access to:
- List available preprocessing steps and their parameters
- List available presets (predefined pipelines)
- Preview pipeline effects on sample text
- Run preprocessing on full datasets (background task)
"""

from datetime import datetime
from pathlib import Path
from typing import Any
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
import polars as pl
from pydantic import BaseModel, Field

from newspaper_explorer.config.base import get_config
from newspaper_explorer.data.preprocessing.pipeline import (
    STEP_CONFIG,
    TextPreprocessor,
)
from newspaper_explorer.data.preprocessing.presets import (
    ANALYSIS_PIPELINES,
    GENERAL_PIPELINES,
)
from newspaper_explorer.data.utils.sources import load_source_config

router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================


class StepParameter(BaseModel):
    """A configurable parameter for a preprocessing step."""

    name: str
    type: str  # "int", "float", "bool", "string", "select"
    default: Any
    description: str
    options: list[str] | None = None  # For select type
    min_value: float | None = None
    max_value: float | None = None


class PreprocessingStepInfo(BaseModel):
    """Information about a preprocessing step for the UI."""

    name: str
    display_name: str
    description: str
    category: str  # normalization, modernization, cleaning, filtering, linguistic, quality
    is_filter: bool  # Does it filter rows vs transform text?
    is_slow: bool  # Requires GPU or API (transnormer, dta-cab, lemmatize)
    parameters: list[StepParameter]


class PresetInfo(BaseModel):
    """Information about a preprocessing preset."""

    name: str
    description: str
    use_case: str
    steps: list[dict[str, Any]]  # Step configs with optional args
    category: str  # "general" or "analysis"


class PipelineStep(BaseModel):
    """A step in the pipeline with optional arguments."""

    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class PreprocessingPreviewRequest(BaseModel):
    """Request to preview a preprocessing pipeline on sample text."""

    text: str
    steps: list[PipelineStep]
    show_intermediate: bool = False


class BatchPreviewRequest(BaseModel):
    """Request to preview a preprocessing pipeline on multiple text samples."""

    texts: list[dict[str, Any]]  # List of text row dicts with metadata
    steps: list[PipelineStep]
    show_intermediate: bool = False


class StepResult(BaseModel):
    """Result of a single preprocessing step."""

    step_name: str
    output: str
    changes_description: str


class BatchStepResult(BaseModel):
    """Result of a batch step showing filtering effects."""

    step_name: str
    input_count: int
    output_count: int
    removed_count: int
    is_filter: bool


class TextSample(BaseModel):
    """A text sample with metadata."""

    text: str
    date: str | None = None
    page_number: int | None = None
    filtered: bool = False


class PreprocessingPreviewResponse(BaseModel):
    """Response from preprocessing preview."""

    original: str
    final: str
    intermediate_steps: list[StepResult]
    stats: dict[str, Any]


class BatchPreviewResponse(BaseModel):
    """Response from batch preprocessing preview."""

    original_samples: list[TextSample]
    processed_samples: list[TextSample]
    step_stats: list[BatchStepResult]
    total_removed: int
    total_remaining: int


class PreprocessingRunRequest(BaseModel):
    """Request to run preprocessing on a full dataset."""

    steps: list[PipelineStep]
    output_name: str | None = None
    text_column: str = "text"
    output_column: str = "text_processed"
    include_original: bool = True
    save_as_preset: str | None = None
    input_path: str | None = None  # Optional path to preprocessed dataset


class PreprocessingRunResponse(BaseModel):
    """Response from starting a preprocessing job."""

    job_id: str
    estimated_time_seconds: int
    output_path: str
    message: str


class PreprocessingStatusResponse(BaseModel):
    """Status of a running or completed preprocessing job."""

    job_id: str
    status: str  # "pending", "running", "completed", "failed"
    progress: float  # 0.0 - 1.0
    current_step: str | None = None
    error: str | None = None
    output_path: str | None = None


# =============================================================================
# Step Metadata Registry
# =============================================================================

# Maps step names to UI-friendly metadata
STEP_UI_METADATA: dict[str, dict[str, Any]] = {
    # === Normalization ===
    "normalize_unicode": {
        "display_name": "Unicode Normalization",
        "description": "Normalize Unicode (NFC), fix mojibake, quotes, ligatures, control characters. RECOMMENDED FIRST STEP.",
        "category": "normalization",
        "is_slow": False,
        "parameters": [],
    },
    "normalize_hyphens": {
        "display_name": "Normalize Hyphens",
        "description": "Normalize hyphen variants (double hyphen, en dash, em dash) to regular hyphen. Run BEFORE dehyphenation.",
        "category": "normalization",
        "is_slow": False,
        "parameters": [
            {
                "name": "mode",
                "type": "select",
                "default": "unify",
                "description": "Normalization mode",
                "options": ["unify", "conservative", "soft_only"],
            }
        ],
    },
    "normalize_whitespace": {
        "display_name": "Normalize Whitespace",
        "description": "Collapse multiple spaces, tabs, newlines to single space.",
        "category": "normalization",
        "is_slow": False,
        "parameters": [
            {
                "name": "keep_newlines",
                "type": "bool",
                "default": False,
                "description": "Preserve line breaks",
            }
        ],
    },
    "normalize_casing": {
        "display_name": "Normalize Casing",
        "description": "Convert text to lowercase, uppercase, or title case.",
        "category": "normalization",
        "is_slow": False,
        "parameters": [
            {
                "name": "mode",
                "type": "select",
                "default": "lower",
                "description": "Casing mode",
                "options": ["lower", "upper", "title"],
            }
        ],
    },
    "normalize_umlauts": {
        "display_name": "Normalize Umlauts",
        "description": "Normalize historical umlaut spellings (ae->ae, oe->oe, ue->ue).",
        "category": "normalization",
        "is_slow": False,
        "parameters": [],
    },
    "normalize_long_s": {
        "display_name": "Long S to Modern S",
        "description": "Convert historical long s (ſ) to modern s.",
        "category": "normalization",
        "is_slow": False,
        "parameters": [
            {
                "name": "mode",
                "type": "select",
                "default": "simple",
                "description": "Normalization mode",
                "options": ["simple", "context-aware"],
            }
        ],
    },
    "dehyphenate": {
        "display_name": "Dehyphenate",
        "description": "Remove line-break hyphens, rejoining split words. Auto-detects line/block level.",
        "category": "normalization",
        "is_slow": False,
        "parameters": [],
    },
    # === Modernization ===
    "modernization_transnormer": {
        "display_name": "Transnormer (Neural)",
        "description": "Neural historical German spelling normalization. GPU recommended.",
        "category": "modernization",
        "is_slow": True,
        "parameters": [
            {
                "name": "model",
                "type": "select",
                "default": "19c",
                "description": "Transnormer model",
                "options": ["19c", "18-19c"],
            },
            {
                "name": "batch_size",
                "type": "int",
                "default": 32,
                "description": "Batch size for inference",
                "min_value": 1,
                "max_value": 256,
            },
            {
                "name": "num_beams",
                "type": "int",
                "default": 4,
                "description": "Beam search width",
                "min_value": 1,
                "max_value": 10,
            },
        ],
    },
    "modernization_dta-cab": {
        "display_name": "DTA-CAB (API)",
        "description": "Historical German normalization via DTA-CAB API. Requires internet.",
        "category": "modernization",
        "is_slow": True,
        "parameters": [
            {
                "name": "batch_size",
                "type": "int",
                "default": 100,
                "description": "Batch size for API calls",
                "min_value": 1,
                "max_value": 500,
            },
            {
                "name": "timeout",
                "type": "int",
                "default": 30,
                "description": "API timeout in seconds",
                "min_value": 5,
                "max_value": 120,
            },
        ],
    },
    # === Cleaning ===
    "remove_diacritics": {
        "display_name": "Remove Diacritics",
        "description": "Remove accents and diacritics (ae->a, oe->o, ue->u).",
        "category": "cleaning",
        "is_slow": False,
        "parameters": [],
    },
    "remove_punctuation": {
        "display_name": "Remove Punctuation",
        "description": "Remove all punctuation marks.",
        "category": "cleaning",
        "is_slow": False,
        "parameters": [],
    },
    "remove_numbers": {
        "display_name": "Remove Numbers",
        "description": "Remove all numeric digits.",
        "category": "cleaning",
        "is_slow": False,
        "parameters": [],
    },
    "remove_stopwords": {
        "display_name": "Remove Stopwords",
        "description": "Remove common German stopwords.",
        "category": "cleaning",
        "is_slow": False,
        "parameters": [
            {
                "name": "language",
                "type": "select",
                "default": "german",
                "description": "Stopword language",
                "options": ["german", "english"],
            }
        ],
    },
    "only_keep_allowed_chars": {
        "display_name": "Remove Invalid Characters",
        "description": "Remove OCR artifacts and invalid characters, keeping only allowed characters.",
        "category": "cleaning",
        "is_slow": False,
        "parameters": [],
    },
    # === Filtering ===
    "filter_by_total_character_length": {
        "display_name": "Filter by Character Length",
        "description": "Remove lines with too few or too many characters.",
        "category": "filtering",
        "is_slow": False,
        "parameters": [
            {
                "name": "min_chars",
                "type": "int",
                "default": 0,
                "description": "Minimum character count",
                "min_value": 0,
            },
            {
                "name": "max_chars",
                "type": "int",
                "default": None,
                "description": "Maximum character count (empty = no limit)",
            },
        ],
    },
    "filter_by_word_count": {
        "display_name": "Filter by Word Count",
        "description": "Remove lines with too few or too many words.",
        "category": "filtering",
        "is_slow": False,
        "parameters": [
            {
                "name": "min_words",
                "type": "int",
                "default": 0,
                "description": "Minimum word count",
                "min_value": 0,
            },
            {
                "name": "max_words",
                "type": "int",
                "default": None,
                "description": "Maximum word count (empty = no limit)",
            },
        ],
    },
    "remove_garbage_words": {
        "display_name": "Remove Garbage Words",
        "description": "Remove words with excessive character repetition (OCR garbage like 'ssss', 'jjjj').",
        "category": "cleaning",
        "is_slow": False,
        "parameters": [
            {
                "name": "min_unique_chars",
                "type": "int",
                "default": 3,
                "description": "Minimum unique characters per word",
                "min_value": 1,
                "max_value": 10,
            },
            {
                "name": "max_repetition_ratio",
                "type": "float",
                "default": 0.3,
                "description": "Maximum ratio of unique to total characters",
                "min_value": 0.1,
                "max_value": 1.0,
            },
            {
                "name": "min_word_length",
                "type": "int",
                "default": 2,
                "description": "Minimum word length to check for garbage",
                "min_value": 1,
                "max_value": 10,
            },
        ],
    },
    "filter_number_only_lines": {
        "display_name": "Filter Number-Only Lines",
        "description": "Remove lines containing only numbers and separators.",
        "category": "filtering",
        "is_slow": False,
        "parameters": [],
    },
    "remove_long_words": {
        "display_name": "Remove Long Words",
        "description": "Remove excessively long words (likely OCR merge errors). Empty lines are handled by filter_empty_lines.",
        "category": "cleaning",
        "is_slow": False,
        "parameters": [
            {
                "name": "max_word_length",
                "type": "int",
                "default": 45,
                "description": "Maximum word length (German compounds rarely exceed 45)",
                "min_value": 20,
                "max_value": 100,
            },
        ],
    },
    "filter_empty_lines": {
        "display_name": "Filter Empty Lines",
        "description": "Remove empty or whitespace-only lines.",
        "category": "filtering",
        "is_slow": False,
        "parameters": [],
    },
    # === Linguistic ===
    "lemmatize_spacy": {
        "display_name": "Lemmatize (spaCy)",
        "description": "Fast lemmatization using spaCy German model.",
        "category": "linguistic",
        "is_slow": True,
        "parameters": [
            {
                "name": "model",
                "type": "select",
                "default": "de_core_news_sm",
                "description": "spaCy model",
                "options": ["de_core_news_sm", "de_core_news_md", "de_core_news_lg"],
            }
        ],
    },
    "lemmatize_germalemma": {
        "display_name": "Lemmatize (GermaLemma)",
        "description": "Thorough lemmatization using GermaLemma. Slower but more accurate.",
        "category": "linguistic",
        "is_slow": True,
        "parameters": [],
    },
    # === Quality ===
    "calculate_quality_metrics": {
        "display_name": "Calculate Quality Metrics",
        "description": "Calculate OCR quality metrics (char/token ratio, OOV rate, etc.).",
        "category": "quality",
        "is_slow": False,
        "parameters": [],
    },
    "filter_by_quality_score": {
        "display_name": "Filter by Quality Score",
        "description": "Filter rows by OCR quality score (good/review/poor).",
        "category": "quality",
        "is_slow": False,
        "parameters": [],
    },
}


# =============================================================================
# Helper Functions
# =============================================================================


def _describe_changes(original: str, processed: str) -> str:
    """Generate a human-readable description of changes between two texts."""
    char_diff = len(processed) - len(original)
    word_diff = len(processed.split()) - len(original.split())

    parts: list[str] = []
    if char_diff != 0:
        parts.append(f"{char_diff:+d} chars")
    if word_diff != 0:
        parts.append(f"{word_diff:+d} words")

    if not parts:
        if original == processed:
            return "No changes"
        return "Content modified"

    return ", ".join(parts)


def _apply_single_step(text: str, step: PipelineStep) -> str:
    """Apply a single preprocessing step to text."""
    df = pl.DataFrame({"text": [text]})
    preprocessor = TextPreprocessor(text_column="text")

    step_dict: dict[str, Any] = {"name": step.name}
    if step.args:
        step_dict["args"] = step.args

    df_result = preprocessor.pipeline(
        df,
        steps=[step_dict],
        output_column="text_out",
    )

    return str(df_result["text_out"][0])


def _normalize_steps(steps: Any) -> list[dict[str, Any]]:
    """Convert steps to normalized dict format."""
    if not isinstance(steps, list):
        return []

    result: list[dict[str, Any]] = []
    for step in steps:
        if isinstance(step, str):
            result.append({"name": step, "args": {}})
        elif isinstance(step, dict):
            result.append({"name": step.get("name", ""), "args": step.get("args", {})})
    return result


# =============================================================================
# API Endpoints
# =============================================================================


@router.get("/steps", response_model=list[PreprocessingStepInfo])
async def get_preprocessing_steps() -> list[PreprocessingStepInfo]:
    """
    Get all available preprocessing steps with descriptions and parameters.

    Returns a list of steps organized by category, with parameter information
    for building the UI configuration forms.
    """
    steps: list[PreprocessingStepInfo] = []

    for step_name, config in STEP_CONFIG.items():
        ui_meta = STEP_UI_METADATA.get(step_name, {})

        # Build parameter list using list comprehension
        parameters = [
            StepParameter(
                name=param["name"],
                type=param["type"],
                default=param["default"],
                description=param.get("description", ""),
                options=param.get("options"),
                min_value=param.get("min_value"),
                max_value=param.get("max_value"),
            )
            for param in ui_meta.get("parameters", [])
        ]

        steps.append(
            PreprocessingStepInfo(
                name=step_name,
                display_name=ui_meta.get("display_name", step_name.replace("_", " ").title()),
                description=ui_meta.get("description", ""),
                category=ui_meta.get("category", "other"),
                is_filter=config.get("filter_only", False),
                is_slow=ui_meta.get("is_slow", False),
                parameters=parameters,
            )
        )

    return steps


@router.get("/presets", response_model=list[PresetInfo])
async def get_preprocessing_presets() -> list[PresetInfo]:
    """
    Get all available preprocessing pipeline presets.

    Returns predefined pipelines for common use cases, organized into
    general-purpose and analysis-specific categories.
    """
    presets: list[PresetInfo] = []

    for name, config in GENERAL_PIPELINES.items():
        presets.append(
            PresetInfo(
                name=name,
                description=str(config.get("description", "")),
                use_case=str(config.get("use_case", "")),
                steps=_normalize_steps(config.get("steps", [])),
                category="general",
            )
        )

    for name, config in ANALYSIS_PIPELINES.items():
        presets.append(
            PresetInfo(
                name=name,
                description=str(config.get("description", "")),
                use_case=str(config.get("use_case", "")),
                steps=_normalize_steps(config.get("steps", [])),
                category="analysis",
            )
        )

    return presets


@router.post("/preview", response_model=PreprocessingPreviewResponse)
async def preview_preprocessing(
    request: PreprocessingPreviewRequest,
) -> PreprocessingPreviewResponse:
    """
    Apply preprocessing pipeline to sample text and return step-by-step results.

    This endpoint is used for the live preview in the UI, allowing users to see
    the effect of each preprocessing step on their text before running on the
    full dataset.
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="Text is required")

    if not request.steps:
        return PreprocessingPreviewResponse(
            original=request.text,
            final=request.text,
            intermediate_steps=[],
            stats={
                "char_diff": 0,
                "original_length": len(request.text),
                "final_length": len(request.text),
            },
        )

    intermediate_results: list[StepResult] = []
    current_text = request.text

    for step in request.steps:
        # Validate step exists
        if step.name not in STEP_CONFIG:
            raise HTTPException(status_code=400, detail=f"Unknown step: {step.name}")

        # Apply step
        try:
            new_text = _apply_single_step(current_text, step)
        except (ValueError, KeyError, RuntimeError) as e:
            raise HTTPException(status_code=500, detail=f"Step {step.name} failed: {e!s}") from e

        if request.show_intermediate:
            intermediate_results.append(
                StepResult(
                    step_name=step.name,
                    output=new_text,
                    changes_description=_describe_changes(current_text, new_text),
                )
            )

        current_text = new_text

    return PreprocessingPreviewResponse(
        original=request.text,
        final=current_text,
        intermediate_steps=intermediate_results,
        stats={
            "char_diff": len(current_text) - len(request.text),
            "original_length": len(request.text),
            "final_length": len(current_text),
            "word_diff": len(current_text.split()) - len(request.text.split()),
        },
    )


@router.post("/preview-batch", response_model=BatchPreviewResponse)
async def preview_preprocessing_batch(
    request: BatchPreviewRequest,
) -> BatchPreviewResponse:
    """
    Apply preprocessing pipeline to multiple text samples and return filter statistics.

    This endpoint is used to preview how filters affect a batch of samples,
    showing how many rows would be removed by each filter step.
    """
    if not request.texts:
        raise HTTPException(status_code=400, detail="Texts are required")

    if not request.steps:
        original_samples = [
            TextSample(
                text=t.get("text", ""),
                date=t.get("date"),
                page_number=t.get("page_number"),
            )
            for t in request.texts
        ]
        return BatchPreviewResponse(
            original_samples=original_samples,
            processed_samples=original_samples,
            step_stats=[],
            total_removed=0,
            total_remaining=len(request.texts),
        )

    # Build original samples
    original_samples = [
        TextSample(
            text=t.get("text", ""),
            date=t.get("date"),
            page_number=t.get("page_number"),
        )
        for t in request.texts
    ]

    # Create DataFrame from texts with index to track which rows survive filtering
    df = pl.DataFrame(
        {
            "_original_idx": list(range(len(request.texts))),
            "text": [t.get("text", "") for t in request.texts],
        }
    )
    preprocessor = TextPreprocessor(text_column="text")

    step_stats: list[BatchStepResult] = []

    for step in request.steps:
        if step.name not in STEP_CONFIG:
            raise HTTPException(status_code=400, detail=f"Unknown step: {step.name}")

        input_count = len(df)
        step_config = STEP_CONFIG.get(step.name, {})
        is_filter = step_config.get("filter_only", False)

        try:
            step_dict: dict[str, Any] = {"name": step.name}
            if step.args:
                step_dict["args"] = step.args

            df = preprocessor.pipeline(
                df,
                steps=[step_dict],
                output_column="text",
            )
        except (ValueError, KeyError, RuntimeError) as e:
            raise HTTPException(status_code=500, detail=f"Step {step.name} failed: {e!s}") from e

        output_count = len(df)
        removed_count = input_count - output_count

        step_stats.append(
            BatchStepResult(
                step_name=step.name,
                input_count=input_count,
                output_count=output_count,
                removed_count=removed_count,
                is_filter=is_filter,
            )
        )

    # Build processed samples - use index to track which rows survived
    # Create a mapping from original index to processed text
    processed_df_rows = df.to_dicts()
    idx_to_processed: dict[int, str] = {
        row["_original_idx"]: row["text"] for row in processed_df_rows
    }

    # Build result maintaining original order
    processed_samples: list[TextSample] = []
    for i, sample in enumerate(original_samples):
        if i in idx_to_processed:
            # Row survived - show processed text
            sample.filtered = False
            processed_samples.append(TextSample(text=idx_to_processed[i], filtered=False))
        else:
            # Row was filtered out
            sample.filtered = True
            processed_samples.append(TextSample(text="", filtered=True))

    total_removed = len(request.texts) - len(df)

    return BatchPreviewResponse(
        original_samples=original_samples,
        processed_samples=processed_samples,
        step_stats=step_stats,
        total_removed=total_removed,
        total_remaining=len(df),
    )


@router.get("/{source_name}/sample")
async def get_sample_text(
    source_name: str,
    count: int = Query(default=5, ge=1, le=50, description="Number of samples to return"),
    sample_type: str = Query(
        default="random",
        description="Sample type: 'random' or 'hyphenated' (for dehyphenation testing)",
    ),
) -> list[dict[str, Any]]:
    """
    Get random sample text lines from a source for preview.

    Returns a list of random text rows with metadata that can be used to test
    preprocessing pipelines.

    Args:
        source_name: The source to sample from
        count: Number of samples to return
        sample_type: 'random' for random samples, 'hyphenated' for lines ending in hyphens
                     (useful for testing dehyphenation)
    """
    config = get_config()

    try:
        source_config = load_source_config(source_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Source not found: {source_name}") from exc

    # Try textblocks first, fall back to lines
    text_path = (
        config.data_dir / "processed" / source_config.dataset_name / "text" / "textblocks.parquet"
    )
    if not text_path.exists():
        text_path = (
            config.data_dir
            / "raw"
            / source_config.dataset_name
            / "text"
            / f"{source_config.dataset_name}_lines.parquet"
        )

    if not text_path.exists():
        raise HTTPException(status_code=404, detail=f"No text data found for {source_name}")

    df = pl.read_parquet(text_path)

    # If we need hyphenated lines for dehyphenation preview
    if sample_type == "hyphenated":
        # Find lines ending with hyphen followed by a word continuation
        hyphenated = df.filter(pl.col("text").str.ends_with("-"))
        if len(hyphenated) > 0:
            # Get a sample of hyphenated lines
            sample_size = min(count, len(hyphenated))
            df = hyphenated.sample(n=sample_size)

    # Sample random rows
    if len(df) > count:
        df = df.sample(n=count)

    # Build result with metadata
    result: list[dict[str, Any]] = []
    for row in df.to_dicts():
        sample: dict[str, Any] = {
            "text": row.get("text", ""),
            "date": str(row.get("date", "")) if row.get("date") else None,
            "page_number": row.get("page_number"),
        }
        result.append(sample)

    return result


# Background job storage (in production, use Redis or database)
_preprocessing_jobs: dict[str, dict[str, Any]] = {}


@router.post("/{source_name}/run", response_model=PreprocessingRunResponse)
async def run_preprocessing_pipeline(
    source_name: str,
    request: PreprocessingRunRequest,
    background_tasks: BackgroundTasks,
) -> PreprocessingRunResponse:
    """
    Start preprocessing pipeline as a background task.

    This endpoint initiates preprocessing on the full dataset. Progress can be
    tracked via the /status endpoint.
    """
    # Validate source exists
    try:
        source_config = load_source_config(source_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Source not found: {source_name}") from exc

    # Generate job ID
    job_id = str(uuid.uuid4())[:8]

    # Convert steps to pipeline format
    steps: list[dict[str, Any]] = [{"name": s.name, "args": s.args} for s in request.steps]

    # Estimate time (rough: 1ms per line for fast steps, 10ms for slow)
    config = get_config()
    text_path = (
        config.data_dir / "processed" / source_config.dataset_name / "text" / "textblocks.parquet"
    )
    if text_path.exists():
        df = pl.read_parquet(text_path, columns=["text"])
        row_count = len(df)
    else:
        row_count = 100000  # Default estimate

    # Check for slow steps
    has_slow_steps = any(
        STEP_UI_METADATA.get(s.name, {}).get("is_slow", False) for s in request.steps
    )
    ms_per_row = 10 if has_slow_steps else 1
    estimated_seconds = int(row_count * ms_per_row / 1000)

    # Initialize job status
    _preprocessing_jobs[job_id] = {
        "status": "pending",
        "progress": 0.0,
        "current_step": None,
        "error": None,
        "output_path": None,
    }

    # Resolve input path if provided
    resolved_input_path = None
    if request.input_path:
        resolved_input_path = Path(request.input_path)
        if not resolved_input_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Input file not found: {request.input_path}"
            )

    # Add background task
    background_tasks.add_task(
        _run_preprocessing_task,
        job_id=job_id,
        source_name=source_name,
        steps=steps,
        text_column=request.text_column,
        output_column=request.output_column,
        input_path=resolved_input_path,
    )

    return PreprocessingRunResponse(
        job_id=job_id,
        estimated_time_seconds=estimated_seconds,
        output_path=f"data/processed/{source_config.dataset_name}/text/preprocessed_{job_id}.parquet",
        message=f"Preprocessing started for {row_count:,} rows",
    )


async def _run_preprocessing_task(
    job_id: str,
    source_name: str,
    steps: list[dict[str, Any]],
    text_column: str,
    output_column: str,
    input_path: Path | None = None,
) -> None:
    """Background task to run preprocessing."""
    _preprocessing_jobs[job_id]["status"] = "running"

    try:
        preprocessor = TextPreprocessor(source=source_name, text_column=text_column)
        result = preprocessor.run(
            steps=steps,
            input_path=input_path,
            output_column=output_column,
            save=True,
        )

        _preprocessing_jobs[job_id].update(
            {
                "status": "completed",
                "progress": 1.0,
                "output_path": str(result.results_path),
            }
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        _preprocessing_jobs[job_id].update(
            {
                "status": "failed",
                "error": str(e),
            }
        )


@router.get("/{source_name}/status/{job_id}", response_model=PreprocessingStatusResponse)
async def get_preprocessing_status(
    source_name: str,  # noqa: ARG001 - kept for URL structure
    job_id: str,
) -> PreprocessingStatusResponse:
    """
    Get status of a running or completed preprocessing job.
    """
    if job_id not in _preprocessing_jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job = _preprocessing_jobs[job_id]

    return PreprocessingStatusResponse(
        job_id=job_id,
        status=str(job.get("status", "unknown")),
        progress=float(job.get("progress", 0.0)),
        current_step=job.get("current_step"),
        error=job.get("error"),
        output_path=job.get("output_path"),
    )


class PreprocessedDatasetInfo(BaseModel):
    """Information about a preprocessed dataset."""

    name: str
    path: str
    created: str
    steps: int
    row_count: int | None = None


@router.get("/{source_name}/datasets", response_model=list[PreprocessedDatasetInfo])
async def list_preprocessed_datasets(source_name: str) -> list[PreprocessedDatasetInfo]:
    """
    List all preprocessed datasets for a source.

    Returns parquet files in the processed/{source}/text/ directory that
    were created by preprocessing pipelines.
    """
    config = get_config()

    try:
        source_config = load_source_config(source_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Source not found: {source_name}") from exc

    processed_path = config.data_dir / "processed" / source_config.dataset_name / "text"

    if not processed_path.exists():
        return []

    datasets: list[PreprocessedDatasetInfo] = []

    # Look for preprocessed parquet files (preprocessed_*.parquet pattern)
    for parquet_file in processed_path.glob("preprocessed_*.parquet"):
        # Get file stats
        stat = parquet_file.stat()
        created = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")

        # Try to get row count and step count from metadata or file
        row_count = None
        steps = 0

        try:
            df = pl.read_parquet(parquet_file, n_rows=0)
            # Estimate steps from column names or just use placeholder
            steps = len([c for c in df.columns if c.startswith("text_")]) or 1
        except (FileNotFoundError, pl.exceptions.ComputeError):
            pass

        datasets.append(
            PreprocessedDatasetInfo(
                name=parquet_file.stem,
                path=str(parquet_file.relative_to(config.data_dir)),
                created=created,
                steps=steps,
                row_count=row_count,
            )
        )

    # Sort by creation date, newest first
    datasets.sort(key=lambda x: x.created, reverse=True)

    return datasets
