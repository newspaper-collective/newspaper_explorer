"""Text preprocessing pipeline for newspaper data.

Provides a TextPreprocessor class that chains together preprocessing steps
from the modular preprocessing modules.

For direct use of preprocessing functions, import from individual modules:
    from newspaper_explorer.data.preprocessing.normalization import normalize_unicode, dehyphenate
    from newspaper_explorer.data.preprocessing.modernization import transnormer, dta_cab
    from newspaper_explorer.data.preprocessing.cleaning import remove_stopwords, remove_punctuation
    from newspaper_explorer.data.preprocessing.filtering import filter_by_total_character_length
    from newspaper_explorer.data.preprocessing.lemmatization import lemmatize_spacy, lemmatize_germalemma

For high-level preprocessing with automatic input/output handling:
    from newspaper_explorer.data.preprocessing.pipeline import TextPreprocessor
    preprocessor = TextPreprocessor(source="der_tag")
    result = preprocessor.run(steps=["normalize_unicode", "normalize_casing"])
"""

from datetime import datetime
import logging
from pathlib import Path
import time
from typing import Any, Optional, Union

import polars as pl

from newspaper_explorer.config.base import get_config
from newspaper_explorer.data.preprocessing.cleaning import (
    only_keep_allowed_chars,
    remove_diacritics,
    remove_garbage_words,
    remove_long_words,
    remove_numbers,
    remove_punctuation,
    remove_short_words,
    remove_stopwords,
)
from newspaper_explorer.data.preprocessing.filtering import (
    filter_by_total_character_length,
    filter_by_word_count,
    filter_empty_lines,
    filter_lines_without_alphabetic_chars,
    filter_number_only_lines,
)
from newspaper_explorer.data.preprocessing.lemmatization import (
    lemmatize_germalemma,
    lemmatize_spacy,
)
from newspaper_explorer.data.preprocessing.modernization import (
    dta_cab,
    transnormer,
)
from newspaper_explorer.data.preprocessing.normalization import (
    dehyphenate,
    normalize_casing,
    normalize_hyphens,
    normalize_long_s,
    normalize_umlauts,
    normalize_unicode,
    normalize_whitespace,
)
from newspaper_explorer.data.preprocessing.presets import (
    StepType,
    get_step_args,
    get_step_name,
)
from newspaper_explorer.data.preprocessing.validation import (
    calculate_quality_metrics,
    filter_by_quality_score,
)
from newspaper_explorer.data.utils.metadata import (
    find_metadata_for_parquet,
    load_metadata,
)
from newspaper_explorer.data.utils.results import save_preprocessing_results
from newspaper_explorer.data.utils.sources import load_source_config
from newspaper_explorer.models.data.metadata import PreprocessingMetadata, PreprocessingResult

logger = logging.getLogger(__name__)


# Step configuration: maps step names to their processing function and behavior
# - "func": The preprocessing function to call
# - "filter_only": If True, step filters rows but doesn't transform text (no output column)
# - "extra_args": Additional fixed arguments to pass
# - "special": If set, requires special handling (e.g., "dehyphenate", "transnormer")
STEP_CONFIG: dict[str, dict[str, Any]] = {
    # === Normalization (normalization.py) ===
    "normalize_unicode": {"func": normalize_unicode},
    "normalize_hyphens": {"func": normalize_hyphens},
    "normalize_whitespace": {"func": normalize_whitespace},
    "normalize_casing": {"func": normalize_casing},
    "normalize_umlauts": {"func": normalize_umlauts},
    "normalize_long_s": {"func": normalize_long_s, "extra_args": {"mode": "simple"}},
    "dehyphenate": {"func": dehyphenate},
    # === Modernization (modernization.py) ===
    "modernization_transnormer": {"func": transnormer, "special": "transnormer"},
    "modernization_dta-cab": {"func": dta_cab},
    # === Cleaning (cleaning.py) ===
    "remove_diacritics": {"func": remove_diacritics},
    "remove_punctuation": {"func": remove_punctuation},
    "remove_numbers": {"func": remove_numbers},
    "remove_stopwords": {"func": remove_stopwords},
    "remove_long_words": {"func": remove_long_words},
    "remove_short_words": {"func": remove_short_words},
    "remove_garbage_words": {"func": remove_garbage_words},
    "only_keep_allowed_chars": {"func": only_keep_allowed_chars},
    # === Lemmatization (lemmatization.py) ===
    "lemmatize_spacy": {"func": lemmatize_spacy},
    "lemmatize_germalemma": {"func": lemmatize_germalemma},
    # === Filtering (filtering.py) ===
    "filter_by_total_character_length": {
        "func": filter_by_total_character_length,
        "filter_only": True,
    },
    "filter_by_word_count": {"func": filter_by_word_count, "filter_only": True},
    "filter_number_only_lines": {"func": filter_number_only_lines, "filter_only": True},
    "filter_lines_without_alphabetic_chars": {
        "func": filter_lines_without_alphabetic_chars,
        "filter_only": True,
    },
    "filter_empty_lines": {"func": filter_empty_lines, "filter_only": True},
    # === Validation (validation.py) ===
    "calculate_quality_metrics": {"func": calculate_quality_metrics, "filter_only": True},
    "filter_by_quality_score": {
        "func": filter_by_quality_score,
        "filter_only": True,
        "no_args": True,
    },
}


class TextPreprocessor:
    """
    Chains together multiple text preprocessing steps in sequence.

    This class provides:
    - pipeline(): Apply preprocessing steps to a DataFrame (pure transformation)
    - run(): Full preprocessing workflow with I/O (load, process, save) - requires source

    Example (full workflow with I/O):
        >>> preprocessor = TextPreprocessor(source="der_tag")
        >>> result = preprocessor.run(
        ...     steps=["normalize_unicode", "normalize_casing", "remove_stopwords"]
        ... )
        >>> print(f"Output: {result.results_path}")

    Example (pure transformation without I/O):
        >>> preprocessor = TextPreprocessor()
        >>> df = preprocessor.pipeline(
        ...     df,
        ...     steps=["normalize_unicode", "normalize_casing"],
        ...     output_column="text_processed"
        ... )

    Note:
        All original DataFrame columns (including IDs and metadata) are preserved
        in the output, so you can still trace processed text back to its source.

        For more control, you can import and use preprocessing functions directly:
        from newspaper_explorer.data.preprocessing.cleaning import remove_stopwords
    """

    def __init__(
        self,
        source: Optional[str] = None,
        *,
        text_column: str = "text",
    ) -> None:
        """
        Initialize preprocessor.

        Args:
            source: Source dataset name (e.g., "der_tag"). Required for run(), optional for pipeline().
            text_column: Name of the column containing text to process
        """
        self.source = source
        self.text_column = text_column
        self._config = get_config()
        # Only load source config if source is provided (needed for run(), not pipeline())
        self._source_config = load_source_config(source) if source else None
        logger.debug(f"TextPreprocessor initialized for source: {source}, column: {text_column}")

    def pipeline(
        self,
        df: pl.DataFrame,
        steps: list[StepType],
        output_column: str = "text_processed",
        batch_size: int = 32,
        num_beams: int = 4,
        num_gpus: int = 1,
        *,
        use_cache: bool = True,
    ) -> pl.DataFrame:
        """
        Apply a pipeline of preprocessing steps.

        Steps can be either:
        - A string: step name (e.g., "normalize_unicode")
        - A dict: {"name": "step_name", "args": {"param": value}}

        Available steps:
        - normalize_unicode: Unicode normalization (NFC + ftfy + character translation)
        - normalize_whitespace: Normalize whitespace
        - normalize_casing: Convert to lowercase
        - normalize_umlauts: Normalize historical umlauts (ae->ae, oe->oe, ue->ue)
        - normalize_long_s: Normalize long s to modern s (simple/context-aware/preserve)
        - dehyphenate: Remove line-break hyphens (auto-detects line/block level)
        - transnormer: Neural normalization with Transnormer
        - dta_cab: API normalization with DTA-CAB
        - remove_diacritics: Remove diacritics
        - remove_punctuation: Remove punctuation
        - remove_numbers: Remove numbers
                - remove_stopwords: Remove German/English stopwords
        - remove_long_words: Remove excessively long words (merged OCR errors)
        - only_keep_allowed_chars: Remove OCR artifacts and invalid characters
        - lemmatize_spacy: Lemmatize with spaCy (fast)
        - lemmatize_germalemma: Lemmatize with GermaLemma (slow)
        - filter_by_total_character_length: Filter out too short/long texts (by character count)
        - filter_by_word_count: Filter out too few/many words (by word count)
        - remove_garbage_words: Remove words with excessive character repetition (OCR garbage)
        - filter_number_only_lines: Remove lines containing only numbers and separators
        - filter_lines_without_alphabetic_chars: Remove lines without any alphabetic characters
        - filter_empty_lines: Remove empty lines
        - calculate_quality_metrics: Calculate OCR quality metrics (char/token ratio, OOV rate, etc.)
        - filter_by_quality_score: Filter rows by quality score (good/review/poor)

        Args:
            df: Input DataFrame (must contain text_column)
            steps: List of step names to apply in order
            output_column: Name for final output column
            batch_size: Batch size for transnormer inference (default: 32)
            num_beams: Number of beams for transnormer generation (default: 4)
            num_gpus: Number of GPUs for parallel transnormer processing (default: 1)

        Returns:
            DataFrame with processed text and all original columns preserved.
            Foreign keys (source_id, issue_id, page_id, text_block_id, line_id)
            are automatically preserved for traceability.

        Note:
            All columns from the input DataFrame are preserved in the output,
            ensuring that foreign keys and metadata remain available after
            preprocessing for linking back to source data.

            Recommended first step: 'normalize_unicode' to handle OCR artifacts,
            unify quotes/hyphens, and remove control characters.
        """
        logger.info(f"Starting preprocessing pipeline with {len(steps)} steps")
        step_names = [get_step_name(s) for s in steps]
        logger.info(f"Steps: {', '.join(step_names)}")

        current_column = self.text_column

        for i, step in enumerate(steps, 1):
            step_name = get_step_name(step)
            step_args = get_step_args(step)
            is_last_step = i == len(steps)
            logger.info(
                f"Step {i}/{len(steps)}: {step_name}" + (f" {step_args}" if step_args else "")
            )
            out_col = output_column if is_last_step else f"_tmp_{i}"

            df, current_column = self._apply_step(
                df=df,
                step=step,
                current_column=current_column,
                out_col=out_col,
                batch_size=batch_size,
                num_beams=num_beams,
                num_gpus=num_gpus,
                use_cache=use_cache,
            )

        # If the last step was filter-only, current_column still points to a temp column
        # We need to rename it to the output column before cleanup
        if current_column != output_column and current_column in df.columns:
            df = df.rename({current_column: output_column})
            current_column = output_column

        # Clean up temporary columns
        temp_cols = [col for col in df.columns if col.startswith("_tmp_")]
        if temp_cols:
            df = df.drop(temp_cols)

        logger.info(f"Preprocessing pipeline completed: {len(df)} rows")
        return df

    def _apply_step(
        self,
        df: pl.DataFrame,
        step: StepType,
        current_column: str,
        out_col: str,
        batch_size: int,
        num_beams: int,
        num_gpus: int,
        *,
        use_cache: bool,
    ) -> tuple[pl.DataFrame, str]:
        """
        Apply a single preprocessing step.

        Args:
            step: Either a string step name or dict with 'name' and 'args' keys

        Returns:
            Tuple of (processed DataFrame, new current column name)
        """
        step_name = get_step_name(step)
        step_args = get_step_args(step)

        if step_name not in STEP_CONFIG:
            raise ValueError(f"Unknown preprocessing step: {step_name}")

        config = STEP_CONFIG[step_name]

        # Handle transnormer special case (needs extra GPU parameters)
        if config.get("special") == "transnormer":
            df = config["func"](
                df,
                input_column=current_column,
                output_column=out_col,
                batch_size=batch_size,
                num_beams=num_beams,
                num_gpus=num_gpus,
                use_cache=use_cache,
            )
            return df, out_col

        # Build kwargs for the function call
        func = config["func"]
        kwargs: dict[str, Any] = {"df": df}

        # Add column arguments based on config
        if not config.get("no_args"):
            # Standard functions with input_column
            kwargs["input_column"] = current_column

            # Add output_column for non-filter steps
            if not config.get("filter_only"):
                kwargs["output_column"] = out_col

        # Add any extra fixed arguments from STEP_CONFIG
        if "extra_args" in config:
            kwargs.update(config["extra_args"])

        # Add/override with step-specific args from the preset
        kwargs.update(step_args)

        # Call the function
        df = func(**kwargs)

        # Filter-only steps don't change the current column
        new_column = current_column if config.get("filter_only") else out_col
        return df, new_column

    def load_previous_metadata(self, input_path: Union[Path, str]) -> Optional[dict[str, Any]]:
        """
        Load preprocessing metadata from input file if it exists.

        Args:
            input_path: Path to input parquet file

        Returns:
            Dictionary with previous preprocessing info, or None if not found
        """
        input_path = Path(input_path)
        metadata_path = find_metadata_for_parquet(input_path)

        if metadata_path:
            try:
                metadata = load_metadata(metadata_path)
                if isinstance(metadata, PreprocessingMetadata):
                    logger.info(
                        f"Found previous preprocessing: {metadata.preprocessing_id} "
                        f"({len(metadata.steps)} steps)"
                    )
                    return metadata.to_dict()
            except (FileNotFoundError, ValueError, OSError) as e:
                logger.debug(f"Could not load preprocessing metadata: {e}")

        return None

    def _create_metadata(
        self,
        steps: list[StepType],
        parameters: dict[str, Any],
        input_df: pl.DataFrame,
        output_df: pl.DataFrame,
        duration_seconds: float,
        previous_preprocessing: Optional[dict[str, Any]] = None,
        preprocessing_id: Optional[str] = None,
    ) -> PreprocessingMetadata:
        """
        Create preprocessing metadata for saving alongside results.

        Args:
            steps: List of preprocessing steps applied (strings or dicts)
            parameters: Processing parameters
            input_df: Input DataFrame (for statistics)
            output_df: Output DataFrame (for statistics)
            duration_seconds: Processing time
            previous_preprocessing: Previous preprocessing metadata if input was preprocessed

        Returns:
            PreprocessingMetadata object ready to save
        """
        # This method is only called from run(), which validates source is not None
        if self.source is None:
            raise ValueError("source must be set to create metadata")
        source: str = self.source  # Narrow type for type checker

        # Convert steps to serializable format (list of step names for metadata)
        step_names = [get_step_name(s) for s in steps]

        # Calculate input/output statistics
        input_data: dict[str, Union[int, list[str]]] = {
            "row_count": len(input_df),
            "columns": list(input_df.columns),
        }
        output_data: dict[str, Union[int, list[str]]] = {
            "row_count": len(output_df),
            "columns": list(output_df.columns),
        }

        return PreprocessingMetadata(
            source=source,
            steps=step_names,
            parameters=parameters,
            input_data=input_data,
            output_data=output_data,
            duration_seconds=duration_seconds,
            previous_preprocessing=previous_preprocessing,
            status="completed",
            completed_at=datetime.now().isoformat(),
            preprocessing_id=preprocessing_id,  # Use explicit or auto-generate
            error_message=None,  # Not needed for completed status
        )

    def run(
        self,
        steps: list[StepType],
        *,
        input_path: Optional[Path] = None,
        output_column: str = "text_processed",
        sample: Optional[int] = None,
        batch_size: int = 32,
        num_beams: int = 4,
        num_gpus: int = 1,
        use_cache: bool = True,
        save: bool = True,
        results_filename: str = "textblocks.parquet",
        preprocessing_id: Optional[str] = None,
    ) -> PreprocessingResult:
        """
        Run the full preprocessing workflow: load, process, and save.

        This is the main entry point for preprocessing operations. It handles:
        - Auto-detecting input path from source if not provided
        - Loading data and checking for previous preprocessing
        - Running the pipeline with timing
        - Creating metadata and saving results

        Args:
            steps: List of preprocessing steps (strings or dicts with args)
            input_path: Path to input parquet file. If None, auto-detects from source.
            output_column: Name for processed text column (default: "text_processed")
            sample: If provided, only process first N rows (for testing)
            batch_size: Batch size for transnormer inference (default: 32)
            num_beams: Number of beams for transnormer generation (default: 4)
            num_gpus: Number of GPUs for parallel transnormer processing (default: 1)
            use_cache: Enable caching for transnormer (default: True)
            save: Whether to save results to disk (default: True)
            results_filename: Filename for output parquet (default: "textblocks.parquet")

        Returns:
            PreprocessingResult with metadata, paths, and statistics

        Raises:
            FileNotFoundError: If input file not found
            ValueError: If source not provided, text column not in DataFrame, or invalid steps

        Example:
            >>> from newspaper_explorer.data.preprocessing.pipeline import TextPreprocessor
            >>> from newspaper_explorer.data.preprocessing.presets import get_preset
            >>> preprocessor = TextPreprocessor(source="der_tag")
            >>> result = preprocessor.run(steps=get_preset("standard"))
            >>> print(f"Processed {result.input_rows} → {result.output_rows} rows")
            >>> print(f"Output: {result.results_path}")
        """
        # Source is required for run() since we need to save results
        if self.source is None or self._source_config is None:
            raise ValueError(
                "Source is required for run(). "
                "Initialize with TextPreprocessor(source='your_source') or use pipeline() directly."
            )

        # Resolve input path
        if input_path is None:
            input_path = (
                self._config.parsed_dir
                / self._source_config.dataset_name
                / "textblocks.parquet"
            )

        if not input_path.exists():
            raise FileNotFoundError(
                f"Input file not found: {input_path}. "
                f"Run 'newspaper-explorer data aggregate --source {self.source}' first."
            )

        logger.info(f"Loading data from {input_path}")

        # Load data
        df = pl.read_parquet(input_path)
        input_df = df  # Keep reference for metadata
        input_rows = len(df)

        logger.info(f"Loaded {input_rows:,} rows")

        # Sample if requested
        if sample is not None and sample < len(df):
            logger.info(f"Sampling first {sample:,} rows for testing")
            df = df.head(sample)

        # Check text column exists
        if self.text_column not in df.columns:
            raise ValueError(
                f"Text column '{self.text_column}' not found. "
                f"Available columns: {', '.join(df.columns)}"
            )

        # Check for previous preprocessing
        previous_preprocessing = self.load_previous_metadata(input_path)

        if previous_preprocessing:
            prev_steps = previous_preprocessing.get("steps", [])
            logger.info(
                f"Found previous preprocessing with {len(prev_steps)} steps: {', '.join(prev_steps)}"
            )

        # Run preprocessing pipeline
        step_names = [get_step_name(s) for s in steps]
        logger.info(f"Starting preprocessing with {len(steps)} steps: {', '.join(step_names)}")
        processing_start = time.time()

        df = self.pipeline(
            df,
            steps=steps,
            output_column=output_column,
            batch_size=batch_size,
            num_beams=num_beams,
            num_gpus=num_gpus,
            use_cache=use_cache,
        )

        duration = time.time() - processing_start
        output_rows = len(df)

        logger.info(
            f"Preprocessing completed: {input_rows:,} → {output_rows:,} rows in {duration:.1f}s"
        )

        # Get sample for preview
        sample_original: Optional[str] = None
        sample_processed: Optional[str] = None
        if len(df) > 0:
            sample_row = df.head(1).to_dicts()[0]
            sample_original = str(sample_row.get(self.text_column, ""))[:500]
            sample_processed = str(sample_row.get(output_column, ""))[:500]

        # Create metadata
        parameters: dict[str, Any] = {
            "text_column": self.text_column,
            "output_column": output_column,
            "batch_size": batch_size,
            "num_beams": num_beams,
            "num_gpus": num_gpus,
            "use_cache": use_cache,
            "sample": sample,
        }

        metadata = self._create_metadata(
            steps=steps,
            parameters=parameters,
            input_df=input_df,
            output_df=df,
            duration_seconds=duration,
            previous_preprocessing=previous_preprocessing,
            preprocessing_id=preprocessing_id,
        )

        # Save results
        if save:
            paths = save_preprocessing_results(
                results_df=df,
                metadata=metadata,
                processed_base_dir=self._config.preprocessed_dir,
                results_filename=results_filename,
            )
            output_dir = paths["output_dir"]
            results_path = paths["results_path"]
            metadata_path = paths["metadata_path"]
            file_size_bytes = results_path.stat().st_size
        else:
            # Return placeholder paths when not saving
            output_dir = Path()
            results_path = Path()
            metadata_path = Path()
            file_size_bytes = 0

        return PreprocessingResult(
            metadata=metadata,
            output_dir=output_dir,
            results_path=results_path,
            metadata_path=metadata_path,
            input_rows=input_rows,
            output_rows=output_rows,
            duration_seconds=duration,
            file_size_bytes=file_size_bytes,
            sample_original=sample_original,
            sample_processed=sample_processed,
        )
