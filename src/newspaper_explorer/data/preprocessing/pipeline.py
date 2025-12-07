"""Text preprocessing pipeline for newspaper data.

Provides a simple pipeline function that chains together preprocessing steps
from the modular preprocessing modules.

For direct use of preprocessing functions, import from individual modules:
    from newspaper_explorer.data.preprocessing.normalization import normalize_unicode, dehyphenate
    from newspaper_explorer.data.preprocessing.modernization import transnormer, dta_cab
    from newspaper_explorer.data.preprocessing.cleaning import remove_stopwords, remove_punctuation
    from newspaper_explorer.data.preprocessing.filtering import filter_by_total_character_length
    from newspaper_explorer.data.preprocessing.lemmatization import lemmatize_spacy, lemmatize_germalemma
"""

from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Optional, Union

import polars as pl

from newspaper_explorer.data.preprocessing.cleaning import (
    only_keep_allowed_chars,
    remove_diacritics,
    remove_numbers,
    remove_punctuation,
    remove_stopwords,
)
from newspaper_explorer.data.preprocessing.filtering import (
    filter_by_char_token_ratio,
    filter_by_max_word_length,
    filter_by_total_character_length,
    filter_by_word_count,
    filter_number_only_lines,
    filter_repeating_chars,
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
    dehyphenate_auto,
    normalize_casing,
    normalize_long_s,
    normalize_umlauts,
    normalize_unicode,
    normalize_whitespace,
)
from newspaper_explorer.data.preprocessing.validation import (
    calculate_quality_metrics,
    filter_by_quality_score,
)
from newspaper_explorer.data.utils.metadata import (
    find_metadata_for_parquet,
    load_metadata,
)
from newspaper_explorer.models.data.metadata import PreprocessingMetadata

logger = logging.getLogger(__name__)


# Step configuration: maps step names to their processing function and behavior
# - "func": The preprocessing function to call
# - "filter_only": If True, step filters rows but doesn't transform text (no output column)
# - "extra_args": Additional fixed arguments to pass
# - "special": If set, requires special handling (e.g., "dehyphenate", "transnormer")
STEP_CONFIG: dict[str, dict[str, Any]] = {
    # === Normalization (normalization.py) ===
    "normalize-unicode": {"func": normalize_unicode},
    "normalize-whitespace": {"func": normalize_whitespace},
    "normalize-casing": {"func": normalize_casing},
    "normalize-umlauts": {"func": normalize_umlauts},
    "normalize-long-s": {"func": normalize_long_s, "extra_args": {"mode": "simple"}},
    "dehyphenate": {"func": dehyphenate_auto},
    # === Modernization (modernization.py) ===
    "modernize-transnormer": {"func": transnormer, "special": "transnormer"},
    "modernize-dtacab": {"func": dta_cab},
    # === Cleaning (cleaning.py) ===
    "remove-diacritics": {"func": remove_diacritics},
    "remove-punctuation": {"func": remove_punctuation},
    "remove-numbers": {"func": remove_numbers},
    "remove-stopwords": {"func": remove_stopwords},
    "filter-allowed-chars": {"func": only_keep_allowed_chars},
    # === Lemmatization (lemmatization.py) ===
    "lemmatize-spacy": {"func": lemmatize_spacy},
    "lemmatize-germalemma": {"func": lemmatize_germalemma},
    # === Filtering (filtering.py) ===
    "filter-length": {"func": filter_by_total_character_length, "filter_only": True},
    "filter-word-count": {"func": filter_by_word_count, "filter_only": True},
    "filter-repeating-chars": {"func": filter_repeating_chars},
    "filter-number-only": {"func": filter_number_only_lines, "filter_only": True},
    "filter-char-token-ratio": {"func": filter_by_char_token_ratio, "filter_only": True},
    "filter-word-length": {"func": filter_by_max_word_length, "filter_only": True},
    # === Validation (validation.py) ===
    "calculate-quality": {"func": calculate_quality_metrics, "filter_only": True},
    "filter-by-quality": {"func": filter_by_quality_score, "filter_only": True, "no_args": True},
}


class TextPreprocessor:
    """
    Chains together multiple text preprocessing steps in sequence.

    This class provides a convenient pipeline() method that applies preprocessing
    steps (like normalization, cleaning, lemmatization) one after another to a text
    column in a Polars DataFrame.

    Example:
        >>> preprocessor = TextPreprocessor(text_column="text")
        >>> df = preprocessor.pipeline(
        ...     df,
        ...     steps=["normalize-unicode", "normalize-casing", "remove-stopwords"],
        ...     output_column="text_processed"
        ... )

    Note:
        All original DataFrame columns (including IDs and metadata) are preserved
        in the output, so you can still trace processed text back to its source.

        For more control, you can import and use preprocessing functions directly:
        from newspaper_explorer.data.preprocessing.cleaning import remove_stopwords
    """

    def __init__(self, text_column: str = "text") -> None:
        """
        Initialize preprocessor.

        Args:
            text_column: Name of the column containing text to process
        """
        self.text_column = text_column
        logger.debug(f"TextPreprocessor initialized for column: {text_column}")

    def pipeline(
        self,
        df: pl.DataFrame,
        steps: list[str],
        output_column: str = "text_processed",
        batch_size: int = 32,
        num_beams: int = 4,
        num_gpus: int = 1,
        *,
        use_cache: bool = True,
    ) -> pl.DataFrame:
        """
        Apply a pipeline of preprocessing steps.

        Available steps:
        - normalize-unicode: Unicode normalization (NFC + ftfy + character translation)
        - normalize-whitespace: Normalize whitespace
        - normalize-casing: Convert to lowercase
        - normalize-umlauts: Normalize historical umlauts (ae→ä, oe→ö, ue→ü)
        - normalize-long-s: Normalize long s (ſ) to modern s (simple/context-aware/preserve)
        - dehyphenate: Remove line-break hyphens
        - modernize-transnormer: Neural normalization with Transnormer
        - modernize-dtacab: API normalization with DTA-CAB
        - remove-diacritics: Remove diacritics
        - remove-punctuation: Remove punctuation
        - remove-numbers: Remove numbers
        - remove-stopwords: Remove German stopwords
        - filter-allowed-chars: Remove OCR artifacts and invalid characters
        - lemmatize-spacy: Lemmatize with spaCy (fast)
        - lemmatize-germalemma: Lemmatize with GermaLemma (slow)
        - filter-length: Filter out too short/long texts (by character count)
        - filter-word-count: Filter out too few/many words (by word count)
        - filter-repeating-chars: Remove words with excessive character repetition (OCR garbage)
        - filter-number-only: Remove lines containing only numbers and separators
        - filter-char-token-ratio: Remove lines with high character-to-token ratio (OCR fragmentation)
        - filter-word-length: Remove lines with excessively long words (merged OCR errors)
        - calculate-quality: Calculate OCR quality metrics (char/token ratio, OOV rate, quality score)
        - filter-by-quality: Filter rows by quality score (good/review/poor)

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

            Recommended first step: 'normalize-unicode' to handle OCR artifacts,
            unify quotes/hyphens, and remove control characters.
        """  # noqa: RUF002
        logger.info(f"Starting preprocessing pipeline with {len(steps)} steps")
        logger.info(f"Steps: {', '.join(steps)}")

        current_column = self.text_column

        for i, step in enumerate(steps, 1):
            logger.info(f"Step {i}/{len(steps)}: {step}")
            out_col = output_column if i == len(steps) else f"_tmp_{i}"

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

        # Clean up temporary columns
        temp_cols = [col for col in df.columns if col.startswith("_tmp_")]
        if temp_cols:
            df = df.drop(temp_cols)

        logger.info(f"Preprocessing pipeline completed: {len(df)} rows")
        return df

    def _apply_step(
        self,
        df: pl.DataFrame,
        step: str,
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

        Returns:
            Tuple of (processed DataFrame, new current column name)
        """
        if step not in STEP_CONFIG:
            raise ValueError(f"Unknown preprocessing step: {step}")

        config = STEP_CONFIG[step]

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

        # Add any extra fixed arguments
        if "extra_args" in config:
            kwargs.update(config["extra_args"])

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

    def create_metadata(
        self,
        source: str,
        steps: list[str],
        parameters: dict[str, Any],
        input_df: pl.DataFrame,
        output_df: pl.DataFrame,
        duration_seconds: float,
        previous_preprocessing: Optional[dict[str, Any]] = None,
    ) -> PreprocessingMetadata:
        """
        Create preprocessing metadata for saving alongside results.

        Args:
            source: Source dataset name
            steps: List of preprocessing steps applied
            parameters: Processing parameters
            input_df: Input DataFrame (for statistics)
            output_df: Output DataFrame (for statistics)
            duration_seconds: Processing time
            previous_preprocessing: Previous preprocessing metadata if input was preprocessed

        Returns:
            PreprocessingMetadata object ready to save
        """
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
            steps=steps,
            parameters=parameters,
            input_data=input_data,
            output_data=output_data,
            duration_seconds=duration_seconds,
            previous_preprocessing=previous_preprocessing,
            status="completed",
            completed_at=datetime.now().isoformat(),
            preprocessing_id=None,  # Will be auto-generated
            error_message=None,  # Not needed for completed status
        )
