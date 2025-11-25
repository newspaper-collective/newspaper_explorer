"""
Text preprocessing pipeline for newspaper data.

Provides a simple pipeline function that chains together preprocessing steps
of the modular preprocessing modules.

For direct use of preprocessing functions, import from individual modules:
    from newspaper_explorer.data.preprocessing.historical import normalize_historical
    from newspaper_explorer.data.preprocessing.cleaning import lowercase
    from newspaper_explorer.data.preprocessing.filtering import remove_stopwords
    from newspaper_explorer.data.preprocessing.linguistic import lemmatize_spacy
"""

from datetime import datetime
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

import polars as pl

from newspaper_explorer.data.models import PreprocessingMetadata
from newspaper_explorer.data.preprocessing.cleaning import lowercase, normalize_whitespace
from newspaper_explorer.data.preprocessing.filtering import (
    clean_ocr_artifacts,
    filter_by_char_token_ratio,
    filter_by_length,
    filter_by_word_count,
    filter_excessive_word_length,
    filter_number_only_lines,
    filter_repeating_chars,
    remove_numbers,
    remove_punctuation,
    remove_stopwords,
)
from newspaper_explorer.data.preprocessing.linguistic import (
    dehyphenate,
    dehyphenate_lines,
    lemmatize_germalemma,
    lemmatize_spacy,
)

# Import all preprocessing functions
from newspaper_explorer.data.preprocessing.normalization import (
    dta_cab,
    normalize_long_s,
    normalize_unicode,
    remove_diacritics,
    simple,
    transnormer,
)
from newspaper_explorer.data.preprocessing.presets import get_preset, list_presets
from newspaper_explorer.data.preprocessing.validation import (
    calculate_quality_metrics,
    filter_by_quality_score,
    summarize_quality,
)

logger = logging.getLogger(__name__)


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
        ...     steps=["normalize", "lowercase", "remove-stopwords"],
        ...     output_column="text_processed"
        ... )

    Note:
        All original DataFrame columns (including IDs and metadata) are preserved
        in the output, so you can still trace processed text back to its source.

        For more control, you can import and use preprocessing functions directly:
        from newspaper_explorer.data.preprocessing.cleaning import lowercase
    """

    def __init__(self, text_column: str = "text"):
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
        steps: List[str],
        output_column: str = "text_processed",
        batch_size: int = 32,
        num_beams: int = 4,
        num_gpus: int = 1,
        use_cache: bool = True,
    ) -> pl.DataFrame:
        """
        Apply a pipeline of preprocessing steps.

        Available steps:
        - normalize-unicode: Unicode normalization (NFC + ftfy + character translation)
        - normalize: Normalize historical German characters
        - normalize-long-s: Normalize long s (ſ) to modern s (simple/context-aware/preserve)
        - remove-diacritics: Remove diacritics
        - normalize-whitespace: Normalize whitespace
        - normalize-transnormer: Neural normalization with Transnormer
        - normalize-dtacab: API normalization with DTA-CAB
        - lowercase: Convert to lowercase
        - remove-punctuation: Remove punctuation
        - remove-numbers: Remove numbers
        - remove-stopwords: Remove German stopwords
        - dehyphenate: Remove line-break hyphens
        - lemmatize-spacy: Lemmatize with spaCy (fast)
        - lemmatize: Lemmatize with GermaLemma (slow)
        - filter-length: Filter out too short/long texts (by character count)
        - filter-word-count: Filter out too few/many words (by word count)
        - filter-repeating-chars: Remove words with excessive character repetition (OCR garbage)
        - filter-number-only: Remove lines containing only numbers and separators
        - filter-char-token-ratio: Remove lines with high character-to-token ratio (OCR fragmentation)
        - filter-word-length: Remove lines with excessively long words (merged OCR errors)
        - calculate-quality: Calculate OCR quality metrics (char/token ratio, OOV rate, quality score)
        - filter-by-quality: Filter rows by quality score (good/review/poor)
        - clean-ocr: Remove OCR artifacts and invalid characters

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
        """
        logger.info(f"Starting preprocessing pipeline with {len(steps)} steps")
        logger.info(f"Steps: {', '.join(steps)}")

        current_column = self.text_column

        for i, step in enumerate(steps, 1):
            logger.info(f"Step {i}/{len(steps)}: {step}")

            if i == len(steps):
                out_col = output_column
            else:
                out_col = f"_tmp_{i}"

            if step == "normalize-unicode":
                df = normalize_unicode(
                    df,
                    input_column=current_column,
                    output_column=out_col,
                )
            elif step == "normalize":
                df = simple(df, input_column=current_column, output_column=out_col)
            elif step == "normalize-long-s":
                df = normalize_long_s(
                    df,
                    input_column=current_column,
                    output_column=out_col,
                    mode="simple",
                )
            elif step == "remove-diacritics":
                df = remove_diacritics(
                    df,
                    input_column=current_column,
                    output_column=out_col,
                )
            elif step == "normalize-whitespace":
                df = normalize_whitespace(
                    df,
                    input_column=current_column,
                    output_column=out_col,
                )
            elif step == "normalize-transnormer":
                df = transnormer(
                    df,
                    input_column=current_column,
                    output_column=out_col,
                    batch_size=batch_size,
                    num_beams=num_beams,
                    num_gpus=num_gpus,
                    use_cache=use_cache,
                )
            elif step == "normalize-dtacab":
                df = dta_cab(
                    df,
                    input_column=current_column,
                    output_column=out_col,
                )
            elif step == "lowercase":
                df = lowercase(
                    df,
                    input_column=current_column,
                    output_column=out_col,
                )
            elif step == "remove-punctuation":
                df = remove_punctuation(
                    df,
                    text_column=current_column,
                    input_column=current_column,
                    output_column=out_col,
                )
            elif step == "remove-numbers":
                df = remove_numbers(
                    df,
                    text_column=current_column,
                    input_column=current_column,
                    output_column=out_col,
                )
            elif step == "remove-stopwords":
                df = remove_stopwords(
                    df,
                    text_column=current_column,
                    input_column=current_column,
                    output_column=out_col,
                )
            elif step == "dehyphenate":
                # Use line-level dehyphenation if coordinate columns are available
                required_cols = ["text_block_id", "line_id", "x", "y", "width"]
                if all(col in df.columns for col in required_cols):
                    logger.info("Using line-level dehyphenation (preserves line structure)")
                    df = dehyphenate_lines(
                        df,
                        text_column=current_column,
                        output_column=out_col,
                        text_block_id_column="text_block_id",
                        line_id_column="line_id",
                        x_column="x",
                        y_column="y",
                        width_column="width",
                    )
                else:
                    logger.info("Using simple dehyphenation (line structure not available)")
                    df = dehyphenate(
                        df,
                        text_column=current_column,
                        input_column=current_column,
                        output_column=out_col,
                    )
            elif step == "lemmatize-spacy":
                df = lemmatize_spacy(
                    df,
                    text_column=current_column,
                    input_column=current_column,
                    output_column=out_col,
                )
            elif step == "lemmatize":
                df = lemmatize_germalemma(
                    df,
                    text_column=current_column,
                    input_column=current_column,
                    output_column=out_col,
                )
            elif step == "filter-length":
                df = filter_by_length(df, text_column=current_column, input_column=current_column)
                # Note: filter-length doesn't change the column, so keep current_column
                current_column = current_column
            elif step == "filter-word-count":
                df = filter_by_word_count(
                    df, text_column=current_column, input_column=current_column
                )
                # Note: filter-word-count doesn't change the column, so keep current_column
                current_column = current_column
            elif step == "filter-repeating-chars":
                df = filter_repeating_chars(
                    df,
                    text_column=current_column,
                    input_column=current_column,
                    output_column=out_col,
                )
            elif step == "filter-number-only":
                df = filter_number_only_lines(
                    df, text_column=current_column, input_column=current_column
                )
                # Note: filter-number-only removes rows, doesn't change column
                current_column = current_column
            elif step == "filter-char-token-ratio":
                df = filter_by_char_token_ratio(
                    df, text_column=current_column, input_column=current_column
                )
                # Note: filter removes rows, doesn't change column
                current_column = current_column
            elif step == "filter-word-length":
                df = filter_excessive_word_length(
                    df, text_column=current_column, input_column=current_column
                )
                # Note: filter removes rows, doesn't change column
                current_column = current_column
            elif step == "calculate-quality":
                df = calculate_quality_metrics(
                    df, text_column=current_column, input_column=current_column
                )
                # Note: adds quality columns, doesn't change current column
                current_column = current_column
            elif step == "filter-by-quality":
                df = filter_by_quality_score(df)
                # Note: filters rows, doesn't change current column
                current_column = current_column
            elif step == "clean-ocr":
                df = clean_ocr_artifacts(
                    df,
                    text_column=current_column,
                    input_column=current_column,
                    output_column=out_col,
                )
            else:
                raise ValueError(f"Unknown preprocessing step: {step}")

            current_column = out_col

        temp_cols = [col for col in df.columns if col.startswith("_tmp_")]
        if temp_cols:
            df = df.drop(temp_cols)

        logger.info("Preprocessing pipeline complete")
        return df

    def load_previous_metadata(self, input_path: Union[Path, str]) -> Optional[Dict[str, Any]]:
        """
        Load preprocessing metadata from input file if it exists.

        Args:
            input_path: Path to input parquet file

        Returns:
            Dictionary with previous preprocessing info, or None if not found
        """
        from newspaper_explorer.data.utils.metadata import (
            PreprocessingMetadata,
            find_metadata_for_parquet,
            load_metadata,
        )

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
            except Exception as e:
                logger.debug(f"Could not load preprocessing metadata: {e}")

        return None

    def create_metadata(
        self,
        source: str,
        steps: List[str],
        parameters: Dict[str, Any],
        input_df: pl.DataFrame,
        output_df: pl.DataFrame,
        duration_seconds: float,
        previous_preprocessing: Optional[Dict[str, Any]] = None,
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
        from newspaper_explorer.data.utils.metadata import (
            PreprocessingMetadata,
            extract_input_stats,
            extract_output_stats,
        )

        metadata = PreprocessingMetadata(
            source=source,
            steps=steps,
            parameters=parameters,
            input_data=extract_input_stats(input_df),
            output_data=extract_output_stats(output_df),
            duration_seconds=duration_seconds,
            previous_preprocessing=previous_preprocessing,
            status="completed",
            completed_at=datetime.now().isoformat(),
            preprocessing_id=None,  # Will be auto-generated
            error_message=None,  # Not needed for completed status
        )

        return metadata


def get_recommended_pipeline(name: str = "standard") -> List[str]:
    """
    Get a recommended preprocessing pipeline configuration.

    Available pipelines:

    General-purpose:
    - minimal: Preserve original text, only fix critical OCR issues
    - basic: Fast OCR cleanup (recommended start)
    - standard: General text analysis (default choice)
    - search: Optimized for search and matching
    - analysis: Word frequency with filtering

    Analysis-specific:
    - entities: Named entity recognition (NER)
    - topics: Topic modeling (BERTopic, etc.)
    - emotions: Emotion classification, sentiment analysis
    - keywords: Keyword extraction, TF-IDF
    - embeddings: Text embeddings, semantic similarity
    - concepts: Concept extraction, semantic networks

    Args:
        name: Pipeline name (default: "standard")

    Returns:
        List of preprocessing step names

    Example:
        >>> from newspaper_explorer.data.preprocessing.pipeline import (
        ...     TextPreprocessor,
        ...     get_recommended_pipeline
        ... )
        >>> preprocessor = TextPreprocessor(text_column="text")
        >>> steps = get_recommended_pipeline("entities")
        >>> df_processed = preprocessor.pipeline(df, steps=steps)

    Note:
        These pipelines do NOT include:
        - normalize-transnormer: GPU-intensive neural normalization
        - normalize-dtacab: API-based normalization (requires DTA-CAB service)
        - lemmatize-spacy / lemmatize: Lemmatization (changes word forms)

        If you need these features, add them manually or create a custom pipeline.
    """
    return get_preset(name)


def list_pipelines() -> Dict[str, Dict[str, Union[str, List[str]]]]:
    """
    List all available recommended pipelines with their descriptions.

    Returns:
        Dictionary mapping pipeline names to their configuration

    Example:
        >>> from newspaper_explorer.data.preprocessing.pipeline import list_pipelines
        >>> pipelines = list_pipelines()
        >>> for name, config in pipelines.items():
        ...     print(f"{name}: {config['description']}")
    """
    return list_presets()
    return list_presets()
